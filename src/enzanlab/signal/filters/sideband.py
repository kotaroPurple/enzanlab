"""Tools for extracting spectral sidebands using complex demodulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter

@dataclass(slots=True)
class SidebandFilter:
    """Single-sideband band-pass filter via complex demodulation.

    The filter isolates a narrow frequency band by translating the selected sideband to
    baseband, applying an FIR low-pass filter, and shifting it back to the original
    center frequency.

    Args:
        sample_rate (float): Sampling frequency in Hz.
        band (tuple[float, float]): Lower and upper frequency bounds in Hz. Values must
            satisfy -sample_rate / 2 < band[0] < band[1] < sample_rate / 2.
        zero_phase (bool): If True, keep the demodulated (baseband) signal instead of
            shifting it back to the original center frequency.
        num_taps (int): Number of taps for the FIR low-pass filter. Should be odd for
            exact linear phase (default: 129).
        window (str): Window specification forwarded to ``scipy.signal.firwin``.

    Example:
        >>> fs = 1_000.0
        >>> filt = SidebandFilter(sample_rate=fs, band=(90.0, 110.0))
        >>> t = np.arange(2_048) / fs
        >>> x = np.exp(1j * 2 * np.pi * 100.0 * t)
        >>> y = filt.filter(x)
        >>> y.shape
        (2048,)
    """

    sample_rate: float
    band: tuple[float, float]
    zero_phase: bool = False
    num_taps: int = 129
    window: str = "hann"
    _taps: NDArray[np.float64] = field(init=False, repr=False)
    _center_frequency: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and design the prototype low-pass filter."""
        self.sample_rate = float(self.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if len(self.band) != 2:
            raise ValueError("band must contain exactly two frequency bounds.")
        low, high = (float(self.band[0]), float(self.band[1]))
        if low >= high:
            raise ValueError("band must satisfy band[0] < band[1].")
        nyquist = 0.5 * self.sample_rate
        if low <= -nyquist or high >= nyquist:
            raise ValueError(
                "Band edges must lie strictly within (-sample_rate/2, sample_rate/2)."
            )

        self.band = (low, high)
        if not isinstance(self.zero_phase, (bool, np.bool_)):
            raise TypeError("zero_phase must be a boolean.")
        self.zero_phase = bool(self.zero_phase)

        self.num_taps = int(self.num_taps)
        if self.num_taps < 3:
            raise ValueError("num_taps must be greater than or equal to 3.")
        if self.num_taps % 2 == 0:
            # Even-length FIRs introduce half-sample delay; odd length keeps group delay integer.
            self.num_taps += 1

        self._center_frequency = 0.5 * (low + high)
        cutoff = 0.5 * (high - low)
        if cutoff <= 0:
            raise ValueError("Computed cutoff frequency must be positive.")

        self._taps = firwin(
            numtaps=self.num_taps,
            cutoff=cutoff,
            window=self.window,
            pass_zero="lowpass",
            fs=self.sample_rate,
        )

    def filter(self, signal: NDArray[np.complex128], axis: int = -1) -> NDArray[np.complex128]:
        """Apply the sideband filter to a signal.

        Args:
            signal (np.ndarray): Input signal containing the targeted sideband. The array
                can be real or complex and of arbitrary shape.
            axis (int): Axis along which the time series is stored. Defaults to the last
                axis.

        Returns:
            np.ndarray: Complex output with the same shape as ``signal`` where only the
            selected sideband remains.

        Raises:
            ValueError: If ``signal`` has zero length along the filtering axis.

        Example:
            >>> filt = SidebandFilter(sample_rate=1_000.0, band=(90.0, 110.0))
            >>> t = np.arange(1_024) / 1_000.0
            >>> x = np.cos(2 * np.pi * 100.0 * t)
            >>> y = filt.filter(x)
            >>> y.dtype
            dtype('complex128')
        """
        data = np.asarray(signal, dtype=np.complex128)
        if data.shape == ():
            raise ValueError("signal must not be scalar.")

        data = np.moveaxis(data, axis, -1)
        n_samples = data.shape[-1]
        if n_samples == 0:
            raise ValueError("signal must contain at least one sample along the target axis.")

        time = np.arange(n_samples, dtype=np.float64) / self.sample_rate
        shift_frequency = self._center_frequency
        demod_phase = np.exp(-1j * 2.0 * np.pi * shift_frequency * time)
        baseband = data * demod_phase

        baseband_filtered = lfilter(self._taps, 1.0, baseband, axis=-1)
        if self.zero_phase:
            filtered = baseband_filtered
        else:
            remod_phase = np.exp(1j * 2.0 * np.pi * shift_frequency * time)
            filtered = baseband_filtered * remod_phase

        filtered = np.moveaxis(filtered, -1, axis)
        return filtered

    @property
    def taps(self) -> NDArray[np.float64]:
        """Return a copy of the prototype low-pass filter taps."""
        return self._taps.copy()
