"""Provide tools for extracting spectral sidebands using complex demodulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfiltfilt


@dataclass(slots=True)
class SidebandFilter:
    """Provide a single-sideband band-pass filter via complex demodulation.

    The filter isolates a narrow frequency band by translating the selected sideband to
    baseband, applying an IIR low-pass filter, and (optionally) shifting it back to the
    original center frequency.

    Args:
        sample_rate (float): Sampling frequency in Hz.
        band (tuple[float, float]): Lower and upper frequency bounds in Hz. Values must
            satisfy -sample_rate / 2 < band[0] < band[1] < sample_rate / 2.
        zero_phase (bool): If True, keep the demodulated (baseband) signal instead of
            shifting it back to the original center frequency.
        filter_order (int): Order of the Butterworth low-pass filter applied after
            demodulation (default: 6).
        remodulate (bool): If True, shift the filtered baseband signal back to the
            original center frequency (default: True).

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
    filter_order: int = 6
    remodulate: bool = True
    _sos: NDArray[np.float64] = field(init=False, repr=False)
    _center_frequency: float = field(init=False, repr=False)
    _transient_samples: int = field(init=False, repr=False)

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
        if not isinstance(self.remodulate, (bool, np.bool_)):
            raise TypeError("remodulate must be a boolean.")
        self.remodulate = bool(self.remodulate)

        self.filter_order = int(self.filter_order)
        if self.filter_order < 1:
            raise ValueError("filter_order must be greater than or equal to 1.")

        self._center_frequency = 0.5 * (low + high)
        cutoff = 0.5 * (high - low)
        if cutoff <= 0:
            raise ValueError("Computed cutoff frequency must be positive.")

        self._sos = butter(
            N=self.filter_order,
            Wn=cutoff,
            btype="low",
            fs=self.sample_rate,
            output="sos",
        )  # type: ignore
        self._transient_samples = max(1, 4 * self.filter_order)

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

        if self.zero_phase:
            baseband_filtered = sosfiltfilt(self._sos, baseband, axis=-1)
        else:
            baseband_filtered = sosfilt(self._sos, baseband, axis=-1)

        filtered = baseband_filtered
        if self.remodulate:
            remod_phase = np.exp(1j * 2.0 * np.pi * shift_frequency * time)
            filtered = baseband_filtered * remod_phase

        filtered = np.moveaxis(filtered, -1, axis)  # type: ignore
        return filtered  # type: ignore

    @property
    def sos(self) -> NDArray[np.float64]:
        """Return a copy of the IIR low-pass section coefficients."""
        return self._sos.copy()

    @property
    def transient_samples(self) -> int:
        """Return a heuristic count of samples affected by filter transients."""
        return self._transient_samples
