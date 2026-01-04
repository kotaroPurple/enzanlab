"""FFT-based correlation utilities.

This module provides windowed, frame-wise (STFT-like) autocorrelation
computed via FFT (Wiener-Khinchin).

Notes:
- This implementation computes **linear** autocorrelation for each frame by
  zero-padding before FFT (avoiding circular wrap-around).
- By default, the output is the *biased* autocorrelation (no division by the
  number of overlapping samples). You can request an unbiased estimate.
"""


from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows


def _next_pow2(n: int) -> int:
    """Return the next power of two >= n.

    Args:
        n: Input value.

    Returns:
        Next power of two greater than or equal to n.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _resolve_window(
    window: Literal["tukey", "hann", "hamming", "boxcar"],
    frame_length: int,
    *,
    dtype: np.dtype,
    tukey_alpha: float,
) -> NDArray[np.floating]:
    """Resolve window name to a 1D ndarray of length frame_length.

    Args:
        window: Window name.
        frame_length: Frame length.
        dtype: Target dtype.
        tukey_alpha: Shape parameter for Tukey window.

    Returns:
        Window array of shape (frame_length,).
    """
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if window == "tukey":
        return windows.tukey(frame_length, alpha=tukey_alpha, sym=True).astype(dtype, copy=False)
    if window == "hann":
        return windows.hann(frame_length, sym=True).astype(dtype, copy=False)
    if window == "hamming":
        return windows.hamming(frame_length, sym=True).astype(dtype, copy=False)
    if window == "boxcar":
        return windows.boxcar(frame_length, sym=True).astype(dtype, copy=False)
    raise ValueError(f"window must be one of 'tukey', 'hann', 'hamming', 'boxcar', got {window!r}")


def stft_autocorrelation(
    x: NDArray[np.floating] | NDArray[np.integer] | NDArray[np.complexfloating],
    *,
    frame_length: int,
    hop_length: int,
    window: Literal["tukey", "hann", "hamming", "boxcar"] = "tukey",
    max_lag: int | None = None,
    center: bool = False,
    pad_mode: str = "reflect",
    detrend: Literal["none", "mean"] = "mean",
    normalize: bool = False,
    nccf: bool = False,
    unbiased: bool = False,
    nfft: int | None = None,
    tukey_alpha: float = 0.5,
) -> tuple[NDArray[np.floating] | NDArray[np.complexfloating], NDArray[np.integer]]:
    """Compute STFT-like (sliding-frame) autocorrelation via FFT.

    Args:
        x: Input 1D signal. Complex signals are supported.
        frame_length: Frame (window) length in samples. (処理幅)
        hop_length: Hop length in samples between adjacent frames. (スライド幅)
        window: Window name. Supported: "tukey", "hann", "hamming", "boxcar".
        max_lag: Maximum lag (in samples) to return. If None, returns 0..frame_length-1.
        center: If True, pad so frames are centered like typical STFT.
        pad_mode: Padding mode for np.pad when center=True.
        detrend: "mean" subtracts the frame mean before computing autocorrelation.
            "none" leaves the frame as-is.
        normalize: If True, normalize by R[0] (so that R[0]=1 when possible).
        nccf: If True, apply per-lag NCCF normalization using overlap energy.
        unbiased: If True, divide each lag by the number of overlapping samples.
        nfft: FFT length. If None, uses next power of 2 >= 2*frame_length.
        tukey_alpha: Shape parameter for Tukey window (0..1) when window="tukey".

    Returns:
        Tuple of:
            - Autocorrelation per frame, shape (n_frames, n_lags). Complex if input is complex.
            - Lags in samples, shape (n_lags,).

    Notes:
        This computes the linear autocorrelation of the *windowed* frames.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}")
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    dtype = np.result_type(x.dtype, np.float64)
    x = x.astype(dtype, copy=False)

    if center:
        pad = frame_length // 2
        x = np.pad(array=x, pad_width=(pad, pad), mode=pad_mode)

    if max_lag is None:
        max_lag = frame_length - 1
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    if max_lag >= frame_length:
        raise ValueError(
            f"max_lag must be < frame_length ({frame_length}), got {max_lag}"
        )

    w = _resolve_window(window, frame_length, dtype=dtype, tukey_alpha=tukey_alpha)

    if nfft is None:
        # Need at least 2*frame_length to get linear autocorrelation via FFT.
        nfft = _next_pow2(2 * frame_length)
    if nfft < 2 * frame_length:
        raise ValueError(f"nfft must be >= 2*frame_length ({2*frame_length}), got {nfft}")

    n = len(x)
    if n < frame_length:
        empty = np.empty((0, max_lag + 1), dtype=dtype)
        return empty, np.arange(max_lag + 1, dtype=int)

    n_frames = 1 + (n - frame_length) // hop_length
    r = np.empty((n_frames, max_lag + 1), dtype=dtype)

    # Precompute unbiased normalization if requested
    if unbiased:
        denom = (frame_length - np.arange(max_lag + 1)).astype(dtype)
    else:
        denom = None

    for i in range(n_frames):
        start = i * hop_length
        frame = x[start : start + frame_length]

        if detrend == "mean":
            frame = frame - frame.mean()
        elif detrend != "none":
            raise ValueError(f"detrend must be 'none' or 'mean', got {detrend!r}")

        frame_w = frame * w

        # FFT-based autocorrelation: r = IFFT(|FFT(frame_w)|^2)
        if np.iscomplexobj(frame_w):
            F = np.fft.fft(frame_w, n=nfft)
            S = F * np.conj(F)
            acf = np.fft.ifft(S, n=nfft)
        else:
            F = np.fft.rfft(frame_w, n=nfft)
            S = F * np.conj(F)
            acf = np.fft.irfft(S, n=nfft)

        # For linear autocorr, take first frame_length samples (lags 0..frame_length-1)
        acf = acf[: frame_length]

        # Keep only requested lags
        acf = acf[: max_lag + 1]

        if denom is not None:
            acf = acf / denom

        if nccf:
            power = np.abs(frame_w) ** 2
            prefix = np.concatenate(([0], np.cumsum(power)))
            lags = np.arange(max_lag + 1)
            energy_a = prefix[frame_length - lags]
            energy_b = prefix[frame_length] - prefix[lags]
            denom = np.sqrt(energy_a * energy_b)
            acf = np.divide(acf, denom, out=np.zeros_like(acf), where=denom > 0)
            r0 = acf[0]
            if np.isfinite(r0) and r0 != 0:
                acf = acf / r0
        elif normalize:
            r0 = acf[0]
            if np.isfinite(r0) and r0 != 0:
                acf = acf / r0

        r[i, :] = acf

    lags = np.arange(max_lag + 1, dtype=int)
    return r, lags
