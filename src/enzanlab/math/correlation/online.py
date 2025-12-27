

r"""Online (streaming) autocorrelation with exponential forgetting.

This module implements an exponentially-weighted moving autocorrelation (online ACF),
useful for tracking a dominant period (e.g., heart-rate / RRI) without recomputing
frame-wise FFT autocorrelation.

The core update is:

    R_t(\tau) = \lambda R_{t-1}(\tau) + (1-\lambda) x_t x_{t-\tau}

where 0 < \lambda < 1 is the forgetting factor and \tau is the lag in samples.

Notes:
- This is *not* the same as a windowed (finite) autocorrelation; it is an
    exponentially weighted statistic.
- To focus on heart-rate, restrict the lag range to a plausible BPM band.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DetrendMode = Literal["none", "ema_mean"]


@dataclass(slots=True)
class OnlineAutocorrelation:
    r"""Exponentially-forgotten (online) autocorrelation estimator.

    This class maintains an exponentially weighted autocorrelation sequence
    for lags 0..max_lag and updates it sample-by-sample.

    The estimator is well-suited for online fundamental-period tracking because:
    - update cost is O(max_lag) per sample,
    - the autocorrelation peak tends to move smoothly over time.

    Args:
        min_lag: Minimum lag (in samples) to maintain. Returned lags are
            min_lag..max_lag inclusive.
        max_lag: Maximum lag (in samples) to maintain. Returned lags are
            min_lag..max_lag inclusive.
        forgetting_factor: Forgetting factor \lambda in (0, 1). Larger values
            remember longer history (smoother but slower to react).
        detrend: Detrending mode.
            - "none": use raw samples.
            - "ema_mean": subtract an exponentially weighted moving average
                (EMA) of the signal before correlation (helps when DC/slow drift
                contaminates the peak).
        mean_forgetting_factor: Forgetting factor for the EMA mean when
            detrend="ema_mean". If None, uses forgetting_factor.
        unbiased: If True, apply a simple lag-dependent correction to reduce
            bias introduced by exponential weighting. This is approximate and
            mainly useful for comparing magnitudes across lags; peak *location*
            often works fine without it.

    Attributes:
        r: Current autocorrelation values, shape (max_lag-min_lag+1,).
        lags: Lags in samples, shape (max_lag-min_lag+1,).

    Examples:
        >>> acf = OnlineAutocorrelation(max_lag=200, forgetting_factor=0.995)
        >>> for x_t in stream:
        ...     r = acf.update(x_t)
        ...     # Use r to track a peak lag.

    """

    max_lag: int
    forgetting_factor: float
    min_lag: int = 0
    r: np.ndarray = field(init=False, repr=False)
    lags: np.ndarray = field(init=False, repr=False)
    _dtype: np.dtype = field(init=False, repr=False)
    _buf: deque[complex | float] = field(init=False, repr=False)
    _n_seen: int = field(init=False, repr=False)
    _mean: complex | float = field(init=False, repr=False)
    _w: np.ndarray | None = field(init=False, repr=False)
    detrend: DetrendMode = "none"
    mean_forgetting_factor: float | None = None
    unbiased: bool = False

    def __post_init__(self) -> None:
        """Initialize internal buffers and validate configuration."""
        if self.min_lag < 0:
            raise ValueError("min_lag must be >= 0")
        if self.max_lag < 0:
            raise ValueError("max_lag must be >= 0")
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag must be <= max_lag")
        if not (0.0 < self.forgetting_factor < 1.0):
            raise ValueError("forgetting_factor must be in (0, 1)")
        if self.mean_forgetting_factor is None:
            self.mean_forgetting_factor = self.forgetting_factor
        if not (0.0 < float(self.mean_forgetting_factor) < 1.0):
            raise ValueError("mean_forgetting_factor must be in (0, 1)")

        self._dtype = np.float64  # type: ignore
        self.r = np.zeros(self.max_lag - self.min_lag + 1, dtype=self._dtype)
        self.lags = np.arange(self.min_lag, self.max_lag + 1, dtype=int)

        # A ring buffer holding the most recent samples (raw), length max_lag+1.
        self._buf = deque([0.0] * (self.max_lag + 1), maxlen=self.max_lag + 1)
        self._n_seen: int = 0

        # EMA mean for detrending
        self._mean = 0.0

        # Optional approximate bias-correction for exponential weighting.
        # For each lag, effective weight sum differs because x(t-τ) becomes valid
        # only after τ samples. We track per-lag weight sums.
        if self.unbiased:
            self._w = np.zeros(self.max_lag - self.min_lag + 1, dtype=np.float64)
        else:
            self._w = None

    def reset(self) -> None:
        """Reset internal state to zeros."""
        self.r.fill(0.0)
        self._buf.clear()
        self._buf.extend([0.0] * (self.max_lag + 1))
        self._n_seen = 0
        self._mean = 0.0
        if self._w is not None:
            self._w.fill(0.0)

    def update(self, x_t: float | complex) -> np.ndarray:
        """Update the online autocorrelation with one new sample.

        Args:
            x_t: New sample value. Complex values are supported.

        Returns:
            Current autocorrelation array for lags 0..max_lag.

        Notes:
            For early times (when fewer than max_lag samples have been seen),
            large lags effectively correlate with zeros from the initial buffer.
            If that matters, use `valid_lag_max()` to restrict peak search.
        """
        x_arr = np.asarray(x_t)
        x = complex(x_arr) if np.iscomplexobj(x_arr) else float(x_arr)
        if np.iscomplexobj(x_arr) and self._dtype != np.complex128:
            self._promote_to_complex()
        self._n_seen += 1

        if self.detrend == "ema_mean":
            a = float(self.mean_forgetting_factor)  # type: ignore
            self._mean = a * self._mean + (1.0 - a) * x
            x = x - self._mean
        elif self.detrend != "none":
            raise ValueError(f"Unsupported detrend mode: {self.detrend!r}")

        # Snapshot the previous samples for lagged products.
        # buf[0] is the oldest in the deque; convert to array for vector ops.
        prev = np.fromiter(self._buf, dtype=self._dtype)
        # We want x(t-τ). With deque holding last max_lag+1 samples, the newest
        # sample currently in the buffer corresponds to x(t-1) (before appending x).
        # Align so that prev_last is x(t-1), prev[-1].
        # After we append x, it becomes x(t).
        prev_rev = prev[::-1]
        if self.min_lag == 0:
            # Build lagged vector: [x(t), x(t-1), ..., x(t-max_lag)]
            lagged = np.empty(self.max_lag + 1, dtype=self._dtype)
            lagged[0] = x
            lagged[1:] = prev_rev[0 : self.max_lag]
        else:
            # Build lagged vector for lags min_lag..max_lag
            lagged = prev_rev[self.min_lag - 1 : self.max_lag]

        lam = self.forgetting_factor
        one_minus = 1.0 - lam

        # Update correlation: R <- lam*R + (1-lam)* x(t) * conj(x(t-τ))
        self.r = lam * self.r + one_minus * (x * np.conj(lagged))

        if self._w is not None:
            # Track per-lag weight sums to approximately de-bias magnitudes.
            # We update weights similarly: w <- lam*w + (1-lam)*1 for lags that are valid.
            # A lag τ becomes valid only after τ samples. Before that, lagged values are
            # influenced by the initial zeros, so we keep weights at 0.
            valid_max = self.valid_lag_max()
            valid_count = max(0, min(valid_max, self.max_lag) - self.min_lag + 1)
            # Update valid lags only
            self._w[:valid_count] = lam * self._w[:valid_count] + one_minus
            # Leave others unchanged
            if valid_count < self._w.size:
                self._w[valid_count:] = lam * self._w[valid_count:]

            # Apply correction in-place for returned value only (keep internal r unchanged).
            # Return corrected copy.
            w = self._w
            out = self.r.copy()
            mask = w > 0
            out[mask] = out[mask] / w[mask]
        else:
            out = self.r

        # Store the detrended sample to keep lagged products consistent.
        self._buf.append(x)

        return out

    def valid_lag_max(self) -> int:
        r"""Return the maximum lag that is fully supported by observed samples.

        Returns:
            Maximum valid lag such that x(t-\tau) is based on actual seen samples.

        Notes:
            For sample index t (1-based count of updates), the largest valid lag is
            min(max_lag, t-1).
        """
        return min(self.max_lag, max(0, self._n_seen - 1))

    def get(self, *, normalize: bool = False) -> np.ndarray:
        """Get the current autocorrelation.

        Args:
            normalize: If True, normalize by lag-0 value so that r[0] == 1
                when possible. If min_lag > 0, normalization uses the first
                available lag (min_lag).

        Returns:
            Autocorrelation array for lags 0..max_lag. If unbiased=True, returns
            the approximately de-biased values.
        """
        if self._w is not None:
            r = self.r.copy()
            mask = self._w > 0
            r[mask] = r[mask] / self._w[mask]
        else:
            r = self.r
        if normalize:
            r0 = r[0]
            if np.isfinite(r0) and r0 != 0:
                r = r / r0
        return r

    def _promote_to_complex(self) -> None:
        """Promote internal state to complex dtype for complex inputs."""
        self._dtype = np.complex128  # type: ignore
        self.r = self.r.astype(self._dtype, copy=False)
        self._buf = deque((complex(v) for v in self._buf), maxlen=self.max_lag + 1)
        self._mean = complex(self._mean)
