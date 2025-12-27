"""Tests for FFT-based STFT autocorrelation."""

import numpy as np
import pytest

from enzanlab.math.correlation.fft_based import stft_autocorrelation


def test_stft_autocorrelation_real_matches_correlate() -> None:
    """Matches numpy.correlate for real signals."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r, lags = stft_autocorrelation(
        x,
        frame_length=4,
        hop_length=4,
        window="boxcar",
        max_lag=3,
        detrend="none",
        nfft=8,
    )
    expected = np.correlate(x, x, mode="full")[3:7]
    assert r.shape == (1, 4)
    np.testing.assert_allclose(r[0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(lags, np.arange(4))


def test_stft_autocorrelation_complex_matches_correlate() -> None:
    """Matches numpy.correlate for complex signals."""
    x = np.array([1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.5j, 0.5 - 0.5j])
    r, _ = stft_autocorrelation(
        x,
        frame_length=4,
        hop_length=4,
        window="boxcar",
        max_lag=3,
        detrend="none",
        nfft=8,
    )
    expected = np.correlate(x, x, mode="full")[3:7]
    np.testing.assert_allclose(r[0], expected, rtol=1e-12, atol=1e-12)


def test_stft_autocorrelation_unbiased_constant_is_one() -> None:
    """Unbiased estimate returns ones for constant input."""
    x = np.ones(8, dtype=float)
    r, _ = stft_autocorrelation(
        x,
        frame_length=4,
        hop_length=4,
        window="boxcar",
        max_lag=3,
        detrend="none",
        unbiased=True,
        nfft=8,
    )
    expected = np.ones(4, dtype=float)
    np.testing.assert_allclose(r[0], expected, rtol=1e-12, atol=1e-12)


def test_stft_autocorrelation_invalid_window_raises() -> None:
    """Raises for unsupported window name."""
    x = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="window must be one of"):
        stft_autocorrelation(
            x,
            frame_length=4,
            hop_length=2,
            window="gauss",  # type: ignore
        )


def test_stft_autocorrelation_max_lag_validation() -> None:
    """Raises for invalid max_lag."""
    x = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="max_lag must be < frame_length"):
        stft_autocorrelation(
            x,
            frame_length=4,
            hop_length=2,
            window="tukey",
            max_lag=4,
        )
