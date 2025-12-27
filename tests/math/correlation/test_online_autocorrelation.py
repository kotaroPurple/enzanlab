"""Tests for online autocorrelation."""

import numpy as np

from enzanlab.math.correlation.online import OnlineAutocorrelation


def test_online_autocorrelation_complex_updates() -> None:
    """Handles complex inputs and uses conjugate lagged samples."""
    acf = OnlineAutocorrelation(max_lag=2, forgetting_factor=0.5, detrend="none")

    r1 = acf.update(1.0 + 1.0j)
    expected_r1 = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    np.testing.assert_allclose(r1, expected_r1, rtol=1e-12, atol=1e-12)

    r2 = acf.update(2.0 - 1.0j)
    expected_r2 = np.array([3.0 + 0.0j, 0.5 - 1.5j, 0.0 + 0.0j])
    np.testing.assert_allclose(r2, expected_r2, rtol=1e-12, atol=1e-12)
