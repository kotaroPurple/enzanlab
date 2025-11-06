"""Unit tests for the sideband filter."""

from __future__ import annotations

import numpy as np
import pytest

from enzanlab.signal.filters.sideband import SidebandFilter


def _steady_slice(transient: int, n_samples: int) -> slice:
    """Return slice indices that avoid the filter's transient region."""
    delay = max(transient, 1)
    if 2 * delay >= n_samples:
        return slice(None)
    return slice(delay, n_samples - delay)


@pytest.mark.parametrize(
    ("band", "component"),
    [
        ((100.0, 140.0), "positive"),
        ((-200.0, -160.0), "negative"),
    ],
)
def test_sideband_filter_recovers_single_tone(
    band: tuple[float, float], component: str
) -> None:
    """Verify the filter keeps only the requested spectral component."""
    fs = 2_048.0
    n_samples = 8_192
    t = np.arange(n_samples, dtype=np.float64) / fs

    positive_freq = 120.0
    negative_freq = 180.0
    signal = (
        np.exp(1j * 2.0 * np.pi * positive_freq * t)
        + np.exp(-1j * 2.0 * np.pi * negative_freq * t)
    )

    if component == "positive":
        expected = np.exp(1j * 2.0 * np.pi * positive_freq * t)
    else:
        expected = np.exp(-1j * 2.0 * np.pi * negative_freq * t)

    filt = SidebandFilter(sample_rate=fs, band=band, filter_order=8)
    filtered = filt.filter(signal)

    steady = _steady_slice(filt.transient_samples, n_samples)
    numerator = np.linalg.norm(filtered[steady] - expected[steady])
    denominator = np.linalg.norm(expected[steady])
    assert numerator / denominator < 5e-2


def test_zero_phase_returns_baseband() -> None:
    """Ensure zero-phase, non-remodulated output is a unit baseband tone."""
    fs = 2_048.0
    n_samples = 8_192
    t = np.arange(n_samples, dtype=np.float64) / fs

    tone = np.exp(1j * 2.0 * np.pi * 120.0 * t)
    filt = SidebandFilter(
        sample_rate=fs,
        band=(100.0, 140.0),
        zero_phase=True,
        remodulate=False,
        filter_order=8,
    )
    filtered = filt.filter(tone)

    steady = _steady_slice(filt.transient_samples, n_samples)
    baseband = filtered[steady]
    reference = np.ones_like(baseband)
    numerator = np.linalg.norm(baseband - reference)
    denominator = np.linalg.norm(reference)
    assert numerator / denominator < 5e-2


def test_zero_phase_with_remodulation_restores_frequency() -> None:
    """Ensure zero-phase filtering with remodulation restores the original tone."""
    fs = 2_048.0
    n_samples = 8_192
    t = np.arange(n_samples, dtype=np.float64) / fs

    tone = np.exp(1j * 2.0 * np.pi * 120.0 * t)
    filt = SidebandFilter(
        sample_rate=fs,
        band=(100.0, 140.0),
        zero_phase=True,
        remodulate=True,
        filter_order=8,
    )
    filtered = filt.filter(tone)

    steady = _steady_slice(filt.transient_samples, n_samples)
    reference = tone[steady]
    numerator = np.linalg.norm(filtered[steady] - reference)
    denominator = np.linalg.norm(reference)
    assert numerator / denominator < 5e-2


def test_invalid_band_raises() -> None:
    """Confirm invalid band definitions raise configuration errors."""
    with pytest.raises(ValueError):
        SidebandFilter(sample_rate=1_000.0, band=(200.0, 200.0))
