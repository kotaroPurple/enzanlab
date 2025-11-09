import numpy as np
import pytest

from enzanlab.math.dmd.hankel_onlinedmd import (
    HankelOnlineDMD,
    HankelOnlineDMDConfig,
)


def _generate_sine(num_samples: int, frequency: float, dt: float) -> np.ndarray:
    t = np.arange(num_samples) * dt
    return np.sin(2 * np.pi * frequency * t)


def test_update_requires_initialization() -> None:
    """."""
    model = HankelOnlineDMD(HankelOnlineDMDConfig(window_size=8))
    with pytest.raises(RuntimeError):
        model.update(0.0)


def test_initialize_requires_enough_samples() -> None:
    """."""
    model = HankelOnlineDMD(HankelOnlineDMDConfig(window_size=8))
    with pytest.raises(ValueError):
        model.initialize(np.zeros(8))  # type: ignore


def test_hankel_online_dmd_tracks_sine_wave() -> None:
    """."""
    config = HankelOnlineDMDConfig(window_size=30, r_max=4, lambda_=0.995)
    model = HankelOnlineDMD(config)

    dt = 1.0 / 100.0
    target_freq = 2.0
    total_samples = 500
    series = _generate_sine(total_samples, target_freq, dt)

    init_len = config.window_size + 5
    model.initialize(series[:init_len])  # type: ignore
    model.update_many(series[init_len:])

    assert model.initialized
    assert model.time_index() == total_samples

    freqs = model.get_mode_frequencies(dt=dt)
    assert freqs.size > 0
    # Frequency magnitude should match the true frequency within tolerance
    assert np.any(np.isclose(np.abs(freqs), target_freq, atol=0.2))

    growth = model.get_mode_growth_rates(dt=dt)
    amps = model.get_mode_amplitudes()
    assert freqs.shape == growth.shape == amps.shape

    forward = model.reconstruct_mode_time_series(n_samples=10)
    assert forward.shape[0] == freqs.size
    assert forward.shape[1] == config.window_size + 9
