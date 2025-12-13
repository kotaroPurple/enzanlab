import numpy as np

from enzanlab.math.geometry.geodesic_tracker import GeodesicTracker1D


def test_process_matches_incremental() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(50)
    theta = 0.4 * np.sin(2 * np.pi * 0.02 * t)
    r = 1.0 + 0.05 * np.sin(2 * np.pi * 0.003 * t)
    z = r * np.exp(1j * theta)
    z += 0.01 * (rng.standard_normal(t.shape) + 1j * rng.standard_normal(t.shape))

    tracker = GeodesicTracker1D(win=16, smooth_u=0.1)
    s_batch, u_batch, _ = tracker.process(z)

    tracker.reset()
    s_inc = []
    u_inc = []
    for zi in z:
        s_i, u_i = tracker.update(zi)
        s_inc.append(s_i)
        u_inc.append(u_i)

    assert np.allclose(s_batch, s_inc)
    assert np.allclose(u_batch, u_inc)
