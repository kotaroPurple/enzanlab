"""Example usage of GeodesicTracker1D on noisy polar data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from enzanlab.math.geometry.geodesic_tracker import GeodesicTracker1D


def main() -> None:
    """Example GeodesicTracker1D"""
    T = 2000
    t = np.arange(T)

    theta = 0.8 * np.sin(2 * np.pi * 0.01 * t)
    theta += 0.1 * np.sin(2 * np.pi * 0.1 * t)
    r = 1.0 + 0.03 * np.sin(2 * np.pi * 0.003 * t)
    x = r * np.cos(theta) + 0.03 * np.random.randn(T)
    y = r * np.sin(theta) + 0.03 * np.random.randn(T)
    points = np.c_[x, y].astype(float)

    tracker = GeodesicTracker1D(win=64, smooth_u=0.2)
    s, u, xy = tracker.update(points)

    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory in plane (x vs y)")

    idx = np.linspace(0, T - 1, 25, dtype=int)
    scale = 0.15
    for i in idx:
        plt.arrow(
            xy[i, 0],
            xy[i, 1],
            scale * u[i, 0],
            scale * u[i, 1],
            length_includes_head=True,
            head_width=0.02,
            head_length=0.03,
        )

    plt.axis("equal")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, s, alpha=0.7)
    plt.xlabel("time index t")
    plt.ylabel("s(t)")
    plt.title("Tracked 1D coordinate s(t) (geodesic-ish)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
