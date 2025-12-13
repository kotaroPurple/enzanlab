
"""Geodesic tracker for a 1D principal curve on the 2D plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class GeodesicTracker1D:
    """Incremental tracker for a 1D principal curve (isomap-inspired).

    The tracker consumes 2D samples sequentially and maintains the
    intrinsic coordinate ``s(t)`` plus the tangent direction ``u(t)``.
    """

    def __init__(self, win: int = 32, smooth_u: float = 0.0) -> None:
        self.win = win
        self.smooth_u = smooth_u
        self._x_buffer: list[np.ndarray] = []
        self._s_values: list[float] = []
        self._u_values: list[np.ndarray] = []
        self._u_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear internal state."""
        self._x_buffer.clear()
        self._s_values.clear()
        self._u_values.clear()
        self._u_prev = None

    def update(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consume one or more 2D samples and update the geodesic estimate.

        Args:
            points (np.ndarray): Samples with shape (2,) or (n, 2).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of ``s``, ``u``,
            and raw points for all data seen so far.
        """
        pts_arr = np.asarray(points, dtype=float)
        if pts_arr.ndim == 1:
            pts_arr = np.expand_dims(pts_arr, axis=0)

        for x_t in pts_arr:
            self._update_one(x_t)

        s = np.asarray(self._s_values, dtype=float)
        u = np.asarray(self._u_values, dtype=float)
        x = np.asarray(self._x_buffer, dtype=float)
        return s, u, x

    def _update_one(self, x_t: np.ndarray) -> None:
        """Update state with a single 2D sample."""
        self._x_buffer.append(x_t)

        if len(self._x_buffer) == 1:
            u_t = np.array([1.0, 0.0], dtype=float)
            s_t = 0.0
        else:
            pts = np.vstack(self._x_buffer[-min(self.win, len(self._x_buffer)) :])
            u_t = self._local_tangent_pca(pts)
            if self._u_prev is not None and u_t @ self._u_prev < 0:
                u_t = -u_t
            if self.smooth_u > 0.0 and self._u_prev is not None:
                u_t = (1.0 - self.smooth_u) * u_t + self.smooth_u * self._u_prev
                u_t = u_t / (np.linalg.norm(u_t) + 1e-12)

            dx = self._x_buffer[-1] - self._x_buffer[-2]
            ds = float(dx @ u_t)
            s_t = self._s_values[-1] + ds

        self._u_prev = u_t
        self._s_values.append(s_t)
        self._u_values.append(u_t)

    @staticmethod
    def _local_tangent_pca(points_2d: np.ndarray) -> np.ndarray:
        """Estimate local tangent direction using PCA."""
        if len(points_2d) == 1:
            return np.array([1.0, 0.0], dtype=float)

        X = points_2d - points_2d.mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(len(X) - 1, 1)
        w, V = np.linalg.eigh(C)
        u = V[:, np.argmax(w)]
        u = u / (np.linalg.norm(u) + 1e-12)
        return u


def main() -> None:
    # ---- synth data (same as previous example) ----
    T = 2000
    t = np.arange(T)

    theta = 0.8 * np.sin(2 * np.pi * 0.01 * t)  # oscillate along an arc
    theta += 0.1 * np.sin(2 * np.pi * 0.1 * t)
    r = 1.0 + 0.03 * np.sin(2 * np.pi * 0.003 * t)  # slow drift in radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x += 0.03 * np.random.randn(T)  # noise
    y += 0.03 * np.random.randn(T)
    points = np.c_[x, y].astype(float)

    tracker = GeodesicTracker1D(win=64, smooth_u=0.2)
    s, u, xy = tracker.update(points)

    # ---- Visualization 1: trajectory in IQ plane + a few tangent vectors ----
    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory in plane (x vs y)")

    # show sparse tangent arrows (scaled for visibility)
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

    # ---- Visualization 2: 1D coordinate s(t) ----
    plt.figure()
    plt.plot(t, s, alpha=0.7)
    plt.xlabel("time index t")
    plt.ylabel("s(t)")
    plt.title("Tracked 1D coordinate s(t) (geodesic-ish)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
