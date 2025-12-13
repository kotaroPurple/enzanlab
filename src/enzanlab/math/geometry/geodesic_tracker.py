
"""Geodesic tracker for 1D principal curve on the complex plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def local_tangent_pca(points_2d: np.ndarray) -> np.ndarray:
    """Estimate local tangent direction using PCA.

    Args:
        points_2d (np.ndarray): Samples with shape (n, 2).

    Returns:
        np.ndarray: Unit tangent vector (2,).
    """
    if len(points_2d) == 1:
        return np.array([1.0, 0.0], dtype=float)

    X = points_2d - points_2d.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C)
    u = V[:, np.argmax(w)]
    u = u / (np.linalg.norm(u) + 1e-12)
    return u


class GeodesicTracker1D:
    """Incremental tracker for a 1D principal curve (isomap-inspired).

    The tracker consumes complex samples sequentially and maintains the
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

    def update(self, z: complex) -> tuple[float, np.ndarray]:
        """Consume one complex sample and update the geodesic estimate.

        Args:
            z (complex): Incoming complex sample.

        Returns:
            tuple[float, np.ndarray]: Current ``s`` value and tangent vector ``u``.
        """
        x_t = np.array([z.real, z.imag], dtype=float)
        self._x_buffer.append(x_t)

        if len(self._x_buffer) == 1:
            u_t = np.array([1.0, 0.0], dtype=float)
            s_t = 0.0
        else:
            pts = np.vstack(self._x_buffer[-min(self.win, len(self._x_buffer)) :])
            u_t = local_tangent_pca(pts)
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
        return s_t, u_t

    def process(self, z_series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch-process a complex sequence.

        Args:
            z_series (np.ndarray): Complex array of shape (T,).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``s``, ``u``, and ``x`` arrays.
        """
        self.reset()
        for z in z_series:
            self.update(z)
        s = np.asarray(self._s_values, dtype=float)
        u = np.asarray(self._u_values, dtype=float)
        x = np.c_[z_series.real, z_series.imag].astype(float)
        return s, u, x


def main() -> None:
    # ---- synth data (same as previous example) ----
    T = 2000
    t = np.arange(T)

    theta = 0.8 * np.sin(2 * np.pi * 0.01 * t)  # oscillate along an arc
    theta += 0.1 * np.sin(2 * np.pi * 0.1 * t)
    r = 1.0 + 0.03 * np.sin(2 * np.pi * 0.003 * t)  # slow drift in radius
    z = r * np.exp(1j * theta)
    z = z + 0.03 * (np.random.randn(T) + 1j * np.random.randn(T))  # noise

    tracker = GeodesicTracker1D(win=64, smooth_u=0.2)
    s, u, x = tracker.process(z)

    # ---- Visualization 1: trajectory in IQ plane + a few tangent vectors ----
    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Trajectory in IQ plane (Re vs Im)")

    # show sparse tangent arrows (scaled for visibility)
    idx = np.linspace(0, T - 1, 25, dtype=int)
    scale = 0.15
    for i in idx:
        plt.arrow(
            x[i, 0],
            x[i, 1],
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
