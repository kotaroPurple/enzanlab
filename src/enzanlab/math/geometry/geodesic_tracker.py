
"""Geodesic tracker for a 1D principal curve on the 2D plane."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GeodesicTracker1D:
    """Incremental tracker for a 1D principal curve (isomap-inspired).

    The tracker consumes 2D samples sequentially and maintains the
    intrinsic coordinate ``s(t)`` plus the tangent direction ``u(t)``.
    """

    def __init__(self, win: int = 32, smooth_u: float = 0.0) -> None:
        self.win = win
        self.smooth_u = smooth_u
        self._x_buffer: list[NDArray[np.float64]] = []
        self._s_values: list[float] = []
        self._u_values: list[NDArray[np.float64]] = []
        self._u_prev: NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Clear internal state."""
        self._x_buffer.clear()
        self._s_values.clear()
        self._u_values.clear()
        self._u_prev = None

    def update(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Consume one or more 2D samples and update the geodesic estimate.

        Args:
            points (NDArray[np.float64]): Samples with shape (2,) or (n, 2).

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            Arrays of ``s``, ``u``, and raw points for all data seen so far.
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

    def _update_one(self, x_t: NDArray[np.float64]) -> None:
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
    def _local_tangent_pca(points_2d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate local tangent direction using PCA."""
        if len(points_2d) == 1:
            return np.array([1.0, 0.0], dtype=float)

        X = points_2d - points_2d.mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(len(X) - 1, 1)
        w, V = np.linalg.eigh(C)
        u = V[:, np.argmax(w)]
        u = u / (np.linalg.norm(u) + 1e-12)
        return u
