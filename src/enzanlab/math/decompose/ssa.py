
"""Singular Spectrum Analysis (SSA) implementation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class SSA:
    """Singular Spectrum Analysis (SSA) model.

    Args:
        window_length: Window length for Hankel embedding.
    """
    def __init__(self, window_length: int) -> None:
        self.window_length = int(window_length)
        self.u: NDArray[np.floating] | None = None
        self.s: NDArray[np.floating] | None = None
        self.vt: NDArray[np.floating] | None = None
        self._hankel_matrix: NDArray[np.floating] | None = None

    def _hankel(
            self, series: NDArray[np.floating], window_length: int) -> NDArray[np.floating]:
        """Build a Hankel matrix from a 1D series.

        Args:
            series: Input 1D series.
            window_length: Window length for embedding.

        Returns:
            Hankel matrix with shape (window_length, column_count).
        """
        series = np.asarray(series)
        series_length = len(series)
        column_count = series_length - window_length + 1
        if window_length <= 0 or column_count <= 0:
            raise ValueError("window_length must satisfy 1 <= window_length <= len(series)")
        return np.column_stack([series[i:i + window_length] for i in range(column_count)])

    def _diagonal_averaging(self, matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reconstruct a series by diagonal averaging.

        Args:
            matrix: Matrix to average along anti-diagonals.

        Returns:
            Reconstructed 1D series.
        """
        row_count, column_count = matrix.shape
        series_length = row_count + column_count - 1
        diagonal_indices = np.add.outer(
            np.arange(row_count, dtype=int),
            np.arange(column_count, dtype=int),
        ).ravel()
        flattened = matrix.ravel()
        values = np.bincount(diagonal_indices, weights=flattened, minlength=series_length)
        weights = np.bincount(diagonal_indices, minlength=series_length)
        return values / weights

    def fit(self, series: NDArray[np.floating]) -> "SSA":
        """Fit the SSA model to a 1D series.

        Args:
            series: Input 1D series.

        Returns:
            Fitted SSA instance.
        """
        self._hankel_matrix = self._hankel(series, self.window_length)
        self.u, self.s, self.vt = np.linalg.svd(self._hankel_matrix, full_matrices=False)
        return self

    def reconstruct(self, indices: int|Sequence[int]|NDArray[np.integer]) -> NDArray[np.floating]:
        """Reconstruct a component or components from the SSA decomposition.

        Args:
            indices: Singular component index or indices.

        Returns:
            Reconstructed 1D series from selected components.
        """
        if self.u is None or self.s is None or self.vt is None:
            raise ValueError("Call fit(x) before reconstruct().")
        if isinstance(indices, int):
            component_index = int(indices)
            reconstructed_matrix = self.s[component_index] * np.outer(
                self.u[:, component_index],
                self.vt[component_index],
            )
        else:
            component_indices = np.asarray(indices, dtype=int)
            reconstructed_matrix = (
                self.u[:, component_indices] * self.s[component_indices]
            ) @ self.vt[component_indices]
        return self._diagonal_averaging(reconstructed_matrix)

    def get_svd(self) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Get the SVD components of the fitted Hankel matrix.

        Returns:
            Tuple of (U, S, VT) from the SVD.
        """
        if self.u is None or self.s is None or self.vt is None:
            raise ValueError("Call fit(x) before get_svd().")
        return self.u, self.s, self.vt
