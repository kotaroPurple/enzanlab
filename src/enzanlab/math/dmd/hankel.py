
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def array_to_hankel_matrix(data: NDArray, window_size: int) -> NDArray:
    """Convert a 1D array into a Hankel matrix using sliding windows.

    Args:
        data (NDArray): The 1D array to convert.
        window_size (int): The number of rows in the Hankel matrix.

    Raises:
        ValueError: If the data is not 1D.

    Returns:
        NDArray: Hankel matrix
    """
    if data.ndim != 1:
        raise ValueError()
    return sliding_window_view(data, len(data) - window_size + 1)


def flatten_hankel_matrix(hankel_mat: NDArray) -> NDArray:
    """Flatten a Hankel matrix into a 1D array by averaging anti-diagonals.

    Args:
        hankel_mat (NDArray): The Hankel matrix to flatten.

    Returns:
        NDArray: 1D array
    """
    n_rows, n_cols = hankel_mat.shape
    row_indices = np.arange(n_rows)[:, None]
    col_indices = np.arange(n_cols)[None, :]
    indices = (row_indices + col_indices).ravel()
    counts = np.bincount(indices, minlength=n_rows + n_cols - 1)
    sums = np.zeros_like(counts, dtype=hankel_mat.dtype)
    np.add.at(sums, indices, hankel_mat.ravel())
    return sums / counts


class HankelSignal:
    """Generate Hankel Matrix from 1D array signal in an online manner."""
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._array = np.zeros(self._window_size)

    def initialize(self, values: NDArray[np.floating|np.complexfloating]) -> NDArray:
        """Initialize the Hankel signal with the first window of values.

        Args:
            values (NDArray): The initial values for the Hankel signal.

        Raises:
            ValueError: If the shape of the values does not match the window size.

        Returns:
            NDArray: The initialized Hankel signal.
        """
        if self._array.shape != values.shape:
            raise ValueError()
        self._array[...] = values
        return self._array

    def update(self, value: float|complex) -> NDArray:
        """Update the Hankel signal with a new value."""
        self._array[:-1] = self._array[1:]
        self._array[-1] = value
        return self._array
