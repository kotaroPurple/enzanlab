
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def array_to_hankel_matrix(data: NDArray, window_size: int) -> NDArray:
    if data.ndim != 1:
        raise ValueError()
    return sliding_window_view(data, len(data) - window_size + 1)


def flatten_hankel_matrix(hankel_mat: NDArray) -> NDArray:
    n_rows, n_cols = hankel_mat.shape
    row_indices = np.arange(n_rows)[:, None]
    col_indices = np.arange(n_cols)[None, :]
    indices = (row_indices + col_indices).ravel()
    counts = np.bincount(indices, minlength=n_rows + n_cols - 1)
    sums = np.zeros_like(counts, dtype=hankel_mat.dtype)
    np.add.at(sums, indices, hankel_mat.ravel())
    return sums / counts


class HankelSignal:
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._array = np.zeros(self._window_size)

    def initialize(self, values: NDArray) -> NDArray:
        if self._array.shape != values.shape:
            raise ValueError()
        self._array[...] = values
        return self._array

    def update(self, value: float) -> NDArray:
        self._array[:-1] = self._array[1:]
        self._array[-1] = value
        return self._array


if __name__ == '__main__':
    data = np.arange(30)
    han = array_to_hankel_matrix(data, window_size=10)
    rec = flatten_hankel_matrix(han)
    print(han.shape)
    # print(han)
    print(rec.shape)
    print(rec)
    hankel_signal = HankelSignal(10)
    print(hankel_signal.initialize(data[:10]))
    a = hankel_signal.update(data[10])
    print(a)
    print()
    # complex-valued verification
    phase = np.linspace(0, np.pi, 30)
    complex_data = np.exp(1j * phase)
    complex_han = array_to_hankel_matrix(complex_data, window_size=10)
    complex_rec = flatten_hankel_matrix(complex_han)
    manual_check = []
    row_idx = np.arange(complex_han.shape[0])[:, None]
    col_idx = np.arange(complex_han.shape[1])[None, :]
    for diag_idx in range(complex_han.shape[0] + complex_han.shape[1] - 1):
        mask = (row_idx + col_idx) == diag_idx
        manual_check.append(complex_han[mask].mean())
    manual_check = np.array(manual_check)
    print('Complex Hankel shape:', complex_han.shape)
    print('Flattened complex result (first 5):', complex_rec[:5])
    print('Manual anti-diagonal average matches:', np.allclose(complex_rec, manual_check))
