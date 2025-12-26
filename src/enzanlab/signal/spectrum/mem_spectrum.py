
import numpy as np
from numpy.typing import NDArray


def _autocorr_mem(x: NDArray, order: int) -> NDArray:
    """Calculate autocorrelation for MEM from lag 0 to ``order``.

    Args:
        x (NDArray): Input signal of shape (n,).
        order (int): Maximum lag.

    Returns:
        NDArray: Autocorrelation values of shape (order + 1,).
    """
    n_samples = len(x)
    r = np.zeros(order + 1, dtype=complex)
    for k in range(order + 1):
        # vdot は前を共役する: vdot(a, b) = sum(conj(a)*b)
        r[k] = np.vdot(x[: n_samples - k], x[k:]) / (n_samples - k)
    # エネルギーなので実数に寄せる
    r[0] = r[0].real
    return r


def _levinson_durbin(
    r: NDArray,
    order: int,
    stability_eps: float = 1e-12,
) -> tuple[NDArray, float]:
    """Solve complex Levinson-Durbin recursion.

    Args:
        r (NDArray): Autocorrelation values of shape (order + 1,).
        order (int): Model order.
        stability_eps (float): Lower bound for numerical stability.

    Returns:
        tuple[NDArray, float]: AR coefficients of shape (order,) and
            the final prediction error.

    Raises:
        ValueError: If ``stability_eps`` is not positive.
    """
    if stability_eps <= 0:
        raise ValueError("stability_eps must be positive")

    a = np.zeros(order + 1, dtype=complex)  # a[0]は使わない
    e = max(r[0].real, stability_eps)

    for m in range(1, order + 1):
        # 反射係数
        acc = r[m]
        if m > 1:
            # r[m-1], r[m-2], ..., r[1] の順で内積
            acc += np.dot(a[1:m], r[m-1:0:-1])
        if e <= stability_eps:
            break
        k = -acc / e

        # 係数更新
        a_new = a.copy()
        a_new[m] = k
        if m > 1:
            # 複素の場合はここで共役が入る
            a_new[1:m] = a[1:m] + k * np.conj(a[m-1:0:-1])
        a = a_new

        # 誤差更新（丸めで負にならないようにする）
        reduction = 1.0 - (k * np.conj(k)).real
        if reduction < stability_eps:
            reduction = stability_eps
        e = e * reduction

    return a[1:], float(e)


def mem_spectrum(
    x: NDArray,
    order: int = 12,
    n_freq: int = 512,
    fs: float = 1.0,
    stability_eps: float = 1e-12,
) -> tuple[NDArray, NDArray]:
    """Estimate MEM spectrum (complex signal supported).

    Args:
        x (NDArray): Input signal of shape (n,).
        order (int): AR model order.
        n_freq (int): Number of frequency bins.
        fs (float): Sampling frequency in Hz.
        stability_eps (float): Lower bound for numerical stability.

    Returns:
        tuple[NDArray, NDArray]: Frequencies in Hz and power spectrum.
    """
    r = _autocorr_mem(x, order)
    a, noise_var = _levinson_durbin(r, order, stability_eps=stability_eps)

    freqs = np.linspace(0, fs/2, n_freq)
    w = 2 * np.pi * freqs / fs

    den = np.ones_like(w, dtype=complex)
    for k in range(1, order + 1):
        den += a[k-1] * np.exp(-1j * w * k)

    # 丸めでごく小さい負になるのを防ぐ
    noise_var = max(noise_var, stability_eps)
    Pxx = noise_var / (np.abs(den) ** 2)

    return freqs, Pxx
