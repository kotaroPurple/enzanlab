
import numpy as np


def autocorr_mem(x: np.ndarray, order: int) -> np.ndarray:
    """
    MEM用の自己相関を0..orderまで計算する（複素対応）。
    r[0] は実数に落としておく。
    """
    x = np.asarray(x)
    N = len(x)
    r = np.zeros(order + 1, dtype=complex)
    for k in range(order + 1):
        # vdot は前を共役する: vdot(a, b) = sum(conj(a)*b)
        r[k] = np.vdot(x[:N-k], x[k:]) / (N - k)
    # エネルギーなので実数に寄せる
    r[0] = r[0].real
    return r


def levinson_durbin(r: np.ndarray, order: int, stability_eps: float = 1e-12):
    """
    複素自己相関に対するLevinson–Durbin.
    予測誤差eが丸めでマイナスにならないように安全側に倒す。
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

    return a[1:], e


def mem_spectrum(x: np.ndarray,
                 order: int = 12,
                 n_freq: int = 512,
                 fs: float = 1.0,
                 stability_eps: float = 1e-12):
    """
    MEMスペクトル推定（複素対応）
    """
    r = autocorr_mem(x, order)
    a, noise_var = levinson_durbin(r, order, stability_eps=stability_eps)

    freqs = np.linspace(0, fs/2, n_freq)
    w = 2 * np.pi * freqs / fs

    den = np.ones_like(w, dtype=complex)
    for k in range(1, order + 1):
        den += a[k-1] * np.exp(-1j * w * k)

    # 丸めでごく小さい負になるのを防ぐ
    noise_var = max(noise_var, stability_eps)
    Pxx = noise_var / (np.abs(den) ** 2)

    return freqs, Pxx


if __name__ == "__main__":
    # 動作テスト：複素IQっぽい信号
    fs = 100.0  # 100 Hz サンプリング
    N = 200
    t = np.arange(N) / fs

    f1 = 5.1   # 5 Hz
    f2 = 12.2  # 12 Hz
    # 解析信号っぽく e^{j 2πft} を足す
    x = np.exp(1j * 2*np.pi*f1*t) + 0.6*np.exp(1j * 2*np.pi*f2*t)
    x += 0.1 * (np.random.randn(N) + 1j*np.random.randn(N))  # 複素ノイズ

    freqs, Pxx = mem_spectrum(x, order=16, n_freq=512, fs=fs)

    # あとは matplotlib で plot すれば OK
    import matplotlib.pyplot as plt
    # plt.plot(freqs, 10*np.log10(Pxx))
    plt.plot(freqs, np.sqrt(Pxx))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.show()
