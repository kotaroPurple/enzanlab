
import matplotlib.pyplot as plt
import numpy as np

from enzanlab.signal.spectrum.mem_spectrum import mem_spectrum


def main() -> None:
    """Run a quick demo with a complex IQ-like signal."""
    np.random.seed(1)
    fs = 100.0  # 100 Hz サンプリング
    n_samples = 200
    t = np.arange(n_samples) / fs

    f1 = 5.1  # 5 Hz
    f2 = 12.2  # 12 Hz

    x = np.exp(1j * 2 * np.pi * f1 * t) + 0.6 * np.exp(1j * 2 * np.pi * f2 * t)
    x += 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

    freqs, pxx = mem_spectrum(x, order=16, n_freq=512, fs=fs)

    # plot
    spectrum = np.sqrt(pxx)
    max_idx = int(np.argmax(spectrum))
    max_freq = float(freqs[max_idx])
    max_val = float(spectrum[max_idx])

    # local maxima (exclude edges)
    local_mask = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:])
    local_indices = np.where(local_mask)[0] + 1

    # keep readable: annotate peaks above 10% of the global max, up to 6 peaks
    threshold = 0.1 * max_val
    candidate_indices = local_indices[spectrum[local_indices] >= threshold]
    candidate_indices = candidate_indices[candidate_indices != max_idx]
    candidate_indices = candidate_indices[np.argsort(spectrum[candidate_indices])[::-1]]
    peak_indices = candidate_indices[:6]

    plt.plot(freqs, spectrum)
    plt.scatter([max_freq], [max_val], color="tab:red", zorder=3)
    plt.annotate(
        f"{max_freq:.2f} Hz",
        xy=(max_freq, max_val),
        xytext=(8, 0),
        textcoords="offset points",
        color="tab:red",
    )

    for idx in peak_indices:
        freq = float(freqs[idx])
        val = float(spectrum[idx])
        plt.scatter([freq], [val], color="tab:orange", zorder=3)
        plt.annotate(
            f"{freq:.2f} Hz",
            xy=(freq, val),
            xytext=(8, 5),
            textcoords="offset points",
            color="tab:orange",
        )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.show()


if __name__ == "__main__":
    main()
