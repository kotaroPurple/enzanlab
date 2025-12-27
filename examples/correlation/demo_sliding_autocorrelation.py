
"""Demo for STFT-like sliding autocorrelation using FFT."""

import matplotlib.pyplot as plt
import numpy as np

from enzanlab.math.correlation.fft_based import stft_autocorrelation


def main() -> None:
    """Run a demo of sliding autocorrelation and visualize the result."""
    sample_rate = 100.0
    duration = 10.0
    t = np.arange(int(sample_rate * duration)) / sample_rate

    signal = (
        0.7 * np.sin(2.0 * np.pi * 2.0 * t)
        + 0.4 * np.sin(2.0 * np.pi * 4.0 * t)
        + 0.1 * np.random.default_rng(0).standard_normal(t.shape)
    )

    frame_length = 200
    hop_length = 25
    max_lag = 150

    r, lags = stft_autocorrelation(
        signal,
        frame_length=frame_length,
        hop_length=hop_length,
        window="tukey",
        tukey_alpha=0.5,
        max_lag=max_lag,
        detrend="mean",
        normalize=True,
    )

    print(f"Computed autocorrelation: frames={r.shape[0]}, lags={r.shape[1]}")

    time_axis = (np.arange(len(signal)) / sample_rate).astype(float)
    frame_times = (np.arange(r.shape[0]) * hop_length / sample_rate).astype(float)
    lag_seconds = lags / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axes[0].plot(time_axis, signal, color="tab:blue", linewidth=1.0)
    axes[0].set_title("Input signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    im = axes[1].imshow(
        r.T,
        origin="lower",
        aspect="auto",
        extent=[frame_times[0], frame_times[-1], lag_seconds[0], lag_seconds[-1]],
        cmap="magma",
    )
    axes[1].set_title("Sliding autocorrelation (normalized)")
    axes[1].set_xlabel("Frame time [s]")
    axes[1].set_ylabel("Lag [s]")
    fig.colorbar(im, ax=axes[1], label="Correlation")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
