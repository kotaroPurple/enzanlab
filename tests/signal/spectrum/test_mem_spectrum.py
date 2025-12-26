
import numpy as np

from enzanlab.signal.spectrum.mem_spectrum import mem_spectrum


def test_mem_spectrum_returns_positive_power_for_sine_wave() -> None:
    """MEMスペクトルは正の値を返すことを確認する"""
    np.random.seed(42)
    fs = 100.0
    n_samples = 256
    freq = 8.0
    order = 20
    signal = np.sin(2 * np.pi * freq * np.arange(n_samples) / fs)

    freqs, power = mem_spectrum(signal, order=order, n_freq=256, fs=fs)

    assert np.all(power >= 0.0)
    assert np.max(power) > 0.0

    peak_freq = freqs[np.argmax(power)]
    assert abs(peak_freq - freq) < fs / len(freqs)


def test_mem_spectrum_handles_complex_signals() -> None:
    """MEMが複素信号を扱えることを確認する"""
    fs = 200.0
    n_samples = 512
    freq = 20.0
    signal = np.exp(1j * 2 * np.pi * freq * np.arange(n_samples) / fs)

    freqs, power = mem_spectrum(signal, order=30, n_freq=512, fs=fs)

    assert power[np.argmax(power)] == np.max(power)
    assert np.isfinite(power).all()
    assert np.max(power) > 0.0
