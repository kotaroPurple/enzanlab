"""Test Online DMD stability and frequency tracking."""

import matplotlib.pyplot as plt
import numpy as np

from enzanlab.math.dmd.hankel import HankelSignal, array_to_hankel_matrix, flatten_hankel_matrix
from enzanlab.math.dmd.online_dmd import OnlineDMD
from enzanlab.math.dmd.plot import (
    OnlineDMDSnapshot,
    animate_eigenvalues_on_complex_plane,
    plot_singular_value_spectrogram,
    snapshot_from_model,
)


class SignalGenerator:
    """Generate various test signals for DMD analysis."""

    def __init__(self, sample_rate: float = 100.0):
        """Initialize signal generator.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.time = 0.0

    def reset_time(self):
        """Reset time counter."""
        self.time = 0.0

    def generate_sine_wave(
            self, frequency: float, amplitude: float, noise_level: float = 0.0) -> float:
        """Generate single sine wave sample.

        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        signal = amplitude * np.sin(2 * np.pi * frequency * self.time)
        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_multi_frequency(
            self, frequencies: list[float], amplitudes: list[float], noise_level: float = 0.0) \
                -> float:
        """Generate multi-frequency signal sample.

        Args:
            frequencies: list of frequencies in Hz
            amplitudes: list of amplitudes
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        signal = 0.0
        for freq, amp in zip(frequencies, amplitudes, strict=True):
            signal += amp * np.sin(2 * np.pi * freq * self.time)

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_time_varying(self, noise_level: float = 0.0) -> float:
        """Generate time-varying signal sample.

        Args:
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        t = self.time

        if t < 3:
            # First 3 seconds: 1Hz + 2Hz
            signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t)
        elif t < 6:
            # Next 3 seconds: 1Hz + 3Hz
            signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*3*t)
        else:
            # After 6 seconds: 1Hz + 2Hz + 4Hz
            signal = (np.sin(2*np.pi*1*t)
                + 0.5*np.sin(2*np.pi*2*t)
                + 0.3*np.sin(2*np.pi*4*t))

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_chirp(
            self, f0: float = 1.0, f1: float = 2.5, a0: float = 1.0, a1: float = 1.0, \
                duration: float = 10.0, noise_level: float = 0.0) -> float:
        """Generate chirp signal sample (linear frequency sweep).

        Args:
            f0: Starting frequency in Hz
            f1: Ending frequency in Hz
            a0: Starting amplitude
            a1: Ending amplitude
            duration: Total duration for sweep
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        t = self.time

        # Linear frequency sweep
        amp = a0 + (a1 - a0) * (t / duration)
        if t <= duration:
            # Instantaneous frequency: f(t) = f0 + (f1-f0)*t/duration
            # Phase: φ(t) = 2π * [f0*t + (f1-f0)*t²/(2*duration)]
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
            signal = amp * np.sin(phase)
        else:
            # After duration, use final frequency
            signal = np.sin(2 * np.pi * f1 * t)

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def get_current_time(self) -> float:
        """Get current time."""
        return self.time


def test_chirp_tracking():
    """Test DMD tracking of chirp signal."""
    print("Testing Online DMD with Chirp Signal...")

    # Parameters
    sample_rate = 100
    dt = 1.0 / sample_rate
    duration = 10.0
    samples = int(duration * sample_rate)

    # Generate chirp signal (1Hz to 5Hz over 8 seconds)
    generator = SignalGenerator(sample_rate)
    _signal_data = []
    _time_data = []

    initial_size_rate = 1.5

    f0 = 1.0
    f1 = 1.2
    a0 = 1.0
    a1 = 1.3
    for i in range(samples):
        sample = generator.generate_chirp(
            f0=f0,
            f1=f1,
            a0=a0,
            a1=a1,
            duration=duration,
            noise_level=0.05,
        )
        _signal_data.append(sample)
        _time_data.append(i * dt)

    signal_data = np.array(_signal_data)
    time_data = np.array(_time_data)

    # Test different DMD configurations
    configs = [
        {
            "window_size": 100,
            "max_rank": 4,
            "forgetting_factor": 1.0,
            "tau": 0.01,
            "name": "Standard DMD",
        },
        {
            "window_size": 100,
            "max_rank": 4,
            "forgetting_factor": 0.99,
            "tau": 0.01,
            "name": "Forgetting λ=0.99",
        },
        {
            "window_size": 100,
            "max_rank": 4,
            "forgetting_factor": 0.97,
            "tau": 0.01,
            "name": "Forgetting λ=0.97",
        },
    ]

    results = {}

    for config in configs:
        print(f"Testing {config['name']}...")

        # Initialize DMD
        window_size = config['window_size']
        dmd = OnlineDMD(
            n_dim=window_size,
            r_max=config['max_rank'],
            lambda_=config['forgetting_factor'],
            tau_add=config['tau'],
        )

        # Initialize with first portion
        # init_length = config['window_size'] + 10
        init_length = int(initial_size_rate * sample_rate)
        init_data = array_to_hankel_matrix(signal_data[:init_length], window_size)
        dmd.initialize(init_data)

        hankel = HankelSignal(window_size)
        hankel.initialize(init_data[:, -1])

        # Track frequency evolution
        freq_evolution: list[float] = []
        time_points: list[float] = []
        growth_rates: list[float] = []
        amps: list[float] = []
        snapshots: list[OnlineDMDSnapshot] = []
        current_step = init_length
        try:
            snapshots.append(snapshot_from_model(dmd, step_index=current_step, dt=dt))
        except ValueError:
            pass

        # Process signal and record dominant frequency
        short_samples = 10
        for i in range(init_length, samples, short_samples):  # Every 10 samples (0.1s)
            # new sample and make x vector
            for value in signal_data[i:i + short_samples]:
                new_vector = hankel.update(value)
                dmd.update(new_vector)
                current_step += 1
                try:
                    snapshots.append(snapshot_from_model(dmd, step_index=current_step, dt=dt))
                except ValueError:
                    continue

            # Get current analysis
            try:
                frequencies = dmd.get_mode_frequencies(dt=dt)
                amplitudes = dmd.get_mode_amplitudes()
                growth = dmd.get_mode_growth_rates(dt=dt)

                if len(frequencies) > 0 and len(amplitudes) > 0:
                    # Find dominant frequency
                    amp_magnitudes = np.abs(amplitudes)
                    dominant_idx = np.argmax(amp_magnitudes)

                    freq_evolution.append(abs(frequencies[dominant_idx]))
                    time_points.append(i * dt)
                    growth_rates.append(growth[dominant_idx])
                    amps.append(amp_magnitudes[dominant_idx])

            except Exception as e:
                print(f"Analysis failed at t={i*dt:.1f}s: {e}")

        # Reconstruct per-mode signals (forward and backward) from the last state
        mode_signal_length = sample_rate  # 1 second worth of samples
        forward_mode_states = np.empty((0, window_size, 0), dtype=np.complex128)
        backward_mode_states = np.empty((0, window_size, 0), dtype=np.complex128)
        try:
            # (rank, window_size, signal_length)
            forward_mode_states = dmd.reconstruct_mode_signals(mode_signal_length, backward=False)
            backward_mode_states = dmd.reconstruct_mode_signals(mode_signal_length, backward=True)
        except ValueError as err:
            print(f"Mode reconstruction skipped for {config['name']}: {err}")

        results[config['name']] = {
            'time': time_points,
            'frequency': freq_evolution,
            'growth_rates': growth_rates,
            'amplitudes': amps,
            'config': config,
            'forward_mode_signals': forward_mode_states,
            'backward_mode_signals': backward_mode_states,
            'snapshots': snapshots,
        }

    # Theoretical chirp frequency
    theoretical_time = np.linspace(0, duration, 100)
    theoretical_freq = []
    for t in theoretical_time:
        if t <= duration:
            f_inst = f0 + (f1 - f0) * t / duration  # Linear sweep
        else:
            f_inst = 5.0
        theoretical_freq.append(f_inst)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Online DMD Frequency Tracking Performance', fontsize=16)

    # Original chirp signal
    axes[0, 0].plot(time_data, signal_data)
    axes[0, 0].set_title('Chirp Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    ## reconstruct
    for name, result in results.items():
        forward_signals = result['forward_mode_signals']
        reconstructed = None
        if name.startswith('Starndard'):
            continue
        # backward_signals = result['backward_mode_signals']
        for j in range(forward_signals.shape[0]):
            sub_data = flatten_hankel_matrix(forward_signals[j, :, :]).real  # type: ignore
            if reconstructed is None:
                reconstructed = sub_data.copy()
            else:
                reconstructed += sub_data
        if reconstructed is not None:
            axes[0, 0].plot(
                np.arange(len(reconstructed)) * dt,
                reconstructed,
                color='gray', alpha=0.4)

    # Frequency tracking comparison
    axes[0, 1].plot(theoretical_time, theoretical_freq, 'k--', linewidth=1, label='True Frequency')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['frequency']:
            axes[0, 1].plot(
                result['time'], result['frequency'],
                color=colors[i % len(colors)], marker='o', markersize=3, label=name, alpha=0.5)

    axes[0, 1].set_title('Frequency Tracking Comparison')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Detected Frequency (Hz)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(f0 * 0.8, f1 * 1.2)

    # Amplitudes
    axes[1, 0].set_title('Mode Amplitudes')
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['amplitudes']:
            axes[1, 0].plot(
                result['time'], result['amplitudes'],
                color=colors[i % len(colors)], marker='o', markersize=3,
                label=name, alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Tracking error analysis
    axes[1, 1].set_title('Tracking Error Analysis')
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['frequency']:
            # Interpolate theoretical frequency at measurement times
            theoretical_interp = np.interp(result['time'], theoretical_time, theoretical_freq)
            tracking_error = np.abs(np.array(result['frequency']) - theoretical_interp)

            axes[1, 1].plot(
                result['time'], tracking_error,
                color=colors[i % len(colors)], marker='o', markersize=3,
                label=f"{name} (avg: {np.mean(tracking_error):.2f}Hz)", alpha=0.8)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Tracking Error (Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    reference_name = configs[0]["name"]
    snapshot_history: list[OnlineDMDSnapshot] = results.get(reference_name, {}).get("snapshots", [])
    if snapshot_history:
        spec_fig, spec_ax = plt.subplots(figsize=(10, 4))
        plot_singular_value_spectrogram(snapshot_history, dt=dt, ax=spec_ax)
        spec_fig.tight_layout()

        anim_fig, anim_ax = plt.subplots(figsize=(6, 6))
        animation = animate_eigenvalues_on_complex_plane(snapshot_history, ax=anim_ax)
        anim_fig._online_dmd_animation = animation  # type: ignore

    plt.show()

    # Print analysis summary
    print("\n=== Online DMD Stability Analysis ===")
    for name, result in results.items():
        if result['time'] and result['frequency']:
            theoretical_interp = np.interp(result['time'], theoretical_time, theoretical_freq)
            tracking_error = np.abs(np.array(result['frequency']) - theoretical_interp)
            avg_growth = np.mean(np.abs(result['growth_rates'])) if result['growth_rates'] else 0

            print(f"\n{name}:")
            print(f"  Average tracking error: {np.mean(tracking_error):.3f} Hz")
            print(f"  Max tracking error: {np.max(tracking_error):.3f} Hz")
            print(f"  Average |growth rate|: {avg_growth:.3f}")
            print(f"  Stability: {'Good' if avg_growth < 1.0 else 'Poor'}")

    return results


if __name__ == "__main__":
    results = test_chirp_tracking()
