from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from .online_dmd import OnlineDMD

__all__ = [
    "OnlineDMDSnapshot",
    "snapshot_from_model",
    "plot_singular_value_spectrogram",
    "animate_eigenvalues_on_complex_plane",
]


@dataclass(frozen=True)
class OnlineDMDSnapshot:
    """Container for OnlineDMD state at a single timestep."""

    time: float
    eigenvalues: NDArray[np.complex128]
    singular_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        eigvals = np.asarray(self.eigenvalues, dtype=np.complex128).reshape(-1)
        singular = np.asarray(self.singular_values, dtype=np.float64).reshape(-1)
        if eigvals.shape[0] != singular.shape[0]:
            raise ValueError(
                "eigenvalues and singular_values must have the same length "
                f"(got {eigvals.shape[0]} vs {singular.shape[0]})."
            )
        object.__setattr__(self, "eigenvalues", eigvals)
        object.__setattr__(self, "singular_values", singular)


def snapshot_from_model(model: OnlineDMD, step_index: int, dt: float = 1.0) -> OnlineDMDSnapshot:
    """Capture the current OnlineDMD state as a snapshot."""
    eigvals, _ = model.eigs()
    if eigvals is None:
        raise ValueError("OnlineDMD instance has no eigenvalues yet.")
    singular = np.asarray(model.S, dtype=np.float64)
    time_value = float(step_index) * float(dt)
    return OnlineDMDSnapshot(
        time=time_value,
        eigenvalues=np.asarray(eigvals, dtype=np.complex128),
        singular_values=singular,
    )


def _compute_frequencies(eigvals: NDArray[np.complex128], dt: float) -> NDArray[np.float64]:
    """Convert eigenvalues to modal frequencies in Hz."""
    eigvals = np.asarray(eigvals, dtype=np.complex128).reshape(-1)
    result = np.full(eigvals.shape, np.nan, dtype=np.float64)
    nonzero_mask = ~np.isclose(eigvals, 0.0)
    safe = eigvals[nonzero_mask]
    if safe.size:
        result[nonzero_mask] = np.imag(np.log(safe)) / (2.0 * np.pi * dt)
    return result


def plot_singular_value_spectrogram(
    history: Sequence[OnlineDMDSnapshot],
    dt: float = 1.0,
    ax: Axes | None = None,
    cmap: str = "viridis",
    point_size: float = 40.0,
) -> Axes:
    """Plot time-frequency scatter where color encodes singular value magnitude."""
    if not history:
        raise ValueError("history must contain at least one snapshot.")

    times: list[np.ndarray] = []
    freqs: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    for snap in history:
        freq = _compute_frequencies(snap.eigenvalues, dt)
        mag = np.abs(snap.singular_values)
        if freq.shape[0] != mag.shape[0]:
            raise ValueError("Each snapshot must have matching eigenvalue and singular value counts.")
        mask = ~np.isnan(freq)
        if not np.any(mask):
            continue
        times.append(np.full(np.count_nonzero(mask), snap.time, dtype=np.float64))
        freqs.append(freq[mask])
        magnitudes.append(mag[mask])

    if not times:
        raise ValueError("All snapshots yielded empty frequency data.")

    time_values = np.concatenate(times)
    freq_values = np.concatenate(freqs)
    magnitude_values = np.concatenate(magnitudes)

    ax = ax or plt.gca()
    vmin = float(np.min(magnitude_values))
    vmax = float(np.max(magnitude_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    ax.scatter(
        time_values,
        freq_values,
        c=magnitude_values,
        s=point_size,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )

    mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    mappable.set_array(magnitude_values)
    colorbar = ax.figure.colorbar(mappable, ax=ax)
    colorbar.set_label("Singular value magnitude")

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Online DMD singular values over time")
    return ax


def animate_eigenvalues_on_complex_plane(
    history: Sequence[OnlineDMDSnapshot],
    ax: Axes | None = None,
    interval: int = 200,
    point_size: float = 60.0,
    cmap: str = "viridis",
    show_unit_circle: bool = True,
) -> FuncAnimation:
    """Create an animation showing eigenvalue evolution on the complex plane."""
    if not history:
        raise ValueError("history must contain at least one snapshot.")

    eigenvalue_series = [snap.eigenvalues for snap in history if snap.eigenvalues.size > 0]
    all_eigvals = np.concatenate(eigenvalue_series) if eigenvalue_series else np.array([], dtype=np.complex128)
    singular_series = [np.abs(snap.singular_values) for snap in history if snap.singular_values.size > 0]
    all_singular = np.concatenate(singular_series) if singular_series else np.array([1.0])

    ax = ax or plt.gca()
    fig = ax.figure

    if all_eigvals.size:
        real_vals = all_eigvals.real
        imag_vals = all_eigvals.imag
        real_span = real_vals.max() - real_vals.min()
        imag_span = imag_vals.max() - imag_vals.min()
        real_margin = max(0.05 * (real_span if real_span > 0 else 1.0), 0.05)
        imag_margin = max(0.05 * (imag_span if imag_span > 0 else 1.0), 0.05)
        ax.set_xlim(real_vals.min() - real_margin, real_vals.max() + real_margin)
        ax.set_ylim(imag_vals.min() - imag_margin, imag_vals.max() + imag_margin)
    else:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    ax.axhline(0.0, color="lightgray", linewidth=0.8)
    ax.axvline(0.0, color="lightgray", linewidth=0.8)
    if show_unit_circle:
        theta = np.linspace(0.0, 2.0 * np.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", linewidth=0.8)

    vmin = float(np.min(all_singular))
    vmax = float(np.max(all_singular))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    first = history[0]
    initial_offsets = np.column_stack((first.eigenvalues.real, first.eigenvalues.imag)) if first.eigenvalues.size else np.empty((0, 2))
    scatter = ax.scatter(
        initial_offsets[:, 0] if initial_offsets.size else [],
        initial_offsets[:, 1] if initial_offsets.size else [],
        c=np.abs(first.singular_values) if first.singular_values.size else [],
        s=point_size,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Online DMD eigenvalue evolution")

    mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    mappable.set_array(all_singular)
    colorbar = fig.colorbar(mappable, ax=ax)
    colorbar.set_label("Singular value magnitude")

    def _update(frame_index: int) -> tuple:
        snap = history[frame_index]
        eigvals = snap.eigenvalues
        offsets = np.column_stack((eigvals.real, eigvals.imag)) if eigvals.size else np.empty((0, 2))
        scatter.set_offsets(offsets)
        magnitudes = np.abs(snap.singular_values)
        scatter.set_array(magnitudes)
        scatter.set_sizes(np.full(magnitudes.shape, point_size))
        time_text.set_text(f"t = {snap.time:.3f}")
        return scatter, time_text

    def _init() -> tuple:
        scatter.set_offsets(initial_offsets)
        scatter.set_array(np.abs(first.singular_values))
        scatter.set_sizes(np.full(first.singular_values.shape, point_size))
        time_text.set_text(f"t = {first.time:.3f}")
        return scatter, time_text

    animation = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(history),
        interval=interval,
        blit=False,
        repeat=True,
    )

    return animation
