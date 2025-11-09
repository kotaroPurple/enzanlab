"""Hankel-aware Online DMD utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hankel import HankelSignal, array_to_hankel_matrix, flatten_hankel_matrix
from .online_dmd import OnlineDMD


@dataclass(slots=True)
class HankelOnlineDMDConfig:
    """Configuration for :class:`HankelOnlineDMD`.

    The defaults mirror :class:`OnlineDMD` so the wrapper behaves identically when the
    Hankel lifting is ignored.
    """

    window_size: int
    r_max: int = 10
    lambda_: float = 1.0
    tau_add: float = 1e-2
    tau_rel: float = 1e-3
    tau_energy: float = 0.99
    mean_center: bool = True


class HankelOnlineDMD:
    """Online DMD tailored for streaming 1D signals via Hankel lifting.

    The class mirrors the workflow implemented in ``examples/dmd/simple_onlinedmd.py``:

    1. A univariate time series is lifted to a Hankel matrix.
    2. The matrix initializes :class:`OnlineDMD`.
    3. Each new sample updates the trailing Hankel window and therefore the DMD state.

    Args:
        config: Hyper-parameters that are forwarded to :class:`OnlineDMD`.
    """

    def __init__(self, config: HankelOnlineDMDConfig) -> None:
        self.window_size = int(config.window_size)
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2 for Hankel lifting.")

        self._dmd = OnlineDMD(
            n_dim=self.window_size,
            r_max=config.r_max,
            lambda_=config.lambda_,
            tau_add=config.tau_add,
            tau_rel=config.tau_rel,
            tau_energy=config.tau_energy,
            mean_center=config.mean_center,
        )
        self._signal = HankelSignal(self.window_size)
        self._initialized = False
        self._samples_seen = 0

    @property
    def model(self) -> OnlineDMD:
        """Expose the underlying :class:`OnlineDMD` instance."""
        return self._dmd

    @property
    def initialized(self) -> bool:
        """Whether :meth:`initialize` has been called."""
        return self._initialized

    def initialize(self, initial_series: Sequence[float | complex]) -> None:
        """Bootstrap the online model using an initial 1D series.

        Args:
            initial_series: Samples used to build the first Hankel matrix. The length must
                be at least ``window_size + 1`` so that we can form paired snapshots for DMD.
        """
        data = np.asarray(initial_series)
        if data.ndim != 1:
            raise ValueError("initial_series must be 1D.")
        if data.size < self.window_size + 1:
            raise ValueError(
                "Need at least window_size + 1 samples to initialize OnlineDMD.")

        hankel_matrix = array_to_hankel_matrix(data, self.window_size)
        hankel_matrix = np.asarray(hankel_matrix, dtype=np.complex128)
        self._dmd.initialize(hankel_matrix)
        self._signal.initialize(hankel_matrix[:, -1])
        self._initialized = True
        self._samples_seen = data.size

    def update(self, value: float | complex) -> NDArray:
        """Insert a new scalar sample and update the Online DMD state."""
        if not self._initialized:
            raise RuntimeError("Call initialize before streaming updates.")

        vector = self._signal.update(value)
        self._dmd.update(vector)
        self._samples_seen += 1
        return vector

    def update_many(self, values: Iterable[float | complex]) -> None:
        """Convenience helper that feeds multiple values sequentially."""
        for value in values:
            self.update(value)

    def get_mode_frequencies(self, dt: float = 1.0) -> NDArray:
        """Delegate to :meth:`OnlineDMD.get_mode_frequencies`."""
        return self._dmd.get_mode_frequencies(dt=dt)

    def get_mode_amplitudes(self) -> NDArray:
        """Delegate to :meth:`OnlineDMD.get_mode_amplitudes`."""
        return self._dmd.get_mode_amplitudes()

    def get_mode_growth_rates(self, dt: float = 1.0) -> NDArray:
        """Delegate to :meth:`OnlineDMD.get_mode_growth_rates`."""
        return self._dmd.get_mode_growth_rates(dt=dt)

    def reconstruct_mode_time_series(
            self, n_samples: int, backward: bool = False) -> NDArray:
        """Flatten reconstructed Hankel states into time-domain signals.

        Each mode first reconstructs its Hankel state trajectory via
        :meth:`OnlineDMD.reconstruct_mode_signals`. The anti-diagonal average collapses the
        Hankel representation back to the original time axis, yielding
        ``(n_modes, window_size + n_samples - 1)`` samples.

        Args:
            n_samples: Number of Hankel columns to generate per mode.
            backward: When ``True``, integrates the dynamics backward in time.

        Returns:
            Array of flattened per-mode signals. When the Online DMD model has no modes yet
            (e.g. right after initialization), an empty array is returned.
        """
        states = self._dmd.reconstruct_mode_signals(
            n_samples=n_samples, backward=backward)
        if states.size == 0:
            return np.empty((0, self.window_size + n_samples - 1), dtype=np.complex128)

        flattened = np.empty(
            (states.shape[0], self.window_size + n_samples - 1),
            dtype=np.complex128,
        )
        for idx, mode_state in enumerate(states):
            flattened[idx] = flatten_hankel_matrix(mode_state.astype(np.complex128))
        return flattened

    def time_index(self) -> int:
        """Return how many samples (including the initialization) have been processed."""
        if not self._initialized:
            return 0
        return self._samples_seen
