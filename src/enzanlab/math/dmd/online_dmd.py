
import numpy as np
from numpy.typing import NDArray


class OnlineDMD:
    """Online DMD with Incremental SVD (V 非保持版, NumPy only)

    仕様:
        - ランク増加/維持を残差ノルム rho の相対閾値で判定
        - ランク上限 r_max（超えたら上位特異値でトランケーション）
        - 特異値トリミング: 相対値閾 + 二乗累積エネルギー閾を併用
        - 忘却係数 lambda_ を Σ と H に同時適用（weighted online DMD と整合）
        - 交差項 H は rank-1 更新:  H <- λH + (U^T x_next) v_last^T
            ※ v_last は "小 SVD の右特異行列の最後の列"（np.linalg.svd の Vt の最後の列を使う）

    Example:
        odmd = OnlineDMD(n_dim, r_max=10, lambda_=1.0, tau_add=1e-2)
        odmd.initialize(X)  # X.shape == (n, m)
        for x in stream: odmd.update(x)   # x.shape == (n,)
        Atilde = odmd.A_tilde()
        evals, modes = odmd.eigs()
    """

    def __init__(
            self,
            n_dim: int,
            r_max: int = 10,
            lambda_: float = 1.0,
            tau_add: float = 1e-2,
            tau_rel: float = 1e-3,
            tau_energy: float = 0.99,
            mean_center: bool = True) -> None:
        self.n = int(n_dim)
        self.r_max = int(r_max)
        self.lambda_ = float(lambda_)
        self.tau_add = float(tau_add)
        self.tau_rel = float(tau_rel)
        self.tau_energy = float(tau_energy)
        self.mean_center = bool(mean_center)

        # 保持するのは U, S, C のみ（V は保持しない）
        self.U = np.empty(0, dtype=np.complex128)  # (n, r)
        self.S = np.empty(0)  # (r,)   singular values
        self.H = np.empty(0, dtype=np.complex128)  # (r, r) = U^T X' V への低ランク交差項

        # ストリーミング用バッファ
        self._x_prev = np.empty(0, dtype=np.complex128)  # 次に取り込む列
        self._mean = np.zeros(self.n, dtype=np.complex128)
        self._mean_acc = np.zeros(self.n, dtype=np.complex128)
        self._weight_sum = 0.0

    def __repr__(self) -> str:
        """."""
        current_rank = self.S.shape[0] if self.S.size > 0 else 0
        return (
            f"OnlineDMD: dim={self.n}, rank={current_rank}, \
                r_max={self.r_max}, lambda_={self.lambda_}")

    def initialize(self, X: NDArray) -> None:
        """バッチ初期化。X.shape = (n, m)

        X = U Σ V^T の SVD を計算し、self.U, self.S, self.H を初期化。
        m > 1 の場合は X' = X[:, 1:] を使って self.H = U.T @ X' @ V[1:, :] を初期化。
        _x_prev は X の最後の列に設定。
        """
        X = np.asarray(X, dtype=np.complex128)
        n, m = X.shape
        if n != self.n:
            raise ValueError(f"Input X has incompatible dimension {n}, expected {self.n}")

        # 平均統計量を初期化
        self._mean[:] = 0.0
        self._mean_acc[:] = 0.0
        self._weight_sum = 0.0

        if self.mean_center:
            processed_X = np.empty_like(X, dtype=np.complex128)
            for idx in range(m):
                x_col = X[:, idx]
                self._mean_acc = self.lambda_ * self._mean_acc + x_col
                self._weight_sum = self.lambda_ * self._weight_sum + 1.0
                self._mean = self._mean_acc / max(self._weight_sum, 1e-15)
                processed_X[:, idx] = x_col - self._mean
        else:
            processed_X = X.copy()

        # 忘却係数処理
        coeff_array = self.lambda_ ** np.arange(m - 1, -1, -1)
        weighted_X = processed_X * coeff_array

        # U, S, Vt = np.linalg.svd(X, full_matrices=True)
        U, S, Vt = np.linalg.svd(weighted_X[:, :-1] * self.lambda_, full_matrices=True)
        r = min(self.r_max, S.size)
        self.U = U[:, :r]
        self.S = S[:r]
        V = Vt.T[:, :r]

        if m > 1:
            # 入力X は Y=AX の Y,X を共に含むため 1 列目を落として整列させる
            Y = weighted_X[:, 1:]
            self.H = self.U.conj().T @ Y @ V
        else:
            self.H = np.zeros((r, r), dtype=np.complex128)

        self._x_prev = weighted_X[:, -1].copy()

    def update(self, x_new: NDArray) -> None:
        """新しい観測 x_new (shape=(n,)) を入れ更新する

        x_prev を SVD に取り込み 小 SVD の Vt から v_last を取り出し x_new で C を rank-1 更新
        """
        x_new_raw = np.asarray(x_new, dtype=np.complex128).reshape(-1)
        assert x_new_raw.shape[0] == self.n
        if self.U.size == 0:
            raise RuntimeError("OnlineDMD must be initialized with initialize() before update().")

        if self.mean_center:
            # 平均統計量を更新し中心化した観測を取得
            self._mean_acc = self.lambda_ * self._mean_acc + x_new_raw
            self._weight_sum = self.lambda_ * self._weight_sum + 1.0
            self._mean = self._mean_acc / max(self._weight_sum, 1e-15)
            x_new_proc = x_new_raw - self._mean
        else:
            x_new_proc = x_new_raw

        # ここから毎回: x_prev（中心化済み）を取り込み、x_new で H を更新
        x_prev = self._x_prev
        U, S = self.U, self.S
        r = S.shape[0]

        # 既存空間への射影と残差
        p = U.conj().T @ x_prev  # (r,)
        r_vec = x_prev - U @ p  # (n,)
        rho = np.linalg.norm(r_vec)
        q_vec = r_vec / rho if rho > 0 else np.zeros_like(r_vec)

        # ランク増加の判定（相対閾値 & 上限）
        add_rank = (rho > self.tau_add * max(np.linalg.norm(x_prev), 1e-12)) and (r < self.r_max)

        # 忘却係数適用
        self.H *= self.lambda_
        S *= self.lambda_

        if add_rank:
            K = np.block([
                [np.diag(S), p.reshape(-1, 1)],
                [np.zeros((1, r), dtype=np.complex128), np.array([[rho]])]
            ])  # (r+1, r+1)

            Ut_r, St_r, Vt_r = np.linalg.svd(K, full_matrices=True)  # K = Ut * diag(St) * Vt
            U_aug = np.column_stack((U, q_vec))

            # update H
            z_old = U.conj().T @ x_new_proc  # (r,)
            z_rho = np.vdot(q_vec, x_new_proc)  # scalar
            H_aug = np.block([
                [self.H, z_old[:, None]],
                [np.zeros((1, self.H.shape[1]), dtype=np.complex128), np.array([[z_rho]])]  # type: ignore
            ])
            U_new = U_aug @ Ut_r  # (n, r)
            S_new = St_r
        else:
            # --- ランク維持: 薄い (r x (r+1)) の SVD ---
            Kthin = np.hstack([np.diag(S), p.reshape(-1, 1)])  # (r, r+1)
            Ut_r, St_r, Vt_r = np.linalg.svd(Kthin, full_matrices=False)  # Vt_r: (r, r+1)
            U_new = U @ Ut_r  # (n, r)
            S_new = St_r  # (r,)

            z_old = U.conj().T @ x_new_proc  # (r,)
            H_aug = np.column_stack([self.H, z_old[:, None]])  # (r, r+1)

        # H
        H_new = Ut_r.conj().T @ H_aug @ Vt_r.conj().T

        # ランク縮小（相対特異値 + エネルギー閾）
        U_new, S_new, H_new = self._truncate_rank(U_new, S_new, H_new)

        # 保存
        self.U, self.S, self.H = U_new, S_new, H_new
        self._x_prev = x_new_proc.copy()

    def A_tilde(self) -> None|NDArray:
        """低ランク小行列 Ã = C Σ^{-1} （shape = (r, r)）"""
        if self.U.size == 0:
            return None
        Sinv = 1.0 / np.maximum(self.S, 1e-15)
        return self.H @ np.diag(Sinv)

    def eigs(self):
        """(固有値, DMD モード)。モード = U @ W（Ã W = W Λ）。"""
        A = self.A_tilde()
        if A is None:
            return None, None
        evals, W = np.linalg.eig(A)
        modes = self.U @ W
        return evals, modes

    def get_mode_amplitudes(self, x_init: NDArray|None = None) -> NDArray:
        """Get DMD mode amplitudes for given initial state."""
        _, modes = self.eigs()
        if x_init is None:
            x_init = np.array(self._x_prev)
        if modes is None:
            raise ValueError("No modes available")
        return np.linalg.lstsq(modes, x_init, rcond=None)[0]

    def get_mode_frequencies(self, dt: float = 1.0) -> NDArray:
        """Get frequencies of DMD modes.

        Args:
            dt: Time step size

        Returns:
        Frequencies in Hz (if dt is in seconds)
        """
        eigvals = self.eigs()[0]
        if eigvals is None:
            raise ValueError("No modes available")
        return np.imag(np.log(eigvals)) / (2 * np.pi * dt)

    def get_mode_growth_rates(self, dt: float = 1.0) -> NDArray:
        """Get growth rates of DMD modes.

        Args:
            dt: Time step size

        Returns:
            Growth rates (positive = growing, negative = decaying)
        """
        eigvals = self.eigs()[0]
        if eigvals is None:
            raise ValueError("No modes available")
        return np.real(np.log(eigvals)) / dt

    def reconstruct_signal(self, x_init: NDArray|None, n_samples: int) -> NDArray:
        """Reconstruct a signal from the learned DMD model.

        Args:
            x_init: Initial state vector (shape=(n,)). If None, uses the last stored state.
            n_samples: Number of samples to reconstruct (>= 1).

        Returns:
            Array of reconstructed samples with shape (n, n_samples).
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        evals, modes = self.eigs()
        if evals is None or modes is None:
            raise ValueError("No modes available for reconstruction.")
        if x_init is None:
            x_init = np.array(self._x_prev)
        x_init = np.asarray(x_init, dtype=np.complex128).reshape(-1)
        if x_init.shape[0] != self.n:
            raise ValueError(f"Initial state dimension {x_init.shape[0]} does not match {self.n}.")
        amplitudes = np.linalg.lstsq(modes, x_init, rcond=None)[0]
        time_indices = np.arange(n_samples)
        time_dynamics = (evals[:, None] ** time_indices) * amplitudes[:, None]
        return modes @ time_dynamics

    def reconstruct_mode_signals(
            self,
            n_samples: int,
            x_init: NDArray | None = None,
            backward: bool = False) -> NDArray:
        """Reconstruct per-mode state contributions over time.

        Args:
            n_samples: Number of timesteps to generate (>= 1).
            x_init: Initial state vector. Defaults to the most recent state.
            backward: If True, propagates dynamics backward in time.

        Returns:
            Array with shape (n_modes, n, n_samples) containing complex-valued
            contributions of each mode. Sum along axis=0 to recover the full
            state trajectory. When no modes are available, returns an empty
            array with shape (0, self.n, n_samples).
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        eigvals, modes = self.eigs()
        if eigvals is None or modes is None or eigvals.size == 0:
            return np.empty((0, self.n, n_samples), dtype=np.complex128)

        if x_init is None:
            x_init = np.array(self._x_prev)
        x_init = np.asarray(x_init, dtype=np.complex128).reshape(-1)
        if x_init.shape[0] != self.n:
            raise ValueError(f"Initial state dimension {x_init.shape[0]} does not match {self.n}.")

        amplitudes = np.linalg.lstsq(modes, x_init, rcond=None)[0]
        time_idx = np.arange(n_samples)
        exponent = -time_idx if backward else time_idx

        n_modes = eigvals.size
        dynamics = np.empty((n_modes, n_samples), dtype=np.complex128)
        if backward:
            zero_mask = np.isclose(eigvals, 0.0)
            nonzero_mask = ~zero_mask
            if np.any(nonzero_mask):
                dynamics[nonzero_mask] = eigvals[nonzero_mask, None] ** exponent
            dynamics[zero_mask] = 0.0
            dynamics[zero_mask, 0] = 1.0  # definition: initial sample matches amplitude
        else:
            dynamics = eigvals[:, None] ** exponent

        modal_scalars = amplitudes[:, None] * dynamics
        mode_vectors = modes.T[:, :, None]
        mode_signals = mode_vectors * modal_scalars[:, None, :]
        return mode_signals

    def _truncate_rank(self, U: NDArray, S: NDArray, H: NDArray):
        """Apply relative singular value + cumulative energy truncation."""
        if S.size == 0:
            return U, S, H

        # Relative threshold (ensure at least the leading singular value is kept)
        lead = float(S[0])
        if lead <= 0:
            r_rel = 1
        else:
            rel_mask = (S / lead) >= self.tau_rel
            if not np.any(rel_mask):
                rel_mask[0] = True
            r_rel = int(np.max(np.nonzero(rel_mask))) + 1

        # Energy threshold
        energy = float(np.sum(S ** 2))
        if energy <= 0:
            r_energy = 1
        else:
            cum_energy = np.cumsum(S ** 2) / energy
            idx = np.searchsorted(cum_energy, self.tau_energy, side="left")
            r_energy = int(idx) + 1
        r_energy = max(1, min(r_energy, S.size))

        # Combine constraints with r_max
        r_limit = min(self.r_max, S.size, r_rel)
        r_keep = max(1, min(r_limit, r_energy))

        return U[:, :r_keep], S[:r_keep], H[:r_keep, :r_keep]
