
import matplotlib.pyplot as plt
import numpy as np


def local_tangent_pca(points_2d: np.ndarray) -> np.ndarray:
    X = points_2d - points_2d.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C)
    u = V[:, np.argmax(w)]
    u = u / (np.linalg.norm(u) + 1e-12)
    return u


def track_geodesic_1d(z_complex: np.ndarray, win: int = 32, smooth_u: float = 0.0):
    x = np.c_[z_complex.real, z_complex.imag].astype(float)
    T = len(x)
    s = np.zeros(T, dtype=float)
    u = np.zeros((T, 2), dtype=float)

    m0 = min(win, T)
    u_prev = local_tangent_pca(x[:m0])
    if T >= 2 and (x[1] - x[0]) @ u_prev < 0:
        u_prev = -u_prev
    u[0] = u_prev

    for t in range(1, T):
        start = max(0, t - win + 1)
        pts = x[start:t+1]
        u_t = local_tangent_pca(pts)
        if u_t @ u_prev < 0:
            u_t = -u_t
        if smooth_u > 0.0:
            u_t = (1.0 - smooth_u) * u_t + smooth_u * u_prev
            u_t = u_t / (np.linalg.norm(u_t) + 1e-12)
        dx = x[t] - x[t - 1]
        ds = dx @ u_t
        s[t] = s[t - 1] + ds
        u[t] = u_t
        u_prev = u_t

    return s, u, x


def main() -> None:
    # ---- synth data (same as previous example) ----
    T = 2000
    t = np.arange(T)

    theta = 0.8 * np.sin(2*np.pi*0.01*t)         # oscillate along an arc
    theta += 0.1 * np.sin(2*np.pi*0.1*t)
    r = 1.0 + 0.03*np.sin(2*np.pi*0.003*t)       # slow drift in radius
    z = r * np.exp(1j * theta)
    z = z + 0.03*(np.random.randn(T) + 1j*np.random.randn(T))  # noise

    s, u, x = track_geodesic_1d(z, win=64, smooth_u=0.2)

    # ---- Visualization 1: trajectory in IQ plane + a few tangent vectors ----
    plt.figure()
    plt.plot(x[:,0], x[:,1])
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Trajectory in IQ plane (Re vs Im)")

    # show sparse tangent arrows (scaled for visibility)
    idx = np.linspace(0, T-1, 25, dtype=int)
    scale = 0.15
    for i in idx:
        plt.arrow(x[i,0], x[i,1], scale*u[i,0], scale*u[i,1],
                length_includes_head=True, head_width=0.02, head_length=0.03)

    plt.axis("equal")
    plt.tight_layout()

    # ---- Visualization 2: 1D coordinate s(t) ----
    plt.figure()
    # plt.plot(t, theta, alpha=0.7)
    plt.plot(t, s, alpha=0.7)
    plt.xlabel("time index t")
    plt.ylabel("s(t)")
    plt.title("Tracked 1D coordinate s(t) (geodesic-ish)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
