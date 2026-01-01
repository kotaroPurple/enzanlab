
import time

import numpy as np

from enzanlab.viz.live_plotter import LivePlotter


def main() -> None:
    """Run a demo of LivePlotter."""
    dbg = LivePlotter(layout=(1, 2), figsize=(10, 4), show=False, autoscale=False)

    dbg.add_line(0, "signal", color="C0")
    dbg.add_scatter(1, "point", s=5, c="C2")

    dbg.set_limits(0, xlim=(0.0, 10.0), ylim=(-1.0, 1.0))
    dbg.set_limits(1, xlim=(0.0, 10.0), ylim=(-1.0, 1.0))

    x = np.linspace(0, 2*np.pi, 300)

    for k in range(1, len(x)):
        # --- 処理 ---
        y = np.sin(x[:k])
        time.sleep(0.01)

        # --- 表示 ----
        if k % 5 == 0:
            dbg.update({
                "signal": (x[:k], y),
                "point": (x[:k][-30:], y[-30:])
            })

        if dbg.is_open() is False:
            break


if __name__ == "__main__":
    main()
