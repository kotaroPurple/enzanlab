
import numpy as np

from enzanlab.viz.live_plotter import LivePlotter


def main() -> None:
    """Run a demo of LivePlotter."""
    dbg = LivePlotter(
        nrows=1,
        ncols=3,
        figsize=(10, 4),
        show=False,
        autoscale=False,
        width_ratios=(1, 1, 1),
        tight_layout=True,
        pause_time=0.05,
    )

    dbg.add_ax("signal_ax", 0, slice(0, 2))
    dbg.add_ax("point_ax", 0, 2)

    dbg.add_line(
        "signal_ax",
        "signal",
        color="C0",
        label="signal",
        label_position="upper_left",
    )
    dbg.add_scatter(
        "point_ax",
        "point",
        s=5,
        c="C2",
        label="recent points",
        label_position="lower_right",
    )

    dbg.set_limits("signal_ax", xlim=(0.0, 10.0), ylim=(-1.0, 1.0))
    dbg.set_limits("point_ax", xlim=(0.0, 10.0), ylim=(-1.0, 1.0))

    x = np.linspace(0, 2 * np.pi, 300)

    for k in range(1, len(x)):
        # --- 処理 ---
        y = np.sin(x[:k])

        # --- 表示 ----
        if k % 5 == 0:
            dbg.update(
                {
                    "signal": (x[:k], y),
                    "point": (x[:k][-30:], y[-30:]),
                }
            )

        if dbg.is_open() is False:
            break


if __name__ == "__main__":
    main()
