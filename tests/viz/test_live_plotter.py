"""Tests for LivePlotter."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from enzanlab.viz.live_plotter import LivePlotter


def test_live_plotter_init_validation() -> None:
    """Raises for invalid initialization parameters."""
    with pytest.raises(ValueError, match="nrows and ncols must be > 0"):
        LivePlotter(nrows=0, ncols=1, show=False)
    with pytest.raises(ValueError, match="pause_time must be >= 0"):
        LivePlotter(show=False, pause_time=-0.1)
    with pytest.raises(ValueError, match="draw_every_n must be > 0"):
        LivePlotter(show=False, draw_every_n=0)
    with pytest.raises(ValueError, match="blit=True is not supported with autoscale=True"):
        LivePlotter(show=False, blit=True, autoscale=True)


def test_add_ax_validates_name_and_overlap() -> None:
    """Validates duplicate names and overlapping regions."""
    dbg = LivePlotter(nrows=2, ncols=2, show=False, pause_time=0.0)
    dbg.add_ax("left", slice(0, 2), 0)

    with pytest.raises(KeyError, match="Axis name already exists"):
        dbg.add_ax("left", 0, 1)
    with pytest.raises(ValueError, match="overlaps an existing axis"):
        dbg.add_ax("overlap", 1, 0)
    with pytest.raises(IndexError, match="row index out of range"):
        dbg.add_ax("bad_index", 10, 1)


def test_add_line_scatter_text_and_update() -> None:
    """Updates line, scatter and text artists correctly."""
    dbg = LivePlotter(nrows=1, ncols=1, show=False, pause_time=0.0, flush_events=False)
    dbg.add_ax("ax", 0, 0)
    line = dbg.add_line("ax", "line")
    scatter = dbg.add_scatter("ax", "scatter")
    text = dbg.add_text("ax", "text", 0.0, 0.0, "init", transform="axes")

    dbg.update(
        {
            "line": (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            "scatter": (
                np.array([0.25, 0.5]),
                np.array([0.75, 1.25]),
                np.array([0.1, 0.9]),
                (0.0, 1.0),
            ),
            "text": (0.2, 0.8, "updated"),
        }
    )

    np.testing.assert_allclose(line.get_xdata(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(line.get_ydata(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(scatter.get_offsets(), np.array([[0.25, 0.75], [0.5, 1.25]]))
    np.testing.assert_allclose(scatter.get_clim(), np.array([0.0, 1.0]))
    assert text.get_text() == "updated"
    np.testing.assert_allclose(text.get_position(), (0.2, 0.8))
    assert text.get_transform() == dbg.axes["ax"].transAxes


def test_update_payload_validation_for_line_scatter_text() -> None:
    """Raises when payload format is invalid for each artist type."""
    dbg = LivePlotter(nrows=1, ncols=1, show=False, pause_time=0.0, flush_events=False)
    dbg.add_ax("ax", 0, 0)
    dbg.add_line("ax", "line")
    dbg.add_scatter("ax", "scatter")
    dbg.add_text("ax", "text", 0.0, 0.0, "init")

    with pytest.raises(ValueError, match="Invalid payload length for line"):
        dbg.update({"line": (np.array([0.0]), np.array([1.0]), "bad")})
    with pytest.raises(ValueError, match="Invalid payload length for scatter"):
        dbg.update({"scatter": (np.array([0.0]),)})
    with pytest.raises(ValueError, match="Invalid payload length for text"):
        dbg.update({"text": (0.0, 1.0)})
    with pytest.raises(ValueError, match="third element must be str"):
        dbg.update({"text": (0.0, 1.0, np.array([1.0]))})


def test_draw_every_n_throttles_draw_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Draws only at configured update intervals."""
    dbg = LivePlotter(
        nrows=1,
        ncols=1,
        show=False,
        pause_time=0.0,
        flush_events=False,
        draw_every_n=2,
    )
    dbg.add_ax("ax", 0, 0)
    dbg.add_line("ax", "line")

    draw_calls: list[int] = []

    def _draw_idle() -> None:
        draw_calls.append(1)

    monkeypatch.setattr(dbg.fig.canvas, "draw_idle", _draw_idle)
    dbg.update({"line": (np.array([0.0]), np.array([1.0]))})
    dbg.update({"line": (np.array([0.0, 1.0]), np.array([1.0, 2.0]))})
    assert len(draw_calls) == 1


def test_label_creation_and_position_update() -> None:
    """Creates legend from labels and updates legend position."""
    dbg = LivePlotter(nrows=1, ncols=1, show=False, pause_time=0.0)
    dbg.add_ax("ax", 0, 0)
    dbg.add_line("ax", "line", label="signal", label_position="upper_left")
    dbg.set_label_position("ax", "lower_right")

    legend = dbg.axes["ax"].get_legend()
    assert legend is not None
    labels = [entry.get_text() for entry in legend.get_texts()]
    assert labels == ["signal"]
