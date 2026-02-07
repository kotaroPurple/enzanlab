
"""Lightweight live debugger for Matplotlib visualizations."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

ColorRange = tuple[float, float]
ColorSpec = str | Sequence[float] | np.ndarray
AxisKey = str
LabelPosition = Literal["upper_left", "lower_left", "upper_right", "lower_right"]

_LEGEND_LOCATIONS: dict[LabelPosition, str] = {
    "upper_left": "upper left",
    "lower_left": "lower left",
    "upper_right": "upper right",
    "lower_right": "lower right",
}


class LivePlotter:
    """Lightweight wrapper for interactive Matplotlib updates."""

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (8, 6),
        *,
        show: bool = True,
        autoscale: bool = False,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
        tight_layout: bool = False,
        pause_time: float = 0.001,
        draw_every_n: int = 1,
        flush_events: bool = True,
        blit: bool = False,
    ) -> None:
        """Initialize a figure and axes layout.

        Args:
            nrows: Number of GridSpec rows.
            ncols: Number of GridSpec columns.
            figsize: Figure size in inches.
            show: If True, show the figure immediately (non-blocking).
            autoscale: If True, autoscale axes after each update.
            width_ratios: Relative column widths for the grid.
            height_ratios: Relative row heights for the grid.
            tight_layout: If True, call fig.tight_layout() after setup.
            pause_time: Pause duration [s] per update. Set 0 to disable pause.
            draw_every_n: Render every N update() calls.
            flush_events: If True, flush GUI events after rendering.
            blit: If True, use blitting when supported for faster updates.
        """
        if nrows <= 0 or ncols <= 0:
            raise ValueError("nrows and ncols must be > 0")
        if pause_time < 0:
            raise ValueError("pause_time must be >= 0")
        if draw_every_n <= 0:
            raise ValueError("draw_every_n must be > 0")
        if blit and autoscale:
            raise ValueError("blit=True is not supported with autoscale=True")

        self._nrows = nrows
        self._ncols = ncols
        self.fig = plt.figure(figsize=figsize)
        grid = self.fig.add_gridspec(
            nrows,
            ncols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        self._grid = grid
        self._occupied = np.zeros((nrows, ncols), dtype=bool)

        self.axes: dict[str, plt.Axes] = {}
        self.artists: dict[str, Any] = {}
        self.artist_axes: dict[str, plt.Axes] = {}
        self._legend_locs: dict[str, str] = {}
        self.autoscale = autoscale
        self._tight_layout = tight_layout
        self._pause_time = pause_time
        self._draw_every_n = draw_every_n
        self._flush_events = flush_events
        self._blit = blit
        self._update_count = 0
        self._blit_supported = bool(getattr(self.fig.canvas, "supports_blit", False))
        self._use_blit = self._blit and self._blit_supported
        self._blit_backgrounds: dict[plt.Axes, Any] = {}
        self._blit_invalidated = True

        if tight_layout:
            self.fig.tight_layout()

        if show:
            plt.show(block=False)

    def add_ax(self, name: str, row: slice | int, col: slice | int) -> plt.Axes:
        """Add a new axis using GridSpec slices.

        Args:
            name: Axis name to register (used as key in axes).
            row: Row index or slice for spanning rows.
            col: Column index or slice for spanning columns.

        Returns:
            The created Matplotlib Axes.
        """
        if name in self.axes:
            raise KeyError(f"Axis name already exists: {name!r}")

        row_indices = self._selector_to_indices(row, self._nrows, "row")
        col_indices = self._selector_to_indices(col, self._ncols, "col")

        if self._occupied[np.ix_(row_indices, col_indices)].any():
            raise ValueError("Requested GridSpec region overlaps an existing axis")

        ax = self.fig.add_subplot(self._grid[row, col])
        self.axes[name] = ax
        self._occupied[np.ix_(row_indices, col_indices)] = True
        self._invalidate_blit()
        if self._tight_layout:
            self.fig.tight_layout()
        return ax

    def add_line(
        self,
        ax: AxisKey,
        name: str,
        *,
        label: str | None = None,
        label_position: LabelPosition | None = None,
        **kwargs: Any,
    ) -> Any:
        """Register a line artist on the specified axis.

        Args:
            ax: Axis key name registered by add_ax().
            name: Artist name for updates.
            label: Legend label for this artist.
            label_position: Corner for legend placement on this axis.
            **kwargs: Passed through to Matplotlib's plot.

        Returns:
            The created line artist.
        """
        if label is not None and "label" in kwargs:
            raise ValueError("Specify either label argument or kwargs['label'], not both")
        if label is not None:
            kwargs["label"] = label
        target_ax = self._resolve_ax(ax)
        line, = target_ax.plot([], [], **kwargs)
        line.set_animated(self._use_blit)
        self.artists[name] = line
        self.artist_axes[name] = target_ax
        self._update_legend(ax, target_ax, label_position=label_position)
        self._invalidate_blit()
        return line

    def add_scatter(
        self,
        ax: AxisKey,
        name: str,
        *,
        label: str | None = None,
        label_position: LabelPosition | None = None,
        **kwargs: Any,
    ) -> Any:
        """Register a scatter artist on the specified axis.

        Args:
            ax: Axis key name registered by add_ax().
            name: Artist name for updates.
            label: Legend label for this artist.
            label_position: Corner for legend placement on this axis.
            **kwargs: Passed through to Matplotlib's scatter.

        Returns:
            The created scatter artist.
        """
        if label is not None and "label" in kwargs:
            raise ValueError("Specify either label argument or kwargs['label'], not both")
        if label is not None:
            kwargs["label"] = label
        target_ax = self._resolve_ax(ax)
        sc = target_ax.scatter([], [], **kwargs)
        sc.set_animated(self._use_blit)
        self.artists[name] = sc
        self.artist_axes[name] = target_ax
        self._update_legend(ax, target_ax, label_position=label_position)
        self._invalidate_blit()
        return sc

    def set_label_position(self, ax: AxisKey, position: LabelPosition) -> None:
        """Set legend corner for labels on an axis.

        Args:
            ax: Axis key name registered by add_ax().
            position: Legend corner ("upper_left", "lower_left", "upper_right", "lower_right").
        """
        target_ax = self._resolve_ax(ax)
        self._update_legend(ax, target_ax, label_position=position)
        self._invalidate_blit()

    def set_limits(
        self,
        ax: AxisKey,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Set axis limits explicitly.

        Args:
            ax: Axis key name registered by add_ax().
            xlim: X-axis limits as (min, max).
            ylim: Y-axis limits as (min, max).
        """
        target_ax = self._resolve_ax(ax)
        if xlim is not None:
            target_ax.set_xlim(*xlim)
        if ylim is not None:
            target_ax.set_ylim(*ylim)
        self._invalidate_blit()

    def set_title(self, ax: AxisKey, title: str) -> None:
        """Set axis title explicitly.

        Args:
            ax: Axis key name registered by add_ax().
            title: Title text to set.
        """
        target_ax = self._resolve_ax(ax)
        target_ax.set_title(title)
        self._invalidate_blit()

    def is_open(self) -> bool:
        """Check whether the figure window is still open."""
        return plt.fignum_exists(self.fig.number)

    def _resolve_ax(self, ax: AxisKey) -> plt.Axes:
        if ax in self.axes:
            return self.axes[ax]
        available = ", ".join(sorted(self.axes.keys()))
        raise KeyError(f"Unknown axis key: {ax!r}. Available axes: [{available}]")

    def _update_legend(
        self,
        ax_key: AxisKey,
        axis: plt.Axes,
        *,
        label_position: LabelPosition | None = None,
    ) -> None:
        if label_position is not None:
            self._legend_locs[ax_key] = _LEGEND_LOCATIONS[label_position]
        loc = self._legend_locs.get(ax_key, _LEGEND_LOCATIONS["upper_right"])

        handles, labels = axis.get_legend_handles_labels()
        valid = [
            (handle, text)
            for handle, text in zip(handles, labels, strict=False)
            if text and not text.startswith("_")
        ]
        legend = axis.get_legend()
        if not valid:
            if legend is not None:
                legend.remove()
            return
        valid_handles, valid_labels = zip(*valid, strict=False)
        axis.legend(valid_handles, valid_labels, loc=loc)
        self._invalidate_blit()

    def _selector_to_indices(
        self,
        selector: slice | int,
        size: int,
        label: str,
    ) -> list[int]:
        if isinstance(selector, int):
            index = selector if selector >= 0 else size + selector
            if index < 0 or index >= size:
                raise IndexError(f"{label} index out of range: {selector}")
            return [index]
        if isinstance(selector, slice):
            values = list(range(*selector.indices(size)))
            if not values:
                raise ValueError(f"{label} slice selects no indices: {selector!r}")
            return values
        raise TypeError(f"{label} selector must be int or slice, got {type(selector)!r}")

    def update(
        self,
        data: Mapping[
            str,
            tuple[np.ndarray, np.ndarray]
            | tuple[np.ndarray, np.ndarray, ColorSpec]
            | tuple[np.ndarray, np.ndarray, ColorSpec, ColorRange],
        ],
    ) -> None:
        """Update artists with new data.

        Args:
            data: Mapping from artist name to data payload.
                - (x, y) for lines or scatters.
                - (x, y, color) for scatters with colors.
                - (x, y, color, (vmin, vmax)) to update color range.
        """
        if not plt.fignum_exists(self.fig.number):
            return
        updated_axes: set[plt.Axes] = set()
        for name, payload in data.items():
            if name not in self.artists:
                continue

            if len(payload) < 2 or len(payload) > 4:
                raise ValueError(
                    f"Invalid payload length for {name!r}: {len(payload)} (expected 2-4)"
                )
            x = payload[0]
            y = payload[1]
            color = payload[2] if len(payload) >= 3 else None
            color_range = payload[3] if len(payload) == 4 else None
            if color_range is not None:
                if len(color_range) != 2:
                    raise ValueError(
                        f"Invalid color range for {name!r}: {color_range!r} "
                        "(expected finite (vmin, vmax))"
                    )
                vmin, vmax = color_range
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    raise ValueError(
                        f"Invalid color range for {name!r}: {color_range!r} "
                        "(expected finite (vmin, vmax))"
                    )

            artist = self.artists[name]
            if hasattr(artist, "set_data"):
                artist.set_data(x, y)
            elif hasattr(artist, "set_offsets"):
                x = np.asarray(x)
                y = np.asarray(y)
                if x.size == 0 or y.size == 0:
                    artist.set_offsets(np.empty((0, 2)))
                else:
                    artist.set_offsets(np.c_[x, y])
                if color is not None:
                    self._apply_scatter_color(artist, color, color_range)
            updated_axes.add(self.artist_axes[name])

            if self.autoscale:
                ax = self.artist_axes.get(name)
                if ax is not None:
                    if hasattr(artist, "get_offsets"):
                        offsets = artist.get_offsets()
                        if offsets.size > 0:
                            ax.update_datalim(offsets)
                    else:
                        ax.relim()
                    ax.autoscale_view()
                    self._invalidate_blit()

        self._update_count += 1
        if self._update_count % self._draw_every_n != 0:
            return

        if self._can_use_blit() and updated_axes:
            self._draw_with_blit(updated_axes)
        else:
            self.fig.canvas.draw_idle()
            if self._flush_events:
                self.fig.canvas.flush_events()

        if self._pause_time > 0:
            plt.pause(self._pause_time)

    def _can_use_blit(self) -> bool:
        return self._use_blit

    def _invalidate_blit(self) -> None:
        self._blit_invalidated = True
        self._blit_backgrounds.clear()

    def _draw_with_blit(self, updated_axes: set[plt.Axes]) -> None:
        if self._blit_invalidated:
            self.fig.canvas.draw()
            for axis in self.axes.values():
                self._blit_backgrounds[axis] = self.fig.canvas.copy_from_bbox(axis.bbox)
            self._blit_invalidated = False

        for axis in updated_axes:
            background = self._blit_backgrounds.get(axis)
            if background is None:
                self._invalidate_blit()
                self.fig.canvas.draw()
                if self._flush_events:
                    self.fig.canvas.flush_events()
                return
            self.fig.canvas.restore_region(background)
            for artist_name, artist_axis in self.artist_axes.items():
                if artist_axis is axis:
                    axis.draw_artist(self.artists[artist_name])
            self.fig.canvas.blit(axis.bbox)

        if self._flush_events:
            self.fig.canvas.flush_events()

    def _apply_scatter_color(
        self,
        artist: Any,
        color: ColorSpec,
        color_range: ColorRange | None,
    ) -> None:
        if isinstance(color, str):
            artist.set_facecolors([to_rgba(color)])
            return
        if isinstance(color, (tuple, list)) and len(color) in (3, 4):
            artist.set_facecolors([to_rgba(color)])
            return
        color_array = np.asarray(color)
        if color_array.ndim == 1 and color_array.size in (3, 4):
            artist.set_facecolors([to_rgba(color_array)])
            return
        artist.set_array(color_array)
        if color_range is not None:
            vmin, vmax = color_range
            artist.set_clim(vmin=vmin, vmax=vmax)
