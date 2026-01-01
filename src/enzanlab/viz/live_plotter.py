
"""Lightweight live debugger for Matplotlib visualizations."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

ColorRange = tuple[float, float]
ColorSpec = str | Sequence[float] | np.ndarray
AxisKey = str


class LivePlotter:
    """Lightweight wrapper for interactive Matplotlib updates."""

    def __init__(
        self,
        layout: tuple[int, int] = (1, 1),
        figsize: tuple[float, float] = (8, 6),
        *,
        show: bool = True,
        autoscale: bool = False,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
        tight_layout: bool = False,
    ) -> None:
        """Initialize a figure and axes layout.

        Args:
            layout: Subplot grid size as (rows, cols).
            figsize: Figure size in inches.
            show: If True, show the figure immediately (non-blocking).
            autoscale: If True, autoscale axes after each update.
            width_ratios: Relative column widths for the grid.
            height_ratios: Relative row heights for the grid.
            tight_layout: If True, call fig.tight_layout() after setup.
        """
        rows, cols = layout
        self.fig = plt.figure(figsize=figsize)
        grid = self.fig.add_gridspec(
            rows,
            cols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        self._grid = grid

        self.axes: dict[str, plt.Axes] = {}
        self.artists: dict[str, Any] = {}
        self.artist_axes: dict[str, plt.Axes] = {}
        self.autoscale = autoscale
        self._tight_layout = tight_layout

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
        ax = self.fig.add_subplot(self._grid[row, col])
        self.axes[name] = ax
        if self._tight_layout:
            self.fig.tight_layout()
        return ax

    def add_line(self, ax: AxisKey, name: str, **kwargs: Any) -> Any:
        """Register a line artist on the specified axis.

        Args:
            ax: Axis key as "ax0", flat index, or (row, col).
            name: Artist name for updates.
            **kwargs: Passed through to Matplotlib's plot.

        Returns:
            The created line artist.
        """
        target_ax = self._resolve_ax(ax)
        line, = target_ax.plot([], [], **kwargs)
        self.artists[name] = line
        self.artist_axes[name] = target_ax
        return line

    def add_scatter(self, ax: AxisKey, name: str, **kwargs: Any) -> Any:
        """Register a scatter artist on the specified axis.

        Args:
            ax: Axis key as "ax0", flat index, or (row, col).
            name: Artist name for updates.
            **kwargs: Passed through to Matplotlib's scatter.

        Returns:
            The created scatter artist.
        """
        target_ax = self._resolve_ax(ax)
        sc = target_ax.scatter([], [], **kwargs)
        self.artists[name] = sc
        self.artist_axes[name] = target_ax
        return sc

    def set_limits(
        self,
        ax: AxisKey,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Set axis limits explicitly.

        Args:
            ax: Axis key (e.g., "ax0").
            xlim: X-axis limits as (min, max).
            ylim: Y-axis limits as (min, max).
        """
        target_ax = self._resolve_ax(ax)
        if xlim is not None:
            target_ax.set_xlim(*xlim)
        if ylim is not None:
            target_ax.set_ylim(*ylim)

    def set_title(self, ax: AxisKey, title: str) -> None:
        """Set axis title explicitly.

        Args:
            ax: Axis key as "ax0", flat index, or (row, col).
            title: Title text to set.
        """
        target_ax = self._resolve_ax(ax)
        target_ax.set_title(title)

    def is_open(self) -> bool:
        """Check whether the figure window is still open."""
        return plt.fignum_exists(self.fig.number)

    def _resolve_ax(self, ax: AxisKey) -> plt.Axes:
        if isinstance(ax, str):
            return self.axes[ax]
        raise TypeError(f"Unsupported axis key: {ax!r}")

    def update(
        self,
        data: dict[
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
        for name, payload in data.items():
            if name not in self.artists:
                continue
            # Safely index payload rather than relying on tuple unpacking,
            # which can fail for sequences like numpy arrays.
            try:
                x = payload[0]
                y = payload[1]
            except Exception:
                continue
            color = payload[2] if len(payload) >= 3 else None
            color_range = payload[3] if len(payload) >= 4 else None
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

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _apply_scatter_color(
        self,
        artist: Any,
        color: ColorSpec,
        color_range: ColorRange|None,
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
