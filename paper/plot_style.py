"""Common look of all paper figures (notebook and experiment reports).

Every method has a fixed identity (label, color, marker) that is the same in every figure; colors
are never assigned by rank or plotting order. Figures are made at their final size in the IEEEtran
paper (panels of one column width) with 8 pt fonts, so LaTeX should not scale them.

Usage::

    from paper import plot_style as ps

    ps.apply()
    fig, ax = ps.figure()
    ps.plot(ax, n, values, "cel4o_d")
    ps.n_axis(ax, n)
    ps.legend(ax)
    ps.save(fig, FIGURES_DIR / "name.pdf")
"""

from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import Bbox, ScaledTranslation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.typing import RcKeyType

COLUMN_WIDTH = 3.5  # [in] IEEEtran \columnwidth
PANEL_HEIGHT = 1.7  # [in]
REFERENCE_COLOR = "#52514e"
GRID_COLOR = "#e6e5e0"
BAND_ALPHA = 0.15
# Padding of the saved bounding box [in]. matplotlib computes text extents from the line metrics,
# but some glyphs (e.g. "f", "[") reach beyond them; in rotated y labels they overshoot to the left.
SAVE_PADDING = 0.01
SAVE_PADDING_LEFT = 0.03
# Distance of aligned y labels from the axes [in] (fits tick labels up to the width of "10^-2").
YLABEL_OFFSET = 0.3
# Fixed margins of single panels (``figure``) [in]. Horizontally, these figures are saved with the
# full column width, so that the axes of figures placed below each other align (LaTeX centers
# figures of different widths differently). Vertically, the saved figure is cropped to its content.
AXES_LEFT = YLABEL_OFFSET + 0.15  # aligned y label (8 pt) + padding
AXES_RIGHT = 0.05  # half of the last x tick label reaching beyond the axes
AXES_BOTTOM = 0.4  # x tick labels and x label (cropped)
AXES_TOP = 0.1  # half of the top y tick label (cropped)


@dataclass(frozen=True)
class Identity:
    label: str
    color: str
    marker: str


# Colors from a palette validated for color vision deficiencies (CVD). In the L4 plot (chol,
# sym sqrt, DCT, cel4o-D, cel4o-Q), all pairs have a normal-vision distance >= 15.6 and a CVD
# distance >= 6.9 (DCT vs. cel4o-Q). In the filter plots (chol, sym sqrt, cel4o-D, cel4o-Q, EKF),
# chol (blue) vs. EKF (violet) is a known weak pair (normal vision 11, CVD 4.5): chol gets the
# well-contrasting blue and DCT/EKF stay distinguishable at least for normal vision, so these two
# rely on their markers (circle vs. down triangle), as do the other CVD pairs; lines always carry
# markers.
IDENTITIES = {
    "chol": Identity("chol", "#2a78d6", "o"),
    "sym_sqrt": Identity("sym sqrt", "#eda100", "s"),
    "cel4o_d": Identity("cel4o-D", "#008300", "D"),
    "cel4o_q": Identity("cel4o-Q", "#e34948", "^"),
    "dct": Identity("DCT", "#1baf7a", "P"),
    "ekf": Identity("EKF", "#6250d6", "v"),
}

# Keys of ``paper.filters.CONSTRUCTORS`` -> identity
FILTER_IDENTITY = {
    "ukf_chol": "chol",
    "ukf_sqrtm": "sym_sqrt",
    "ukf_d": "cel4o_d",
    "ukf_q": "cel4o_q",
    "ekf": "ekf",
}

RC_PARAMS: dict[RcKeyType, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelpad": 1.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.25,
    "legend.borderpad": 0.2,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "pdf.fonttype": 42,  # embedded TrueType fonts (required by IEEE PDF eXpress)
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

# Legend handles of curves drawn by ``plot_shared`` (first line -> tuple of proxy lines).
_SHARED_HANDLES: dict[Artist, tuple[Line2D, ...]] = {}
# Figures from ``figure``, which ``save`` keeps at full width.
_FULL_WIDTH: weakref.WeakSet[Figure] = weakref.WeakSet()


def apply():
    """Sets the matplotlib defaults for all paper figures."""
    mpl.rcParams.update(RC_PARAMS)


def figure(*, height: float = PANEL_HEIGHT) -> tuple[Figure, Axes]:
    """Figure with a single panel of one column width. The axes have the same horizontal position
    and size in all such figures (and the same height for the same ``height``), and the y label
    is aligned (``align_ylabel``), so that figures placed below each other align."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, height))
    fig.subplots_adjust(
        left=AXES_LEFT / COLUMN_WIDTH,
        right=1 - AXES_RIGHT / COLUMN_WIDTH,
        bottom=AXES_BOTTOM / height,
        top=1 - AXES_TOP / height,
    )
    align_ylabel(ax)
    _FULL_WIDTH.add(fig)
    return fig, ax


def panels(num_of_panels: int, *, height: float = PANEL_HEIGHT) -> tuple[Figure, list[Axes]]:
    """Figure with ``num_of_panels`` side-by-side panels of one column width each."""
    fig, axes = plt.subplots(
        1,
        num_of_panels,
        figsize=(num_of_panels * COLUMN_WIDTH, height),
        squeeze=False,
        layout="constrained",
    )
    return fig, list(axes[0])


def plot(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    identity: str,
    *,
    label: str | None = None,
    dashed: bool = False,
    marker: bool = True,
) -> Line2D:
    """Curve of a method in its identity's color and marker. Dashed curves are used for
    randomized variants of a method (e.g. random LDQ for cel4o-D)."""
    style = IDENTITIES[identity]
    return ax.plot(
        x,
        y,
        color=style.color,
        marker=style.marker if marker else None,
        markerfacecolor="white" if dashed else style.color,
        linestyle="--" if dashed else "-",
        label=style.label if label is None else label,
    )[0]


def band(ax: Axes, x: npt.ArrayLike, low: npt.ArrayLike, high: npt.ArrayLike, identity: str):
    """Uncertainty band (confidence interval or +/- standard deviation) of a method."""
    ax.fill_between(x, low, high, color=IDENTITIES[identity].color, alpha=BAND_ALPHA, linewidth=0)


def plot_shared(
    ax: Axes, x: npt.ArrayLike, y: npt.ArrayLike, identities: Sequence[str], *, label: str
) -> Line2D:
    """A curve that holds for several methods, drawn with alternating dashes (and alternating
    markers) in their colors. ``legend`` shows it with the same alternating dashes."""
    dash = 3.0  # [pt]
    lines = []
    proxies = []
    for index, identity in enumerate(identities):
        style = IDENTITIES[identity]
        dashes = (index * dash, (dash, (len(identities) - 1) * dash))
        lines.append(
            ax.plot(
                x,
                y,
                color=style.color,
                linestyle=dashes,
                marker=style.marker,
                markevery=slice(index, None, len(identities)),
                label=label if index == 0 else "_nolegend_",
            )[0]
        )
        proxies.append(Line2D([], [], color=style.color, linestyle=dashes))
    _SHARED_HANDLES[lines[0]] = tuple(proxies)
    return lines[0]


def reference(ax: Axes, x: npt.ArrayLike, y: npt.ArrayLike, *, label: str) -> Line2D:
    """Reference curve that is not a method (e.g. an asymptotic order)."""
    return ax.plot(x, y, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, label=label)[0]


def reference_hline(ax: Axes, y: float, *, label: str) -> Line2D:
    """Horizontal reference line (e.g. a nominal value)."""
    return ax.axhline(y, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, label=label, zorder=1)


def n_axis(ax: Axes, ticks: Sequence[float], *, label: str = "$n$"):
    """Logarithmic dimension axis with ticks exactly at ``ticks`` (plain numbers)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(ticks) / 1.07, max(ticks) * 1.07)
    ax.set_xlabel(label)


def align_ylabel(ax: Axes, offset: float = YLABEL_OFFSET):
    """Puts the y label at a fixed distance [in] from the axes instead of next to the tick
    labels, so that the axes of figures with differently wide tick labels align when the figures
    are placed below each other (the offset must exceed the widest tick label)."""
    transform = ax.transAxes + ScaledTranslation(-offset, 0, ax.figure.dpi_scale_trans)
    ax.yaxis.set_label_coords(0, 0.5, transform=transform)


def _handles(ax: Axes) -> tuple[list, list[str]]:
    handles, labels = ax.get_legend_handles_labels()
    return [_SHARED_HANDLES.get(handle, handle) for handle in handles], labels


def legend(ax: Axes, *, rows: Sequence[Sequence[str]] | None = None, **kwargs):
    """Legend of an axes (supports curves from ``plot_shared``).

    :param rows: Optional arrangement of the entries, given by their labels row by row (all rows
        of equal length). Entries that are not listed are omitted.
    """
    handles, labels = _handles(ax)
    if rows is not None:
        num_of_columns = len(rows[0])
        assert all(len(row) == num_of_columns for row in rows)
        handle_by_label = dict(zip(labels, handles, strict=True))
        # matplotlib fills legends column by column.
        labels = [row[column] for column in range(num_of_columns) for row in rows]
        handles = [handle_by_label[label] for label in labels]
        kwargs["ncol"] = num_of_columns
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, **kwargs)


def figure_legend(fig: Figure, ax: Axes, **kwargs):
    """One legend for a whole multi-panel figure, built from the entries of ``ax``."""
    handles, labels = _handles(ax)
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, **kwargs)


def save(fig: Figure, path: Path):
    """Saves the figure cropped to its content (tight bounding box with minimal padding).
    Figures from ``figure`` keep their full width (only cropped vertically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()  # finalizes the (constrained) layout
    tight = fig.get_tightbbox()
    x0, x1 = tight.x0 - SAVE_PADDING_LEFT, tight.x1 + SAVE_PADDING
    if fig in _FULL_WIDTH:
        width = fig.get_figwidth()
        if x0 < 0 or x1 > width:
            warnings.warn(
                f"{path.name}: content reaches beyond the fixed margins "
                f"({x0:.3f} in left, {x1 - width:.3f} in right)",
                stacklevel=2,
            )
        x0, x1 = 0.0, width
    bbox = Bbox.from_extents(x0, tight.y0 - SAVE_PADDING, x1, tight.y1 + SAVE_PADDING)
    fig.savefig(path, bbox_inches=bbox)
