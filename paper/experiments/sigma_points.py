"""2D marginal of the sigma points of N(0, I_n): standard (chol = sym sqrt) for n = 2 and n = 25
versus cel4o-D and cel4o-Q for n = 25.

Run from the repository root::

    python -m paper.experiments.sigma_points

Writes ``sigma_points_2d_marginal.pdf`` to the figures directory.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle

from compact_matrix_sqrt import compact_matrix_sqrt_Q, compact_matrix_sqrt_D

from .. import plot_style as ps
from ..paths import FIGURES_DIR


def standard_sigma_points(n: int, kappa: float = 1.0) -> np.ndarray:
    """
    Standard unscented-transform sigma points for N(0, I_n).

    Returns
    -------
    points : ndarray, shape (2*n+1, n)
    """
    r = np.sqrt(n + kappa)

    points = np.zeros((2 * n + 1, n))
    points[1:n + 1] = r * np.eye(n)
    points[n + 1:] = -r * np.eye(n)

    return points


def compact_sigma_points(n: int, kappa: float = 1.0, matrix_sqrt:str='Q') -> np.ndarray:
    """
    Sigma points using compact_matrix_sqrt_Q.
    """
    # Covariance square root for N(0, I_n).
    if matrix_sqrt=='Q':  S = compact_matrix_sqrt_Q(np.eye(n))
    elif matrix_sqrt=='D': S = compact_matrix_sqrt_D(np.eye(n))
    else: raise ValueError ("matrix_sqrt must be 'D' or 'Q'")
    r = np.sqrt(n + kappa)

    points = np.zeros((2 * n + 1, n))
    points[1:n + 1] = r * S.T
    points[n + 1:] = -r * S.T

    return points


# The paper includes this figure at 0.7\linewidth, so it is drawn at that final size (fonts stay at
# 7/8 pt); marker sizes and line widths are given for the full column width and scaled by SCALE.
SCALE = 0.7
WIDTH = SCALE * ps.COLUMN_WIDTH  # [in]

# The standard sigma points of N(0, I) are the same for chol and sym sqrt. This figure comes before
# the paper distinguishes them, so both are "standard": n = 2 in chol's color and marker, n = 25 in sym
# sqrt's.
STANDARD_2_MARKER: dict[str, Any] = {
    "marker": ps.IDENTITIES["chol"].marker,
    "markersize": SCALE * 7,
    "color": ps.IDENTITIES["chol"].color,
    "markeredgecolor": "white",
    "markeredgewidth": SCALE * 0.5,
}
STANDARD_25_MARKER: dict[str, Any] = {
    "marker": ps.IDENTITIES["sym_sqrt"].marker,
    "markersize": SCALE * 7,
    "color": ps.IDENTITIES["sym_sqrt"].color,
    "markeredgecolor": "white",
    "markeredgewidth": SCALE * 0.5,
}


def plot_sigma_point_comparison():
    sigma_2 = standard_sigma_points(n=2, kappa=1)
    sigma_compact_q = compact_sigma_points(n=25, kappa=1, matrix_sqrt='Q')
    sigma_compact_d = compact_sigma_points(n=25, kappa=1, matrix_sqrt='D')

    # 2D marginals
    p2 = sigma_2[:, :2]
    marginals = {"cel4o_q": sigma_compact_q[:, 7:9], "cel4o_d": sigma_compact_d[:, 7:9]}

    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH), layout="constrained")

    # Gaussian density (peak at 70 % gray) and 1, 2, 3 sigma circles
    lim = 5.6  # outermost points at +-sqrt(26) = 5.1
    x = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(x, x)
    ax.imshow(np.exp(-0.5 * (X**2 + Y**2)), extent=(-lim, lim, -lim, lim), origin="lower",
              cmap="Greys", vmin=0, vmax=1 / 0.7, interpolation="bilinear", zorder=0)
    for radius in (1, 2, 3):
        ax.add_patch(Circle((0, 0), radius, fill=False, color=ps.REFERENCE_COLOR,
                                linestyle=":", linewidth=SCALE * 0.6, zorder=1))
    ax.grid(False)

    # Standard sigma points, n = 2 (low-dimensional reference)
    ax.plot(p2[:, 0], p2[:, 1], linestyle="none", **STANDARD_2_MARKER,
            label="standard, $n=2$", zorder=3)
    # At the origin, the central points of all sets coincide: the n = 2 one is drawn smaller on top
    # of the n = 25 square (whose corners stay visible) and below the cel4o markers.
    center_marker: dict[str, Any] = STANDARD_2_MARKER | {
        "markersize": SCALE * 6,
        "markeredgewidth": 0,
    }
    ax.plot([0], [0], linestyle="none", **center_marker, zorder=4.5)

    # Standard sigma points, n = 25: only the unique visible marginal locations. 47 sigma points
    # coincide at the origin: 1 central point + 2*(25-2) points from dimensions 3,...,25.
    r25 = np.sqrt(25 + 1)
    p25_visible = np.array([[r25, 0], [-r25, 0], [0, r25], [0, -r25], [0, 0]])
    ax.plot(p25_visible[:, 0], p25_visible[:, 1], linestyle="none", **STANDARD_25_MARKER,
            label="standard, $n=25$", zorder=4)
    marker = DrawingArea(SCALE * 10, SCALE * 10, 0, 0)
    marker.add_artist(Line2D([SCALE * 5], [SCALE * 5], **STANDARD_25_MARKER))
    label = HPacker(children=[TextArea("47 ×", textprops={"fontsize": 7}), marker],
                    pad=0, sep=1.5, align="center")
    ax.add_artist(AnnotationBbox(
        label, (0.05, -0.05), xybox=(2.9, -3.4), frameon=False, box_alignment=(0.5, 1.0),
        arrowprops={"arrowstyle": "-", "color": ps.REFERENCE_COLOR, "linewidth": SCALE * 0.6,
                    "shrinkA": 1, "shrinkB": SCALE * 3},
        zorder=10,
    ))

    # cel4o sigma points, n = 25
    for identity, points in marginals.items():
        style = ps.IDENTITIES[identity]
        ax.plot(points[:, 0], points[:, 1], linestyle="none", marker=style.marker,
                markersize=SCALE * 3.5, color=style.color, markeredgecolor="white",
                markeredgewidth=SCALE * 0.4, label=f"{style.label}, $n=25$", zorder=5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ps.legend(ax, loc="upper right")
    path = FIGURES_DIR / "sigma_points_2d_marginal.pdf"
    ps.save(fig, path)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    ps.apply()
    plot_sigma_point_comparison()
