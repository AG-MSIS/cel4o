"""Helpers shared by the experiments over the dimension n (L4 loss, runtime).

Every experiment keeps a log ``<name>.npy`` in the data directory whose rows are
``[n, value_1, value_2, ...]``. Entries are computed per dimension and the log is saved after each
one, so an interrupted computation resumes where it stopped. (The results are then not identical
to an uninterrupted run, since the random generator starts again from its seed.)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..paths import DATA_DIR, FIGURES_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def dimensions(max_n: int) -> list[int]:
    """Dimensions 3, 5, 8, 12, ... (growing by a factor of 1.5, rounded up) up to ``max_n``."""
    ns = []
    n = 3
    while n <= max_n:
        ns.append(n)
        n = math.ceil(1.5 * n)
    return ns


def parse_args(
    description: str | None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Command line of an experiment. Without ``--compute``/``--recompute``, only the summary and
    the figure are written from the stored log."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compute", action="store_true", help="compute the dimensions missing in the log"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="discard the log and compute all dimensions"
    )
    parser.add_argument("--no-plot", action="store_true", help="do not write the figure")
    parser.add_argument("--max-n", type=int, default=100, help="largest dimension (default: 100)")
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    if add_arguments is not None:
        add_arguments(parser)
    return parser.parse_args()


def load_log(path: Path) -> npt.NDArray[np.float64]:
    """Loads a log, sorted by dimension."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run with --compute first")
    log = np.load(path)
    return log[np.argsort(log[:, 0])]


def update_log(
    path: Path,
    ns: Sequence[int],
    compute_entry: Callable[[int], Sequence[float]],
    *,
    recompute: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes ``compute_entry(n)`` (a row starting with ``n``) for every dimension that is
    missing in the log at ``path`` (or for all with ``recompute``), saving after each one.

    :return: The log, sorted by dimension.
    """
    log = None if recompute or not path.exists() else np.load(path)
    for n in ns:
        if log is not None and n in log[:, 0]:
            continue
        entry = np.array([compute_entry(n)], dtype=np.float64)
        assert entry[0, 0] == n
        log = entry if log is None else np.concatenate([log, entry], axis=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, log)
    if log is None:
        raise ValueError("No dimensions to compute")
    return log[np.argsort(log[:, 0])]


def format_table(
    header: Sequence[str], rows: npt.ArrayLike, formats: Sequence[str]
) -> str:
    """Fixed-width text table with right-aligned columns."""
    cells = [list(header)] + [
        [format_.format(value) for format_, value in zip(formats, row, strict=True)]
        for row in np.asarray(rows)
    ]
    widths = [max(len(row[column]) for row in cells) for column in range(len(header))]
    lines = ["  ".join(cell.rjust(width) for cell, width in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def write_text(path: Path, text: str):
    """Prints ``text`` and writes it to ``path``."""
    print(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(f"\nWrote {path}")
