"""Runtime of the square-root methods over the dimension n.

For every dimension and method, random SPSD matrices are decomposed for about
``--seconds-per-benchmark`` seconds and the median time is logged.

Run from the repository root, for example::

    python -m paper.experiments.runtime                # summary and figure from the stored log
    python -m paper.experiments.runtime --recompute    # rerun all benchmarks (~30 min)

Writes ``timelogs.npy`` (log, times in s) and ``timelogs.txt`` (summary) to the data directory
and ``runtime.pdf`` to the figures directory.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from compact_matrix_sqrt import compact_matrix_sqrt_D, compact_matrix_sqrt_Q, dct_matrix
from compact_matrix_sqrt.algorithms import (
    compact_matrix_sqrt_D_python,
    compact_matrix_sqrt_Q_python,
    random_SPSD,
)

from .. import plot_style as ps
from .common import dimensions, format_table, load_log, parse_args, update_log, write_text

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    import numpy.typing as npt

NAME = "timelogs"
FIGURE_NAME = "runtime"
COLUMNS = (
    "n",
    "chol",
    "DCT",
    "cel4o-D C++",
    "cel4o-D Python",
    "cel4o-Q C++",
    "cel4o-Q Python",
)


def cholesky (Sigma):
    return np.linalg.cholesky (Sigma)

def dct (Sigma):
    return np.linalg.cholesky(Sigma) @ dct_matrix (Sigma.shape[0], True)

def cms_cel4o_D_cpp (Sigma):
    return compact_matrix_sqrt_D (Sigma)

def cms_cel4o_D_python (Sigma):
    return compact_matrix_sqrt_D_python (Sigma)

def cms_cel4o_Q_cpp (Sigma):
    return compact_matrix_sqrt_Q (Sigma)

def cms_cel4o_Q_python (Sigma):
    return compact_matrix_sqrt_Q_python (Sigma)


def warm_up():
    dct (np.eye(10))
    cms_cel4o_D_python (np.eye(10))
    cms_cel4o_D_cpp (np.eye(10))
    cholesky (np.eye(10))
    cms_cel4o_Q_python (np.eye(10))
    cms_cel4o_Q_cpp (np.eye(10))


def benchmark(f, n, rng, repeatsfors=30, max_repeats=100000):
    times = []
    totalT = 0
    while totalT <= repeatsfors and len(times)<=max_repeats:
        Sigma = random_SPSD (rng, n)
        t0 = time.perf_counter()
        f(Sigma)
        t = time.perf_counter() - t0
        times.append(t)
        totalT += t
    return np.median(times)


def compute_entry(n: int, rng: np.random.Generator, seconds: float) -> list[float]:
    """One row of the log (see ``COLUMNS``)."""
    t_cholesky = benchmark(cholesky, n, rng, seconds)
    t_dct = benchmark(dct, n, rng, seconds)
    t_d_cpp = benchmark(cms_cel4o_D_cpp, n, rng, seconds)
    t_d_python = benchmark(cms_cel4o_D_python, n, rng, seconds)
    t_q_cpp = benchmark(cms_cel4o_Q_cpp, n, rng, seconds)
    t_q_python = benchmark(cms_cel4o_Q_python, n, rng, seconds)
    entry = [n, t_cholesky, t_dct, t_d_cpp, t_d_python, t_q_cpp, t_q_python]
    print(f"n: {n}")
    print(f"cel4o-Q Python: {1000*t_q_python:.5f} ms")
    print(f"cel4o-Q C++:    {1000*t_q_cpp:.5f} ms")
    print(f"cel4o-D Python: {1000*t_d_python:.5f} ms")
    print(f"cel4o-D C++:    {1000*t_d_cpp:.5f} ms")
    print(f"dct: {1000*t_dct:.5f} ms")
    print(f"Cholesky: {1000*t_cholesky:.5f} ms")

    print(f"Relative time to Cholesky: Q {t_q_cpp/t_cholesky:.3f} D {t_d_cpp/t_cholesky:.2f}")
    print(f"Relative time Python to C++: {t_q_python / t_q_cpp:.1f}x {t_d_python / t_d_cpp:.1f}x")
    return entry


def summarize(tlog: npt.NDArray[np.float64]) -> str:
    n = tlog[:, 0]
    t_cholesky = tlog[:, 1]
    t_d_cpp = tlog[:, 3]
    t_d_python = tlog[:, 4]
    t_q_cpp = tlog[:, 5]
    t_q_python = tlog[:, 6]
    factors = np.stack(
        [n, t_d_cpp / t_cholesky, t_q_cpp / t_cholesky, t_d_python / t_d_cpp, t_q_python / t_q_cpp],
        axis=-1,
    )
    times_ms = np.concatenate([n[:, np.newaxis], 1000 * tlog[:, 1:]], axis=-1)
    return "\n".join(
        [
            f"dimensions n: {[int(_) for _ in n]}",
            "",
            "median runtime [ms]",
            format_table(COLUMNS, times_ms, ["{:.0f}"] + ["{:.5f}"] * (len(COLUMNS) - 1)),
            "",
            "relative runtime",
            format_table(
                ("n", "D/chol", "Q/chol", "D Python/C++", "Q Python/C++"),
                factors,
                ["{:.0f}", "{:.2f}", "{:.3f}", "{:.1f}", "{:.1f}"],
            ),
            "",
            f"Mean factor cel4o-D/Cholesky {np.mean(t_d_cpp / t_cholesky):.2f}",
            f"Mean factor cel4o-Q/Cholesky {np.mean(t_q_cpp / t_cholesky):.2f}",
            f"Mean factor cel4o-D python/cpp {np.mean(t_d_python / t_d_cpp):.2f}",
            f"Mean factor cel4o-Q python/cpp {np.mean(t_q_python / t_q_cpp):.2f}",
        ]
    )


def plot(tlog: npt.NDArray[np.float64], path: Path):
    n = tlog[:, 0]
    t_cholesky = tlog[:, 1]
    t_dct = tlog[:, 2]
    t_d_cpp = tlog[:, 3]
    t_q_cpp = tlog[:, 5]

    fig, ax = ps.figure()

    ps.plot(ax, n, 1000 * t_q_cpp, "cel4o_q", label="cel4o-Q C++")
    ps.plot(ax, n, 1000 * t_d_cpp, "cel4o_d", label="cel4o-D C++")
    ps.plot(ax, n, 1000 * t_dct, "dct")
    ps.plot(ax, n, 1000 * t_cholesky, "chol")
    ps.reference(ax, n, 0.00001 * n**3, label="$\\mathcal{O}(n^3)$ reference")

    ps.n_axis(ax, [int(_) for _ in n])

    ax.set_yscale("log")
    ax.set_ylim(0.001, 2.5)
    ax.set_ylabel("runtime [ms]")
    ps.align_ylabel(ax)  # aligned with the L4 plot

    ps.legend(ax, loc="upper left")

    ps.save(fig, path)
    plt.close(fig)
    print(f"Wrote {path}")


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--seconds-per-benchmark",
        type=float,
        default=30,
        help="total measured time per method and dimension (default: 30)",
    )


def main():
    args = parse_args(__doc__, _add_arguments)
    log_path = args.data_dir / f"{NAME}.npy"
    if args.compute or args.recompute:
        warm_up()
        rng = np.random.default_rng(args.seed)
        tlog = update_log(
            log_path,
            dimensions(args.max_n),
            lambda n: compute_entry(n, rng, args.seconds_per_benchmark),
            recompute=args.recompute,
        )
    else:
        tlog = load_log(log_path)
    write_text(args.data_dir / f"{NAME}.txt", summarize(tlog))
    if not args.no_plot:
        ps.apply()
        plot(tlog, args.figures_dir / f"{FIGURE_NAME}.pdf")


if __name__ == "__main__":
    main()
