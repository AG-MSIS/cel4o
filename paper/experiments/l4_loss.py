"""Weighted L4 loss of the square-root methods over the dimension n.

For every dimension, the worst case of DCT, cel4o-D and cel4o-Q is searched with CMA-ES over
lower triangular matrices L, and mean and standard deviation of the randomized methods
(random LDQ^DCT, random LQ) are sampled. The identity (worst case of chol / sym sqrt) and the DCT
itself are evaluated directly.

Run from the repository root, for example::

    python -m paper.experiments.l4_loss              # summary and figure from the stored log
    python -m paper.experiments.l4_loss --compute    # compute missing dimensions first (needs cma)

Writes ``l4logs.npy`` (log) and ``l4logs.txt`` (summary) to the data directory and ``l4_loss.pdf``
to the figures directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from compact_matrix_sqrt import cel4o_D, cel4o_Q, dct_matrix, weighted_l4_loss
from compact_matrix_sqrt.algorithms import adversarial, random_L, random_Q

from .. import plot_style as ps
from .common import dimensions, format_table, load_log, parse_args, update_log, write_text

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

NAME = "l4logs"
FIGURE_NAME = "l4_loss"
COLUMNS = (
    "n",
    "I",
    "DCT",
    "adv DCT",
    "rnd LDQ",
    "rnd LDQ std",
    "cel4o-D",
    "rnd LQ",
    "rnd LQ std",
    "cel4o-Q",
)


def cma_worst_case(algorithm, n, L0=None, iterations=300, sigma0=0.3, rng=None):
    import cma
    if rng is None:
        rng = np.random.default_rng()
    rows, cols = np.tril_indices(n)

    def vector_to_L(x):
        L = np.zeros((n, n))
        L[rows, cols] = x
        return L

    def objective(x):
        L = vector_to_L(x)
        return -weighted_l4_loss(algorithm(L))

    if not isinstance(L0,list): L0 = [L0]
    L_best = None
    L4_best = None
    for L0i in L0:
        if L0i is None: x0 = rng.standard_normal(len(rows))
        else: x0 = L0i[rows, cols]
        l4  = objective(x0)
        if L4_best is None or l4<L4_best:
            L_best = vector_to_L(x0)
            L4_best= l4
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'verbose': -9})
        for _ in range(iterations):
            xs = es.ask()
            es.tell(xs, [objective(x) for x in xs])

        l4 = es.result.fbest
        if L4_best is None or l4<L4_best:
            L_best = vector_to_L(es.result.xbest)
            L4_best= l4
    return L_best, -L4_best


def stat_random_LQ (n, rng, iterations = 5000)->tuple[float, float]:
    samples = [
       weighted_l4_loss (random_L(rng, n)@random_Q (rng, n))
       for _ in range(iterations)
    ]
    return float(np.mean(samples)), float(np.std(samples))


def stat_random_LDQ (n, rng, L, Q, iterations = 5000):
    samples = [
       weighted_l4_loss (L*rng.choice([-1.0, 1.0], size=n)@Q)
       for _ in range(iterations)
    ]
    return float(np.mean(samples)), float(np.std(samples))


def compute_entry(n: int, rng: np.random.Generator) -> list[float]:
    """One row of the log (see ``COLUMNS``)."""
    Q_dct = dct_matrix (n, True)

    def alg_dct (L):
        return L@Q_dct

    def alg_cel4o_d (L):
        return cel4o_D (L, Q_dct)

    def alg_cel4o_q (L):
        return cel4o_Q (L)

    L0 = [None]*20 + [np.eye(n), adversarial(Q_dct)]
    l4_I      = weighted_l4_loss (np.eye(n))
    l4_DCT    = weighted_l4_loss (Q_dct)
    _, l4_AdvDCT = cma_worst_case (alg_dct, n, L0=L0, rng=rng)
    l4_randLDQ, l4s_randLDQ = stat_random_LDQ (n, rng, adversarial(Q_dct), Q_dct)
    _, l4_cel4o_D = cma_worst_case (alg_cel4o_d, n, L0=L0, rng=rng)
    l4_randLQ, l4s_randLQ = stat_random_LQ (n, rng)
    _, l4_cel4o_Q = cma_worst_case (alg_cel4o_q, n, L0=L0, rng=rng)
    entry = [n, l4_I, l4_DCT, l4_AdvDCT, l4_randLDQ, l4s_randLDQ, l4_cel4o_D, l4_randLQ, l4s_randLQ, l4_cel4o_Q]
    print(f"n {n}: I {l4_I:.1f}, DCT {l4_DCT:.1f}, adv DCT {l4_AdvDCT:.1f}, rnd-LDQ {l4_randLDQ:.2f}+/-{l4s_randLDQ:.2f}, cel4o-D {l4_cel4o_D:.2f} rnd-LQ {l4_randLQ:.2f}+/-{l4s_randLQ:.2f}, cel4o-Q {l4_cel4o_Q:.2f}")
    return entry


def summarize(log: npt.NDArray[np.float64]) -> str:
    n = log[:, 0]
    return "\n".join(
        [
            f"dimensions n: {[int(_) for _ in n]}",
            "",
            format_table(COLUMNS, log, ["{:.0f}"] + ["{:.4f}"] * (len(COLUMNS) - 1)),
            "",
            f"l4_I approx {np.mean(log[:, 1] / n):.4}n",
            f"l4_AdvDCT approx {np.mean(log[:, 3] / n):.4}n",
        ]
    )


def plot(log: npt.NDArray[np.float64], path: Path):
    n = log[:, 0]
    l4_I = log[:, 1]
    l4_AdvDCT = log[:, 3]
    l4_randLDQ = log[:, 4]
    l4s_randLDQ = log[:, 5]
    l4_cel4o_D = log[:, 6]
    l4_randLQ = log[:, 7]
    l4s_randLQ = log[:, 8]
    l4_cel4o_Q = log[:, 9]

    fig, ax = ps.figure()

    # Deterministic curves
    ps.plot_shared(ax, n, l4_I, ("chol", "sym_sqrt"), label="chol / sym sqrt")
    ps.plot(ax, n, l4_AdvDCT, "dct")

    # Random LDQ / LQ: mean +/- standard deviation, dashed in the color of the method that
    # derandomizes it
    ps.band(ax, n, l4_randLDQ - l4s_randLDQ, l4_randLDQ + l4s_randLDQ, "cel4o_d")
    ps.plot(ax, n, l4_randLDQ, "cel4o_d", label="random $LDQ^{DCT}$", dashed=True)
    ps.band(ax, n, l4_randLQ - l4s_randLQ, l4_randLQ + l4s_randLQ, "cel4o_q")
    ps.plot(ax, n, l4_randLQ, "cel4o_q", label="random $LQ$", dashed=True)

    ps.plot(ax, n, l4_cel4o_D, "cel4o_d")
    ps.plot(ax, n, l4_cel4o_Q, "cel4o_q")

    ps.n_axis(ax, [int(_) for _ in n])
    ax.set_ylim(1, 4)
    ax.set_ylabel(r"$\mathcal{L}_4$")
    ps.align_ylabel(ax)  # aligned with the runtime plot
    ps.legend(
        ax,
        loc="lower right",
        rows=[
            ["chol / sym sqrt", "DCT"],
            ["random $LQ$", "cel4o-Q"],
            ["random $LDQ^{DCT}$", "cel4o-D"],
        ],
    )

    ps.save(fig, path)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    args = parse_args(__doc__)
    log_path = args.data_dir / f"{NAME}.npy"
    if args.compute or args.recompute:
        rng = np.random.default_rng(args.seed)
        log = update_log(
            log_path,
            dimensions(args.max_n),
            lambda n: compute_entry(n, rng),
            recompute=args.recompute,
        )
    else:
        log = load_log(log_path)
    write_text(args.data_dir / f"{NAME}.txt", summarize(log))
    if not args.no_plot:
        ps.apply()
        plot(log, args.figures_dir / f"{FIGURE_NAME}.pdf")


if __name__ == "__main__":
    main()
