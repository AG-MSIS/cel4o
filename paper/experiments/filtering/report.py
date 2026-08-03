"""Evaluates a run of ``paper.experiments.filtering.run``: writes a text summary (per state
dimension and aggregated over all of them, see ``aggregate``), a LaTeX table and ``summary.json``
to the run directory, and one single-column figure per metric (coverage, RMSE,
sigma-point flip rate over the state dimension, position RMSE relative to chol) plus an overview
with the absolute metrics to the figures directory (``filtering_<metric>.pdf``,
``filtering_relative_position_rmse.pdf``, ``filtering_summary.png/.pdf``).

Run from the repository root, for example::

    python -m paper.experiments.filtering.report paper/data/filter-<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ... import plot_style as ps
from ...filters import CONSTRUCTORS
from ...paths import FIGURES_DIR
from ..common import write_text
from .aggregate import aggregate, format_aggregates
from .metrics import (
    COVERAGE_PROBABILITY,
    paired_summary,
    relative_summary,
    rho_quantiles,
    rho_regime_fractions,
    summarize,
)
from .run import load_run
from .scenario import ScenarioConfig

SHORT_NAMES = {key: ps.IDENTITIES[identity].label for key, identity in ps.FILTER_IDENTITY.items()}
# metric, y label, factor (fractions are shown in percent), legend location (empty corner)
PANELS = [
    ("position_coverage", "position coverage [%]", 100, "lower left"),
    ("joint_coverage", "95% NEES coverage [%]", 100, "lower left"),
    ("position_rmse", "position RMSE per axis [m]", 1, "upper right"),
    ("flip_rate", "flipped updates [%]", 100, "upper left"),
]
RHO_EDGES = (0.1, 0.15, 0.3, 0.5)
# Baselines against which the paired differences are tested.
REFERENCES = ["ukf_chol", "ukf_sqrtm"]
# Filter whose position RMSE the others are shown relative to.
RELATIVE_REFERENCE = "ukf_chol"
# Order of the filters in the figures (legends and drawing): cel4o-Q before cel4o-D, as in all
# paper figures. Filters that are not listed follow in run order.
FIGURE_ORDER = ["ukf_chol", "ukf_sqrtm", "ukf_q", "ukf_d", "ekf"]


def figure_order(filter_keys: list[str]) -> list[str]:
    def rank(key: str) -> int:
        return FIGURE_ORDER.index(key) if key in FIGURE_ORDER else len(FIGURE_ORDER)

    return sorted(filter_keys, key=rank)


def evaluate(directory: Path) -> tuple[dict, dict]:
    meta, results = load_run(directory)
    summaries = {n: summarize(data, meta["filters"]) for n, data in results.items()}
    rho = {
        n: {
            "quantiles": rho_quantiles(data).tolist(),
            "fractions": rho_regime_fractions(data, RHO_EDGES).tolist(),
        }
        for n, data in results.items()
    }
    references = [_ for _ in REFERENCES if _ in meta["filters"]]
    paired = {n: paired_summary(data, meta["filters"], references) for n, data in results.items()}
    relative_position_rmse = {
        n: relative_summary(data, meta["filters"], RELATIVE_REFERENCE, "position_rmse")
        for n, data in results.items()
    }
    num_of_episodes = min(data["true"].shape[0] for data in results.values())
    return meta, {
        "summaries": summaries,
        "paired": paired,
        "relative_position_rmse": relative_position_rmse,
        "rho": rho,
        "num_of_episodes": num_of_episodes,
    }


def format_summary(meta: dict, evaluation: dict) -> str:
    filters = meta["filters"]
    lines = []
    for num_of_vehicles, summary in sorted(evaluation["summaries"].items()):
        q = evaluation["rho"][num_of_vehicles]["quantiles"]
        f = evaluation["rho"][num_of_vehicles]["fractions"]
        lines.append(
            f"\nN={num_of_vehicles} (n={4 * num_of_vehicles}) "
            f"rho q50/q90/q99={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} "
            f"rho% <.1|.1-.15|.15-.3|.3-.5|>.5: " + " ".join(f"{100 * _:4.1f}" for _ in f)
        )
        lines.append(
            f"  {'filter':<16} {'pos cov [95% CI]':>22} {'joint cov':>8} {'pos RMSE':>8} "
            f"{'div':>6} {'flip':>6}"
        )
        for key in filters:
            s = summary[key]
            pc = s["position_coverage"]
            lines.append(
                f"  {CONSTRUCTORS[key][1]:<16} {pc[0]:6.3f} [{pc[1]:.3f},{pc[2]:.3f}] "
                f"{s['joint_coverage'][0]:8.3f} {s['position_rmse'][0]:8.3f} "
                f"{s['diverged'][0]:6.3f} {s['flip_rate'][0]:6.3f}"
            )
        lines.append("  paired differences (mean [95% CI]) of joint coverage / position RMSE:")
        for key, comparisons in evaluation["paired"][num_of_vehicles].items():
            for reference, d in comparisons.items():
                if key in REFERENCES:
                    continue
                jc, rmse = d["joint_coverage"], d["position_rmse"]
                lines.append(
                    f"    {SHORT_NAMES[key]:>8} - {SHORT_NAMES[reference]:<8} "
                    f"{jc[0]:+.3f} [{jc[1]:+.3f},{jc[2]:+.3f}]  "
                    f"{rmse[0]:+.3f} [{rmse[1]:+.3f},{rmse[2]:+.3f}]"
                )
    return "\n".join(lines)


def baselines(meta: dict, results: dict[int, dict[str, np.ndarray]]) -> dict[str, float]:
    """Per-axis RMS values that put the filter RMSEs into perspective.

    ``spread_*``: RMS of the true positions / velocities, i.e. the RMSE of always estimating 0.
    ``open_loop_*``: RMSE (averaged over the episode) of extrapolating the initial mean without any
    measurement, computed from the exact open-loop covariance (identical for all vehicles).
    """
    config = ScenarioConfig.from_dict(meta["scenario"])
    true = np.concatenate([np.reshape(data["true"], (-1, 4)) for data in results.values()])
    system = config.system(1)
    covariance = config.initial_covariance(1)
    jacobian = system.dynamics_jacobian(None, config.dt)
    variances = []
    for _ in range(config.num_of_steps):
        covariance = jacobian @ covariance @ jacobian.T + system.dynamics_covariance(config.dt)
        variances.append(np.diag(covariance))
    variances = np.array(variances)
    return {
        "spread_position": float(np.sqrt(np.mean(np.square(true[:, :2])))),
        "spread_velocity": float(np.sqrt(np.mean(np.square(true[:, 2:])))),
        "open_loop_position": float(np.sqrt(np.mean(variances[:, :2]))),
        "open_loop_position_final": float(np.sqrt(np.mean(variances[-1, :2]))),
        "open_loop_velocity": float(np.sqrt(np.mean(variances[:, 2:]))),
    }


def latex_table(meta: dict, evaluation: dict, filter_keys: list[str]) -> str:
    """Compact table for the paper: rows = filters, columns = state dimension, one block with the
    joint coverage (value closest to the nominal coverage in bold) and one with the position RMSE
    per axis (smallest in bold). A dagger marks values that are significantly better than all
    reference filters in the table (paired 95% CI of the difference excludes 0). The leading
    comment states the largest marginal CI half-widths for the caption."""
    summaries = evaluation["summaries"]
    paired = evaluation["paired"]
    nums_of_vehicles = sorted(summaries)
    columns = len(nums_of_vehicles) + 1
    references = [_ for _ in REFERENCES if _ in filter_keys]
    lines = [
        f"% {evaluation['num_of_episodes']} episodes per n. Largest half-width of the marginal 95% "
        + "bootstrap CIs: joint coverage {:.1f} pp, position RMSE {:.2f} m.".format(
            *(
                max(
                    factor * (summaries[n][key][metric][2] - summaries[n][key][metric][1]) / 2
                    for n in nums_of_vehicles
                    for key in filter_keys
                )
                for metric, factor in (("joint_coverage", 100), ("position_rmse", 1))
            )
        ),
        "% \\dagger: significantly better than "
        + " and ".join(SHORT_NAMES[_] for _ in references)
        + " (95% paired bootstrap CI of the per-episode difference excludes 0).",
        r"{\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabular}{@{}l" + "r" * len(nums_of_vehicles) + "@{}}",
        r"\hline",
        r"\(n\) & " + " & ".join(rf"\({4 * _}\)" for _ in nums_of_vehicles) + r" \\",
    ]
    blocks = [
        ("joint_coverage", "joint 95\\,\\% coverage [\\%]", 100, 1, True),
        ("position_rmse", "position RMSE per axis [m]", 1, 2, False),
    ]
    for metric, title, factor, decimals, is_coverage in blocks:
        lines += [r"\hline", rf"\multicolumn{{{columns}}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]

        def score(value: float, is_coverage=is_coverage, factor=factor, decimals=decimals):
            target = COVERAGE_PROBABILITY if is_coverage else 0.0
            return round(abs(factor * value - factor * target), decimals)

        for key in filter_keys:
            cells = [SHORT_NAMES[key]]
            for n in nums_of_vehicles:
                value = summaries[n][key][metric][0]
                best = min(score(summaries[n][_][metric][0]) for _ in filter_keys)
                text = f"{factor * value:.{decimals}f}"
                if score(value) == best:
                    text = rf"\textbf{{{text}}}"
                if (
                    references
                    and key not in references
                    and all(
                        score(value) < score(summaries[n][reference][metric][0])
                        and (
                            paired[n][key][reference][metric][1] > 0
                            or paired[n][key][reference][metric][2] < 0
                        )
                        for reference in references
                    )
                ):
                    text += r"\rlap{\(^\dagger\)}"
                cells.append(text)
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", ""]
    return "\n".join(lines)


def plot_metric(ax, meta: dict, evaluation: dict, metric: str, factor: float):
    """Value and 95% CI band of a metric over the state dimension for every filter."""
    summaries = evaluation["summaries"]
    nums_of_vehicles = sorted(summaries)
    dims = [4 * _ for _ in nums_of_vehicles]
    for key in figure_order(meta["filters"]):
        values = factor * np.array([summaries[n][key][metric] for n in nums_of_vehicles])
        if np.all(np.isnan(values[:, 0])):
            continue  # e.g. flip rate of the EKF
        identity = ps.FILTER_IDENTITY[key]
        ps.band(ax, dims, values[:, 1], values[:, 2], identity)
        ps.plot(ax, dims, values[:, 0], identity)
    # References are drawn last so that they are the last legend entry (they stay behind the
    # curves via zorder).
    if "coverage" in metric:
        ps.reference_hline(ax, factor * COVERAGE_PROBABILITY, label="nominal")
    ps.n_axis(ax, dims)


def plot_relative_rmse(ax, meta: dict, evaluation: dict):
    """Position RMSE relative to ``RELATIVE_REFERENCE`` (in percent) with paired 95% CI bands."""
    relative = evaluation["relative_position_rmse"]
    nums_of_vehicles = sorted(relative)
    dims = [4 * _ for _ in nums_of_vehicles]
    for key in figure_order(meta["filters"]):
        values = 100 * np.array([relative[n][key] for n in nums_of_vehicles])
        identity = ps.FILTER_IDENTITY[key]
        if key != RELATIVE_REFERENCE:
            ps.band(ax, dims, values[:, 1], values[:, 2], identity)
        ps.plot(ax, dims, values[:, 0], identity)
    ps.n_axis(ax, dims)
    ax.set_ylabel(f"position RMSE rel. to {SHORT_NAMES[RELATIVE_REFERENCE]} [%]")


def plot(meta: dict, evaluation: dict, directory: Path) -> list[Path]:
    """Writes ``filtering_<metric>.pdf`` per metric (one column wide) and
    ``filtering_summary.png/.pdf`` with all of them.

    :return: The written files.
    """
    paths = []
    for metric, ylabel, factor, legend_location in PANELS:
        fig, ax = ps.figure()
        plot_metric(ax, meta, evaluation, metric, factor)
        ax.set_ylabel(ylabel)
        ps.legend(ax, loc=legend_location, ncol=3)
        paths.append(directory / f"filtering_{metric}.pdf")
        ps.save(fig, paths[-1])
        plt.close(fig)

    fig, ax = ps.figure()
    plot_relative_rmse(ax, meta, evaluation)
    ps.legend(ax, loc="lower left", ncol=3)
    paths.append(directory / "filtering_relative_position_rmse.pdf")
    ps.save(fig, paths[-1])
    plt.close(fig)

    fig, axes = ps.panels(len(PANELS))
    for ax, (metric, ylabel, factor, _) in zip(axes, PANELS, strict=True):
        plot_metric(ax, meta, evaluation, metric, factor)
        ax.set_ylabel(ylabel)
    ps.figure_legend(fig, axes[0], loc="outside upper center", ncol=len(meta["filters"]) + 1)
    paths += [directory / "filtering_summary.png", directory / "filtering_summary.pdf"]
    for path in paths[-2:]:
        ps.save(fig, path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--table-filters",
        nargs="+",
        default=["ukf_chol", "ukf_sqrtm", "ukf_d", "ukf_q"],
        choices=list(CONSTRUCTORS),
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    ps.apply()
    meta, evaluation = evaluate(args.directory)
    _, results = load_run(args.directory)
    evaluation["baselines"] = baselines(meta, results)
    evaluation["aggregates"] = aggregate(
        results, meta["filters"], [_ for _ in REFERENCES if _ in meta["filters"]]
    )
    b = evaluation["baselines"]
    write_text(
        args.directory / "summary.txt",
        "\n".join(
            [
                f"scenario: {meta['scenario']}",
                format_summary(meta, evaluation),
                "",
                (
                    f"baselines (per axis): true positions RMS {b['spread_position']:.2f} m, "
                    f"velocities RMS {b['spread_velocity']:.2f} m/s; open loop (no measurements) "
                    f"position RMSE {b['open_loop_position']:.2f} m (final "
                    f"{b['open_loop_position_final']:.2f} m), velocity RMSE "
                    f"{b['open_loop_velocity']:.2f} m/s"
                ),
                "",
                format_aggregates(evaluation["aggregates"], meta["filters"]),
            ]
        ),
    )
    (args.directory / "table.tex").write_text(latex_table(meta, evaluation, args.table_filters))
    figure_paths = plot(meta, evaluation, args.figures_dir)
    with open(args.directory / "summary.json", "w") as f:
        json.dump(
            {str(n): s for n, s in evaluation["summaries"].items()}
            | {"rho": {str(n): r for n, r in evaluation["rho"].items()}}
            | {"baselines": evaluation["baselines"]}
            | {"aggregates": evaluation["aggregates"]}
            | {"paired": {str(n): p for n, p in evaluation["paired"].items()}}
            | {
                "relative_position_rmse": {
                    str(n): r for n, r in evaluation["relative_position_rmse"].items()
                }
            },
            f,
            indent=2,
        )
    print(f"Wrote table.tex and summary.json to {args.directory}")
    for path in figure_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
