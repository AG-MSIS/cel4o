"""Statistics aggregated over the state dimension n, for statements such as "the overconfidence of
cel4o-D is x% of that of chol" or "cel4o-D reduces the position RMSE by y% compared to chol".

- Overconfidence: the coverage deficit ``95% - coverage`` (the coverage counterpart of NEES / dof
  > 1; it is not robust against outliers like the NEES). Relative to a reference filter, it is
  the ratio of the deficits averaged over n (a ratio of means, since the deficits at small n are
  small and their ratios noisy).
- Position RMSE: the mean over n of ``RMSE / RMSE_reference - 1``.

Every n is weighted equally, so the "over all n" values depend on the chosen dimensions. The
values at the largest n are given as well. CIs are bootstrapped with
``metrics.bootstrap_over_dimensions``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ... import plot_style as ps
from ..common import format_table
from .metrics import COVERAGE_PROBABILITY, METRICS, bootstrap_over_dimensions

SHORT_NAMES = {key: ps.IDENTITIES[identity].label for key, identity in ps.FILTER_IDENTITY.items()}
Statistic = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def mean_deficit(means: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Coverage deficit averaged over n; ``means``: coverage (..., dimensions, 1)."""
    return np.mean(COVERAGE_PROBABILITY - means[..., 0], axis=-1)


def deficit_ratio(means: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ratio of the coverage deficits averaged over n; ``means``: coverage of the filter and the
    reference (..., dimensions, 2)."""
    deficits = np.mean(COVERAGE_PROBABILITY - means, axis=-2)
    return deficits[..., 0] / deficits[..., 1]


def relative_rmse(means: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Relative RMSE change averaged over n; ``means``: MSE of the filter and the reference
    (..., dimensions, 2)."""
    return np.mean(np.sqrt(means[..., 0] / means[..., 1]) - 1, axis=-1)


def _aggregate(
    results: dict[int, dict],
    nums_of_vehicles: list[int],
    name: str,
    keys: list[str],
    statistic: Statistic,
) -> tuple[float, float, float]:
    per_episode = METRICS[name][0]
    values = [np.stack([per_episode(results[n], key) for key in keys]) for n in nums_of_vehicles]
    return bootstrap_over_dimensions(values, statistic)


def aggregate(
    results: dict[int, dict], filter_keys: list[str], references: list[str]
) -> dict[str, dict]:
    """``{selection: {statistic: {key: value} or {reference: {key: value}}}}`` with values
    ``(estimate, low, high)``; selections are all n and the largest n."""
    nums_of_vehicles = sorted(results)
    selections = {
        "all n": nums_of_vehicles,
        f"n = {4 * nums_of_vehicles[-1]}": nums_of_vehicles[-1:],
    }
    aggregates = {}
    for selection, nums in selections.items():
        aggregates[selection] = {
            f"{coverage}_deficit": {
                key: _aggregate(results, nums, coverage, [key], mean_deficit) for key in filter_keys
            }
            for coverage in ("joint_coverage", "position_coverage")
        } | {
            name: {
                reference: {
                    key: _aggregate(results, nums, metric, [key, reference], statistic)
                    for key in filter_keys
                    if key != reference
                }
                for reference in references
            }
            for name, metric, statistic in (
                ("joint_overconfidence_ratio", "joint_coverage", deficit_ratio),
                ("position_overconfidence_ratio", "position_coverage", deficit_ratio),
                ("relative_position_rmse", "position_rmse", relative_rmse),
            )
        }
    return aggregates


def _cell(value: tuple[float, float, float], *, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{100 * value[0]:{sign}.1f} [{100 * value[1]:{sign}.1f}, {100 * value[2]:{sign}.1f}]"


def format_aggregates(aggregates: dict[str, dict], filter_keys: list[str]) -> str:
    selections = list(aggregates)
    lines = [
        "Aggregated over the state dimension (every n weighted equally), with 95% bootstrap CIs",
        "(episodes resampled per n, paired across filters).",
        "",
        (
            f"Overconfidence = coverage deficit {100 * COVERAGE_PROBABILITY:.0f}% - coverage [pp], "
            "mean over n:"
        ),
        format_table(
            ["filter"] + [f"{c} cov., {s}" for c in ("joint", "position") for s in selections],
            [
                [SHORT_NAMES[key]]
                + [
                    _cell(aggregates[s][f"{c}_coverage_deficit"][key])
                    for c in ("joint", "position")
                    for s in selections
                ]
                for key in filter_keys
            ],
            ["{}"] * (1 + 2 * len(selections)),
        ),
    ]
    references = list(aggregates[selections[0]]["relative_position_rmse"])
    for reference in references:
        keys = [key for key in filter_keys if key != reference]
        lines += [
            "",
            (
                f"Relative to {SHORT_NAMES[reference]} [%]: overconfidence ratio (mean deficit / "
                f"mean deficit of {SHORT_NAMES[reference]}; 100 = as overconfident) and position "
                f"RMSE change (mean over n of RMSE / RMSE of {SHORT_NAMES[reference]} - 1):"
            ),
            format_table(
                ["filter"]
                + [
                    f"{c}, {s}"
                    for c in ("joint overconf.", "position overconf.", "position RMSE")
                    for s in selections
                ],
                [
                    [SHORT_NAMES[key]]
                    + [
                        _cell(aggregates[s][name][reference][key], signed=signed)
                        for name, signed in (
                            ("joint_overconfidence_ratio", False),
                            ("position_overconfidence_ratio", False),
                            ("relative_position_rmse", True),
                        )
                        for s in selections
                    ]
                    for key in keys
                ],
                ["{}"] * (1 + 3 * len(selections)),
            ),
        ]
    return "\n".join(lines)
