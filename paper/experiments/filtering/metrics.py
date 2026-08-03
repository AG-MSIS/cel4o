"""Metrics on stacked episode results (leading axis = episode, see ``simulate.run_episode``).

All metrics are first computed per episode and then aggregated over episodes, so that confidence
intervals can be obtained by bootstrapping over (independent) episodes rather than over
(correlated) steps. Rates are plain means over episodes; RMSEs are pooled, i.e. the square root of
the mean over episodes of the per-episode MSE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

COVERAGE_PROBABILITY = 0.95
FLIP_THRESHOLD = np.pi / 2


def position_coverage(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode fraction of (step, vehicle) whose true position lies inside the filter's 95%
    position ellipse. Steps after divergence count as not covered.
    (episodes,)
    """
    threshold = chi2.ppf(COVERAGE_PROBABILITY, df=2)
    with np.errstate(invalid="ignore"):
        covered = data[f"{key}/position_nees"] <= threshold
    return np.mean(covered, axis=(1, 2))


def joint_coverage(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode fraction of steps whose true state lies inside the filter's 95% ellipsoid.
    (episodes,)
    """
    threshold = chi2.ppf(COVERAGE_PROBABILITY, df=data["true"].shape[-1])
    with np.errstate(invalid="ignore"):
        covered = data[f"{key}/nees"] <= threshold
    return np.mean(covered, axis=1)


def errors(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Estimation error per (episode, step, vehicle, [x, y, vx, vy]); NaN after divergence."""
    num_of_vehicles = data["true"].shape[-1] // 4
    shape = (*data["true"].shape[:-1], num_of_vehicles, 4)
    return np.reshape(data["true"], shape) - np.reshape(data[f"{key}/mean"], shape)


def position_mse(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode position MSE per axis (i.e. over x and y separately, not the squared Euclidean
    distance) over all steps before divergence.
    (episodes,)
    """
    return np.nanmean(np.square(errors(data, key)[..., :2]), axis=(1, 2, 3))


def velocity_mse(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode velocity MSE per axis over all steps before divergence.
    (episodes,)
    """
    return np.nanmean(np.square(errors(data, key)[..., 2:]), axis=(1, 2, 3))


def diverged(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode indicator whether the filter failed numerically.
    (episodes,)
    """
    return (data[f"{key}/diverged_at"] < data["true"].shape[1]).astype(np.float64)


def flip_rate(data: dict, key: str) -> npt.NDArray[np.float64]:
    """Per-episode fraction of bearing updates in which at least one sigma point's predicted
    bearing deviates by more than 90 degrees from the predicted mean bearing (NaN for filters
    without sigma points).
    (episodes,)
    """
    residual = data[f"{key}/max_sigma_residual"]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN episodes, e.g. for the EKF
        return np.nanmean(
            np.where(np.isnan(residual), np.nan, residual > FLIP_THRESHOLD), axis=(1, 2)
        )


Transform = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None

# name -> (per-episode values, transform). The metric is ``transform(mean over episodes)``.
METRICS: dict[str, tuple[Callable[[dict, str], npt.NDArray[np.float64]], Transform]] = {
    "position_coverage": (position_coverage, None),
    "joint_coverage": (joint_coverage, None),
    "position_rmse": (position_mse, np.sqrt),
    "velocity_rmse": (velocity_mse, np.sqrt),
    "diverged": (diverged, None),
    "flip_rate": (flip_rate, None),
}


def metric_value(data: dict, key: str, name: str) -> float:
    """Point estimate of a metric (without confidence interval)."""
    per_episode, transform = METRICS[name]
    values = per_episode(data, key)
    values = values[np.isfinite(values)]
    return float(_aggregate(values, transform)) if len(values) > 0 else np.nan


def _aggregate(values: npt.NDArray[np.float64], transform: Transform, axis=None):
    mean = np.mean(values, axis=axis)
    return mean if transform is None else transform(mean)


def bootstrap_ci(
    values: npt.NDArray[np.float64],
    transform: Transform = None,
    *,
    reference_values: npt.NDArray[np.float64] | None = None,
    relative: bool = False,
    num_of_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``transform(mean(values))`` and its bootstrap percentile confidence interval, resampling
    episodes.

    If ``reference_values`` (of the same episodes) are given, the statistic is the difference
    ``transform(mean(values)) - transform(mean(reference_values))`` and episodes are resampled
    jointly (paired bootstrap). With ``relative``, the statistic is the ratio minus one instead of
    the difference.
    """
    valid = np.isfinite(values)
    if reference_values is not None:
        valid &= np.isfinite(reference_values)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    index = rng.integers(0, np.sum(valid), size=(num_of_resamples, np.sum(valid)))

    def statistic(selection, axis=None):
        result = _aggregate(values[valid][selection], transform, axis)
        if reference_values is not None:
            reference = _aggregate(reference_values[valid][selection], transform, axis)
            result = result / reference - 1 if relative else result - reference
        return result

    resampled = statistic(index, axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled, [alpha, 1 - alpha])
    return float(statistic(slice(None))), float(low), float(high)


def bootstrap_over_dimensions(
    values: Sequence[npt.NDArray[np.float64]],
    statistic: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    *,
    num_of_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A statistic of the episode means at several state dimensions and its bootstrap percentile
    confidence interval.

    :param values: Per dimension, the per-episode values (quantities, episodes) of all quantities
        that enter the statistic (e.g. a metric of a filter and of a reference filter, which ran on
        the same episodes). Episodes are resampled independently per dimension (each dimension has
        its own episodes), but jointly for all quantities of a dimension (paired bootstrap).
    :param statistic: Maps episode means (..., dimensions, quantities) to (...).
    """
    rng = np.random.default_rng(seed)
    means = []
    resampled = []
    for values_of_dimension in values:
        valid = values_of_dimension[:, np.all(np.isfinite(values_of_dimension), axis=0)]
        index = rng.integers(0, valid.shape[1], size=(num_of_resamples, valid.shape[1]))
        means.append(np.mean(valid, axis=1))
        resampled.append(np.mean(valid[:, index], axis=-1).T)  # (resamples, quantities)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(statistic(np.stack(resampled, axis=1)), [alpha, 1 - alpha])
    return float(statistic(np.array(means))), float(low), float(high)


def summarize(
    data: dict, filter_keys: list[str]
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Value and 95% CI of every metric for every filter."""
    return {
        key: {
            name: bootstrap_ci(per_episode(data, key), transform)
            for name, (per_episode, transform) in METRICS.items()
        }
        for key in filter_keys
    }


def paired_summary(
    data: dict, filter_keys: list[str], references: list[str]
) -> dict[str, dict[str, dict[str, tuple[float, float, float]]]]:
    """Difference ``key - reference`` of every metric with a paired 95% CI.

    All filters run on the same episodes, so resampling episodes jointly removes the
    episode-to-episode variation that both filters share. This is the appropriate test whether one
    filter is better than another (marginal CIs of both filters may overlap although the
    difference is significant).
    """
    return {
        key: {
            reference: {
                name: bootstrap_ci(
                    per_episode(data, key),
                    transform,
                    reference_values=per_episode(data, reference),
                )
                for name, (per_episode, transform) in METRICS.items()
            }
            for reference in references
            if reference != key
        }
        for key in filter_keys
    }


def relative_summary(
    data: dict, filter_keys: list[str], reference: str, name: str
) -> dict[str, tuple[float, float, float]]:
    """Ratio ``key / reference - 1`` of metric ``name`` for every filter with a paired 95% CI."""
    per_episode, transform = METRICS[name]
    return {
        key: bootstrap_ci(
            per_episode(data, key),
            transform,
            reference_values=per_episode(data, reference),
            relative=True,
        )
        for key in filter_keys
    }


def rho_quantiles(
    data: dict, quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
) -> npt.NDArray[np.float64]:
    """Quantiles of rho over all (episode, step, vehicle)."""
    rho = data["rho"]
    return np.quantile(rho[np.isfinite(rho)], quantiles)


def rho_regime_fractions(
    data: dict, edges: tuple[float, ...] = (0.1, 0.15, 0.3, 0.5)
) -> npt.NDArray[np.float64]:
    """Fractions of observations with rho in [0, e_0), [e_0, e_1), ..., [e_last, inf)."""
    rho = data["rho"]
    rho = rho[np.isfinite(rho)]
    bins = np.concatenate([[0.0], edges, [np.inf]])
    return np.histogram(rho, bins=bins)[0] / len(rho)
