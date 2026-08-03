"""Compare matrix square roots in an unscented transform to polar coordinates.

The state has 25 dimensions, and the nonlinear function is applied to all
coordinate pairs ``(0, 1), (2, 3), ...``.  For every covariance scale and
coordinate pair, random means are drawn uniformly from the unit circle.  A
high-order two-dimensional Gauss-Hermite integration is used as ground truth,
and errors are averaged over random means and coordinate pairs.

Run, for example::

    python -m paper.experiments.polar_matrix_sqrt --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.polynomial.hermite import hermgauss
from tqdm import tqdm

from paper.filters.ukf_utils import (
    SigmaPointMethod,
    covariance_of_sigma_points,
    generate_sigma_points,
    generate_sigma_weights,
    mean_of_sigma_points,
    sigma_point_direction_count,
)
from paper.paths import DATA_DIR, FIGURES_DIR
from paper.systems.vehicles_2d import AngleBatch


METHODS = tuple(SigmaPointMethod)


@dataclass
class Estimate:
    mean: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    cross_covariance: npt.NDArray[np.float64]


class PolarMeasurement:
    """A bearing/range value on S1 x R with a two-dimensional tangent space."""

    def __init__(self, values: npt.ArrayLike):
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.shape != (2,):
            raise ValueError("polar values must have shape (2,)")

    def __sub__(self, other: PolarMeasurement) -> npt.NDArray[np.float64]:
        difference = self.values - other.values
        difference[0] = AngleBatch.normalize(difference[0])
        return difference

    @classmethod
    def mean(cls, states: list[PolarMeasurement], *,
        weights: npt.NDArray[np.float64]) -> PolarMeasurement:
        values = np.stack([state.values for state in states])
        angle = np.arctan2(weights @ np.sin(values[:, 0]),
                           weights @ np.cos(values[:, 0]))
        radius = weights @ values[:, 1]
        return cls(np.array([angle, radius]))


def covariance_matrix(n: int, stddev: float, direction: float | npt.ArrayLike,
    pair: tuple[int, int] = (0, 1)
) -> npt.NDArray[np.float64]:
    """Build Sigma with orientable ellipses in coordinate pairs.

    Along ``direction`` the standard deviation is ``0.75 * stddev``; in
    the perpendicular direction it is ``1.25 * stddev``.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    pairs = [pair] if np.ndim(direction) == 0 else coordinate_pairs(n)
    directions = np.full(len(pairs), float(direction)) if np.ndim(direction) == 0 else np.asarray(direction)
    if directions.shape != (len(pairs),):
        raise ValueError("direction must be scalar or have one entry per coordinate pair")
    covariance = stddev**2 * np.eye(n, dtype=np.float64)
    active_variances = stddev**2 * np.diag([0.75**2, 1.25**2])
    for active_pair, active_direction in zip(pairs, directions):
        if not (0 <= active_pair[0] < n and 0 <= active_pair[1] < n and active_pair[0] != active_pair[1]):
            raise ValueError("pair must contain two distinct valid dimensions")
        cosine, sine = np.cos(active_direction), np.sin(active_direction)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        covariance[np.ix_(active_pair, active_pair)] = rotation @ active_variances @ rotation.T
    return covariance


def coordinate_pairs(n: int) -> list[tuple[int, int]]:
    """Return coordinate pairs ``(0, 1), (2, 3), ...``."""
    if n < 2:
        raise ValueError("n must be at least 2")
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def measurement(x0: npt.ArrayLike, x1: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return bearing and range, with the measurement dimension last."""
    return np.stack((np.arctan2(x1, x0), np.hypot(x0, x1)), axis=-1)


def transform(points: npt.NDArray[np.float64], pair: tuple[int, int] = (0, 1)) -> list[PolarMeasurement]:
    """Apply the configured measurement to points stored by row."""
    return [PolarMeasurement(measurement(x[pair[0]], x[pair[1]])) for x in points]


def ut_estimate_from_points(mean: npt.NDArray[np.float64], points: list,
    weights: npt.NDArray[np.float64], pair: tuple[int, int] = (0, 1)
) -> Estimate:
    """Evaluate the polar transform for one pair on a sigma-point set."""
    point_array = np.asarray(points)
    measurements = transform(point_array, pair)
    measurement_mean = mean_of_sigma_points(measurements, weights=weights)
    covariance = covariance_of_sigma_points(
        measurements, measurement_mean, weights=weights
    )
    active_points = [x[list(pair)] for x in points]
    cross_covariance = covariance_of_sigma_points(
        active_points, mean[list(pair)], measurements, measurement_mean, weights=weights
    )
    return Estimate(measurement_mean.values, covariance, cross_covariance)


def ut_estimate(mean: npt.NDArray[np.float64], covariance: npt.NDArray[np.float64],
    method: SigmaPointMethod, kappa: float, pair: tuple[int, int] = (0, 1)
) -> Estimate:
    points = generate_sigma_points(mean, covariance, method, kappa=kappa)
    weights = generate_sigma_weights(
        len(mean),
        kappa=kappa,
        num_directions=sigma_point_direction_count(len(mean), method),
    )
    return ut_estimate_from_points(mean, points, weights, pair)

@lru_cache(maxsize=None)
def standard_normal_hermgauss(
    order: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return cached Gauss-Hermite nodes and weights for N(0, 1)."""
    if order < 1:
        raise ValueError("order must be positive")
    z, w = hermgauss(order)
    z = np.sqrt(2.0) * z
    w = w / np.sqrt(np.pi)
    z.setflags(write=False)
    w.setflags(write=False)
    return z, w


def polar_gauss_hermite(
    mean: npt.ArrayLike, cov: npt.ArrayLike, order: int = 100
) -> Estimate:
    """High-order 2D Gauss-Hermite reference for bearing and range."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.shape != (2,):
        raise ValueError("mean must have shape (2,)")
    if cov.shape != (2, 2):
        raise ValueError("cov must have shape (2, 2)")
    z, w = standard_normal_hermgauss(order)

    # x = mean + L @ z, z ~ N(0, I)
    L = np.linalg.cholesky(cov)

    # Tensor-product 2D quadrature
    z0, z1 = np.meshgrid(z, z, indexing="ij")
    weights = np.outer(w, w)

    Z = np.stack((z0.ravel(), z1.ravel()))
    X = mean[:, None] + L @ Z
    weights = weights.ravel()

    # Use exactly the same measurement as the sigma-point path.
    Y = measurement(X[0], X[1])

    y_mean = np.array([
        np.arctan2(weights @ np.sin(Y[:, 0]), weights @ np.cos(Y[:, 0])),
        weights @ Y[:, 1],
    ])
    dy = Y - y_mean
    dy[:, 0] = AngleBatch.normalize(dy[:, 0])
    y_covariance = (dy * weights[:, None]).T @ dy
    dx = X - mean[:, None]
    cross_cov = dx @ (weights[:, None] * dy)

    return Estimate(y_mean, y_covariance, cross_cov)


def atan2_gauss_hermite(
    mean: npt.ArrayLike, cov: npt.ArrayLike, order: int = 100
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Backward-compatible angular projection of :func:`polar_gauss_hermite`."""
    estimate = polar_gauss_hermite(mean, cov, order)
    return (float(estimate.mean[0]), float(estimate.covariance[0, 0]),
            estimate.cross_covariance[:, 0])

def ground_truth(mean: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    pair: tuple[int, int] = (0, 1),
    *, gauss_hermite_order: int = 100,
) -> Estimate:
    lp = list(pair)
    return polar_gauss_hermite(
        mean[lp], covariance[np.ix_(pair, pair)], order=gauss_hermite_order
    )


def linearized_estimate(
    mean: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    pair: tuple[int, int] = (0, 1),
) -> Estimate:
    """Propagate moments with the first-order polar linearization."""
    active_mean = np.asarray(mean[list(pair)], dtype=np.float64)
    active_covariance = np.asarray(covariance[np.ix_(pair, pair)], dtype=np.float64)
    squared_radius = float(active_mean @ active_mean)
    if squared_radius == 0.0:
        raise ValueError("polar coordinates cannot be linearized at the origin")
    radius = np.sqrt(squared_radius)
    jacobian = np.array([
        [-active_mean[1] / squared_radius, active_mean[0] / squared_radius],
        [active_mean[0] / radius, active_mean[1] / radius],
    ])
    return Estimate(
        mean=measurement(active_mean[0], active_mean[1]),
        covariance=jacobian @ active_covariance @ jacobian.T,
        cross_covariance=active_covariance @ jacobian.T,
    )

def monte_carlo_estimate(
    mean: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    standard_normals: npt.NDArray[np.float64],
    pair: tuple[int, int] = (0, 1),
) -> Estimate:
    """Ground truth using only the analytically sufficient 2D marginal."""
    active_mean = mean[list(pair)]
    active_covariance = covariance[np.ix_(pair, pair)]
    samples = active_mean + standard_normals @ np.linalg.cholesky(active_covariance).T
    measurements = measurement(samples[:, 0], samples[:, 1])
    measurement_mean = np.array([
        np.arctan2(np.mean(np.sin(measurements[:, 0])),
                   np.mean(np.cos(measurements[:, 0]))),
        np.mean(measurements[:, 1]),
    ])
    residual = measurements - measurement_mean
    residual[:, 0] = AngleBatch.normalize(residual[:, 0])
    centered = samples - active_mean
    measurement_covariance = residual.T @ residual / len(samples)
    cross_covariance = centered.T @ residual / len(samples)
    return Estimate(measurement_mean, measurement_covariance, cross_covariance)


def errors(estimate: Estimate, truth: Estimate) -> npt.NDArray[np.float64]:
    # Embed the angular component as (cos, sin), then append the range.  This
    # makes the Euclidean error invariant to the chosen angle convention.
    mean_error = np.linalg.norm([
        np.cos(estimate.mean[0]) - np.cos(truth.mean[0]),
        np.sin(estimate.mean[0]) - np.sin(truth.mean[0]),
        estimate.mean[1] - truth.mean[1],
    ])
    covariance_error = np.linalg.norm(estimate.covariance - truth.covariance)
    cross_error = np.linalg.norm(estimate.cross_covariance - truth.cross_covariance)
    return np.array([mean_error, covariance_error, cross_error])


def run_experiment(stddevs: npt.NDArray[np.float64], *,
    n: int = 25, repeats: int = 40, gauss_hermite_order: int = 100,
    kappa: float = 1.0, seed: int = 0,
) -> tuple[
    dict[SigmaPointMethod, npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return method errors, ground-truth scales, and linearized errors.

    Ground-truth columns are the Frobenius norms of the 2x2 measurement
    covariance and the 2x2 cross-covariance.
    """
    rng = np.random.default_rng(seed)
    result = {method: np.zeros((len(stddevs), 3)) for method in METHODS}
    reference_scales = np.zeros((len(stddevs), 2), dtype=np.float64)
    linearized_errors = np.zeros((len(stddevs), 3), dtype=np.float64)
    pairs = coordinate_pairs(n)

    for scale_index, stddev in enumerate(tqdm(stddevs, desc="polar", unit="scale")):
        accumulated = {method: np.zeros(3) for method in METHODS}
        accumulated_reference = np.zeros(2, dtype=np.float64)
        accumulated_linearized = np.zeros(3, dtype=np.float64)
        for _ in range(repeats):
            phases = rng.uniform(-np.pi, np.pi, size=len(pairs))
            ellipse_directions = rng.uniform(-np.pi, np.pi, size=len(pairs))
            mean = np.zeros(n)
            for pair, phase in zip(pairs, phases):
                mean[list(pair)] = [np.cos(phase), np.sin(phase)]
            covariance = covariance_matrix(n, float(stddev), ellipse_directions)
            method_points = {}
            method_weights = {}
            for method in METHODS:
                method_points[method] = generate_sigma_points(mean, covariance, method, kappa=kappa)
                method_weights[method] = generate_sigma_weights(n, kappa=kappa,
                    num_directions=sigma_point_direction_count(n, method))

            for pair in pairs:
                truth = ground_truth(
                    mean, covariance, pair,
                    gauss_hermite_order=gauss_hermite_order,
                )
                accumulated_reference += (
                    np.linalg.norm(truth.covariance),
                    np.linalg.norm(truth.cross_covariance),
                )
                accumulated_linearized += errors(
                    linearized_estimate(mean, covariance, pair), truth
                )
                for method in METHODS:
                    estimate = ut_estimate_from_points(mean, method_points[method],
                        method_weights[method], pair)
                    accumulated[method] += errors(estimate, truth)
        for method in METHODS:
            result[method][scale_index] = accumulated[method] / (repeats * len(pairs))
        reference_scales[scale_index] = accumulated_reference / (repeats * len(pairs))
        linearized_errors[scale_index] = accumulated_linearized / (repeats * len(pairs))
    return result, reference_scales, linearized_errors


def parse_method(value: str) -> SigmaPointMethod:
    """Parse a SigmaPointMethod name for argparse."""
    try:
        return SigmaPointMethod[value.upper()]
    except KeyError as error:
        choices = ", ".join(method.name for method in SigmaPointMethod)
        raise argparse.ArgumentTypeError(f"method must be one of: {choices}") from error


def errors_relative_to_method(results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
    reference_method: SigmaPointMethod, eps: float = 1e-15
) -> dict[SigmaPointMethod, npt.NDArray[np.float64]]:
    """Return method errors divided by the selected reference method."""
    reference = results.get(reference_method)
    if reference is None:
        raise ValueError(f"{reference_method.name} results are required for relative plotting")
    return {method: values / np.maximum(reference, eps) for method, values in results.items()}


def errors_relative_to_sigma_fourth(
    stddevs: npt.NDArray[np.float64],
    results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
) -> dict[SigmaPointMethod, npt.NDArray[np.float64]]:
    """Return method errors divided by ``sigma**4``."""
    stddevs = np.asarray(stddevs, dtype=np.float64)
    if stddevs.ndim != 1:
        raise ValueError("stddevs must be one-dimensional")
    if np.any(~np.isfinite(stddevs)) or np.any(stddevs <= 0.0):
        raise ValueError("stddevs must contain only finite positive values")
    denominator = stddevs[:, None]**4
    normalized = {}
    for method, values in results.items():
        if values.ndim != 2 or values.shape[0] != len(stddevs):
            raise ValueError(
                f"{method.name} results must have one row per standard deviation"
            )
        normalized[method] = values / denominator
    return normalized


def mean_sigma_fourth_normalized_errors(
    stddevs: npt.NDArray[np.float64],
    results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
    linearized_errors: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Average ``error / sigma**4`` over all investigated sigma values."""
    normalized = errors_relative_to_sigma_fourth(stddevs, results)
    linearized_errors = np.asarray(linearized_errors, dtype=np.float64)
    if linearized_errors.shape != (len(stddevs), 3):
        raise ValueError("linearized_errors must have shape (len(stddevs), 3)")
    denominator = np.asarray(stddevs, dtype=np.float64)[:, None]**4
    summary = {
        method.name: np.mean(values, axis=0)
        for method, values in normalized.items()
    }
    summary["LINEARIZED_REFERENCE"] = np.mean(
        linearized_errors / denominator, axis=0
    )
    return summary


def _mean_polar_sort_key(
    item: tuple[object, npt.NDArray[np.float64]],
) -> tuple[bool, float, str]:
    """Sort curves by descending mean polar-mean error, non-finite last."""
    name, values = item
    mean_error = float(np.mean(values[:, 0] if values.ndim == 2 else values[0]))
    return (
        not np.isfinite(mean_error),
        -mean_error if np.isfinite(mean_error) else 0.0,
        str(name),
    )


def _latex_decimal(value: float) -> str:
    """Format a floating-point value with three decimal places for LaTeX."""
    if not np.isfinite(value):
        return rf"\text{{{value}}}"
    return f"{value:.3f}"


def write_sigma_fourth_summary(
    stddevs: npt.NDArray[np.float64],
    results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
    linearized_errors: npt.NDArray[np.float64],
    output: Path,
) -> dict[str, npt.NDArray[np.float64]]:
    """Print and save mean sigma^4-normalized errors in the three categories."""
    summary = mean_sigma_fourth_normalized_errors(
        stddevs, results, linearized_errors
    )
    summary = dict(sorted(summary.items(), key=_mean_polar_sort_key))
    labels = ("polar mean", "measurement covariance", "cross-covariance")
    name_width = max(len(name) for name in summary)
    print("\nMean sigma^4-normalized errors (mean over sigma values):")
    print(
        f"{'method':<{name_width}}  "
        + "  ".join(f"{label:>24}" for label in labels)
    )
    for name, values in summary.items():
        print(f"{name:<{name_width}}  " + "  ".join(f"{value:24.3f}" for value in values))

    latex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        (r"Method & Polar mean & Measurement covariance "
         r"& Cross-covariance \\"),
        r"\hline",
    ]
    for name, values in summary.items():
        latex_name = name.replace("_", r"\_")
        latex_values = " & ".join(_latex_decimal(float(value)) for value in values)
        latex_lines.append(f"{latex_name} & {latex_values} " + r"\\")
    latex_lines.extend([r"\hline", r"\end{tabular}", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(latex_lines), encoding="utf-8")
    return summary


def methods_by_first_circular_mean(
    results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
) -> list[SigmaPointMethod]:
    """Order methods top-to-bottom by their first circular-mean value."""
    def sort_key(method: SigmaPointMethod) -> tuple[bool, float, str]:
        value = float(results[method][0, 0])
        return (not np.isfinite(value), -value if np.isfinite(value) else 0.0, method.name)

    return sorted(results, key=sort_key)


def method_plot_label(method: SigmaPointMethod) -> str:
    """Return a compact method name for plot legends."""
    return method.name.replace("_RANDOM_Q", "_RAN").replace(
        "_RANDOM", "_RAN"
    ).replace("_OPTIMIZED", "_OPT")


def plot_results(stddevs: npt.NDArray[np.float64], results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
    output: Path, *, circular_mean_ymin: float | None = None,
    reference_method: SigmaPointMethod | None = None,
    normalize_by_sigma_fourth: bool = True,
    reference_scales: npt.NDArray[np.float64] | None = None,
    linearized_errors: npt.NDArray[np.float64] | None = None) -> None:
    labels = ("polar mean", "measurement covariance", "cross-covariance")
    if reference_method is not None:
        plot_values = errors_relative_to_method(results, reference_method)
        ylabel = f"error / {reference_method.name} error"
    elif normalize_by_sigma_fourth:
        plot_values = errors_relative_to_sigma_fourth(stddevs, results)
        ylabel = r"error / $\sigma^4$"
    else:
        plot_values = results
        ylabel = "mean relative error"
    method_order = [
        method for method, _ in sorted(plot_values.items(), key=_mean_polar_sort_key)
    ]
    if linearized_errors is not None:
        linearized_errors = np.asarray(linearized_errors, dtype=np.float64)
        if linearized_errors.shape != (len(stddevs), 3):
            raise ValueError("linearized_errors must have shape (len(stddevs), 3)")
        if reference_method is not None:
            plotted_linearized_errors = linearized_errors / np.maximum(
                results[reference_method][:, :3], 1e-15
            )
        elif normalize_by_sigma_fourth:
            plotted_linearized_errors = linearized_errors / stddevs[:, None]**4
        else:
            plotted_linearized_errors = linearized_errors
    if reference_scales is not None:
        reference_scales = np.asarray(reference_scales, dtype=np.float64)
        if reference_scales.shape != (len(stddevs), 2):
            raise ValueError("reference_scales must have shape (len(stddevs), 2)")
        plotted_reference_scales = reference_scales / stddevs[:, None]**4 \
            if normalize_by_sigma_fourth else reference_scales
    fig, axes = plt.subplots(1, len(labels), figsize=(18, 4.2), sharex=True)
    for column, (axis, label) in enumerate(zip(axes, labels, strict=True)):
        for method in method_order:
            values = plot_values[method]
            axis.plot(stddevs, values[:, column], marker=".", markersize=4,
                linewidth=1.2, label=method_plot_label(method))
        axis.set_title(label)
        axis.set_xlabel("standard deviation sigma")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.set_yscale("log")
        if column == 0 and circular_mean_ymin is not None:
            axis.set_ylim(bottom=circular_mean_ymin)
        axis.grid(True, which="both", alpha=0.3)
        if reference_scales is not None and column in (1, 2):
            # Reference curves are contextual and must not change the visible
            # error range, particularly at the small-sigma end.
            method_ylim = axis.get_ylim()
            axis.plot(stddevs, plotted_reference_scales[:, column - 1],
                color="black", linewidth=0.7, zorder=1,
            )
            axis.set_ylim(method_ylim)
        if linearized_errors is not None:
            # This is a contextual EKF-style baseline, not another competing
            # sigma-point method, and must therefore not affect autoscaling.
            method_ylim = axis.get_ylim()
            axis.plot(stddevs, plotted_linearized_errors[:, column],
                color="black", linewidth=0.7, linestyle="--", label="linearized propagation", zorder=1,
            )
            axis.set_ylim(method_ylim)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    polar_means = {
        method_plot_label(method): float(np.mean(plot_values[method][:, 0]))
        for method in method_order
    }
    if linearized_errors is not None:
        polar_means["linearized propagation"] = float(
            np.mean(plotted_linearized_errors[:, 0])
        )
    legend_entries = sorted(
        zip(handles, legend_labels),
        key=lambda entry: (
            not np.isfinite(polar_means[entry[1]]),
            (-polar_means[entry[1]]
             if np.isfinite(polar_means[entry[1]]) else 0.0),
            entry[1],
        ),
    )
    axes[0].legend(
        [entry[0] for entry in legend_entries],
        [entry[1] for entry in legend_entries],
        loc="lower left", ncol=2,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def save_results(
    stddevs: npt.NDArray[np.float64],
    results: dict[SigmaPointMethod, npt.NDArray[np.float64]],
    reference_scales: npt.NDArray[np.float64],
    linearized_errors: npt.NDArray[np.float64],
    output: Path, *,
    n: int, repeats: int, gauss_hermite_order: int, kappa: float, seed: int,
) -> None:
    """Store numerical results and experiment parameters as a compressed NPZ."""
    arrays = {f"errors_{method.name}": values for method, values in results.items()}
    np.savez_compressed(
        output,
        stddevs=stddevs,
        error_columns=np.array(
            [
                "polar_mean_euclidean",
                "measurement_covariance_frobenius",
                "cross_covariance_frobenius",
            ]
        ),
        n=np.array(n),
        repeats=np.array(repeats),
        gauss_hermite_order=np.array(gauss_hermite_order),
        kappa=np.array(kappa),
        seed=np.array(seed),
        ground_truth_measurement_covariance_norm=reference_scales[:, 0],
        ground_truth_cross_covariance_norm=reference_scales[:, 1],
        linearized_errors=linearized_errors,
        **arrays,
    )


def load_results(input_path: Path) -> tuple[
    npt.NDArray[np.float64],
    dict[SigmaPointMethod, npt.NDArray[np.float64]],
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    """Load saved experiment results from an NPZ file."""
    with np.load(input_path) as archive:
        stddevs = np.asarray(archive["stddevs"], dtype=np.float64)
        legacy_names = {
            SigmaPointMethod.CEL4O_D_DCT: "CEL4O_D",
            SigmaPointMethod.CEL4O_D_RANDOM: "CEL4O_D_N_RANDOM_Q",
        }
        results = {}
        for method in METHODS:
            key = f"errors_{method.name}"
            if key not in archive.files and method in legacy_names:
                key = f"errors_{legacy_names[method]}"
            if key in archive.files:
                # Older archives contain a fourth singularity-distance column.
                results[method] = np.asarray(archive[key], dtype=np.float64)[:, :3]
        reference_scales = None
        covariance_key = (
            "ground_truth_measurement_covariance_norm"
            if "ground_truth_measurement_covariance_norm" in archive.files
            else "ground_truth_angular_variance"
        )
        reference_keys = (covariance_key, "ground_truth_cross_covariance_norm")
        if all(key in archive.files for key in reference_keys):
            reference_scales = np.column_stack(
                [np.asarray(archive[key], dtype=np.float64) for key in reference_keys]
            )
        linearized_errors = (
            np.asarray(archive["linearized_errors"], dtype=np.float64)
            if "linearized_errors" in archive.files else None
        )
    return stddevs, results, reference_scales, linearized_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--gauss-hermite-order", type=int, default=100,
        help="Gauss-Hermite order per dimension for ground truth (default: 100)")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-stddev", type=float, default=5e-3)
    parser.add_argument("--max-stddev", type=float, default=1.0)
    parser.add_argument("--num-stddevs", type=int, default=40)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "polar_matrix_sqrt_errors.png")
    parser.add_argument("--plot-from-results", type=Path,
        help="Load an existing NPZ result file and only redraw the UKF error plot")
    parser.add_argument("--reference-method", type=parse_method,
        help="Normalize errors by this method instead of sigma^4")
    parser.add_argument("--absolute-errors", action="store_true",
        help="Plot absolute errors without sigma^4 or method normalization")
    parser.add_argument("--circular-mean-ymin", type=float, default=0.0,
        help="Lower y-limit for the circular mean plot; use 0 to disable clipping")
    parser.add_argument("--results-output", type=Path,
        help="NPZ output path (default: data directory, stem of --output, suffix .npz)",
    )
    parser.add_argument("--summary-output", type=Path,
        help=("LaTeX summary output path (default: same directory and stem as "
              "--output, followed by _sigma4_summary.tex)"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    circular_mean_ymin = None if args.circular_mean_ymin <= 0.0 else args.circular_mean_ymin
    if args.gauss_hermite_order < 1:
        raise ValueError("--gauss-hermite-order must be positive")
    reference_method = None if args.absolute_errors else args.reference_method
    normalize_by_sigma_fourth = not args.absolute_errors and reference_method is None
    if args.plot_from_results is not None:
        stddevs, results, reference_scales, linearized_errors = load_results(
            args.plot_from_results
        )
        plot_results(stddevs, results, args.output, circular_mean_ymin=circular_mean_ymin,
            reference_method=reference_method,
            normalize_by_sigma_fourth=normalize_by_sigma_fourth,
            reference_scales=reference_scales,
            linearized_errors=linearized_errors)
        summary_output = args.summary_output or args.output.with_name(
            f"{args.output.stem}_sigma4_summary.tex"
        )
        if linearized_errors is not None:
            write_sigma_fourth_summary(
                stddevs, results, linearized_errors, summary_output
            )
            print(f"Wrote {args.output} and {summary_output} from {args.plot_from_results}")
        else:
            print(f"Wrote {args.output} from {args.plot_from_results}")
            print("No summary written: loaded results contain no linearized errors.")
        return

    stddevs = np.geomspace(args.min_stddev, args.max_stddev, args.num_stddevs)
    results, reference_scales, linearized_errors = run_experiment(stddevs,
        n=args.n, repeats=args.repeats, gauss_hermite_order=args.gauss_hermite_order,
        kappa=args.kappa, seed=args.seed,
    )
    plot_results(stddevs, results, args.output, circular_mean_ymin=circular_mean_ymin,
        reference_method=reference_method,
        normalize_by_sigma_fourth=normalize_by_sigma_fourth,
        reference_scales=reference_scales,
        linearized_errors=linearized_errors)
    results_output = args.results_output or DATA_DIR / args.output.with_suffix(".npz").name
    save_results(stddevs, results, reference_scales, linearized_errors, results_output,
                 n=args.n, repeats=args.repeats, gauss_hermite_order=args.gauss_hermite_order,
                 kappa=args.kappa, seed=args.seed,
    )
    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}_sigma4_summary.tex"
    )
    write_sigma_fourth_summary(stddevs, results, linearized_errors, summary_output)
    print(f"Wrote {args.output}, {results_output}, and {summary_output}")


if __name__ == "__main__":
    main()
