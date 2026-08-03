"""Tests of the experiment code (moved from tests/test_compact_matrix_sqrt.py)."""

import numpy as np
import pytest

from compact_matrix_sqrt import dct_matrix
from paper.experiments.polar_matrix_sqrt import (
    atan2_gauss_hermite,
    errors_relative_to_sigma_fourth,
    method_plot_label,
    methods_by_first_circular_mean,
)
from paper.filters.ukf_utils import (
    SigmaPointMethod,
    covariance_of_sigma_points,
    generate_sigma_points,
    generate_sigma_weights,
    sigma_point_direction_count,
)


@pytest.mark.parametrize("method, num_directions", [
    (SigmaPointMethod.CEL4O_D_RANDOM, 4),
    (SigmaPointMethod.CEL4O_D_2N_RANDOM_Q, 8),
])
def test_generate_sigma_points_cel4o_D_random_Q_is_reproducible(method: SigmaPointMethod,
    num_directions: int
) -> None:
    n = 4
    mean = np.zeros(n)
    covariance = np.eye(n)

    points1 = np.asarray(generate_sigma_points(mean, covariance, method, kappa=1.0))
    points2 = np.asarray(generate_sigma_points(mean, covariance, method, kappa=1.0))

    assert sigma_point_direction_count(n, method) == num_directions
    assert points1.shape == (2 * num_directions + 1, n)
    np.testing.assert_allclose(points1, points2)


@pytest.mark.parametrize("method", [
    SigmaPointMethod.CEL4O_D_DCT,
    SigmaPointMethod.CEL4O_D_RANDOM,
    SigmaPointMethod.CEL4O_D_OPTIMIZED,
])
def test_generate_sigma_points_cel4o_D_n_variants(
    method: SigmaPointMethod,
) -> None:
    n = 4
    if method is SigmaPointMethod.CEL4O_D_OPTIMIZED:
        from paper.filters import ukf_utils

        ukf_utils._Q_CACHE[(n, n)] = dct_matrix(
            n, make_orthonormal=True
        ).T
    covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    points = np.asarray(
        generate_sigma_points(np.zeros(n), covariance, method, kappa=1.0)
    )

    assert points.shape == (2 * n + 1, n)
    np.testing.assert_allclose(
        covariance_of_sigma_points(
            [point for point in points], np.zeros(n),
            weights=generate_sigma_weights(n, kappa=1.0),
        ),
        covariance,
        atol=1e-12,
    )


def test_errors_relative_to_sigma_fourth() -> None:
    stddevs = np.array([0.5, 2.0])
    values = np.array([[1.0, 2.0], [4.0, 8.0]])

    normalized = errors_relative_to_sigma_fourth(
        stddevs, {SigmaPointMethod.CEL4O_D: values}
    )

    np.testing.assert_allclose(
        normalized[SigmaPointMethod.CEL4O_D],
        values / stddevs[:, None]**4,
    )


def test_methods_sorted_by_first_circular_mean_top_to_bottom() -> None:
    results = {
        SigmaPointMethod.CHOLESKY: np.array([[2.0, 100.0]]),
        SigmaPointMethod.SQRTM: np.array([[5.0, 0.0]]),
        SigmaPointMethod.CEL4O_D_DCT: np.array([[1.0, 200.0]]),
    }

    assert methods_by_first_circular_mean(results) == [
        SigmaPointMethod.SQRTM,
        SigmaPointMethod.CHOLESKY,
        SigmaPointMethod.CEL4O_D_DCT,
    ]


@pytest.mark.parametrize("method, label", [
    (SigmaPointMethod.CEL4O_D_DCT, "CEL4O_D_DCT"),
    (SigmaPointMethod.CEL4O_D_RANDOM, "CEL4O_D_RAN"),
    (SigmaPointMethod.CEL4O_D_OPTIMIZED, "CEL4O_D_OPT"),
    (SigmaPointMethod.CEL4O_D_2N_RANDOM_Q, "CEL4O_D_2N_RAN"),
    (SigmaPointMethod.CEL4O_D_2N_OPTIMIZED, "CEL4O_D_2N_OPT"),
])
def test_method_plot_label(method: SigmaPointMethod, label: str) -> None:
    assert method_plot_label(method) == label


def test_atan2_gauss_hermite_symmetric_case() -> None:
    variance = 0.05**2
    angle, angle_variance, cross_covariance = atan2_gauss_hermite(
        np.array([1.0, 0.0]), variance * np.eye(2)
    )

    assert angle == pytest.approx(0.0, abs=1e-15)
    assert angle_variance == pytest.approx(variance, rel=0.02)
    assert cross_covariance[0] == pytest.approx(0.0, abs=1e-15)
    assert cross_covariance[1] == pytest.approx(variance, rel=0.02)
