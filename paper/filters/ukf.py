from typing import override

import numpy as np
import numpy.typing as npt

from ..systems import System
from .base import Filter
from .ukf_utils import (
    SigmaPointMethod,
    covariance_of_sigma_points,
    generate_sigma_points,
    generate_sigma_weights,
    mean_of_sigma_points,
    sigma_point_direction_count,
)


class UKF(Filter):
    def __init__(
        self,
        system: System,
        initial_mean,
        initial_covariance: npt.NDArray[np.float64],
        *,
        sigma_point_method: SigmaPointMethod,
        kappa: float,
    ):
        super().__init__(system)
        self._mean = initial_mean
        self._covariance = initial_covariance
        self._sigma_point_method = sigma_point_method
        self._kappa = kappa
        n = self._covariance.shape[0]
        self._weights = generate_sigma_weights(
            n,
            kappa=self._kappa,
            num_directions=sigma_point_direction_count(n, self._sigma_point_method),
        )
        # Diagnostic of the last update: Largest absolute residual of a sigma measurement w.r.t.
        # the mean sigma measurement, per measurement component. Values above pi/2 for bearings
        # indicate that the sigma points "wrapped around" the observed object.
        self.last_max_sigma_residual: npt.NDArray[np.float64] | None = None

    @override
    def predict(self, dt: float):
        sigma_points = [
            self._system.step(_, dt)
            for _ in generate_sigma_points(
                self._mean, self._covariance, self._sigma_point_method, kappa=self._kappa
            )
        ]

        self._mean = mean_of_sigma_points(sigma_points, weights=self._weights)
        self._covariance = _symmetrize(
            covariance_of_sigma_points(sigma_points, self._mean, weights=self._weights)
            + self._system.dynamics_covariance(dt)
        )

    @override
    def update(self, measurement, noise_covariance: npt.NDArray[np.float64]):
        sigma_points = generate_sigma_points(
            self._mean, self._covariance, self._sigma_point_method, kappa=self._kappa
        )
        sigma_measurements = [self._system.sensor(_) for _ in sigma_points]
        mean_of_sigma_measurements = mean_of_sigma_points(sigma_measurements, weights=self._weights)
        covariance_of_sigma_measurements = (
            covariance_of_sigma_points(
                sigma_measurements, mean_of_sigma_measurements, weights=self._weights
            )
            + noise_covariance
        )
        covariance_of_states_and_measurements = covariance_of_sigma_points(
            sigma_points,
            self._mean,
            sigma_measurements,
            mean_of_sigma_measurements,
            weights=self._weights,
        )
        self.last_max_sigma_residual = np.max(
            np.abs(np.stack([_ - mean_of_sigma_measurements for _ in sigma_measurements])), axis=0
        )
        # K = P_xz S^-1 <=> K^T = S^-1 P_xz^T since S is symmetric.
        kalman_gain = np.linalg.solve(
            covariance_of_sigma_measurements, covariance_of_states_and_measurements.T
        ).T
        update = kalman_gain @ (measurement - mean_of_sigma_measurements)
        covariance = _symmetrize(
            self._covariance - kalman_gain @ covariance_of_states_and_measurements.T
        )
        sigma_points = generate_sigma_points(
            self._mean, covariance, self._sigma_point_method, offset=update, kappa=self._kappa
        )
        self._mean = mean_of_sigma_points(sigma_points, weights=self._weights)
        self._covariance = _symmetrize(
            covariance_of_sigma_points(sigma_points, self._mean, weights=self._weights)
        )

    @property
    @override
    def mean(self):
        return self._mean

    @property
    @override
    def covariance(self) -> npt.NDArray[np.float64]:
        return self._covariance

    @property
    @override
    def sigma_points(self):
        return generate_sigma_points(
            self._mean, self._covariance, self._sigma_point_method, kappa=self._kappa
        )


def _symmetrize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)
