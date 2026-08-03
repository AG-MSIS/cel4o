from typing import override

import numpy as np
import numpy.typing as npt

from ..systems import System
from .base import Filter


class EKF(Filter):
    """Extended Kalman filter. Requires ``dynamics_jacobian`` and ``sensor_jacobian`` of the
    system (w.r.t. the tangent space).

    The measurement may live on a [+]-manifold (the innovation is computed with [-]), but the
    state must be a vector space: The covariance is not moved to the updated mean as reference
    (see ``update``), which is only exact if [+] is the vector addition.
    """

    def __init__(self, system: System, initial_mean, initial_covariance: npt.NDArray[np.float64]):
        super().__init__(system)
        self._mean = initial_mean
        self._covariance = initial_covariance

    @override
    def predict(self, dt: float):
        jacobian = self._system.dynamics_jacobian(self._mean, dt)
        self._mean = self._system.step(self._mean, dt)
        self._covariance = _symmetrize(
            jacobian @ self._covariance @ jacobian.T + self._system.dynamics_covariance(dt)
        )

    @override
    def update(self, measurement, noise_covariance: npt.NDArray[np.float64]):
        jacobian = self._system.sensor_jacobian(self._mean)
        innovation = measurement - self._system.sensor(self._mean)
        innovation_covariance = jacobian @ self._covariance @ jacobian.T + noise_covariance
        # K = P H^T S^-1 <=> K^T = S^-1 H P since S and P are symmetric.
        kalman_gain = np.linalg.solve(innovation_covariance, jacobian @ self._covariance).T
        self._mean = self._mean + kalman_gain @ innovation
        # Joseph form
        a = np.eye(self._covariance.shape[0]) - kalman_gain @ jacobian
        self._covariance = _symmetrize(
            a @ self._covariance @ a.T + kalman_gain @ noise_covariance @ kalman_gain.T
        )
        # This covariance is expressed in the tangent space at the prior mean. On a general
        # [+]-manifold, it would have to be moved to the updated mean as reference:
        # P <- J P J^T with J = d/dd ((prior [+] (K y + d)) [-] posterior) at d = 0.
        # For vector-space states, J = I, so this step is omitted.

    @property
    @override
    def mean(self):
        return self._mean

    @property
    @override
    def covariance(self) -> npt.NDArray[np.float64]:
        return self._covariance


def _symmetrize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)
