from .base import Filter
from .ekf import EKF
from .ukf import UKF
from .ukf_utils import SigmaPointMethod

KAPPA = 1.0

CONSTRUCTORS = {
    "ukf_sqrtm": (
        lambda system, mean, covariance: UKF(
            system, mean, covariance, sigma_point_method=SigmaPointMethod.SQRTM, kappa=KAPPA
        ),
        "UKF (sym sqrt)",
    ),
    "ukf_chol": (
        lambda system, mean, covariance: UKF(
            system, mean, covariance, sigma_point_method=SigmaPointMethod.CHOLESKY, kappa=KAPPA
        ),
        "UKF (chol)",
    ),
    "ukf_d": (
        lambda system, mean, covariance: UKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.CEL4O_D_DCT,
            kappa=KAPPA,
        ),
        "UKF (cel4o-D)",
    ),
    "ukf_q": (
        lambda system, mean, covariance: UKF(
            system, mean, covariance, sigma_point_method=SigmaPointMethod.CEL4O_Q, kappa=KAPPA
        ),
        "UKF (cel4o-Q)",
    ),
    "ekf": (
        lambda system, mean, covariance: EKF(system, mean, covariance),
        "EKF",
    ),
}

__all__ = ["CONSTRUCTORS", "EKF", "Filter", "UKF"]
