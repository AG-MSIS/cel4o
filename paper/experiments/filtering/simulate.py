"""Simulation of single episodes of the multi-vehicle bearing-only tracking experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...filters import CONSTRUCTORS, UKF, Filter
from ...systems import SystemRunner, tangent_sample
from .scenario import ScenarioConfig, sample_initial_mean

if TYPE_CHECKING:
    from ...systems.vehicles_2d import Vehicle2D, VehicleCollection2D

FILTER_ERRORS = (np.linalg.LinAlgError, FloatingPointError, ValueError, RuntimeError)


def episode_seed(base_seed: int, num_of_vehicles: int, episode: int) -> np.random.SeedSequence:
    """Seed of an episode. It depends only on these three numbers, i.e. results are reproducible
    independently of how episodes are distributed to worker processes."""
    return np.random.SeedSequence([base_seed, num_of_vehicles, episode])


def position_blocks(
    covariance: npt.NDArray[np.float64], num_of_vehicles: int
) -> npt.NDArray[np.float64]:
    """Extracts the 2x2 position covariance of every vehicle.

    :param covariance: (..., 4 * num_of_vehicles, 4 * num_of_vehicles)
    :return: (..., num_of_vehicles, 2, 2)
    """
    index = 4 * np.arange(num_of_vehicles)[:, np.newaxis] + np.arange(2)
    return covariance[..., index[:, :, np.newaxis], index[:, np.newaxis, :]]


def observation_rho(
    system: Vehicle2D, mean: VehicleCollection2D, covariance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Ratio of the relative position standard deviation (RMS over both axes) to the distance
    between observer and observed object, for every current observation.

    :return: (num_of_vehicles,)
    """
    num_of_vehicles = mean.position.shape[0]
    blocks = position_blocks(covariance, num_of_vehicles)
    rho = np.zeros(num_of_vehicles, dtype=np.float64)
    for vehicle, target in enumerate(system.observation_matrix[:, 0]):
        if target < num_of_vehicles:
            relative = mean.position[target] - mean.position[vehicle]
            relative_covariance = (
                blocks[vehicle]
                + blocks[target]
                - covariance[4 * vehicle : 4 * vehicle + 2, 4 * target : 4 * target + 2]
                - covariance[4 * target : 4 * target + 2, 4 * vehicle : 4 * vehicle + 2]
            )
        else:
            relative = system.landmarks[target - num_of_vehicles] - mean.position[vehicle]
            relative_covariance = blocks[vehicle]
        rho[vehicle] = np.sqrt(max(np.trace(relative_covariance), 0) / 2) / np.linalg.norm(relative)
    return rho


def run_episode(
    config: ScenarioConfig,
    num_of_vehicles: int,
    filter_keys: list[str],
    seed: np.random.SeedSequence,
    *,
    rho_reference: str | None = None,
    record_sigma_points: bool = False,
) -> dict[str, npt.NDArray]:
    """Simulates one episode and runs all filters on the same measurements.

    If a filter fails numerically, it is marked as diverged at that step and its entries are NaN
    from then on (which counts as not covered).

    :param rho_reference: Filter whose prior is used to compute ``rho``. Defaults to ``ukf_d`` if
        present, otherwise the first filter.
    :param record_sigma_points: Whether to store the prior sigma points of every UKF update (only
        sensible for single episodes, e.g. for the animation).
    :return: Flat dictionary of arrays. Global entries are ``true`` (steps, n), ``measurement``
        (steps, N), ``observation`` (steps, N) and ``rho`` (steps, N). Per filter ``key``, there
        are ``key/mean`` (steps, n), ``key/position_covariance`` (steps, N, 2, 2), ``key/nees``
        (steps,), ``key/position_nees`` (steps, N), ``key/max_sigma_residual`` (steps, N),
        ``key/diverged_at`` () and optionally ``key/sigma_points`` (steps, points, n).
    """
    if rho_reference is None:
        rho_reference = "ukf_d" if "ukf_d" in filter_keys else filter_keys[0]
    rng = np.random.default_rng(seed)
    num_of_steps = config.num_of_steps
    system = config.system(num_of_vehicles)
    n = system.tangent_dim
    initial_covariance = config.initial_covariance(num_of_vehicles)
    initial_mean = sample_initial_mean(config, num_of_vehicles, rng)
    runner = SystemRunner(system, initial_mean + tangent_sample(initial_covariance, rng), rng)

    estimators: dict[str, Filter] = {
        key: CONSTRUCTORS[key][0](system, initial_mean, initial_covariance) for key in filter_keys
    }

    result: dict[str, npt.NDArray] = {
        "true": np.zeros((num_of_steps, n), dtype=np.float64),
        "measurement": np.zeros((num_of_steps, num_of_vehicles), dtype=np.float64),
        "observation": np.zeros((num_of_steps, num_of_vehicles), dtype=np.int32),
        "rho": np.full((num_of_steps, num_of_vehicles), np.nan, dtype=np.float64),
    }
    for key in filter_keys:
        result[f"{key}/mean"] = np.full((num_of_steps, n), np.nan)
        result[f"{key}/position_covariance"] = np.full(
            (num_of_steps, num_of_vehicles, 2, 2), np.nan
        )
        result[f"{key}/nees"] = np.full(num_of_steps, np.nan)
        result[f"{key}/position_nees"] = np.full((num_of_steps, num_of_vehicles), np.nan)
        result[f"{key}/max_sigma_residual"] = np.full((num_of_steps, num_of_vehicles), np.nan)
        result[f"{key}/diverged_at"] = np.array(num_of_steps)
        if record_sigma_points and isinstance(estimators[key], UKF):
            result[f"{key}/sigma_points"] = np.full(
                (num_of_steps, len(estimators[key].sigma_points), n), np.nan
            )

    for step in range(num_of_steps):
        runner.step(config.dt, noisy=True)
        runner.trigger()
        measurement = runner.measure(noisy=True)
        truth = runner.state
        result["true"][step] = truth.numpy()
        result["measurement"][step] = measurement.numpy()
        result["observation"][step] = system.observation_matrix[:, 0]

        for key, estimator in estimators.items():
            if result[f"{key}/diverged_at"] < num_of_steps:
                continue
            try:
                with np.errstate(invalid="raise", over="raise"):
                    estimator.predict(config.dt)
                    if key == rho_reference:
                        result["rho"][step] = observation_rho(
                            system, estimator.mean, estimator.covariance
                        )
                    if f"{key}/sigma_points" in result:
                        result[f"{key}/sigma_points"][step] = np.stack(
                            [_.numpy() for _ in estimator.sigma_points]
                        )
                    estimator.update(measurement, system.sensor_covariance())
                    mean = estimator.mean.numpy()
                    covariance = estimator.covariance
                    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))):
                        raise FloatingPointError("non-finite estimate")
                    error = truth - estimator.mean
                    nees = error @ np.linalg.solve(covariance, error)
                    blocks = position_blocks(covariance, num_of_vehicles)
                    position_error = np.reshape(error, (num_of_vehicles, 4))[:, :2]
                    position_nees = np.einsum(
                        "vi,vi->v",
                        position_error,
                        np.linalg.solve(blocks, position_error[..., np.newaxis])[..., 0],
                    )
            except FILTER_ERRORS:
                result[f"{key}/diverged_at"] = np.array(step)
                continue
            result[f"{key}/mean"][step] = mean
            result[f"{key}/position_covariance"][step] = blocks
            result[f"{key}/nees"][step] = nees
            result[f"{key}/position_nees"][step] = position_nees
            residual = getattr(estimator, "last_max_sigma_residual", None)
            if residual is not None:
                result[f"{key}/max_sigma_residual"][step] = residual

    return result
