"""Scenario definition for the multi-vehicle bearing-only tracking experiment.

All bearing measurements are invariant to a common scaling of positions, velocities and the
acceleration noise. Therefore, the geometry is fixed and only one dimensionless knob,
``noise_scale``, scales the bearing noise, the (square root of the) acceleration noise density and
the initial standard deviations together.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...systems.vehicles_2d import Vehicle2D, VehicleCollection2D


@dataclass(frozen=True)
class ScenarioConfig:
    noise_scale: float = (
        4.0  # [1] Chosen by calibrate.py (data/calibration-2026-11-02T11:31:09.json).
    )
    dt: float = 0.1  # [s]
    num_of_steps: int = 150  # [1]

    # Geometry: Vehicles start inside a disc and move with constant nominal velocity in random
    # directions. Landmarks are placed equidistantly on a ring around the start disc.
    start_radius: float = 10.0  # [m]
    min_separation: float = 2.0  # [m]
    min_speed: float = 0.3  # [m/s]
    max_speed: float = 1.0  # [m/s]
    num_of_landmarks: int = 6  # [1]
    landmark_radius: float = 15.0  # [m]
    landmark_observation_probability: float = 0.2  # [1]

    # Noise at noise_scale = 1.
    sensor_noise_stddev: float = 0.05  # [rad]
    acceleration_noise_stddev: float = 0.05  # [m/s^2 / sqrt(Hz)]
    initial_position_stddev: float = 0.2  # [m]
    initial_velocity_stddev: float = 0.05  # [m/s]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> ScenarioConfig:
        return cls(**values)

    def replace(self, **changes) -> ScenarioConfig:
        return dataclasses.replace(self, **changes)

    def landmarks(self) -> npt.NDArray[np.float64]:
        angles = 2 * np.pi * np.arange(self.num_of_landmarks) / self.num_of_landmarks
        return self.landmark_radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def system(self, num_of_vehicles: int) -> Vehicle2D:
        return Vehicle2D(
            num_of_vehicles=num_of_vehicles,
            landmarks=self.landmarks(),
            num_of_observations=1,
            landmark_observation_probability=self.landmark_observation_probability,
            dynamic_noise_stddev=self.noise_scale * self.acceleration_noise_stddev,
            sensor_noise_stddev=self.noise_scale * self.sensor_noise_stddev,
        )

    def initial_covariance(self, num_of_vehicles: int) -> npt.NDArray[np.float64]:
        block = np.square(
            self.noise_scale
            * np.array(
                [
                    self.initial_position_stddev,
                    self.initial_position_stddev,
                    self.initial_velocity_stddev,
                    self.initial_velocity_stddev,
                ],
                dtype=np.float64,
            )
        )
        return np.diag(np.tile(block, num_of_vehicles))


def sample_initial_mean(
    config: ScenarioConfig, num_of_vehicles: int, rng: np.random.Generator
) -> VehicleCollection2D:
    """Draws well-separated start positions in a disc and random nominal velocities."""
    for _ in range(1000):
        positions = np.zeros((num_of_vehicles, 2), dtype=np.float64)
        count = 0
        for _ in range(1000 * num_of_vehicles):
            if count == num_of_vehicles:
                break
            radius = config.start_radius * np.sqrt(rng.uniform())
            angle = rng.uniform(-np.pi, np.pi)
            candidate = radius * np.array([np.cos(angle), np.sin(angle)])
            if np.all(
                np.linalg.norm(positions[:count] - candidate, axis=-1) >= config.min_separation
            ):
                positions[count] = candidate
                count += 1
        if count == num_of_vehicles:
            break
    else:
        raise RuntimeError("Could not place vehicles with the requested minimum separation.")

    headings = rng.uniform(-np.pi, np.pi, size=num_of_vehicles)
    speeds = rng.uniform(config.min_speed, config.max_speed, size=num_of_vehicles)
    velocities = speeds[:, np.newaxis] * np.stack([np.cos(headings), np.sin(headings)], axis=-1)
    return VehicleCollection2D(positions, velocities)
