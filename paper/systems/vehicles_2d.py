from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from .base import GENERATOR, System

if TYPE_CHECKING:
    from collections.abc import Iterable


class VehicleCollection2D:
    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64]):
        """
        :param position: A batch of 2D positions (n, 2).
        :param velocity: A batch of 2D velocities (n, 2).
        """
        self.position = position
        self.velocity = velocity

    def __add__(self, delta: npt.NDArray[np.float64]) -> VehicleCollection2D:
        stacked = np.reshape(delta, (-1, 4))
        return VehicleCollection2D(self.position + stacked[:, :2], self.velocity + stacked[:, 2:])

    def __sub__(self, other: VehicleCollection2D) -> npt.NDArray[np.float64]:
        return np.reshape(
            np.concatenate(
                [self.position - other.position, self.velocity - other.velocity],
                axis=1,
            ),
            (-1,),
        )

    def __repr__(self):
        return f"position: {self.position}, velocity: {self.velocity}"

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Self:
        stacked = np.reshape(array, (-1, 4))
        return cls(stacked[:, :2], stacked[:, 2:])

    def numpy(self) -> npt.NDArray[np.float64]:
        return np.reshape(np.concatenate([self.position, self.velocity], axis=1), (-1,))

    @classmethod
    def mean(
        cls, states: Iterable[VehicleCollection2D], *, weights: npt.NDArray[np.float64]
    ) -> Self:
        mean = np.sum(np.stack([_.numpy() for _ in states], axis=-1) * weights, axis=-1)
        return cls.from_numpy(mean)


class AngleBatch:
    def __init__(self, angles: npt.NDArray[np.float64]):
        self.angles = angles

    def __add__(self, delta: npt.NDArray[np.float64]) -> AngleBatch:
        return AngleBatch(AngleBatch.normalize(self.angles + delta))

    def __sub__(self, other: AngleBatch) -> npt.NDArray[np.float64]:
        return AngleBatch.normalize(self.angles - other.angles)

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Self:
        return cls(array)

    def numpy(self) -> npt.NDArray[np.float64]:
        return self.angles

    @classmethod
    def mean(cls, states: Iterable[AngleBatch], *, weights: npt.NDArray[np.float64]) -> Self:
        as_np = np.stack([_.numpy() for _ in states], axis=-1)
        c = np.cos(as_np) * weights
        s = np.sin(as_np) * weights
        return cls.from_numpy(np.arctan2(np.sum(s, axis=-1), np.sum(c, axis=-1)))

    @staticmethod
    def normalize(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.arctan2(np.sin(x), np.cos(x))


class Vehicle2D(System):
    def __init__(
        self,
        num_of_vehicles: int,
        landmarks: npt.ArrayLike,
        num_of_observations: int = 1,
        landmark_observation_probability: float = 0.1,
        dynamic_noise_stddev: float = 1.0,
        sensor_noise_stddev: float = 1.0,
    ):
        """Constructor.

        :param num_of_vehicles: The overall number of vehicles.
        :param landmarks: Positions of the landmarks that can be observed by all vehicles.
            (num_of_landmarks, 2)
        :param num_of_observations: The number observations per vehicle.
        :param landmark_observation_probability: The probability to observe a landmark instead of
            another vehicle.
        :param dynamic_noise_stddev: The velocity noise standard deviation per second
            [L/T / sqrt(T)].
        :param sensor_noise_stddev: The bearing noise standard deviation [rad].
        """
        super().__init__(
            4 * num_of_vehicles,
            4 * num_of_vehicles,
            num_of_vehicles * num_of_observations,
        )

        self._landmarks = np.asarray(landmarks, dtype=np.float64)
        assert self._landmarks.ndim == 2
        assert self._landmarks.shape[1] == 2

        self._num_of_vehicles = num_of_vehicles
        self._num_of_observations = num_of_observations
        self._landmark_observation_probability = landmark_observation_probability

        self._dynamic_noise_stddev = dynamic_noise_stddev
        self._sensor_noise_stddev = sensor_noise_stddev

        self._observation_matrix = np.zeros(
            (self._num_of_vehicles, self._num_of_observations), dtype=np.int32
        )  # (num_of_vehicles, num_of_observations)

    @property
    def landmarks(self) -> npt.NDArray[np.float64]:
        return self._landmarks

    @property
    def observation_matrix(self) -> npt.NDArray[np.int32]:
        """Index of the observed object per vehicle and observation. Indices below
        ``num_of_vehicles`` refer to vehicles, the others to ``landmarks[index - num_of_vehicles]``.
        (num_of_vehicles, num_of_observations)
        """
        return self._observation_matrix

    @observation_matrix.setter
    def observation_matrix(self, value: npt.NDArray[np.int32]):
        assert value.shape == (self._num_of_vehicles, self._num_of_observations)
        self._observation_matrix = value

    def dynamics(self, state: VehicleCollection2D, dt: float) -> VehicleCollection2D:
        return VehicleCollection2D(state.position + state.velocity * dt, state.velocity)

    def dynamics_covariance(self, dt) -> npt.NDArray[np.float64]:
        # Continuous time white acceleration noise
        c = self._dynamic_noise_stddev**2
        block = c * np.array(
            [
                [dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**3 / 3, 0, dt**2 / 2],
                [dt**2 / 2, 0, dt, 0],
                [0, dt**2 / 2, 0, dt],
            ],
            dtype=np.float64,
        )
        return cast(
            "npt.NDArray[np.float64]",
            scipy.linalg.block_diag(*(block for _ in range(self._num_of_vehicles))),
        )

    def dynamics_jacobian(self, state: VehicleCollection2D, dt: float) -> npt.NDArray[np.float64]:
        block = np.eye(4, dtype=np.float64)
        block[0, 2] = dt
        block[1, 3] = dt
        return cast(
            "npt.NDArray[np.float64]",
            scipy.linalg.block_diag(*(block for _ in range(self._num_of_vehicles))),
        )

    def trigger(self, rng: np.random.Generator = GENERATOR):
        chosen_vehicle = rng.integers(
            0, self._num_of_vehicles - 1, size=(self._num_of_vehicles, self._num_of_observations)
        )
        chosen_vehicle += (
            chosen_vehicle >= np.arange(self._num_of_vehicles, dtype=int)[:, np.newaxis]
        )

        chosen_landmark = rng.integers(
            self._num_of_vehicles,
            self._num_of_vehicles + len(self._landmarks),
            size=(self._num_of_vehicles, self._num_of_observations),
        )
        use_landmark = (
            rng.uniform(size=(self._num_of_vehicles, self._num_of_observations))
            < self._landmark_observation_probability
        )
        self._observation_matrix = np.where(
            use_landmark, chosen_landmark, chosen_vehicle
        )  # (num_of_vehicles, num_of_observations)

    def _relative_observations(self, state: VehicleCollection2D) -> npt.NDArray[np.float64]:
        observables = np.concatenate(
            [state.position, self._landmarks], axis=0
        )  # (num_of_vehicles + num_of_landmarks, 2)
        return (
            observables[self._observation_matrix, :] - state.position[:, np.newaxis, :]
        )  # (num_of_vehicles, num_of_observations, 2)

    def sensor(self, state: VehicleCollection2D) -> AngleBatch:
        relative_observations = self._relative_observations(state)
        all_bearings = np.atan2(
            relative_observations[..., 1], relative_observations[..., 0]
        )  # (num_of_vehicles, num_of_observations)
        return AngleBatch(np.reshape(all_bearings, (-1,)))

    def sensor_jacobian(self, state: VehicleCollection2D) -> npt.NDArray[np.float64]:
        relative_observations = self._relative_observations(state)
        # d atan2(y, x) / d(x, y) = (-y, x) / (x^2 + y^2)
        gradient = np.stack(
            [-relative_observations[..., 1], relative_observations[..., 0]], axis=-1
        ) / np.sum(np.square(relative_observations), axis=-1, keepdims=True)
        jacobian = np.zeros((self.measurement_dim, self.tangent_dim), dtype=np.float64)
        for vehicle in range(self._num_of_vehicles):
            for observation in range(self._num_of_observations):
                row = vehicle * self._num_of_observations + observation
                target = self._observation_matrix[vehicle, observation]
                jacobian[row, 4 * vehicle : 4 * vehicle + 2] -= gradient[vehicle, observation]
                if target < self._num_of_vehicles:
                    jacobian[row, 4 * target : 4 * target + 2] += gradient[vehicle, observation]
        return jacobian

    def sensor_covariance(self) -> npt.NDArray[np.float64]:
        return np.diag(
            np.array([self._sensor_noise_stddev**2] * self.measurement_dim, dtype=np.float64)
        )
