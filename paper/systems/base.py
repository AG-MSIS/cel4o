from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

# Fallback generator for callers that don't pass their own. Experiments should always pass an
# explicitly seeded generator to make runs reproducible.
GENERATOR = np.random.Generator(np.random.PCG64())


class System(ABC):
    def __init__(self, state_dim, tangent_dim, measurement_dim):
        self.state_dim = state_dim
        self.tangent_dim = tangent_dim
        self.measurement_dim = measurement_dim

    @abstractmethod
    def dynamics(self, state, dt: float):
        raise NotImplementedError

    @abstractmethod
    def dynamics_covariance(self, dt: float) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def sensor(self, state):
        raise NotImplementedError

    @abstractmethod
    def sensor_covariance(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def dynamics_jacobian(self, state, dt: float) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def sensor_jacobian(self, state) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def trigger(self, rng: np.random.Generator = GENERATOR):
        pass

    def step(self, state, dt: float, noisy: bool = False, rng: np.random.Generator = GENERATOR):
        state = self.dynamics(state, dt)
        if noisy:
            state = state + tangent_sample(self.dynamics_covariance(dt), rng)
        return state


class SystemRunner:
    def __init__(self, system: System, state, rng: np.random.Generator = GENERATOR):
        self._system = system
        self._state = state
        self._rng = rng

    def step(self, dt: float, noisy: bool = False):
        self._state = self._system.step(self._state, dt, noisy=noisy, rng=self._rng)

    def trigger(self):
        self._system.trigger(self._rng)

    def measure(self, noisy: bool = False):
        z = self._system.sensor(self._state)
        if noisy:
            z += tangent_sample(self._system.sensor_covariance(), self._rng)
        return z

    @property
    def state(self):
        return self._state


def tangent_sample(
    covariance: npt.NDArray[np.float64], rng: np.random.Generator = GENERATOR
) -> npt.NDArray[np.float64]:
    return rng.multivariate_normal(np.zeros_like(covariance[0, :]), covariance, method="eigh")
