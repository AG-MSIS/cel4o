from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..systems import System


class Filter(ABC):
    def __init__(self, system: System):
        self._system = system

    @abstractmethod
    def predict(self, dt: float):
        raise NotImplementedError

    @abstractmethod
    def update(self, measurement, noise_covariance: npt.NDArray[np.float64]):
        raise NotImplementedError

    @property
    @abstractmethod
    def mean(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def covariance(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def sigma_points(self) -> list:
        raise NotImplementedError
