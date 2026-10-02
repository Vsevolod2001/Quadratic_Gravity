"""Base class for numerical schemes."""

from abc import ABC, abstractmethod

import numpy as np

from ..equations.base import ItoEquation


class Scheme(ABC):
    """Interface for one-step schemes for Itô equations."""

    @abstractmethod
    def step(
        self,
        equation: ItoEquation,
        dt: float,
        rng: np.random.Generator,
        x: np.ndarray,
    ) -> np.ndarray:
        """Advance the state by one time step."""
