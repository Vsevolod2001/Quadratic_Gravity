"""Base class for autonomous Itô equations."""

from abc import ABC, abstractmethod

import numpy as np


class ItoEquation(ABC):
    """An autonomous Itô equation ``dX = a(X) dt + B(X) dW``."""

    def __init__(self, var_dim: int = 1, noise_dim: int = 1) -> None:
        self.var_dim = var_dim
        self.noise_dim = noise_dim

    @abstractmethod
    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        """Return the drift vector ``a(x)`` with shape ``(var_dim,)``."""

    @abstractmethod
    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        """Return the diffusion matrix ``B(x)``.

        Its shape must be ``(var_dim, noise_dim)``.
        """

    def shape_check(self, x: np.ndarray | None = None) -> None:
        """Validate coefficient shapes at a supplied state or at zero."""

        if x is None:
            x = np.zeros((self.var_dim,))

        drift = self.drift_vector(x)
        diffusion = self.diffusion_matrix(x)

        if drift.shape != (self.var_dim,):
            raise ValueError(
                "Incorrect drift shape: "
                f"expected {(self.var_dim,)}, got {drift.shape}"
            )

        expected_diffusion_shape = (self.var_dim, self.noise_dim)
        if diffusion.shape != expected_diffusion_shape:
            raise ValueError(
                "Incorrect diffusion shape: "
                f"expected {expected_diffusion_shape}, got {diffusion.shape}"
            )


# Compatibility with the class name used in the development notebook.
Ito_equation = ItoEquation
