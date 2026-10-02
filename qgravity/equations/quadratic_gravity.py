"""Quadratic-gravity Itô system."""

import numpy as np

from .base import ItoEquation


class QuadraticGravity(ItoEquation):

    """
    The state is ordered as ``(beta, alpha, g)`` and is driven by one
    Wiener process.
    """

    def __init__(self, sigma: float, nu: float) -> None:
        super().__init__(var_dim=3, noise_dim=1)
        self.sigma = sigma
        self.nu = nu

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        beta, alpha, _ = x
        return np.array(
            [
                beta**2 - self.nu * alpha**2,
                beta * alpha,
                alpha,
            ],
            dtype=float,
        )

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([[self.sigma], [0.0], [0.0]], dtype=float)


# Compatibility with the class name used in the development notebook.
Quadratic_gravity = QuadraticGravity
