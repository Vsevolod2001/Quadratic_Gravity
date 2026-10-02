"""Ornstein–Uhlenbeck equation used as a reference model."""

import numpy as np

from .base import ItoEquation


class OrnsteinUhlenbeck(ItoEquation):
    """Scalar equation ``dX = -beta (X - alpha) dt + sigma dW``."""

    def __init__(self, alpha: float, beta: float, sigma: float) -> None:
        super().__init__(var_dim=1, noise_dim=1)
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        return np.array(-self.beta * (x - self.alpha), dtype=float)

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([[self.sigma]], dtype=float)


# Compatibility with the spelling used in the development notebook.
Ornstein_Ulenbek = OrnsteinUhlenbeck
