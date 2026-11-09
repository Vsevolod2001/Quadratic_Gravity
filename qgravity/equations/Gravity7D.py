"""gravity7D with derivatives Itô system."""

import numpy as np

from .base import ItoEquation


class Gravity7D(ItoEquation):
    """
    The state is ordered as ``(A, alpha, G, beta, beta1, beta2)`` and is
    driven by one Wiener process.
    """

    def __init__(self, sigma: float, eps: float) -> None:
        super().__init__(var_dim=6, noise_dim=1)
        self.sigma = sigma
        self.eps = eps

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        A, alpha, G, beta, beta1, beta2 = x
        return np.array(
            [
                2.0 * beta * A,
                alpha * beta,
                alpha,
                beta1,
                beta2,
                (2.0 * beta**2 * beta1 
                 - 8.0 * beta * beta2 
                 - 4.0 * beta1**2.0 
                 + 24.0 * beta**4.0 
                 - self.eps * A * (3.0 * beta**2.0 + beta1) 
                 - (3.0 * A * self.eps**2.0) / 98.0),
            ],
            dtype=float,
        )

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array(
            [[0.0], [0.0], [0.0], [0.0], [0.0], [self.sigma]],
            dtype=float,
        )


# Compatibility with the class name used in the development notebook.
Gravity_7D = Gravity7D