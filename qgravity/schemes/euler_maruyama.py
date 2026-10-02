"""Euler–Maruyama scheme."""

import numpy as np

from ..equations.base import ItoEquation
from .base import Scheme


class EulerMaruyama(Scheme):
    """Explicit Euler–Maruyama method."""

    def step(
        self,
        equation: ItoEquation,
        dt: float,
        rng: np.random.Generator,
        x: np.ndarray,
    ) -> np.ndarray:
        d_w = np.sqrt(dt) * rng.standard_normal(size=equation.noise_dim)
        drift = equation.drift_vector(x) * dt
        noise = equation.diffusion_matrix(x) @ d_w
        return x + drift + noise


# Short name retained for convenient use and notebook compatibility.
Euler = EulerMaruyama
