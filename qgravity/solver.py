"""Trajectory solver for autonomous Itô equations."""

import numpy as np

from .equations.base import ItoEquation
from .schemes.base import Scheme


class ItoSolver:
    """Generate one trajectory on a uniform time grid."""

    def __init__(
        self,
        equation: ItoEquation,
        scheme: Scheme,
        time_step: float,
        total_time: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.eq = equation
        self.scheme = scheme
        self.total_time = total_time
        self.time_step = time_step

        self.N_steps = round(total_time / time_step)
        self.tau_grid = time_step * np.arange(self.N_steps + 1)
        self.rng = np.random.default_rng() if rng is None else rng

        self.eq.shape_check()

    def solve_one_process(self, initial_cond: np.ndarray) -> np.ndarray:
        """Return one trajectory with shape ``(N_steps + 1, var_dim)``."""

        x = np.asarray(initial_cond, dtype=float).copy()
        self.initial_cond_check(x)

        trajectory = np.empty((self.N_steps + 1, self.eq.var_dim))
        trajectory[0] = x

        for step_index in range(1, self.N_steps + 1):
            x = self.scheme.step(self.eq, self.time_step, self.rng, x)
            trajectory[step_index] = x

        return trajectory

    def initial_cond_check(self, x: np.ndarray) -> None:
        """Validate the shape of an initial state."""

        if x.shape != (self.eq.var_dim,):
            raise ValueError(
                "Incorrect initial-condition shape: "
                f"expected {(self.eq.var_dim,)}, got {x.shape}"
            )


# Compatibility with the class name used in the development notebook.
Ito_solver = ItoSolver
