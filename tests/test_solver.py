"""Tests for Euler–Maruyama and the trajectory solver."""

import unittest

import numpy as np

from qgravity import EulerMaruyama, ItoSolver, QuadraticGravity
from qgravity.equations.base import ItoEquation


class ConstantEquation(ItoEquation):
    def __init__(self, drift: float, diffusion: float) -> None:
        super().__init__(var_dim=1, noise_dim=1)
        self.drift = drift
        self.diffusion = diffusion

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([self.drift])

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([[self.diffusion]])


class TwoDimensionalEquation(ItoEquation):
    def __init__(self) -> None:
        super().__init__(var_dim=2, noise_dim=2)

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([1.0, -2.0])

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([[2.0, 0.0], [0.0, 3.0]])


class FixedRng:
    def standard_normal(self, size: int) -> np.ndarray:
        if size != 2:
            raise AssertionError(f"Expected two noises, got {size}")
        return np.array([0.5, -1.0])


class SolverTests(unittest.TestCase):
    def test_deterministic_euler_step(self) -> None:
        equation = ConstantEquation(drift=2.0, diffusion=0.0)
        scheme = EulerMaruyama()
        state = scheme.step(
            equation,
            dt=0.25,
            rng=np.random.default_rng(1),
            x=np.array([1.0]),
        )

        np.testing.assert_allclose(state, [1.5])

    def test_multidimensional_noise_step(self) -> None:
        state = EulerMaruyama().step(
            TwoDimensionalEquation(),
            dt=0.25,
            rng=FixedRng(),
            x=np.array([10.0, 20.0]),
        )

        np.testing.assert_allclose(state, [10.75, 18.0])

    def test_solver_grid_and_trajectory_shape(self) -> None:
        solver = ItoSolver(
            equation=QuadraticGravity(sigma=0.0, nu=1.0),
            scheme=EulerMaruyama(),
            time_step=0.1,
            total_time=0.3,
            rng=np.random.default_rng(1),
        )

        trajectory = solver.solve_one_process([1, 1, 0])

        np.testing.assert_allclose(solver.tau_grid, [0.0, 0.1, 0.2, 0.3])
        self.assertEqual(trajectory.shape, (4, 3))
        self.assertEqual(trajectory.dtype, np.dtype(float))

    def test_seeded_solvers_are_reproducible(self) -> None:
        equation = ConstantEquation(drift=0.0, diffusion=1.0)

        first = ItoSolver(
            equation,
            EulerMaruyama(),
            time_step=0.1,
            total_time=0.5,
            rng=np.random.default_rng(7),
        ).solve_one_process([0.0])
        second = ItoSolver(
            equation,
            EulerMaruyama(),
            time_step=0.1,
            total_time=0.5,
            rng=np.random.default_rng(7),
        ).solve_one_process([0.0])

        np.testing.assert_array_equal(first, second)

    def test_incorrect_initial_shape_is_rejected(self) -> None:
        solver = ItoSolver(
            equation=QuadraticGravity(sigma=0.0, nu=1.0),
            scheme=EulerMaruyama(),
            time_step=0.1,
            total_time=0.3,
            rng=np.random.default_rng(1),
        )

        with self.assertRaisesRegex(ValueError, "initial-condition shape"):
            solver.solve_one_process([1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
