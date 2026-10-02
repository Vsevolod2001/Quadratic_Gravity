"""Tests for projected trajectory generation."""

import unittest

import numpy as np

from qgravity import EulerMaruyama, ItoEquation, ItoSolver, TrajectoryEnsemble


class ThreeComponentEquation(ItoEquation):
    def __init__(self, diffusion: float = 0.0) -> None:
        super().__init__(var_dim=3, noise_dim=1)
        self.diffusion = diffusion

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([1.0, 2.0, 3.0])

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.array([[self.diffusion], [0.0], [0.0]])


def make_ensemble(seed: int, diffusion: float = 0.0) -> TrajectoryEnsemble:
    solver = ItoSolver(
        equation=ThreeComponentEquation(diffusion),
        scheme=EulerMaruyama(),
        time_step=0.5,
        total_time=1.0,
        rng=np.random.default_rng(seed),
    )
    return TrajectoryEnsemble(
        solver=solver,
        initial_cond=[10.0, 20.0, 30.0],
        component_indices=(2, 0),
    )


class TrajectoryEnsembleTests(unittest.TestCase):
    def test_selected_components_and_shapes(self) -> None:
        ensemble_generator = make_ensemble(seed=1)

        trajectory = ensemble_generator.generate_one()
        ensemble = ensemble_generator.solve_many(2)

        np.testing.assert_allclose(
            trajectory,
            [[30.0, 10.0], [31.5, 10.5], [33.0, 11.0]],
        )
        self.assertEqual(trajectory.shape, (3, 2))
        self.assertEqual(ensemble.shape, (2, 3, 2))

    def test_iterator_returns_requested_number(self) -> None:
        trajectories = list(make_ensemble(seed=2).iter_trajectories(3))
        self.assertEqual(len(trajectories), 3)
        self.assertTrue(all(item.shape == (3, 2) for item in trajectories))

    def test_initial_condition_is_copied(self) -> None:
        initial_cond = np.array([10.0, 20.0, 30.0])
        solver = ItoSolver(
            ThreeComponentEquation(),
            EulerMaruyama(),
            time_step=0.5,
            total_time=1.0,
            rng=np.random.default_rng(3),
        )
        ensemble_generator = TrajectoryEnsemble(solver, initial_cond, (0, 1))
        initial_cond[:] = -1.0

        np.testing.assert_allclose(
            ensemble_generator.generate_one()[0],
            [10.0, 20.0],
        )

    def test_seeded_ensembles_are_reproducible(self) -> None:
        first = make_ensemble(seed=4, diffusion=1.0).solve_many(3)
        second = make_ensemble(seed=4, diffusion=1.0).solve_many(3)
        np.testing.assert_array_equal(first, second)

    def test_invalid_component_index_is_rejected(self) -> None:
        with self.assertRaises(IndexError):
            TrajectoryEnsemble(
                make_ensemble(seed=5).solver,
                [10.0, 20.0, 30.0],
                (0, 3),
            )

    def test_zero_trajectories_has_well_defined_shape(self) -> None:
        ensemble = make_ensemble(seed=6).solve_many(0)
        self.assertEqual(ensemble.shape, (0, 3, 2))


if __name__ == "__main__":
    unittest.main()
