"""Tests for interpolation and statistics of scale factors."""

import unittest

import numpy as np

from qgravity import Scale_factor_calculator


class ScaleFactorCalculatorTests(unittest.TestCase):
    def test_interpolation_on_different_g_grids(self) -> None:
        # The first trajectory has alpha = 1 + g, the second alpha = 1 + 2g.
        trajectories = np.array(
            [
                [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
                [[1.0, 0.0], [4.0, 1.5], [7.0, 3.0]],
            ]
        )
        calculator = Scale_factor_calculator(trajectories, n_points=5)

        t_grid = calculator.calculate_t_grid()
        a_trajs = calculator.calculate_a_trajs()

        self.assertEqual(calculator.n_samples, 2)
        np.testing.assert_allclose(t_grid, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(
            a_trajs,
            [[1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        )
        self.assertIs(t_grid, calculator.t_grid)
        self.assertIs(a_trajs, calculator.a_trajs)

    def test_statistics_are_calculated_across_samples(self) -> None:
        trajectories = np.array(
            [
                [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
                [[1.0, 0.0], [4.0, 1.5], [7.0, 3.0]],
            ]
        )
        calculator = Scale_factor_calculator(trajectories, n_points=5)

        t_grid, a_mean, a_err = calculator.calculate_statistics()

        for array in (t_grid, a_mean, a_err):
            self.assertEqual(array.shape, (5,))
        np.testing.assert_allclose(a_mean, [1.0, 1.75, 2.5, 3.25, 4.0])
        np.testing.assert_allclose(a_err, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_single_sample_has_zero_standard_deviation(self) -> None:
        trajectories = np.array([[[2.0, 0.0], [4.0, 1.0], [2.0, 2.0]]])
        calculator = Scale_factor_calculator(trajectories, n_points=5)

        t_grid, a_mean, a_err = calculator.calculate_statistics()

        np.testing.assert_allclose(t_grid, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(a_mean, [2.0, 3.0, 4.0, 3.0, 2.0])
        np.testing.assert_array_equal(a_err, np.zeros(5))


if __name__ == "__main__":
    unittest.main()
