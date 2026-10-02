"""Tests for the supplied Itô equations."""

import unittest

import numpy as np

from qgravity.equations import OrnsteinUhlenbeck, QuadraticGravity


class EquationTests(unittest.TestCase):
    def test_quadratic_gravity_coefficients(self) -> None:
        equation = QuadraticGravity(sigma=0.25, nu=2.0)
        x = np.array([2.0, 3.0, 5.0])

        np.testing.assert_allclose(
            equation.drift_vector(x),
            np.array([-14.0, 6.0, 3.0]),
        )
        np.testing.assert_allclose(
            equation.diffusion_matrix(x),
            np.array([[0.25], [0.0], [0.0]]),
        )

    def test_ornstein_uhlenbeck_coefficients(self) -> None:
        equation = OrnsteinUhlenbeck(alpha=1.5, beta=2.0, sigma=0.3)
        x = np.array([2.0])

        np.testing.assert_allclose(equation.drift_vector(x), [-1.0])
        np.testing.assert_allclose(equation.diffusion_matrix(x), [[0.3]])

    def test_coefficients_are_float_with_integer_inputs(self) -> None:
        cases = [
            (QuadraticGravity(sigma=1, nu=1), np.array([2, 3, 5])),
            (OrnsteinUhlenbeck(alpha=1, beta=2, sigma=1), np.array([2])),
        ]

        for equation, x in cases:
            with self.subTest(equation=type(equation).__name__):
                self.assertEqual(equation.drift_vector(x).dtype, np.dtype(float))
                self.assertEqual(
                    equation.diffusion_matrix(x).dtype,
                    np.dtype(float),
                )


if __name__ == "__main__":
    unittest.main()
