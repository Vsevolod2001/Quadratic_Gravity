"""Tests for abstract interfaces and coefficient contracts."""

import unittest

import numpy as np

from qgravity import ItoEquation, Scheme


class IncorrectEquation(ItoEquation):
    def __init__(self) -> None:
        super().__init__(var_dim=2, noise_dim=1)

    def drift_vector(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.zeros((2, 1))

    def diffusion_matrix(self, x: np.ndarray) -> np.ndarray:
        del x
        return np.zeros((2, 1))


class InterfaceTests(unittest.TestCase):
    def test_abstract_classes_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            ItoEquation()

        with self.assertRaises(TypeError):
            Scheme()

    def test_incorrect_drift_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "drift shape"):
            IncorrectEquation().shape_check(np.zeros(2))


if __name__ == "__main__":
    unittest.main()
