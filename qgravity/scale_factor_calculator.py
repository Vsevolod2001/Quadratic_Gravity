"""Scale factors interpolated from an ensemble of Itô trajectories."""

import numpy as np


class Scale_factor_calculator:
    """Calculate scale-factor trajectories on a common time grid.

    The input has shape ``(n_samples, n_times, 2)`` with components ordered
    as ``(alpha, g)``.
    """

    def __init__(self, ito_trajectories: np.ndarray, n_points: int) -> None:
        self.alpha_trajs = ito_trajectories[:, :, 0]
        self.g_trajs = ito_trajectories[:, :, 1]
        self.n_samples = ito_trajectories.shape[0]
        self.n_points = n_points

    def calculate_t_grid(self) -> np.ndarray:
        """Store and return the common grid ending at the smallest final g."""

        t_max = np.min(self.g_trajs[:, -1])
        self.t_grid = np.linspace(0, t_max, self.n_points)
        return self.t_grid

    def calculate_a_trajs(self) -> np.ndarray:
        """Interpolate each trajectory onto the previously calculated grid."""

        self.a_trajs = np.empty((self.n_samples, self.n_points), dtype=float)

        for i in range(self.n_samples):
            self.a_trajs[i] = np.interp(
                self.t_grid,
                self.g_trajs[i],
                self.alpha_trajs[i],
            )

        return self.a_trajs

    def calculate_statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the grid, interpolated trajectories, mean and std.

        Return ``(t_grid, a_mean, a_err)``, each of shape ``(n_points,)``.
        ``a_err`` is the standard deviation across trajectories.
        """

        self.calculate_t_grid()
        self.calculate_a_trajs()

        a_mean = np.mean(self.a_trajs, axis=0)
        a_err = np.std(self.a_trajs, axis=0)
        return self.t_grid, a_mean, a_err
