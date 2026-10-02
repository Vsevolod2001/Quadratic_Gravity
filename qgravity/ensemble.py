"""Generation of projected trajectory ensembles."""

from collections.abc import Iterator, Sequence

import numpy as np

from .solver import ItoSolver


class TrajectoryEnsemble:
    """Generate selected components of trajectories from a fixed solver.

    Component indices are a pair of zero-based indices ``(i, j)``. Every
    generated trajectory starts from the same initial condition, while the
    solver's random-number generator advances between trajectories.
    """

    def __init__(
        self,
        solver: ItoSolver,
        initial_cond: Sequence[float] | np.ndarray,
        component_indices: tuple[int, int],
    ) -> None:
        self.solver = solver
        self.initial_cond = np.asarray(initial_cond, dtype=float).copy()
        self.solver.initial_cond_check(self.initial_cond)

        if any(i < 0 or i >= self.solver.eq.var_dim for i in component_indices):
            raise IndexError("Component index is outside the state vector")

        self.component_indices = component_indices

    def generate_one(self) -> np.ndarray:
        """Return one projected trajectory.

        The returned shape is ``(N_steps + 1, number_of_components)``.
        """

        trajectory = self.solver.solve_one_process(self.initial_cond)
        return np.take(trajectory, self.component_indices, axis=1)

    def iter_trajectories(
        self,
        number_of_trajectories: int,
    ) -> Iterator[np.ndarray]:
        """Return an iterator that produces trajectories one by one."""

        return (self.generate_one() for _ in range(number_of_trajectories))

    
    def solve_many(self, number_of_trajectories: int) -> np.ndarray:
        
        """Generate and store a complete projected ensemble.

        The returned shape is
        ``(number_of_trajectories, N_steps + 1, number_of_components)``.
        """

        ensemble = np.empty(
            (
                number_of_trajectories,
                self.solver.N_steps + 1,
                len(self.component_indices),
            )
        )

        for trajectory_index, trajectory in enumerate(
            self.iter_trajectories(number_of_trajectories)
        ):
            ensemble[trajectory_index] = trajectory

        return ensemble
