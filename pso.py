"""
FlowCast v2 — Particle Swarm Optimization
Metaheuristic optimizer for ML hyperparameter tuning (Paper 2: Eatesam et al.)
"""
import numpy as np
from typing import Callable, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float = -np.inf


class PSOOptimizer:
    """
    Particle Swarm Optimization for hyperparameter tuning.
    Minimizes the objective function (lower is better).
    """

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

    def optimize(
        self,
        objective_fn: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        param_types: Dict[str, str] = None,
    ) -> Tuple[Dict, float, List[float]]:
        """
        Run PSO optimization.

        Args:
            objective_fn: function(params_dict) -> score (higher is better)
            param_bounds: {name: (lower, upper)}
            param_types: {name: "int"|"float"}, defaults to "float"

        Returns:
            best_params, best_score, convergence_history
        """
        param_types = param_types or {}
        names = list(param_bounds.keys())
        dim = len(names)
        lower = np.array([param_bounds[n][0] for n in names])
        upper = np.array([param_bounds[n][1] for n in names])

        # Initialize swarm
        particles = []
        for _ in range(self.n_particles):
            pos = self.rng.uniform(lower, upper)
            vel = self.rng.uniform(-(upper - lower) * 0.1, (upper - lower) * 0.1)
            particles.append(Particle(
                position=pos, velocity=vel,
                best_position=pos.copy(), best_score=-np.inf
            ))

        global_best_pos = particles[0].position.copy()
        global_best_score = -np.inf
        history = []

        for iteration in range(self.n_iterations):
            for p in particles:
                # Decode parameters
                params = self._decode(p.position, names, lower, upper, param_types)

                # Evaluate
                try:
                    score = objective_fn(params)
                except Exception as e:
                    score = -np.inf

                # Update personal best
                if score > p.best_score:
                    p.best_score = score
                    p.best_position = p.position.copy()

                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = p.position.copy()

            # Update velocities and positions
            w = self.w * (self.w_decay ** iteration)
            for p in particles:
                r1 = self.rng.rand(dim)
                r2 = self.rng.rand(dim)
                p.velocity = (
                    w * p.velocity
                    + self.c1 * r1 * (p.best_position - p.position)
                    + self.c2 * r2 * (global_best_pos - p.position)
                )
                p.position = np.clip(p.position + p.velocity, lower, upper)

            history.append(global_best_score)
            if self.verbose and (iteration + 1) % 10 == 0:
                best_params = self._decode(global_best_pos, names, lower, upper, param_types)
                print(f"  PSO iter {iteration+1}/{self.n_iterations}: "
                      f"best_score={global_best_score:.4f}")

        best_params = self._decode(global_best_pos, names, lower, upper, param_types)
        return best_params, global_best_score, history

    def _decode(self, position, names, lower, upper, param_types):
        params = {}
        for i, name in enumerate(names):
            val = np.clip(position[i], lower[i], upper[i])
            if param_types.get(name) == "int":
                val = int(round(val))
            params[name] = val
        return params