from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


@dataclass
class OptimizerConfig:
    max_iters: int = 2000
    neighbor_samples: int = 60
    random_seed: int = 7
    maximize: bool = True  # True: maximize objective, False: minimize objective


class HillClimbingOptimizer:
    """
    Hill Climbing over permutations:
    - Start from an initial ordering.
    - Sample neighbors via random swaps.
    - Move to best improving neighbor.
    """

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def optimize(
        self,
        initial_order: Sequence[int],
        objective: Callable[[Sequence[int]], float],
    ) -> Tuple[List[int], float]:
        order = list(initial_order)
        best_val = float(objective(order))

        n = len(order)
        if n <= 2:
            return order, best_val

        for _ in range(self.config.max_iters):
            improved = False
            candidate_best_order = order
            candidate_best_val = best_val

            for _k in range(self.config.neighbor_samples):
                i, j = self._rng.integers(0, n, size=2)
                if i == j:
                    continue
                neigh = order.copy()
                neigh[i], neigh[j] = neigh[j], neigh[i]
                val = float(objective(neigh))
                better = val > candidate_best_val if self.config.maximize else val < candidate_best_val
                if better:
                    candidate_best_val = val
                    candidate_best_order = neigh
                    improved = True

            if not improved:
                break
            order = candidate_best_order
            best_val = candidate_best_val

        return order, best_val


@dataclass
class GeneticConfig:
    population_size: int = 80
    generations: int = 250
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    elite_frac: float = 0.15
    random_seed: int = 13
    maximize: bool = False  # for waiting-time objectives, default minimize


class GeneticAlgorithmOptimizer:
    """
    Permutation GA (order-based):
    - Fitness = objective(order)
    - Supports maximize/minimize
    """

    def __init__(self, config: GeneticConfig | None = None):
        self.config = config or GeneticConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self._rng.integers(0, n, size=2).tolist())
        child = [-1] * n
        child[a:b] = p1[a:b]
        fill = [g for g in p2 if g not in child]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child

    def _mutate_swap(self, order: List[int]) -> None:
        n = len(order)
        i, j = self._rng.integers(0, n, size=2)
        if i != j:
            order[i], order[j] = order[j], order[i]

    def optimize(
        self,
        initial_order: Sequence[int],
        objective: Callable[[Sequence[int]], float],
    ) -> Tuple[List[int], float]:
        n = len(initial_order)
        if n <= 2:
            o = list(initial_order)
            return o, float(objective(o))

        pop: List[List[int]] = []
        base = list(initial_order)
        pop.append(base)
        for _ in range(self.config.population_size - 1):
            x = base.copy()
            self._rng.shuffle(x)
            pop.append(x)

        def is_better(a: float, b: float) -> bool:
            return a > b if self.config.maximize else a < b

        best = pop[0]
        best_val = float(objective(best))

        elite_n = max(1, int(self.config.elite_frac * self.config.population_size))

        for _gen in range(self.config.generations):
            vals = np.array([float(objective(ind)) for ind in pop], dtype=float)
            # Track global best
            idx_best = int(np.argmax(vals) if self.config.maximize else np.argmin(vals))
            if is_better(float(vals[idx_best]), best_val):
                best = pop[idx_best].copy()
                best_val = float(vals[idx_best])

            # Elitism
            elite_idx = np.argsort(-vals)[:elite_n] if self.config.maximize else np.argsort(vals)[:elite_n]
            new_pop = [pop[int(i)].copy() for i in elite_idx]

            # Tournament selection
            while len(new_pop) < self.config.population_size:
                a, b = self._rng.integers(0, len(pop), size=2)
                p1 = pop[int(a)] if is_better(float(vals[int(a)]), float(vals[int(b)])) else pop[int(b)]
                c, d = self._rng.integers(0, len(pop), size=2)
                p2 = pop[int(c)] if is_better(float(vals[int(c)]), float(vals[int(d)])) else pop[int(d)]

                child = p1.copy()
                if self._rng.random() < self.config.crossover_rate:
                    child = self._order_crossover(p1, p2)
                if self._rng.random() < self.config.mutation_rate:
                    self._mutate_swap(child)
                new_pop.append(child)

            pop = new_pop

        return best, best_val


def total_waiting_time(
    order: Sequence[int],
    *,
    eta_minutes: Sequence[float],
    separation_min: float,
) -> float:
    """
    Computes total waiting time given an ordering.
    eta_minutes: desired event times (e.g., ETA to runway) for each index.
    separation_min: fixed spacing between consecutive operations.
    Waiting = sum(max(0, scheduled - eta)).
    """
    eta = np.asarray(eta_minutes, dtype=float)
    t = 0.0
    total = 0.0
    for idx in order:
        desired = float(eta[int(idx)])
        total += max(0.0, t - desired)
        t += float(separation_min)
    return float(total)
