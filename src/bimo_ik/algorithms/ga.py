"""Genetic Algorithm with explicit memory modes (cold / injection).

Operators transcribe the canonical GA from reference/main.py: elitism
(top-2), roulette selection proportional to inverse fitness, arithmetic
crossover (rate 0.7), per-gene Gaussian mutation (rate 0.2, sigma 0.1 rad),
and renewal of the worst 20% non-elite individuals each generation.
Differences from the reference are policy, not operators: injected
``numpy.random.Generator``, optional injection seeding of the initial
population, and (fitness, pose_error) tracking.

There is deliberately NO persistence mode for the GA: carrying a swarm's
velocities and personal bests across points has no principled GA analog, and
fabricating one would blur the factor the study isolates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    CountingEvaluator,
    Evaluate,
    MemoryMode,
    PointResult,
    injected_population,
)


@dataclass(frozen=True)
class GAParams:
    pop: int = 40
    n_elite: int = 2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.1
    renewal_fraction: float = 0.2


class GA:
    def __init__(
        self,
        params: GAParams,
        memory: MemoryMode,
        lower: np.ndarray,
        upper: np.ndarray,
        injection_fraction: float = 0.25,
        injection_sigma: float = 0.1,
    ):
        memory = MemoryMode(memory)
        if memory is MemoryMode.PERSISTENCE:
            raise ValueError("persistence is PSO-exclusive by design; GA supports cold/injection")
        self.params = params
        self.memory = memory
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.injection_fraction = injection_fraction
        self.injection_sigma = injection_sigma
        self.last_initial_population: np.ndarray | None = None  # inspection hook

    def reset(self) -> None:
        """GA keeps no cross-point state; provided for interface symmetry."""
        self.last_initial_population = None

    def solve_point(
        self,
        evaluate: Evaluate,
        n_iter: int,
        rng: np.random.Generator,
        prev_best: np.ndarray | None = None,
    ) -> PointResult:
        ev = CountingEvaluator(evaluate)
        p = self.params
        pop_size = p.pop
        dim = len(self.lower)

        population = injected_population(
            pop_size, self.lower, self.upper, rng, self.memory, prev_best,
            self.injection_fraction, self.injection_sigma,
        )
        self.last_initial_population = population.copy()

        scores = np.empty(pop_size)
        pose_errors = np.empty(pop_size)
        for i in range(pop_size):
            scores[i], pose_errors[i] = ev(population[i])

        fitness_hist = np.empty(n_iter)
        pe_hist = np.empty(n_iter)
        div_hist = np.empty(n_iter)
        ev_hist = np.empty(n_iter, dtype=np.int64)

        n_children = pop_size - p.n_elite
        for gen in range(n_iter):
            elite_idx = np.argsort(scores)[: p.n_elite]
            elite = population[elite_idx].copy()

            # roulette selection, proportional to inverse fitness
            inv = 1.0 / (scores + 1e-12)
            prob = inv / inv.sum()
            parents = population[rng.choice(pop_size, size=pop_size, replace=True, p=prob)]

            children: list[np.ndarray] = []
            for j in range(0, n_children, 2):
                p1, p2 = parents[j], parents[j + 1]
                if rng.random() < p.crossover_rate:
                    a = rng.random()
                    c1 = a * p1 + (1 - a) * p2
                    c2 = (1 - a) * p1 + a * p2
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    mask = rng.random(dim) < p.mutation_rate
                    c[mask] += rng.normal(0.0, p.mutation_sigma, int(mask.sum()))
                children.append(np.clip(c1, self.lower, self.upper))
                children.append(np.clip(c2, self.lower, self.upper))
            while len(children) < n_children:
                extra = parents[-1].copy()
                mask = rng.random(dim) < p.mutation_rate
                extra[mask] += rng.normal(0.0, p.mutation_sigma, int(mask.sum()))
                children.append(np.clip(extra, self.lower, self.upper))

            population = np.vstack([elite, np.array(children[:n_children])])

            # renewal: replace the worst non-elite (scored on this landscape)
            n_renew = max(1, int(p.renewal_fraction * n_children))
            scores_ne = np.array([ev(ind)[0] for ind in population[p.n_elite:]])
            worst = np.argsort(scores_ne)[-n_renew:]
            population[p.n_elite + worst] = rng.uniform(
                self.lower, self.upper, (n_renew, dim)
            )

            for i in range(pop_size):
                scores[i], pose_errors[i] = ev(population[i])

            best = int(np.argmin(scores))
            fitness_hist[gen] = scores[best]
            pe_hist[gen] = pose_errors[best]
            div_hist[gen] = float(np.std(population))
            ev_hist[gen] = ev.calls

        best = int(np.argmin(scores))
        return PointResult(
            best_q=population[best].copy(),
            best_fitness=float(scores[best]),
            best_pose_error=float(pose_errors[best]),
            fitness_history=fitness_hist,
            pose_error_history=pe_hist,
            diversity_history=div_hist,
            evals_history=ev_hist,
            n_evaluations=ev.calls,
        )
