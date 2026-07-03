"""Shared optimizer machinery: memory modes, per-point results, eval counting.

The initialization/memory policy is an explicit experimental factor:

- ``cold``        : fresh uniform-random population at every trajectory point
                    (the prior study's PSO/GA behavior).
- ``injection``   : fresh population, but a fixed fraction is seeded at the
                    previous point's best solution plus Gaussian jitter.
- ``persistence`` : one population instance carries its full state across all
                    points (PSO-only by design; there is no principled GA
                    analog of carrying velocities/personal bests).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

Evaluate = Callable[[np.ndarray], tuple[float, float]]


class MemoryMode(str, Enum):
    COLD = "cold"
    INJECTION = "injection"
    PERSISTENCE = "persistence"


@dataclass
class PointResult:
    """Outcome of optimizing a single trajectory point."""

    best_q: np.ndarray
    best_fitness: float
    best_pose_error: float
    fitness_history: np.ndarray      # incumbent (best-by-fitness) after each iteration
    pose_error_history: np.ndarray   # pose error of that incumbent after each iteration
    diversity_history: np.ndarray    # population std after each iteration
    evals_history: np.ndarray        # cumulative evaluate() calls after each iteration
    n_evaluations: int               # total evaluate() calls for this point


class CountingEvaluator:
    """Wraps evaluate(q) -> (fitness, pose_error), counting every call."""

    def __init__(self, evaluate: Evaluate):
        self._evaluate = evaluate
        self.calls = 0

    def __call__(self, q: np.ndarray) -> tuple[float, float]:
        self.calls += 1
        return self._evaluate(q)


def injected_population(
    pop: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    memory: MemoryMode,
    prev_best: np.ndarray | None,
    injection_fraction: float,
    injection_sigma: float,
) -> np.ndarray:
    """Initial population for cold/injection modes.

    Injection seeds ceil(fraction * pop) individuals at prev_best + Gaussian
    jitter (clipped to bounds); everything else is uniform random. At the
    first point (prev_best is None) injection degenerates to cold — there is
    nothing to inject yet.
    """
    dim = len(lower)
    X = rng.uniform(lower, upper, (pop, dim))
    if memory is MemoryMode.INJECTION and prev_best is not None:
        k = int(np.ceil(injection_fraction * pop))
        seeds = prev_best + rng.normal(0.0, injection_sigma, (k, dim))
        X[:k] = np.clip(seeds, lower, upper)
    return X
