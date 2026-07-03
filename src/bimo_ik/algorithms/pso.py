"""Particle Swarm Optimization with explicit memory modes.

The update rule and hyperparameters transcribe the canonical PSOBase from
reference/main.py (inertia 0.7, c1 = c2 = 1.5, velocity clamped to 20% of the
joint range, positions clipped to bounds, asynchronous pbest/gbest updates).
Differences from the reference are policy, not dynamics:

- randomness comes from an injected ``numpy.random.Generator`` (never global);
- ``persistence`` mode reproduces PSODinamico.atualizar_landscape semantics:
  the swarm state (X, V, pbest, gbest) survives across trajectory points and
  the personal/global memories are re-evaluated under the new landscape
  before flight resumes;
- the evaluator returns (fitness, pose_error); the incumbent's pose error is
  tracked alongside its fitness for the effort-vs-epsilon analysis.
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
class PSOParams:
    pop: int = 40
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    v_max_frac: float = 0.2


class PSO:
    def __init__(
        self,
        params: PSOParams,
        memory: MemoryMode,
        lower: np.ndarray,
        upper: np.ndarray,
        injection_fraction: float = 0.25,
        injection_sigma: float = 0.1,
    ):
        self.params = params
        self.memory = MemoryMode(memory)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.v_max = params.v_max_frac * (self.upper - self.lower)
        self.injection_fraction = injection_fraction
        self.injection_sigma = injection_sigma
        self._state: dict | None = None  # persistent swarm (persistence mode only)
        self.last_initial_population: np.ndarray | None = None  # inspection hook

    @property
    def state(self) -> dict | None:
        """Persistent swarm state (inspection hook for tests)."""
        return self._state

    def reset(self) -> None:
        """Forget all cross-point state (start of a new independent run)."""
        self._state = None
        self.last_initial_population = None

    def solve_point(
        self,
        evaluate: Evaluate,
        n_iter: int,
        rng: np.random.Generator,
        prev_best: np.ndarray | None = None,
    ) -> PointResult:
        ev = CountingEvaluator(evaluate)
        pop = self.params.pop
        dim = len(self.lower)

        if self.memory is MemoryMode.PERSISTENCE and self._state is not None:
            # Landscape swap: keep X, V and the pbest POSITIONS; re-evaluate
            # the memories so stale values never steer the flight.
            st = self._state
            X, V, pbest_x = st["X"], st["V"], st["pbest_x"]
            pbest_f = np.empty(pop)
            pbest_pe = np.empty(pop)
            for i in range(pop):
                pbest_f[i], pbest_pe[i] = ev(pbest_x[i])
            g = int(np.argmin(pbest_f))
            gbest_x = pbest_x[g].copy()
            gbest_f, gbest_pe = float(pbest_f[g]), float(pbest_pe[g])
        else:
            X = injected_population(
                pop, self.lower, self.upper, rng, self.memory, prev_best,
                self.injection_fraction, self.injection_sigma,
            )
            V = rng.uniform(-0.5, 0.5, (pop, dim)) * (self.upper - self.lower) * 0.1
            pbest_x = X.copy()
            pbest_f = np.empty(pop)
            pbest_pe = np.empty(pop)
            for i in range(pop):
                pbest_f[i], pbest_pe[i] = ev(X[i])
            g = int(np.argmin(pbest_f))
            gbest_x = X[g].copy()
            gbest_f, gbest_pe = float(pbest_f[g]), float(pbest_pe[g])

        self.last_initial_population = X.copy()

        fitness_hist = np.empty(n_iter)
        pe_hist = np.empty(n_iter)
        div_hist = np.empty(n_iter)
        ev_hist = np.empty(n_iter, dtype=np.int64)

        w, c1, c2 = self.params.inertia, self.params.cognitive, self.params.social
        for it in range(n_iter):
            r1 = rng.random((pop, dim))
            r2 = rng.random((pop, dim))
            V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
            V = np.clip(V, -self.v_max, self.v_max)
            X = np.clip(X + V, self.lower, self.upper)

            for i in range(pop):
                f, pe = ev(X[i])
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_pe[i] = pe
                    pbest_x[i] = X[i].copy()
                if f < gbest_f:
                    gbest_f, gbest_pe = f, pe
                    gbest_x = X[i].copy()

            fitness_hist[it] = gbest_f
            pe_hist[it] = gbest_pe
            div_hist[it] = float(np.std(X))
            ev_hist[it] = ev.calls

        if self.memory is MemoryMode.PERSISTENCE:
            self._state = {
                "X": X, "V": V,
                "pbest_x": pbest_x, "pbest_f": pbest_f, "pbest_pe": pbest_pe,
                "gbest_x": gbest_x.copy(), "gbest_f": gbest_f, "gbest_pe": gbest_pe,
            }

        return PointResult(
            best_q=gbest_x.copy(),
            best_fitness=gbest_f,
            best_pose_error=gbest_pe,
            fitness_history=fitness_hist,
            pose_error_history=pe_hist,
            diversity_history=div_hist,
            evals_history=ev_hist,
            n_evaluations=ev.calls,
        )
