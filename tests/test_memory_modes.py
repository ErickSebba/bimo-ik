"""Proof tests for the memory factor: each mode does exactly what it claims.

These use a cheap synthetic objective (distance to a movable center) so the
swarm/population mechanics are tested in isolation from the robot model.
"""

import numpy as np
import pytest

from bimo_ik.algorithms import GA, GAParams, MemoryMode, PSO, PSOParams

LOWER = -np.ones(6)
UPPER = np.ones(6)
POP = 12
K_INJ = int(np.ceil(0.25 * POP))  # injected individuals


def sphere(center):
    center = np.asarray(center, dtype=float)

    def evaluate(q):
        d = float(np.linalg.norm(q - center))
        return d, d / 2.0

    return evaluate


def make_pso(memory):
    return PSO(PSOParams(pop=POP), memory, LOWER, UPPER,
               injection_fraction=0.25, injection_sigma=0.1)


def make_ga(memory):
    return GA(GAParams(pop=POP), memory, LOWER, UPPER,
              injection_fraction=0.25, injection_sigma=0.1)


# --- cold ---------------------------------------------------------------------

def test_pso_cold_keeps_no_state_between_points():
    opt = make_pso(MemoryMode.COLD)
    rng = np.random.default_rng(0)
    r1 = opt.solve_point(sphere([0.5] * 6), 5, rng)
    assert opt.state is None
    x1 = opt.last_initial_population.copy()
    opt.solve_point(sphere([-0.5] * 6), 5, rng, prev_best=r1.best_q)
    assert opt.state is None
    x2 = opt.last_initial_population
    # fresh uniform draw, not derived from point 1
    assert not np.allclose(x1, x2)
    # and NOT clustered around the previous best
    near = np.sum(np.linalg.norm(x2 - r1.best_q, axis=1) < 0.5)
    assert near <= 2  # uniform chance hits, never a seeded cluster


def test_cold_initial_population_matches_fresh_rng_draw():
    # the cold init at point i+1 must be exactly what a fresh optimizer would
    # draw from the same generator state: zero information carryover.
    opt = make_pso(MemoryMode.COLD)
    rng_a = np.random.default_rng(123)
    opt.solve_point(sphere([0.2] * 6), 3, rng_a, prev_best=None)
    opt.solve_point(sphere([0.4] * 6), 3, rng_a, prev_best=np.zeros(6))
    second_init = opt.last_initial_population.copy()

    rng_b = np.random.default_rng(123)
    fresh = make_pso(MemoryMode.COLD)
    fresh.solve_point(sphere([0.2] * 6), 3, rng_b, prev_best=None)
    fresh2 = make_pso(MemoryMode.COLD)  # brand-new object, same rng stream
    fresh2.solve_point(sphere([0.4] * 6), 3, rng_b, prev_best=np.zeros(6))
    np.testing.assert_array_equal(second_init, fresh2.last_initial_population)


# --- injection ----------------------------------------------------------------

@pytest.mark.parametrize("factory", [make_pso, make_ga])
def test_injection_seeds_exact_fraction_near_previous_best(factory):
    opt = factory(MemoryMode.INJECTION)
    rng = np.random.default_rng(1)
    prev = np.array([0.3, -0.2, 0.1, 0.4, -0.1, 0.2])
    opt.solve_point(sphere([0.0] * 6), 2, rng, prev_best=prev)
    X = opt.last_initial_population
    dists = np.linalg.norm(X - prev, axis=1)
    # first K_INJ rows are jittered copies of prev (sigma 0.1 -> tight)
    assert np.all(dists[:K_INJ] < 0.6)
    # the rest is uniform: overwhelmingly far from prev in 6-D
    assert np.median(dists[K_INJ:]) > 0.6


@pytest.mark.parametrize("factory", [make_pso, make_ga])
def test_injection_degenerates_to_cold_at_first_point(factory):
    opt = factory(MemoryMode.INJECTION)
    rng = np.random.default_rng(2)
    opt.solve_point(sphere([0.0] * 6), 2, rng, prev_best=None)
    X = opt.last_initial_population
    # nothing to inject: population is a plain uniform draw
    rng_ref = np.random.default_rng(2)
    np.testing.assert_array_equal(X, rng_ref.uniform(LOWER, UPPER, (POP, 6)))


# --- persistence (PSO only) ---------------------------------------------------

def test_persistence_carries_swarm_state_across_points():
    opt = make_pso(MemoryMode.PERSISTENCE)
    rng = np.random.default_rng(3)
    opt.solve_point(sphere([0.5] * 6), 5, rng)
    assert opt.state is not None
    x_end = opt.state["X"].copy()
    v_end = opt.state["V"].copy()
    pbest_end = opt.state["pbest_x"].copy()

    opt.solve_point(sphere([-0.5] * 6), 0, rng)  # 0 iterations: swap only
    # the swarm entering point 2 is exactly the swarm that left point 1
    np.testing.assert_array_equal(opt.last_initial_population, x_end)
    np.testing.assert_array_equal(opt.state["V"], v_end)
    np.testing.assert_array_equal(opt.state["pbest_x"], pbest_end)


def test_persistence_reevaluates_memories_on_landscape_swap():
    opt = make_pso(MemoryMode.PERSISTENCE)
    rng = np.random.default_rng(4)
    opt.solve_point(sphere([0.5] * 6), 5, rng)
    pbest_positions = opt.state["pbest_x"].copy()

    new_eval = sphere([-0.7] * 6)
    result = opt.solve_point(new_eval, 0, rng)  # swap + re-evaluation only
    expected = min(new_eval(x)[0] for x in pbest_positions)
    assert result.best_fitness == pytest.approx(expected, abs=1e-12)
    # and the stale value from the old landscape is gone
    old_eval = sphere([0.5] * 6)
    assert result.best_fitness != pytest.approx(
        min(old_eval(x)[0] for x in pbest_positions)
    )


def test_persistence_reset_forgets_everything():
    opt = make_pso(MemoryMode.PERSISTENCE)
    rng = np.random.default_rng(5)
    opt.solve_point(sphere([0.5] * 6), 3, rng)
    x_end = opt.state["X"].copy()
    opt.reset()
    assert opt.state is None
    opt.solve_point(sphere([0.5] * 6), 3, rng)
    assert not np.allclose(opt.last_initial_population, x_end)


def test_ga_refuses_persistence():
    with pytest.raises(ValueError):
        make_ga(MemoryMode.PERSISTENCE)


# --- bookkeeping ----------------------------------------------------------------

def test_pso_evaluation_accounting():
    opt = make_pso(MemoryMode.COLD)
    rng = np.random.default_rng(6)
    r = opt.solve_point(sphere([0.0] * 6), 7, rng)
    assert r.n_evaluations == POP * (1 + 7)  # init + one sweep per iteration
    assert r.evals_history[-1] == r.n_evaluations
    assert np.all(np.diff(r.evals_history) == POP)


def test_ga_evaluation_accounting():
    opt = make_ga(MemoryMode.COLD)
    rng = np.random.default_rng(7)
    r = opt.solve_point(sphere([0.0] * 6), 7, rng)
    # init pop + per generation: (pop - elite) renewal scoring + full rescore
    per_gen = (POP - 2) + POP
    assert r.n_evaluations == POP + 7 * per_gen
    assert r.evals_history[-1] == r.n_evaluations


@pytest.mark.parametrize("factory", [make_pso, make_ga])
def test_incumbent_fitness_never_increases_within_a_point(factory):
    opt = factory(MemoryMode.COLD)
    rng = np.random.default_rng(8)
    r = opt.solve_point(sphere([0.4] * 6), 30, rng)
    assert np.all(np.diff(r.fitness_history) <= 1e-15)
    assert r.best_fitness == pytest.approx(r.fitness_history[-1])


def test_pso_converges_on_easy_problem():
    opt = make_pso(MemoryMode.COLD)
    rng = np.random.default_rng(9)
    r = opt.solve_point(sphere([0.3] * 6), 60, rng)
    assert r.best_fitness < 1e-2


def test_ga_converges_on_easy_problem():
    opt = make_ga(MemoryMode.COLD)
    rng = np.random.default_rng(10)
    r = opt.solve_point(sphere([0.3] * 6), 60, rng)
    assert r.best_fitness < 5e-2
