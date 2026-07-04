"""Effort-vs-epsilon: how much work each method needs to reach an accuracy level.

Derived post hoc from the per-iteration incumbent pose-error histories that
the runner records at a fixed iteration cap. For each epsilon in the sweep
and each trajectory point, effort is the first iteration (1-based; 0 = the
initial population already satisfied epsilon) at which the incumbent's pose
error crossed below epsilon, together with the cumulative evaluation count
at that moment. Points that never cross within the cap are RIGHT-CENSORED:
recorded at the cap with censored=True, never discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EffortTable:
    """Effort data for one run: arrays shaped (n_epsilon, n_points)."""

    epsilon_values: np.ndarray     # (n_epsilon,)
    iterations: np.ndarray         # first-crossing iteration; cap where censored
    evaluations: np.ndarray        # cumulative evals at crossing; total where censored
    censored: np.ndarray           # bool


def first_crossing(pose_error_history: np.ndarray, epsilon: float) -> int | None:
    """First 1-based iteration whose incumbent pose error is <= epsilon.

    Returns None if the history never crosses (right-censored).
    Note: iteration 0 (the initial population) is not part of the recorded
    history; a first-iteration crossing therefore returns 1.
    """
    below = np.nonzero(pose_error_history <= epsilon)[0]
    if len(below) == 0:
        return None
    return int(below[0]) + 1


def effort_table(
    pose_error_hist: np.ndarray,   # (n_points, max_iter)
    evals_hist: np.ndarray,        # (n_points, max_iter) cumulative within point
    epsilon_values: np.ndarray,
) -> EffortTable:
    epsilon_values = np.asarray(epsilon_values, dtype=float)
    n_points, max_iter = pose_error_hist.shape
    n_eps = len(epsilon_values)

    iterations = np.full((n_eps, n_points), max_iter, dtype=np.int64)
    evaluations = np.empty((n_eps, n_points), dtype=np.int64)
    censored = np.zeros((n_eps, n_points), dtype=bool)

    for j in range(n_points):
        history = pose_error_hist[j]
        for k, eps in enumerate(epsilon_values):
            crossing = first_crossing(history, eps)
            if crossing is None:
                censored[k, j] = True
                evaluations[k, j] = evals_hist[j, -1]
            else:
                iterations[k, j] = crossing
                evaluations[k, j] = evals_hist[j, crossing - 1]
    return EffortTable(epsilon_values, iterations, evaluations, censored)


def summarize_effort(tables: list[EffortTable]) -> dict:
    """Aggregate effort tables (e.g. all runs of one cell) into per-epsilon
    summaries.

    For each epsilon: the fraction of points that reached it, and the median
    iterations/evaluations. The median treats censored points at the cap —
    a conservative, standard convention that keeps censored data in the
    statistic (valid as long as the median itself is uncensored; the
    reached-fraction is reported alongside so readers can tell).
    """
    eps = tables[0].epsilon_values
    for t in tables:
        if not np.array_equal(t.epsilon_values, eps):
            raise ValueError("effort tables have mismatched epsilon sweeps")

    iterations = np.hstack([t.iterations for t in tables])    # (n_eps, total_points)
    evaluations = np.hstack([t.evaluations for t in tables])
    censored = np.hstack([t.censored for t in tables])

    return {
        "epsilon_values": eps.copy(),
        "fraction_reached": 1.0 - censored.mean(axis=1),
        "median_iterations": np.median(iterations, axis=1),
        "median_evaluations": np.median(evaluations, axis=1),
        "n_points": iterations.shape[1],
        "n_censored": censored.sum(axis=1),
    }
