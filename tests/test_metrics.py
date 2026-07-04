"""Metrics: accuracy from FK, kinematic proxies, effort-vs-epsilon censoring."""

import numpy as np
import pytest

from bimo_ik.kinematics import fk_pose
from bimo_ik.metrics import (
    accuracy_summary,
    effort_table,
    kinematic_summary,
    pose_errors,
    summarize_effort,
)
from bimo_ik.metrics.effort import first_crossing
from bimo_ik.trajectories.generators import Trajectory


def _exact_trajectory(q_path):
    positions, rotations = [], []
    for q in q_path:
        p, R = fk_pose(q)
        positions.append(p)
        rotations.append(R)
    return Trajectory("exact", np.array(positions), np.array(rotations))


# --- accuracy -----------------------------------------------------------------

def test_perfect_tracking_scores_zero():
    rng = np.random.default_rng(0)
    q_path = rng.uniform(-0.5, 0.5, (5, 6))
    traj = _exact_trajectory(q_path)
    summary = accuracy_summary(q_path, traj)
    # arccos near 1 is only sqrt(eps)-accurate (~1.5e-8 rad); that IS zero
    # for this metric's purposes.
    for value in summary.values():
        assert value == pytest.approx(0.0, abs=1e-6)


def test_position_and_orientation_errors_are_separate():
    q_path = np.zeros((2, 6))
    traj = _exact_trajectory(q_path)
    # shift only the target positions -> orientation error stays zero
    shifted = Trajectory("shifted", traj.positions + [3.0, 0.0, 4.0], traj.rotations)
    pos_err, ori_err = pose_errors(q_path, shifted)
    np.testing.assert_allclose(pos_err, 5.0, atol=1e-12)
    np.testing.assert_allclose(ori_err, 0.0, atol=1e-12)


# --- kinematic ------------------------------------------------------------------

def test_kinematic_summary_on_known_path():
    # single joint moving 0 -> 0.1 -> 0.3 (steps 0.1, 0.2)
    q_path = np.zeros((3, 6))
    q_path[1, 0] = 0.1
    q_path[2, 0] = 0.3
    m = kinematic_summary(q_path)
    assert m["smoothness_rad"] == pytest.approx(0.3)
    assert m["smoothness_index"] == pytest.approx(0.15)
    assert m["jerk_rad"] == pytest.approx(0.1)  # |0.2 - 0.1|
    assert m["peak_joint_velocity_rad"] == pytest.approx(0.2)
    assert m["peak_joint_acceleration_rad"] == pytest.approx(0.1)


def test_constant_path_has_zero_motion():
    m = kinematic_summary(np.ones((10, 6)))
    assert m["smoothness_rad"] == 0.0
    assert m["jerk_rad"] == 0.0


# --- effort-vs-epsilon ----------------------------------------------------------

def test_first_crossing():
    history = np.array([0.5, 0.3, 0.09, 0.04, 0.04])
    assert first_crossing(history, 0.1) == 3
    assert first_crossing(history, 0.5) == 1
    assert first_crossing(history, 0.01) is None


def test_effort_table_records_censoring_at_cap():
    max_iter = 5
    pose_hist = np.array(
        [
            [0.5, 0.3, 0.09, 0.04, 0.04],   # crosses 0.1 at iter 3
            [0.5, 0.4, 0.35, 0.30, 0.29],   # never crosses 0.1 -> censored
        ]
    )
    evals = np.cumsum(np.full((2, max_iter), 10), axis=1)  # 10 evals/iter
    table = effort_table(pose_hist, evals, np.array([0.1]))

    assert table.iterations[0, 0] == 3
    assert table.evaluations[0, 0] == 30
    assert not table.censored[0, 0]

    assert table.censored[0, 1]
    assert table.iterations[0, 1] == max_iter          # recorded at the cap
    assert table.evaluations[0, 1] == evals[1, -1]     # full budget spent


def test_effort_is_monotone_in_epsilon():
    rng = np.random.default_rng(1)
    # decreasing histories
    hist = np.sort(rng.uniform(0, 1, (4, 50)), axis=1)[:, ::-1]
    evals = np.cumsum(np.full((4, 50), 7), axis=1)
    eps = np.array([0.5, 0.2, 0.05, 0.01])
    table = effort_table(hist, evals, eps)
    # tighter epsilon can never need FEWER iterations
    assert np.all(np.diff(table.iterations, axis=0) >= 0)
    # and censoring can only grow
    assert np.all(table.censored[:-1] <= table.censored[1:])


def test_summarize_effort_aggregates_and_counts_censoring():
    pose_a = np.array([[0.5, 0.05, 0.05]])
    pose_b = np.array([[0.5, 0.4, 0.3]])
    evals = np.cumsum(np.full((1, 3), 10), axis=1)
    eps = np.array([0.1])
    t_a = effort_table(pose_a, evals, eps)
    t_b = effort_table(pose_b, evals, eps)
    s = summarize_effort([t_a, t_b])
    assert s["fraction_reached"][0] == pytest.approx(0.5)
    assert s["n_censored"][0] == 1
    assert s["n_points"] == 2
    # median over {2, cap=3}
    assert s["median_iterations"][0] == pytest.approx(2.5)


def test_summarize_effort_rejects_mismatched_sweeps():
    evals = np.cumsum(np.full((1, 3), 10), axis=1)
    t_a = effort_table(np.array([[0.5, 0.05, 0.05]]), evals, np.array([0.1]))
    t_b = effort_table(np.array([[0.5, 0.05, 0.05]]), evals, np.array([0.2]))
    with pytest.raises(ValueError):
        summarize_effort([t_a, t_b])
