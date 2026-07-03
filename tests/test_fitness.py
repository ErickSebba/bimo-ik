"""Fitness: dimensionless, deterministic, correct schedule endpoints."""

import numpy as np
import pytest

from bimo_ik.fitness import FitnessWeights, make_evaluator, scheduled_weights
from bimo_ik.kinematics import fk_pose, joint_range_norm, max_reach

BASE = FitnessWeights(w_pos=1.0, w_ori=1.0, w_joint=0.1)


def _evaluator(weights=BASE, q_ref=None):
    target_pos, target_rot = fk_pose(np.array([0.3, 0.5, -0.2, 0.1, 0.4, -0.3]))
    return make_evaluator(
        target_pos,
        target_rot,
        np.zeros(6) if q_ref is None else q_ref,
        weights,
        BASE,
        max_reach(),
        joint_range_norm(),
    )


def test_exact_solution_scores_zero_pose_error():
    q_star = np.array([0.3, 0.5, -0.2, 0.1, 0.4, -0.3])
    evaluate = _evaluator()
    fitness, pose_error = evaluate(q_star)
    assert pose_error == pytest.approx(0.0, abs=1e-12)
    # only the continuity term remains
    assert fitness == pytest.approx(
        0.1 * np.linalg.norm(q_star) / joint_range_norm(), abs=1e-12
    )


def test_deterministic_no_noise():
    evaluate = _evaluator()
    q = np.array([0.1, -0.4, 0.9, 0.0, 1.0, -1.2])
    assert evaluate(q) == evaluate(q)


def test_terms_are_dimensionless():
    # Every term lies in a comparable O(1) range over the joint space.
    evaluate = _evaluator()
    rng = np.random.default_rng(5)
    from bimo_ik.kinematics import JOINT_LOWER, JOINT_UPPER

    for _ in range(200):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        fitness, pose_error = evaluate(q)
        assert 0.0 <= pose_error <= 3.0
        assert 0.0 <= fitness <= 3.5


def test_pose_error_uses_base_weights_not_scheduled():
    # In an adaptive cell the fitness changes with the schedule but the
    # epsilon criterion (pose_error) must not.
    q = np.array([0.1, -0.4, 0.9, 0.0, 1.0, -1.2])
    f_static, pe_static = _evaluator(BASE)(q)
    f_adaptive, pe_adaptive = _evaluator(scheduled_weights(BASE, 1.0))(q)
    assert pe_static == pytest.approx(pe_adaptive, abs=1e-15)
    assert f_static != pytest.approx(f_adaptive)


def test_schedule_endpoints():
    w0 = scheduled_weights(BASE, 0.0)
    w1 = scheduled_weights(BASE, 1.0)
    assert w0.w_pos == pytest.approx(0.7) and w1.w_pos == pytest.approx(1.3)
    assert w0.w_ori == pytest.approx(0.7) and w1.w_ori == pytest.approx(1.3)
    assert w0.w_joint == pytest.approx(0.15) and w1.w_joint == pytest.approx(0.05)


def test_schedule_rejects_out_of_range_progress():
    with pytest.raises(ValueError):
        scheduled_weights(BASE, 1.5)


def test_continuity_term_pulls_toward_reference():
    q_ref = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    evaluate = _evaluator(q_ref=q_ref)
    f_at_ref, _ = evaluate(q_ref)
    f_far, _ = evaluate(q_ref + 1.0)
    # same pose error contribution is not guaranteed, but at q_ref the
    # continuity term is exactly zero:
    _, pe_ref = evaluate(q_ref)
    assert f_at_ref == pytest.approx(pe_ref, abs=1e-12)
    assert f_far > f_at_ref
