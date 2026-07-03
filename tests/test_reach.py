"""Characteristic scales: computed from the model, bounding, attainable."""

import numpy as np
import pytest

from bimo_ik.kinematics import (
    DH_PARAMS,
    JOINT_LOWER,
    JOINT_UPPER,
    fk_position,
    joint_range_norm,
    max_reach,
)


def test_max_reach_value_follows_dh_table():
    # sum of per-row translation lengths sqrt(a^2 + d^2) for the BIMO table
    expected = 184.0 + 222.0 + 205.0 + 75.0
    assert max_reach() == pytest.approx(expected)
    # and it is genuinely derived from the table, not a constant:
    scaled = DH_PARAMS.copy()
    scaled[:, 1:3] *= 2.0
    assert max_reach(scaled) == pytest.approx(2.0 * expected)


def test_max_reach_bounds_all_sampled_positions():
    rng = np.random.default_rng(0)
    l_reach = max_reach()
    for _ in range(2000):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        assert np.linalg.norm(fk_position(q)) <= l_reach + 1e-9


def test_max_reach_is_attained_by_stretched_configuration():
    # q2 = +90 deg raises the planar chain to vertical and q5 = +90 deg aligns
    # the tool axis with it; all link lengths then add up along +z, so the
    # bound is tight (and the configuration is within joint limits).
    q_stretched = np.array([0.0, np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
    assert np.all(q_stretched >= JOINT_LOWER) and np.all(q_stretched <= JOINT_UPPER)
    assert np.linalg.norm(fk_position(q_stretched)) == pytest.approx(max_reach(), rel=1e-12)


def test_joint_range_norm():
    assert joint_range_norm() == pytest.approx(
        float(np.linalg.norm(JOINT_UPPER - JOINT_LOWER))
    )
    assert joint_range_norm(np.zeros(3), np.ones(3)) == pytest.approx(np.sqrt(3.0))
