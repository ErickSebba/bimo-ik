"""Forward-kinematics regression tests against hand-derived poses and the
canonical implementation from reference/main.py (transcribed inline)."""

import numpy as np
import pytest

from bimo_ik.kinematics import (
    DH_PARAMS,
    JOINT_LOWER,
    JOINT_UPPER,
    N_JOINTS,
    fk_pose,
    fk_position,
    link_positions,
)

# --- canonical FK transcribed independently from reference/main.py ----------

_CANONICAL_DH = [
    [0, 184, 0, np.pi / 2],
    [0, 0, 222, 0],
    [0, 0, 205, 0],
    [0, 0, 0, -np.pi / 2],
    [0, 0, 0, np.pi / 2],
    [0, 75, 0, 0],
]


def _canonical_dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _canonical_fk(q):
    T = np.eye(4, dtype=float)
    for i in range(6):
        th, d, a, al = _CANONICAL_DH[i]
        T = T @ _canonical_dh_matrix(q[i] + th, d, a, al)
    return T[:3, 3]


# --- hand-derived anchors ----------------------------------------------------

def test_home_pose_position():
    # Derived by hand-composing the six DH transforms at q = 0:
    # shoulder lift d1=184 (z), links a2+a3=427 along x, tool d6=75 along -y.
    pos = fk_position(np.zeros(6))
    np.testing.assert_allclose(pos, [427.0, -75.0, 184.0], atol=1e-9)


def test_home_pose_rotation():
    # At q = 0 the accumulated rotation reduces to Rx(90 deg).
    _, R = fk_pose(np.zeros(6))
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_base_rotation_rotates_whole_arm():
    # Rotating joint 1 by 90 deg maps the home position through Rz(90 deg).
    q = np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fk_position(q), [75.0, 427.0, 184.0], atol=1e-9)


def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        _, R = fk_pose(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_matches_canonical_fk_on_random_configurations():
    rng = np.random.default_rng(42)
    for _ in range(200):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        np.testing.assert_allclose(fk_position(q), _canonical_fk(q), atol=1e-9)


def test_link_positions_chain():
    pts = link_positions(np.zeros(6))
    assert pts.shape == (N_JOINTS + 1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(pts[1], [0.0, 0.0, 184.0], atol=1e-12)
    np.testing.assert_allclose(pts[-1], fk_position(np.zeros(6)), atol=1e-12)


def test_dh_table_shape_and_limits():
    assert DH_PARAMS.shape == (6, 4)
    assert np.all(JOINT_LOWER < JOINT_UPPER)
