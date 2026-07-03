"""SO(3) utilities: geodesic error (with the mandatory clip guard), quaternions, SLERP."""

import numpy as np
import pytest

from bimo_ik.kinematics import (
    matrix_from_quat,
    quat_from_matrix,
    rotation_error,
    rotation_vector,
    slerp,
)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def test_error_of_identity_is_zero():
    assert rotation_error(np.eye(3), np.eye(3)) == 0.0


def test_error_of_known_rotations():
    assert rotation_error(np.eye(3), _rot_x(np.pi / 2)) == pytest.approx(np.pi / 2)
    assert rotation_error(np.eye(3), _rot_z(np.pi)) == pytest.approx(np.pi)
    # relative rotation between two non-identity matrices
    assert rotation_error(_rot_z(0.3), _rot_z(0.3 + 1.1)) == pytest.approx(1.1)


def test_clip_guard_prevents_nan():
    # A rotation matrix whose float trace exceeds 3: without the clip,
    # arccos((trace-1)/2) would be NaN.
    R = np.eye(3) * (1.0 + 1e-15)
    assert (np.trace(R) - 1.0) / 2.0 > 1.0
    err = rotation_error(np.eye(3), R)
    assert not np.isnan(err)
    assert err == 0.0
    # Same guard on the -1 side (trace slightly below -1).
    R_pi = _rot_x(np.pi) @ np.diag([1.0 + 1e-15] * 3)
    err_pi = rotation_error(np.eye(3), R_pi)
    assert not np.isnan(err_pi)
    assert err_pi == pytest.approx(np.pi)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(100):
        v = rng.normal(size=4)
        q = v / np.linalg.norm(v)
        R = matrix_from_quat(q)
        q_back = quat_from_matrix(R)
        # q and -q encode the same rotation
        assert min(np.linalg.norm(q - q_back), np.linalg.norm(q + q_back)) < 1e-9
        np.testing.assert_allclose(matrix_from_quat(q_back), R, atol=1e-9)


def test_slerp_endpoints_and_midpoint():
    q0 = quat_from_matrix(np.eye(3))
    q1 = quat_from_matrix(_rot_z(np.pi / 2))
    np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-12)
    np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-12)
    R_mid = matrix_from_quat(slerp(q0, q1, 0.5))
    assert rotation_error(R_mid, _rot_z(np.pi / 4)) < 1e-9


def test_slerp_takes_shortest_arc():
    q0 = quat_from_matrix(_rot_z(0.1))
    q1 = -quat_from_matrix(_rot_z(0.4))  # antipodal representative
    R_mid = matrix_from_quat(slerp(q0, q1, 0.5))
    assert rotation_error(R_mid, _rot_z(0.25)) < 1e-9


def test_slerp_angular_velocity_is_constant():
    q0 = quat_from_matrix(np.eye(3))
    q1 = quat_from_matrix(_rot_x(2.0))
    ts = np.linspace(0, 1, 11)
    Rs = [matrix_from_quat(slerp(q0, q1, t)) for t in ts]
    steps = [rotation_error(Rs[i], Rs[i + 1]) for i in range(10)]
    np.testing.assert_allclose(steps, 0.2, atol=1e-9)


def test_rotation_vector_log_map():
    np.testing.assert_allclose(rotation_vector(np.eye(3)), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rotation_vector(_rot_x(0.7)), [0.7, 0, 0], atol=1e-9)
    # robust near pi
    v = rotation_vector(_rot_z(np.pi - 1e-9))
    assert np.linalg.norm(v) == pytest.approx(np.pi - 1e-9, abs=1e-6)
    # norm of the log map equals the geodesic error
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4)
        R = matrix_from_quat(q / np.linalg.norm(q))
        assert np.linalg.norm(rotation_vector(R)) == pytest.approx(
            rotation_error(np.eye(3), R), abs=1e-9
        )
