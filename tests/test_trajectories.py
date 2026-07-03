"""Trajectory generators + reachability guard."""

import numpy as np
import pytest

from bimo_ik.kinematics import matrix_from_quat, rotation_error
from bimo_ik.trajectories import (
    Trajectory,
    TrajectorySpec,
    UnreachablePoseError,
    build_trajectory,
    default_trajectory_specs,
    solve_ik,
    validate_trajectory,
)

SPECS = default_trajectory_specs(30)


def test_default_specs_cover_three_kinds():
    assert set(SPECS) == {"line", "arc", "corner"}
    assert all(spec.n_points == 30 for spec in SPECS.values())


def test_line_endpoints_match_prior_paper():
    traj = build_trajectory(SPECS["line"])
    np.testing.assert_allclose(traj.positions[0], [150.0, 100.0, 200.0])
    np.testing.assert_allclose(traj.positions[-1], [300.0, -100.0, 200.0])
    # straight: every point on the segment
    d = traj.positions[-1] - traj.positions[0]
    for p in traj.positions:
        cross = np.cross(p - traj.positions[0], d)
        assert np.linalg.norm(cross) < 1e-6


def test_arc_geometry():
    traj = build_trajectory(SPECS["arc"])
    start = np.array(SPECS["arc"].start)
    end = np.array(SPECS["arc"].end)
    center = (start + end) / 2.0
    radius = np.linalg.norm(end - start) / 2.0
    np.testing.assert_allclose(traj.positions[0], start, atol=1e-9)
    np.testing.assert_allclose(traj.positions[-1], end, atol=1e-9)
    # all points on the circle, constant height
    dists = np.linalg.norm(traj.positions - center, axis=1)
    np.testing.assert_allclose(dists, radius, atol=1e-9)
    assert np.allclose(traj.positions[:, 2], 200.0)


def test_corner_geometry():
    traj = build_trajectory(SPECS["corner"])
    corner = np.array(SPECS["corner"].corner)
    # the corner waypoint appears exactly once
    hits = np.sum(np.all(np.isclose(traj.positions, corner, atol=1e-9), axis=1))
    assert hits == 1
    # every point lies on one of the two legs
    start = np.array(SPECS["corner"].start)
    end = np.array(SPECS["corner"].end)
    for p in traj.positions:
        on1 = np.linalg.norm(np.cross(p - start, corner - start)) < 1e-6
        on2 = np.linalg.norm(np.cross(p - corner, end - corner)) < 1e-6
        assert on1 or on2


def test_orientations_interpolate_smoothly_by_arc_length():
    for spec in SPECS.values():
        traj = build_trajectory(spec)
        R_start = matrix_from_quat(spec.quat_start)
        R_end = matrix_from_quat(spec.quat_end)
        assert rotation_error(traj.rotations[0], R_start) < 1e-9
        assert rotation_error(traj.rotations[-1], R_end) < 1e-9
        total = rotation_error(R_start, R_end)
        steps = [
            rotation_error(traj.rotations[i], traj.rotations[i + 1])
            for i in range(len(traj) - 1)
        ]
        # no orientation discontinuities: every step well below the total sweep
        assert max(steps) < 0.35 * total
        assert sum(steps) == pytest.approx(total, rel=1e-6)


def test_corner_spec_requires_waypoint():
    with pytest.raises(ValueError):
        TrajectorySpec("bad", "corner", 30, (0, 0, 0), (1, 1, 1))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        TrajectorySpec("bad", "spiral", 30, (0, 0, 0), (1, 1, 1))


def test_all_default_trajectories_pass_the_guard():
    for spec in SPECS.values():
        errors = validate_trajectory(build_trajectory(spec), tol=1e-3)
        assert np.all(errors <= 1e-3)


def test_guard_rejects_unreachable_position():
    # far outside the workspace (reach bound is 686 mm)
    positions = np.array([[1200.0, 0.0, 200.0]])
    rotations = np.array([np.eye(3)])
    bad = Trajectory("bad", positions, rotations)
    with pytest.raises(UnreachablePoseError):
        validate_trajectory(bad, n_starts=3)


def test_guard_rejects_unreachable_orientation():
    # position reachable, but the pose forces the tool axis somewhere the
    # wrist cannot go at full stretch: at (0, 0, 686) the arm is fully
    # extended upward and the tool must point +z; demanding tool -z there
    # is impossible.
    positions = np.array([[0.0, 0.0, 686.0]])
    rotations = np.array([np.diag([1.0, -1.0, -1.0])])  # tool z -> -z world
    bad = Trajectory("bad-ori", positions, rotations)
    with pytest.raises(UnreachablePoseError):
        validate_trajectory(bad, n_starts=3)


def test_solve_ik_certifies_known_pose():
    from bimo_ik.kinematics import fk_pose

    q_true = np.array([0.2, 0.4, 0.9, -0.3, 0.8, 0.5])
    pos, rot = fk_pose(q_true)
    sol = solve_ik(pos, rot, q_seed=q_true + 0.2)
    assert sol.pose_error < 1e-6
