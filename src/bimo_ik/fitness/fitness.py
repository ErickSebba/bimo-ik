"""Dimensionless full-pose fitness.

    f(q) = w_pos * (e_pos / L_reach)
         + w_ori * (e_ori / pi)
         + w_joint * (||q - q_ref|| / ||joint range||)

Each term is normalized by a characteristic scale of the problem (arm reach
computed from the DH table, the maximal geodesic angle pi, and the norm of
the joint-range vector), so the weights are directly comparable dimensionless
numbers rather than unit-dependent magic constants. The objective is fully
deterministic: there is deliberately no stochastic term of any kind.

The evaluator also returns the *pose error*
    e = w_pos_base * (e_pos / L_reach) + w_ori_base * (e_ori / pi)
computed with the BASE (unscheduled) weights and no continuity term. This is
the epsilon criterion of the effort-vs-epsilon analysis; using base weights
for every experimental cell keeps the criterion identical across landscape
modes, and excluding the continuity term keeps "accuracy effort" from being
conflated with the very memory effect under study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..kinematics.dh import fk_pose
from ..kinematics.orientation import rotation_error

Evaluator = Callable[[np.ndarray], tuple[float, float]]


@dataclass(frozen=True)
class FitnessWeights:
    """Dimensionless fitness weights."""

    w_pos: float = 1.0
    w_ori: float = 1.0
    w_joint: float = 0.1


def scheduled_weights(base: FitnessWeights, progress: float) -> FitnessWeights:
    """Adaptive landscape schedule (an explicit ablation factor).

    For trajectory progress t in [0, 1]:
      accuracy terms (position and orientation) scale by (0.7 + 0.6 t),
      the continuity term scales by (1.5 - t).

    This mirrors the undocumented weight schedule found in the prior study's
    PSO-DFL fitness, now declared and tested as an experimental factor.
    """
    t = float(progress)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {t}")
    accuracy = 0.7 + 0.6 * t
    continuity = 1.5 - t
    return FitnessWeights(
        w_pos=base.w_pos * accuracy,
        w_ori=base.w_ori * accuracy,
        w_joint=base.w_joint * continuity,
    )


def make_evaluator(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q_ref: np.ndarray,
    weights: FitnessWeights,
    base_weights: FitnessWeights,
    l_reach: float,
    q_range_norm: float,
) -> Evaluator:
    """Build evaluate(q) -> (fitness, pose_error) for one trajectory point.

    fitness    : minimized by the optimizer (uses `weights`, which may be the
                 scheduled variant in adaptive-landscape cells).
    pose_error : the epsilon criterion (uses `base_weights`, no continuity
                 term), computed from the same forward-kinematics call.
    """
    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)

    def evaluate(q: np.ndarray) -> tuple[float, float]:
        q = np.asarray(q, dtype=float)
        position, rotation = fk_pose(q)
        e_pos = np.linalg.norm(position - target_position) / l_reach
        e_ori = rotation_error(rotation, target_rotation) / np.pi
        e_joint = np.linalg.norm(q - q_ref) / q_range_norm
        fitness = weights.w_pos * e_pos + weights.w_ori * e_ori + weights.w_joint * e_joint
        pose_error = base_weights.w_pos * e_pos + base_weights.w_ori * e_ori
        return float(fitness), float(pose_error)

    return evaluate
