"""Pose math shared by training-time retrieval and the runtime Solidify memory.

Poses are (x, y, heading) in world units / radians. Keeping one definition of
"distance between views" and "relative pose features" for both training and
inference is what lets the model trust what the harness retrieves.
"""
from __future__ import annotations

import numpy as np

POSE_FEAT_DIM = 5  # (local_dx, local_dy, sin dtheta, cos dtheta, valid)
POS_SCALE = 4.0


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def view_distance(query: np.ndarray, poses: np.ndarray, angle_weight: float = 1.5) -> np.ndarray:
    """Distance from one query pose [3] to many poses [..., 3]."""
    d_pos = np.hypot(poses[..., 0] - query[0], poses[..., 1] - query[1])
    d_ang = np.abs(wrap(poses[..., 2] - query[2]))
    return d_pos + angle_weight * d_ang


def relative_pose_features(query: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Features of memory poses [..., 3] expressed in the query camera frame -> [..., 5]."""
    dx, dy = poses[..., 0] - query[0], poses[..., 1] - query[1]
    c, s = np.cos(query[2]), np.sin(query[2])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    dth = wrap(poses[..., 2] - query[2])
    return np.stack([lx / POS_SCALE, ly / POS_SCALE, np.sin(dth), np.cos(dth), np.ones_like(lx)], -1).astype(np.float32)
