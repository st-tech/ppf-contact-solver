# File: _scene_transform_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Quaternion / TRS helpers and TransformAnimation dataclass.

Split out of ``_scene_.py``: these helpers thin-wrap Rust kernels in
``_ppf_cts_py`` and back :class:`TransformAnimation`, the sparse rigid-body
animation used by static moving objects.
"""

from dataclasses import dataclass

import numpy as np

from . import _rust  # type: ignore[attr-defined]


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions (w,x,y,z)."""
    return _rust.scene_quat_slerp(
        [float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])],
        [float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])],
        float(t),
    )


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return _rust.scene_quat_to_mat3(
        [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    )


def _axis_angle_to_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Axis-angle to quaternion (w,x,y,z). Angle in degrees."""
    return _rust.scene_axis_angle_to_quat(
        [float(axis[0]), float(axis[1]), float(axis[2])], float(angle_deg)
    )


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (Hamilton product). Both in (w,x,y,z) format."""
    return _rust.scene_quat_multiply(
        [float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])],
        [float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])],
    )


def _mat3_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w,x,y,z)."""
    return _rust.scene_mat3_to_quat(np.ascontiguousarray(m, dtype=np.float64))


def _apply_transform_to_verts(
    local_vert: np.ndarray,
    translation: np.ndarray,
    quaternion: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply T * R * S transform to local vertices. Returns world positions."""
    return _rust.scene_apply_trs_to_verts(
        np.ascontiguousarray(local_vert, dtype=np.float64),
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ],
        [float(scale[0]), float(scale[1]), float(scale[2])],
    )


@dataclass
class TransformAnimation:
    """Sparse rigid-body transform animation for a static object.

    ``local_vert`` holds the pre-transform local geometry the T*R*S
    keyframes are applied to. For a normalized object it must already
    carry the normalize pre-step ``(v - center) / max(bbox)`` baked in,
    so ``evaluate(0)`` matches the object's built static base.
    """

    local_vert: np.ndarray
    times: list[float]
    translations: list[np.ndarray]
    quaternions: list[np.ndarray]
    scales: list[np.ndarray]

    def max_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    def evaluate(self, time: float) -> np.ndarray:
        """Compute world-space vertex positions at the given time."""
        translations = [
            [float(t[0]), float(t[1]), float(t[2])] for t in self.translations
        ]
        quaternions = [
            [float(q[0]), float(q[1]), float(q[2]), float(q[3])] for q in self.quaternions
        ]
        scales = [
            [float(s[0]), float(s[1]), float(s[2])] for s in self.scales
        ]
        return _rust.scene_transform_animation_evaluate(
            np.ascontiguousarray(self.local_vert, dtype=np.float64),
            [float(t) for t in self.times],
            translations, quaternions, scales, float(time),
        )
