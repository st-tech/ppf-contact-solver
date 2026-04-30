# File: _scene_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import colorsys
import json as _json
import os
import pickle
import shutil
import warnings


class ValidationError(ValueError):
    """ValueError with structured violation data for visualization."""

    def __init__(self, message, violations=None):
        super().__init__(message)
        self.violations = violations or []

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

from numba import njit, prange
from tqdm.auto import tqdm

from ._asset_ import AssetManager
from ._intersection_ import check_self_intersection
from ._invisible_collider_ import check_invisible_collider_violations
from ._param_ import ParamHolder, object_param
from ._plot_ import Plot, PlotManager
from ._proximity_ import check_contact_offset_violation
from ._render_ import MitsubaRenderer, Rasterizer
from ._utils_ import Utils

# Numba-optimized utility functions for scene building



@njit(parallel=True, cache=True)
def compute_face_to_vertex_counts(
    tris: np.ndarray,
    n_verts: int,
    out_counts: np.ndarray,
):
    """Count faces per vertex in parallel."""
    n_tris = len(tris)

    # Reset counts
    for i in prange(n_verts):
        out_counts[i] = 0

    # Count - need atomic operations, so use sequential for correctness
    for ti in range(n_tris):
        out_counts[tris[ti, 0]] += 1
        out_counts[tris[ti, 1]] += 1
        out_counts[tris[ti, 2]] += 1


@njit(parallel=True, cache=True)
def compute_face_to_vertex_weights(
    counts: np.ndarray,
    out_weights: np.ndarray,
    epsilon: float = 0.0001,
):
    """Compute inverse weights from counts in parallel."""
    n = len(counts)
    for i in prange(n):
        out_weights[i] = 1.0 / (counts[i] + epsilon)


EPS = 1e-3


class SceneManager:
    """SceneManager class. Use this to manage scenes.

    Example:
        Create a scene through the app's scene manager and build it::

            app = App.create("demo")
            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            fixed = scene.build()
    """

    def __init__(self, plot: Optional[PlotManager], asset: AssetManager):
        """Initialize the scene manager."""
        self._plot = plot
        self._asset = asset
        self._scene: dict[str, Scene] = {}

    def create(self, name: str = "") -> "Scene":
        """Create a new scene.

        If a scene with the given name already exists, it is replaced.

        Args:
            name (str): The name of the scene to create. If empty, defaults to ``"scene"``.

        Returns:
            Scene: The created scene.

        Example:
            Create two named scenes side by side::

                cloth_scene = app.scene.create("cloth")
                rods_scene = app.scene.create("rods")
        """
        if name == "":
            name = "scene"

        if name in self._scene:
            del self._scene[name]

        scene = Scene(name, self._plot, self._asset)
        self._scene[name] = scene
        return scene

    def select(self, name: str, create: bool = True) -> "Scene":
        """Select a scene.

        If the scene exists, it is returned. If it does not exist and ``create`` is True, a new scene is created and returned.

        Args:
            name (str): The name of the scene to select.
            create (bool, optional): Whether to create a new scene if it does not exist. Defaults to True.

        Returns:
            Scene: The selected (or newly created) scene.

        Example:
            Fetch an existing scene by name, creating it lazily::

                scene = app.scene.select("cloth")
                scene.add("sheet")
        """
        if create and name not in self._scene:
            return self.create(name)
        else:
            return self._scene[name]

    def remove(self, name: str):
        """Remove a scene from the manager.

        Args:
            name (str): The name of the scene to remove.

        Example:
            Drop a scene that is no longer needed::

                app.scene.remove("cloth")
        """
        if name in self._scene:
            del self._scene[name]

    def clear(self):
        """Clear all the scenes in the manager.

        Example:
            Remove every scene before rebuilding from scratch::

                app.scene.clear()
                scene = app.scene.create()
        """
        self._scene = {}

    def list(self) -> list[str]:
        """List all the scenes in the manager.

        Returns:
            list[str]: A list of scene names.

        Example:
            Inspect which scenes are currently registered::

                for name in app.scene.list():
                    print(name)
        """
        return list(self._scene.keys())


class WallParam:
    """A class to hold wall parameters."""

    def __init__(self):
        self._param = {
            "contact-gap": 1e-3,
            "friction": 0.0,
            "active-duration": -1.0,
            "thickness": 1.0,
        }

    def list(self) -> dict[str, float]:
        """List all the parameters for the wall.

        Returns:
            dict[str, float]: A dictionary of wall parameters.
        """
        return self._param

    def set(self, name: str, value: float) -> "WallParam":
        """Set a parameter for the wall.

        Args:
            name (str): The parameter name (must already exist in the defaults).
            value (float): The new value.

        Returns:
            WallParam: The updated wall parameters.
        """
        if name not in self._param:
            raise Exception(f"unknown parameter {name}")
        else:
            self._param[name] = value
        return self


class Wall:
    """An invisible wall class.

    Example:
        Box a trampoline in with four invisible walls::

            gap = 0.025
            scene.add.invisible.wall([1 + gap, 0, 0], [-1, 0, 0])
            scene.add.invisible.wall([-1 - gap, 0, 0], [1, 0, 0])
            scene.add.invisible.wall([0, 0, 1 + gap], [0, 0, -1])
            scene.add.invisible.wall([0, 0, -1 - gap], [0, 0, 1])
    """

    def __init__(self):
        """Initialize the wall."""
        self._entry = []
        self._transition = "linear"
        self._param = WallParam()

    def get_entry(self) -> list[tuple[list[float], float]]:
        """Get a list of time-dependent wall entries.

        Returns:
            list[tuple[list[float], float]]: A list of time-dependent entries, each containing a position and time.

        Example:
            Inspect every keyframed position on an animated wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.move_to([0, 0.2, 0], 2.0)
                for pos, t in wall.get_entry():
                    print(t, pos)
        """
        return self._entry

    def add(self, pos: list[float], normal: list[float]) -> "Wall":
        """Add the initial wall entry.

        Args:
            pos (list[float]): The position of the wall.
            normal (list[float]): The outer normal of the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Place a ground wall at y=0 with the outer normal facing up::

                wall = Wall().add([0, 0, 0], [0, 1, 0])
        """
        if len(self._entry):
            raise Exception("wall already exists")
        else:
            self._normal = normal
            self._entry.append((pos, 0.0))
            return self

    def _check_time(self, time: float):
        """Check if the time is valid.

        Args:
            time (float): The time to check.
        """
        if time <= self._entry[-1][1]:
            raise Exception("time must be greater than the last time")

    def move_by(self, delta: list[float], time: float) -> "Wall":
        """Move the wall by a positional delta at a specific time.

        Args:
            delta (list[float]): The positional delta to move the wall.
            time (float): The absolute time to move the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Raise the ground wall by 0.2 units at t=2 seconds::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.move_by([0, 0.2, 0], 2.0)
        """
        self._check_time(time)
        prev = self._entry[-1][0]
        pos = [prev[i] + delta[i] for i in range(len(prev))]
        self._entry.append((pos, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Wall":
        """Move the wall to an absolute position at a specific time.

        Args:
            pos (list[float]): The target position of the wall.
            time (float): The absolute time to move the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Animate a piston wall toward an absolute target::

                wall = scene.add.invisible.wall([0, 0, 0], [1, 0, 0])
                wall.move_to([0.5, 0, 0], 3.0)
        """
        self._check_time(time)
        self._entry.append((pos, time))
        return self

    def interp(self, transition: str) -> "Wall":
        """Set the transition type for the wall.

        Args:
            transition (str): The transition type. Defaults to ``"linear"`` if never set.

        Returns:
            Wall: The invisible wall.

        Example:
            Ease the wall motion with a smoothstep instead of a linear ramp::

                wall.move_to([0.5, 0, 0], 3.0).interp("smooth")
        """
        self._transition = transition
        return self

    @property
    def normal(self) -> list[float]:
        """Get the wall normal.

        Example:
            Read the outer normal back from an added wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                print(wall.normal)
        """
        return self._normal

    @property
    def entry(self) -> list[tuple[list[float], float]]:
        """Get the wall entries.

        Example:
            Iterate the wall animation keyframes::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                for pos, t in wall.entry:
                    print(t, pos)
        """
        return self._entry

    @property
    def transition(self) -> str:
        """Get the wall transition.

        Example:
            Check the interpolation mode of an animated wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0]).interp("smooth")
                assert wall.transition == "smooth"
        """
        return self._transition

    @property
    def param(self) -> WallParam:
        """Get the wall parameters.

        Example:
            Tweak a wall parameter through the returned holder::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.param.set("friction", 0.2)
        """
        return self._param


class SphereParam:
    """A class to hold sphere parameters."""

    def __init__(self):
        self._param = {
            "contact-gap": 1e-3,
            "friction": 0.0,
            "active-duration": -1.0,
            "thickness": 1.0,
        }

    def list(self) -> dict[str, float]:
        """List all the parameters for the sphere.

        Returns:
            dict[str, float]: A dictionary of sphere parameters.
        """
        return self._param

    def set(self, name: str, value: float) -> "SphereParam":
        """Set a parameter for the sphere.

        Args:
            name (str): The parameter name (must already exist in the defaults).
            value (float): The new value.

        Returns:
            SphereParam: The updated sphere parameters.
        """
        if name not in self._param:
            raise Exception(f"unknown parameter {name}")
        else:
            self._param[name] = value
        return self


class Sphere:
    """An invisible sphere class.

    Example:
        Add an inverted hemispherical bowl as a collider::

            scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
    """

    def __init__(self):
        """Initialize the sphere."""
        self._entry = []
        self._hemisphere = False
        self._invert = False
        self._transition = "linear"
        self._param = SphereParam()

    def hemisphere(self) -> "Sphere":
        """Turn the sphere into a hemisphere, so the top half becomes empty, like a bowl.

        Example:
            Cup-shaped collider by combining ``invert`` and ``hemisphere``::

                scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
        """
        self._hemisphere = True
        return self

    def invert(self) -> "Sphere":
        """Invert the sphere, so the inside becomes empty and the outside becomes solid.

        Example:
            Trap objects inside an inverted sphere boundary::

                scene.add.invisible.sphere([0, 0, 0], 2.0).invert()
        """
        self._invert = True
        return self

    def interp(self, transition: str) -> "Sphere":
        """Set the transition type for the sphere.

        Args:
            transition (str): The transition type. Defaults to ``"linear"`` if never set.

        Returns:
            Sphere: The sphere.

        Example:
            Ease the sphere motion with a smoothstep::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.move_to([0, 0.5, 0], 2.0).interp("smooth")
        """
        self._transition = transition
        return self

    def get_entry(self) -> list[tuple[list[float], float, float]]:
        """Get the time-dependent sphere entries.

        Example:
            Iterate over an animated sphere's keyframes (position, radius, time)::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.transform_to([0, 0.5, 0], 0.5, 2.0)
                for pos, r, t in sphere.get_entry():
                    print(t, r, pos)
        """
        return self._entry

    def add(self, pos: list[float], radius: float) -> "Sphere":
        """Add an invisible sphere information.

        Args:
            pos (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Instantiate a unit sphere at the origin::

                sphere = Sphere().add([0, 0, 0], 1.0)
        """
        if len(self._entry):
            raise Exception("sphere already exists")
        else:
            self._entry.append((pos, radius, 0.0))
            return self

    def _check_time(self, time: float):
        """Check if the time is valid.

        Args:
            time (float): The time to check.
        """
        if time <= self._entry[-1][2]:
            raise Exception(
                f"time must be greater than the last time. last time is {self._entry[-1][2]:f}"
            )

    def transform_to(self, pos: list[float], radius: float, time: float) -> "Sphere":
        """Change the sphere to a new position and radius at a specific time.

        Args:
            pos (list[float]): The target position of the sphere.
            radius (float): The target radius of the sphere.
            time (float): The absolute time to transform the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Grow the collider sphere as it slides into place::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.transform_to([0, 0.5, 0], 0.5, 2.0)
        """
        self._check_time(time)
        self._entry.append((pos, radius, time))
        return self

    def move_by(self, delta: list[float], time: float) -> "Sphere":
        """Move the sphere by a positional delta at a specific time.

        Args:
            delta (list[float]): The positional delta to move the sphere.
            time (float): The absolute time to move the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Slide a collider sphere 1 unit along +x by t=2::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.move_by([1, 0, 0], 2.0)
        """
        self._check_time(time)
        prev = self._entry[-1][0]
        pos = [prev[i] + delta[i] for i in range(len(prev))]
        radius = self._entry[-1][1]
        self._entry.append((pos, radius, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Sphere":
        """Move the sphere to an absolute position at a specific time.

        Args:
            pos (list[float]): The target position of the sphere.
            time (float): The absolute time to move the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Drop a ball from above to y=0 by t=1::

                sphere = scene.add.invisible.sphere([0, 1, 0], 0.25)
                sphere.move_to([0, 0, 0], 1.0)
        """
        self._check_time(time)
        radius = self._entry[-1][1]
        self._entry.append((pos, radius, time))
        return self

    def radius(self, radius: float, time: float) -> "Sphere":
        """Change the radius of the sphere at a specific time.

        Args:
            radius (float): The target radius of the sphere.
            time (float): The absolute time to change the radius.

        Returns:
            Sphere: The sphere.

        Example:
            Inflate the sphere from 0.25 to 0.5 by t=1::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.radius(0.5, 1.0)
        """
        self._check_time(time)
        pos = self._entry[-1][0]
        self._entry.append((pos, radius, time))
        return self

    @property
    def entry(self) -> list[tuple[list[float], float, float]]:
        """Get the sphere entries.

        Example:
            Inspect the keyframed (position, radius, time) list::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                for pos, r, t in sphere.entry:
                    print(t, r, pos)
        """
        return self._entry

    @property
    def is_hemisphere(self) -> bool:
        """Get whether sphere is hemisphere.

        Example:
            Check the hemisphere flag on a collider::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                assert sphere.is_hemisphere in (True, False)
        """
        return self._hemisphere

    @property
    def is_inverted(self) -> bool:
        """Get whether sphere is inverted.

        Example:
            An inverted sphere acts as a containing bowl::

                sphere = scene.add.invisible.sphere([0, 0, 0], 1.0, invert=True)
                assert sphere.is_inverted
        """
        return self._invert

    @property
    def transition(self) -> str:
        """Get the sphere transition.

        Example:
            Read back the interpolation mode::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5).interp("smooth")
                assert sphere.transition == "smooth"
        """
        return self._transition

    @property
    def param(self) -> SphereParam:
        """Get the sphere parameters.

        Example:
            Adjust a sphere parameter through the returned holder::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.param.set("friction", 0.2)
        """
        return self._param


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions (w,x,y,z)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _axis_angle_to_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Axis-angle to quaternion (w,x,y,z). Angle in degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (Hamilton product). Both in (w,x,y,z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _mat3_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w,x,y,z)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _apply_transform_to_verts(
    local_vert: np.ndarray,
    translation: np.ndarray,
    quaternion: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply T * R * S transform to local vertices. Returns world positions."""
    R = _quat_to_mat3(quaternion)
    S = np.diag(scale)
    return (local_vert @ (R @ S).T) + translation


@dataclass
class TransformAnimation:
    """Sparse rigid-body transform animation for a static object."""

    local_vert: np.ndarray
    times: list[float]
    translations: list[np.ndarray]
    quaternions: list[np.ndarray]
    scales: list[np.ndarray]

    def max_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    def evaluate(self, time: float) -> np.ndarray:
        """Compute world-space vertex positions at the given time."""
        if not self.times:
            return self.local_vert.copy()
        if time <= self.times[0]:
            return _apply_transform_to_verts(
                self.local_vert, self.translations[0],
                self.quaternions[0], self.scales[0],
            )
        if time >= self.times[-1]:
            return _apply_transform_to_verts(
                self.local_vert, self.translations[-1],
                self.quaternions[-1], self.scales[-1],
            )
        for i in range(len(self.times) - 1):
            if self.times[i] <= time < self.times[i + 1]:
                t = (time - self.times[i]) / (self.times[i + 1] - self.times[i])
                trans = (1 - t) * self.translations[i] + t * self.translations[i + 1]
                quat = _quat_slerp(self.quaternions[i], self.quaternions[i + 1], t)
                scl = (1 - t) * self.scales[i] + t * self.scales[i + 1]
                return _apply_transform_to_verts(self.local_vert, trans, quat, scl)
        return _apply_transform_to_verts(
            self.local_vert, self.translations[-1],
            self.quaternions[-1], self.scales[-1],
        )


@dataclass
class SpinData:
    """Represents spinning data for a set of vertices."""

    center: np.ndarray
    axis: np.ndarray
    angular_velocity: float
    t_start: float
    t_end: float


@dataclass
class PinKeyframe:
    """Represents a single keyframe for pinned vertices."""

    position: np.ndarray
    time: float


class Operation:
    """Base class for pin operations that can be applied in sequence."""

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Apply the operation to vertices at a given time.

        Args:
            vertex: Current vertex positions (may be transformed by previous operations).
            time: Current simulation time.

        Returns:
            Transformed vertex positions.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def get_time_range(self) -> tuple[float, float]:
        """Get the time range this operation is active.

        Returns:
            (t_start, t_end) tuple.
        """
        raise NotImplementedError("Subclasses must implement get_time_range()")


@dataclass
class MoveByOperation(Operation):
    """Move operation with position delta and time range."""

    delta: np.ndarray
    t_start: float
    t_end: float
    transition: str = "linear"
    bezier_handles: Optional[tuple] = None

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Apply position delta to current vertex position over time range."""
        if time < self.t_start:
            return vertex
        if time >= self.t_end:
            return vertex + self.delta
        progress = _eased_progress(
            time, self.t_start, self.t_end,
            self.transition, self.bezier_handles,
        )
        return vertex + self.delta * progress

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


def _bezier_progress(t: float, handles: tuple) -> float:
    """Evaluate cubic Bezier easing given normalized time t in [0,1].
    handles = ((hr_t, hr_y), (hl_t, hl_y)) — normalized control points.
    P0=(0,0), P1=(hr_t, hr_y), P2=(hl_t, hl_y), P3=(1,1).
    Solves B_x(u)=t via Newton, returns B_y(u).
    """
    hr, hl = handles
    p1x, p2x = hr[0], hl[0]
    p1y, p2y = hr[1], hl[1]
    u = t
    for _ in range(8):
        omu = 1.0 - u
        omu2 = omu * omu
        u2 = u * u
        u3 = u2 * u
        bx = 3.0 * omu2 * u * p1x + 3.0 * omu * u2 * p2x + u3
        dbx = 3.0 * omu2 * p1x + 6.0 * omu * u * (p2x - p1x) + 3.0 * u2 * (1.0 - p2x)
        if abs(dbx) < 1e-10:
            break
        u = u - (bx - t) / dbx
        u = max(0.0, min(1.0, u))
    omu = 1.0 - u
    omu2 = omu * omu
    u2 = u * u
    u3 = u2 * u
    return max(0.0, min(1.0, 3.0 * omu2 * u * p1y + 3.0 * omu * u2 * p2y + u3))


def _eased_progress(
    time: float, t_start: float, t_end: float,
    transition: str, bezier_handles: Optional[tuple],
) -> float:
    """Linear progress in [0,1] over [t_start, t_end], optionally eased."""
    progress = (time - t_start) / (t_end - t_start)
    if transition == "bezier" and bezier_handles is not None:
        return _bezier_progress(progress, bezier_handles)
    if transition == "smooth":
        return progress * progress * (3.0 - 2.0 * progress)
    return progress


@dataclass
class MoveToOperation(Operation):
    """Move operation with absolute target positions and time range."""

    target: np.ndarray
    t_start: float
    t_end: float
    transition: str = "linear"
    bezier_handles: Optional[tuple] = None

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Overwrite vertex positions with target over time range."""
        if time < self.t_start:
            return vertex
        if time >= self.t_end:
            return self.target.copy()
        progress = _eased_progress(
            time, self.t_start, self.t_end,
            self.transition, self.bezier_handles,
        )
        return vertex * (1 - progress) + self.target * progress

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class TransformKeyframeOperation(Operation):
    """Evaluate sparse TRS keyframes with slerp on rotation.

    Each call recomputes positions as ``R(t)*S(t)*local + T(t) - rest_T``.
    The ``-rest_T`` cancels the per-object displacement that
    ``Scene.time()`` re-adds after all ops, so the final vertex lands at
    the interpolated world pose with a rotation-aware slerp instead of a
    linear chord through the mesh.
    """

    local_vert: np.ndarray
    times: list
    translations: list
    quaternions: list
    scales: list
    segments: list
    rest_translation: np.ndarray

    def _eval(self, T, Q, S) -> np.ndarray:
        return _apply_transform_to_verts(self.local_vert, T, Q, S) - self.rest_translation

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        if not self.times:
            return vertex
        if time <= self.times[0]:
            return self._eval(self.translations[0], self.quaternions[0], self.scales[0])
        if time >= self.times[-1]:
            return self._eval(self.translations[-1], self.quaternions[-1], self.scales[-1])
        for i in range(len(self.times) - 1):
            t0, t1 = self.times[i], self.times[i + 1]
            if t0 <= time < t1:
                progress = (time - t0) / (t1 - t0)
                seg = self.segments[i]
                interp = seg.get("interpolation", "LINEAR")
                if interp == "BEZIER":
                    progress = _bezier_progress(
                        progress, (seg["handle_right"], seg["handle_left"])
                    )
                elif interp == "CONSTANT":
                    progress = 0.0
                elif interp != "LINEAR":
                    raise ValueError(
                        f"Unsupported interpolation '{interp}' in "
                        "transform_keyframes op"
                    )
                T = (1 - progress) * self.translations[i] + progress * self.translations[i + 1]
                Q = _quat_slerp(self.quaternions[i], self.quaternions[i + 1], progress)
                S = (1 - progress) * self.scales[i] + progress * self.scales[i + 1]
                return self._eval(T, Q, S)
        return self._eval(self.translations[-1], self.quaternions[-1], self.scales[-1])

    def get_time_range(self) -> tuple[float, float]:
        if not self.times:
            return (0.0, 0.0)
        return (self.times[0], self.times[-1])


@dataclass
class SpinOperation(Operation):
    """Spin operation with rotation parameters."""

    center: np.ndarray
    axis: np.ndarray
    angular_velocity: float
    t_start: float
    t_end: float
    center_mode: str = "absolute"

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Apply Rodrigues rotation if within time range."""
        t = min(time, self.t_end) - self.t_start
        if t <= 0:
            return vertex

        radian_velocity = self.angular_velocity / 180.0 * np.pi
        angle = radian_velocity * t
        axis = self.axis / np.linalg.norm(self.axis)

        # Rodrigues rotation formula
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        points = vertex - self.center
        rotated = (
            points * cos_theta
            + np.cross(axis, points) * sin_theta
            + np.outer(np.dot(points, axis), axis) * (1.0 - cos_theta)
        )
        return rotated + self.center

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class ScaleOperation(Operation):
    """Scale operation around a center point."""

    center: np.ndarray
    factor: float
    t_start: float
    t_end: float
    transition: str = "linear"
    center_mode: str = "absolute"

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Apply scaling interpolating from 1.0 to target factor over time range."""
        if time < self.t_start:
            return vertex

        if time >= self.t_end:
            # Apply full scale
            points = vertex - self.center
            return points * self.factor + self.center
        else:
            # Interpolate scale factor from 1.0 to target
            progress = (time - self.t_start) / (self.t_end - self.t_start)
            if self.transition == "smooth":
                progress = progress * progress * (3.0 - 2.0 * progress)

            current_factor = 1.0 + (self.factor - 1.0) * progress
            points = vertex - self.center
            return points * current_factor + self.center

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class TorqueOperation(Operation):
    """Torque operation — applies rotational force around a PCA axis."""

    axis_component: int  # 0=PC1, 1=PC2, 2=PC3
    magnitude: float
    hint_vertex: int = -1  # Blender vertex index for axis orientation hint
    t_start: float = 0.0
    t_end: float = float("inf")

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Torque is a force, not a kinematic operation. Returns vertex unchanged."""
        return vertex

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class PinData:
    """Represents pinning data for a set of vertices."""

    index: list[int]
    operations: list[Operation]
    unpin_time: Optional[float] = None
    transition: str = "linear"
    pull_strength: float = 0.0
    pin_group_id: str = ""
    # Static moving objects use the pin-shell as an implementation detail
    # (every vertex is pinned to drive a rigid-body motion). The user
    # never asked for these pins, so the preview should not render them
    # as pin markers.
    hide_in_preview: bool = False


class PinHolder:
    """Class to manage pinning behavior of objects."""

    _pin_counter = 0

    def __init__(self, obj: "Object", indices: list[int]):
        """Initialize pin object.

        Args:
            obj (Object): The object to pin.
            indices (list[int]): The indices of the vertices to pin.
        """
        self._obj = obj
        PinHolder._pin_counter += 1
        self._data = PinData(
            index=indices,
            operations=[],
            pin_group_id=f"{obj.name}:pin_{PinHolder._pin_counter}",
        )

    def interp(self, transition: str) -> "PinHolder":
        """Set the transition type for the pinning.

        Args:
            transition (str): The transition type. Supported values are ``"linear"``, ``"smooth"``, and ``"bezier"``. Default is ``"linear"``.

        Returns:
            PinHolder: The pinholder with the updated transition type.
        """
        self._data.transition = transition
        return self

    def unpin(self, time: float) -> "PinHolder":
        """Unpin the object at a specified time.

        Args:
            time (float): The time at which to unpin the vertices.

        Returns:
            PinHolder: The pinholder with the unpin time set.
        """
        if time < 0.0:
            raise Exception("unpin time must be non-negative")
        self._data.unpin_time = time
        return self

    def move_by(
        self, delta_pos, t_start: float = 0.0, t_end: float = 1.0,
        transition: Optional[str] = None, bezier_handles=None,
    ) -> "PinHolder":
        """Move the object by a positional delta over a specified time range.

        Args:
            delta_pos (list[float]): The positional delta to apply.
            t_start (float): The start time. Defaults to 0.0.
            t_end (float): The end time. Defaults to 1.0.
            transition (str, optional): Override interpolation type.
            bezier_handles: Bezier control points for easing.

        Returns:
            PinHolder: The pinholder with the updated position.

        Example:
            Slide a pinned sphere 8 units along +x between t=0 and t=5::

                scene.add("sphere").pin().move_by(
                    [8, 0, 0], t_start=0.0, t_end=5.0,
                )
        """
        delta_pos = np.array(delta_pos).reshape((-1, 3))

        if len(delta_pos) == 1 and len(self.index) > 1:
            delta_pos = np.tile(delta_pos, (len(self.index), 1))
        elif len(delta_pos) != len(self.index):
            raise Exception("delta_pos must have the same length as pin")

        if t_end <= t_start:
            raise Exception("t_end must be greater than t_start")

        self._data.operations.append(
            MoveByOperation(
                delta=delta_pos,
                t_start=t_start,
                t_end=t_end,
                transition=transition if transition else self._data.transition,
                bezier_handles=bezier_handles,
            )
        )
        return self

    def transform_keyframes(
        self,
        local_vert: np.ndarray,
        times: list,
        translations: list,
        quaternions: list,
        scales: list,
        segments: list,
        rest_translation: np.ndarray,
    ) -> "PinHolder":
        """Pin vertices to sparse TRS keyframes evaluated with slerp.

        Produces ``R(t)*S(t)*local + T(t) - rest_T`` per vertex, so a
        rotating object follows an arc instead of a linear chord.

        Per-segment ``interpolation`` is one of ``"LINEAR"``,
        ``"BEZIER"``, or ``"CONSTANT"``.

        Example:
            Animate a static (pinned) object with three TRS keyframes,
            a Bezier segment then a linear one::

                import numpy as np

                times        = [0.0, 1.0, 2.0]
                translations = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
                quaternions  = [[1, 0, 0, 0], [1, 0, 0, 0], [0.707, 0, 0.707, 0]]
                scales       = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                segments     = [
                    {"interpolation": "BEZIER"},
                    {"interpolation": "LINEAR"},
                ]

                obj = scene.add("sphere").pin()
                obj.transform_keyframes(
                    local_vert=obj.object.vertex(False),
                    times=times,
                    translations=translations,
                    quaternions=quaternions,
                    scales=scales,
                    segments=segments,
                    rest_translation=np.zeros(3),
                )
        """
        n = len(times)
        if len(translations) != n or len(quaternions) != n or len(scales) != n:
            raise ValueError(
                "transform_keyframes: T/Q/S arrays must match times length"
            )
        if n > 0 and len(segments) != n - 1:
            raise ValueError(
                f"transform_keyframes: expected {n - 1} segments, got {len(segments)}"
            )
        # Supported per-segment interpolation types.  The runtime (preview
        # and Rust simulator) respects exactly these; anything else would
        # silently degrade to LINEAR, which is surprising, so we reject up
        # front.
        allowed = {"LINEAR", "BEZIER", "CONSTANT"}
        for i, s in enumerate(segments):
            interp = s.get("interpolation", "LINEAR")
            if interp not in allowed:
                raise ValueError(
                    f"transform_keyframes segment {i}: unsupported "
                    f"interpolation '{interp}'. Supported: {sorted(allowed)}"
                )
        self._data.operations.append(
            TransformKeyframeOperation(
                local_vert=np.asarray(local_vert, dtype=np.float64),
                times=list(times),
                translations=[np.asarray(t, dtype=np.float64) for t in translations],
                quaternions=[np.asarray(q, dtype=np.float64) for q in quaternions],
                scales=[np.asarray(s, dtype=np.float64) for s in scales],
                segments=list(segments),
                rest_translation=np.asarray(rest_translation, dtype=np.float64),
            )
        )
        return self

    def move_to(
        self, target_pos, t_start: float = 0.0, t_end: float = 1.0,
        transition: Optional[str] = None, bezier_handles=None,
    ) -> "PinHolder":
        """Move the object to absolute target positions over a specified time range.

        Args:
            target_pos (list[float]): The target positions (absolute).
            t_start (float): The start time. Defaults to 0.0.
            t_end (float): The end time. Defaults to 1.0.
            transition (str, optional): Override interpolation type.
            bezier_handles: Bezier control points for easing ((hr_t, hr_v), (hl_t, hl_v)).

        Returns:
            PinHolder: The pinholder with the updated position.
        """
        target_pos = np.array(target_pos).reshape((-1, 3))

        if len(target_pos) == 1:
            initial_vertices = self._obj.vertex(False)[self._data.index]
            current_center = np.array(self._obj.position)
            delta = target_pos[0] - current_center
            target_pos = initial_vertices + delta
        elif len(target_pos) != len(self.index):
            raise Exception("target_pos must have the same length as pin")

        if t_end <= t_start:
            raise Exception("t_end must be greater than t_start")

        self._data.operations.append(
            MoveToOperation(
                target=target_pos,
                t_start=t_start,
                t_end=t_end,
                transition=transition if transition else self._data.transition,
                bezier_handles=bezier_handles,
            )
        )
        return self

    def scale(
        self,
        scale: float,
        t_start: float = 0.0,
        t_end: float = 1.0,
        center: Optional[list[float]] = None,
        center_mode: str = "absolute",
    ) -> "PinHolder":
        """Scale the object by a specified factor over a time range.

        Interpolates the scale factor from 1.0 to the target scale over [t_start, t_end].

        Args:
            scale (float): The target scaling factor.
            t_start (float): The start time of the scaling. Defaults to 0.0.
            t_end (float): The end time of the scaling. Defaults to 1.0.
            center (Optional[list[float]]): The center point for scaling. If not provided, uses the origin (0, 0, 0).
            center_mode (str): "centroid" to compute center from vertex positions at runtime, or "absolute" for fixed coordinate.

        Returns:
            PinHolder: The pinholder with the updated scaling.
        """
        center_point = np.array(center) if center is not None else np.zeros(3)

        self._data.operations.append(
            ScaleOperation(
                center=center_point,
                factor=scale,
                t_start=t_start,
                t_end=t_end,
                transition=self._data.transition,
                center_mode=center_mode,
            )
        )
        return self

    def pull(self, strength: float = 1.0) -> "PinHolder":
        """Set the pull strength applied to the pinned vertices.

        Args:
            strength (float, optional): The pull strength. Defaults to 1.0.

        Returns:
            PinHolder: The pinholder with the updated pull strength.
        """
        self._data.pull_strength = strength
        return self

    def spin(
        self,
        center: Optional[list[float]] = None,
        axis: Optional[list[float]] = None,
        angular_velocity: float = 360.0,
        t_start: float = 0.0,
        t_end: float = float("inf"),
        center_mode: str = "absolute",
    ) -> "PinHolder":
        """Add a spin operation to the pin.

        Args:
            center: Center of rotation. Defaults to [0, 0, 0].
            axis: Rotation axis. Defaults to [0, 1, 0].
            angular_velocity: Rotation speed in degrees/second.
            t_start: Start time of spin.
            t_end: End time of spin.
            center_mode: "centroid" to compute center from vertex positions at runtime, or "absolute" for fixed coordinate.

        Returns:
            PinHolder: The pinholder with the spin operation added.
        """
        if axis is None:
            axis = [0.0, 1.0, 0.0]
        if center is None:
            center = [0.0, 0.0, 0.0]

        self._data.operations.append(
            SpinOperation(
                center=np.array(center),
                axis=np.array(axis),
                angular_velocity=angular_velocity,
                t_start=t_start,
                t_end=t_end,
                center_mode=center_mode,
            )
        )
        return self

    def torque(
        self,
        magnitude: float = 1.0,
        axis_component: int = 2,
        hint_vertex: int = -1,
        t_start: float = 0.0,
        t_end: float = float("inf"),
    ) -> "PinHolder":
        """Add a torque operation to the pin.

        Applies a constant rotational force around a PCA-computed axis.
        The center is always the centroid of the pin vertices.

        Args:
            magnitude: Torque in N·m.
            axis_component: 0=PC1 (major), 1=PC2 (middle), 2=PC3 (minor).
            hint_vertex: Vertex index for axis orientation hint.
            t_start: Start time in seconds.
            t_end: End time in seconds.

        Returns:
            PinHolder: The pinholder with the torque operation added.
        """
        self._data.operations.append(
            TorqueOperation(
                axis_component=axis_component,
                magnitude=magnitude,
                hint_vertex=hint_vertex,
                t_start=t_start,
                t_end=t_end,
            )
        )
        return self

    @property
    def data(self) -> Optional[PinData]:
        """Get the pinning data.

        Returns:
            PinData: The pinning data.
        """
        return self._data

    @property
    def index(self) -> list[int]:
        """Get pinned vertex indices."""
        return self._data.index

    @property
    def operations(self) -> list[Operation]:
        """Get list of operations."""
        return self._data.operations

    @property
    def unpin_time(self) -> Optional[float]:
        """Get the time at which vertices should unpin."""
        return self._data.unpin_time

    @property
    def pull_strength(self) -> float:
        """Get pull force strength."""
        return self._data.pull_strength

    @property
    def transition(self) -> str:
        """Get the transition type."""
        return self._data.transition

    @property
    def pin_group_id(self) -> str:
        """Get the pin group identifier."""
        return self._data.pin_group_id


class EnumColor(Enum):
    """Dynamic face color enumeration."""

    NONE = 0
    AREA = 1


def _compute_triangle_areas_vectorized(vert: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Compute triangle areas using vectorized operations."""
    v0 = vert[tri[:, 0]]
    v1 = vert[tri[:, 1]]
    v2 = vert[tri[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def _compute_area(vert: np.ndarray, tri: np.ndarray, area: np.ndarray):
    """Compute areas for all triangles and store in the provided array."""
    area[:] = _compute_triangle_areas_vectorized(vert, tri)


def _compute_area_change(
    vert: np.ndarray, tri: np.ndarray, init_area: np.ndarray, rat: np.ndarray
):
    """Compute area change ratios for all triangles."""
    current_areas = _compute_triangle_areas_vectorized(vert, tri)
    rat[:] = current_areas / init_area


class FixedScene:
    """A fixed scene class.

    ``FixedScene`` is the immutable, validated result of :meth:`Scene.build`.
    Hand it to :meth:`SessionManager.create` to drive a solver run.

    Example:
        Build a scene, inspect it, then pass it into a session::

            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            fixed = scene.build().report()
            fixed.preview()
            session = app.session.create(fixed).build()
    """

    def __init__(
        self,
        plot: Optional[PlotManager],
        name: str,
        map_by_name: dict[str, list[int]],
        displacement: np.ndarray,
        vert: tuple[np.ndarray, np.ndarray],
        color: np.ndarray,
        dyn_face_color: list[EnumColor],
        dyn_face_intensity: list[float],
        vel: np.ndarray,
        uv: list[np.ndarray],
        rod: np.ndarray,
        tri: np.ndarray,
        tet: np.ndarray,
        rod_param: dict[str, list[Any]],
        tri_param: dict[str, list[Any]],
        tet_param: dict[str, list[Any]],
        wall: list[Wall],
        sphere: list[Sphere],
        rod_vert_range: tuple[int, int],
        shell_vert_range: tuple[int, int],
        rod_count: int,
        shell_count: int,
        tri_is_collider: np.ndarray,
        rod_is_collider: np.ndarray,
        pinned_vertices: Optional[set[int]] = None,
        static_vert_for_check: Optional[np.ndarray] = None,
        static_tri_for_check: Optional[np.ndarray] = None,
        surface_map_by_name: Optional[dict[str, tuple]] = None,
        concat_rest_vert: Optional[np.ndarray] = None,
        rest_vert_mask: Optional[np.ndarray] = None,
    ):
        """Initialize the fixed scene.

        Args:
            plot (Optional[PlotManager]): The plot manager.
            name (str): The name of the scene.
            map_by_name (dict[str, list[int]]): Mapping from object name to per-object vertex index arrays.
            displacement (np.ndarray): The per-object displacement vectors.
            vert (tuple[np.ndarray, np.ndarray]): The vertices of the scene. The first array is the displacement map reference, the second is local positions.
            color (np.ndarray): The colors of the vertices.
            dyn_face_color (list[EnumColor]): The dynamic face colors.
            dyn_face_intensity (list[float]): The dynamic face color intensities.
            vel (np.ndarray): The velocities of the vertices.
            uv (list[np.ndarray]): The per-face UV coordinates for shell faces.
            rod (np.ndarray): The rod elements.
            tri (np.ndarray): The triangle elements.
            tet (np.ndarray): The tetrahedral elements.
            rod_param (dict[str, list[Any]]): The parameters for the rod elements.
            tri_param (dict[str, list[Any]]): The parameters for the triangle elements.
            tet_param (dict[str, list[Any]]): The parameters for the tetrahedral elements.
            wall (list[Wall]): The invisible walls.
            sphere (list[Sphere]): The invisible spheres.
            rod_vert_range (tuple[int, int]): The index range of the rod vertices.
            shell_vert_range (tuple[int, int]): The index range of the shell vertices.
            rod_count (int): The number of rod elements.
            shell_count (int): The number of shell elements.
            tri_is_collider (np.ndarray): Boolean array indicating collider triangles.
            rod_is_collider (np.ndarray): Boolean array indicating collider rod edges.
            pinned_vertices (Optional[set[int]]): Set of pinned vertex indices. Used to skip checking pinned vertices against invisible colliders.
            static_vert_for_check (Optional[np.ndarray]): Static mesh vertices for intersection check.
            static_tri_for_check (Optional[np.ndarray]): Static mesh triangles for intersection check.
            surface_map_by_name (Optional[dict[str, tuple]]): Frame-embedding surface maps for tetrahedralized objects.
            concat_rest_vert (Optional[np.ndarray]): Concatenated rest-shape vertices for objects whose pins release at ``unpin_time``.
            rest_vert_mask (Optional[np.ndarray]): Per-vertex uint8 mask marking entries in ``concat_rest_vert`` that are valid.
        """

        self._map_by_name = map_by_name
        self._plot = plot
        self._name = name
        self._displacement = displacement
        self._vert = vert
        self._color = color
        self._dyn_face_color = dyn_face_color
        self._dyn_face_intensity = dyn_face_intensity
        self._vel = vel
        self._velocity_schedules = {}
        self._collision_windows_data = {}
        self._uv = uv
        self._rod = rod
        self._tri = tri
        self._tet = tet
        self._rod_param = rod_param
        self._tri_param = tri_param
        self._tet_param = tet_param
        self._concat_rest_vert = concat_rest_vert
        self._rest_vert_mask = rest_vert_mask
        self._pin: list[PinData] = []
        self._spin: list[SpinData] = []
        self._static_vert = (np.zeros(0, dtype=np.uint32), np.zeros(0))
        self._static_color = np.zeros((0, 0))
        self._static_tri = np.zeros((0, 0))
        self._stitch_ind = np.zeros((0, 0))
        self._stitch_w = np.zeros((0, 0))
        self._static_param = {}
        self._static_transform_animations: list[tuple[int, TransformAnimation]] = []
        self._excluded_from_output: set[str] = set()
        self._wall = wall
        self._sphere = sphere
        self._rod_vert_range = rod_vert_range
        self._shell_vert_range = shell_vert_range
        self._rod_count = rod_count
        self._shell_count = shell_count
        self._has_dyn_color = any(entry != EnumColor.NONE for entry in dyn_face_color)

        self._surface_map_by_name = surface_map_by_name

        # Violation flags - set during validation checks
        self._has_self_intersection = False
        self._has_contact_offset_violation = False
        self._has_wall_violation = False
        self._has_sphere_violation = False

        assert len(self._vert[0]) == len(self._color)
        assert len(self._vert[1]) == len(self._color)
        assert len(self._tri) == len(self._dyn_face_color)
        assert len(self._uv) == shell_count

        for key, value in self._rod_param.items():
            if value:
                assert len(value) == len(self._rod), (
                    f"{key} has {len(value)} entries, but rod has {len(self._rod)} rods"
                )
        for key, value in self._tri_param.items():
            if value:
                assert len(value) == len(self._tri), (
                    f"{key} has {len(value)} entries, but tri has {len(self._tri)} faces"
                )
        for key, value in self._tet_param.items():
            if value:
                assert len(value) == len(self._tet), (
                    f"{key} has {len(value)} entries, but tet has {len(self._tet)} tets"
                )

        # Check for self-intersections (including static colliders)
        if len(self._tri) > 0:
            dynamic_verts = self._vert[1] + self._displacement[self._vert[0]]

            # Combine dynamic and static meshes for intersection check
            if (
                static_vert_for_check is not None
                and static_tri_for_check is not None
                and len(static_tri_for_check) > 0
            ):
                n_dyn_verts = len(dynamic_verts)
                combined_verts = np.vstack([dynamic_verts, static_vert_for_check])
                combined_tris = np.vstack(
                    [self._tri, static_tri_for_check + n_dyn_verts]
                )
                # Static triangles are all colliders (fully pinned)
                static_is_collider = np.ones(len(static_tri_for_check), dtype=bool)
                combined_is_collider = np.concatenate(
                    [tri_is_collider, static_is_collider]
                )
            else:
                combined_verts = dynamic_verts
                combined_tris = self._tri
                combined_is_collider = tri_is_collider

            rod_edges = self._rod if len(self._rod) > 0 else None
            intersection_pairs = check_self_intersection(
                combined_verts, combined_tris, combined_is_collider,
                rod_edges=rod_edges, verbose=True,
            )
            if len(intersection_pairs) > 0:
                self._has_self_intersection = True

        # Check rod contact-offset constraints
        n_tris = len(self._tri)
        n_rods = len(self._rod)
        tri_offset = self._tri_param.get("contact-offset", [])
        rod_offset = self._rod_param.get("contact-offset", [])
        if rod_offset and n_rods > 0 and n_tris > 0:
            dv = self._vert[1] + self._displacement[self._vert[0]]
            # Precompute triangle data for proximity check
            tri_v0 = dv[self._tri[:, 0]]
            tri_v1 = dv[self._tri[:, 1]]
            tri_v2 = dv[self._tri[:, 2]]
            tri_offset_arr = np.array(tri_offset, dtype=np.float64) if tri_offset else np.zeros(n_tris)
            for ri, rod in enumerate(self._rod):
                offset = rod_offset[ri] if ri < len(rod_offset) else 0
                if offset <= 0:
                    continue
                edge_vec = dv[rod[1]] - dv[rod[0]]
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len <= offset:
                    raise ValueError(
                        f"Contact offset ({offset:.4f}) exceeds rod edge "
                        f"length ({edge_len:.4f}). Reduce the offset or "
                        f"increase mesh resolution."
                    )
                # Check each endpoint against all triangles
                for vi in [rod[0], rod[1]]:
                    p = dv[vi]
                    v0p = p - tri_v0
                    e0 = tri_v1 - tri_v0
                    e1 = tri_v2 - tri_v0
                    d00 = np.sum(e0 * e0, axis=1)
                    d01 = np.sum(e0 * e1, axis=1)
                    d11 = np.sum(e1 * e1, axis=1)
                    d20 = np.sum(v0p * e0, axis=1)
                    d21 = np.sum(v0p * e1, axis=1)
                    denom = d00 * d11 - d01 * d01
                    valid = np.abs(denom) > 1e-30
                    b = np.where(valid, (d11 * d20 - d01 * d21) / np.where(valid, denom, 1), 0)
                    c = np.where(valid, (d00 * d21 - d01 * d20) / np.where(valid, denom, 1), 0)
                    a = 1.0 - b - c
                    a = np.clip(a, 0, 1)
                    b = np.clip(b, 0, 1)
                    c = np.clip(c, 0, 1)
                    s = a + b + c
                    s = np.where(s > 0, s, 1)
                    a /= s; b /= s; c /= s
                    proj = a[:, None] * tri_v0 + b[:, None] * tri_v1 + c[:, None] * tri_v2
                    dist = np.linalg.norm(p - proj, axis=1)
                    required = offset + tri_offset_arr
                    violations = np.where(valid & (dist < required))[0]
                    if len(violations) > 0:
                        ti = violations[0]
                        raise ValueError(
                            f"Rod vertex within contact offset of triangle "
                            f"(dist={dist[ti]:.4f} < offset={required[ti]:.4f}). "
                            f"Reduce contact-offset or move the curve away."
                        )

        # Check for contact-offset separation violations (excluding collider elements)
        has_tri_offset = tri_offset and any(o > 0 for o in tri_offset)
        has_rod_offset = rod_offset and any(o > 0 for o in rod_offset)

        contact_violations = []
        has_elements = n_tris > 0 or n_rods > 0
        if has_elements and (has_tri_offset or has_rod_offset or n_rods > 0):
            dynamic_verts = self._vert[1] + self._displacement[self._vert[0]]

            combined_is_collider = np.concatenate(
                [
                    tri_is_collider
                    if len(tri_is_collider)
                    else np.zeros(0, dtype=bool),
                    rod_is_collider
                    if len(rod_is_collider)
                    else np.zeros(0, dtype=bool),
                ]
            )

            tri_offset_arr = (
                np.array(tri_offset, dtype=np.float64)
                if tri_offset
                else np.zeros(n_tris, dtype=np.float64)
            )
            rod_offset_arr = (
                np.array(rod_offset, dtype=np.float64)
                if rod_offset
                else np.zeros(n_rods, dtype=np.float64)
            )
            combined_offset = np.concatenate([tri_offset_arr, rod_offset_arr])

            contact_violations = check_contact_offset_violation(
                dynamic_verts,
                F=self._tri if n_tris > 0 else None,
                E=self._rod if n_rods > 0 else None,
                is_collider=combined_is_collider,
                contact_offset=combined_offset,
                verbose=True,
            )
            if len(contact_violations) > 0:
                self._has_contact_offset_violation = True

        # Check for invisible collider violations (walls and spheres)
        wall_violations = []
        sphere_violations = []
        if wall or sphere:
            dynamic_verts = self._vert[1] + self._displacement[self._vert[0]]
            wall_violations, sphere_violations = check_invisible_collider_violations(
                dynamic_verts,
                wall,
                sphere,
                pinned_vertices,
                verbose=True,
            )
            if wall_violations:
                self._has_wall_violation = True
            if sphere_violations:
                self._has_sphere_violation = True

        # Collect all violations and raise once with structured data
        all_violations = []
        messages = []

        if self._has_self_intersection:
            si_dynamic = self._vert[1] + self._displacement[self._vert[0]]
            si_tris_data = []
            n_tri_tri = 0
            n_rod_tri = 0
            for p in intersection_pairs[:100]:
                tri_positions = []
                is_rod = p[0] == -1
                for ti in p:
                    if 0 <= ti < len(self._tri):
                        t = self._tri[ti]
                        tri_positions.append(si_dynamic[t].tolist())
                si_tris_data.append(tri_positions)
                if is_rod:
                    n_rod_tri += 1
                else:
                    n_tri_tri += 1
            all_violations.append({
                "type": "self_intersection",
                "tris": si_tris_data,
                "count": len(intersection_pairs),
            })
            parts = []
            if n_tri_tri > 0:
                parts.append(f"{n_tri_tri} tri-tri")
            if n_rod_tri > 0:
                parts.append(f"{n_rod_tri} rod-tri")
            messages.append(
                f"{len(intersection_pairs)} self-intersections ({', '.join(parts)})."
            )

        if self._has_contact_offset_violation:
            dynamic_verts = self._vert[1] + self._displacement[self._vert[0]]
            pairs_data = []
            for ei, ej in contact_violations[:100]:
                ei_type = "triangle" if ei < n_tris else "edge"
                ej_type = "triangle" if ej < n_tris else "edge"
                # Include world-space vertex positions for visualization
                ei_pos = []
                if ei_type == "triangle":
                    t = self._tri[ei]
                    ei_pos = dynamic_verts[t].tolist()
                elif ei < n_tris + len(self._rod):
                    e = self._rod[ei - n_tris]
                    ei_pos = dynamic_verts[e].tolist()
                ej_pos = []
                if ej_type == "triangle":
                    t = self._tri[ej]
                    ej_pos = dynamic_verts[t].tolist()
                elif ej < n_tris + len(self._rod):
                    e = self._rod[ej - n_tris]
                    ej_pos = dynamic_verts[e].tolist()
                pairs_data.append({
                    "ei_type": ei_type, "ej_type": ej_type,
                    "ei_pos": ei_pos, "ej_pos": ej_pos,
                })
            all_violations.append({
                "type": "contact_offset",
                "pairs": pairs_data,
                "count": len(contact_violations),
            })
            first_ei, first_ej = contact_violations[0]
            messages.append(
                f"{len(contact_violations)} element pairs too close."
            )

        if wall_violations:
            wv_dynamic = self._vert[1] + self._displacement[self._vert[0]]
            all_violations.append({
                "type": "wall",
                "vertices": [
                    {"pos": wv_dynamic[vi].tolist(), "wall": int(wi), "dist": float(d)}
                    for vi, wi, d in wall_violations[:100]
                ],
                "count": len(wall_violations),
            })
            vi, wall_idx, signed_dist = wall_violations[0]
            messages.append(
                f"{len(wall_violations)} vertices violate wall constraints. "
                f"First: vertex {vi}, {-signed_dist:.6f} units wrong side of wall {wall_idx}."
            )

        if sphere_violations:
            sv_dynamic = self._vert[1] + self._displacement[self._vert[0]]
            all_violations.append({
                "type": "sphere",
                "vertices": [
                    {"pos": sv_dynamic[vi].tolist(), "sphere": int(si), "dist": float(d)}
                    for vi, si, d in sphere_violations[:100]
                ],
                "count": len(sphere_violations),
            })
            vi, sphere_idx, dist = sphere_violations[0]
            messages.append(
                f"{len(sphere_violations)} vertices violate sphere constraints. "
                f"First: vertex {vi}, sphere {sphere_idx}."
            )

        if all_violations:
            raise ValidationError(" | ".join(messages), violations=all_violations)

        if len(self._tri):
            self._area = np.zeros(len(self._tri))
            _compute_area(self._vert[1], self._tri, self._area)
        else:
            self._area = np.zeros(0)

        if self._has_dyn_color:
            # Compute vertex weights for face-to-vertex averaging (numba-optimized)
            n_verts = len(self._vert[0])
            tri_arr = np.ascontiguousarray(self._tri, dtype=np.int64)
            vert_face_count = np.zeros(n_verts, dtype=np.float64)
            compute_face_to_vertex_counts(tri_arr, n_verts, vert_face_count)
            self._face_to_vert_weights = np.zeros(n_verts, dtype=np.float64)
            compute_face_to_vertex_weights(vert_face_count, self._face_to_vert_weights)
        else:
            self._face_to_vert_weights = None

    @property
    def tri_param(self) -> dict[str, list[Any]]:
        """Get the triangle parameters.

        Example:
            Inspect per-triangle parameter arrays captured at build time::

                fixed = scene.build()
                for key, values in fixed.tri_param.items():
                    print(key, len(values))
        """
        return self._tri_param

    @property
    def has_violations(self) -> bool:
        """Check if the scene has any violations that prevent simulation.

        Example:
            Guard the solver launch behind a validation check::

                fixed = scene.build()
                if fixed.has_violations:
                    print(fixed.get_violation_messages())
        """
        return (
            self._has_self_intersection
            or self._has_contact_offset_violation
            or self._has_wall_violation
            or self._has_sphere_violation
        )

    def get_violation_messages(self) -> list[str]:
        """Get a list of violation messages for the scene.

        Example:
            Print any validation violations before running the solver::

                for msg in fixed.get_violation_messages():
                    print(msg)
        """
        messages = []
        if self._has_self_intersection:
            messages.append("Scene has self-intersections")
        if self._has_contact_offset_violation:
            messages.append("Scene has contact-offset violations")
        if self._has_wall_violation:
            messages.append("Scene has wall constraint violations")
        if self._has_sphere_violation:
            messages.append("Scene has sphere constraint violations")
        return messages

    def report(self) -> "FixedScene":
        """Print a summary of the scene.

        Returns:
            FixedScene: The fixed scene (for chaining).

        Example:
            Chain a summary printout into the build step::

                fixed = scene.build().report()
                fixed.preview()
        """
        data = {}
        data["#vert"] = len(self._vert[1])
        if len(self._rod):
            data["#rod"] = len(self._rod)
        if len(self._tri):
            data["#tri"] = len(self._tri)
        if len(self._tet):
            data["#tet"] = len(self._tet)
        if len(self._pin):
            data["#pin"] = sum([len(pin.index) for pin in self._pin])
        if len(self._static_vert) and len(self._static_tri):
            data["#static_vert"] = len(self._static_vert[1])
            data["#static_tri"] = len(self._static_tri)
        if len(self._stitch_ind) and len(self._stitch_w):
            data["#stitch_ind"] = len(self._stitch_ind)
        for key, value in data.items():
            if isinstance(value, int):
                data[key] = [f"{value:,}"]
            elif isinstance(value, float):
                data[key] = [f"{value:.2e}"]
            else:
                data[key] = [str(value)]

        from IPython.display import HTML, display

        from ._utils_ import dict_to_html_table

        if self._plot is not None and self._plot.is_jupyter_notebook():
            html = dict_to_html_table(data, classes="table")
            display(HTML(html))
        else:
            print(data)
        return self

    def color(self, vert: np.ndarray, hint: Optional[dict] = None) -> np.ndarray:
        """Compute the per-vertex color for the scene given a vertex array.

        Args:
            vert (np.ndarray): The current vertex positions.
            hint (dict, optional): Optional hints for the color computation (e.g. ``"max-area"``). Defaults to None.

        Returns:
            np.ndarray: The per-vertex colors of the scene.

        Example:
            Compute colors for the current vertex positions before previewing::

                fixed = scene.build()
                V = fixed.vertex()
                colors = fixed.color(V, hint={"max-area": 2.0})
                fixed.preview(V)
        """
        if hint is None:
            hint = {}
        if self._has_dyn_color:
            assert self._face_to_vert_weights is not None
            assert self._area is not None

            max_area = 2.0

            if "max-area" in hint:
                max_area = hint["max-area"]

            rat = np.zeros(len(self._tri))
            face_color = np.zeros((len(self._tri), 3))
            intensity = np.zeros(len(self._tri))
            _compute_area_change(vert, self._tri, self._area, rat)

            for i in range(len(face_color)):
                if self._dyn_face_color[i] != EnumColor.NONE:
                    val = max(0.0, min(1.0, (rat[i] - 1.0) / (max_area - 1.0)))
                    intensity[i] = self._dyn_face_intensity[i]
                    hue = 240.0 * (1.0 - val) / 360.0
                    face_color[i] = np.array(colorsys.hsv_to_rgb(hue, 0.75, 1.0))

            # Face-to-vertex averaging (replaces sparse matrix dot product)
            num_verts = len(self._face_to_vert_weights)
            vert_intensity = np.zeros(num_verts)
            np.add.at(vert_intensity, self._tri.ravel(), np.repeat(intensity, 3))
            vert_intensity *= self._face_to_vert_weights

            vert_face_color = np.zeros((num_verts, 3))
            np.add.at(
                vert_face_color, self._tri.ravel(), np.repeat(face_color, 3, axis=0)
            )
            vert_face_color *= self._face_to_vert_weights[:, None]

            color = (1.0 - vert_intensity[:, None]) * self._color + vert_intensity[
                :, None
            ] * vert_face_color
            return color
        else:
            return self._color

    def vertex(self, transform: bool = True) -> np.ndarray:
        """Get the vertices of the scene.

        Args:
            transform (bool, optional): Whether to transform the vertices. Defaults to True.

        Returns:
            np.ndarray: The vertices of the scene.

        Example:
            Fetch the initial world-space vertex positions::

                vert = fixed.vertex()
                print(vert.shape)
        """
        if transform:
            return self._vert[1] + self._displacement[self._vert[0]]
        else:
            return self._vert[1]

    def export(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        path: str,
        include_static: bool = True,
        args: Optional[dict] = None,
        delete_exist: bool = False,
    ) -> "FixedScene":
        """Export the scene to a mesh file.

        The vertices and vertex colors must be supplied explicitly so callers
        can pass time-evaluated positions.

        Args:
            vert (np.ndarray): The vertices of the scene.
            color (np.ndarray): The colors of the vertices.
            path (str): The path to the mesh file. Supported formats include ``.ply`` and ``.obj``.
            include_static (bool, optional): Whether to include the static mesh. Defaults to True.
            args (dict, optional): Additional arguments passed to the renderer.
            delete_exist (bool, optional): Whether to delete any existing file at the path. Defaults to False.

        Returns:
            FixedScene: The fixed scene.

        Example:
            Write out the scene as a .ply at the initial time::

                vert = fixed.vertex()
                color = fixed.color(vert)
                fixed.export(vert, color, "/tmp/scene.ply", delete_exist=True)
        """

        if args is None:
            args = {}
        image_path = path + ".png"
        if delete_exist:
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(image_path):
                os.remove(image_path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        seg, tri = self._rod, None
        if not os.path.exists(path) or not os.path.exists(image_path):
            if include_static and len(self._static_vert) and len(self._static_tri):
                static_vert = (
                    self._static_vert[1] + self._displacement[self._static_vert[0]]
                )
                tri = np.concatenate([self._tri, self._static_tri + len(vert)])
                vert = np.concatenate([vert, static_vert], axis=0)
                color = np.concatenate([color, self._static_color], axis=0)
            else:
                tri = self._tri

        if tri is not None and len(tri) == 0:
            tri = np.array([[0, 0, 0]])

        # Check if rendering should be skipped (e.g., on Windows headless)
        skip_render = args.get("skip_render", False)

        # Export mesh file (also in CI mode when skip_render is set)
        if not os.path.exists(path) and (Utils.ci_name() is None or skip_render):
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=vert, faces=tri, vertex_colors=color, process=False
            )
            mesh.export(path)

        # Skip rendering if skip_render is set
        if not skip_render and not os.path.exists(image_path):
            renderer_type = args.get("renderer", "software")
            if Utils.ci_name() is not None:
                args["width"] = 320
                args["height"] = 240
            if renderer_type == "mitsuba":
                assert shutil.which("mitsuba") is not None
                renderer = MitsubaRenderer(args)
            elif renderer_type == "software":
                renderer = Rasterizer(args)
            else:
                raise Exception("unsupported renderer")

            assert tri is not None
            assert color is not None
            renderer.render(vert, color, seg, tri, image_path)

        return self

    def export_fixed(self, path: str, delete_exist: bool) -> "FixedScene":
        """Export the fixed scene into a set of data files that are read by the simulator.

        Args:
            path (str): The path to the output directory.
            delete_exist (bool): Whether to delete the existing directory.

        Returns:
            FixedScene: The fixed scene.

        Example:
            Typically invoked internally by :meth:`Session.build`, but can be
            called directly to materialize the solver's data directory::

                fixed = scene.build()
                fixed.export_fixed("/tmp/my_scene_data", delete_exist=True)
        """

        steps = 14
        pbar = tqdm(total=steps, desc="build session")

        if os.path.exists(path):
            if delete_exist:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            else:
                raise Exception(f"file {path} already exists")
        else:
            os.makedirs(path)
        pbar.update(1)

        map_path = os.path.join(path, "map.pickle")
        with open(map_path, "wb") as f:
            exported_map = {k: v for k, v in self._map_by_name.items()
                           if k not in self._excluded_from_output}
            pickle.dump(exported_map, f)
        if self._surface_map_by_name:
            # Wire format v2: frame-embedding coefs replace barycentric weights.
            # Wrapped in a versioned envelope so legacy clients (which expected
            # a bare dict of bary maps) fail loudly instead of silently
            # applying the wrong reconstruction math.
            surface_map_path = os.path.join(path, "surface_map.pickle")
            with open(surface_map_path, "wb") as f:
                pickle.dump(
                    {"version": 2, "maps": self._surface_map_by_name}, f
                )
        pbar.update(1)

        info_path = os.path.join(path, "info.toml")
        with open(info_path, "w") as f:
            f.write("[count]\n")
            f.write(f"vert = {len(self._vert[1])}\n")
            f.write(f"rod = {len(self._rod)}\n")
            f.write(f"tri = {len(self._tri)}\n")
            f.write(f"tet = {len(self._tet)}\n")
            f.write(f"static_vert = {len(self._static_vert[1])}\n")
            f.write(f"static_tri = {len(self._static_tri)}\n")
            f.write(f"pin_block = {len(self._pin)}\n")
            f.write(f"wall = {len(self._wall)}\n")
            f.write(f"sphere = {len(self._sphere)}\n")
            f.write(f"stitch = {len(self._stitch_ind)}\n")
            f.write(f"rod_vert_start = {self._rod_vert_range[0]}\n")
            f.write(f"rod_vert_end = {self._rod_vert_range[1]}\n")
            f.write(f"shell_vert_start = {self._shell_vert_range[0]}\n")
            f.write(f"shell_vert_end = {self._shell_vert_range[1]}\n")
            f.write(f"rod_count = {self._rod_count}\n")
            f.write(f"shell_count = {self._shell_count}\n")
            f.write(f"has_rest_vert = {str(self._concat_rest_vert is not None).lower()}\n")
            f.write("\n")

            if self._static_transform_animations:
                total_kf = sum(len(a.times) for _, a in self._static_transform_animations)
                f.write("[static_transform]\n")
                f.write(f"object_count = {len(self._static_transform_animations)}\n")
                f.write(f"total_keyframes = {total_kf}\n")
                kf_counts = [len(a.times) for _, a in self._static_transform_animations]
                f.write(f"keyframe_counts = {kf_counts}\n")
                vert_counts = [len(a.local_vert) for _, a in self._static_transform_animations]
                f.write(f"vert_counts = {vert_counts}\n")
                vert_offsets = [off for off, _ in self._static_transform_animations]
                f.write(f"vert_offsets = {vert_offsets}\n")
                f.write("\n")

            for i, pin in enumerate(self._pin):
                f.write(f"[pin-{i}]\n")
                f.write(f"operation_count = {len(pin.operations)}\n")
                f.write(f"pin = {len(pin.index)}\n")
                f.write(f"pull = {float(pin.pull_strength)}\n")
                if pin.unpin_time is not None:
                    f.write(f"unpin_time = {float(pin.unpin_time)}\n")
                if pin.pin_group_id:
                    f.write(f'pin_group_id = "{pin.pin_group_id}"\n')
                f.write("\n")

                # Write operation metadata
                for j, op in enumerate(pin.operations):
                    f.write(f"[pin-{i}-op-{j}]\n")
                    if isinstance(op, MoveByOperation):
                        f.write('type = "move_by"\n')
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                        f.write(f'transition = "{op.transition}"\n')
                    elif isinstance(op, MoveToOperation):
                        f.write('type = "move_to"\n')
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                        f.write(f'transition = "{op.transition}"\n')
                    elif isinstance(op, SpinOperation):
                        f.write('type = "spin"\n')
                        f.write(f'center_mode = "{op.center_mode}"\n')
                        f.write(f"center_x = {float(op.center[0])}\n")
                        f.write(f"center_y = {float(op.center[1])}\n")
                        f.write(f"center_z = {float(op.center[2])}\n")
                        f.write(f"axis_x = {float(op.axis[0])}\n")
                        f.write(f"axis_y = {float(op.axis[1])}\n")
                        f.write(f"axis_z = {float(op.axis[2])}\n")
                        f.write(f"angular_velocity = {float(op.angular_velocity)}\n")
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                    elif isinstance(op, ScaleOperation):
                        f.write('type = "scale"\n')
                        f.write(f'center_mode = "{op.center_mode}"\n')
                        f.write(f"center_x = {float(op.center[0])}\n")
                        f.write(f"center_y = {float(op.center[1])}\n")
                        f.write(f"center_z = {float(op.center[2])}\n")
                        f.write(f"factor = {float(op.factor)}\n")
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                        f.write(f'transition = "{op.transition}"\n')
                    elif isinstance(op, TorqueOperation):
                        f.write('type = "torque"\n')
                        f.write(f"axis_component = {int(op.axis_component)}\n")
                        f.write(f"magnitude = {float(op.magnitude)}\n")
                        f.write(f"hint_vertex = {int(op.hint_vertex)}\n")
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                    elif isinstance(op, TransformKeyframeOperation):
                        f.write('type = "transform_keyframes"\n')
                        f.write(f"keyframe_count = {len(op.times)}\n")
                        f.write(f"t_start = {float(op.times[0]) if op.times else 0.0}\n")
                        f.write(f"t_end = {float(op.times[-1]) if op.times else 0.0}\n")
                        rt = op.rest_translation
                        f.write(f"rest_tx = {float(rt[0])}\n")
                        f.write(f"rest_ty = {float(rt[1])}\n")
                        f.write(f"rest_tz = {float(rt[2])}\n")
                    f.write("\n")

            for i, wall in enumerate(self._wall):
                normal = wall.normal
                f.write(f"[wall-{i}]\n")
                f.write(f"keyframe = {len(wall.entry)}\n")
                f.write(f"nx = {float(normal[0])}\n")
                f.write(f"ny = {float(normal[1])}\n")
                f.write(f"nz = {float(normal[2])}\n")
                f.write(f'transition = "{wall.transition}"\n')
                for key, value in wall.param.list().items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")

            for i, sphere in enumerate(self._sphere):
                f.write(f"[sphere-{i}]\n")
                f.write(f"keyframe = {len(sphere.entry)}\n")
                f.write(f"hemisphere = {'true' if sphere.is_hemisphere else 'false'}\n")
                f.write(f"invert = {'true' if sphere.is_inverted else 'false'}\n")
                f.write(f'transition = "{sphere.transition}"\n')
                for key, value in sphere.param.list().items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")
        pbar.update(1)

        bin_path = os.path.join(path, "bin")
        os.makedirs(bin_path)
        param_path = os.path.join(bin_path, "param")
        os.makedirs(param_path)
        pbar.update(1)

        def export_param(param: dict[str, list[Any]], basepath: str, name: str):
            """Export parameters to a binary file."""
            for key, value in param.items():
                if value:
                    filepath = os.path.join(basepath, f"{name}-{key}.bin")
                    if key == "model":
                        model_map = {
                            "arap": 0,
                            "stvk": 1,
                            "baraff-witkin": 2,
                            "snhk": 3,
                        }
                        assert all(name in model_map for name in value)
                        np.array(
                            [model_map[name] for name in value], dtype=np.uint8
                        ).tofile(filepath)
                    else:
                        np.array(value, dtype=np.float32).tofile(filepath)

        self._displacement.astype(np.float64).tofile(
            os.path.join(bin_path, "displacement.bin")
        )
        self._vert[0].astype(np.uint32).tofile(os.path.join(bin_path, "vert_dmap.bin"))
        self._vert[1].astype(np.float64).tofile(os.path.join(bin_path, "vert.bin"))
        if self._concat_rest_vert is not None:
            self._concat_rest_vert.astype(np.float64).tofile(
                os.path.join(bin_path, "rest_vert.bin")
            )
            self._rest_vert_mask.astype(np.uint8).tofile(
                os.path.join(bin_path, "rest_vert_mask.bin")
            )
        self._color.astype(np.float32).tofile(os.path.join(bin_path, "color.bin"))
        self._vel.astype(np.float32).tofile(os.path.join(bin_path, "vel.bin"))

        # Write velocity schedule and collision windows as dyn_param entries
        dyn_entries = {}
        dyn_entries.update(self._velocity_schedules)
        # Collision windows: each entry is a (t_start, t_end) pair written as 2 floats
        for key, windows in self._collision_windows_data.items():
            dyn_entries[key] = windows

        if dyn_entries:
            dyn_path = os.path.join(path, "dyn_param.txt")
            mode = "a" if os.path.exists(dyn_path) else "w"
            with open(dyn_path, mode) as f:
                for key, entries in dyn_entries.items():
                    f.write(f"[{key}]\n")
                    for entry in entries:
                        if isinstance(entry, (list, tuple)) and len(entry) == 2:
                            if isinstance(entry[1], (list, tuple)):
                                # velocity: (time, [vx, vy, vz])
                                t, vel = entry
                                items = " ".join(str(float(x)) for x in vel)
                                f.write(f"{t} {items}\n")
                            else:
                                # collision window: (t_start, t_end)
                                f.write(f"{float(entry[0])} {float(entry[1])}\n")

        pbar.update(1)

        if self._uv:
            with open(os.path.join(bin_path, "uv.bin"), "wb") as f:
                for uv in self._uv:
                    uv.astype(np.float32).tofile(f)
        pbar.update(1)

        if len(self._rod):
            self._rod.astype(np.uint64).tofile(os.path.join(bin_path, "rod.bin"))
            export_param(self._rod_param, param_path, "rod")
        pbar.update(1)

        if len(self._tri):
            self._tri.astype(np.uint64).tofile(os.path.join(bin_path, "tri.bin"))
            export_param(self._tri_param, param_path, "tri")
        pbar.update(1)

        if len(self._tet):
            self._tet.astype(np.uint64).tofile(os.path.join(bin_path, "tet.bin"))
            export_param(self._tet_param, param_path, "tet")
        pbar.update(1)

        if len(self._static_vert[0]):
            self._static_vert[0].astype(np.uint32).tofile(
                os.path.join(bin_path, "static_vert_dmap.bin")
            )
            self._static_vert[1].astype(np.float64).tofile(
                os.path.join(bin_path, "static_vert.bin")
            )
            self._static_tri.astype(np.uint64).tofile(
                os.path.join(bin_path, "static_tri.bin")
            )
            self._static_color.astype(np.float32).tofile(
                os.path.join(bin_path, "static_color.bin")
            )
            export_param(self._static_param, param_path, "static")
        if self._static_transform_animations:
            all_local = np.vstack([a.local_vert for _, a in self._static_transform_animations])
            all_local.astype(np.float64).tofile(
                os.path.join(bin_path, "static_local_vert.bin")
            )
            all_times = []
            all_trans = []
            all_quats = []
            all_scales = []
            for _, anim in self._static_transform_animations:
                all_times.extend(anim.times)
                all_trans.extend(anim.translations)
                all_quats.extend(anim.quaternions)
                all_scales.extend(anim.scales)
            np.array(all_times, dtype=np.float64).tofile(
                os.path.join(bin_path, "static_transform_time.bin")
            )
            np.array(all_trans, dtype=np.float64).tofile(
                os.path.join(bin_path, "static_transform_translation.bin")
            )
            np.array(all_quats, dtype=np.float64).tofile(
                os.path.join(bin_path, "static_transform_quaternion.bin")
            )
            np.array(all_scales, dtype=np.float64).tofile(
                os.path.join(bin_path, "static_transform_scale.bin")
            )
        pbar.update(1)

        if len(self._stitch_ind) and len(self._stitch_w):
            self._stitch_ind.astype(np.uint64).tofile(
                os.path.join(bin_path, "stitch_ind.bin")
            )
            self._stitch_w.astype(np.float32).tofile(
                os.path.join(bin_path, "stitch_w.bin")
            )
        pbar.update(1)

        for i, pin in enumerate(self._pin):
            # Write pin indices
            with open(os.path.join(bin_path, f"pin-ind-{i}.bin"), "wb") as f:
                np.array(pin.index, dtype=np.uint64).tofile(f)

            # Write operation data
            for j, op in enumerate(pin.operations):
                if isinstance(op, MoveByOperation):
                    # MoveBy operations need to write position delta to binary file
                    op_path = os.path.join(bin_path, f"pin-{i}-op-{j}.bin")
                    with open(op_path, "wb") as f:
                        np.array(op.delta, dtype=np.float64).tofile(f)
                elif isinstance(op, MoveToOperation):
                    # MoveTo operations need to write target positions to binary file
                    op_path = os.path.join(bin_path, f"pin-{i}-op-{j}.bin")
                    with open(op_path, "wb") as f:
                        np.array(op.target, dtype=np.float64).tofile(f)
                elif isinstance(op, TransformKeyframeOperation):
                    # TRS keyframes: separate binaries for verts, times, TRS
                    # arrays, and per-segment interpolation metadata.
                    base = os.path.join(bin_path, f"pin-{i}-op-{j}")
                    np.asarray(op.local_vert, dtype=np.float64).tofile(base + ".bin")
                    np.asarray(op.times, dtype=np.float64).tofile(
                        base + "-time.bin"
                    )
                    np.asarray(op.translations, dtype=np.float64).tofile(
                        base + "-translation.bin"
                    )
                    np.asarray(op.quaternions, dtype=np.float64).tofile(
                        base + "-quaternion.bin"
                    )
                    np.asarray(op.scales, dtype=np.float64).tofile(
                        base + "-scale.bin"
                    )
                    # Per-segment: 1 byte interp code (0=linear, 1=bezier,
                    # 2=constant) and 4 f64 handles. Written as two files to
                    # avoid struct padding headaches. Validation already
                    # rejected unknown interpolations at op-creation time.
                    interp_map = {"LINEAR": 0, "BEZIER": 1, "CONSTANT": 2}
                    segs = op.segments
                    interp_codes = np.asarray(
                        [interp_map[s.get("interpolation", "LINEAR")]
                         for s in segs],
                        dtype=np.uint8,
                    )
                    handles = np.asarray(
                        [[s.get("handle_right", [1/3, 0.0])[0],
                          s.get("handle_right", [1/3, 0.0])[1],
                          s.get("handle_left", [2/3, 1.0])[0],
                          s.get("handle_left", [2/3, 1.0])[1]]
                         for s in segs],
                        dtype=np.float64,
                    )
                    interp_codes.tofile(base + "-interp.bin")
                    handles.tofile(base + "-handles.bin")
                # Spin and Scale operations have all data in info.toml
        pbar.update(1)

        for i, wall in enumerate(self._wall):
            with open(os.path.join(bin_path, f"wall-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _ in wall.entry for p in pos], dtype=np.float64
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"wall-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, t in wall.entry], dtype=np.float64)
                timing.tofile(f)
        pbar.update(1)

        for i, sphere in enumerate(self._sphere):
            with open(os.path.join(bin_path, f"sphere-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _, _ in sphere.entry for p in pos], dtype=np.float64
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"sphere-radius-{i}.bin"), "wb") as f:
                radius = np.array([r for _, r, _ in sphere.entry], dtype=np.float32)
                radius.tofile(f)
            with open(os.path.join(bin_path, f"sphere-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, _, t in sphere.entry], dtype=np.float64)
                timing.tofile(f)
        pbar.update(1)
        pbar.close()
        return self

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the bounding box of the scene.

        Returns:
            tuple[np.ndarray, np.ndarray]: The maximum and minimum coordinates of the bounding box.

        Example:
            Print the extents of the built scene::

                hi, lo = fixed.bbox()
                print("size:", hi - lo)
        """
        vert = self._vert[1] + self._displacement[self._vert[0]]
        return (np.max(vert, axis=0), np.min(vert, axis=0))

    def center(self) -> np.ndarray:
        """Compute the area-weighted center of the scene.

        Returns:
            np.ndarray: The area-weighted center of the scene.

        Example:
            Aim a camera at the scene's area-weighted center::

                target = fixed.center()
                fixed.preview(options={"lookat": target.tolist()})
        """
        vert = self._vert[1] + self._displacement[self._vert[0]]
        tri = self._tri
        center = np.zeros(3)
        area_sum = 0
        for f in tri:
            a, b, c = vert[f[0]], vert[f[1]], vert[f[2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            center += area * (a + b + c) / 3.0
            area_sum += area
        if area_sum == 0:
            raise Exception("no area")
        else:
            return center / area_sum

    def _average_tri_area(self) -> float:
        """Compute the average triangle area of the scene.

        Returns:
            float: The average triangle area of the scene.
        """
        if len(self._area):
            return self._area.mean()
        else:
            return 0.0

    def set_pin(self, pin: list[PinData]):
        """Set the pinning data of all the objects.

        Args:
            pin (list[PinData]): A list of pinning data.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject pinning data into a fixed scene::

                fixed = scene.build()
                fixed.set_pin(list_of_pin_data)
        """
        self._pin = pin

    def set_spin(self, spin: list[SpinData]):
        """Set the spinning data of all the objects.

        Args:
            spin (list[SpinData]): A list of spinning data.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject spin data::

                fixed = scene.build()
                fixed.set_spin(list_of_spin_data)
        """
        self._spin = spin

    def set_static(
        self,
        vert: tuple[np.ndarray, np.ndarray],
        tri: np.ndarray,
        color: np.ndarray,
        param: dict[str, list[Any]],
        transform_animations: Optional[list[tuple[int, TransformAnimation]]] = None,
    ):
        """Set the static mesh data.

        Args:
            vert (tuple[np.ndarray, np.ndarray]): The vertices of the static mesh. The first array is the displacement map reference; the second is local positions.
            tri (np.ndarray): The triangle elements of the static mesh.
            color (np.ndarray): The colors of the static mesh.
            param (dict[str, list[Any]]): Parameters for the static mesh elements.
            transform_animations: Optional list of ``(vert_offset, TransformAnimation)`` for animated static objects.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to attach a static collider mesh::

                fixed = scene.build()
                fixed.set_static((ref, V), F, colors, {"area": areas})
        """
        self._static_vert = vert
        self._static_tri = tri
        self._static_color = color
        self._static_param = param
        if transform_animations:
            self._static_transform_animations = transform_animations

    def set_stitch(self, ind: np.ndarray, w: np.ndarray):
        """Set the stitch data.

        Args:
            ind (np.ndarray): The stitch indices.
            w (np.ndarray): The stitch weights.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject stitch constraints::

                fixed = scene.build()
                fixed.set_stitch(stitch_indices, stitch_weights)
        """
        self._stitch_ind = ind
        self._stitch_w = w

    def time(self, time: float) -> np.ndarray:
        """Compute the vertex positions at a specific time.

        Args:
            time (float): The time to compute the vertex positions.

        Returns:
            np.ndarray: The vertex positions at the specified time.

        Example:
            Evaluate the scene midway through a pin animation::

                vert_mid = fixed.time(0.5)
                print(vert_mid.shape)
        """
        vert = self._vert[1].copy()
        initial = self._vert[1]

        for pin in self._pin:
            # Reset to initial before applying ops (matches Rust solver
            # which computes position = initial + ops independently per pin)
            vert[pin.index] = initial[pin.index]
            for op in pin.operations:
                vert[pin.index] = op.apply(vert[pin.index], time)

        vert += time * self._vel
        vert += self._displacement[self._vert[0]]
        return vert

    def static_time(self, time: float) -> np.ndarray:
        """Compute animated static vertex positions at a specific time.

        Args:
            time (float): The time to evaluate.

        Returns:
            np.ndarray: The static vertex positions at the specified time.

        Example:
            Sample the animated collider mesh halfway through the animation::

                fixed = scene.build()
                V_static = fixed.static_time(0.5)
                print(V_static.shape)
        """
        if not self._static_transform_animations:
            if len(self._static_vert[1]) == 0:
                return np.zeros((0, 3))
            return self._static_vert[1] + self._displacement[self._static_vert[0]]
        base = self._static_vert[1] + self._displacement[self._static_vert[0]]
        result = base.copy()
        for vert_offset, anim in self._static_transform_animations:
            n = len(anim.local_vert)
            result[vert_offset:vert_offset + n] = anim.evaluate(time)
        return result

    def preview(
        self,
        vert: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        show_slider: bool = True,
        engine: str = "threejs",
    ) -> Optional["Plot"]:
        """Preview the scene.

        Args:
            vert (Optional[np.ndarray], optional): The vertices to preview. Defaults to None, in which case ``self.vertex()`` is used.
            options (dict, optional): The options for the plot. Defaults to None.
            show_slider (bool, optional): Whether to show the time slider. Defaults to True.
            engine (str, optional): The rendering engine. Defaults to ``"threejs"``.

        Returns:
            Optional[Plot]: The plot object if in a Jupyter notebook, otherwise None.

        Example:
            Preview the initial scene with a custom camera and no pin markers::

                opts = {"eye": [0, 1.4, 2.5], "pin": False, "wireframe": True}
                fixed.preview(options=opts)
        """
        if options is None:
            options = {}
        default_opts = {
            "flat_shading": False,
            "wireframe": True,
            "stitch": True,
            "pin": True,
        }
        options = dict(options)
        for key, value in default_opts.items():
            if key not in options:
                options[key] = value

        if self._plot is not None and self._plot.is_jupyter_notebook():
            if vert is None:
                vert = self.vertex()
            assert vert is not None
            color = self.color(vert, options)
            assert len(color) == len(vert)
            tri = self._tri.copy()
            edge = self._rod.copy()
            pts = np.zeros(0)
            plotter = self._plot.create(engine)

            n_dynamic_vert = len(vert)
            has_static = len(self._static_vert[1]) > 0 or self._static_transform_animations
            if has_static:
                static_vert = self.static_time(0.0)
                static_color = np.zeros_like(static_vert)
                static_color[:, :] = self._static_color
                if len(tri):
                    tri = np.vstack([tri, self._static_tri + len(vert)])
                else:
                    tri = self._static_tri + len(vert)
                vert = np.vstack([vert, static_vert])
                color = np.vstack([color, static_color])
            assert vert is not None and color is not None
            assert len(color) == len(vert)

            if options["stitch"] and len(self._stitch_ind) and len(self._stitch_w):
                stitch_vert, stitch_edge = [], []
                for ind, w in zip(self._stitch_ind, self._stitch_w, strict=False):
                    # 4D format: ind=[src, t0, t1, t2], w=[ws, w1, w2, w3]
                    src = vert[ind[0]]
                    target = w[1] * vert[ind[1]] + w[2] * vert[ind[2]] + w[3] * vert[ind[3]]
                    idx0 = len(stitch_vert) + len(vert)
                    idx1 = idx0 + 1
                    stitch_vert.append(src)
                    stitch_vert.append(target)
                    stitch_edge.append([idx0, idx1])
                stitch_vert = np.array(stitch_vert)
                stitch_edge = np.array(stitch_edge)
                stitch_color = np.tile(np.array([1.0, 1.0, 1.0]), (len(stitch_vert), 1))
                vert = np.vstack([vert, stitch_vert])
                edge = np.vstack([edge, stitch_edge]) if len(edge) else stitch_edge
                color = np.vstack([color, stitch_color])

            if options["pin"] and self._pin:
                # Triangle area is the natural marker scale for surface
                # meshes; for rod-only scenes, fall back to a fraction of
                # the average rod edge length so pin dots stay visible.
                if len(self._area):
                    options["pts_scale"] = float(np.sqrt(self._area.mean()))
                elif len(self._rod):
                    rest = self._vert[1]
                    diff = rest[self._rod[:, 0]] - rest[self._rod[:, 1]]
                    options["pts_scale"] = 0.5 * float(
                        np.linalg.norm(diff, axis=1).mean()
                    )
                pts = []
                for pin in self._pin:
                    # Static-moving pin-shells pin every vertex as an
                    # implementation detail; hide those dots in preview.
                    if pin.hide_in_preview:
                        continue
                    pts.extend(pin.index)
                pts = np.array(pts)

            plotter.plot(vert, color, tri, edge, pts, options)

            has_vel = np.linalg.norm(self._vel) > 0
            has_static_anim = bool(self._static_transform_animations)
            if show_slider and (self._pin or has_vel or has_static_anim):
                max_time = 0
                if self._pin:
                    for pin in self._pin:
                        for op in pin.operations:
                            _, t_end = op.get_time_range()
                            if t_end == float("inf"):
                                max_time = max(max_time, 1.0)
                            else:
                                max_time = max(max_time, t_end)
                if has_vel:
                    max_time = max(max_time, 1.0)
                for _, anim in self._static_transform_animations:
                    max_time = max(max_time, anim.max_time())
                if max_time > 0:

                    def update(time=0):
                        dyn_vert = self.time(time)
                        if has_static:
                            static_v = self.static_time(time)
                            combined = np.vstack([dyn_vert, static_v])
                        else:
                            combined = dyn_vert
                        plotter.update(combined)

                    from ipywidgets import interact

                    interact(update, time=(0, max_time, 0.01))
            return plotter
        else:
            return None


class SceneInfo:
    """Lightweight metadata handle carrying the scene name.

    Example:
        Look up the name of the scene you are currently editing::

            print(scene.info.name)
    """

    def __init__(self, name: str, scene: "Scene"):
        self._scene = scene
        self.name = name


class InvisibleAdder:
    """Helper for attaching invisible colliders (walls and spheres) to a scene.

    Obtained via ``scene.add.invisible``.

    Example:
        Add a ground wall and a ball collider together::

            scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
            scene.add.invisible.sphere([0, 0.5, 0], 0.25)
    """

    def __init__(self, scene: "Scene"):
        self._scene = scene

    def sphere(self, position: list[float], radius: float) -> Sphere:
        """Add an invisible sphere to the scene.

        Args:
            position (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.
        Returns:
            Sphere: The invisible sphere.

        Example:
            Place an inverted hemispherical bowl under the cloth::

                scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
        """
        sphere = Sphere().add(position, radius)
        self._scene.sphere_list.append(sphere)
        return sphere

    def wall(self, position: list[float], normal: list[float]) -> Wall:
        """Add an invisible wall to the scene.

        Args:
            position (list[float]): The position of the wall.
            normal (list[float]): The outer normal of the wall.
        Returns:
            Wall: The invisible wall.

        Example:
            Seal a simulation box with four side walls::

                scene.add.invisible.wall([1, 0, 0], [-1, 0, 0])
                scene.add.invisible.wall([-1, 0, 0], [1, 0, 0])
                scene.add.invisible.wall([0, 0, 1], [0, 0, -1])
                scene.add.invisible.wall([0, 0, -1], [0, 0, 1])
        """
        wall = Wall().add(position, normal)
        self._scene.wall_list.append(wall)
        return wall


class ObjectAdder:
    """Factory for introducing meshes into a :class:`Scene`.

    Reached as ``scene.add``. Calling it returns an :class:`Object` that can
    be chained with transforms, pins, and colors. Invisible colliders live
    under ``scene.add.invisible``.

    Example:
        Drop a cloth sheet and an invisible ground wall into a scene::

            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
            fixed = scene.build()
    """

    def __init__(self, scene: "Scene"):
        self._scene = scene
        self.invisible = InvisibleAdder(
            scene
        )  #: InvisibleAdder: The invisible object adder.

    def __call__(self, mesh_name: str, ref_name: str = "") -> "Object":
        """Add a mesh to the scene.

        Args:
            mesh_name (str): The name of the mesh to add.
            ref_name (str, optional): The reference name of the object.

        Returns:
            Object: The added object.

        Example:
            Drop two instances of a registered asset into the scene,
            the second as a static collider::

                scene = app.scene.create()
                scene.add("sheet").at(0, 0.6, 0)
                scene.add("sphere").at(0, 0, 0).pin()  # static collider
                scene = scene.build()
        """
        if ref_name == "":
            ref_name = mesh_name
            count = 0
            while ref_name in self._scene.object_dict:
                count += 1
                ref_name = f"{mesh_name}_{count}"
        mesh_list = self._scene.asset_manager.list()
        if mesh_name not in mesh_list:
            raise Exception(f"mesh_name '{mesh_name}' does not exist")
        elif ref_name in self._scene.object_dict:
            raise Exception(f"ref_name '{ref_name}' already exists")
        else:
            obj = Object(self._scene.asset_manager, mesh_name)
            self._scene.object_dict[ref_name] = obj
            # Auto-set UV from asset if available
            asset_data = self._scene.asset_manager.fetch.get(mesh_name)
            if "UV" in asset_data and "F" in asset_data:
                vertex_uv = asset_data["UV"]
                faces = asset_data["F"]
                # Convert vertex UV to per-face UV
                face_uv = []
                for f in faces:
                    uv_per_face = np.array(
                        [
                            vertex_uv[f[0]],
                            vertex_uv[f[1]],
                            vertex_uv[f[2]],
                        ]
                    )
                    face_uv.append(uv_per_face)
                obj.set_uv(face_uv)
            return obj


class Scene:
    """A scene class.

    A ``Scene`` collects objects, pins, invisible colliders, and stitch data,
    then compiles them into a :class:`FixedScene` via :meth:`build`.

    Example:
        Build a tiny drape scene from registered assets::

            scene = app.scene.create()
            sheet = scene.add("sheet").at(0, 0.6, 0)
            sheet.pin(sheet.grab([-1, 0, -1]) + sheet.grab([1, 0, -1]))
            scene.add("sphere").at(0, 0, 0).pin()
            fixed = scene.build().report()
    """

    def __init__(self, name: str, plot: Optional[PlotManager], asset: AssetManager):
        self._name = name
        self._plot = plot
        self._asset = asset
        self._object: dict[str, Object] = {}
        self._sphere: list[Sphere] = []
        self._wall: list[Wall] = []
        self._surface_map_by_name: dict[str, tuple] = {}
        self._explicit_merge_pairs: list[dict] = []
        self._cross_stitch: list[dict] = []
        self.add = ObjectAdder(self)  #: ObjectAdder: The object adder.
        self.info = SceneInfo(name, self)  #: SceneInfo: The scene information.

    def clear(self) -> "Scene":
        """Clear all objects from the scene.

        Returns:
            Scene: The cleared scene.

        Example:
            Start from a blank slate before re-adding objects::

                scene.clear()
                scene.add("sheet").at(0, 0.6, 0)
        """
        self._object.clear()
        return self

    def set_explicit_merge_pairs(self, pairs: list[dict]):
        """Set explicit per-vertex merge pairs captured at snap time.

        Each entry must contain non-empty ``source_uuid`` and ``target_uuid``;
        missing keys raise :class:`ValueError`.

        Example:
            Typically invoked internally by the Blender add-on decoder, but
            can be called directly to wire up per-vertex merge constraints::

                scene.set_explicit_merge_pairs([
                    {"source_uuid": "a", "target_uuid": "b",
                     "source_vert": 0, "target_vert": 12},
                ])
        """
        for i, pair in enumerate(pairs):
            source_uuid = pair.get("source_uuid", "")
            target_uuid = pair.get("target_uuid", "")
            if not source_uuid or not target_uuid:
                raise ValueError(
                    f"Merge pair [{i}] missing required source_uuid/target_uuid: "
                    f"source_uuid={source_uuid!r} target_uuid={target_uuid!r}"
                )
        self._explicit_merge_pairs = pairs

    def set_surface_map(
        self,
        name: str,
        tri_indices: np.ndarray,
        coefs: np.ndarray,
        surf_tri: np.ndarray,
    ):
        """Store frame-embedding surface mapping for a tetrahedralized object.

        Args:
            name: Object UUID key.
            tri_indices: Closest triangle index in tet surface per original vertex (N,).
            coefs: Frame coefficients (c1, c2, c3) per original vertex (N, 3).
            surf_tri: Surface triangles of the tet mesh (Q, 3).

        Example:
            Typically invoked internally by the Blender add-on decoder when a
            TetMesh is registered, but can be called directly to attach a
            surface map::

                scene.set_surface_map("obj-uuid", tri_idx, coefs, surf_tri)
        """
        if not name:
            raise ValueError("set_surface_map requires a non-empty object UUID key")
        self._surface_map_by_name[name] = (tri_indices, coefs, surf_tri)

    def select(self, name: str) -> "Object":
        """Select an object from the scene by its name.

        Args:
            name (str): The reference name of the object to select.

        Returns:
            Object: The selected object.

        Example:
            Adjust an already-added object by ref name::

                scene.add("sheet")
                sheet = scene.select("sheet")
                sheet.at(0, 0.6, 0)
        """
        if name not in self._object:
            raise Exception(f"object {name} does not exist")
        else:
            return self._object[name]

    def min(self, axis: str) -> float:
        """Get the minimum value of the scene along a specific axis.

        Args:
            axis (str): The axis to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The minimum vertex coordinate along the specified axis.

        Example:
            Place a ground wall just below the lowest vertex::

                y_min = scene.min("y")
                scene.add.invisible.wall([0, y_min - 0.01, 0], [0, 1, 0])
        """
        result = float("inf")
        _axis = {"x": 0, "y": 1, "z": 2}
        for obj in self._object.values():
            vert = obj.vertex(True)
            if vert is not None:
                result = min(result, np.min(vert[:, _axis[axis]]))
        return result

    def max(self, axis: str) -> float:
        """Get the maximum value of the scene along a specific axis.

        Args:
            axis (str): The axis to get the maximum value along, either "x", "y", or "z".

        Returns:
            float: The maximum vertex coordinate along the specified axis.

        Example:
            Place a ceiling wall just above the highest vertex::

                y_max = scene.max("y")
                scene.add.invisible.wall([0, y_max + 0.01, 0], [0, -1, 0])
        """
        result = float("-inf")
        _axis = {"x": 0, "y": 1, "z": 2}
        for obj in self._object.values():
            vert = obj.vertex(True)
            if vert is not None:
                result = max(result, np.max(vert[:, _axis[axis]]))
        return result

    @property
    def sphere_list(self) -> list[Sphere]:
        """Get the list of spheres.

        Example:
            Enumerate invisible sphere colliders currently on the scene::

                for sphere in scene.sphere_list:
                    print(sphere.entry)
        """
        return self._sphere

    @property
    def wall_list(self) -> list[Wall]:
        """Get the list of walls.

        Example:
            Enumerate invisible wall colliders currently on the scene::

                for wall in scene.wall_list:
                    print(wall.normal, wall.entry)
        """
        return self._wall

    @property
    def object_dict(self) -> dict[str, "Object"]:
        """Get the object dictionary.

        Example:
            Iterate every added object by its reference name::

                for name, obj in scene.object_dict.items():
                    print(name, obj.obj_type)
        """
        return self._object

    @property
    def asset_manager(self) -> AssetManager:
        """Get the asset manager.

        Example:
            Reach into the asset manager attached to this scene::

                mgr = scene.asset_manager
                print(mgr.list())
        """
        return self._asset

    def build(self, progress_callback=None) -> FixedScene:
        """Build the fixed scene from the current scene.

        Args:
            progress_callback: Optional callable ``f(fraction, step_info)`` invoked as the build progresses.

        Returns:
            FixedScene: The built fixed scene.

        Example:
            Compile the scene and chain a summary print::

                fixed = scene.build().report()
                session = app.session.create(fixed).build()
        """
        total_steps = 12
        completed_steps = 0

        def report(step_info: str):
            if progress_callback is not None:
                progress_callback(completed_steps / total_steps, step_info)

        def advance(step_info: str):
            nonlocal completed_steps
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(completed_steps / total_steps, step_info)

        pbar = tqdm(total=total_steps, desc="build scene")
        report("Building scene: preparing objects...")
        for _, obj in self._object.items():
            obj.update_static()
        pbar.update(1)
        advance("Building scene: preparing objects...")

        concat_count = 0
        dyn_objects = [
            (name, obj) for name, obj in self._object.items() if not obj.static
        ]
        n = len(dyn_objects)
        for i, (_, obj) in enumerate(dyn_objects):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.75, 1.0)
            if obj.object_color is None:
                obj.default_color(r, g, b)

        # Build vertex alias map from merge pairs
        # Objects are registered by UUID; merge pairs carry source_uuid/target_uuid
        vertex_alias: dict[tuple[str, int], tuple[str, int]] = {}
        if self._explicit_merge_pairs:
            for pair in self._explicit_merge_pairs:
                source_name = pair.get("source_uuid", "")
                target_name = pair.get("target_uuid", "")
                index_pairs = pair.get("pairs", [])
                if not source_name or not target_name:
                    raise ValueError("Merge pair missing source_uuid/target_uuid")
                source_obj = self._object.get(source_name)
                target_obj = self._object.get(target_name)
                if source_obj is None or target_obj is None:
                    raise ValueError(
                        f"Merge pair references unknown object: "
                        f"source={source_name!r} target={target_name!r}"
                    )
                if source_obj.static or target_obj.static:
                    continue
                merged_count = 0
                for src_i, tgt_i in index_pairs:
                    vertex_alias[(target_name, int(tgt_i))] = (source_name, int(src_i))
                    merged_count += 1
                if merged_count > 0:
                    print(
                        f"Explicit merge pair {source_name} <-> {target_name}: "
                        f"{merged_count} vertices aliased"
                    )
        def resolve_alias(name, vi):
            while (name, vi) in vertex_alias:
                name, vi = vertex_alias[(name, vi)]
            return name, vi

        def add_entry(obj_name, map, entry):
            nonlocal concat_count
            for e in entry:
                for vi in e:
                    if map[vi] != -1:
                        continue
                    target_name, target_vi = resolve_alias(obj_name, vi)
                    if target_name != obj_name or target_vi != vi:
                        target_map = map_by_name[target_name]
                        if target_map[target_vi] != -1:
                            map[vi] = target_map[target_vi]
                        else:
                            target_map[target_vi] = concat_count
                            map[vi] = concat_count
                            concat_count += 1
                    else:
                        map[vi] = concat_count
                        concat_count += 1

        map_by_name = {}
        for name, obj in dyn_objects:
            vert = obj.get("V")
            if vert is not None:
                map_by_name[name] = np.full(len(vert), -1, dtype=np.int64)

        pbar.update(1)
        advance("Building scene: indexing rod topology...")
        for name, obj in dyn_objects:
            if obj.get("T") is None:
                map = map_by_name[name]
                edge = obj.get("E")
                if edge is not None:
                    add_entry(name, map, edge)
        rod_vert_start, rod_vert_end = 0, concat_count

        for name, obj in dyn_objects:
            if obj.get("T") is None:
                map, tri = map_by_name[name], obj.get("F")
                if tri is not None:
                    add_entry(name, map, tri)
        shell_vert_start, shell_vert_end = rod_vert_end, concat_count

        pbar.update(1)
        advance("Building scene: indexing shell topology...")
        for name, obj in dyn_objects:
            map, tri = map_by_name[name], obj.get("F")
            if tri is not None:
                add_entry(name, map, tri)

        pbar.update(1)
        advance("Building scene: finalizing vertex map...")
        for name, obj in dyn_objects:
            vert = obj.get("V")
            if vert is not None:
                map = map_by_name[name]
                unassigned = np.where(map == -1)[0]
                for i in unassigned:
                    target_name, target_vi = resolve_alias(name, int(i))
                    if target_name != name or target_vi != int(i):
                        target_map = map_by_name[target_name]
                        if target_map[target_vi] != -1:
                            map[i] = target_map[target_vi]
                            continue
                        target_map[target_vi] = concat_count
                        map[i] = concat_count
                        concat_count += 1
                    else:
                        map[i] = concat_count
                        concat_count += 1

        dmap = {}
        concat_displacement = []
        concat_vert_dmap = np.zeros(concat_count, dtype=np.uint32)
        concat_vert = np.zeros((concat_count, 3))
        concat_color = np.zeros((concat_count, 3))
        concat_dyn_tri_color = []
        concat_dyn_tri_intensity = []
        concat_vel = np.zeros((concat_count, 3))
        concat_uv = []
        concat_pin = []
        concat_rod = []
        concat_tri = []
        concat_tet = []
        concat_static_vert_dmap = []
        concat_static_vert = []
        concat_static_tri = []
        concat_static_color = []
        concat_stitch_ind = []
        concat_stitch_w = []
        concat_rod_param = {}
        concat_tri_param = {}
        concat_tet_param = {}
        concat_static_param = {}

        def vec_map(map, elm):
            map_arr = np.asarray(map)
            return map_arr[elm]

        def extend_param(
            param: ParamHolder,
            concat_param: dict[str, list],
            count: int,
        ):
            if len(concat_param.keys()):
                assert param.key_list() == list(concat_param.keys()), (
                    f"param keys mismatch: {param.key_list()} vs {list(concat_param.keys())}"
                )
            for key, value in param.items():
                if key not in concat_param:
                    concat_param[key] = []
                concat_param[key].extend([value] * count)

        for name, obj in self._object.items():
            dmap[name] = len(concat_displacement)
            concat_displacement.append(obj.position)
        concat_displacement = np.array(concat_displacement)

        has_merge = bool(vertex_alias)

        pbar.update(1)
        advance("Building scene: gathering dynamic vertices...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            vert = obj.vertex(False)
            if vert is not None:
                concat_vert[map] = vert
                concat_vert_dmap[map] = [dmap[name]] * len(map)
                concat_vel[map] = obj.object_velocity
                concat_color[map] = obj.get("color")

        # Collect velocity schedule entries (stored on FixedScene, exported by session)
        velocity_schedules = {}
        for name, obj in dyn_objects:
            if obj._velocity_schedule:
                dmap_idx = dmap[name]
                key = f"velocity:{dmap_idx}"
                velocity_schedules[key] = [(t, list(v)) for t, v in obj._velocity_schedule]

        # Collect collision window entries (max 8 per object)
        MAX_COLLISION_WINDOWS = 8
        collision_windows = {}
        for name, obj in dyn_objects:
            if obj._collision_windows:
                if len(obj._collision_windows) > MAX_COLLISION_WINDOWS:
                    raise ValueError(
                        f"Object '{name}' has {len(obj._collision_windows)} collision windows "
                        f"(max {MAX_COLLISION_WINDOWS})"
                    )
                dmap_idx = dmap[name]
                key = f"collision_window:{dmap_idx}"
                collision_windows[key] = obj._collision_windows

        # Collect pinned vertices per object for per-element collider check
        pinned_verts_by_obj: dict[str, set[int]] = {}
        for name, obj in dyn_objects:
            if obj.pin_list:
                pinned_verts = set()
                for p in obj.pin_list:
                    pinned_verts.update(p.index)
                pinned_verts_by_obj[name] = pinned_verts
            else:
                pinned_verts_by_obj[name] = set()

        def is_elem_collider(name: str, elem_verts: list[int]) -> bool:
            """Check if all vertices of an element are pinned."""
            pinned = pinned_verts_by_obj.get(name, set())
            return all(v in pinned for v in elem_verts)

        concat_rod_is_collider = []
        concat_tri_is_collider = []

        pbar.update(1)
        advance("Building scene: assembling rods...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            if obj.obj_type == "rod":
                edge = obj.get("E")
                if edge is not None:
                    t = vec_map(map, edge)
                    if has_merge:
                        filtered_t = []
                        filtered_collider = []
                        for idx, (e_mapped, e_orig) in enumerate(zip(t, edge)):
                            if e_mapped[0] != e_mapped[1]:
                                filtered_t.append(e_mapped)
                                filtered_collider.append(
                                    is_elem_collider(name, e_orig.tolist())
                                )
                        concat_rod.extend(filtered_t)
                        concat_rod_is_collider.extend(filtered_collider)
                        extend_param(obj.param, concat_rod_param, len(filtered_t))
                    else:
                        concat_rod.extend(t)
                        for e in edge:
                            concat_rod_is_collider.append(
                                is_elem_collider(name, e.tolist())
                            )
                        extend_param(obj.param, concat_rod_param, len(t))
        rod_count = len(concat_rod)

        pbar.update(1)
        advance("Building scene: assembling shell triangles...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet, tri = obj.get("T"), obj.get("F")
            if tri is not None and tet is None:
                t = vec_map(map, tri)
                if has_merge:
                    uv_list = obj.uv_coords
                    for idx, (t_mapped, face) in enumerate(zip(t, tri)):
                        if len(set(t_mapped)) == 3:
                            concat_tri.append(t_mapped)
                            if uv_list is not None:
                                concat_uv.append(uv_list[idx])
                            else:
                                concat_uv.append(np.zeros((2, 3), dtype=np.float32))
                            concat_dyn_tri_color.append(obj.dynamic_color)
                            concat_dyn_tri_intensity.append(obj.dynamic_intensity)
                            concat_tri_is_collider.append(
                                is_elem_collider(name, face.tolist())
                            )
                    # Param count must match filtered tri count — count how many we added
                    added = sum(
                        1 for t_mapped in t if len(set(t_mapped)) == 3
                    )
                    extend_param(obj.param, concat_tri_param, added)
                else:
                    concat_tri.extend(t)
                    if obj.uv_coords is not None:
                        concat_uv.extend(obj.uv_coords)
                    else:
                        concat_uv.extend(
                            [np.zeros((2, 3), dtype=np.float32)] * len(t)
                        )
                    concat_dyn_tri_color.extend([obj.dynamic_color] * len(t))
                    concat_dyn_tri_intensity.extend([obj.dynamic_intensity] * len(t))
                    for face in tri:
                        concat_tri_is_collider.append(
                            is_elem_collider(name, face.tolist())
                        )
                    extend_param(obj.param, concat_tri_param, len(t))
        shell_count = len(concat_tri)

        pbar.update(1)
        advance("Building scene: assembling solid surfaces...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet, tri = obj.get("T"), obj.get("F")
            if tet is not None and tri is not None:
                t = vec_map(map, tri)
                if has_merge:
                    added = 0
                    for t_mapped, face in zip(t, tri):
                        if len(set(t_mapped)) == 3:
                            concat_tri.append(t_mapped)
                            concat_dyn_tri_color.append(obj.dynamic_color)
                            concat_dyn_tri_intensity.append(obj.dynamic_intensity)
                            concat_tri_is_collider.append(
                                is_elem_collider(name, face.tolist())
                            )
                            added += 1
                    extend_param(obj.param, concat_tri_param, added)
                else:
                    concat_tri.extend(t)
                    concat_dyn_tri_color.extend([obj.dynamic_color] * len(t))
                    concat_dyn_tri_intensity.extend([obj.dynamic_intensity] * len(t))
                    for face in tri:
                        concat_tri_is_collider.append(
                            is_elem_collider(name, face.tolist())
                        )
                    extend_param(obj.param, concat_tri_param, len(t))

        pbar.update(1)
        advance("Building scene: assembling tetrahedra...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet = obj.get("T")
            if tet is not None:
                t = vec_map(map, tet)
                if has_merge:
                    added = 0
                    for t_mapped in t:
                        if len(set(t_mapped)) == 4:
                            concat_tet.append(t_mapped)
                            added += 1
                    extend_param(obj.param, concat_tet_param, added)
                else:
                    concat_tet.extend(t)
                    extend_param(obj.param, concat_tet_param, len(t))

        pbar.update(1)
        advance("Building scene: collecting pins and stitches...")
        for name, obj in dyn_objects:
            map = map_by_name[name]
            hide_preview_pins = bool(getattr(obj, "_is_static_moving", False))
            for p in obj.pin_list:
                # Map torque hint_vertex through vertex map (local → global)
                mapped_ops = []
                for op in p.operations:
                    if isinstance(op, TorqueOperation) and op.hint_vertex >= 0:
                        from dataclasses import replace
                        op = replace(op, hint_vertex=map[op.hint_vertex])
                    mapped_ops.append(op)
                concat_pin.append(
                    PinData(
                        index=[map[vi] for vi in p.index],
                        operations=mapped_ops,
                        unpin_time=p.unpin_time,
                        pull_strength=p.pull_strength,
                        transition=p.transition,
                        pin_group_id=p.pin_group_id,
                        hide_in_preview=hide_preview_pins,
                    )
                )
            stitch_ind = obj.get("Ind")
            stitch_w = obj.get("W")
            if stitch_ind is not None and stitch_w is not None:
                mapped_ind = vec_map(map, stitch_ind)
                for si, sw in zip(mapped_ind, stitch_w, strict=False):
                    if len(si) == 3:
                        # Convert 3-index to 4-index: duplicate last vertex
                        concat_stitch_ind.append([si[0], si[1], si[2], si[2]])
                    else:
                        concat_stitch_ind.append(list(si))
                    if len(sw) == 2:
                        # Convert 2-weight to 4-weight: [1.0, 1-t, t, 0.0]
                        concat_stitch_w.append([1.0, float(sw[0]), float(sw[1]), 0.0])
                    elif len(sw) == 4:
                        concat_stitch_w.append(list(sw))
                    else:
                        concat_stitch_w.append([1.0, float(sw[0]), float(sw[1]), 0.0])

        # Add cross-object stitch with proper per-object mapping
        for cs in self._cross_stitch:
            source_map = map_by_name.get(cs["source_name"])
            target_map = map_by_name.get(cs["target_name"])
            if source_map is None or target_map is None:
                raise ValueError(
                    f"Cross-stitch references unknown object: "
                    f"source={cs['source_name']!r} target={cs['target_name']!r}"
                )
            for ind, w in zip(cs["ind"], cs["w"]):
                global_ind = [
                    source_map[int(ind[0])],
                    target_map[int(ind[1])],
                    target_map[int(ind[2])],
                    target_map[int(ind[3])],
                ]
                concat_stitch_ind.append(global_ind)
                concat_stitch_w.append(list(w))

        # Compute global set of pinned vertices for invisible collider checks
        global_pinned_vertices: set[int] = set()
        for pin in concat_pin:
            global_pinned_vertices.update(pin.index)

        # Build rest_vert: when ALL solver-space vertices of an object are
        # covered by pins that have unpin_time, compute the deformed shape
        # at unpin time (using solver-space positions and operations).
        concat_rest_vert = concat_vert.copy()
        rest_vert_mask = np.zeros(concat_count, dtype=np.uint8)
        for name, obj in dyn_objects:
            indices = map_by_name[name]
            obj_indices_set = set(indices.tolist())
            # Find all concat_pin entries whose indices overlap this object
            obj_pins = [
                p for p in concat_pin
                if set(p.index) & obj_indices_set
            ]
            if not obj_pins:
                continue
            covered = set()
            all_have_duration = True
            for p in obj_pins:
                if p.unpin_time is None:
                    all_have_duration = False
                    break
                covered.update(p.index)
            if not all_have_duration or not obj_indices_set.issubset(covered):
                continue
            # All verts pinned with duration — compute rest shape at unpin
            # using solver-space positions and operations
            for p in obj_pins:
                positions = concat_rest_vert[p.index].copy()
                for op in p.operations:
                    positions = op.apply(positions, p.unpin_time)
                concat_rest_vert[p.index] = positions
                rest_vert_mask[p.index] = 1
        has_rest_vert = bool(rest_vert_mask.any())

        pbar.update(1)
        advance("Building scene: assembling static geometry...")
        concat_static_transform_anims: list[tuple[int, TransformAnimation]] = []
        for name, obj in self._object.items():
            if obj.static:
                color = obj.get("color")
                offset = len(concat_static_vert)
                tri, vert = obj.get("F"), obj.get("V")
                if tri is not None and vert is not None:
                    concat_static_tri.extend(tri + offset)
                    concat_static_vert.extend(obj.apply_transform(vert, False))
                    concat_static_color.extend([color] * len(vert))
                    concat_static_vert_dmap.extend([dmap[name]] * len(vert))
                    extend_param(
                        obj.param,
                        concat_static_param,
                        len(tri),
                    )
                    if obj._transform_animation is not None:
                        concat_static_transform_anims.append((offset, obj._transform_animation))
        pbar.update(1)
        advance("Building scene: finalizing fixed scene...")

        for key in ["model"]:
            concat_rod_param[key] = []
            concat_static_param[key] = []

        for key in ["poiss-rat"]:
            concat_rod_param[key] = []

        # Rods support strain-limit (alongside shells), so do not clear it from
        # concat_rod_param. Shrink-x/y remain rod/tet/static-irrelevant.
        # Uniform shrink is solid-only; rods and statics drop it.
        for key in ["strain-limit"]:
            concat_tet_param[key] = []
            concat_static_param[key] = []
        for key in ["shrink-x", "shrink-y"]:
            concat_rod_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []
        for key in ["shrink"]:
            concat_rod_param[key] = []
            concat_tri_param[key] = []
            concat_static_param[key] = []

        for key in ["friction", "contact-gap", "contact-offset", "bend"]:
            concat_tet_param[key] = []

        # Bend plasticity is shells+rods (per feedback_plasticity_scope rods
        # get bend-only). Keep rod/tri values; strip for tet/static.
        for key in ["bend-plasticity", "bend-plasticity-threshold",
                    "bend-rest-from-geometry"]:
            concat_tet_param[key] = []
            concat_static_param[key] = []

        for key in ["young-mod", "poiss-rat", "bend", "density"]:
            concat_static_param[key] = []

        for key in ["length-factor"]:
            concat_tri_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []

        # Shrink/extend invalidates strain limiting on shells: the two
        # formulations both rewrite the rest shape and combining them makes
        # the per-face strain bound ill-defined. Reject explicit conflicts
        # rather than silently picking a winner.
        sx = concat_tri_param.get("shrink-x", [])
        sy = concat_tri_param.get("shrink-y", [])
        sl = concat_tri_param.get("strain-limit", [])
        if sx and sy and sl:
            for i, (x, y, s) in enumerate(zip(sx, sy, sl)):
                if (float(x) != 1.0 or float(y) != 1.0) and float(s) > 0.0:
                    raise ValueError(
                        f"Shell face {i}: shrink (x={float(x)}, y={float(y)}) "
                        f"conflicts with strain-limit ({float(s)}). "
                        "Set strain-limit to 0 or keep shrink at 1.0."
                    )

        # Compute static vertices with displacement for intersection check
        static_vert_for_check = None
        static_tri_for_check = None
        if len(concat_static_vert) > 0 and len(concat_static_tri) > 0:
            static_vert_dmap_arr = np.array(concat_static_vert_dmap)
            static_vert_for_check = (
                np.array(concat_static_vert) + concat_displacement[static_vert_dmap_arr]
            )
            static_tri_for_check = np.array(concat_static_tri)

        fixed = FixedScene(
            self._plot,
            self.info.name,
            map_by_name,
            concat_displacement,
            (concat_vert_dmap, concat_vert),
            concat_color,
            concat_dyn_tri_color,
            concat_dyn_tri_intensity,
            concat_vel,
            concat_uv,
            np.array(concat_rod),
            np.array(concat_tri),
            np.array(concat_tet),
            concat_rod_param,
            concat_tri_param,
            concat_tet_param,
            self._wall,
            self._sphere,
            (rod_vert_start, rod_vert_end),
            (shell_vert_start, shell_vert_end),
            rod_count,
            shell_count,
            np.array(concat_tri_is_collider, dtype=bool),
            np.array(concat_rod_is_collider, dtype=bool),
            global_pinned_vertices if global_pinned_vertices else None,
            static_vert_for_check,
            static_tri_for_check,
            self._surface_map_by_name if self._surface_map_by_name else None,
            concat_rest_vert=concat_rest_vert if has_rest_vert else None,
            rest_vert_mask=rest_vert_mask if has_rest_vert else None,
        )

        for name, obj in dyn_objects:
            if obj._exclude_from_output:
                fixed._excluded_from_output.add(name)

        if velocity_schedules:
            fixed._velocity_schedules = velocity_schedules
        if collision_windows:
            fixed._collision_windows_data = collision_windows

        if len(concat_pin):
            fixed.set_pin(concat_pin)

        if len(concat_static_vert):
            fixed.set_static(
                (np.array(concat_static_vert_dmap), np.array(concat_static_vert)),
                np.array(concat_static_tri),
                np.array(concat_static_color),
                concat_static_param,
                concat_static_transform_anims if concat_static_transform_anims else None,
            )

        if len(concat_stitch_ind) and len(concat_stitch_w):
            fixed.set_stitch(
                np.array(concat_stitch_ind),
                np.array(concat_stitch_w),
            )

        pbar.update(1)
        advance("Building scene: validating constraints...")
        pbar.close()

        return fixed


class Object:
    """The object class.

    An ``Object`` is a placed instance of a registered asset. It carries a
    4x4 transform, per-vertex colors, pins, stitches, and material
    parameters. Instances are created via ``scene.add("<mesh_name>")`` and
    are typically configured through chainable methods.

    Example:
        Chain placement, material, and pinning onto a cloth sheet::

            sheet = scene.add("sheet").at(0, 0.6, 0).jitter()
            sheet.param.set("strain-limit", 0.05)
            sheet.pin(sheet.grab([-1, 0, -1]) + sheet.grab([1, 0, -1]))
    """

    def __init__(self, asset: AssetManager, name: str):
        self._asset = asset
        self._name = name
        self._static = False
        self._param = ParamHolder(object_param(self.obj_type))
        self.clear()

    @property
    def name(self) -> str:
        """Get name of the object.

        Example:
            Read the asset reference name back from an object::

                obj = scene.add("sheet")
                assert obj.name == "sheet"
        """
        return self._name

    @property
    def static(self) -> bool:
        """Get whether the object is static.

        Example:
            Pinning every vertex turns an object into a static collider::

                obj = scene.add("sphere").pin().object
                assert obj.static
        """
        return self._static

    @property
    def param(self) -> ParamHolder:
        """Get the material parameters of the object.

        Returns:
            ParamHolder: The material parameters of the object.

        Example:
            Configure the Young's modulus through the returned holder::

                scene.add("sheet").param.set("young-mod", 1e5)
        """
        return self._param

    @property
    def obj_type(self) -> str:
        """Get the type of the object.

        Returns:
            str: The type of the object, either "rod", "tri", or "tet".

        Example:
            Branch on object topology when iterating the scene::

                for obj in scene.object_dict.values():
                    if obj.obj_type == "tet":
                        obj.param.set("young-mod", 1e6)
        """
        return self._asset.fetch.get_type(self._name)

    @property
    def object_color(self) -> Optional[list[float]]:
        """Get the object color.

        Example:
            Read the RGB color previously assigned via :meth:`color`::

                obj = scene.add("sheet").color(0.9, 0.3, 0.3)
                print(obj.object_color)
        """
        color = self._color
        if color is None:
            return None
        elif isinstance(color, list):
            return color
        else:
            return color.tolist()

    @property
    def position(self) -> list[float]:
        """Get the object translation from the transform matrix.

        Example:
            Read back the translation set by :meth:`at`::

                obj = scene.add("sheet").at(0, 0.6, 0)
                assert obj.position == [0.0, 0.6, 0.0]
        """
        return self._transform[:3, 3].tolist()

    @property
    def object_velocity(self) -> list[float] | np.ndarray:
        """Get the object velocity.

        Example:
            Inspect the initial velocity set via :meth:`velocity`::

                obj = scene.add("sheet").velocity(0, 0, -1)
                print(obj.object_velocity)
        """
        return self._velocity

    @property
    def uv_coords(self) -> Optional[list[np.ndarray]]:
        """Get the UV coordinates.

        Example:
            Check whether the asset carries UV data for texturing::

                obj = scene.add("sheet")
                if obj.uv_coords is not None:
                    print(obj.uv_coords[0].shape)
        """
        return self._uv

    @property
    def dynamic_color(self) -> EnumColor:
        """Get the dynamic color type.

        Example:
            Inspect the dynamic color mode previously selected::

                obj = scene.add("sheet")
                print(obj.dynamic_color)
        """
        return self._dyn_color

    @property
    def dynamic_intensity(self) -> float:
        """Get the dynamic color intensity.

        Example:
            Read the scalar intensity used by dynamic coloring::

                obj = scene.add("sheet")
                print(obj.dynamic_intensity)
        """
        return self._dyn_intensity

    @property
    def pin_list(self) -> list[PinHolder]:
        """Get the list of pin holders.

        Example:
            Iterate pin holders attached to an object::

                obj = scene.add("sheet")
                obj.pin([0, 1, 2])
                for holder in obj.pin_list:
                    print(len(holder.index))
        """
        return self._pin

    def clear(self):
        """Clear the object data.

        Example:
            Reset an object's transform and pin state::

                obj = scene.select("sheet")
                obj.clear()
                obj.at(0, 0.6, 0)
        """
        self._transform = np.eye(4)  # Single 4x4 matrix for all transforms
        self._color: Union[np.ndarray, list[float], None] = None
        self._dyn_color = EnumColor.NONE
        self._dyn_intensity = 1.0
        self._static_color = [0.75, 0.75, 0.75]
        self._default_color = [1.0, 0.85, 0.0]
        self._velocity = [0.0, 0.0, 0.0]
        self._velocity_schedule = []
        self._collision_windows = []
        self._pin: list[PinHolder] = []
        self._normalize = False
        self._stitch = None
        self._uv = None
        self._transform_animation: Optional[TransformAnimation] = None
        self._exclude_from_output = False
        # Flagged by the decoder when the object is a static-moving mesh
        # driven by a pin-shell (either fcurve or UI-op). Propagates into
        # PinData.hide_in_preview so the JupyterLab viewer doesn't draw
        # the all-vertex pin as user-facing pin markers.
        self._is_static_moving = False

    def report(self):
        """Report the object data.

        Example:
            Print a summary of an object after configuring it::

                obj = scene.add("sheet").at(0, 0.6, 0)
                obj.pin(obj.grab([-1, 0, -1]))
                obj.report()
        """
        print("transform:")
        print(self._transform)
        print("color:", self._color)
        print("velocity:", self._velocity)
        print("normalize:", self._normalize)
        self.update_static()
        if self.static:
            print("pin: static")
        else:
            print("pin:", sum([len(p.index) for p in self._pin]))

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the bounding box of the object.

        Returns:
            tuple[np.ndarray, np.ndarray]: The dimensions and center of the bounding box.

        Example:
            Read a sheet's size and center in world space::

                sheet = scene.add("sheet")
                size, center = sheet.bbox()
                print(size, center)
        """
        vert = self.get("V")
        if vert is None:
            raise Exception("vertex does not exist")
        else:
            transformed = self.apply_transform(vert, False)
            max_x, max_y, max_z = np.max(transformed, axis=0)
            min_x, min_y, min_z = np.min(transformed, axis=0)
            return (
                np.array(
                    [
                        max_x - min_x,
                        max_y - min_y,
                        max_z - min_z,
                    ]
                ),
                np.array(
                    [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2]
                ),
            )

    def normalize(self) -> "Object":
        """Normalize the object so that it fits within a unit cube.

        Returns:
            Object: The normalized object.

        Example:
            Normalize a tetrahedral asset before scaling it into place::

                arm = scene.add("armadillo").normalize().scale(0.75)
                arm.at(0, 1, 0)
        """
        if self._normalize:
            raise Exception("already normalized")
        else:
            self._bbox, self._center = self.bbox()
            self._normalize = True
            return self

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get an associated value of the object with respect to the key.

        Args:
            key (str): The key of the value.
        Returns:
            Optional[np.ndarray]: The value associated with the key.

        Example:
            Fetch the rest-pose vertices and face list of an asset::

                sheet = scene.add("sheet")
                V = sheet.get("V")
                F = sheet.get("F")
        """
        if key == "color":
            if self._color is not None:
                return np.array(self._color)
            else:
                if self.static:
                    return np.array(self._static_color)
                else:
                    return np.array(self._default_color)
        elif key == "Ind":
            if self._stitch is not None:
                return self._stitch[0]
            else:
                return None
        elif key == "W":
            if self._stitch is not None:
                return self._stitch[1]
            else:
                return None
        else:
            result = self._asset.fetch.get(self._name)
            if key in result:
                return result[key]
            else:
                return None

    def vertex(self, translate: bool) -> np.ndarray:
        """Get the transformed vertices of the object.

        Args:
            translate (bool): Whether to translate the vertices.

        Returns:
            np.ndarray: The transformed vertices.

        Example:
            Read world-space vertex positions after ``.at`` has been set::

                sheet = scene.add("sheet").at(0, 0.6, 0)
                world_vert = sheet.vertex(True)
        """
        vert = self.get("V")
        if vert is None:
            raise Exception("vertex does not exist")
        else:
            return self.apply_transform(vert, translate)

    def grab(self, direction: list[float], eps: float = 1e-3) -> list[int]:
        """Select vertices that are furthest along a specified direction.

        Args:
            direction (list[float]): The direction vector.
            eps (float, optional): Tolerance (in dot-product units) from the maximum. Defaults to 1e-3.

        Returns:
            list[int]: The indices of the selected vertices.

        Example:
            Pin the two top corners of a sheet to hang it::

                sheet = scene.add("sheet")
                sheet.pin(sheet.grab([-1, 1, 0]) + sheet.grab([1, 1, 0]))
        """
        vert = self.vertex(False)
        val = np.max(np.dot(vert, np.array(direction)))
        return np.where(np.dot(vert, direction) > val - eps)[0].tolist()

    def mat4x4(self, matrix: np.ndarray) -> "Object":
        """Set the full 4x4 transformation matrix directly.

        Replaces the current transform entirely.  The matrix is applied as::

            world_pos = matrix[:3,:3] @ local_pos + matrix[:3,3]

        Args:
            matrix (np.ndarray): A 4x4 transformation matrix.

        Returns:
            Object: The object with the updated transform.

        Example:
            Apply a pre-computed transform imported from Blender::

                import numpy as np
                M = np.eye(4)
                M[:3, 3] = [0, 0.6, 0]
                scene.add("sheet").mat4x4(M)
        """
        self._transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return self

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get the current 4x4 transformation matrix.

        Example:
            Inspect the composed translation, rotation, and scale::

                obj = scene.add("sheet").at(0, 0.6, 0)
                print(obj.transform_matrix)
        """
        return self._transform

    def at(self, x: float, y: float, z: float) -> "Object":
        """Set the translation component of the transform.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.

        Returns:
            Object: The object with the updated position.

        Example:
            Drop a sheet 0.6 units above the origin::

                scene.add("sheet").at(0, 0.6, 0)
        """
        self._transform[:3, 3] = [x, y, z]
        return self

    def jitter(self, r: float = 1e-2) -> "Object":
        """Add random jitter to the translation.

        Args:
            r (float, optional): The jitter magnitude.

        Returns:
            Object: The object with the jittered position.

        Example:
            Break symmetry for a falling armadillo on a trampoline::

                scene.add("armadillo").at(0, 1, 0).jitter().velocity(0, -5, 0)
        """
        self._transform[0, 3] += r * np.random.random()
        self._transform[1, 3] += r * np.random.random()
        self._transform[2, 3] += r * np.random.random()
        return self

    def scale(self, _scale: float) -> "Object":
        """Apply uniform scale to the transform.

        Args:
            _scale (float): The scale factor.

        Returns:
            Object: The object with the updated scale.

        Example:
            Shrink an armadillo to 0.75 of its original size::

                scene.add("armadillo").scale(0.75).at(0, 1, 0)
        """
        S = np.eye(4)
        S[0, 0] = S[1, 1] = S[2, 2] = _scale
        self._transform = self._transform @ S
        return self

    def rotate(self, angle: float, axis: str) -> "Object":
        """Apply rotation around a specified axis to the transform.

        Args:
            angle (float): The rotation angle in degrees.
            axis (str): The rotation axis ('x', 'y', or 'z').

        Returns:
            Object: The object with the updated rotation.

        Example:
            Stand a sheet upright by rotating 90 degrees around x::

                scene.add("sheet").rotate(90, "x").at(0, 0.5, 0)
        """
        theta = angle / 180.0 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4)
        if axis.lower() == "x":
            R[1, 1] = c; R[1, 2] = -s
            R[2, 1] = s; R[2, 2] = c
        elif axis.lower() == "y":
            R[0, 0] = c; R[0, 2] = s
            R[2, 0] = -s; R[2, 2] = c
        elif axis.lower() == "z":
            R[0, 0] = c; R[0, 1] = -s
            R[1, 0] = s; R[1, 1] = c
        else:
            raise Exception("invalid axis")
        pos = self._transform[:3, 3].copy()
        self._transform = R @ self._transform
        self._transform[:3, 3] = pos
        return self

    def move(
        self, delta, t_start: float = 0.0, t_end: float = 1.0
    ) -> "Object":
        """Animate the static object by a translational delta over time.

        Args:
            delta: [dx, dy, dz] translation delta in world space.
            t_start (float): Start time in seconds.
            t_end (float): End time in seconds.

        Returns:
            Object: The object with the animation added.

        Example:
            Slide a static (fully pinned) collider along +x between t=0 and t=2::

                scene.add("sphere").at(-1, 0, 0).pin()
                scene.select("sphere").move([2, 0, 0], t_start=0.0, t_end=2.0)
        """
        delta = np.asarray(delta, dtype=np.float64)
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        self._ensure_transform_animation()
        anim = self._transform_animation
        start_trans = anim.translations[-1].copy()
        start_quat = anim.quaternions[-1].copy()
        start_scale = anim.scales[-1].copy()
        if anim.times[-1] < t_start:
            anim.times.append(t_start)
            anim.translations.append(start_trans.copy())
            anim.quaternions.append(start_quat.copy())
            anim.scales.append(start_scale.copy())
        anim.times.append(t_end)
        anim.translations.append(start_trans + delta)
        anim.quaternions.append(start_quat.copy())
        anim.scales.append(start_scale.copy())
        return self

    def animate_rotate(
        self,
        axis,
        angle: float,
        center=None,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> "Object":
        """Animate the static object by rotating around an axis over time.

        Args:
            axis: [ax, ay, az] rotation axis (will be normalized).
            angle (float): Rotation angle in degrees.
            center: Center of rotation [cx, cy, cz]. Defaults to object centroid.
            t_start (float): Start time in seconds.
            t_end (float): End time in seconds.

        Returns:
            Object: The object with the animation added.

        Example:
            Rotate a pinned static collider 180 degrees around y over 2 seconds::

                roller = scene.add("cylinder").at(0, 0.5, 0).pin()
                scene.select("cylinder").animate_rotate(
                    [0, 1, 0], 180.0, t_start=0.0, t_end=2.0,
                )
        """
        axis = np.asarray(axis, dtype=np.float64)
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        self._ensure_transform_animation()
        anim = self._transform_animation
        start_trans = anim.translations[-1].copy()
        start_quat = anim.quaternions[-1].copy()
        start_scale = anim.scales[-1].copy()
        if center is None:
            center = np.mean(
                _apply_transform_to_verts(anim.local_vert, start_trans, start_quat, start_scale),
                axis=0,
            )
        else:
            center = np.asarray(center, dtype=np.float64)
        rot_quat = _axis_angle_to_quat(axis, angle)
        R = _quat_to_mat3(rot_quat)
        new_trans = center + R @ (start_trans - center)
        new_quat = _quat_multiply(rot_quat, start_quat)
        if anim.times[-1] < t_start:
            anim.times.append(t_start)
            anim.translations.append(start_trans.copy())
            anim.quaternions.append(start_quat.copy())
            anim.scales.append(start_scale.copy())
        anim.times.append(t_end)
        anim.translations.append(new_trans)
        anim.quaternions.append(new_quat)
        anim.scales.append(start_scale.copy())
        return self

    def _ensure_transform_animation(self):
        """Initialize TransformAnimation from current transform if not already set."""
        if self._transform_animation is not None:
            return
        vert = self.get("V")
        if vert is None:
            raise ValueError("Object has no vertices; cannot create transform animation")
        R = self._transform[:3, :3]
        t = self._transform[:3, 3]
        scale_x = np.linalg.norm(R[:, 0])
        scale_y = np.linalg.norm(R[:, 1])
        scale_z = np.linalg.norm(R[:, 2])
        scale = np.array([scale_x, scale_y, scale_z], dtype=np.float64)
        R_norm = R.copy()
        if scale_x > 0:
            R_norm[:, 0] /= scale_x
        if scale_y > 0:
            R_norm[:, 1] /= scale_y
        if scale_z > 0:
            R_norm[:, 2] /= scale_z
        quat = _mat3_to_quat(R_norm)
        self._transform_animation = TransformAnimation(
            local_vert=vert.copy(),
            times=[0.0],
            translations=[t.copy()],
            quaternions=[quat],
            scales=[scale],
        )

    def max(self, dim: str) -> float:
        """Get the maximum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the maximum value along, either "x", "y", or "z".

        Returns:
            float: The maximum coordinate value.

        Example:
            Check that a sheet sits above y=0::

                sheet = scene.add("sheet").at(0, 0.6, 0)
                assert sheet.min("y") >= 0
                print(sheet.max("y"))
        """
        vert = self.vertex(True)
        return np.max([x[{"x": 0, "y": 1, "z": 2}[dim]] for x in vert])

    def min(self, dim: str) -> float:
        """Get the minimum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The minimum coordinate value.

        Example:
            Lift an object so its lowest point sits at y=0::

                obj = scene.add("armadillo")
                obj.at(0, -obj.min("y"), 0)
        """
        vert = self.vertex(True)
        return np.min([x[{"x": 0, "y": 1, "z": 2}[dim]] for x in vert])

    def apply_transform(self, x: np.ndarray, translate: bool) -> np.ndarray:
        """Apply the object's transformation to a set of vertices.

        Args:
            x (np.ndarray): The vertices to transform (N, 3).
            translate (bool): Whether to include the translation component.

        Returns:
            np.ndarray: The transformed vertices (N, 3).

        Example:
            Manually map a batch of points through the object's current transform::

                obj = scene.add("sheet").at(0, 0.6, 0)
                world_pts = obj.apply_transform(obj.get("V"), True)
        """
        if len(x.shape) == 1:
            raise Exception("vertex should be 2D array")
        v = x.copy()
        if self._normalize:
            v = (v - self._center) / np.max(self._bbox)
        M = self._transform
        v = (M[:3, :3] @ v.T).T
        if translate:
            v = v + M[:3, 3]
        return v

    def static_color(self, red: float, green: float, blue: float) -> "Object":
        """Set the static color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated static color.

        Example:
            Tint a fully pinned collider a light gray::

                scene.add("sphere").pin()
                scene.select("sphere").static_color(0.75, 0.75, 0.75)
        """
        self._static_color = [red, green, blue]
        return self

    def default_color(self, red: float, green: float, blue: float) -> "Object":
        """Set the default color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated default color.

        Example:
            Override the auto-assigned hue for a dynamic object::

                scene.add("sheet").default_color(1.0, 0.85, 0.0)
        """
        self._default_color = [red, green, blue]
        return self

    def color(self, red: float, green: float, blue: float) -> "Object":
        """Set the color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated color.

        Example:
            Color a falling armadillo light gray::

                scene.add("armadillo").color(0.75, 0.75, 0.75)
        """
        self._color = [red, green, blue]
        return self

    def vert_color(self, color: np.ndarray) -> "Object":
        """Set the vertex colors of the object.

        Args:
            color (np.ndarray): The vertex colors.

        Returns:
            Object: The object with the updated vertex colors.

        Example:
            Paint each vertex of a sheet with its own RGB::

                import numpy as np
                sheet = scene.add("sheet")
                n = len(sheet.get("V"))
                sheet.vert_color(np.random.rand(n, 3))
        """
        self._color = color
        return self

    def direction_color(self, x: float, y: float, z: float) -> "Object":
        """Set the color along the direction of the object.

        Args:
            x (float): The x-component of the direction.
            y (float): The y-component of the direction.
            z (float): The z-component of the direction.

        Returns:
            Object: The object with the updated color.

        Example:
            Shade a cylinder along its long axis::

                scene.add("cylinder").direction_color(1, 0, 0)
        """
        vertex = self.vertex(False)
        vals = vertex.dot([x, y, z])
        min_val, max_val = np.min(vals), np.max(vals)
        color = np.zeros((len(vertex), 3))
        for i, val in enumerate(vals):
            y = (val - min_val) / (max_val - min_val)
            hue = 240.0 * (1.0 - y) / 360.0
            color[i] = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        return self.vert_color(color)

    def cylinder_color(
        self, center: list[float], direction: list[float], up: list[float]
    ) -> "Object":
        """Set the color along the cylinder direction.

        Args:
            center (list[float]): The center of the cylinder.
            direction (list[float]): The direction of the cylinder.
            up (list[float]): The up vector of the cylinder.

        Returns:
            Object: The object with the updated color.

        Example:
            Apply a cylinder gradient used in the twist demo::

                obj = scene.add("cylinder").cylinder_color(
                    [0, 0, 0], [1, 0, 0], [0, 1, 0],
                )
        """
        ey = np.array(up)
        ex = np.cross(np.array(direction), ey)

        vertex = self.vertex(False) - np.array(center)
        x = np.dot(vertex, ex)
        y = np.dot(vertex, ey)
        angle = np.arctan2(y, x)
        angle = np.mod(angle, 2 * np.pi) / (2 * np.pi)
        color = np.zeros((len(vertex), 3))
        for i, z in enumerate(angle):
            color[i] = colorsys.hsv_to_rgb(z, 0.75, 1.0)
        return self.vert_color(color)

    def dyn_color(self, color: str, intensity: float = 0.75) -> "Object":
        """Set the dynamic color of the object.

        Args:
            color (str): The dynamic color type. Currently only ``"area"`` is supported.
            intensity (float, optional): Blend intensity of the dynamic color. Defaults to 0.75.

        Returns:
            Object: The object with the updated dynamic color.

        Example:
            Highlight stretched triangles on a trampoline sheet::

                scene.add("sheet").dyn_color("area", 1.0)
        """
        if color == "area":
            self._dyn_color = EnumColor.AREA
            self._dyn_intensity = intensity
        else:
            raise Exception("invalid color type")
        return self

    def velocity(self, u: float, v: float, w: float, t: float = 0.0) -> "Object":
        """Set the velocity of the object.

        Args:
            u (float): The velocity in the x-direction.
            v (float): The velocity in the y-direction.
            w (float): The velocity in the z-direction.
            t (float): Time in seconds. 0.0 sets initial velocity; >0 adds a timed override.

        Returns:
            Object: The object with the updated velocity.

        Example:
            Give an armadillo a downward initial velocity of 5 m/s::

                scene.add("armadillo").at(0, 1, 0).velocity(0, -5, 0)
        """
        if self.static:
            raise Exception("object is static")
        if t <= 0.0:
            self._velocity = np.array([u, v, w])
        else:
            self._velocity_schedule.append((t, [u, v, w]))
        return self

    def velocity_schedule(self, schedule: list) -> "Object":
        """Set a list of (time, [vx, vy, vz]) velocity overrides.

        Example:
            Kick an object, then stop it a second later::

                obj = scene.add("armadillo").at(0, 1, 0)
                obj.velocity_schedule([
                    (0.0, [0, -5, 0]),
                    (1.0, [0, 0, 0]),
                ])
        """
        self._velocity_schedule = schedule
        return self

    def collision_windows(self, windows: list) -> "Object":
        """Set collision active time windows: list of (t_start, t_end) pairs.

        Example:
            Enable contact only during t in [0.2, 1.0] and [2.0, 3.0]::

                scene.add("sheet").collision_windows([(0.2, 1.0), (2.0, 3.0)])
        """
        self._collision_windows = windows
        return self

    def update_static(self):
        """Recompute whether the object is static.

        When every vertex is pinned and no pin carries operations, a pull
        strength, or an unpin time, the object is treated as static. The
        result is cached on ``self._static``.

        Example:
            Typically invoked internally by :meth:`Scene.build` after pins
            are finalized, but can be called directly to refresh the cached
            flag after editing pin data in place::

                obj = scene.select("sheet")
                obj.pin()
                obj.update_static()
                assert obj.static
        """
        if not self._pin:
            self._static = False
            return

        for p in self._pin:
            if len(p.operations) > 0 or p.pull_strength or p.unpin_time is not None:
                return

        vert = self.get("V")
        if vert is None:
            self._static = False
            return

        vert_flag = np.zeros(len(vert))
        for p in self._pin:
            for i in p.index:
                vert_flag[i] = 1
        self._static = np.sum(vert_flag) == len(vert)

    def pin(self, ind: Optional[list[int]] = None) -> PinHolder:
        """Set specified vertices as pinned.

        An object with every vertex pinned is a *static collider* —
        its motion is prescribed, not simulated.  Use the returned
        :class:`PinHolder` to animate it via
        :meth:`PinHolder.move_by`, :meth:`PinHolder.move_to`, or
        :meth:`PinHolder.transform_keyframes`.

        Args:
            ind (Optional[list[int]], optional): The indices of the vertices to pin.
            If None, all vertices are pinned. Defaults to None.

        Returns:
            PinHolder: The pin holder.

        Example:
            Static sphere that slides across the scene::

                (scene.add("sphere")
                      .at(-1, 0, 0)
                      .pin()
                      .move_by([8, 0, 0], t_start=0.0, t_end=5.0))
        """
        if ind is None:
            vert: np.ndarray = self.vertex(False)
            ind = list(range(len(vert)))

        holder = PinHolder(self, ind)
        self._pin.append(holder)
        return holder

    def stitch(self, name: str) -> "Object":
        """Apply stitch to the object.

        Args:
            name (str): The name of stitch registered in the asset manager.

        Returns:
            Object: The stitched object.

        Example:
            Attach a glue stitch registered earlier in the asset manager::

                app.asset.add.stitch("glue", stitch_data)
                scene.add("dress").stitch("glue").rotate(-90, "x")
        """
        if self.static:
            raise Exception("object is static")
        else:
            stitch = self._asset.fetch.get(name)
            if "Ind" not in stitch:
                raise Exception("Ind not found in stitch")
            elif "W" not in stitch:
                raise Exception("W not found in stitch")
            else:
                self._stitch = (stitch["Ind"], stitch["W"])
                return self

    def set_uv(self, uv: list[np.ndarray]) -> "Object":
        """Set the UV coordinates of the object.

        Args:
            uv (list[np.ndarray]): The UV coordinates for each face.

        Returns:
            Object: The object with the updated UV coordinates.

        Example:
            Supply per-face UVs for a triangulated sheet::

                import numpy as np
                sheet = scene.add("sheet")
                n_faces = len(sheet.get("F"))
                uv = [np.zeros((3, 2), dtype=np.float32) for _ in range(n_faces)]
                sheet.set_uv(uv)
        """
        if self.obj_type != "tri":
            raise Exception("UV coordinates are only applicable to triangular meshes")
        else:
            self._uv = uv
            return self

    def direction(self, _ex: list[float], _ey: list[float]) -> "Object":
        """Set two orthogonal directions of a shell required for Baraff-Witkin model.

        Args:
            _ex (list[float]): The 3D x-direction vector.
            _ey (list[float]): The 3D y-direction vector.

        Returns:
            Object: The object with the updated direction.

        Example:
            Pin the warp and weft directions of a flat sheet in the xz plane::

                sheet = scene.add("sheet")
                sheet.param.set("model", "baraff-witkin")
                sheet.direction([1, 0, 0], [0, 0, 1])
        """
        vert, tri = self.vertex(False), self.get("F")
        ex = np.array(_ex)
        ex = ex / np.linalg.norm(ex)
        ey = np.array(_ey)
        ey = ey / np.linalg.norm(ey)
        if abs(np.dot(ex, ey)) > EPS:
            raise Exception(f"ex and ey must be orthogonal. ex: {ex}, ey: {ey}")
        elif vert is None:
            raise Exception("vertex does not exist")
        elif tri is None:
            raise Exception("face does not exist")
        else:
            uv = []
            for t in tri:
                a, b, c = vert[t]
                n = np.cross(b - a, c - a)
                n = n / np.linalg.norm(n)
                if abs(np.dot(n, _ex)) > EPS:
                    raise Exception(
                        f"ex must be orthogonal to the face normal. normal: {n}"
                    )
                elif abs(np.dot(n, _ey)) > EPS:
                    raise Exception(
                        f"ey must be orthogonal to the face normal. normal: {n}"
                    )
                uv.append(
                    np.array(
                        [
                            [a.dot(ex), a.dot(ey)],
                            [b.dot(ex), b.dot(ey)],
                            [c.dot(ex), c.dot(ey)],
                        ]
                    )
                )
            self._uv = uv
        return self
