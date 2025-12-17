# File: _scene_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import builtins
import colorsys
import os
import pickle
import shutil

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from tqdm import tqdm

from ._asset_ import AssetManager
from ._param_ import ParamHolder, object_param
from ._plot_ import Plot, PlotManager
from ._render_ import MitsubaRenderer, OpenGLRenderer
from ._utils_ import Utils

EPS = 1e-3


class SceneManager:
    """SceneManager class. Use this to manage scenes."""

    def __init__(self, plot: PlotManager | None, asset: AssetManager):
        """Initialize the scene manager."""
        self._plot = plot
        self._asset = asset
        self._scene: dict[str, Scene] = {}

    def create(self, name: str = "") -> "Scene":
        """Create a new scene.

        Create a scene only if the name does not exist. Raise an exception if the name already exists.

        Args:
            name (str): The name of the scene to create. If not provided, it will use the current time as the name.

        Returns:
            Scene: The created scene.
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

        If the scene exists, it will be selected. If it does not exist and create is True, a new scene will be created.

        Args:
            name (str): The name of the scene to select.
            create (bool, optional): Whether to create a new scene if it does not exist. Defaults to True.
        """
        if create and name not in self._scene:
            return self.create(name)
        else:
            return self._scene[name]

    def remove(self, name: str):
        """Remove a scene from the manager.

        Args:
            name (str): The name of the scene to remove.
        """
        if name in self._scene:
            del self._scene[name]

    def clear(self):
        """Clear all the scenes in the manager."""
        self._scene = {}

        """List all the scenes in the manager.

        Returns:
            list[str]: A list of scene names.
        """
        return list(self._scene.keys())


class WallParam:
    """A class to hold wall parameters."""

    def __init__(self):
        self._param = {
            "contact-gap": 1e-3,
            "friction": 0.0,
        }

    def list(self) -> dict[str, float]:
        """List all the parameters for the wall.

        Returns:
            dict[str, float]: A dictionary of wall parameters.
        """
        return self._param

    def set(self, name: str, value: float) -> "WallParam":
        """Set a parameter for the wall."""
        if name not in self._param:
            raise Exception(f"unknown parameter {name}")
        else:
            self._param[name] = value
        return self


class Wall:
    """An invisible wall class."""

    def __init__(self):
        """Initialize the wall."""
        self._entry = []
        self._transition = "linear"
        self._param = WallParam()

    def get_entry(self) -> list[tuple[list[float], float]]:
        """Get a list of time-dependent wall entries.

        Returns:
            list[tuple[list[float], float]]: A list of time-dependent entries, each containing a position and time.
        """
        return self._entry

    def add(self, pos: list[float], normal: list[float]) -> "Wall":
        """Add an invisible wall information.

            pos (list[float]): The position of the wall.
            normal (list[float]): The outer normal of the wall.

        Returns:
            Wall: The invisible wall.
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
        """
        self._check_time(time)
        pos = self._entry[-1][0] + delta
        self._entry.append((pos, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Wall":
        """Move the wall to an absolute position at a specific time.

        Args:
            pos (list[float]): The target position of the wall.
            time (float): The absolute time to move the wall.

        Returns:
            Wall: The invisible wall.
        """
        self._check_time(time)
        self._entry.append((pos, time))
        return self

    def interp(self, transition: str) -> "Wall":
        """Set the transition type for the wall."""
        self._transition = transition
        return self

    @property
    def normal(self) -> list[float]:
        """Get the wall normal."""
        return self._normal

    @property
    def entry(self) -> list[tuple[list[float], float]]:
        """Get the wall entries."""
        return self._entry

    @property
    def transition(self) -> str:
        """Get the wall transition."""
        return self._transition

    @property
    def param(self) -> WallParam:
        """Get the wall parameters."""
        return self._param


class SphereParam:
    """A class to hold wall parameters."""

    def __init__(self):
        self._param = {
            "contact-gap": 1e-3,
            "friction": 0.0,
        }

    def list(self) -> dict[str, float]:
        """List all the parameters for the sphere.

        Returns:
            dict[str, float]: A dictionary of sphere parameters.
        """
        return self._param

    def set(self, name: str, value: float) -> "SphereParam":
        """Set a parameter for the wall."""
        if name not in self._param:
            raise Exception(f"unknown parameter {name}")
        else:
            self._param[name] = value
        return self


class Sphere:
    """An invisible sphere class."""

    def __init__(self):
        """Initialize the sphere."""
        self._entry = []
        self._hemisphere = False
        self._invert = False
        self._transition = "linear"
        self._param = SphereParam()

    def hemisphere(self) -> "Sphere":
        """Turn the sphere into a hemisphere, so the half of the sphere top becomes empty, like a bowl."""
        self._hemisphere = True
        return self

    def invert(self) -> "Sphere":
        """Invert the sphere, so the inside becomes empty and the outside becomes solid."""
        self._invert = True
        return self

    def interp(self, transition: str) -> "Sphere":
        """Set the transition type for the sphere."""
        self._transition = transition
        return self

    def get_entry(self) -> list[tuple[list[float], float, float]]:
        """Get the time-dependent sphere entries."""
        return self._entry

    def add(self, pos: list[float], radius: float) -> "Sphere":
        """Add an invisible sphere information.

        Args:
            pos (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.

        Returns:
            Sphere: The sphere.
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
            Spere: The sphere.
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
        """
        self._check_time(time)
        pos = self._entry[-1][0] + delta
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
        """
        self._check_time(time)
        pos = self._entry[-1][0]
        self._entry.append((pos, radius, time))
        return self

    @property
    def entry(self) -> list[tuple[list[float], float, float]]:
        """Get the sphere entries."""
        return self._entry

    @property
    def is_hemisphere(self) -> bool:
        """Get whether sphere is hemisphere."""
        return self._hemisphere

    @property
    def is_inverted(self) -> bool:
        """Get whether sphere is inverted."""
        return self._invert

    @property
    def transition(self) -> str:
        """Get the sphere transition."""
        return self._transition

    @property
    def param(self) -> SphereParam:
        """Get the sphere parameters."""
        return self._param


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

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Apply position delta to current vertex position over time range."""
        if time < self.t_start:
            return vertex
        elif time >= self.t_end:
            return vertex + self.delta
        else:
            # Interpolate delta and add to current position
            progress = (time - self.t_start) / (self.t_end - self.t_start)
            if self.transition == "smooth":
                progress = progress * progress * (3.0 - 2.0 * progress)
            return vertex + self.delta * progress

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class MoveToOperation(Operation):
    """Move operation with absolute target positions and time range."""

    target: np.ndarray
    t_start: float
    t_end: float
    transition: str = "linear"

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        """Overwrite vertex positions with target over time range."""
        if time < self.t_start:
            return vertex
        elif time >= self.t_end:
            return self.target.copy()
        else:
            # Interpolate from current position to target
            progress = (time - self.t_start) / (self.t_end - self.t_start)
            if self.transition == "smooth":
                progress = progress * progress * (3.0 - 2.0 * progress)
            return vertex * (1 - progress) + self.target * progress

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class SpinOperation(Operation):
    """Spin operation with rotation parameters."""

    center: np.ndarray
    axis: np.ndarray
    angular_velocity: float
    t_start: float
    t_end: float

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
class PinData:
    """Represents pinning data for a set of vertices."""

    index: list[int]
    operations: list[Operation]
    unpin_time: Optional[float] = None
    transition: str = "linear"
    pull_strength: float = 0.0


class PinHolder:
    """Class to manage pinning behavior of objects."""

    def __init__(self, obj: "Object", indices: list[int]):
        """Initialize pin object.

        Args:
            obj (Object): The object to pin.
            indices (list[int]): The indices of the vertices to pin.
        """
        self._obj = obj
        self._data = PinData(
            index=indices,
            operations=[],
        )

    def interp(self, transition: str) -> "PinHolder":
        """Set the transition type for the pinning.

        Args:
            transition (str): The transition type. Currently supported: "smooth", "linear". Default is "linear".

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
        self, delta_pos, t_start: float = 0.0, t_end: float = 1.0
    ) -> "PinHolder":
        """Move the object by a positional delta over a specified time range.

        Args:
            delta_pos (list[float]): The positional delta to apply.
            t_start (float): The start time. Defaults to 0.0.
            t_end (float): The end time. Defaults to 1.0.

        Returns:
            PinHolder: The pinholder with the updated position.
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
                transition=self._data.transition,
            )
        )
        return self

    def move_to(
        self, target_pos, t_start: float = 0.0, t_end: float = 1.0
    ) -> "PinHolder":
        """Move the object to absolute target positions over a specified time range.

        Args:
            target_pos (list[float]): The target positions (absolute).
            t_start (float): The start time. Defaults to 0.0.
            t_end (float): The end time. Defaults to 1.0.

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
                transition=self._data.transition,
            )
        )
        return self

    def scale(
        self,
        scale: float,
        t_start: float = 0.0,
        t_end: float = 1.0,
        center: Optional[list[float]] = None,
    ) -> "PinHolder":
        """Scale the object by a specified factor over a time range.

        Interpolates the scale factor from 1.0 to the target scale over [t_start, t_end].

        Args:
            scale (float): The target scaling factor.
            t_start (float): The start time of the scaling. Defaults to 0.0.
            t_end (float): The end time of the scaling. Defaults to 1.0.
            center (Optional[list[float]]): The center point for scaling. If not provided, uses the origin (0, 0, 0).

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
            )
        )
        return self

    def pull(self, strength: float = 1.0) -> "PinHolder":
        """Pull the object at specified vertices.

        Args:
            strength (float, optional): The pull strength. Defaults to 1.0.

        Returns:
            PinHolder: The pinholder with the pinned and pulled vertices.
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
    ) -> "PinHolder":
        """Add a spin operation to the pin.

        Args:
            center: Center of rotation. Defaults to [0, 0, 0].
            axis: Rotation axis. Defaults to [0, 1, 0].
            angular_velocity: Rotation speed in degrees/second.
            t_start: Start time of spin.
            t_end: End time of spin.

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
            )
        )
        return self

    @property
    def data(self) -> PinData | None:
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
    """A fixed scene class."""

    def __init__(
        self,
        plot: PlotManager | None,
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
        merge: bool,
    ):
        """Initialize the fixed scene.

        Args:
            plot (PlotManager): The plot manager.
            name (str): The name of the scene.
            map (dict[str, int]): The mapping of vetex indices to their original indices.
            displacement (np.ndarray): The displacement of the vertices.
            vert (np.ndarray, np.ndarray): The vertices of the scene. The first array is the displacement map reference.
            color (np.ndarray): The colors of the vertices.
            dyn_face_color (list[EnumColor]): The dynamic face colors.
            dyn_face_intensity (list[float]): The dynamic face color intensity.
            vel (np.ndarray): The velocities of the vertices.
            uv (np.ndarray): The UV coordinates of the vertices.
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
        self._uv = uv
        self._rod = rod
        self._tri = tri
        self._tet = tet
        self._rod_param = rod_param
        self._tri_param = tri_param
        self._tet_param = tet_param
        self._pin: list[PinData] = []
        self._spin: list[SpinData] = []
        self._static_vert = (np.zeros(0, dtype=np.uint32), np.zeros(0))
        self._static_color = np.zeros((0, 0))
        self._static_tri = np.zeros((0, 0))
        self._stitch_ind = np.zeros((0, 0))
        self._stitch_w = np.zeros((0, 0))
        self._static_param = {}
        self._wall = wall
        self._sphere = sphere
        self._rod_vert_range = rod_vert_range
        self._shell_vert_range = shell_vert_range
        self._rod_count = rod_count
        self._shell_count = shell_count
        self._has_dyn_color = any(entry != EnumColor.NONE for entry in dyn_face_color)

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

        if len(self._tri):
            self._area = np.zeros(len(self._tri))
            _compute_area(self._vert[1], self._tri, self._area)
        else:
            self._area = np.zeros(0)

        if self._has_dyn_color:
            sum = np.zeros(len(self._vert[0])) + 0.0001
            rows, cols, vals = [], [], []
            for i, f in enumerate(self._tri):
                for j in f:
                    rows.append(j)
                    cols.append(i)
                    vals.append(1.0)
                    sum[j] += 1
            self._face_to_vert_mat = csr_matrix(
                (vals, (rows, cols)), shape=(len(sum), len(self._tri))
            )
            self._face_to_vert_mat = self._face_to_vert_mat.multiply(1.0 / sum[:, None])
        else:
            self._face_to_vert_mat = None

        # Detect and merge duplicate vertices using hash-based approach
        if merge:
            vertex_hash_map = {}

            # Compute minimal edge length from triangles and edges
            min_edge_length = float("inf")

            # Check edges from rods
            for edge in self._rod:
                v0 = self._vert[1][edge[0]] + self._displacement[self._vert[0][edge[0]]]
                v1 = self._vert[1][edge[1]] + self._displacement[self._vert[0][edge[1]]]
                edge_length = np.linalg.norm(v1 - v0)
                if edge_length > 0:
                    min_edge_length = min(min_edge_length, edge_length)

            # Check edges from triangles
            for tri in self._tri:
                for i in range(3):
                    v0 = (
                        self._vert[1][tri[i]]
                        + self._displacement[self._vert[0][tri[i]]]
                    )
                    v1 = (
                        self._vert[1][tri[(i + 1) % 3]]
                        + self._displacement[self._vert[0][tri[(i + 1) % 3]]]
                    )
                    edge_length = np.linalg.norm(v1 - v0)
                    if edge_length > 0:
                        min_edge_length = min(min_edge_length, edge_length)

            # Set epsilon based on minimal edge length
            if min_edge_length == float("inf"):
                # No edges found, use default epsilon
                epsilon = 1e-5
                scale_factor = 1e5
            else:
                epsilon = float(0.1 * min_edge_length)
                scale_factor = 1.0 / epsilon

            # Build set of vertices used in tetrahedra to exclude them
            tet_vertices = set()
            for tet in self._tet:
                tet_vertices.update(tet)

            # Calculate actual positions and hash them (excluding tetrahedra vertices)
            for i in range(len(self._vert[0])):
                # Skip vertices that belong to tetrahedra
                if i in tet_vertices:
                    continue

                actual_pos = self._vert[1][i] + self._displacement[self._vert[0][i]]

                # Create hash from scaled coordinates using large prime numbers
                hash_val = (
                    int(actual_pos[0] * scale_factor) * 73856093
                    + int(actual_pos[1] * scale_factor) * 19349663
                    + int(actual_pos[2] * scale_factor) * 83492791
                )

                if hash_val not in vertex_hash_map:
                    vertex_hash_map[hash_val] = []
                vertex_hash_map[hash_val].append((i, actual_pos))

            # Find duplicate groups (handles transitive duplicates)
            duplicate_groups = []  # List of sets of duplicate vertex indices
            vertex_to_group = {}  # Map vertex index to its duplicate group

            for _, vertices in vertex_hash_map.items():
                if len(vertices) > 1:
                    # Check all pairs in this hash bucket
                    for i in range(len(vertices)):
                        for j in range(i + 1, len(vertices)):
                            idx1, pos1 = vertices[i]
                            idx2, pos2 = vertices[j]

                            # Check if positions are exactly the same (within epsilon)
                            if np.allclose(pos1, pos2, rtol=0, atol=epsilon):
                                # Find or create groups for these vertices
                                group1 = vertex_to_group.get(idx1)
                                group2 = vertex_to_group.get(idx2)

                                if group1 is None and group2 is None:
                                    # Create new group
                                    new_group = {idx1, idx2}
                                    duplicate_groups.append(new_group)
                                    vertex_to_group[idx1] = new_group
                                    vertex_to_group[idx2] = new_group
                                elif group1 is None:
                                    # Add idx1 to group2
                                    assert group2 is not None  # Type hint for pyright
                                    group2.add(idx1)
                                    vertex_to_group[idx1] = group2
                                elif group2 is None:
                                    # Add idx2 to group1
                                    group1.add(idx2)
                                    vertex_to_group[idx2] = group1
                                elif group1 is not group2:
                                    # Merge two groups
                                    group1.update(group2)
                                    duplicate_groups.remove(group2)
                                    for v in group2:
                                        vertex_to_group[v] = group1

            # Build merge mapping and perform merging
            if duplicate_groups:
                print(
                    f"Found {len(duplicate_groups)} groups of duplicate vertices to merge:"
                )
                total_duplicates = builtins.sum(
                    len(group) - 1 for group in duplicate_groups
                )
                print(f"  Total duplicate vertices to remove: {total_duplicates}")

                # Create old to new index mapping
                old_to_new = {}
                new_index = 0

                # First pass: assign new indices to kept vertices
                for i in range(len(self._vert[0])):
                    group = vertex_to_group.get(i)
                    if group is None:
                        # Not a duplicate, keep it
                        old_to_new[i] = new_index
                        new_index += 1
                    else:
                        # Part of a duplicate group
                        representative = min(group)
                        if i == representative:
                            # This is the representative, keep it
                            old_to_new[i] = new_index
                            new_index += 1

                # Second pass: map duplicates to their representatives
                for i in range(len(self._vert[0])):
                    if i not in old_to_new:
                        # This is a duplicate, find its representative
                        group = vertex_to_group[i]
                        representative = min(group)
                        old_to_new[i] = old_to_new[representative]

                # Create new vertex arrays
                new_vert_dmap = []
                new_vert = []
                new_color = []
                new_vel = []

                for i in range(len(self._vert[0])):
                    if vertex_to_group.get(i) is None or i == min(vertex_to_group[i]):
                        # Keep this vertex
                        new_vert_dmap.append(self._vert[0][i])
                        new_vert.append(self._vert[1][i])
                        new_color.append(self._color[i])
                        new_vel.append(self._vel[i])

                # Update map_by_name
                for name, indices in self._map_by_name.items():
                    self._map_by_name[name] = [old_to_new[i] for i in indices]

                # Update rods
                new_rod = []
                new_rod_param = {key: [] for key in self._rod_param}
                removed_rods = 0

                for i, edge in enumerate(self._rod):
                    remapped = [old_to_new[v] for v in edge]
                    if remapped[0] != remapped[1]:  # Not degenerate
                        new_rod.append(remapped)
                        for key, values in self._rod_param.items():
                            if values:
                                new_rod_param[key].append(values[i])
                    else:
                        removed_rods += 1

                # Update triangles
                new_tri = []
                new_tri_param = {key: [] for key in self._tri_param}
                new_dyn_face_color = []
                new_dyn_face_intensity = []
                removed_tris = 0

                for i, tri in enumerate(self._tri):
                    remapped = [old_to_new[v] for v in tri]
                    if len(set(remapped)) == 3:  # All vertices are different
                        new_tri.append(remapped)
                        for key, values in self._tri_param.items():
                            if values:
                                new_tri_param[key].append(values[i])
                        new_dyn_face_color.append(self._dyn_face_color[i])
                        new_dyn_face_intensity.append(self._dyn_face_intensity[i])
                    else:
                        removed_tris += 1

                # Update tetrahedra
                new_tet = []
                new_tet_param = {key: [] for key in self._tet_param}
                removed_tets = 0

                for i, tet in enumerate(self._tet):
                    remapped = [old_to_new[v] for v in tet]
                    if len(set(remapped)) == 4:  # All vertices are different
                        new_tet.append(remapped)
                        for key, values in self._tet_param.items():
                            if values:
                                new_tet_param[key].append(values[i])
                    else:
                        removed_tets += 1

                # Update pin data
                for pin in self._pin:
                    pin.index = list({old_to_new[i] for i in pin.index})

                # Update stitch data
                new_stitch_ind = []
                new_stitch_w = []
                for ind, w in zip(self._stitch_ind, self._stitch_w, strict=False):
                    remapped = [old_to_new[v] for v in ind]
                    if len(set(remapped)) == len(remapped):
                        new_stitch_ind.append(remapped)
                        new_stitch_w.append(w)

                # Replace data with merged versions
                self._vert = (
                    np.array(new_vert_dmap, dtype=np.uint32),
                    np.array(new_vert),
                )
                self._color = np.array(new_color)
                self._vel = np.array(new_vel)
                self._rod = (
                    np.array(new_rod) if new_rod else np.zeros((0, 2), dtype=np.uint64)
                )
                self._tri = (
                    np.array(new_tri) if new_tri else np.zeros((0, 3), dtype=np.uint64)
                )
                self._tet = (
                    np.array(new_tet) if new_tet else np.zeros((0, 4), dtype=np.uint64)
                )
                self._rod_param = new_rod_param
                self._tri_param = new_tri_param
                self._tet_param = new_tet_param
                self._dyn_face_color = new_dyn_face_color
                self._dyn_face_intensity = new_dyn_face_intensity
                self._stitch_ind = (
                    np.array(new_stitch_ind) if new_stitch_ind else self._stitch_ind
                )
                self._stitch_w = (
                    np.array(new_stitch_w) if new_stitch_w else self._stitch_w
                )

                # Report results
                print("Vertex merge complete:")
                print(
                    f"  Vertices: {len(old_to_new)} -> {len(new_vert)} (removed {total_duplicates})"
                )
                if removed_rods > 0:
                    print(f"  Removed {removed_rods} degenerate edges")
                if removed_tris > 0:
                    print(f"  Removed {removed_tris} degenerate triangles")
                if removed_tets > 0:
                    print(f"  Removed {removed_tets} degenerate tetrahedra")

                # Re-compute face areas and face-to-vertex matrix
                if len(self._tri):
                    self._area = np.zeros(len(self._tri))
                    _compute_area(self._vert[1], self._tri, self._area)

                    if self._has_dyn_color:
                        sum = np.zeros(len(self._vert[0])) + 0.0001
                        rows, cols, vals = [], [], []
                        for i, f in enumerate(self._tri):
                            for j in f:
                                rows.append(j)
                                cols.append(i)
                                vals.append(1.0)
                                sum[j] += 1
                        self._face_to_vert_mat = csr_matrix(
                            (vals, (rows, cols)), shape=(len(sum), len(self._tri))
                        )
                        self._face_to_vert_mat = self._face_to_vert_mat.multiply(
                            1.0 / sum[:, None]
                        )

    @property
    def tri_param(self) -> dict[str, list[Any]]:
        """Get the triangle parameters."""
        return self._tri_param

    def report(self) -> "FixedScene":
        """Print a summary of the scene."""
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

        if self._plot is not None and self._plot.is_jupyter_notebook():
            df = pd.DataFrame(data)
            html = df.to_html(classes="table", index=False)
            display(HTML(html))
        else:
            print(data)
        return self

    def color(self, vert: np.ndarray, hint: Optional[dict] = None) -> np.ndarray:
        """Compute the color of the scene given the vertex array.

        Args:
            vert (np.ndarray): The vertices of the scene.
            hint (dict, optional): The hint for the color computation. Defaults to {}.

        Returns:
            color (np.ndarray): The vertex color of the scene.
        """
        if hint is None:
            hint = {}
        if self._has_dyn_color:
            assert self._face_to_vert_mat is not None
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
            intensity = self._face_to_vert_mat.dot(intensity)
            color = (1.0 - intensity[:, None]) * self._color + intensity[
                :, None
            ] * self._face_to_vert_mat.dot(face_color)
            return color
        else:
            return self._color

    def vertex(self, transform: bool = True) -> np.ndarray:
        """Get the vertices of the scene.

        Args:
            transform (bool, optional): Whether to transform the vertices. Defaults to True.

        Returns:
            np.ndarray: The vertices of the scene.
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

        Export the scene to a mesh file. The vertices must be explicitly provided.

        Args:
            vert (np.ndarray): The vertices of the scene.
            color (np.ndarray): The colors of the vertices.
            path (str): The path to the mesh file. Supported formats are `.ply`, `.obj`
            include_static (bool, optional): Whether to include the static mesh. Defaults to True.
            args (dict, optional): Additional arguments passed to a renderer.
            delete_exist (bool, optional): Whether to delete the existing file. Defaults to False.

        Returns:
            FixedScene: The fixed scene.
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

        # Skip rendering on Windows (pyrender doesn't work in headless mode)
        if not skip_render and not os.path.exists(image_path):
            if Utils.ci_name() is not None:
                args["width"] = 320
                args["height"] = 240
            if "renderer" in args:
                if args["renderer"] == "mitsuba":
                    assert shutil.which("mitsuba") is not None
                    renderer = MitsubaRenderer(args)
                elif args["renderer"] == "opengl":
                    renderer = OpenGLRenderer(args)
                else:
                    raise Exception("unsupported renderer")
            else:
                renderer = OpenGLRenderer(args)

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
        """

        steps = 14
        pbar = tqdm(total=steps, desc="build session", ncols=70)

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
            pickle.dump(self._map_by_name, f)
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
            f.write("\n")

            for i, pin in enumerate(self._pin):
                f.write(f"[pin-{i}]\n")
                f.write(f"operation_count = {len(pin.operations)}\n")
                f.write(f"pin = {len(pin.index)}\n")
                f.write(f"pull = {float(pin.pull_strength)}\n")
                if pin.unpin_time is not None:
                    f.write(f"unpin_time = {float(pin.unpin_time)}\n")
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
                        f.write(f"center_x = {float(op.center[0])}\n")
                        f.write(f"center_y = {float(op.center[1])}\n")
                        f.write(f"center_z = {float(op.center[2])}\n")
                        f.write(f"factor = {float(op.factor)}\n")
                        f.write(f"t_start = {float(op.t_start)}\n")
                        f.write(f"t_end = {float(op.t_end)}\n")
                        f.write(f'transition = "{op.transition}"\n')
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
        self._color.astype(np.float32).tofile(os.path.join(bin_path, "color.bin"))
        self._vel.astype(np.float32).tofile(os.path.join(bin_path, "vel.bin"))
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
        """
        vert = self._vert[1] + self._displacement[self._vert[0]]
        return (np.max(vert, axis=0), np.min(vert, axis=0))

    def center(self) -> np.ndarray:
        """Compute the area-weighted center of the scene.

        Returns:
            np.ndarray: The area-weighted center of the scene.
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
            pin_data (list[PinData]): A list of pinning data.
        """
        self._pin = pin

    def set_spin(self, spin: list[SpinData]):
        """Set the spinning data of all the objects.

        Args:
            spin_data (list[SpinData]): A list of spinning data.
        """
        self._spin = spin

    def set_static(
        self,
        vert: tuple[np.ndarray, np.ndarray],
        tri: np.ndarray,
        color: np.ndarray,
        param: dict[str, list[Any]],
    ):
        """Set the static mesh data.

        Args:
            vert (np.ndarray, np.ndarray): The vertices of the static mesh. The first array is the displacement map reference.
            tri (np.ndarray): The triangle elements of the static mesh.
            color (np.ndarray): The colors of the static mesh.
        """
        self._static_vert = vert
        self._static_tri = tri
        self._static_color = color
        self._static_param = param

    def set_stitch(self, ind: np.ndarray, w: np.ndarray):
        """Set the stitch data.

        Args:
            ind (np.ndarray): The stitch indices.
            w (np.ndarray): The stitch weights.
        """
        self._stitch_ind = ind
        self._stitch_w = w

    def time(self, time: float) -> np.ndarray:
        """Compute the vertex positions at a specific time.

        Args:
            time (float): The time to compute the vertex positions.

        Returns:
            np.ndarray: The vertex positions at the specified time.
        """
        vert = self._vert[1].copy()

        for pin in self._pin:
            # Apply all operations in strict order
            for op in pin.operations:
                vert[pin.index] = op.apply(vert[pin.index], time)

        vert += time * self._vel
        vert += self._displacement[self._vert[0]]
        return vert

    def check_intersection(self) -> "FixedScene":
        """Check for self-intersections and intersections with the static mesh.

        Returns:
            FixedScene: The fixed scene.
        """
        import open3d as o3d

        if len(self._vert) and len(self._tri):
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self._vert),
                o3d.utility.Vector3iVector(self._tri),
            )
            if o3d_mesh.is_self_intersecting():
                print("WARNING: mesh is self-intersecting")
            if len(self._static_vert) and len(self._static_tri):
                o3d_static = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(self._static_vert),
                    o3d.utility.Vector3iVector(self._static_tri),
                )
                if o3d_static.is_self_intersecting():
                    print("WARNING: static mesh is self-intersecting")
                if o3d_static.is_intersecting(o3d_mesh):
                    print("WARNING: mesh is intersecting with static mesh")
        return self

    def preview(
        self,
        vert: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        show_slider: bool = True,
        engine: str = "threejs",
    ) -> Optional["Plot"]:
        """Preview the scene.

        Args:
            vert (Optional[np.ndarray], optional): The vertices to preview. Defaults to None.
            options (dict, optional): The options for the plot. Defaults to {}.
            show_slider (bool, optional): Whether to show the time slider. Defaults to True.
            engine (str, optional): The rendering engine. Defaults to "pythreejs".

        Returns:
            Optional[Plot]: The plot object if in a Jupyter notebook, otherwise None.
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

            if len(self._static_vert[1]):
                static_vert = (
                    self._static_vert[1] + self._displacement[self._static_vert[0]]
                )
                static_color = np.zeros_like(static_vert)
                static_color[:, :] = self._static_color
                if len(tri):
                    tri = np.vstack([tri, self._static_tri + len(vert)])
                else:
                    tri = self._static_tri + len(vert)
                vert = np.vstack([vert, static_vert])
                color = np.vstack([color, static_color])
                assert len(color) == len(vert)

            if options["stitch"] and len(self._stitch_ind) and len(self._stitch_w):
                stitch_vert, stitch_edge = [], []
                for ind, w in zip(self._stitch_ind, self._stitch_w, strict=False):
                    x0, y0, y1 = vert[ind[0]], vert[ind[1]], vert[ind[2]]
                    w0, w1 = w[0], w[1]
                    idx0 = len(stitch_vert) + len(vert)
                    idx1 = idx0 + 1
                    stitch_vert.append(x0)
                    stitch_vert.append(w0 * y0 + w1 * y1)
                    stitch_edge.append([idx0, idx1])
                stitch_vert = np.array(stitch_vert)
                stitch_edge = np.array(stitch_edge)
                stitch_color = np.tile(np.array([1.0, 1.0, 1.0]), (len(stitch_vert), 1))
                vert = np.vstack([vert, stitch_vert])
                edge = np.vstack([edge, stitch_edge]) if len(edge) else stitch_edge
                color = np.vstack([color, stitch_color])

            if options["pin"] and self._pin:
                options["pts_scale"] = np.sqrt(self._average_tri_area())
                pts = []
                for pin in self._pin:
                    pts.extend(pin.index)
                pts = np.array(pts)

            plotter.plot(vert, color, tri, edge, pts, options)

            has_vel = np.linalg.norm(self._vel) > 0
            if show_slider and (self._pin or has_vel):
                max_time = 0
                if self._pin:
                    # Find max time across all operations
                    for pin in self._pin:
                        for op in pin.operations:
                            if isinstance(op, (MoveByOperation, MoveToOperation)):
                                max_time = max(max_time, op.t_end)
                            elif isinstance(op, SpinOperation):
                                if op.t_end == float("inf"):
                                    max_time = max(max_time, 1.0)
                                else:
                                    max_time = max(max_time, op.t_end)
                            elif isinstance(op, ScaleOperation):
                                max_time = max(max_time, op.t_end)
                if has_vel:
                    max_time = max(max_time, 1.0)
                if max_time > 0:

                    def update(time=0):
                        vert = self.time(time)
                        plotter.update(vert)

                    from ipywidgets import interact

                    interact(update, time=(0, max_time, 0.01))
            return plotter
        else:
            return None


class SceneInfo:
    def __init__(self, name: str, scene: "Scene"):
        self._scene = scene
        self.name = name


class InvisibleAdder:
    def __init__(self, scene: "Scene"):
        self._scene = scene

    def sphere(self, position: list[float], radius: float) -> Sphere:
        """Add an invisible sphere to the scene.

        Args:
            position (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.
        Returns:
            Sphere: The invisible sphere.
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
        """
        wall = Wall().add(position, normal)
        self._scene.wall_list.append(wall)
        return wall


class ObjectAdder:
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
            return obj


class Scene:
    """A scene class."""

    def __init__(self, name: str, plot: PlotManager | None, asset: AssetManager):
        self._name = name
        self._plot = plot
        self._asset = asset
        self._object: dict[str, Object] = {}
        self._sphere: list[Sphere] = []
        self._wall: list[Wall] = []
        self.add = ObjectAdder(self)  #: ObjectAdder: The object adder.
        self.info = SceneInfo(name, self)  #: SceneInfo: The scene information.

    def clear(self) -> "Scene":
        """Clear all objects from the scene.

        Returns:
            Scene: The cleared scene.
        """
        self._object.clear()
        return self

    def select(self, name: str) -> "Object":
        """Select an object from the scene by its name.

        Returns:
            Object: The selected object.
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
            axis (str): The axis to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The maximum vertex coordinate along the specified axis.
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
        """Get the list of spheres."""
        return self._sphere

    @property
    def wall_list(self) -> list[Wall]:
        """Get the list of walls."""
        return self._wall

    @property
    def object_dict(self) -> dict[str, "Object"]:
        """Get the object dictionary."""
        return self._object

    @property
    def asset_manager(self) -> AssetManager:
        """Get the asset manager."""
        return self._asset

    def build(self, merge: bool = False) -> FixedScene:
        """Build the fixed scene from the current scene.

        Args:
            merge (bool, optional): Whether to merge duplicate vertices. Defaults to False.

        Returns:
            FixedScene: The built fixed scene.
        """
        pbar = tqdm(total=11, desc="build scene", ncols=70)
        for _, obj in self._object.items():
            obj.update_static()

        concat_count = 0
        dyn_objects = [
            (name, obj) for name, obj in self._object.items() if not obj.static
        ]
        n = len(dyn_objects)
        for i, (_, obj) in enumerate(dyn_objects):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.75, 1.0)
            if obj.object_color is None:
                obj.default_color(r, g, b)

        def add_entry(
            map,
            entry,
        ):
            nonlocal concat_count
            for e in entry:
                for vi in e:
                    if map[vi] == -1:
                        map[vi] = concat_count
                        concat_count += 1

        map_by_name = {}
        for name, obj in dyn_objects:
            vert = obj.get("V")
            if vert is not None:
                map_by_name[name] = [-1] * len(vert)

        pbar.update(1)
        for name, obj in dyn_objects:
            if obj.get("T") is None:
                map = map_by_name[name]
                edge = obj.get("E")
                if edge is not None:
                    add_entry(
                        map,
                        edge,
                    )
        rod_vert_start, rod_vert_end = 0, concat_count

        for name, obj in dyn_objects:
            if obj.get("T") is None:
                map, tri = map_by_name[name], obj.get("F")
                if tri is not None:
                    add_entry(
                        map,
                        tri,
                    )
        shell_vert_start, shell_vert_end = rod_vert_end, concat_count

        pbar.update(1)
        for name, obj in dyn_objects:
            map, tri = map_by_name[name], obj.get("F")
            if tri is not None:
                add_entry(
                    map,
                    tri,
                )

        pbar.update(1)
        for name, obj in dyn_objects:
            vert = obj.get("V")
            if vert is not None:
                map = map_by_name[name]
                for i in range(len(vert)):
                    if map[i] == -1:
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
            result = elm.copy()
            for i in range(len(elm)):
                result[i] = [map[vi] for vi in elm[i]]
            return result

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

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            vert = obj.vertex(False)
            if vert is not None:
                concat_vert[map] = vert
                concat_vert_dmap[map] = [dmap[name]] * len(map)
                concat_vel[map] = obj.object_velocity
                concat_color[map] = obj.get("color")

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            if obj.obj_type == "rod":
                edge = obj.get("E")
                t = vec_map(map, edge)
                concat_rod.extend(t)
                extend_param(obj.param, concat_rod_param, len(t))
        rod_count = len(concat_rod)

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet, tri = obj.get("T"), obj.get("F")
            if tri is not None and tet is None:
                t = vec_map(map, tri)
                concat_tri.extend(t)
                if obj.uv_coords is not None:
                    concat_uv.extend(obj.uv_coords)
                else:
                    concat_uv.extend([np.zeros((2, 3), dtype=np.float32)] * len(t))
                concat_dyn_tri_color.extend([obj.dynamic_color] * len(t))
                concat_dyn_tri_intensity.extend([obj.dynamic_intensity] * len(t))
                extend_param(obj.param, concat_tri_param, len(t))
        shell_count = len(concat_tri)

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet, tri = obj.get("T"), obj.get("F")
            if tet is not None and tri is not None:
                t = vec_map(map, tri)
                concat_tri.extend(t)
                concat_dyn_tri_color.extend([obj.dynamic_color] * len(t))
                concat_dyn_tri_intensity.extend([obj.dynamic_intensity] * len(t))
                extend_param(
                    obj.param,
                    concat_tri_param,
                    len(t),
                )

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            tet = obj.get("T")
            if tet is not None:
                t = vec_map(map, tet)
                concat_tet.extend(t)
                extend_param(
                    obj.param,
                    concat_tet_param,
                    len(t),
                )

        pbar.update(1)
        for name, obj in dyn_objects:
            map = map_by_name[name]
            for p in obj.pin_list:
                concat_pin.append(
                    PinData(
                        index=[map[vi] for vi in p.index],
                        operations=p.operations,
                        unpin_time=p.unpin_time,
                        pull_strength=p.pull_strength,
                        transition=p.transition,
                    )
                )
            stitch_ind = obj.get("Ind")
            stitch_w = obj.get("W")
            if stitch_ind is not None and stitch_w is not None:
                concat_stitch_ind.extend(vec_map(map, stitch_ind))
                concat_stitch_w.extend(stitch_w)

        pbar.update(1)
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
        pbar.update(1)

        for key in ["model"]:
            concat_rod_param[key] = []
            concat_static_param[key] = []

        for key in ["poiss-rat"]:
            concat_rod_param[key] = []

        for key in ["strain-limit", "shrink"]:
            concat_rod_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []

        for key in ["friction", "contact-gap", "contact-offset", "bend"]:
            concat_tet_param[key] = []

        for key in ["young-mod", "poiss-rat", "bend", "density"]:
            concat_static_param[key] = []

        for key in ["length-factor"]:
            concat_tri_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []

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
            merge,
        )

        if len(concat_pin):
            fixed.set_pin(concat_pin)

        if len(concat_static_vert):
            fixed.set_static(
                (np.array(concat_static_vert_dmap), np.array(concat_static_vert)),
                np.array(concat_static_tri),
                np.array(concat_static_color),
                concat_static_param,
            )

        if len(concat_stitch_ind) and len(concat_stitch_w):
            fixed.set_stitch(
                np.array(concat_stitch_ind),
                np.array(concat_stitch_w),
            )

        pbar.close()
        return fixed


class Object:
    """The object class."""

    def __init__(self, asset: AssetManager, name: str):
        self._asset = asset
        self._name = name
        self._static = False
        self._param = ParamHolder(object_param(self.obj_type))
        self.clear()

    @property
    def name(self) -> str:
        """Get name of the object."""
        return self._name

    @property
    def static(self) -> bool:
        """Get whether the object is static."""
        return self._static

    @property
    def param(self) -> ParamHolder:
        """Get the material parameters of the object.

        Returns:
            ObjectParam: The material parameters of the object.
        """
        return self._param

    @property
    def obj_type(self) -> str:
        """Get the type of the object.

        Returns:
            str: The type of the object, either "rod", "tri", or "tet".
        """
        return self._asset.fetch.get_type(self._name)

    @property
    def object_color(self) -> Optional[list[float]]:
        """Get the object color."""
        if self._color is None:
            return None
        elif isinstance(self._color, np.ndarray):
            return self._color.tolist()
        else:
            return self._color

    @property
    def position(self) -> list[float]:
        """Get the object position."""
        return self._at

    @property
    def object_velocity(self) -> list[float] | np.ndarray:
        """Get the object velocity."""
        return self._velocity

    @property
    def uv_coords(self) -> Optional[list[np.ndarray]]:
        """Get the UV coordinates."""
        return self._uv

    @property
    def dynamic_color(self) -> EnumColor:
        """Get the dynamic color type."""
        return self._dyn_color

    @property
    def dynamic_intensity(self) -> float:
        """Get the dynamic color intensity."""
        return self._dyn_intensity

    @property
    def pin_list(self) -> list[PinHolder]:
        """Get the list of pin holders."""
        return self._pin

    def clear(self):
        """Clear the object data."""
        self._at = [0.0, 0.0, 0.0]
        self._scale = 1.0
        self._rotation = np.eye(3)
        self._color = None
        self._dyn_color = EnumColor.NONE
        self._dyn_intensity = 1.0
        self._static_color = [0.75, 0.75, 0.75]
        self._default_color = [1.0, 0.85, 0.0]
        self._velocity = [0.0, 0.0, 0.0]
        self._pin: list[PinHolder] = []
        self._normalize = False
        self._stitch = None
        self._uv = None

    def report(self):
        """Report the object data."""
        print("at:", self._at)
        print("scale:", self._scale)
        print("rotation:")
        print(self._rotation)
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
        """Normalize the object  so that it fits within a unit cube.

        Returns:
            Object: The normalized object.
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
        """
        vert = self.get("V")
        if vert is None:
            raise Exception("vertex does not exist")
        else:
            return self.apply_transform(vert, translate)

    def grab(self, direction: list[float], eps: float = 1e-3) -> list[int]:
        """Grab vertices max towards a specified direction.

        Args:
            direction (list[float]): The direction vector.
            eps (float, optional): The distance threshold.

        Returns:
            list[int]: The indices of the grabbed vertices.
        """
        vert = self.vertex(False)
        val = np.max(np.dot(vert, np.array(direction)))
        return np.where(np.dot(vert, direction) > val - eps)[0].tolist()

    def at(self, x: float, y: float, z: float) -> "Object":
        """Set the position of the object.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.

        Returns:
            Object: The object with the updated position.
        """
        self._at = [x, y, z]
        return self

    def jitter(self, r: float = 1e-2) -> "Object":
        """Add random jitter to the position of the object.

        Args:
            r (float, optional): The jitter magnitude.

        Returns:
            Object: The object with the jittered position.
        """
        dx = np.random.random()
        dy = np.random.random()
        dz = np.random.random()
        self._at[0] += r * dx
        self._at[1] += r * dy
        self._at[2] += r * dz
        return self

    def scale(self, _scale: float) -> "Object":
        """Set the scale of the object.

        Args:
            _scale (float): The scale factor.

        Returns:
            Object: The object with the updated scale.
        """
        self._scale = _scale
        return self

    def rotate(self, angle: float, axis: str) -> "Object":
        """Rotate the object around a specified axis.

        Args:
            angle (float): The rotation angle in degrees.
            axis (str): The rotation axis ('x', 'y', or 'z').

        Returns:
            Object: The object with the updated rotation.
        """
        theta = angle / 180.0 * np.pi
        if axis.lower() == "x":
            self._rotation = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
                @ self._rotation
            )
        elif axis.lower() == "y":
            self._rotation = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
                @ self._rotation
            )
        elif axis.lower() == "z":
            self._rotation = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
                @ self._rotation
            )
        else:
            raise Exception("invalid axis")
        return self

    def max(self, dim: str) -> float:
        """Get the maximum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the maximum value along, either "x", "y", or "z".

        Returns:
            float: The maximum coordinate value.
        """
        vert = self.vertex(True)
        return np.max([x[{"x": 0, "y": 1, "z": 2}[dim]] for x in vert])

    def min(self, dim: str) -> float:
        """Get the minimum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The minimum coordinate value.
        """
        vert = self.vertex(True)
        return np.min([x[{"x": 0, "y": 1, "z": 2}[dim]] for x in vert])

    def apply_transform(self, x: np.ndarray, translate: bool) -> np.ndarray:
        """Apply the object's transformation to a set of vertices.

        Args:
            x (np.ndarray): The vertices to transform.
            translate (bool, optional): Whether to translate the vertices.

        Returns:
            np.ndarray: The transformed vertices.
        """
        if len(x.shape) == 1:
            raise Exception("vertex should be 2D array")
        else:
            x = x.transpose()
        if self._normalize:
            x = (x - self._center) / np.max(self._bbox)
        x = self._rotation @ x
        x = x * self._scale
        if translate:
            x += np.array(self._at).reshape((3, 1))
        return x.transpose()

    def static_color(self, red: float, green: float, blue: float) -> "Object":
        """Set the static color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated static color.
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
        """
        self._color = [red, green, blue]
        return self

    def vert_color(self, color: np.ndarray) -> "Object":
        """Set the vertex colors of the object.

        Args:
            color (np.ndarray): The vertex colors.

        Returns:
            Object: The object with the updated vertex colors.
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

        Aergs:
            center (list[float]): The center of the cylinder.
            direction (list[float]): The direction of the cylinder.
            up (list[float]): The up vector of the cylinder.

        Returns:
            Object: The object with the updated color.
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
            color (str): The dynamic color type.

        Returns:
            Object: The object with the updated dynamic color.
        """
        if color == "area":
            self._dyn_color = EnumColor.AREA
            self._dyn_intensity = intensity
        else:
            raise Exception("invalid color type")
        return self

    def velocity(self, u: float, v: float, w: float) -> "Object":
        """Set the velocity of the object.
        If the object is static, an exception is raised.

        Args:
            u (float): The velocity in the x-direction.
            v (float): The velocity in the y-direction.
            w (float): The velocity in the z-direction.

        Returns:
            Object: The object with the updated velocity.
        """
        if self.static:
            raise Exception("object is static")
        else:
            self._velocity = np.array([u, v, w])
            return self

    def update_static(self):
        """Check if the object is static.
        When all the vertices are pinned and the object is not moving,
        it is considered static.

        Returns:
            bool: True if the object is static, False otherwise.
        """
        if not self._pin:
            self._static = False
            return

        for p in self._pin:
            if len(p.operations) > 0 or p.pull_strength:
                return False

        vert = self.get("V")
        if vert is None:
            self._static = False
            return

        vert_flag = np.zeros(len(vert))
        for p in self._pin:
            for i in p.index:
                vert_flag[i] = 1
        self._static = np.sum(vert_flag) == len(vert)

    def pin(self, ind: list[int] | None = None) -> PinHolder:
        """Set specified vertices as pinned.

        Args:
            ind (Optional[list[int]], optional): The indices of the vertices to pin.
            If None, all vertices are pinned. Defaults to None.

        Returns:
            PinHolder: The pin holder.
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
