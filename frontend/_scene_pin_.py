# File: _scene_pin_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Pin operations, PinHolder, PinData, SpinData, and pin TOML helpers.

Split out of ``_scene_.py``: these classes describe pin animation
(MoveBy, MoveTo, Spin, Scale, Torque, TransformKeyframes) and the
``PinHolder`` user-facing builder that mirrors a Rust validator.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]


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
        if self.bezier_handles is not None:
            hr, hl = self.bezier_handles
            handles = ((float(hr[0]), float(hr[1])), (float(hl[0]), float(hl[1])))
        else:
            handles = None
        return _rust.scene_move_by_apply(
            np.ascontiguousarray(vertex, dtype=np.float64),
            np.ascontiguousarray(self.delta, dtype=np.float64),
            float(time), float(self.t_start), float(self.t_end),
            self.transition, handles,
        )

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


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
        if self.bezier_handles is not None:
            hr, hl = self.bezier_handles
            handles = ((float(hr[0]), float(hr[1])), (float(hl[0]), float(hl[1])))
        else:
            handles = None
        return _rust.scene_move_to_apply(
            np.ascontiguousarray(vertex, dtype=np.float64),
            np.ascontiguousarray(self.target, dtype=np.float64),
            float(time), float(self.t_start), float(self.t_end),
            self.transition, handles,
        )

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

    def apply(self, vertex: np.ndarray, time: float) -> np.ndarray:
        translations = [
            [float(t[0]), float(t[1]), float(t[2])] for t in self.translations
        ]
        quaternions = [
            [float(q[0]), float(q[1]), float(q[2]), float(q[3])] for q in self.quaternions
        ]
        scales = [
            [float(s[0]), float(s[1]), float(s[2])] for s in self.scales
        ]
        rest = [
            float(self.rest_translation[0]),
            float(self.rest_translation[1]),
            float(self.rest_translation[2]),
        ]
        return _rust.scene_transform_keyframe_apply(
            np.ascontiguousarray(vertex, dtype=np.float64),
            np.ascontiguousarray(self.local_vert, dtype=np.float64),
            [float(t) for t in self.times],
            translations, quaternions, scales,
            list(self.segments), rest, float(time),
        )

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
        return _rust.scene_spin_apply(
            np.ascontiguousarray(vertex, dtype=np.float64),
            [float(self.center[0]), float(self.center[1]), float(self.center[2])],
            [float(self.axis[0]), float(self.axis[1]), float(self.axis[2])],
            float(self.angular_velocity),
            float(self.t_start), float(self.t_end), float(time),
        )

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
        return _rust.scene_scale_apply(
            np.ascontiguousarray(vertex, dtype=np.float64),
            [float(self.center[0]), float(self.center[1]), float(self.center[2])],
            float(self.factor), float(self.t_start), float(self.t_end),
            self.transition, float(time),
        )

    def get_time_range(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)


@dataclass
class TorqueOperation(Operation):
    """Torque operation: applies rotational force around a PCA axis."""

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


def _flatten_bezier_handles(op: "Operation"):
    """Flatten an op's `bezier_handles = ((hr_x, hr_y), (hl_x, hl_y))`
    pair into the `[hr_x, hr_y, hl_x, hl_y]` list the Rust TOML
    formatter writes (and that the solver-side reader picks up). None
    when the op carries no handles, signaling linear fallback at apply
    time.
    """
    h = getattr(op, "bezier_handles", None)
    if h is None:
        return None
    hr, hl = h
    return [float(hr[0]), float(hr[1]), float(hl[0]), float(hl[1])]


def _pin_op_to_toml_dict(op: "Operation") -> dict:
    """Repackage a pin Operation into the descriptor dict consumed by
    `_rust.scene_format_pin_toml`. The Rust kernel walks a flat list of
    these and emits the entire pin/op TOML block in one allocation.
    """
    if isinstance(op, MoveByOperation):
        return {
            "type": "move_by",
            "t_start": float(op.t_start),
            "t_end": float(op.t_end),
            "transition": op.transition,
            "bezier_handles": _flatten_bezier_handles(op),
        }
    if isinstance(op, MoveToOperation):
        return {
            "type": "move_to",
            "t_start": float(op.t_start),
            "t_end": float(op.t_end),
            "transition": op.transition,
            "bezier_handles": _flatten_bezier_handles(op),
        }
    if isinstance(op, SpinOperation):
        return {
            "type": "spin",
            "center_mode": op.center_mode,
            "center": [float(op.center[0]), float(op.center[1]), float(op.center[2])],
            "axis": [float(op.axis[0]), float(op.axis[1]), float(op.axis[2])],
            "angular_velocity": float(op.angular_velocity),
            "t_start": float(op.t_start),
            "t_end": float(op.t_end),
        }
    if isinstance(op, ScaleOperation):
        return {
            "type": "scale",
            "center_mode": op.center_mode,
            "center": [float(op.center[0]), float(op.center[1]), float(op.center[2])],
            "factor": float(op.factor),
            "t_start": float(op.t_start),
            "t_end": float(op.t_end),
            "transition": op.transition,
            "bezier_handles": _flatten_bezier_handles(op),
        }
    if isinstance(op, TorqueOperation):
        return {
            "type": "torque",
            "axis_component": int(op.axis_component),
            "magnitude": float(op.magnitude),
            "hint_vertex": int(op.hint_vertex),
            "t_start": float(op.t_start),
            "t_end": float(op.t_end),
        }
    if isinstance(op, TransformKeyframeOperation):
        rt = op.rest_translation
        return {
            "type": "transform_keyframes",
            "keyframe_count": len(op.times),
            "t_start": float(op.times[0]) if op.times else 0.0,
            "t_end": float(op.times[-1]) if op.times else 0.0,
            "rest_translation": [float(rt[0]), float(rt[1]), float(rt[2])],
        }
    raise TypeError(f"unsupported pin operation type: {type(op).__name__}")


def _pin_to_toml_dict(pin: "PinData") -> dict:
    """Repackage a PinData into the dict consumed by Rust's pin TOML
    formatter."""
    return {
        "operation_count": len(pin.operations),
        "pin_count": len(pin.index),
        "pull_strength": float(pin.pull_strength),
        "unpin_time": float(pin.unpin_time) if pin.unpin_time is not None else None,
        "pin_group_id": pin.pin_group_id if pin.pin_group_id else None,
        "ops": [_pin_op_to_toml_dict(op) for op in pin.operations],
    }


class PinHolder:
    """Class to manage pinning behavior of objects."""

    _pin_counter = 0

    def __init__(self, obj: "Object", indices: list[int]):  # noqa: F821
        """Initialize pin object.

        Args:
            obj (Object): The object to pin.
            indices (list[int]): The indices of the vertices to pin.
        """
        self._obj = obj
        PinHolder._pin_counter += 1
        pin_group_id = f"{obj.name}:pin_{PinHolder._pin_counter}"
        self._data = PinData(
            index=indices,
            operations=[],
            pin_group_id=pin_group_id,
        )
        # Parallel Rust mirror, kept in lockstep with `self._data`.
        # Builder methods route validation through the mirror first;
        # downstream callers (decoder, scene builder) keep reading the
        # Python `PinData` dataclass, so we hand them an authoritative
        # Python view that the Rust holder validated.
        self._rust = _rust.PinHolder(list(indices), pin_group_id)

    def __getstate__(self) -> dict:
        # The Rust mirror is not picklable. _data is canonical; the
        # mirror is a parallel validator for builder calls. After
        # unpickle the holder is read-only (loaded as part of a frozen
        # FixedSession), so __setstate__ rebuilds an empty mirror.
        return {"obj": self._obj, "data": self._data}

    def __setstate__(self, state: dict) -> None:
        self._obj = state["obj"]
        self._data = state["data"]
        self._rust = _rust.PinHolder(self._data.index, self._data.pin_group_id)

    def interp(self, transition: str) -> "PinHolder":
        """Set the transition type for the pinning.

        Args:
            transition (str): The transition type. Supported values are ``"linear"``, ``"smooth"``, and ``"bezier"``. Default is ``"linear"``.

        Returns:
            PinHolder: The pinholder with the updated transition type.
        """
        self._rust.interp(transition)
        self._data.transition = transition
        return self

    def unpin(self, time: float) -> "PinHolder":
        """Unpin the object at a specified time.

        Args:
            time (float): The time at which to unpin the vertices.

        Returns:
            PinHolder: The pinholder with the unpin time set.
        """
        self._rust.unpin(time)
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
        delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape((-1, 3))

        if len(delta_pos) == 1 and len(self.index) > 1:
            delta_pos = np.tile(delta_pos, (len(self.index), 1))
        elif len(delta_pos) != len(self.index):
            raise Exception("delta_pos must have the same length as pin")

        if t_end <= t_start:
            raise Exception("t_end must be greater than t_start")

        # Rust validates length + t_end ordering; we already
        # reshaped above so the call is idempotent.
        self._rust.move_by(
            delta_pos, t_start, t_end, transition, bezier_handles,
        )
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
        local_arr = np.asarray(local_vert, dtype=np.float64)
        translations_arr = [np.asarray(t, dtype=np.float64) for t in translations]
        quaternions_arr = [np.asarray(q, dtype=np.float64) for q in quaternions]
        scales_arr = [np.asarray(s, dtype=np.float64) for s in scales]
        rest_arr = np.asarray(rest_translation, dtype=np.float64)
        self._rust.transform_keyframes(
            local_arr,
            list(times),
            translations_arr,
            quaternions_arr,
            scales_arr,
            list(segments),
            rest_arr,
        )
        self._data.operations.append(
            TransformKeyframeOperation(
                local_vert=local_arr,
                times=list(times),
                translations=translations_arr,
                quaternions=quaternions_arr,
                scales=scales_arr,
                segments=list(segments),
                rest_translation=rest_arr,
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
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape((-1, 3))

        if len(target_pos) == 1:
            # Singleton-target: resolve against the object's current
            # position. This depends on `self._obj`, which lives only
            # on the Python side, so we expand here before forwarding
            # the resolved (N, 3) array to the Rust mirror.
            initial_vertices = self._obj.vertex(False)[self._data.index]
            current_center = np.array(self._obj.position)
            delta = target_pos[0] - current_center
            target_pos = initial_vertices + delta
        elif len(target_pos) != len(self.index):
            raise Exception("target_pos must have the same length as pin")

        if t_end <= t_start:
            raise Exception("t_end must be greater than t_start")

        self._rust.move_to(
            target_pos, t_start, t_end, transition, bezier_handles,
        )
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

        self._rust.scale(scale, t_start, t_end, center_point, center_mode)
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
        self._rust.pull(strength)
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

        self._rust.spin(
            np.array(center, dtype=np.float64),
            np.array(axis, dtype=np.float64),
            angular_velocity, t_start, t_end, center_mode,
        )
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
        self._rust.torque(
            magnitude, axis_component, hint_vertex, t_start, t_end,
        )
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
