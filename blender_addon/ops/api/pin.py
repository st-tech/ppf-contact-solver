"""``_Pin`` proxy: a pinned vertex group bound to a dynamics group.

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ...core.utils import get_vertices_in_group, parse_vertex_index, set_linear_interpolation
from ...models.collection_utils import safe_update_index
from .._api_markers import blender_api


# ---------------------------------------------------------------------------
# Pin proxy
# ---------------------------------------------------------------------------

@blender_api
class _Pin:
    """A pinned vertex group bound to a dynamics group.

    Created via ``Group.create_pin(object_name, vertex_group_name)``.
    Every mutating method returns ``self`` so operations chain.

    Example::

        pin = group.create_pin("Cloth", "hem")
        pin.move(delta=(0, 0, 1.0), frame=60)  # lift hem over 60 frames
        pin.unpin(frame=120)                   # release at frame 120
    """

    def __init__(self, group_uuid: str, object_name: str, vertex_group_name: str):
        object.__setattr__(self, "_group_uuid", group_uuid)
        object.__setattr__(self, "_object_name", object_name)
        object.__setattr__(self, "_vertex_group_name", vertex_group_name)
        object.__setattr__(self, "_initial_keyframed", False)
        object.__setattr__(self, "_unpin_frame", None)
        # Resolve UUID at construction so subsequent lookups are
        # rename-safe within the same session.
        from ...core.uuid_registry import get_or_create_object_uuid
        obj = bpy.data.objects.get(object_name)
        object.__setattr__(
            self, "_object_uuid",
            get_or_create_object_uuid(obj) if obj else "",
        )

    @property
    @blender_api
    def object_name(self) -> str:
        """Name of the mesh object this pin belongs to.

        Example::

            pin = group.create_pin("Cloth", "hem")
            print(pin.object_name)  # "Cloth"
        """
        return object.__getattribute__(self, "_object_name")

    @property
    @blender_api
    def vertex_group_name(self) -> str:
        """Name of the vertex group this pin targets.

        Example::

            pin = group.create_pin("Cloth", "hem")
            print(pin.vertex_group_name)  # "hem"
        """
        return object.__getattribute__(self, "_vertex_group_name")

    def _find_pin_item(self):
        """Look up the PinVertexGroupItem for this pin via UUID."""
        from ...models.groups import get_group_by_uuid
        group = get_group_by_uuid(
            bpy.context.scene,
            object.__getattribute__(self, "_group_uuid"),
        )
        if group is None:
            return None, None
        obj_uuid = object.__getattribute__(self, "_object_uuid")
        vg_name = object.__getattribute__(self, "_vertex_group_name")
        for pin_item in group.pin_vertex_groups:
            if pin_item.object_uuid == obj_uuid:
                from ...models.groups import decode_vertex_group_identifier
                _, item_vg = decode_vertex_group_identifier(pin_item.name)
                if item_vg == vg_name:
                    return group, pin_item
        return group, None

    @blender_api
    def pull(self, strength: float = 1.0) -> "_Pin":
        """Use pull force instead of hard pin constraint.

        Pull allows the vertices to move but applies a restoring
        force toward their target position.

        Args:
            strength: Pull force strength (default 1.0).

        Returns:
            ``self`` for chaining.

        Example::

            group.create_pin("Cloth", "shoulder").pull(strength=2.5)
        """
        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            pin_item.use_pull = True
            pin_item.pull_strength = strength
        return self

    @blender_api
    def spin(self, axis: tuple[float, float, float] = (1, 0, 0),
             angular_velocity: float = 360.0,
             flip: bool = False,
             center: tuple[float, float, float] | None = None,
             center_mode: str | None = None,
             center_direction: tuple[float, float, float] | None = None,
             center_vertex: int | None = None,
             frame_start: int = 1, frame_end: int = 60,
             transition: str = "LINEAR") -> "_Pin":
        """Add a spin operation to this pin.

        Args:
            axis: Rotation axis vector.
            angular_velocity: Degrees per second.
            flip: Reverse spin direction.
            center: Center of rotation (for ABSOLUTE mode).
            center_mode: ``"CENTROID"``, ``"ABSOLUTE"``, ``"MAX_TOWARDS"``, or
                ``"VERTEX"``.  If ``None``, inferred from other args
                (``None`` center → ``"CENTROID"``).
            center_direction: Direction for ``MAX_TOWARDS`` mode.
            center_vertex: Vertex index for ``VERTEX`` mode.
            frame_start: Start frame.
            frame_end: End frame.
            transition: ``"LINEAR"`` or ``"SMOOTH"``.

        Returns:
            ``self`` for chaining.

        Example::

            # Spin about the centroid at 180 deg/s for frames 1-60
            pin.spin(axis=(0, 0, 1), angular_velocity=180.0)
            # Spin about an absolute world-space pivot
            pin.spin(axis=(0, 1, 0), center=(0, 0, 1),
                     frame_start=30, frame_end=90)
        """
        if center_mode is None:
            if center_vertex is not None:
                center_mode = "VERTEX"
            elif center_direction is not None:
                center_mode = "MAX_TOWARDS"
            elif center is not None:
                center_mode = "ABSOLUTE"
            else:
                center_mode = "CENTROID"

        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            op = pin_item.operations.add()
            op.op_type = "SPIN"
            op.spin_center_mode = center_mode
            if center is not None:
                op.spin_center = center
            if center_direction is not None:
                op.spin_center_direction = center_direction
            if center_vertex is not None:
                op.spin_center_vertex = center_vertex
            op.spin_axis = axis
            op.spin_angular_velocity = angular_velocity
            op.spin_flip = flip
            op.frame_start = frame_start
            op.frame_end = frame_end
            op.transition = transition
            pin_item.operations.move(len(pin_item.operations) - 1, 0)
        return self

    @blender_api
    def scale(self, factor: float = 1.0,
              center: tuple[float, float, float] | None = None,
              center_mode: str | None = None,
              center_direction: tuple[float, float, float] | None = None,
              center_vertex: int | None = None,
              frame_start: int = 1, frame_end: int = 60,
              transition: str = "LINEAR") -> "_Pin":
        """Add a scale operation to this pin.

        Args:
            factor: Scale factor.
            center: Center point (for ``ABSOLUTE`` mode).
            center_mode: ``"CENTROID"``, ``"ABSOLUTE"``, ``"MAX_TOWARDS"``, or
                ``"VERTEX"``.  If ``None``, inferred from other args
                (``None`` center → ``"CENTROID"``).
            center_direction: Direction for ``MAX_TOWARDS`` mode.
            center_vertex: Vertex index for ``VERTEX`` mode.
            frame_start: Start frame.
            frame_end: End frame.
            transition: ``"LINEAR"`` or ``"SMOOTH"``.

        Returns:
            ``self`` for chaining.

        Example::

            # Shrink to 50% over frames 1-60 about the centroid
            pin.scale(factor=0.5, transition="SMOOTH")
        """
        if center_mode is None:
            if center_vertex is not None:
                center_mode = "VERTEX"
            elif center_direction is not None:
                center_mode = "MAX_TOWARDS"
            elif center is not None:
                center_mode = "ABSOLUTE"
            else:
                center_mode = "CENTROID"

        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            op = pin_item.operations.add()
            op.op_type = "SCALE"
            op.scale_center_mode = center_mode
            op.scale_factor = factor
            if center is not None:
                op.scale_center = center
            if center_direction is not None:
                op.scale_center_direction = center_direction
            if center_vertex is not None:
                op.scale_center_vertex = center_vertex
            op.frame_start = frame_start
            op.frame_end = frame_end
            op.transition = transition
            pin_item.operations.move(len(pin_item.operations) - 1, 0)
        return self

    @blender_api
    def torque(self, magnitude: float = 1.0,
               axis_component: str = "PC3",
               flip: bool = False,
               frame_start: int = 1, frame_end: int = 60) -> "_Pin":
        """Add a torque operation to this pin.

        Applies a rotational force around a PCA-computed axis.

        Args:
            magnitude: Torque in N·m.
            axis_component: ``"PC1"`` (major), ``"PC2"`` (middle), or
                ``"PC3"`` (minor).
            flip: Reverse torque direction.
            frame_start: Start frame.
            frame_end: End frame.

        Returns:
            ``self`` for chaining.

        Example::

            pin.torque(magnitude=2.0, axis_component="PC1",
                       frame_start=1, frame_end=30)
        """
        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            op = pin_item.operations.add()
            op.op_type = "TORQUE"
            op.torque_magnitude = magnitude
            op.torque_axis_component = axis_component
            op.torque_flip = flip
            op.frame_start = frame_start
            op.frame_end = frame_end
            pin_item.operations.move(len(pin_item.operations) - 1, 0)
        return self

    @blender_api
    def move_by(self, delta: tuple[float, float, float] = (0, 0, 0),
                frame_start: int = 1, frame_end: int = 60,
                transition: str = "LINEAR") -> "_Pin":
        """Ramp a translation of the pinned vertices over a frame range.

        Args:
            delta: ``(dx, dy, dz)`` offset.
            frame_start: Start frame.
            frame_end: End frame.
            transition: ``"LINEAR"`` or ``"SMOOTH"``.

        Returns:
            ``self`` for chaining.

        Example::

            # Lift 1.0m along +Z between frames 10 and 90
            pin.move_by(delta=(0, 0, 1.0),
                        frame_start=10, frame_end=90,
                        transition="SMOOTH")
        """
        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            op = pin_item.operations.add()
            op.op_type = "MOVE_BY"
            op.delta = delta
            op.frame_start = frame_start
            op.frame_end = frame_end
            op.transition = transition
            pin_item.operations.move(len(pin_item.operations) - 1, 0)
        return self

    @blender_api
    def unpin(self, frame: int) -> "_Pin":
        """Mark this pin to be released at the given frame.

        Sets the duration on the underlying UI property so the encoder
        and clear logic are aware.  Also prevents future ``move(frame=N)``
        calls where *N* >= *frame*.

        Args:
            frame: Frame number at which the pin is released.

        Returns:
            ``self`` for chaining.

        Example::

            pin.move(delta=(0, 0, 1), frame=60).unpin(frame=120)
        """
        object.__setattr__(self, "_unpin_frame", frame)

        # Update the PinVertexGroupItem in the UI
        _, pin_item = self._find_pin_item()
        if pin_item is not None:
            pin_item.use_pin_duration = True
            pin_item.pin_duration = frame
        return self

    @blender_api
    def move(
        self,
        delta: tuple[float, float, float] = (0, 0, 0),
        frame: int | None = None,
    ) -> "_Pin":
        """Move pin vertices by *delta* and optionally keyframe at *frame*.

        On the first call with *frame*, the current vertex positions are
        automatically keyframed at the current scene frame before any
        movement is applied.  Ignored if *frame* >= the unpin frame.

        Args:
            delta: ``(dx, dy, dz)`` offset to apply (default no movement).
            frame: Frame number to keyframe at.  ``None`` means no keyframe.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If the target object is missing, not a mesh, or
                the vertex group does not exist on it.

        Example::

            pin = group.create_pin("Cloth", "hem")
            pin.move(delta=(0, 0, 1.0), frame=60)   # auto-keyframes start
            pin.move(delta=(0.5, 0, 0), frame=120)  # adds another keyframe
        """
        unpin_frame = object.__getattribute__(self, "_unpin_frame")
        if frame is not None and unpin_frame is not None and frame >= unpin_frame:
            return self

        object_uuid = object.__getattribute__(self, "_object_uuid")
        vertex_group_name = object.__getattribute__(self, "_vertex_group_name")

        if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")

        from ...core.uuid_registry import get_object_by_uuid
        obj = get_object_by_uuid(object_uuid) if object_uuid else None
        if not obj or obj.type != "MESH":
            raise ValueError(f"Object for pin not found or not a mesh")
        vg = obj.vertex_groups.get(vertex_group_name)
        if not vg:
            raise ValueError(f"Vertex group '{vertex_group_name}' not found")

        saved_frame = bpy.context.scene.frame_current

        # Auto-keyframe initial position on first timed move
        if frame is not None and not object.__getattribute__(self, "_initial_keyframed"):
            self._insert_keyframe(obj, vg)
            object.__setattr__(self, "_initial_keyframed", True)

        if frame is not None:
            bpy.context.scene.frame_current = frame

        # Apply movement
        if delta != (0, 0, 0):
            d = Vector(delta)
            for idx in get_vertices_in_group(obj, vg):
                obj.data.vertices[idx].co += d

        # Insert keyframe at target frame
        if frame is not None:
            self._insert_keyframe(obj, vg)
            bpy.context.scene.frame_current = saved_frame
            from ...core.animation import save_pin_keyframes
            save_pin_keyframes(bpy.context)
        return self

    def _insert_keyframe(self, obj, vg):
        """Insert a positional keyframe for this pin's vertices with LINEAR interpolation."""
        for idx in get_vertices_in_group(obj, vg):
            obj.data.vertices[idx].keyframe_insert(data_path="co")

        if obj.data.animation_data and obj.data.animation_data.action:
            set_linear_interpolation(obj.data.animation_data.action)

    @blender_api
    def clear_keyframes(self) -> "_Pin":
        """Delete all positional keyframes for this pin's vertices.

        Returns:
            ``self`` for chaining.

        Example::

            pin = group.create_pin("Cloth", "hem")
            pin.move(delta=(0, 0, 1.0), frame=60)
            pin.clear_keyframes()  # wipe the animation, keep the pin
        """
        from ...core.utils import _get_fcurves
        from ...core.uuid_registry import get_object_by_uuid

        object_uuid = object.__getattribute__(self, "_object_uuid")
        vertex_group_name = object.__getattribute__(self, "_vertex_group_name")

        obj = get_object_by_uuid(object_uuid) if object_uuid else None
        if not obj or obj.type != "MESH":
            return self
        vg = obj.vertex_groups.get(vertex_group_name)
        if not vg:
            return self

        pin_indices = set(get_vertices_in_group(obj, vg))

        if not obj.data.animation_data or not obj.data.animation_data.action:
            return self
        fcurves = _get_fcurves(obj.data.animation_data.action)
        to_remove = []
        for fc in fcurves:
            if fc.data_path.startswith("vertices[") and ".co" in fc.data_path:
                idx = parse_vertex_index(fc.data_path)
                if idx is not None and idx in pin_indices:
                    to_remove.append(fc)
        for fc in to_remove:
            fcurves.remove(fc)
        return self

    @blender_api
    def delete(self) -> None:
        """Remove this pin from its group.

        Raises:
            ValueError: If the owning group or pin item can no longer be
                found (for example, after ``solver.clear()``).

        Example::

            pin = group.create_pin("Cloth", "hem")
            pin.delete()  # remove the pin entry from the group
        """
        group_uuid = object.__getattribute__(self, "_group_uuid")
        object_uuid = object.__getattribute__(self, "_object_uuid")
        vertex_group_name = object.__getattribute__(self, "_vertex_group_name")

        from ...models.groups import (
            get_group_by_uuid,
            decode_vertex_group_identifier,
            invalidate_overlays,
        )

        group = get_group_by_uuid(bpy.context.scene, group_uuid)
        if group is None:
            raise ValueError(f"Group '{group_uuid}' not found")

        # Match by UUID + vg_name (rename-safe) instead of composite name string
        for i in range(len(group.pin_vertex_groups)):
            pin_item = group.pin_vertex_groups[i]
            if pin_item.object_uuid != object_uuid:
                continue
            _, item_vg = decode_vertex_group_identifier(pin_item.name)
            if item_vg != vertex_group_name:
                continue
            group.pin_vertex_groups.remove(i)
            group.pin_vertex_groups_index = safe_update_index(
                group.pin_vertex_groups_index, len(group.pin_vertex_groups)
            )
            invalidate_overlays()
            return
        raise ValueError(
            f"Pin '{vertex_group_name}' not found in group"
        )

    def __repr__(self):
        return (
            f"Pin({self._object_name!r}, {self._vertex_group_name!r})"
        )
