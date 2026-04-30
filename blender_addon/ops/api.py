"""Object-oriented API for the ZOZO Contact Solver.

Usage in Blender scripts::

    from zozo_contact_solver import solver

    # Scene parameters (attribute access)
    solver.param.step_size = 0.004
    solver.param.gravity = (0, 0, -9.8)
    solver.param.project_name = "my_project"

    # Group creation returns a Group proxy
    shell = solver.create_group("Cloth", type="SHELL")
    shell.add("Plane")
    shell.param.shell_density = 1.0
    shell.param.friction = 0.5

    # Pin creation returns a Pin proxy; methods are chainable
    # Initial position is auto-keyframed on first move(frame=...)
    left = shell.create_pin("Plane", "left")
    left.move(delta=(0, 0, 1.0), frame=60)

    # Connection (falls through to bpy.ops)
    solver.connect()
"""

import bpy  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ..core.utils import get_vertices_in_group, parse_vertex_index, set_linear_interpolation
from ..models.collection_utils import (
    generate_unique_name,
    safe_update_index,
    sort_keyframes_by_frame,
    validate_no_duplicate_frame,
)
from ..models.groups import encode_vertex_group_identifier, get_addon_data
from ._api_markers import blender_api


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
        from ..core.uuid_registry import get_or_create_object_uuid
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
        from ..models.groups import get_group_by_uuid
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
                from ..models.groups import decode_vertex_group_identifier
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

        from ..core.uuid_registry import get_object_by_uuid
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
            from ..core.animation import save_pin_keyframes
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
        from ..core.utils import _get_fcurves
        from ..core.uuid_registry import get_object_by_uuid

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

        from ..models.groups import (
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


# ---------------------------------------------------------------------------
# Group proxy
# ---------------------------------------------------------------------------

_PARAM_PROPS = {
    "solid_model", "shell_model",
    "shell_density", "solid_density", "rod_density",
    "shell_young_modulus", "solid_young_modulus", "rod_young_modulus",
    "shell_poisson_ratio", "solid_poisson_ratio",
    "friction",
    "contact_gap", "contact_gap_rat",
    "contact_offset", "contact_offset_rat",
    "enable_strain_limit", "strain_limit",
    "enable_inflate", "inflate_pressure",
    "enable_plasticity", "plasticity", "plasticity_threshold",
    "enable_bend_plasticity", "bend_plasticity", "bend_plasticity_threshold",
    "bend_rest_angle_source",
    "bend", "shrink", "shrink_x", "shrink_y",
    "stitch_stiffness",
}


@blender_api
class _ParamProxy:
    """Proxy for material and simulation parameters on a group.

    Accessed via :attr:`Group.param`.  Attribute access is whitelisted:
    reading or writing a name outside the whitelist raises
    :class:`AttributeError`.

    Whitelisted attributes:

    - **Solver model**: ``solid_model``, ``shell_model``
    - **Density**: ``solid_density``, ``shell_density``, ``rod_density``
    - **Young's modulus**: ``solid_young_modulus``, ``shell_young_modulus``,
      ``rod_young_modulus``
    - **Poisson ratio**: ``solid_poisson_ratio``, ``shell_poisson_ratio``
    - **Contact**: ``friction``, ``contact_gap``, ``contact_gap_rat``,
      ``contact_offset``, ``contact_offset_rat``
    - **Strain limit**: ``enable_strain_limit``, ``strain_limit``
    - **Inflation**: ``enable_inflate``, ``inflate_pressure``
    - **Plasticity**: ``enable_plasticity``, ``plasticity``,
      ``plasticity_threshold``
    - **Bend plasticity**: ``enable_bend_plasticity``, ``bend_plasticity``,
      ``bend_plasticity_threshold``, ``bend_rest_angle_source``
    - **Shell-specific**: ``bend``, ``shrink``, ``shrink_x``, ``shrink_y``,
      ``stitch_stiffness``

    Example::

        # Solver model
        group.param.solid_model = "ARAP"
        group.param.shell_model = "ARAP"

        # Density (kg/m^3 for solid/shell, kg/m for rod)
        group.param.solid_density = 1000.0
        group.param.shell_density = 0.3
        group.param.rod_density = 0.05

        # Young's modulus (Pa) and Poisson ratio
        group.param.solid_young_modulus = 1.0e6
        group.param.shell_young_modulus = 5.0e5
        group.param.rod_young_modulus = 1.0e7
        group.param.solid_poisson_ratio = 0.45
        group.param.shell_poisson_ratio = 0.30

        # Contact
        group.param.friction = 0.5
        group.param.contact_gap = 0.001
        group.param.contact_gap_rat = 0.1
        group.param.contact_offset = 0.002
        group.param.contact_offset_rat = 0.2

        # Strain limit
        group.param.enable_strain_limit = True
        group.param.strain_limit = 1.05

        # Inflation
        group.param.enable_inflate = True
        group.param.inflate_pressure = 100.0

        # Plasticity (shells and tets only)
        group.param.enable_plasticity = True
        group.param.plasticity = 0.3
        group.param.plasticity_threshold = 0.2
        group.param.enable_bend_plasticity = True
        group.param.bend_plasticity = 0.5
        group.param.bend_plasticity_threshold = 0.1
        group.param.bend_rest_angle_source = "REST"

        # Shell-specific
        group.param.bend = 1.0e-4
        group.param.shrink = 0.98
        group.param.shrink_x = 0.99
        group.param.shrink_y = 0.97
        group.param.stitch_stiffness = 5.0e4
    """

    def __init__(self, group_proxy: "_Group"):
        object.__setattr__(self, "_group_proxy", group_proxy)

    def __setattr__(self, key, value):
        if key not in _PARAM_PROPS:
            raise AttributeError(f"Unknown material parameter '{key}'")
        gp = object.__getattribute__(self, "_group_proxy")
        group = gp._get_group()
        setattr(group, key, value)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

    def __getattr__(self, key):
        if key not in _PARAM_PROPS:
            raise AttributeError(f"Unknown material parameter '{key}'")
        gp = object.__getattribute__(self, "_group_proxy")
        group = gp._get_group()
        return getattr(group, key)


@blender_api
class _Group:
    """A dynamics group proxy.

    Created via :meth:`Solver.create_group`.  Material parameters are
    accessed via :attr:`param`.  Every mutating method returns ``self``
    so operations chain.

    Example::

        group = solver.create_group("Shirt", type="SHELL")
        group.add("Shirt").set_overlay_color(0.9, 0.2, 0.1)
        group.param.friction = 0.5
        group.param.shell_density = 1.0
    """

    def __init__(self, uuid: str):
        object.__setattr__(self, "_uuid", uuid)
        object.__setattr__(self, "_param", _ParamProxy(self))

    @property
    @blender_api
    def uuid(self) -> str:
        """The UUID of this group.  Stable across renames.

        Example::

            group = solver.create_group("Shirt", type="SHELL")
            same_group = solver.get_group(group.uuid)
        """
        return object.__getattribute__(self, "_uuid")

    @property
    @blender_api
    def param(self) -> _ParamProxy:
        """Material and simulation parameter proxy.  See :class:`GroupParam`.

        Example::

            group.param.friction = 0.5
            group.param.shell_density = 1.0
        """
        return object.__getattribute__(self, "_param")

    def _get_group(self):
        from ..models.groups import get_group_by_uuid

        group = get_group_by_uuid(bpy.context.scene, self._uuid)
        if group is None:
            raise ValueError(f"Group '{self._uuid}' not found")
        return group

    # -- Overlay -------------------------------------------------------------

    @blender_api
    def set_overlay_color(self, r: float, g: float, b: float, a: float = 1.0) -> "_Group":
        """Set the viewport overlay color for this group and enable it.

        Args:
            r: Red channel in ``[0, 1]``.
            g: Green channel in ``[0, 1]``.
            b: Blue channel in ``[0, 1]``.
            a: Alpha in ``[0, 1]`` (default ``1.0``).

        Returns:
            ``self`` for chaining.

        Example::

            group.set_overlay_color(0.9, 0.2, 0.1)  # red overlay
        """
        from ..ui.dynamics.overlay import apply_object_overlays

        group = self._get_group()
        group.color = (r, g, b, a)
        group.show_overlay_color = True
        apply_object_overlays()
        return self

    # -- Object assignment ---------------------------------------------------

    @blender_api
    def add(self, *object_names: str) -> "_Group":
        """Add mesh objects to this group by name.

        Args:
            *object_names: One or more Blender object names.

        Returns:
            ``self`` for chaining.

        Example::

            group.add("Shirt", "Skirt", "Sleeve")
        """
        import json

        bpy.ops.zozo_contact_solver.add_objects_to_group(
            group_uuid=self._uuid, object_names=json.dumps(list(object_names))
        )
        return self

    @blender_api
    def remove(self, object_name: str) -> "_Group":
        """Remove an object from this group.

        Args:
            object_name: Name of the object to remove.

        Returns:
            ``self`` for chaining.

        Example::

            group.remove("Sleeve")
        """
        bpy.ops.zozo_contact_solver.remove_object_from_group(
            group_uuid=self._uuid, object_name=object_name,
        )
        return self

    # -- Pin management ------------------------------------------------------

    @blender_api
    def create_pin(self, object_name: str, vertex_group_name: str) -> _Pin:
        """Pin a vertex group so its vertices stay fixed during simulation.

        Args:
            object_name: Name of the mesh object.
            vertex_group_name: Name of the vertex group on that object.

        Returns:
            A :class:`Pin` proxy for the newly created pin.

        Raises:
            ValueError: If the object is missing, not a mesh, or the
                vertex group does not exist on it.

        Example::

            pin = group.create_pin("Cloth", "collar")
            pin.move(delta=(0, 0, 0.2), frame=60)
        """
        group = self._get_group()

        obj = bpy.data.objects.get(object_name)
        if obj is None:
            raise ValueError(f"Object '{object_name}' not found")
        if obj.type != "MESH":
            raise ValueError(f"Object '{object_name}' is not a mesh")
        if vertex_group_name not in [vg.name for vg in obj.vertex_groups]:
            raise ValueError(
                f"Vertex group '{vertex_group_name}' not found on '{object_name}'"
            )

        # Validation + locking + raw mutation all go through the
        # MutationService so scripts and MCP clients share one gate.
        from ..core.mutation import MutationError, service
        try:
            service.create_pin(self._uuid, object_name, vertex_group_name)
        except MutationError as e:
            raise ValueError(str(e))
        return _Pin(self._uuid, object_name, vertex_group_name)

    @blender_api
    def get_pins(self) -> list[_Pin]:
        """Return all pins in this group as :class:`Pin` proxies.

        Example::

            for pin in group.get_pins():
                pin.clear_keyframes()
        """
        from ..models.groups import decode_vertex_group_identifier

        group = self._get_group()
        result = []
        for pin in group.pin_vertex_groups:
            obj_name, vg_name = decode_vertex_group_identifier(pin.name)
            if obj_name and vg_name:
                result.append(_Pin(self._uuid, obj_name, vg_name))
        return result

    # -- Keyframe animation (convenience, delegates to each Pin) ---------------

    @blender_api
    def clear_keyframes(self) -> "_Group":
        """Delete all keyframes for all pins in this group.

        Convenience method that calls :meth:`Pin.clear_keyframes` on every
        pin returned by :meth:`get_pins`.

        Returns:
            ``self`` for chaining.

        Example::

            group.clear_keyframes()
        """
        for pin in self.get_pins():
            pin.clear_keyframes()
        return self

    # -- Lifecycle -----------------------------------------------------------

    @blender_api
    def delete(self) -> None:
        """Delete this group and every pin it owns.

        Example::

            group.delete()
        """
        bpy.ops.zozo_contact_solver.delete_group(group_uuid=self._uuid)

    def __repr__(self):
        try:
            group = self._get_group()
            name = group.name or "unnamed"
            return f"Group({name!r}, uuid={self._uuid!r})"
        except ValueError:
            return f"Group(deleted, uuid={self._uuid!r})"


# ---------------------------------------------------------------------------
# Dynamic parameter builder
# ---------------------------------------------------------------------------

_DYN_PARAM_KEY_MAP = {
    "gravity": "GRAVITY",
    "wind": "WIND",
    "air_density": "AIR_DENSITY",
    "air_friction": "AIR_FRICTION",
    "vertex_air_damp": "VERTEX_AIR_DAMP",
}


@blender_api
class _DynParamBuilder:
    """Fluent builder for dynamic scene parameter keyframes.

    Mirrors the frontend ``session.param.dyn()`` API but uses **frames**
    instead of seconds.  Obtained from :meth:`SceneParam.dyn`.

    Valid parameter keys: ``"gravity"``, ``"wind"``, ``"air_density"``,
    ``"air_friction"``, ``"vertex_air_damp"``.

    Frames must be strictly increasing within a chain.  Every mutating
    method returns ``self`` so operations chain.

    Example::

        solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)
    """

    def __init__(self, key: str):
        param_type = _DYN_PARAM_KEY_MAP.get(key)
        if param_type is None:
            raise ValueError(
                f"Unknown dynamic parameter '{key}'. "
                f"Valid keys: {', '.join(_DYN_PARAM_KEY_MAP)}"
            )
        self._param_type = param_type
        self._frame = 1
        state = get_addon_data(bpy.context.scene).state
        # Find or create the DynParamItem
        self._item = None
        for item in state.dyn_params:
            if item.param_type == param_type:
                self._item = item
                break
        if self._item is None:
            self._item = state.dyn_params.add()
            self._item.param_type = param_type
            kf = self._item.keyframes.add()
            kf.frame = 1
            state.dyn_params_index = len(state.dyn_params) - 1

    @blender_api
    def time(self, frame: int) -> "_DynParamBuilder":
        """Advance the frame cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_DynParamBuilder":
        """Hold the previous value at the current cursor frame (step function).

        Returns:
            ``self`` for chaining.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def change(self, value, strength=None) -> "_DynParamBuilder":
        """Set a new value at the current cursor frame.

        Args:
            value: For ``"gravity"``, an ``(x, y, z)`` tuple.
                For ``"wind"``, an ``(x, y, z)`` direction tuple.
                For scalar keys (``"air_density"``, ``"air_friction"``,
                ``"vertex_air_damp"``), a ``float``.
            strength: Wind strength (only for ``"wind"``).

        Returns:
            ``self`` for chaining.

        Example::

            solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)
        """
        self._add_keyframe(use_hold=False, value=value, strength=strength)
        return self

    @blender_api
    def clear(self) -> "_DynParamBuilder":
        """Remove this dynamic parameter entirely.

        Returns:
            ``self`` for chaining (though no further method on this
            builder will do anything meaningful after ``clear()``).

        Example::

            solver.param.dyn("wind").clear()
        """
        state = get_addon_data(bpy.context.scene).state
        removed = False
        for i, item in enumerate(state.dyn_params):
            if item.param_type == self._param_type:
                state.dyn_params.remove(i)
                state.dyn_params_index = safe_update_index(
                    state.dyn_params_index, len(state.dyn_params)
                )
                removed = True
                break
        self._item = None
        if removed:
            from ..models.groups import invalidate_overlays
            invalidate_overlays()
        return self

    def _add_keyframe(self, use_hold=False, value=None, strength=None):
        if self._item is None:
            raise ValueError("Dynamic parameter has been cleared")
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold and value is not None:
            if self._param_type == "GRAVITY":
                kf.gravity_value = tuple(value)
            elif self._param_type == "WIND":
                kf.wind_direction_value = tuple(value)
                if strength is not None:
                    kf.wind_strength_value = float(strength)
            else:
                kf.scalar_value = float(value)
        sort_keyframes_by_frame(self._item.keyframes)


# ---------------------------------------------------------------------------
# Invisible collider builders
# ---------------------------------------------------------------------------


@blender_api
class _ColliderParamProxy:
    """Attribute proxy for invisible-collider parameters.

    Accessed via :attr:`Wall.param` or :attr:`Sphere.param`.  Attribute
    access is whitelisted: reading or writing a name outside the
    whitelist raises :class:`AttributeError`.

    Whitelisted attributes:

    - ``friction``: contact friction coefficient
    - ``contact_gap``: contact gap thickness
    - ``thickness``: wall/sphere shell thickness
    - ``enable_active_duration``: ``True`` to limit collider lifetime
    - ``active_duration``: number of frames the collider is active when
      ``enable_active_duration`` is set

    Example::

        wall = solver.add_wall((0, 0, 0), (0, 0, 1))
        wall.param.friction = 0.5
        wall.param.contact_gap = 0.002
        wall.param.thickness = 0.01
        wall.param.enable_active_duration = True
        wall.param.active_duration = 60  # active for frames 1-60

        sphere = solver.add_sphere((0, 0, 1), 0.5)
        sphere.param.friction = 0.3
    """

    def __init__(self, item):
        object.__setattr__(self, "_item", item)

    def __setattr__(self, key, value):
        item = object.__getattribute__(self, "_item")
        if key in ("contact_gap", "friction", "thickness", "enable_active_duration", "active_duration"):
            setattr(item, key, value)
        else:
            raise AttributeError(f"Unknown collider param '{key}'")

    def __getattr__(self, key):
        item = object.__getattribute__(self, "_item")
        if key in ("contact_gap", "friction", "thickness", "enable_active_duration", "active_duration"):
            return getattr(item, key)
        raise AttributeError(f"Unknown collider param '{key}'")


@blender_api
class _InvisibleWallBuilder:
    """Chainable builder for invisible wall colliders.

    Returned by :meth:`Solver.add_wall`.  Keyframe frames must be
    strictly increasing.  Every mutating method returns ``self``.

    Example::

        solver.add_wall((0, 0, 0), (0, 0, 1)).param.friction = 0.5
        (solver.add_wall((0, 0, 0), (0, 1, 0))
               .time(60).hold().time(61).move_to((0, 1, 0)))
    """

    def __init__(self, position, normal):
        state = get_addon_data(bpy.context.scene).state
        self._item = state.invisible_colliders.add()
        self._item.collider_type = "WALL"
        # Auto-name
        existing = [c.name for c in state.invisible_colliders if c.collider_type == "WALL"]
        self._item.name = generate_unique_name("Wall", existing)
        self._item.position = tuple(position)
        self._item.normal = tuple(normal)
        kf = self._item.keyframes.add()
        kf.frame = 1
        state.invisible_colliders_index = len(state.invisible_colliders) - 1
        self._frame = 1

    @classmethod
    def attach_to_last(cls) -> "_InvisibleWallBuilder":
        """Return a builder bound to the most-recently-added collider
        without re-adding.  Used by ``_Solver.add_wall`` after the service
        has already performed the add."""
        inst = cls.__new__(cls)
        state = get_addon_data(bpy.context.scene).state
        inst._item = state.invisible_colliders[-1]
        inst._frame = 1
        return inst

    @property
    @blender_api
    def param(self) -> _ColliderParamProxy:
        """Collider parameter proxy.  See :class:`ColliderParam`.

        Example::

            wall = solver.add_wall((0, 0, 0), (0, 0, 1))
            wall.param.friction = 0.5
        """
        return _ColliderParamProxy(self._item)

    @blender_api
    def time(self, frame: int) -> "_InvisibleWallBuilder":
        """Advance the keyframe cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_to((0, 0, 0.5)))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_InvisibleWallBuilder":
        """Hold the previous position at the current cursor frame.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).hold().time(90).move_to((0, 0, 0.5)))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def move_to(self, position) -> "_InvisibleWallBuilder":
        """Keyframe a new absolute position at the current cursor frame.

        Args:
            position: ``(x, y, z)`` world-space position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_to((0, 0, 1.0)))
        """
        self._add_keyframe(use_hold=False, position=position)
        return self

    @blender_api
    def move_by(self, delta) -> "_InvisibleWallBuilder":
        """Keyframe a position offset from the previous keyframe.

        Args:
            delta: ``(dx, dy, dz)`` offset added to the previous
                keyframed position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_by((0, 0, 0.25)))
        """
        # Get previous position
        if len(self._item.keyframes) > 0:
            last_kf = self._item.keyframes[len(self._item.keyframes) - 1]
            prev = list(last_kf.position) if last_kf.frame > 1 else list(self._item.position)
        else:
            prev = list(self._item.position)
        new_pos = [prev[i] + delta[i] for i in range(3)]
        self._add_keyframe(use_hold=False, position=new_pos)
        return self

    @blender_api
    def delete(self) -> None:
        """Remove this wall collider from the scene.

        Example::

            wall = solver.add_wall((0, 0, 0), (0, 0, 1))
            wall.delete()
        """
        state = get_addon_data(bpy.context.scene).state
        for i, item in enumerate(state.invisible_colliders):
            if item == self._item:
                state.invisible_colliders.remove(i)
                state.invisible_colliders_index = safe_update_index(
                    state.invisible_colliders_index,
                    len(state.invisible_colliders),
                )
                from ..models.groups import invalidate_overlays
                invalidate_overlays()
                break

    def _add_keyframe(self, use_hold=False, position=None):
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold and position is not None:
            kf.position = tuple(position)
        sort_keyframes_by_frame(self._item.keyframes)


@blender_api
class _InvisibleSphereBuilder:
    """Chainable builder for invisible sphere colliders.

    Returned by :meth:`Solver.add_sphere`.  Keyframe frames must be
    strictly increasing.  Every mutating method returns ``self``.

    Example::

        solver.add_sphere((0, 0, 0), 0.98).invert().hemisphere()
        (solver.add_sphere((0, 0, 0), 1.0)
               .time(60).hold().time(61).radius(0.5))
    """

    def __init__(self, position, radius):
        state = get_addon_data(bpy.context.scene).state
        self._item = state.invisible_colliders.add()
        self._item.collider_type = "SPHERE"
        existing = [c.name for c in state.invisible_colliders if c.collider_type == "SPHERE"]
        self._item.name = generate_unique_name("Sphere", existing)
        self._item.position = tuple(position)
        self._item.radius = float(radius)
        kf = self._item.keyframes.add()
        kf.frame = 1
        state.invisible_colliders_index = len(state.invisible_colliders) - 1
        self._frame = 1

    @classmethod
    def attach_to_last(cls) -> "_InvisibleSphereBuilder":
        """Bind to the most-recently-added sphere collider without
        adding a new one (the service already did)."""
        inst = cls.__new__(cls)
        state = get_addon_data(bpy.context.scene).state
        inst._item = state.invisible_colliders[-1]
        inst._frame = 1
        return inst

    @property
    @blender_api
    def param(self) -> _ColliderParamProxy:
        """Collider parameter proxy.  See :class:`ColliderParam`.

        Example::

            sphere = solver.add_sphere((0, 0, 0), 1.0)
            sphere.param.friction = 0.3
        """
        return _ColliderParamProxy(self._item)

    @blender_api
    def invert(self) -> "_InvisibleSphereBuilder":
        """Flip the sphere inside-out so contact is on the inside surface.

        Returns:
            ``self`` for chaining.

        Example::

            solver.add_sphere((0, 0, 0), 1.0).invert()
        """
        self._item.invert = True
        return self

    @blender_api
    def hemisphere(self) -> "_InvisibleSphereBuilder":
        """Treat this collider as a hemisphere rather than a full sphere.

        Returns:
            ``self`` for chaining.

        Example::

            solver.add_sphere((0, 0, 0), 1.0).hemisphere()
        """
        self._item.hemisphere = True
        return self

    @blender_api
    def time(self, frame: int) -> "_InvisibleSphereBuilder":
        """Advance the keyframe cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).move_to((0, 0, 1.0)))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_InvisibleSphereBuilder":
        """Hold the previous position and radius at the current cursor frame.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).hold().time(90).radius(0.5))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def move_to(self, position) -> "_InvisibleSphereBuilder":
        """Keyframe a new absolute position at the current cursor frame.

        Args:
            position: ``(x, y, z)`` world-space position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).move_to((0, 0, 2.0)))
        """
        self._add_keyframe(use_hold=False, position=position)
        return self

    @blender_api
    def radius(self, r) -> "_InvisibleSphereBuilder":
        """Keyframe a new radius at the current cursor frame.

        Args:
            r: New radius.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).radius(0.25))  # shrink over 60 frames
        """
        self._add_keyframe(use_hold=False, radius=r)
        return self

    @blender_api
    def transform_to(self, position, radius) -> "_InvisibleSphereBuilder":
        """Keyframe both position and radius together.

        Args:
            position: ``(x, y, z)`` world-space position.
            radius: New radius.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).transform_to((0, 0, 1.0), 0.5))
        """
        self._add_keyframe(use_hold=False, position=position, radius=radius)
        return self

    @blender_api
    def delete(self) -> None:
        """Remove this sphere collider from the scene.

        Example::

            sphere = solver.add_sphere((0, 0, 0), 1.0)
            sphere.delete()
        """
        state = get_addon_data(bpy.context.scene).state
        for i, item in enumerate(state.invisible_colliders):
            if item == self._item:
                state.invisible_colliders.remove(i)
                state.invisible_colliders_index = safe_update_index(
                    state.invisible_colliders_index,
                    len(state.invisible_colliders),
                )
                from ..models.groups import invalidate_overlays
                invalidate_overlays()
                break

    def _add_keyframe(self, use_hold=False, position=None, radius=None):
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold:
            if position is not None:
                kf.position = tuple(position)
            else:
                # Keep previous position
                kf.position = tuple(self._item.position)
            if radius is not None:
                kf.radius = float(radius)
            else:
                kf.radius = self._item.radius
        sort_keyframes_by_frame(self._item.keyframes)


# ---------------------------------------------------------------------------
# Scene proxy
# ---------------------------------------------------------------------------

@blender_api
class _SceneProxy:
    """Attribute proxy for scene and SSH/connection parameters.

    Accessed as :attr:`Solver.param`.  Supports both get and set via
    attribute access.  Writes go through the ``zozo_contact_solver.set``
    operator (with auto type coercion), reads fall through to the
    scene's addon state or SSH state.

    ``gravity`` is an alias for ``gravity_3d``.

    Example::

        solver.param.step_size = 0.004
        print(solver.param.gravity)

    Dynamic (keyframed) parameters are accessed via :meth:`dyn`::

        solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
    """

    @blender_api
    def dyn(self, key: str) -> _DynParamBuilder:
        """Select a parameter for dynamic keyframing.

        Args:
            key: One of ``"gravity"``, ``"wind"``, ``"air_density"``,
                ``"air_friction"``, ``"vertex_air_damp"``.

        Returns:
            A chainable :class:`DynParam` builder.

        Raises:
            ValueError: If *key* is not one of the valid dynamic keys.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        return _DynParamBuilder(key)

    def __setattr__(self, key, value):
        bpy.ops.zozo_contact_solver.set(key=str(key), value=str(value))

    _ALIASES = {"gravity": "gravity_3d"}

    def __getattr__(self, key):
        key = self._ALIASES.get(key, key)
        scene = bpy.context.scene
        addon_data = get_addon_data(scene)
        if hasattr(addon_data.state, key):
            return getattr(addon_data.state, key)
        if hasattr(addon_data.ssh_state, key):
            return getattr(addon_data.ssh_state, key)
        raise AttributeError(f"No scene property '{key}'")


# ---------------------------------------------------------------------------
# Solver (top-level entry point)
# ---------------------------------------------------------------------------

@blender_api
class _Solver:
    """Top-level entry point for the ZOZO Contact Solver.

    Available as ``solver`` when imported via::

        from zozo_contact_solver import solver

    Scene parameters are accessed via :attr:`param` (a
    :class:`SceneParam` proxy).  Groups, pins, and invisible colliders
    are created via the methods below.

    Unrecognized attribute access falls through to
    ``bpy.ops.zozo_contact_solver.<name>()``, so every operator
    registered under that namespace (including every MCP handler)
    can be called as a method on ``solver``.

    Example::

        solver.param.gravity = (0, 0, -9.8)
        group = solver.create_group("Sphere", type="SOLID")
        group.add("Sphere")
        group.param.solid_density = 1000
    """

    #: Scene parameter proxy; see :class:`SceneParam`.
    param = _SceneProxy()

    # -- Group lifecycle -----------------------------------------------------

    @blender_api
    def create_group(self, name: str = "", type: str = "SOLID") -> _Group:
        """Create a new dynamics group.

        Args:
            name: Display name for the group.  Empty string leaves the
                auto-generated name in place.
            type: One of ``"SOLID"``, ``"SHELL"``, ``"ROD"``, ``"STATIC"``.

        Returns:
            A :class:`Group` proxy for the newly created group.

        Example::

            group = solver.create_group("Shirt", type="SHELL")
            group.add("Shirt")
        """
        bpy.ops.zozo_contact_solver.create_group()
        uuid = get_addon_data(bpy.context.scene).state.current_group_uuid
        if name:
            bpy.ops.zozo_contact_solver.set(
                group_uuid=uuid, key="name", value=name
            )
        if type != "SOLID":
            bpy.ops.zozo_contact_solver.set_group_type(
                group_uuid=uuid, type=type
            )
        return _Group(uuid)

    @blender_api
    def get_group(self, group_uuid: str) -> _Group:
        """Look up a group by UUID.

        Args:
            group_uuid: UUID string of the group.

        Returns:
            A :class:`Group` proxy.

        Raises:
            KeyError: If the group does not exist.

        Example::

            uuid = solver.get_groups()[0].uuid
            group = solver.get_group(uuid)
        """
        from ..models.groups import get_group_by_uuid

        group = get_group_by_uuid(bpy.context.scene, group_uuid)
        if group is None:
            raise KeyError(f"Group '{group_uuid}' not found")
        return _Group(group_uuid)

    @blender_api
    def get_groups(self) -> list[_Group]:
        """Return :class:`Group` proxies for every active group.

        Example::

            for group in solver.get_groups():
                print(group.uuid)
        """
        from ..models.groups import iterate_active_object_groups

        result = []
        for group in iterate_active_object_groups(bpy.context.scene):
            result.append(_Group(group.uuid))
        return result

    @blender_api
    def delete_all_groups(self) -> "_Solver":
        """Delete every active group and the pins they own.

        Returns:
            ``self`` for chaining.

        Example::

            solver.delete_all_groups()
        """
        bpy.ops.zozo_contact_solver.delete_all_groups()
        return self

    @blender_api
    def clear(self) -> "_Solver":
        """Reset the entire solver state to defaults.

        Deletes every active group, resets scene parameters to their
        property defaults, clears merge pairs, invisible colliders,
        dynamic parameters, previously fetched frames, saved pin
        keyframes, and any residual ``MESH_CACHE`` modifiers on mesh
        objects.
        Call this at the top of any script that needs a clean slate.

        Returns:
            ``self`` for chaining.

        Example::

            solver.clear()
            solver.param.gravity = (0, 0, -9.8)
        """
        from ..models.groups import N_MAX_GROUPS

        root = get_addon_data(bpy.context.scene)
        state = root.state

        # Delete all groups and reset their properties
        for i in range(N_MAX_GROUPS):
            group = getattr(root, f"object_group_{i}", None)
            if group and group.active:
                group.reset_to_defaults()

        # Reset scene parameters to defaults
        bl_props = state.bl_rna.properties
        skip = {"bl_rna", "rna_type", "name", "fetched_frame", "saved_pin_keyframes"}
        for prop in bl_props:
            pid = prop.identifier
            if pid in skip:
                continue
            if hasattr(prop, "default") and hasattr(state, pid):
                try:
                    setattr(state, pid, prop.default)
                except Exception:
                    continue
            elif hasattr(prop, "default_array") and hasattr(state, pid):
                try:
                    setattr(state, pid, tuple(prop.default_array))
                except Exception:
                    continue

        # Clear fetched frames and saved pin keyframes
        state.clear_fetched_frames()
        state.saved_pin_keyframes.clear()

        # Collection properties don't respond to setattr(prop.default), so
        # clear them explicitly, otherwise solver.clear() silently leaves
        # merge pairs and scene colliders behind.
        state.merge_pairs.clear()
        state.merge_pairs_index = 0
        if hasattr(state, "invisible_colliders"):
            state.invisible_colliders.clear()
            state.invisible_colliders_index = 0
        if hasattr(state, "dyn_params"):
            state.dyn_params.clear()
            state.dyn_params_index = 0

        # Remove MESH_CACHE modifiers, PC2 files, and residual animation data
        from ..core.pc2 import cleanup_mesh_cache

        for obj in bpy.data.objects:
            if obj.type == "MESH":
                cleanup_mesh_cache(obj)
                if obj.data.animation_data:
                    obj.data.animation_data_clear()

        # Collection .clear()/.remove() calls above (assigned_objects and
        # pin_vertex_groups via reset_to_defaults, plus merge_pairs,
        # invisible_colliders, dyn_params here) do not trigger update
        # callbacks, so invalidate the overlay cache once now.
        from ..models.groups import invalidate_overlays
        invalidate_overlays()

        return self

    # -- Snap ----------------------------------------------------------------

    @blender_api
    def snap(self, object_a: str, object_b: str) -> "_Solver":
        """Translate *object_a* so its nearest vertex lands on *object_b*.

        Args:
            object_a: Name of the mesh that moves.
            object_b: Name of the mesh that stays in place.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If either object is missing, not a mesh, or
                validation in the underlying mutation service fails.

        Example::

            solver.snap("Shirt", "Mannequin")
        """
        from ..core.mutation import MutationError, service
        try:
            service.snap_to_vertices(object_a, object_b)
        except MutationError as e:
            raise ValueError(str(e))
        return self

    # -- Merge pairs ---------------------------------------------------------

    @blender_api
    def add_merge_pair(self, object_a: str, object_b: str) -> "_Solver":
        """Mark two objects to be merged at their shared contact.

        Args:
            object_a: Name of the first mesh.
            object_b: Name of the second mesh.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If either object is missing, not a mesh, or
                the pair is invalid.

        Example::

            solver.add_merge_pair("SleeveLeft", "BodyLeft")
        """
        from ..core.mutation import MutationError, service
        try:
            service.add_merge_pair(object_a, object_b)
        except MutationError as e:
            raise ValueError(str(e))
        return self

    @blender_api
    def remove_merge_pair(self, object_a: str, object_b: str) -> "_Solver":
        """Remove a previously added merge pair.

        The ordering of *object_a* and *object_b* does not matter; the
        pair is matched by UUID in either direction.

        Args:
            object_a: Name of the first mesh.
            object_b: Name of the second mesh.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If validation fails for the given pair.

        Example::

            solver.remove_merge_pair("SleeveLeft", "BodyLeft")
        """
        from ..core.mutation import MutationError, service
        try:
            service.remove_merge_pair(object_a, object_b)
        except MutationError as e:
            raise ValueError(str(e))
        return self

    @blender_api
    def get_merge_pairs(self) -> list[tuple[str, str]]:
        """Return every merge pair as a list of ``(object_a, object_b)`` tuples.

        Example::

            for a, b in solver.get_merge_pairs():
                print(f"{a} <-> {b}")
        """
        return _raw_get_merge_pairs()

    @blender_api
    def clear_merge_pairs(self) -> "_Solver":
        """Remove every merge pair.

        Returns:
            ``self`` for chaining.

        Example::

            solver.clear_merge_pairs()
        """
        from ..core.mutation import service
        service.clear_merge_pairs()
        return self

    # -- Invisible colliders -------------------------------------------------

    @blender_api
    def add_wall(self, position, normal) -> _InvisibleWallBuilder:
        """Add an invisible infinite-plane wall collider.

        Args:
            position: ``(x, y, z)`` world-space point on the plane.
            normal: ``(nx, ny, nz)`` outward-facing plane normal.
                Need not be unit-length.

        Returns:
            A chainable :class:`Wall` builder bound to the newly added
            collider.

        Raises:
            ValueError: If the position or normal fails vec3 validation.

        Example::

            solver.add_wall(position=(0, 0, 0), normal=(0, 0, 1))
        """
        from ..core.mutation import MutationError, service
        try:
            service.add_invisible_wall(position, normal)
        except MutationError as e:
            raise ValueError(str(e))
        return _InvisibleWallBuilder.attach_to_last()

    @blender_api
    def add_sphere(self, position, radius) -> _InvisibleSphereBuilder:
        """Add an invisible sphere collider.

        Args:
            position: ``(x, y, z)`` world-space center.
            radius: Sphere radius.

        Returns:
            A chainable :class:`Sphere` builder bound to the newly added
            collider.

        Raises:
            ValueError: If the position or radius fails validation.

        Example::

            solver.add_sphere(position=(0, 0, 1.0), radius=0.25)
        """
        from ..core.mutation import MutationError, service
        try:
            service.add_invisible_sphere(position, radius)
        except MutationError as e:
            raise ValueError(str(e))
        return _InvisibleSphereBuilder.attach_to_last()

    @blender_api
    def get_invisible_colliders(self) -> list:
        """Return every invisible collider as a list of ``(type, name)`` tuples.

        *type* is one of ``"WALL"`` or ``"SPHERE"``.

        Example::

            for kind, name in solver.get_invisible_colliders():
                print(kind, name)
        """
        state = get_addon_data(bpy.context.scene).state
        return [(c.collider_type, c.name) for c in state.invisible_colliders]

    @blender_api
    def clear_invisible_colliders(self) -> "_Solver":
        """Remove every invisible collider.

        Returns:
            ``self`` for chaining.

        Example::

            solver.clear_invisible_colliders()
        """
        from ..core.mutation import service
        service.clear_invisible_colliders()
        return self

    # -- Fallback ------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(bpy.ops.zozo_contact_solver, name)


# ---------------------------------------------------------------------------
# Raw state-mutation functions.
#
# These are the only place that writes merge_pairs / invisible_colliders /
# snap state directly.  MutationService public methods validate + lock
# then call these.  _Solver public methods call MutationService (so
# scripts get the same validation MCP callers get).
#
# No validation here; callers must have validated already.
# ---------------------------------------------------------------------------


def _raw_add_merge_pair(object_a: str, object_b: str) -> None:
    from ..core.uuid_registry import get_or_create_object_uuid
    state = get_addon_data(bpy.context.scene).state
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    uuid_a = get_or_create_object_uuid(obj_a) if obj_a else ""
    uuid_b = get_or_create_object_uuid(obj_b) if obj_b else ""
    if not uuid_a or not uuid_b:
        return
    for pair in state.merge_pairs:
        if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or (
            pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a
        ):
            return
    item = state.merge_pairs.add()
    item.object_a = object_a
    item.object_b = object_b
    item.object_a_uuid = uuid_a
    item.object_b_uuid = uuid_b
    state.merge_pairs_index = len(state.merge_pairs) - 1
    from ..ui.dynamics.overlay import apply_object_overlays
    apply_object_overlays()


def _raw_remove_merge_pair(object_a: str, object_b: str) -> None:
    from ..core.uuid_registry import get_or_create_object_uuid
    state = get_addon_data(bpy.context.scene).state
    # Resolve names to UUIDs so the comparison is rename-safe.
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    uuid_a = get_or_create_object_uuid(obj_a) if obj_a else ""
    uuid_b = get_or_create_object_uuid(obj_b) if obj_b else ""
    if not uuid_a or not uuid_b:
        return
    for i in range(len(state.merge_pairs)):
        pair = state.merge_pairs[i]
        if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or (
            pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a
        ):
            state.merge_pairs.remove(i)
            state.merge_pairs_index = safe_update_index(
                state.merge_pairs_index, len(state.merge_pairs)
            )
            from ..ui.dynamics.overlay import apply_object_overlays
            apply_object_overlays()
            return


def _raw_clear_merge_pairs() -> None:
    state = get_addon_data(bpy.context.scene).state
    state.merge_pairs.clear()
    state.merge_pairs_index = -1
    from ..ui.dynamics.overlay import apply_object_overlays
    apply_object_overlays()


def _raw_get_merge_pairs() -> list[tuple[str, str]]:
    """Return existing merge pairs as (name_a, name_b) tuples."""
    state = get_addon_data(bpy.context.scene).state
    return [(p.object_a, p.object_b) for p in state.merge_pairs]


def _raw_snap(object_a: str, object_b: str) -> None:
    from ..core.uuid_registry import get_or_create_object_uuid
    state = get_addon_data(bpy.context.scene).state
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    if not obj_a or not obj_b:
        return
    uid_a = get_or_create_object_uuid(obj_a)
    uid_b = get_or_create_object_uuid(obj_b)
    if not uid_a or not uid_b:
        return
    state.snap_object_a = uid_a
    state.snap_object_b = uid_b
    bpy.ops.object.snap_to_vertices()


def _raw_add_wall(position, normal) -> None:
    _InvisibleWallBuilder(position, normal)


def _raw_add_sphere(position, radius, invert: bool = False, hemisphere: bool = False) -> None:
    builder = _InvisibleSphereBuilder(position, radius)
    if invert:
        builder.invert()
    if hemisphere:
        builder.hemisphere()


def _raw_clear_invisible_colliders() -> None:
    state = get_addon_data(bpy.context.scene).state
    state.invisible_colliders.clear()
    state.invisible_colliders_index = -1
    from ..models.groups import invalidate_overlays
    invalidate_overlays()


def _raw_create_pin(group_uuid: str, object_name: str, vertex_group_name: str) -> None:
    """Add a pin without validation (caller has validated)."""
    from ..models.groups import get_group_by_uuid
    group = get_group_by_uuid(bpy.context.scene, group_uuid)
    if group is None:
        raise ValueError(f"group '{group_uuid}' not found")
    obj = bpy.data.objects.get(object_name)
    from ..core.uuid_registry import compute_vg_hash, get_or_create_object_uuid
    from ..models.groups import decode_vertex_group_identifier
    identifier = encode_vertex_group_identifier(object_name, vertex_group_name)
    uuid_val = get_or_create_object_uuid(obj)
    if not uuid_val:
        raise ValueError(f"object '{object_name}' is not writable (library-linked)")
    vg_hash_val = str(compute_vg_hash(obj, vertex_group_name))
    # Match-by-(UUID, vg_name); rename-safe duplicate detection.
    for p in group.pin_vertex_groups:
        if p.object_uuid != uuid_val:
            continue
        _, p_vg = decode_vertex_group_identifier(p.name)
        if p_vg == vertex_group_name:
            return
    item = group.pin_vertex_groups.add()
    try:
        item.name = identifier
        item.object_uuid = uuid_val
        item.vg_hash = vg_hash_val
        group.pin_vertex_groups_index = len(group.pin_vertex_groups) - 1
    except Exception:
        group.pin_vertex_groups.remove(len(group.pin_vertex_groups) - 1)
        raise
    from ..models.groups import invalidate_overlays
    invalidate_overlays()


solver = _Solver()


# ---------------------------------------------------------------------------
# Public aliases used by the documentation generator.
#
# The underscore-prefixed class names above are the runtime identifiers;
# these aliases are what the docs generator references and what scripts
# should use for type hints (``group: Group = solver.create_group(...)``).
# Runtime behavior is unchanged; aliases are simple name bindings.
# ---------------------------------------------------------------------------

Solver = _Solver
Group = _Group
Pin = _Pin
SceneParam = _SceneProxy
GroupParam = _ParamProxy
ColliderParam = _ColliderParamProxy
DynParam = _DynParamBuilder
Wall = _InvisibleWallBuilder
Sphere = _InvisibleSphereBuilder
