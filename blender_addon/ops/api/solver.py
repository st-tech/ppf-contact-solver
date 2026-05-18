"""``_Solver``: top-level entry point for the ZOZO Contact Solver
scripting API.

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore

from ...models.groups import get_addon_data
from .._api_markers import blender_api
from .collider import _InvisibleSphereBuilder, _InvisibleWallBuilder
from .curve import _CurveBuilder
from .dynamics import _SceneProxy
from .group import _Group


# ---------------------------------------------------------------------------
# Solver (top-level entry point)
# ---------------------------------------------------------------------------

@blender_api
class _Solver:
    """Top-level entry point for the ZOZO Contact Solver.

    Available as ``solver`` when imported via::

        from bl_ext.user_default.ppf_contact_solver.ops.api import solver

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
        from ...models.groups import get_group_by_uuid

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
        from ...models.groups import iterate_active_object_groups

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
        from ...models.groups import N_MAX_GROUPS

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

        # Blender treats CollectionProperty.clear() as a write to the
        # owning ID and blocks it in restricted contexts (load_post,
        # render handlers, scripts launched from an embedded Text
        # data-block during file load). Wrap each .clear() so a
        # restricted context skips silently instead of crashing the
        # script halfway through.
        def _safe_clear(call):
            try:
                call()
            except AttributeError as e:
                if "Writing to ID classes" not in str(e):
                    raise

        _safe_clear(state.clear_fetched_frames)
        _safe_clear(state.saved_pin_keyframes.clear)

        # Collection properties don't respond to setattr(prop.default), so
        # clear them explicitly, otherwise solver.clear() silently leaves
        # merge pairs and scene colliders behind.
        _safe_clear(state.merge_pairs.clear)
        state.merge_pairs_index = 0
        if hasattr(state, "invisible_colliders"):
            _safe_clear(state.invisible_colliders.clear)
            state.invisible_colliders_index = 0
        if hasattr(state, "dyn_params"):
            _safe_clear(state.dyn_params.clear)
            state.dyn_params_index = 0

        # Remove MESH_CACHE modifiers, PC2 files, and residual animation data
        from ...core.pc2 import cleanup_mesh_cache

        for obj in bpy.data.objects:
            if obj.type == "MESH":
                cleanup_mesh_cache(obj)
                if obj.data.animation_data:
                    obj.data.animation_data_clear()

        # Collection .clear()/.remove() calls above (assigned_objects and
        # pin_vertex_groups via reset_to_defaults, plus merge_pairs,
        # invisible_colliders, dyn_params here) do not trigger update
        # callbacks, so invalidate the overlay cache once now.
        from ...models.groups import invalidate_overlays
        invalidate_overlays()

        return self

    # -- Curve construction --------------------------------------------------

    @blender_api
    def create_curve(self, name: str, *, bevel_depth: float = 0.0,
                     bevel_resolution: int = 2, resolution_u: int = 4,
                     dimensions: str = "3D",
                     clear_existing: bool = True) -> _CurveBuilder:
        """Start building a multi-spline Bezier curve object.

        Returns a :class:`Curve` builder.  Use :meth:`Curve.add_spline`
        for each spline, optionally :meth:`Curve.set_material` to color
        them, then :meth:`Curve.finalize` to link the resulting object
        into the scene.

        Args:
            name: Object name.  When ``clear_existing`` is true (the
                default) any existing object with this name is removed
                first so re-running the script starts from a clean
                slate.
            bevel_depth: Tube radius for visualization (Blender's
                ``Curve.bevel_depth``).  ``0`` leaves the curve as a
                wireframe.
            bevel_resolution: Tube cross-section subdivisions
                (``Curve.bevel_resolution``).
            resolution_u: Spline interpolation resolution
                (``Curve.resolution_u``).
            dimensions: ``"3D"`` (default) or ``"2D"``.
            clear_existing: Set ``False`` to skip the same-name cleanup.

        Returns:
            A :class:`Curve` builder.

        Example::

            curve = solver.create_curve("Strands", bevel_depth=3e-3)
            for points, closed in strands:
                curve.add_spline(points, closed=closed)
            obj = curve.finalize()
        """
        return _CurveBuilder(
            name,
            bevel_depth=bevel_depth,
            bevel_resolution=bevel_resolution,
            resolution_u=resolution_u,
            dimensions=dimensions,
            clear_existing=clear_existing,
        )

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
        from ...core import mutation
        try:
            mutation.snap_to_vertices(object_a, object_b)
        except mutation.MutationError as e:
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
        from ...core import mutation
        try:
            mutation.add_merge_pair(object_a, object_b)
        except mutation.MutationError as e:
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
        from ...core import mutation
        try:
            mutation.remove_merge_pair(object_a, object_b)
        except mutation.MutationError as e:
            raise ValueError(str(e))
        return self

    @blender_api
    def get_merge_pairs(self) -> list[tuple[str, str]]:
        """Return every merge pair as a list of ``(object_a, object_b)`` tuples.

        Example::

            for a, b in solver.get_merge_pairs():
                print(f"{a} <-> {b}")
        """
        from ...core.mutation import _raw_get_merge_pairs
        return _raw_get_merge_pairs()

    @blender_api
    def clear_merge_pairs(self) -> "_Solver":
        """Remove every merge pair.

        Returns:
            ``self`` for chaining.

        Example::

            solver.clear_merge_pairs()
        """
        from ...core import mutation
        mutation.clear_merge_pairs()
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
        from ...core import mutation
        try:
            mutation.add_invisible_wall(position, normal)
        except mutation.MutationError as e:
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
        from ...core import mutation
        try:
            mutation.add_invisible_sphere(position, radius)
        except mutation.MutationError as e:
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
        from ...core import mutation
        mutation.clear_invisible_colliders()
        return self

    # -- Fallback ------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(bpy.ops.zozo_contact_solver, name)

solver = _Solver()
