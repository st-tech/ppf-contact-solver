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


def _handler_result(handler, **kwargs) -> dict:
    """Run an MCP handler's body and return the result dict it produces.

    The handlers own the validation, the selection save and restore, and the
    reporting for the mesh, export and deformation-cache operations, so the
    methods below call them instead of restating any of it.

    What is called is ``_original_func``, the body ``@mcp_handler`` wrapped.
    The wrapper answers an MCP client with ``{"status": "error", ...}``
    instead of raising, and a Python caller reading that as a return value
    would see a successful call that did nothing. The body raises ``MCPError``
    instead, and every condition it raises for is one the caller states in the
    call, so it arrives here as the package's ``ValueError``.

    A ``RuntimeError`` from ``bpy.ops`` (an operator poll refusing the
    context, or an operator reporting an error the handler does not restate
    as an ``MCPError``) is not one of those stated conditions, so it passes
    through as itself.  That is loud, never silent.
    """
    from ...mcp.decorators import MCPError

    try:
        return handler._original_func(**kwargs)
    except MCPError as e:
        raise ValueError(str(e))


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
        group.param.solid_density = 100
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
            type: One of ``"SOLID"``, ``"SHELL"``, ``"ROD"``, ``"STATIC"``,
                ``"PDRD"``, ``"SAND"``.

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
        from ...models.groups import iterate_active_object_groups
        from ...core.uuid_registry import resolve_assigned
        from ...ui.dynamics.utils import reset_object_display

        root = get_addon_data(bpy.context.scene)
        state = root.state

        # Delete all groups and reset their properties. Each member's
        # display state is taken back while it is still a member; once
        # the group is reset the object is the user's again and the
        # add-on may not write to it.
        for group in iterate_active_object_groups(bpy.context.scene):
            for assigned in group.assigned_objects:
                member = resolve_assigned(assigned)
                if member is not None:
                    reset_object_display(member)
            group.reset_to_defaults()

        # Reset scene parameters to defaults
        bl_props = state.bl_rna.properties
        skip = {"bl_rna", "rna_type", "name", "fetched_frame"}
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

    # -- Mesh cleaning -------------------------------------------------------
    #
    # Each tool names its targets rather than reading the viewport selection,
    # and the selection is saved and restored around the call.
    #
    # Three of the repairs change the vertex count. The panel confirms that
    # through a dialog and the MCP tools through an "acknowledge" flag; here
    # the call itself is the confirmation, the same way clear() needs none, so
    # there is no such parameter. "clear_stale_caches" is a parameter, because
    # it decides what becomes of the caches the count change invalidates.

    @blender_api
    def scan_meshes(self, *object_names: str, merge_threshold: float = 1e-4,
                    area_eps: float = 0.0) -> dict[str, dict]:
        """Report the geometry the solver rejects, modifying nothing.

        Args:
            *object_names: One or more mesh object names.
            merge_threshold: Vertices closer together than this, in local
                units, count as near-coincident.  Matches Blender's Merge
                by Distance default.
            area_eps: Faces at or below this area, in local units squared,
                count as degenerate.  ``0.0`` reports only exactly
                zero-area faces.

        Returns:
            ``{object_name: report}``.  Each report carries ``object`` (that
            same name), ``n_errors``, ``n_notes``, ``total`` (every defect
            count summed), the ``n_verts`` / ``n_polys`` /
            ``merge_threshold`` / ``area_eps`` it was taken at,
            ``dependents`` (what a vertex-count change on that object would
            affect), and ``defects``.  A note is not a defect: an open quad
            panel is an ordinary cloth mesh.

            ``defects`` is keyed by the eight names below, and ``count`` is
            the only key every entry has.  The rest differ per defect, and
            three entries carry nothing but ``count`` when they find
            nothing, so read anything else behind a non-zero ``count`` or
            through ``dict.get``:

            * ``near_duplicates``: ``min_dist``, ``min_dist_world``,
              ``preview`` (up to eight ``(i, j)`` vertex-index pairs) and
              ``verts`` (every vertex index involved), all four present only
              when ``count`` is non-zero.
            * ``isolated_verts``, ``hanging_verts``: ``preview`` (up to
              eight vertex indices) and ``verts``, always present and empty
              at ``count`` zero.
            * ``degenerate_faces``: ``preview`` (up to eight face indices),
              present only when ``count`` is non-zero.  No ``verts``.
            * ``duplicate_faces``: nothing beyond ``count``.
            * ``surface``: ``boundary``, ``non_manifold`` and
              ``bad_winding``, always present; ``count`` is their sum.  No
              ``preview``, no ``verts``.
            * ``resplittable``: ``max_fold_deg`` and ``past_flip``, present
              only when ``count`` is non-zero.  No ``verts``.
            * ``linked_duplicate``: ``siblings``, the names of the objects
              sharing this mesh datablock, always present.

            The per-vertex ``verts`` lists come back whole here.  The MCP
            tool strips them from its payload, where the counts and the
            previews are what a client acts on.

        Raises:
            ValueError: If a name is missing, is not a mesh, is outside the
                active view layer, or cannot be selected in it (hidden in
                the viewport, hidden by its collection, or carrying Disable
                Selection), since the operators underneath read the
                selection and would skip it.  Also if a named object
                answered with no report.

        Example::

            report = solver.scan_meshes("Shirt")["Shirt"]
            if report["defects"]["near_duplicates"]["count"]:
                solver.merge_by_distance("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning
        from ...mesh_ops.cleaning_ops import clear_scan_report, get_scan_report

        names = list(object_names)
        # Drop the previous reports first, so every report read back below
        # belongs to this call. get_scan_report is keyed by object name and
        # validates only against the vertex and polygon counts, so an object
        # this scan did not reach would otherwise answer with an older report
        # taken at different thresholds.
        for name in names:
            clear_scan_report(name)
        _handler_result(
            mesh_cleaning.scan_meshes,
            object_names=names,
            merge_threshold=merge_threshold,
            area_eps=area_eps,
        )

        reports = {}
        for name in names:
            report = get_scan_report(name)
            if report is None:
                raise ValueError(
                    f"'{name}' produced no scan report. The scan reaches only "
                    "objects the active view layer can select."
                )
            reports[name] = report
        return reports

    @blender_api
    def merge_by_distance(self, *object_names: str,
                          merge_threshold: float = 1e-4,
                          clear_stale_caches: bool = True) -> "_Solver":
        """Weld near-coincident vertices.  Changes the vertex count.

        A surviving pair sits far inside the contact gap, where the cubic
        barrier's ``mass / gap^2`` dynamic stiffness contributes Hessian
        entries many orders of magnitude larger than the rest of the row,
        so the fp32 Newton matrix loses rank and the run stops on the
        solver's SPD guard naming no geometry.  Welding is what makes such
        a mesh simulable.

        The vertex count moves, which invalidates the PC2 display cache
        and any captured deformation on the object, and can shift which
        vertices a pin group holds.  :meth:`scan_meshes` reports those per
        object under ``dependents``.  Run Transfer again afterward.

        Args:
            *object_names: One or more mesh object names.
            merge_threshold: Weld vertices closer together than this, in
                local units.
            clear_stale_caches: Delete the display and capture caches the
                count change invalidates, which is what the panel's
                confirmation does.  Pass ``False`` to keep them, and expect
                the viewport overlay to read data sized for the old vertex
                count until the next Transfer rewrites it.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.merge_by_distance("Shirt", merge_threshold=1e-4)
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.merge_by_distance,
            object_names=list(object_names),
            merge_threshold=merge_threshold,
            acknowledge=True,
            clear_stale_caches=clear_stale_caches,
        )
        return self

    @blender_api
    def remove_loose_vertices(self, *object_names: str,
                              clear_stale_caches: bool = True) -> "_Solver":
        """Delete vertices that belong to no face.  Changes the vertex count.

        The solver averages a vertex's contact parameters over its
        incident faces and aborts when it has none.  Loose edges go with
        the vertices they connect.  Pinned vertices are exempt, since a
        pin holds them regardless, and a SAND particle mesh is skipped
        outright: every grain center is legitimately faceless.

        The vertex count moves, which invalidates the PC2 display cache
        and any captured deformation on the object, and can shift which
        vertices a pin group holds.  Run Transfer again afterward.

        Args:
            *object_names: One or more mesh object names.
            clear_stale_caches: Delete the display and capture caches the
                count change invalidates, which is what the panel's
                confirmation does.  Pass ``False`` to keep them, and expect
                the viewport overlay to read data sized for the old vertex
                count until the next Transfer rewrites it.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.remove_loose_vertices("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.remove_loose_vertices,
            object_names=list(object_names),
            acknowledge=True,
            clear_stale_caches=clear_stale_caches,
        )
        return self

    @blender_api
    def dissolve_degenerate_faces(self, *object_names: str,
                                  merge_threshold: float = 1e-4,
                                  clear_stale_caches: bool = True) -> "_Solver":
        """Collapse zero-area faces and zero-length edges.

        Changes the vertex count.  A face with no area has no defined
        normal, so the contact normal and the bending hinge built on it
        are both undefined.

        The vertex count moves, which invalidates the PC2 display cache
        and any captured deformation on the object, and can shift which
        vertices a pin group holds.  Run Transfer again afterward.

        Args:
            *object_names: One or more mesh object names.
            merge_threshold: Edges shorter than this, in local units, are
                collapsed.
            clear_stale_caches: Delete the display and capture caches the
                count change invalidates, which is what the panel's
                confirmation does.  Pass ``False`` to keep them, and expect
                the viewport overlay to read data sized for the old vertex
                count until the next Transfer rewrites it.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.dissolve_degenerate_faces("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.dissolve_degenerate_faces,
            object_names=list(object_names),
            merge_threshold=merge_threshold,
            acknowledge=True,
            clear_stale_caches=clear_stale_caches,
        )
        return self

    @blender_api
    def delete_duplicate_faces(self, *object_names: str) -> "_Solver":
        """Delete faces that repeat an existing face's vertex set.

        Two faces on the same vertices contribute their contact and
        elastic terms twice.  The vertex count is unchanged, so no cache
        is invalidated.  Run Transfer again afterward.

        Args:
            *object_names: One or more mesh object names.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.delete_duplicate_faces("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.delete_duplicate_faces,
            object_names=list(object_names),
        )
        return self

    @blender_api
    def triangulate_for_solver(self, *object_names: str) -> "_Solver":
        """Triangulate every face with more than three corners.

        Transfer triangulates on its own at encode time, so this is for
        when the triangulation has to be visible and stable in the
        viewport: with the diagonals fixed in the mesh, Blender has none
        left to re-pick from the deformed shape, and the displayed surface
        stops drifting from the simulated one.  The vertex count is
        unchanged, so no cache is invalidated.  For a mesh whose symmetry
        matters under bending, use :meth:`symmetric_triangulate` instead.

        Args:
            *object_names: One or more mesh object names.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.triangulate_for_solver("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.triangulate_for_solver,
            object_names=list(object_names),
        )
        return self

    @blender_api
    def recalculate_normals_outside(self, *object_names: str) -> "_Solver":
        """Make face winding consistent and outward.

        Inconsistent winding flips the normal a face contributes, which
        the contact and inflate terms read.  The vertex count is
        unchanged, so no cache is invalidated, but the encoder captures
        winding at Transfer time, so run Transfer again afterward.

        Args:
            *object_names: One or more mesh object names.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.recalculate_normals_outside("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.recalculate_normals_outside,
            object_names=list(object_names),
        )
        return self

    @blender_api
    def symmetric_triangulate(self, *object_names: str) -> "_Solver":
        """Triangulate by poking each face, keeping the mesh symmetric.

        A single-diagonal triangulation breaks a mirror-symmetric mesh's
        symmetry, which shows up as a lopsided drape under bending.
        Poking inserts a center vertex and fans the face into triangles
        around it instead.

        That adds one vertex per face, so the PC2 display cache and any
        captured deformation on the object are invalidated exactly as the
        count-changing repairs invalidate them.  Nothing here deletes
        them: run Transfer to rewrite the display cache, and Capture
        Deformation (or :meth:`recapture_all_deformations`) to re-take the
        captures.

        Args:
            *object_names: One or more mesh object names.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a name is missing, is not a mesh, or the object
                is outside the active view layer or cannot be selected in
                it (hidden in the viewport, hidden by its collection, or
                carrying Disable Selection).

        Example::

            solver.symmetric_triangulate("Shirt")
        """
        from ...mcp.handlers import mesh_cleaning

        _handler_result(
            mesh_cleaning.symmetric_triangulate,
            object_names=list(object_names),
        )
        return self

    # -- Particle mesh -------------------------------------------------------

    @blender_api
    def convert_to_particle_mesh(self, object_name: str, grain_radius: float,
                                 extra_spacing: float = 0.0,
                                 rng_seed: int = 0) -> int:
        """Replace a solid mesh with the grain cloud a SAND group simulates.

        Destructive: the faces are discarded and the object becomes a
        faceless mesh of loose vertices carrying a render-only Particle
        Mesh modifier.  The grain count is not chosen, it is whatever
        fills the volume at the given separation, which is why it is the
        return value.

        ``grain_radius`` is stamped onto the object and is what the
        encoder reads, in preference to the group's ``sand_grain_radius``.
        The non-overlapping seed spacing derives from it and it is also
        the contact skin, so it is locked once the object is converted and
        the panel shows it read-only.  Pick it before converting.

        Args:
            object_name: Name of a mesh object that has faces and is not
                already a particle mesh.
            grain_radius: Physical grain radius, which is also the contact
                skin.
            extra_spacing: Gap added between grains beyond touching.
                ``0.0`` packs them as densely as non-overlap allows.
            rng_seed: Seed for the Poisson-disk seeding, for a repeatable
                cloud.

        Returns:
            The number of grains seeded.

        Raises:
            ValueError: If the object is missing, is not a mesh, has no
                faces, is already a particle mesh, ``grain_radius`` is not
                positive, or no grain fits inside the mesh.

        Example::

            n_grains = solver.convert_to_particle_mesh(
                "Pile", grain_radius=0.01,
            )
            sand = solver.create_group("Sand", type="SAND")
            sand.add("Pile")
            sand.param.sand_particle_mass = 10.0  # grams per grain
        """
        from ...mcp.handlers import object_ops

        result = _handler_result(
            object_ops.convert_to_particle_mesh,
            object_name=object_name,
            grain_radius=grain_radius,
            extra_spacing=extra_spacing,
            rng_seed=rng_seed,
        )
        return result["grain_count"]

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

    # -- Deformation caches --------------------------------------------------

    @blender_api
    def recapture_all_deformations(self) -> "_Solver":
        """Re-capture every deforming STATIC collider and every animated pin.

        One pass over the whole scene instead of one Capture Deformation
        per object.  The statics are captured first and the pins after,
        since the two share the depsgraph and cannot run at once.

        Note:
            The captures advance on Blender's event loop: this returns
            once the first one has started, and the rest complete over
            later ticks.  A script that keeps running on the same tick
            holds the loop and blocks them, so schedule whatever depends
            on the caches and gate it on :meth:`is_capture_running`.  A
            Blender started with ``--background`` runs no ticks and
            captures nothing.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If there is nothing to re-capture, or a capture or
                bake is already running.

        Example::

            import bpy

            def transfer_when_captured():
                if solver.is_capture_running():
                    return 0.1  # come back in 100 ms
                solver.transfer_data()
                return None

            solver.recapture_all_deformations()
            bpy.app.timers.register(transfer_when_captured)
        """
        from ...mcp.handlers import object_ops

        _handler_result(object_ops.recapture_all_deformations)
        return self

    @blender_api
    def clear_all_deformations(self) -> "_Solver":
        """Delete every captured deformation cache in the scene.

        Covers the STATIC-collider deform caches and the animated-pin
        captures across the active groups, plus any cache orphaned by an
        object that was deleted or taken out of its group, which nothing
        else reaches.  The objects keep their deformers, so
        :meth:`recapture_all_deformations` rebuilds what this removes.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If there is no captured cache to clear, or a
                capture or bake is already running.

        Example::

            solver.clear_all_deformations()
        """
        from ...mcp.handlers import object_ops

        _handler_result(object_ops.clear_all_deformations)
        return self

    @blender_api
    def is_capture_running(self) -> bool:
        """True while a deformation capture is in flight.

        Covers both phases of :meth:`recapture_all_deformations`, the
        STATIC-collider captures and the pin captures, so a script waits
        on the whole pass with one predicate.  Read it from a timer or a
        handler: a capture advances only when the script has handed
        control back to Blender's event loop.

        Example::

            import bpy

            def report_when_done():
                if solver.is_capture_running():
                    return 0.1  # come back in 100 ms
                print("captures finished")
                return None

            bpy.app.timers.register(report_when_done)
        """
        from ...ui.dynamics import pin_capture_ops, static_deform_ops

        return bool(
            static_deform_ops.is_capture_running()
            or pin_capture_ops.is_pin_capture_running()
        )

    # -- Cache export --------------------------------------------------------

    @blender_api
    def export_usd(self, filepath: str) -> "_Solver":
        """Export the simulated mesh sequence as a USD cache.

        A lighter result than baking shape keys: the deformation is
        sampled per frame straight from the solver cache into a file other
        DCC tools play back, and the scene itself is left untouched.

        Every frame must be fetched before the export runs, and the export
        refuses while a run, bake or capture is in flight, and outside
        Object Mode.  Rod and curve objects are not carried by this
        format; :meth:`get_unexportable_curves` names the ones that will
        be left out.

        Args:
            filepath: Destination path, taken as written once a ``//``
                blend-relative prefix is resolved.  The suffix is what
                picks the USD flavor, so give one of ``.usdc`` (crate),
                ``.usda`` (ASCII), ``.usd`` or ``.usdz`` (package).  The
                parent directory must exist.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If frames are unfetched, another solver activity
                is in progress, the scene is in Edit or Sculpt mode, there
                is no simulated mesh sequence, the destination directory
                does not exist, or the export did not complete.

        Example::

            solver.export_usd("/tmp/drape.usdc")
        """
        from ...mcp.handlers import simulation

        _handler_result(simulation.export_usd, filepath=filepath)
        return self

    @blender_api
    def export_alembic(self, filepath: str) -> "_Solver":
        """Export the simulated mesh sequence as an Alembic (ABC) cache.

        A lighter result than baking shape keys: the deformation is
        sampled per frame straight from the solver cache into a file other
        DCC tools play back, and the scene itself is left untouched.

        Every frame must be fetched before the export runs, and the export
        refuses while a run, bake or capture is in flight, and outside
        Object Mode.  Rod and curve objects are not carried by this
        format; :meth:`get_unexportable_curves` names the ones that will
        be left out.

        Args:
            filepath: Destination ``.abc`` path, taken as written once a
                ``//`` blend-relative prefix is resolved.  The parent
                directory must exist.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If frames are unfetched, another solver activity
                is in progress, the scene is in Edit or Sculpt mode, there
                is no simulated mesh sequence, the destination directory
                does not exist, or the export did not complete.

        Example::

            solver.export_alembic("/tmp/drape.abc")
        """
        from ...mcp.handlers import simulation

        _handler_result(simulation.export_alembic, filepath=filepath)
        return self

    @blender_api
    def get_unexportable_curves(self) -> list[str]:
        """Names of the simulated curves a cache export leaves out.

        A rod deforms through a frame-change handler rather than through
        the cache modifier the exporters sample, so a CURVE object carrying
        a solver cache cannot be written to USD or Alembic.  Bake Animation
        is the route that carries one.

        Example::

            missing = solver.get_unexportable_curves()
            if missing:
                print("not exported:", ", ".join(missing))
            solver.export_usd("/tmp/drape.usdc")
        """
        from ...ui.dynamics.export_ops import _excluded_sim_curves

        return _excluded_sim_curves(bpy.context)

    # -- Fallback ------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(bpy.ops.zozo_contact_solver, name)

solver = _Solver()
