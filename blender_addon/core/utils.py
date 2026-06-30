# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from ..models.groups import decode_vertex_group_identifier, iterate_active_object_groups
from .transform import world_matrix


def get_category_name():
    """Get the category name for the add-on."""
    return "ZOZO's Contact Solver"


# Characters that break a connection path when it is interpolated into a shell
# command (``cd {path} && ...``, ``docker exec -w {path} ...``).  Whitespace
# plus shell and glob metacharacters.  ``~`` is intentionally NOT included:
# ``~/work`` is a common, safe remote path and the shell expands it fine.  The
# usual path components (``/``, ``\``, ``:``, ``.``, ``-``, ``_``) are also
# allowed so Windows drive letters and separators pass.
_INVALID_PATH_CHARS = frozenset(
    " \t\n\r"          # whitespace
    "&|;<>()$`\"'"     # shell metacharacters
    "*?[]{}#"          # glob, brace expansion, comment
    "!%^"              # history expansion, env, caret
)


def find_invalid_path_char(path: str) -> str | None:
    """Return the first space or shell-unsafe character in *path*, else ``None``.

    Used to warn about, and refuse, connection paths that would break (or be
    misinterpreted) when interpolated into a shell command.  Returns ``None``
    for an empty or whitespace-only path, so callers can chain
    ``find_invalid_path_char(p) is None`` without rejecting an intentionally
    blank optional path; emptiness is validated separately per connection type.
    """
    for ch in path.strip():
        if ch in _INVALID_PATH_CHARS:
            return ch
    return None


# Windows refuses to open a path of 260 characters or more (the classic
# MAX_PATH limit) unless system-wide long-path support is enabled. The build
# pipeline writes cache files several directories below the solver root, so a
# long root pushes the deepest of them past the limit and the Transfer dies
# with a bare ``FileNotFoundError`` that names a path nobody recognizes. The
# deepest file is the tetrahedralize cache; its full server-side path mirrors
# ``datamodel/app.rs`` ``compose_data_dir`` plus the cache filename:
#
#   <root>\local\share\ppf-cts\git-<branch>\<project>\.cash\
#   <64-hex>__<64-hex>_tetrahedralize_.npz.npz
#
# The git branch is resolved on the server and is ``unknown`` for a packaged
# Windows build (no ``.git``), which is the case this guards.
WINDOWS_MAX_PATH = 260


def projected_windows_cache_path_len(base_path: str, project_name: str) -> int:
    """Length of the longest cache file path the build pipeline writes under
    *base_path* for *project_name* on a Windows server.

    Mirrors the server-side layout described in the ``WINDOWS_MAX_PATH`` note,
    so the value matches the path that would actually appear in a
    ``FileNotFoundError`` (computed exactly for the default, empty-arg
    tetrahedralize cache; a cache with extra tetrahedralize args is longer, so
    this is a lower bound).
    """
    root = base_path.strip().rstrip("/\\")
    hex64 = "f" * 64  # a SHA-256 hex digest is 64 chars
    cache_file = f"{hex64}__{hex64}_tetrahedralize_.npz.npz"
    tail = "\\".join(
        [
            "local", "share", "ppf-cts", "git-unknown",
            project_name.strip() or "unnamed", ".cash", cache_file,
        ]
    )
    return len(root) + len("\\") + len(tail)


def windows_path_too_long(base_path: str, project_name: str) -> int | None:
    """Return the projected deepest cache-path length when it reaches the
    Windows ``MAX_PATH`` limit, else ``None``.

    This is a pure measurement of the path length. It does NOT consider
    whether long-path support is enabled on the host; callers that warn the
    user should also consult :func:`windows_long_paths_enabled`, since with
    long paths on the limit no longer applies and the warning is just noise.

    Returns ``None`` for an empty/whitespace-only path so callers can chain
    ``windows_path_too_long(p, n) is None`` next to
    :func:`find_invalid_path_char` without rejecting a blank field.
    """
    if not base_path.strip():
        return None
    projected = projected_windows_cache_path_len(base_path, project_name)
    return projected if projected >= WINDOWS_MAX_PATH else None


# Cache for windows_long_paths_enabled(). The registry value is honored per
# process at startup (the long-path opt-in is read once), so it won't change
# within a Blender session, and the panel's draw() would otherwise re-read it
# on every redraw. ``None`` means "not yet queried".
_windows_long_paths_enabled: bool | None = None


def _query_windows_long_paths_enabled() -> bool:
    """Read the system-wide long-path flag from the Windows registry.

    ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled``
    is the DWORD that opts the whole system out of the ``MAX_PATH`` limit.
    Returns ``False`` on non-Windows hosts and whenever the value is missing
    or unreadable (i.e. assume the limit applies unless we can prove it
    doesn't).
    """
    import sys

    if sys.platform != "win32":
        return False
    try:
        import winreg  # Windows-only stdlib module.

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return int(value) == 1
    except (OSError, ValueError):
        return False


def windows_long_paths_enabled() -> bool:
    """True when Windows long-path support is enabled system-wide, so paths
    past ``MAX_PATH`` no longer fail.

    ``False`` on non-Windows hosts and whenever the registry flag is unset or
    unreadable. Cached after the first read (see ``_windows_long_paths_enabled``).
    """
    global _windows_long_paths_enabled
    if _windows_long_paths_enabled is None:
        _windows_long_paths_enabled = _query_windows_long_paths_enabled()
    return _windows_long_paths_enabled


def find_invalid_name_char(name: str) -> str | None:
    """Return the first character in *name* that is not filename-safe, else ``None``.

    A project name becomes a single directory-name component on the server and
    is interpolated into commands, so it is held to a stricter rule than a
    path: only letters, digits, ``.``, ``-`` and ``_`` are allowed. That
    rejects spaces, shell/glob metacharacters, AND path separators (``/``,
    ``\\``, ``:``), none of which belong in a single name component. Returns
    ``None`` for an empty or whitespace-only name; emptiness is validated
    separately.
    """
    for ch in name.strip():
        if not (ch.isalnum() or ch in "._-"):
            return ch
    return None


def count_duplicate_faces(obj) -> int:
    """Return how many triangles share their full vertex set with an
    earlier triangle once *obj*'s mesh is tessellated the way the encoder
    tessellates it (Blender's ``loop_triangles``). ``0`` means every
    triangle is unique.

    Two coincident triangles (the same three vertices, in any winding)
    make the solver's bending-hinge builder produce a degenerate
    element and abort the simulation at startup. They almost always come
    from doubled geometry welded with Merge by Distance, common in
    airbag / inflate setups, so the dynamics pipeline rejects them
    up-front and names the object rather than silently dropping faces
    the user may have placed on purpose.

    Returns ``0`` for non-mesh objects (curves, etc.), which have no
    polygons in the Blender mesh sense.

    Used by:
      * ``OBJECT_OT_AddObjectsToGroup`` to refuse doubled meshes at
        assignment time, so the user sees an explicit error popup.
      * ``encoder.mesh._build_obj_data`` to fail the Transfer if a
        doubled mesh got assigned through a path that bypasses the
        operator (older saves, MCP scripts).
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return 0
    from .numpy_mesh_utils import loop_triangle_indices
    seen: set[tuple[int, ...]] = set()
    duplicates = 0
    for tri in loop_triangle_indices(obj.data):
        key = tuple(sorted(int(v) for v in tri))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def find_linked_duplicate_siblings(obj) -> list[str]:
    """Return the names of other Blender objects that share *obj*'s data
    block (i.e. ``obj`` is a Linked Duplicate / shallow copy, typically
    created by Alt-D).

    The simulator works under the assumption that each object owns its
    own mesh data: shared data means a vertex coordinate written for
    one assigned object would silently propagate to its sibling, which
    in turn would cause the encoder to ship inconsistent geometry,
    corrupt PC2 playback, and break the topology hash. The dynamics
    pipeline rejects these objects up-front.

    Returns an empty list when ``obj`` has its own data block, or when
    ``obj.data`` is None (curves with no spline yet, etc.).
    """
    if obj is None or obj.data is None:
        return []
    # ``users`` counts every reference to the data block, including
    # the active object itself. A solo owner has ``users == 1``.
    if obj.data.users <= 1:
        return []
    return [
        o.name
        for o in bpy.data.objects
        if o is not obj and o.data is obj.data
    ]


def get_timer_wait_time():
    """Get the wait time for the timer."""
    return 0.25


def redraw_all_areas(context):
    """Tag all screen areas for redraw."""
    for area in context.screen.areas:
        area.tag_redraw()


def redraw_all_windows(area_type: str | None = None):
    """Tag areas across all windows for redraw, optionally by type.

    Iterates ``window_manager.windows`` (not ``context.screen``) so it is
    safe to call from the public Python API and from ``bpy.app.timers``
    callbacks where the active screen is unreliable or None. Pass
    ``area_type`` (e.g. ``"VIEW_3D"``) to limit the redraw to one editor type,
    or leave it None to tag every area.
    """
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area_type is None or area.type == area_type:
                area.tag_redraw()


def check_vec3(name: str, v, error_cls) -> tuple[float, float, float]:
    """Coerce `v` to a length-3 tuple of floats, raising `error_cls` on failure.

    Callers plug in their layer's exception type (MCPError, ValidationError,
    MutationError, ...) so the message vocabulary is shared but the error
    class stays specific to the boundary that raised.
    """
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise error_cls(f"{name} must be a length-3 list/tuple, got {v!r}")
    try:
        return tuple(float(x) for x in v)
    except (TypeError, ValueError) as e:
        raise error_cls(f"{name} components must be numeric: {e}")


def parse_vertex_index(data_path: str) -> int | None:
    """Parse the vertex index from a data path string."""
    start = data_path.find("[") + 1
    end = data_path.find("]")
    if start >= 0 and end > start:
        try:
            return int(data_path[start:end])
        except ValueError:
            pass
    return None


def _get_fcurves(action):
    """Get fcurves from an action (Blender 5.0+ layered API)."""
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                if bag.fcurves:
                    return bag.fcurves
    return []


_TRANSFORM_PATHS = (
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
)


def has_transform_fcurves(obj) -> bool:
    """True if *obj* has any object-level transform fcurve (loc/rot/scale).

    Used by the static-ops UI and encoder to enforce mutual exclusion:
    a static object with Blender keyframe animation cannot also use
    UI-assigned move/spin/scale ops (only one source of motion at a
    time).
    """
    if obj is None or not hasattr(obj, "animation_data"):
        return False
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    for fc in _get_fcurves(ad.action):
        path = getattr(fc, "data_path", "") or ""
        if any(path == p or path.endswith(f".{p}") for p in _TRANSFORM_PATHS):
            return True
    return False


# Modifier types whose evaluation can move mesh vertices off the
# rest pose. Picked by inspecting Blender's modifier categories; if
# any of these is in the stack AND enabled for the depsgraph, the
# evaluated mesh may differ from obj.data.vertices and a STATIC
# collider needs a Capture Deformation pass to feed the solver.
#
# ``NODES`` is included, but a Geometry Nodes modifier is NOT taken
# at face value: ``has_deforming_modifier_stack`` only counts it when
# its node group actually writes vertex positions (see
# ``_nodes_modifier_can_deform``). A Geometry Nodes modifier can do
# anything from full procedural deformation to a pure normal recompute
# (Blender 4.1+'s default "Smooth by Angle" node group, auto-added on
# most meshes, is the most common case and moves no vertices).
# Counting every NODES modifier as a deformer forces Capture
# Deformation on objects that don't need it; the position-write scan
# excludes those, and ``is_deforming_static_object`` keeps a
# depsgraph-sampling backstop for groups whose writers the scan
# doesn't enumerate.
_DEFORMING_MODIFIER_TYPES = frozenset({
    "ARMATURE",
    "CAST",
    "CLOTH",
    "CORRECTIVE_SMOOTH",
    "CURVE",
    "DISPLACE",
    "HOOK",
    "LAPLACIANDEFORM",
    "LAPLACIANSMOOTH",
    "LATTICE",
    "MESH_DEFORM",
    # Geometry Nodes can displace existing vertices (e.g. a Set Position
    # wave driven by Scene Time). It may also change topology, in which
    # case it is additionally treated as generative for MESH_CACHE
    # placement (see ``_GENERATIVE_MODIFIER_TYPES`` in ``core/pc2.py``);
    # the two classifications are independent. A position-writing GN is
    # treated as a deformer here (gated by ``_nodes_modifier_can_deform``)
    # so the pin overlay, Capture Deformation, and the encoder follow
    # GN-driven motion the same way they follow Armature/Lattice.
    "NODES",
    "SHRINKWRAP",
    "SIMPLE_DEFORM",
    "SMOOTH",
    "SOFT_BODY",
    "SURFACE_DEFORM",
    "VOLUME_DISPLACE",
    "WARP",
    "WAVE",
})


def _has_shape_key_animation(obj) -> bool:
    """True if *obj* has any shape-key value fcurve.

    Shape keys with animated `.value` deform the evaluated mesh even
    though the modifier list looks empty. The encoder must not silently
    ignore them.
    """
    if obj is None or obj.type != "MESH":
        return False
    sk = getattr(obj.data, "shape_keys", None)
    if sk is None:
        return False
    ad = getattr(sk, "animation_data", None)
    if ad is None or ad.action is None:
        return False
    return any(_get_fcurves(ad.action))


# Geometry Nodes node types that can move existing mesh vertices off
# their rest position. A GN modifier whose tree contains none of these
# cannot deform the mesh (it may still recompute normals, set shade-
# smooth flags, assign attributes, ...), so it must not be treated as a
# deformer. Instance-transform nodes (Translate/Rotate/Scale Instances,
# Set Instance Transform) are intentionally excluded: they move
# instances, not the realized collider mesh's own vertices.
_GN_POSITION_WRITING_NODES = frozenset({
    "GeometryNodeSetPosition",
    "GeometryNodeTransform",
    "GeometryNodeDeformCurvesOnSurface",
})


def _nodes_modifier_can_deform(mod) -> bool:
    """True if a Geometry Nodes *mod* can move mesh vertices.

    Scans the modifier's node group, recursing into nested node
    groups, for any node that writes geometry position (see
    ``_GN_POSITION_WRITING_NODES``). A group with no such node, e.g.
    Blender's auto-added "Smooth by Angle" (which only sets shade-
    smooth flags), leaves every vertex at rest and is not a deformer.

    Conservative on failure: if the node group can't be introspected,
    returns True so the caller errs toward treating it as deforming
    (``is_deforming_static_object``'s depsgraph backstop then settles
    it per-frame for the one path that hard-fails on a missing cache).
    """
    ng = getattr(mod, "node_group", None)
    if ng is None:
        return False
    try:
        seen = set()
        stack = [ng]
        while stack:
            tree = stack.pop()
            if tree is None or tree.as_pointer() in seen:
                continue
            seen.add(tree.as_pointer())
            for node in tree.nodes:
                if node.bl_idname in _GN_POSITION_WRITING_NODES:
                    return True
                nested = getattr(node, "node_tree", None)
                if nested is not None:
                    stack.append(nested)
        return False
    except Exception:
        return True


def has_deforming_modifier_stack(obj) -> bool:
    """True if *obj*'s modifier stack contains any vertex-moving deformer.

    Cheap declarative check (no depsgraph round-trip, except a node-tree
    scan for Geometry Nodes modifiers). Only modifiers enabled for
    ``show_viewport`` count — a muted deformer doesn't contribute to what
    Blender or the depsgraph sees.

    A Geometry Nodes modifier counts only when its node group actually
    writes vertex positions: the ubiquitous auto-added "Smooth by Angle"
    group moves nothing and must not be mistaken for a deformer.
    """
    if obj is None or obj.type != "MESH":
        return False
    if not hasattr(obj, "modifiers"):
        return False
    for mod in obj.modifiers:
        if not getattr(mod, "show_viewport", True):
            continue
        if mod.type not in _DEFORMING_MODIFIER_TYPES:
            continue
        if mod.type == "NODES" and not _nodes_modifier_can_deform(mod):
            continue
        return True
    return False


def eval_deform_local_positions(obj, context=None, exclude_modifier_name=None):
    """Return ``(N, 3)`` float32 local-space vertex positions of *obj*
    with its deform-only modifier stack evaluated at the current frame,
    or ``None`` when the result can't stand in for the rest mesh.

    Honors Geometry Nodes / Armature / Lattice deforms so callers can
    capture the shape the artist sees, instead of the undeformed rest
    cage. ``exclude_modifier_name`` (typically the addon's
    ``ContactSolverCache``) is temporarily hidden during evaluation:
    that MESH_CACHE replays prior solver output with OVERWRITE, so
    reading it back would feed the output into the next input.

    Returns ``None`` when the evaluated vertex count differs from the
    base mesh (a generative modifier changed topology, so base
    triangulation / vertex-group indices would no longer line up) or
    when evaluation isn't available.
    """
    import numpy as np

    if obj is None or obj.type != "MESH":
        return None
    if context is None:
        context = bpy.context
    toggled = []
    try:
        if exclude_modifier_name:
            for m in obj.modifiers:
                if m.name == exclude_modifier_name and m.show_viewport:
                    m.show_viewport = False
                    toggled.append(m)
        deps = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        eval_mesh = eval_obj.to_mesh()
        try:
            n = len(eval_mesh.vertices)
            if n != len(obj.data.vertices):
                return None
            co = np.empty(n * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", co)
            return co.reshape(n, 3)
        finally:
            eval_obj.to_mesh_clear()
    except Exception:
        return None
    finally:
        for m in toggled:
            m.show_viewport = True


def validate_bend_reference(source_obj, ref_obj, context=None, group_type="SHELL"):
    """Validate that *ref_obj* is a positions-only topological copy of
    *source_obj*, usable as a bending rest-angle reference.

    The check matches how each group type ships geometry:

    * SHELL and mesh ROD: evaluate ``ref_obj`` through its full modifier /
      geometry-nodes stack and compare the result against ``source_obj``'s
      base mesh. The evaluated reference must have the same vertex count and
      identical connectivity (faces for SHELL, edges for ROD); only vertex
      positions may differ.
    * Curve ROD: sample both curves the way the encoder does (control-point
      level, ``sample_curve``) and compare sampled vertex count + edges.
      Curve modifiers / geometry nodes are not sampled, so a curve reference
      must move its control points (directly, or via a modifier baked into
      the control points), mirroring how the source curve rod is shipped.

    Returns ``(True, "")`` on success, or ``(False, message)`` with a
    user-facing error describing the first mismatch found.
    """
    import numpy as np

    if context is None:
        context = bpy.context
    if source_obj is None:
        return False, "Source object not found."
    if ref_obj is None:
        return False, "Reference object not found."
    if ref_obj == source_obj:
        return False, "The reference object must be different from the object itself."
    if ref_obj.type != source_obj.type:
        return False, (
            f"Reference '{ref_obj.name}' is a {ref_obj.type.title()} but "
            f"'{source_obj.name}' is a {source_obj.type.title()}; the "
            f"reference must be the same object type."
        )

    # Curve rod: compare sampled rod vertices (control-point level).
    if group_type == "ROD" and source_obj.type == "CURVE":
        from mathutils import Matrix  # pyright: ignore
        from .curve_rod import sample_curve
        src_v, src_e, _ = sample_curve(source_obj, Matrix.Identity(4))
        ref_v, ref_e, _ = sample_curve(ref_obj, Matrix.Identity(4))
        if len(ref_v) != len(src_v):
            return False, (
                f"Reference '{ref_obj.name}' samples to {len(ref_v)} rod "
                f"vertices but '{source_obj.name}' samples to {len(src_v)}. A "
                f"reference curve must have the same spline structure (only "
                f"control-point positions may change)."
            )
        if not np.array_equal(np.asarray(ref_e), np.asarray(src_e)):
            return False, (
                f"Reference '{ref_obj.name}' has different rod connectivity "
                f"than '{source_obj.name}'. Only control-point positions may "
                f"change in a reference curve."
            )
        return True, ""

    if source_obj.type != "MESH":
        return False, (
            f"'{source_obj.name}' is neither a mesh nor a curve rod; "
            f"reference rest angles are not supported for it."
        )

    is_rod = group_type == "ROD"
    src_mesh = source_obj.data
    n_src = len(src_mesh.vertices)
    if is_rod:
        src_conn = [tuple(sorted(e.vertices)) for e in src_mesh.edges]
        conn_label = "edge"
    else:
        src_conn = [tuple(p.vertices) for p in src_mesh.polygons]
        conn_label = "face"

    deps = context.evaluated_depsgraph_get()
    eval_obj = ref_obj.evaluated_get(deps)
    eval_mesh = eval_obj.to_mesh()
    try:
        n_ref = len(eval_mesh.vertices)
        if n_ref != n_src:
            return False, (
                f"Reference '{ref_obj.name}' has {n_ref} vertices after "
                f"evaluating its modifiers / geometry nodes, but "
                f"'{source_obj.name}' has {n_src}. A reference must be a "
                f"topological copy with only vertex positions changed."
            )
        if is_rod:
            ref_conn = [tuple(sorted(e.vertices)) for e in eval_mesh.edges]
        else:
            ref_conn = [tuple(p.vertices) for p in eval_mesh.polygons]
    finally:
        eval_obj.to_mesh_clear()

    if ref_conn != src_conn:
        return False, (
            f"Reference '{ref_obj.name}' has different {conn_label} "
            f"connectivity than '{source_obj.name}' after evaluation. Only "
            f"vertex positions may change in a reference object."
        )
    return True, ""


def eval_reference_local_positions(ref_obj, context=None):
    """Return ``(N, 3)`` float32 local-space vertex positions of *ref_obj*
    with its FULL modifier / geometry-nodes stack evaluated at the current
    frame, or ``None`` when evaluation isn't available.

    Unlike :func:`eval_deform_local_positions`, this does NOT gate on the
    evaluated count matching the object's own base mesh: a bending
    reference is validated against the SOURCE object's count by the
    caller (see :func:`validate_bend_reference`), so the count check
    belongs there, not here.
    """
    import numpy as np

    if ref_obj is None or ref_obj.type != "MESH":
        return None
    if context is None:
        context = bpy.context
    deps = context.evaluated_depsgraph_get()
    eval_obj = ref_obj.evaluated_get(deps)
    eval_mesh = eval_obj.to_mesh()
    try:
        n = len(eval_mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        eval_mesh.vertices.foreach_get("co", co)
        return co.reshape(n, 3)
    finally:
        eval_obj.to_mesh_clear()


def _depsgraph_mesh_differs_across_range(obj, context) -> bool:
    """Compare depsgraph-evaluated mesh *shape* at ``frame_start`` and
    ``frame_end``. Returns True if any vertex moves in the object's
    LOCAL space between the two samples.

    Local space is the right comparison: the goal is to detect mesh
    SHAPE changes (drivers poking vertex coords, geometry-node
    deformation that fell through the modifier-name list) and let
    rigid object-transform animation be handled separately by
    ``transform_animation``. A previous version of this function
    compared world-space coords, which conflated own-transform
    fcurves with deformation and forced Capture Deformation on
    rigid-only STATIC objects.
    """
    import numpy as np

    from .pc2 import resume_mesh_cache_display, suspend_mesh_cache_display

    scene = context.scene
    if scene.frame_end <= scene.frame_start:
        return False
    saved = scene.frame_current
    # The addon's own ContactSolverCache (a MESH_CACHE with
    # deform_mode='OVERWRITE') replays the PREVIOUS solver output, so with it
    # enabled the evaluated mesh appears to change across the timeline even when
    # the object is rigid. Suspend it for the duration of the two samples so
    # this measures genuine deformer output only. Otherwise a moving STATIC
    # collider that carries a cache (e.g. one previously simulated, or just
    # replaying results) is misread as deforming and wrongly forced to Capture
    # Deformation. Restored in the finally below.
    cache_prior = suspend_mesh_cache_display(obj)
    try:
        def _sample(f):
            scene.frame_set(int(f))
            dg = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(dg)
            eval_mesh = eval_obj.to_mesh()
            try:
                n = len(eval_mesh.vertices)
                if n == 0:
                    return None
                co = np.empty((n, 3), dtype=np.float32)
                eval_mesh.vertices.foreach_get("co", co.ravel())
                return co
            finally:
                eval_obj.to_mesh_clear()

        a = _sample(scene.frame_start)
        b = _sample(scene.frame_end)
    finally:
        scene.frame_set(saved)
        resume_mesh_cache_display(obj, cache_prior)
    if a is None or b is None or a.shape != b.shape:
        return False
    return bool(np.any(np.abs(a - b) > 1e-6))


def _matrix_world_differs_without_own_fcurves(obj, context) -> bool:
    """Detect motion that comes from a parent / constraint / driver
    rather than the object's own location/rotation/scale fcurves.

    ``transform_animation`` samples ``obj.matrix_world`` only at the
    object's own fcurve keyframes; if there are no own fcurves but
    the world matrix still changes across the timeline, the encoder
    would silently drop the motion. Surface that case so the user
    knows to Capture Deformation (which bakes the evaluated motion
    from every source).

    Returns False when the object DOES have its own loc/rot/scale
    fcurves: the encoder's transform-keyframe sampler covers those,
    so the rigid path is sufficient.
    """
    import numpy as np

    if has_transform_fcurves(obj):
        return False
    scene = context.scene
    if scene.frame_end <= scene.frame_start:
        return False
    saved = scene.frame_current
    try:
        def _mw(f):
            scene.frame_set(int(f))
            dg = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(dg)
            return np.array(eval_obj.matrix_world, dtype=np.float64)
        a = _mw(scene.frame_start)
        b = _mw(scene.frame_end)
    finally:
        scene.frame_set(saved)
    return bool(np.any(np.abs(a - b) > 1e-6))


def _has_nonfcurve_motion_source(obj) -> bool:
    """Cheap (no depsgraph, no writes) check for a motion source the rigid
    own-fcurve ``transform_animation`` path can't capture: a parent, a
    constraint, a transform driver, or an NLA track. Used by the UI to keep
    the Capture Deformation button reachable without running the depsgraph
    sampler on every redraw; the encoder's full check is the authoritative
    gate."""
    if obj is None:
        return False
    if obj.parent is not None:
        return True
    if len(getattr(obj, "constraints", ())) > 0:
        return True
    ad = getattr(obj, "animation_data", None)
    if ad is not None:
        for d in ad.drivers:
            dp = getattr(d, "data_path", "") or ""
            if any(t in dp for t in ("location", "rotation", "scale")):
                return True
        if len(ad.nla_tracks) > 0:
            return True
    return False


def _mesh_shape_could_animate(obj) -> bool:
    """Cheap (no depsgraph, no writes) over-approximation of whether *obj*'s
    evaluated mesh SHAPE can change across the timeline.

    Returns True whenever some source could move this mesh's vertices frame
    to frame: any modifier (a deform/generative modifier, or one whose
    parameters are animated or driven), shape keys (their values can be
    keyed or driven), mesh-level animation data (drivers on vertex coords),
    or a parent (parent-relative armature/lattice deform). A fully inert
    rigid mesh, none of the above, returns False; in that case the
    evaluated mesh shape is provably frame-invariant, so the caller can
    skip ``_depsgraph_mesh_differs_across_range`` and its two whole-scene
    frame evaluations.

    Own loc/rot/scale animation (``obj.animation_data`` action fcurves on a
    mesh with no modifier/shape-key/parent) is deliberately NOT a trigger:
    it moves the object transform, not the mesh shape, and is handled by
    the rigid ``transform_animation`` path, not the deform path.
    """
    if obj is None or obj.type != "MESH":
        return False
    if len(getattr(obj, "modifiers", ())) > 0:
        return True
    if obj.parent is not None:
        return True
    mesh = obj.data
    if mesh is not None:
        if getattr(mesh, "shape_keys", None) is not None:
            return True
        if getattr(mesh, "animation_data", None) is not None:
            return True
    return False


def is_deforming_static_object(obj, context, allow_eval: bool = True) -> bool:
    """True if *obj* needs a Capture Deformation pass for the solver.

    Four-tier detection:
      1. Declarative: deforming modifier stack.
      2. Shape-key animation.
      3. Local-space mesh shape change across the timeline (catches
         driver-only and geometry-node deformation).
      4. Externally driven object motion: ``matrix_world`` changes
         across the timeline AND the object has no own loc/rot/scale
         fcurves (so the rigid ``transform_animation`` path would
         silently miss it).

    Pure own-transform animation (rigid loc/rot/scale fcurves on a
    mesh with constant shape and no parent/constraint motion) returns
    False, and the encoder uses the lighter ``transform_animation``
    path instead.

    ``allow_eval`` gates tiers 3-4, which sample the depsgraph (they call
    ``scene.frame_set`` and temporarily toggle the ContactSolverCache
    modifier). Those mutate scene state and must NOT run from a restricted
    context such as a UI ``draw()`` handler, where Blender forbids ID writes
    and per-redraw frame stepping would be unusable. Pass ``allow_eval=False``
    from draw to get the cheap declarative tiers only; the encoder leaves it
    True for the full, authoritative gate.
    """
    if obj is None or obj.type != "MESH":
        return False
    if has_deforming_modifier_stack(obj):
        return True
    if _has_shape_key_animation(obj):
        return True
    if context is None or not allow_eval:
        return False
    # Tiers 3-4 each step the timeline twice (``scene.frame_set`` at
    # frame_start and frame_end) and re-evaluate the whole-scene depsgraph,
    # so a single call costs two full frame evaluations. On a heavy
    # collision scene with many rigid STATIC colliders this dominates the
    # encode (hundreds of full-scene evals). Gate each behind a cheap,
    # no-depsgraph pre-check that is a strict SUPERSET of the motion it can
    # detect, so an inert rigid collider skips the sampling entirely:
    #   * tier 3 reports a LOCAL mesh-shape change only if some animatable
    #     source can move this mesh's verts (``_mesh_shape_could_animate``);
    #   * tier 4 reports a world-matrix change from a non-own-fcurve source
    #     (parent / constraint / transform driver / NLA), which is exactly
    #     what ``_has_nonfcurve_motion_source`` reports.
    # If the pre-check is False the sampler is provably False, so skipping
    # it changes no result, only cost.
    if _mesh_shape_could_animate(obj) and _depsgraph_mesh_differs_across_range(
        obj, context
    ):
        return True
    if _has_nonfcurve_motion_source(obj) and _matrix_world_differs_without_own_fcurves(
        obj, context
    ):
        return True
    return False


def get_vertices_in_group(obj, vg) -> list[int]:
    """Return vertex indices belonging to the given vertex group.

    For MESH objects, reads from Blender vertex groups.
    For CURVE objects, reads from custom property ``_pin_{vg.name}``.

    Args:
        obj: Blender object (MESH or CURVE).
        vg: Blender vertex group (or object with .name for curve lookup).

    Returns:
        List of vertex indices that belong to *vg*.
    """
    if obj.type == "CURVE":
        import json
        key = f"_pin_{vg.name}"
        raw = obj.get(key)
        if raw:
            return json.loads(raw)
        return []
    indices = []
    if not hasattr(obj.data, "vertices"):
        return indices
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                indices.append(v.index)
                break
    return indices


def pin_covers_all_vertices(obj, vg_name) -> bool:
    """True when the pin's vertex group includes EVERY vertex of the mesh.

    "Track Rest-Pose Deformation" requires such a full pin: with every vertex
    captured, the rest pose IS the captured deformation (the solver drives all
    sim vertices), so no partial-pin reconstruction is needed. A partial pin
    would leave the unpinned region at the undeformed rest and tear the
    boundary, so the feature is gated off for it. Mirrors the decoder's
    ``full_pin`` test (``len(pinned) == n_blender``).

    Implemented as a plain count match: the group's member count equals the
    mesh vertex count. Blender exposes no O(1) vertex-group count, so this scans
    the mesh (O(n)). The panel does not call this every redraw; the Refresh
    button next to the rest-pose toggle runs it on demand and caches the result
    on the pin (full_pin_checked / full_pin_cached). The encoder still calls it
    directly at encode time as the source-of-truth gate.
    """
    if obj is None or getattr(obj, "type", None) != "MESH" or not vg_name:
        return False
    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "vertices"):
        return False
    vg = obj.vertex_groups.get(vg_name)
    if vg is None:
        return False
    n_total = len(data.vertices)
    return n_total > 0 and len(get_vertices_in_group(obj, vg)) == n_total


def set_linear_interpolation(action):
    """Set LINEAR interpolation on all keyframe points in *action*.

    Args:
        action: Blender action containing fcurves.
    """
    for fc in _get_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def get_moving_vertex_indices(obj, exclude=None) -> list[int]:
    from .pc2 import has_mesh_cache

    if exclude is None:
        exclude = []
    # MESH_CACHE modifier means all vertices are animated
    if obj and obj.type == "MESH" and has_mesh_cache(obj):
        return [i for i in range(len(obj.data.vertices)) if i not in exclude]
    return []


def get_pin_vertex_indices(obj, context, frame: int | None = None) -> list[int]:
    """List vertex indices that are pinned (active) at the given frame.

    Args:
        obj: Blender mesh object.
        context: Blender context.
        frame: Current frame number. If given, pins with duration that have
            expired by this frame are excluded (their vertices are no longer
            considered pinned). If ``None``, all pin vertices are returned
            regardless of duration.
    """
    indices = set()
    if obj and hasattr(obj, "vertex_groups") and hasattr(obj.data, "vertices"):
        pin_vg_names = set()

        from .uuid_registry import get_object_uuid
        _obj_uid = get_object_uuid(obj)
        for group in iterate_active_object_groups(context.scene):
            if hasattr(group, "pin_vertex_groups"):
                from .uuid_registry import resolve_pin
                for pin_item in group.pin_vertex_groups:
                    resolve_pin(pin_item)
                    if pin_item.object_uuid != _obj_uid:
                        continue
                    _, vg_name = decode_vertex_group_identifier(pin_item.name)
                    if vg_name:
                        # Pull pins are not hard-pinned — exclude them
                        if pin_item.use_pull:
                            continue
                        # Pins with explicit operations (spin/scale/move_by)
                        # move during simulation — exclude them
                        if any(op.op_type in ("SPIN", "SCALE", "MOVE_BY", "TORQUE") for op in pin_item.operations):
                            continue
                        # If frame is given, skip expired duration-limited pins
                        if frame is not None and pin_item.use_pin_duration:
                            if frame > pin_item.pin_duration:
                                continue
                        pin_vg_names.add(vg_name)

        for vg_name in pin_vg_names:
            vg = obj.vertex_groups.get(vg_name)
            if vg:
                for idx in get_vertices_in_group(obj, vg):
                    indices.add(idx)

    return list(indices)


def get_transform_keyframes(obj, context, fps: float) -> dict | None:
    """Extract sparse object-level transform keyframes for a STATIC object.

    Only extracts keyframes from object-level animation (location, rotation, scale).
    Raises RuntimeError if the object has mesh-level-only animation (shape keys).

    Args:
        obj: The Blender object.
        context: The Blender context.
        fps: Frames per second for time conversion.

    Returns:
        dict with keys "time", "translation", "quaternion", "scale", "segments",
        or None if no animation.
    """
    if not obj or not hasattr(obj, "animation_data"):
        return None

    has_mesh_anim = (
        obj.data
        and hasattr(obj.data, "animation_data")
        and obj.data.animation_data
        and obj.data.animation_data.action
        and any(_get_fcurves(obj.data.animation_data.action))
    )
    if has_mesh_anim:
        raise RuntimeError(
            f"STATIC object '{obj.name}' has mesh-level animation (shape keys). "
            "Only object-level transform animation is supported for STATIC objects."
        )

    if not obj.animation_data or not obj.animation_data.action:
        return None

    fcurves = _get_fcurves(obj.animation_data.action)
    if not fcurves:
        return None

    keyframe_frames = set()
    for fc in fcurves:
        keyframe_frames.update(int(kp.co[0]) for kp in fc.keyframe_points)
    if not keyframe_frames:
        return None

    sorted_frames = sorted(keyframe_frames)
    scene = context.scene
    current_frame = scene.frame_current

    times = []
    translations = []
    quaternions = []
    scales = []
    # Per-segment interpolation between keyframe[i] and keyframe[i+1].
    # Bezier handles are normalized to the segment's [0,1] time range.
    segments = []

    for frame in sorted_frames:
        scene.frame_set(frame)
        mat = world_matrix(obj)
        loc, quat, scale = mat.decompose()
        times.append((frame - 1) / fps)
        translations.append([float(loc.x), float(loc.y), float(loc.z)])
        quaternions.append([float(quat.w), float(quat.x), float(quat.y), float(quat.z)])
        scales.append([float(scale.x), float(scale.y), float(scale.z)])

    # Extract interpolation info from fcurves (use location X as representative)
    loc_fc = None
    for fc in fcurves:
        if "location" in fc.data_path:
            loc_fc = fc
            break
    if loc_fc is None:
        loc_fc = fcurves[0]

    kp_by_frame = {int(kp.co[0]): kp for kp in loc_fc.keyframe_points}
    for i in range(len(sorted_frames) - 1):
        f0 = sorted_frames[i]
        f1 = sorted_frames[i + 1]
        kp0 = kp_by_frame.get(f0)
        kp1 = kp_by_frame.get(f1)
        interp = "LINEAR"
        handle_right = [1.0 / 3.0, 0.0]
        handle_left = [2.0 / 3.0, 1.0]
        if kp0 is not None:
            interp = kp0.interpolation
            if interp == "BEZIER" and kp1 is not None:
                dt = f1 - f0
                dv = kp1.co[1] - kp0.co[1]
                hr = kp0.handle_right
                hl = kp1.handle_left
                hr_x = float((hr[0] - f0) / dt) if dt > 0 else 1.0 / 3.0
                hl_x = float((hl[0] - f0) / dt) if dt > 0 else 2.0 / 3.0
                if abs(dv) > 1e-10:
                    hr_y = float((hr[1] - kp0.co[1]) / dv)
                    hl_y = float((hl[1] - kp0.co[1]) / dv)
                else:
                    hr_y = 0.0
                    hl_y = 1.0
                handle_right = [hr_x, hr_y]
                handle_left = [hl_x, hl_y]
        segments.append({
            "interpolation": interp,
            "handle_right": handle_right,
            "handle_left": handle_left,
        })

    scene.frame_set(current_frame)

    return {
        "time": times,
        "translation": translations,
        "quaternion": quaternions,
        "scale": scales,
        "segments": segments,
    }
