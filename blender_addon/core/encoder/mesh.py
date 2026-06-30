# File: encoder/mesh.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import hashlib
import json
from contextlib import contextmanager

import numpy as np
from mathutils import Vector  # pyright: ignore

from ...models.groups import decode_vertex_group_identifier, get_addon_data, iterate_object_groups
from ..utils import (
    eval_deform_local_positions,
    get_transform_keyframes,
    get_vertices_in_group,
    has_deforming_modifier_stack,
    is_deforming_static_object,
    world_matrix,
)
from . import _swap_axes, _to_solver, resolve_fps

# Minimal diagonal used for a group with no usable extent. Both the
# empty/no-geometry fallback and the co-located-objects floor use this
# single value so a sizeless group always yields the same minimal
# diagonal, keeping the derived contact_gap/contact_offset consistent.
# 1e-6 keeps co-located objects from producing a zero ratio that
# cascades into a NaN contact_gap.
MIN_GROUP_DIAGONAL = 1e-6


@contextmanager
def evaluate_at_frame_one(context):
    """Temporarily move the timeline to frame 1, then restore it.

    Geometry-derived encodings must not depend on where the artist parked
    the playhead. The data payload (vertex buffers) and the param payload
    (the per-group bounding-box diagonal that scales contact-gap /
    contact-offset) both read live mesh state, which for an animated
    collider changes frame to frame. Evaluating them at frame 1 is the
    single source of truth shared by the upload and the later drift check,
    so ``compute_data_hash`` / ``compute_param_hash`` reproduce exactly the
    fingerprint the server stored at upload. Without it, scrubbing the
    timeline silently shrinks an animated object's bounding box and the
    param fingerprint drifts, surfacing as a spurious "Parameters have
    changed since the last transfer" on Run / Resume.
    """
    scene = context.scene
    saved = scene.frame_current
    try:
        scene.frame_set(1)
        yield
    finally:
        scene.frame_set(saved)


def _frame_one_eval_local_verts(obj, context):
    """Frame-1 deform-evaluated local verts for *obj*, or ``None`` when
    no usable eval is available (caller keeps the rest mesh).

    Honors a deform-only Geometry Nodes / Armature / Lattice stack so
    the solver starts in the shape the artist sees at frame 1. Pinned
    vertices are included: the decoder integrates a captured pin track
    as deltas from each vertex's INITIAL position, and the captured
    cache's frame-0 row IS the frame-1 deform pose, so initial ==
    cache[0] telescopes to the exact wave (frame-1 + (cache[k] -
    cache[0]) = cache[k]). Holding pinned verts at rest instead would
    shift the whole track by the frame-1 displacement. The addon's
    MESH_CACHE is excluded so a prior solve's output isn't read back.
    """
    from ..pc2 import MODIFIER_NAME
    return eval_deform_local_positions(
        obj, context, exclude_modifier_name=MODIFIER_NAME,
    )


def compute_mesh_hash(context):
    """Compute a fast hash of mesh topology for validation.

    Returns a dictionary with hash per group type containing:
    - vertex_count: total vertices across all objects in group
    - triangle_count: total triangles (for STATIC/SHELL/SOLID)
    - edge_count: total edges (for ROD, or pure edges for others)
    - object_count: number of objects in group
    - pin_groups: dictionary of pin vertex group info {group_name: vertex_count}
    """
    scene = context.scene
    hash_data = {}

    for group in [group for group in iterate_object_groups(scene) if group.active]:
        group_hash = {
            "vertex_count": 0,
            "triangle_count": 0,
            "edge_count": 0,
            "object_count": sum(1 for obj in group.assigned_objects if obj.included),
            "pin_groups": {},
            # Per-object hashes of static-deform PC2 sidecars so a re-
            # capture flips the upload hash even though the rest mesh
            # topology stayed identical. Keyed by object UUID, value
            # is (n_frames, n_verts) plus a sha256 of the cache bytes.
            "static_deform": {},
        }

        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            from ..uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            if obj is None or obj.type not in ("MESH", "CURVE"):
                continue

            if obj.type == "CURVE" and group.object_type == "ROD":
                # Hash based on original curve structure (stable across runs)
                for s in obj.data.splines:
                    n_pts = len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
                    group_hash["vertex_count"] += n_pts
                    group_hash["edge_count"] += max(0, n_pts - 1)
                continue

            mesh = obj.data
            group_hash["vertex_count"] += len(mesh.vertices)

            if group.object_type == "ROD":
                # ROD objects only have edges
                group_hash["edge_count"] += len(mesh.edges)
            else:
                # STATIC, SHELL, SOLID have polygons. Count polygons (not
                # loop triangles) so this fingerprint stays stable across
                # the loop_triangles encoder switch: it only needs to flip
                # when topology changes, and a polygon count already does
                # that (it is also more sensitive than a triangle count to
                # a quad->two-tris edit, which leaves the triangle count
                # unchanged). Avoids a spurious one-time "topology changed"
                # warning for every existing quad/N-gon scene on upgrade.
                group_hash["triangle_count"] += len(mesh.polygons)
                # Also count pure edges (stitch edges)
                stitch_data = detect_stitch_edges(mesh)
                if stitch_data is not None:
                    group_hash["edge_count"] += len(stitch_data[0])

            # Static-deform PC2 fingerprint (STATIC group only). Hash
            # the in-memory cache bytes when present so re-capturing
            # the modifier stack changes the upload hash; without
            # this, the encoder would happily ship the OLD cache after
            # an Armature tweak because the rest mesh is unchanged.
            if group.object_type == "STATIC" and obj.type == "MESH":
                from ..pc2 import get_static_deform_cache
                cache = get_static_deform_cache(obj)
                if cache is not None:
                    from ..uuid_registry import get_or_create_object_uuid
                    obj_uuid = get_or_create_object_uuid(obj)
                    digest = hashlib.sha256(cache.tobytes()).hexdigest()
                    group_hash["static_deform"][obj_uuid] = {
                        "n_frames": int(cache.shape[0]),
                        "n_verts": int(cache.shape[1]),
                        "sha256": digest,
                    }

        # Include pin vertex group information
        from ..uuid_registry import resolve_pin as _resolve_pin
        for pin_item in group.pin_vertex_groups:
            obj = _resolve_pin(pin_item)
            if obj is None or obj.type != "MESH":
                continue
            _, vg_name = decode_vertex_group_identifier(pin_item.name)
            if not vg_name:
                continue
            vg = obj.vertex_groups.get(vg_name)
            if vg:
                # Count vertices in this vertex group.
                # Key on (object_uuid, vg_hash) so the mesh-hash is stable
                # across object/vertex-group renames.
                vg_vertex_count = len(get_vertices_in_group(obj, vg))
                pin_key = f"{pin_item.object_uuid}:{pin_item.vg_hash}"
                group_hash["pin_groups"][pin_key] = vg_vertex_count

        # Create a simple hash key using group type and UUID
        group_key = f"{group.object_type}_{group.ensure_uuid()}"
        hash_data[group_key] = group_hash

    return hash_data


def compute_group_bounding_box_diagonal(group):
    """Compute the maximal diagonal length of the bounding box that encapsulates all objects in a group."""
    min_coord = None
    max_coord = None

    has_valid_object = False

    from ..uuid_registry import resolve_assigned
    for assigned_obj in group.assigned_objects:
        if not assigned_obj.included:
            continue
        obj = resolve_assigned(assigned_obj)
        if not obj or obj.type not in ("MESH", "CURVE"):
            continue

        world_mat = obj.matrix_world

        if obj.type == "CURVE":
            # Use Blender's bound_box which works for all object types
            for corner in obj.bound_box:
                world_pos = world_mat @ Vector(corner)
                if min_coord is None:
                    min_coord = world_pos.copy()
                    max_coord = world_pos.copy()
                else:
                    for i in range(3):
                        min_coord[i] = min(min_coord[i], world_pos[i])
                        max_coord[i] = max(max_coord[i], world_pos[i])
                has_valid_object = True
        elif obj.type == "MESH":
            mesh = obj.data

            for vert in mesh.vertices:
                world_pos = world_mat @ vert.co

                if min_coord is None:
                    min_coord = world_pos.copy()
                    max_coord = world_pos.copy()
                    has_valid_object = True
                else:
                    min_coord.x = min(min_coord.x, world_pos.x)
                    min_coord.y = min(min_coord.y, world_pos.y)
                    min_coord.z = min(min_coord.z, world_pos.z)
                    max_coord.x = max(max_coord.x, world_pos.x)
                    max_coord.y = max(max_coord.y, world_pos.y)
                    max_coord.z = max(max_coord.z, world_pos.z)

    if not has_valid_object:
        return MIN_GROUP_DIAGONAL

    diagonal_vector = max_coord - min_coord
    return max(diagonal_vector.length, MIN_GROUP_DIAGONAL)


def detect_stitch_edges(mesh):
    """Detect stitch edges (pure edge elements not connected to any triangles).

    Args:
        mesh: Blender mesh object

    Returns:
        tuple: (Ind, W) where Ind is indices array (#x3) and W is weights array (#x2)
               Returns None if no stitch edges are found
    """
    n_edges = len(mesh.edges)
    if n_edges == 0:
        return None

    # A stitch (sewing) edge is a pure edge element used by no face. Every
    # face-boundary edge is referenced by at least one mesh loop via its
    # ``edge_index``; an edge no loop references is therefore loose. Reading
    # the loop->edge map and the edge vertices through ``foreach_get`` keeps
    # this O(loops + edges) in C instead of a per-edge / per-poly Python scan
    # (the prior set-based form dominated ``compute_mesh_hash`` on scenes
    # with many objects). The result is byte-identical: edges stay in
    # ascending index order and each pair is sorted ascending, matching the
    # old ``tuple(sorted(...))`` per edge.
    used = np.zeros(n_edges, dtype=bool)
    n_loops = len(mesh.loops)
    if n_loops:
        loop_edges = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("edge_index", loop_edges)
        used[loop_edges] = True

    stitch_idx = np.nonzero(~used)[0]
    if stitch_idx.size == 0:
        return None

    edge_verts = np.empty(n_edges * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_verts)
    edge_verts = edge_verts.reshape(n_edges, 2)
    se = np.sort(edge_verts[stitch_idx], axis=1)  # each pair ascending

    n = se.shape[0]
    Ind = np.empty((n, 4), dtype=np.uint32)
    Ind[:, 0] = se[:, 0]
    Ind[:, 1] = se[:, 1]
    Ind[:, 2] = se[:, 1]
    Ind[:, 3] = se[:, 1]
    W = np.tile(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    return (Ind, W)


def detect_hanging_stitch_vertices(mesh, pinned=None):
    """Return vertices that sit midway along a stitch (loose) edge with no face.

    A stitch edge is a pure edge element (an edge shared by no polygon);
    these act as sewing seams (see ``detect_stitch_edges``). A normal seam
    runs between two boundary vertices that are themselves corners of cloth
    triangles, so both endpoints pick up surface mass. Subdividing a seam
    (or otherwise leaving a vertex partway along it) inserts a vertex that
    touches only loose edges and belongs to no face.

    The solver aggregates vertex mass from faces, rod edges and tets only;
    a stitch edge contributes none. A face-less stitch vertex therefore ends
    up with zero mass, so its momentum (inertia) Hessian block is zero and
    the linear solve goes singular. The simulation then aborts before the
    first frame with no error, which is exactly the silent failure reported
    for subdivided sewing lines (dissolving the midway vertices fixes it).

    Pinned vertices are exempt. A pin is a Dirichlet condition: the solver
    prescribes the vertex's position, so it carries no free momentum DOF and
    cannot make the system singular even with zero face mass. A curtain that
    hangs by sewing a row of pinned hook vertices to its top edge is the
    canonical valid case (each hook is a face-less stitch vertex, yet fixed).

    Args:
        mesh: Blender mesh data block.
        pinned: optional iterable of vertex indices held fixed by a pin; these
            are excluded from the result.

    Returns:
        list[int]: sorted vertex indices that lie on a stitch edge but on no
        face and on no pin. Empty when the mesh has no such hanging seam
        vertex.
    """
    pinned = set() if pinned is None else set(pinned)
    verts_in_faces = set()
    edges_in_faces = set()
    for poly in mesh.polygons:
        verts = poly.vertices
        n = len(verts)
        for i in range(n):
            v1 = verts[i]
            v2 = verts[(i + 1) % n]
            verts_in_faces.add(v1)
            edges_in_faces.add(tuple(sorted((v1, v2))))

    hanging = set()
    for edge in mesh.edges:
        v1, v2 = edge.vertices[0], edge.vertices[1]
        if tuple(sorted((v1, v2))) in edges_in_faces:
            continue
        if v1 not in verts_in_faces and v1 not in pinned:
            hanging.add(v1)
        if v2 not in verts_in_faces and v2 not in pinned:
            hanging.add(v2)
    return sorted(hanging)


def _group_pinned_vertex_indices(group, obj):
    """Vertex indices of ``obj`` held fixed by any of ``group``'s pin groups.

    Resolves each pin item to its object (syncing object/vertex-group
    renames) and, when it targets ``obj``, collects the pinned vertices.
    Used to exempt pinned vertices from the hanging-stitch rejection: a
    pinned vertex is a fixed Dirichlet point and never goes singular.
    """
    from ...models.groups import decode_vertex_group_identifier
    from ..uuid_registry import resolve_pin
    from ..utils import get_vertices_in_group

    pinned = set()
    for pin_item in group.pin_vertex_groups:
        # A malformed pin (e.g. empty object_uuid) makes resolve_pin raise;
        # skip it here rather than fail the obj-encode path. The genuine
        # pin-encode path still surfaces the error.
        try:
            pin_obj = resolve_pin(pin_item)
        except Exception:
            continue
        if pin_obj is None or pin_obj.name != obj.name:
            continue
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if not vg_name:
            continue
        vg = obj.vertex_groups.get(vg_name)
        if vg is None:
            continue
        pinned.update(get_vertices_in_group(obj, vg))
    return pinned


def detect_isolated_vertices(mesh, pinned=None):
    """Return vertices present in the mesh but in no triangle (in no face).

    The solver derives a collider vertex's contact parameters by averaging
    over its incident faces (and a dynamic vertex's mass likewise comes from
    faces). A vertex that belongs to no face contributes nothing to that
    average: for a STATIC collider the area-weighted average divides by zero
    and the solver aborts the build; for a dynamic mesh the vertex is
    silently dropped. These are stray points left in a model (e.g. detached
    vertices in an imported car body) and should be removed before transfer.

    Pinned vertices are exempt: a pin holds the vertex at a fixed position,
    so a faceless pinned point (e.g. a sewn curtain hook) is valid, not
    stray. The detection keys on ``loop_triangles`` (the same triangulation
    the encoder ships), so it matches exactly the per-face data the solver
    builds on; a vertex on a loose edge but in no face is the separate
    hanging-seam case handled by ``detect_hanging_stitch_vertices``.

    Args:
        mesh: Blender mesh data block.
        pinned: optional iterable of vertex indices held fixed by a pin;
            these are excluded from the result.

    Returns:
        list[int]: sorted vertex indices that lie in no triangle and on no
        pin. Empty when the mesh has no stray vertex.
    """
    from ..numpy_mesh_utils import loop_triangle_indices

    tri = loop_triangle_indices(mesh)
    used = np.unique(tri)
    isolated = np.setdiff1d(
        np.arange(len(mesh.vertices), dtype=used.dtype), used
    )
    pinned = set() if pinned is None else set(pinned)
    return sorted(int(i) for i in isolated if int(i) not in pinned)


def _local_mesh_hash(obj):
    """Compute a content hash of an object's local-space triangulated mesh.

    Two non-linked duplicates with identical geometry (same vertices + the
    same loop-triangle tessellation in local space) produce the same hash,
    enabling deduplication. The triangulation is Blender's own
    ``loop_triangles`` split, matching the ``face`` array the encoder ships,
    so the hash keys on exactly the topology that reaches the solver.
    """
    from ..numpy_mesh_utils import extract_mesh_to_numpy, loop_triangle_indices

    mesh = obj.data
    vertices, _ = extract_mesh_to_numpy(mesh)
    tri = loop_triangle_indices(mesh)
    h = hashlib.sha256()
    h.update(vertices.tobytes())
    h.update(tri.tobytes())
    return h.hexdigest()


def _encode_bend_reference_verts(group, assigned, source_obj, context):
    """Local-space reference vertices for a shell or rod object's bending
    rest angle, or ``None`` when the object has no enabled reference.

    Re-validates topology at encode time and raises ``ValueError`` on any
    mismatch (missing reference, vertex-count or connectivity deviation)
    so a bad reference fails the upload with a clear, actionable message
    instead of silently shipping a misaligned vertex buffer. The returned
    array is in the same vertex order as the object's shipped ``vert``,
    which ``validate_bend_reference`` has confirmed matches one-to-one:

    * SHELL / mesh ROD: the reference's modifier-evaluated local verts.
    * Curve ROD: the reference curve sampled the same way the source curve
      rod is sampled (control-point level).
    """
    if group.object_type not in ("SHELL", "ROD"):
        return None
    if not group.bend_rest_from_reference:
        return None
    if not assigned.bend_ref_enable or not assigned.bend_ref_uuid:
        return None

    from ..uuid_registry import get_object_by_uuid
    from ..utils import eval_reference_local_positions, validate_bend_reference

    ref_obj = get_object_by_uuid(assigned.bend_ref_uuid)
    if ref_obj is None:
        raise ValueError(
            f"Object '{source_obj.name}' in group '{group.name}' has "
            f"Reference Rest Angle enabled but its reference object "
            f"('{assigned.bend_ref_name or assigned.bend_ref_uuid}') is "
            f"missing. Re-pick the reference or turn off Reference Rest Angle."
        )
    ok, msg = validate_bend_reference(source_obj, ref_obj, context, group.object_type)
    if not ok:
        raise ValueError(
            f"Reference Rest Angle for '{source_obj.name}' in group "
            f"'{group.name}': {msg}"
        )

    # Curve rod: sample the reference curve (control-point level), matching
    # how the source curve rod is shipped.
    if group.object_type == "ROD" and ref_obj.type == "CURVE":
        from mathutils import Matrix  # pyright: ignore
        from ..curve_rod import sample_curve
        ref_local, _, _ = sample_curve(ref_obj, Matrix.Identity(4))
        return np.ascontiguousarray(ref_local, dtype=np.float32)

    # Mesh shell or mesh rod: full modifier-evaluated local vertices.
    ref_local = eval_reference_local_positions(ref_obj, context)
    if ref_local is None:
        raise ValueError(
            f"Could not evaluate reference object '{ref_obj.name}' for "
            f"'{source_obj.name}' in group '{group.name}'."
        )
    return np.ascontiguousarray(ref_local, dtype=np.float32)


def _build_obj_data(context, *, persist_topology_hash: bool) -> list:
    """Construct the list-of-group dicts that ``encode_obj`` serializes to CBOR.

    Factored so ``compute_data_hash`` can fingerprint the same source
    of truth without committing the upload-time side effect of
    stamping ``state.mesh_hash_json``. ``persist_topology_hash``
    decides whether the topology summary should land on disk via
    ``state.set_mesh_hash``: ``True`` for upload encoding (so a later
    run can warn on stale topology), ``False`` for hash-only callers.
    """
    scene = context.scene
    state = get_addon_data(scene).state

    # Check for duplicate object names across active groups, and for
    # Linked Duplicates (Alt-D / shallow copies) sneaking in via paths
    # that bypass ``OBJECT_OT_AddObjectsToGroup`` (older saves, MCP
    # mutations, scripts directly mutating ``assigned_objects``). Both
    # are upload-fatal: the first lets a vertex be owned by two
    # groups, the second lets two different assigned objects scribble
    # over each other's mesh data because Blender shares the data
    # block.
    from ..uuid_registry import resolve_assigned
    from ..utils import (
        count_duplicate_faces,
        find_linked_duplicate_siblings,
    )
    seen = {}
    for group in iterate_object_groups(scene):
        if not group.active:
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            uid = assigned.uuid
            if uid in seen:
                raise ValueError(
                    f"Object '{assigned.name}' is assigned to both "
                    f"'{seen[uid]}' and '{group.name}' groups. "
                    f"Each object can only belong to one group."
                )
            seen[uid] = group.name
            if obj is not None:
                siblings = find_linked_duplicate_siblings(obj)
                if siblings:
                    raise ValueError(
                        f"Object '{obj.name}' in group '{group.name}' is a "
                        f"Linked Duplicate (shares mesh data with "
                        f"{siblings[0]!r}). Make it single-user "
                        f"(Object > Relations > Make Single User > Object "
                        f"& Data) before transferring."
                    )
                # SAND bodies are a faceless mesh of loose grain-center
                # vertices, so the duplicate-face / hanging-stitch surface
                # validators do not apply.
                dup_count = (
                    0 if group.object_type == "SAND"
                    else count_duplicate_faces(obj)
                )
                if dup_count > 0:
                    raise ValueError(
                        f"Object '{obj.name}' in group '{group.name}' has "
                        f"{dup_count} duplicate face(s) (two triangles that "
                        f"share the same three vertices). The solver builds "
                        f"a degenerate bending element from coincident "
                        f"faces and aborts. This usually means doubled "
                        f"geometry: select the mesh in Edit Mode and run "
                        f"Mesh > Merge > By Distance (or delete the "
                        f"duplicate faces) before transferring."
                    )
                if obj.type == "MESH" and group.object_type in {"SHELL", "SOLID"}:
                    pinned = _group_pinned_vertex_indices(group, obj)
                    hanging = detect_hanging_stitch_vertices(obj.data, pinned)
                    if hanging:
                        preview = ", ".join(str(i) for i in hanging[:8])
                        if len(hanging) > 8:
                            preview += ", ..."
                        raise ValueError(
                            f"Object '{obj.name}' in group '{group.name}' has "
                            f"{len(hanging)} vertex(es) sitting midway along a "
                            f"sewing (loose) edge with no connected face "
                            f"(vertex index {preview}). A sewing seam must run "
                            f"between two face-connected vertices; a vertex "
                            f"left midway along it carries no surface mass, so "
                            f"the solver builds a singular system and silently "
                            f"aborts before the first frame with no error. "
                            f"Select the mesh in Edit Mode, select those "
                            f"vertices, and run Mesh > Dissolve > Dissolve "
                            f"Vertices (or Limited Dissolve) so each seam is a "
                            f"single edge between two face-connected vertices "
                            f"before transferring."
                        )
                if obj.type == "MESH" and group.object_type == "STATIC":
                    # A STATIC collider with stray vertices (in no triangle)
                    # crashes the solver's collision-mesh build, which averages
                    # each collider vertex's contact parameters over its faces
                    # and aborts when a vertex has none. STATIC colliders are
                    # never pinned, so no pin exemption. (The dynamic SHELL/
                    # SOLID path tolerates a faceless vertex by dropping it
                    # rather than crashing; promoting that to an error too is a
                    # possible follow-up.) The "isolated vert" wording is the
                    # sentinel the Blender panel matches to offer the "Remove
                    # Isolated Vertices" button.
                    isolated = detect_isolated_vertices(obj.data)
                    if isolated:
                        preview = ", ".join(str(i) for i in isolated[:8])
                        if len(isolated) > 8:
                            preview += ", ..."
                        raise ValueError(
                            f"Object '{obj.name}' in group '{group.name}' has "
                            f"{len(isolated)} isolated vertex(es) (present in "
                            f"the mesh but in no triangle/face) at vertex index "
                            f"{preview}. The solver averages each vertex's "
                            f"contact parameters over its connected faces and "
                            f"aborts the build when a vertex has none, so stray "
                            f"points left in an imported model must be removed. "
                            f"Use the \"Remove Isolated Vertices\" button below, "
                            f"or select the mesh in Edit Mode and run Select > "
                            f"All by Trait > Loose Geometry, then Mesh > Delete "
                            f"> Loose, before transferring."
                        )

    # Compute and (optionally) store mesh topology summary.
    mesh_hash = compute_mesh_hash(context)
    if persist_topology_hash:
        state.set_mesh_hash(mesh_hash)

    data = []
    with evaluate_at_frame_one(context):
        return _encode_obj_inner(context, scene, state, data)


def _encode_obj_inner(context, scene, state, data):
    fps = resolve_fps(state)

    # Track canonical meshes for deduplication: hash -> first object name
    canonical_meshes = {}

    for group in [group for group in iterate_object_groups(scene) if group.active]:
        objects_info = []
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            from ..uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            assert obj is not None, f"Object {assigned.name} not found."
            mat = world_matrix(obj)

            # For CURVE objects in ROD groups, sample in local space.
            # The "transform" field (full world_matrix) is sent separately
            # and applied by the decoder, matching mesh encoding behavior.
            if obj.type == "CURVE" and group.object_type == "ROD":
                from mathutils import Matrix  # pyright: ignore
                from ..curve_rod import sample_curve
                vert, edges_array, _ = sample_curve(obj, Matrix.Identity(4))
                stitch_data = None
                uv = []
                tri = None
                mesh_ref = None
            else:
                mesh = obj.data
                edges_array = None
                tri = None
                mesh_ref = None

                if group.object_type == "ROD":
                    # ROD objects use edges only - no triangulation needed.
                    # Encode in local space; the decoder applies "transform"
                    # on top, matching the CURVE+ROD and SHELL/SOLID paths.
                    local_verts = [v.co.copy() for v in mesh.vertices]
                    vert = np.array(local_verts, dtype=np.float32)
                    edges = []
                    for edge in mesh.edges:
                        edges.append(list(edge.vertices))
                    edges_array = np.array(edges, dtype=np.uint32)
                    stitch_data = None
                    uv = []
                elif group.object_type == "SAND":
                    # SAND objects are a faceless mesh of loose vertices:
                    # each vertex is one grain center. Ship local-space
                    # positions via the fast foreach_get path (no per-vertex
                    # list comp) and emit no edges/faces; the decoder applies
                    # "transform" on top, like ROD and SHELL/SOLID.
                    n_v = len(mesh.vertices)
                    flat = np.empty(n_v * 3, dtype=np.float32)
                    mesh.vertices.foreach_get("co", flat)
                    vert = flat.reshape(n_v, 3)
                    uv = []
                    stitch_data = None
                else:
                    # STATIC objects with a static-deform cache always
                    # send canonical (no mesh-ref dedup): each one
                    # carries its own per-frame vertex buffer, and a
                    # shared canonical entry would conflate them.
                    from ..pc2 import has_static_deform_animation
                    _has_sd = (
                        group.object_type == "STATIC"
                        and obj.type == "MESH"
                        and has_static_deform_animation(obj)
                    )

                    # A deform-only modifier stack (Geometry Nodes,
                    # Armature, ...) makes the frame-1 shape diverge from
                    # the rest cage, so two objects that share a rest mesh
                    # can no longer share one canonical buffer. Exclude
                    # them from dedup, same as the static-deform case.
                    _has_deform = has_deforming_modifier_stack(obj)

                    # A shell object with an enabled bending reference ships
                    # a per-object reference vertex buffer that must line up
                    # 1:1 with its own `vert` order, so it can't share a
                    # canonical mesh with a sibling. Force it canonical, same
                    # as the static-deform and deform-stack cases.
                    _has_bend_ref = (
                        group.object_type == "SHELL"
                        and group.bend_rest_from_reference
                        and assigned.bend_ref_enable
                        and bool(assigned.bend_ref_uuid)
                    )
                    _dedup_ok = (
                        not _has_sd and not _has_deform and not _has_bend_ref
                    )

                    # Deduplication: check if this mesh is a duplicate
                    content_hash = _local_mesh_hash(obj)
                    if content_hash in canonical_meshes and _dedup_ok:
                        # Duplicate: reference the canonical mesh, send only transform
                        mesh_ref = canonical_meshes[content_hash]
                        vert = None
                        tri = None
                        uv = []
                        stitch_data = None
                    else:
                        # Canonical: extract local-space mesh + transform
                        from ..uuid_registry import get_or_create_object_uuid
                        if _dedup_ok:
                            canonical_meshes[content_hash] = get_or_create_object_uuid(obj)
                        from ..numpy_mesh_utils import (
                            extract_mesh_to_numpy,
                            loop_triangulate_mesh,
                        )
                        local_verts, _ = extract_mesh_to_numpy(mesh)
                        # Triangulation and UVs come off the rest topology
                        # via Blender's loop_triangles, so quads and N-gons
                        # split exactly the way the viewport shows them
                        # (stable vertex indices). The positions sent to the
                        # solver honor the frame-1 deform when present,
                        # except on pin-driven verts (kept at rest so the
                        # pin delta track isn't double-counted).
                        tri, uv = loop_triangulate_mesh(mesh)
                        eval_verts = _frame_one_eval_local_verts(obj, context)
                        vert = eval_verts if eval_verts is not None else local_verts
                        stitch_data = detect_stitch_edges(mesh)

            info = {}
            if group.object_type == "STATIC":
                # Static-deform cache wins over fcurves and static_ops:
                # the depsgraph already composes parent/object fcurves
                # and modifier deformation into the cache, so the rigid
                # T/R/S path would double-count if both were emitted.
                from ..pc2 import (
                    get_static_deform_cache,
                    has_static_deform_animation,
                )
                _has_sd_cache = (
                    obj.type == "MESH" and has_static_deform_animation(obj)
                )
                if _has_sd_cache:
                    cache = get_static_deform_cache(obj)
                    if cache is None or cache.shape[1] == 0 or cache.shape[0] == 0:
                        raise ValueError(
                            f"STATIC object '{obj.name}' in group "
                            f"'{group.name}' has an empty deformation cache; "
                            "re-run 'Capture Deformation' on this row."
                        )
                    if cache.shape[1] != len(mesh.vertices):
                        raise ValueError(
                            f"STATIC object '{obj.name}' in group "
                            f"'{group.name}': deformation cache has "
                            f"{cache.shape[1]} vertices but the mesh now "
                            f"has {len(mesh.vertices)}. Edit since the "
                            "last capture changed the topology; click "
                            "'Clear Cache' and re-run 'Capture Deformation'."
                        )
                    n_frames = int(cache.shape[0])
                    info["static_deform_animation"] = {
                        "time": [i / fps for i in range(n_frames)],
                        "vert_frames": np.ascontiguousarray(
                            cache, dtype=np.float32,
                        ),
                    }
                else:
                    # No cache: refuse to silently lose the deformation.
                    # Matches the linked-duplicate / duplicate-face
                    # validators in _build_obj_data: hard fail with a
                    # next-action message.
                    if (
                        obj.type == "MESH"
                        and is_deforming_static_object(obj, context)
                    ):
                        raise ValueError(
                            f"STATIC object '{obj.name}' in group "
                            f"'{group.name}' has a deforming modifier "
                            "stack (or shape-key animation) but no "
                            "deformation cache. Click 'Capture "
                            "Deformation' on the assigned-object row to "
                            "record the per-frame mesh, or remove the "
                            "deformer if the motion shouldn't reach the "
                            "solver."
                        )

                transform_kf = (
                    None if _has_sd_cache
                    else get_transform_keyframes(obj, context, fps)
                )
                if transform_kf is not None:
                    info["transform_animation"] = transform_kf
                # UI-assigned static ops: find the AssignedObject entry
                # for this obj and serialize its ops list. Mutually
                # exclusive with Blender fcurve animation AND the new
                # static-deform cache (3-way XOR enforced by the Rust
                # validator at decode time).
                from ..uuid_registry import get_or_create_object_uuid as _get_uuid_static
                _obj_uuid_static = _get_uuid_static(obj)
                _assigned_static = None
                for _a in group.assigned_objects:
                    if _a.uuid == _obj_uuid_static:
                        _assigned_static = _a
                        break
                if (
                    _assigned_static is not None
                    and len(_assigned_static.static_ops) > 0
                    and transform_kf is None
                    and not _has_sd_cache
                ):
                    # Skip if transform_kf is already set OR a deform
                    # cache exists, both take precedence; the UI warns
                    # the user that ops will be ignored.
                    ops_out = []
                    for op in _assigned_static.static_ops:
                        t_start = (op.frame_start - 1) / fps
                        t_end = (op.frame_end - 1) / fps
                        entry = {
                            "op_type": op.op_type,
                            "t_start": float(t_start),
                            "t_end": float(t_end),
                            "transition": str(op.transition).lower(),
                        }
                        # Spin/scale always pivot around the object origin,
                        # which in the pin-shell's local op frame is (0,0,0),
                        # so the decoder sends that as an absolute center.
                        if op.op_type == "MOVE_BY":
                            entry["delta"] = _to_solver(op.delta)
                        elif op.op_type == "SPIN":
                            entry["axis"] = _swap_axes(op.spin_axis)
                            entry["angular_velocity"] = float(op.spin_angular_velocity)
                        elif op.op_type == "SCALE":
                            entry["factor"] = float(op.scale_factor)
                        ops_out.append(entry)
                    info["static_ops"] = ops_out
            else:
                pin_indices = []
                from ..uuid_registry import resolve_pin, get_or_create_object_uuid as _get_uuid
                obj_uuid = _get_uuid(obj)
                for pin_item in group.pin_vertex_groups:
                    resolve_pin(pin_item)
                    if pin_item.object_uuid != obj_uuid:
                        continue
                    _, vg_name = decode_vertex_group_identifier(pin_item.name)
                    if vg_name:
                        if obj.type == "CURVE":
                            key = f"_pin_{vg_name}"
                            raw = obj.get(key)
                            if raw:
                                pin_indices.extend(json.loads(raw))
                        else:
                            vg = obj.vertex_groups.get(vg_name)
                            if vg:
                                pin_indices.extend(get_vertices_in_group(obj, vg))

                if obj.type == "CURVE" and pin_indices:
                    from ..curve_rod import map_cp_pins_to_sampled
                    pin_indices = map_cp_pins_to_sampled(obj, pin_indices)

                if len(pin_indices) > 0:
                    info["pin"] = pin_indices

            from ..uuid_registry import get_or_create_object_uuid
            obj_uuid = get_or_create_object_uuid(obj)
            if not obj_uuid:
                raise RuntimeError(
                    f"Object '{obj.name}' has no UUID after migration. "
                    "All objects must have UUIDs before encoding."
                )
            info["name"] = str(obj.name)
            info["uuid"] = obj_uuid

            if mesh_ref is not None:
                # Duplicate: send reference + per-instance transform
                info["mesh_ref"] = mesh_ref
                info["transform"] = np.array(mat, dtype=np.float64).reshape(4, 4)
            elif vert is not None:
                # Canonical: send local-space vertices + transform
                info["vert"] = vert
                info["transform"] = np.array(mat, dtype=np.float64).reshape(4, 4)
                if group.object_type == "ROD":
                    info["edge"] = edges_array
                elif group.object_type == "SAND":
                    # Faceless granular body: just the loose-vert grain
                    # centers plus the per-grain radius. The radius is the
                    # value locked on the object at Convert time (the seeding
                    # spacing was derived from it), so render and contact agree.
                    info["radius"] = float(
                        obj.get("ppf_grain_radius", group.sand_grain_radius)
                    )
                else:
                    info["face"] = tri
                    if len(uv) > 0:
                        info["uv"] = uv
                    if stitch_data is not None:
                        info["stitch"] = stitch_data

            # Per-object bending reference rest angle (SHELL): ship the
            # reference object's evaluated local-space vertices so the
            # solver computes this object's hinge rest angles from the
            # reference shape instead of its own initial pose. The object
            # is forced canonical above, so `vert` is present and shares
            # the reference's vertex order.
            bend_ref_verts = _encode_bend_reference_verts(group, assigned, obj, context)
            if bend_ref_verts is not None:
                assert vert is not None, (
                    "bend-reference object must ship canonical vertices"
                )
                if len(bend_ref_verts) != len(vert):
                    raise ValueError(
                        f"Reference Rest Angle for '{obj.name}' in group "
                        f"'{group.name}': reference has {len(bend_ref_verts)} "
                        f"vertices but the object ships {len(vert)}."
                    )
                info["bend_rest_vert"] = bend_ref_verts

            # Case-3 (static-deform pin shell): the cache is in solver
            # world space, so we override the rest mesh + world matrix
            # so the pin shell starts AT frame_start's depsgraph pose,
            # not the bind-pose mesh. Then per-frame MoveBy deltas
            # (decoder-side) compose into world-space target positions
            # without needing to re-apply the object's world matrix.
            if (
                "static_deform_animation" in info
                and "vert" in info
            ):
                vf = info["static_deform_animation"]["vert_frames"]
                # vf[0] is the rest pose in solver world space. Use it
                # as the pin shell's local_vert and reset the world
                # transform to identity so transform @ local = vf[0].
                info["vert"] = np.ascontiguousarray(vf[0], dtype=np.float32)
                info["transform"] = np.eye(4, dtype=np.float64)

            objects_info.append(info)

        data.append(
            {
                "object": objects_info,
                "type": str(group.object_type),
            }
        )

    return data


def encode_obj(context) -> bytes:
    # Producer emits CBOR with the schema-version envelope from
    # crates/ppf-cts-formats/src/envelope.rs. The on-disk filename
    # stays `data.pickle` for back-compat with existing project layouts.
    from .cbor_encode import dumps_envelope
    return dumps_envelope("Scene", _build_obj_data(context, persist_topology_hash=True))


def encode_obj_with_hash(context) -> tuple[bytes, str]:
    """Encode the scene data tree and hash the encoded bytes in one pass.

    Builds ``_build_obj_data`` once, encodes to CBOR, and computes the
    SHA-256 of the resulting bytes. Cheaper than calling ``encode_obj``
    plus ``compute_data_hash`` because the data tree is built only once
    and the hash runs on the already-produced wire bytes (vs encoding
    the tree to CBOR a second time). The on-wire bytes are deterministic
    because ``_build_obj_data`` produces a fixed dict-insertion order
    and ``cbor2.dumps`` preserves that order.
    """
    from .cbor_encode import dumps_envelope
    tree = _build_obj_data(context, persist_topology_hash=True)
    encoded = dumps_envelope("Scene", tree)
    return encoded, hashlib.sha256(encoded).hexdigest()


def compute_data_hash(context) -> str:
    """SHA-256 fingerprint of the encoded scene data.

    Hashes the same CBOR bytes that ``encode_obj`` produces, so the
    upload-time hash and the click-time drift hash always agree.
    Side-effect-free: does NOT stamp ``state.mesh_hash_json`` (only the
    upload path should do that). Raises ``ValueError`` for the same
    duplicate-assignment / encoding errors ``encode_obj`` raises.
    """
    from .cbor_encode import dumps_envelope
    tree = _build_obj_data(context, persist_topology_hash=False)
    return hashlib.sha256(dumps_envelope("Scene", tree)).hexdigest()
