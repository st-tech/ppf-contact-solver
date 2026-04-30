# File: encoder/mesh.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import hashlib
import json
import pickle

import bpy  # pyright: ignore
import numpy as np
from mathutils import Vector  # pyright: ignore

from ...models.groups import decode_vertex_group_identifier, get_addon_data, iterate_object_groups
from ..utils import get_transform_keyframes, get_vertices_in_group, world_matrix
from . import _swap_axes, _to_solver


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
                # STATIC, SHELL, SOLID have triangles
                group_hash["triangle_count"] += len(mesh.polygons)
                # Also count pure edges (stitch edges)
                stitch_data = detect_stitch_edges(mesh)
                if stitch_data is not None:
                    group_hash["edge_count"] += len(stitch_data[0])

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
        return 0.001

    diagonal_vector = max_coord - min_coord
    # Floor at 1e-6 so co-located objects don't produce a zero ratio
    # that cascades into a NaN contact_gap.
    return max(diagonal_vector.length, 1e-6)


def detect_stitch_edges(mesh):
    """Detect stitch edges (pure edge elements not connected to any triangles).

    Args:
        mesh: Blender mesh object

    Returns:
        tuple: (Ind, W) where Ind is indices array (#x3) and W is weights array (#x2)
               Returns None if no stitch edges are found
    """
    edges_in_faces = set()
    for poly in mesh.polygons:
        for i in range(len(poly.vertices)):
            v1 = poly.vertices[i]
            v2 = poly.vertices[(i + 1) % len(poly.vertices)]
            edge = tuple(sorted([v1, v2]))
            edges_in_faces.add(edge)

    stitch_edges = []
    for edge in mesh.edges:
        edge_verts = tuple(sorted([edge.vertices[0], edge.vertices[1]]))
        if edge_verts not in edges_in_faces:
            stitch_edges.append(edge_verts)

    if not stitch_edges:
        return None

    Ind = []
    W = []
    for edge in stitch_edges:
        v1, v2 = edge
        Ind.append([v1, v2, v2, v2])
        W.append([1.0, 1.0, 0.0, 0.0])

    return (np.array(Ind, dtype=np.uint32), np.array(W, dtype=np.float32))


def _local_mesh_hash(obj):
    """Compute a content hash of an object's local-space triangulated mesh.

    Two non-linked duplicates with identical geometry (same vertices + faces
    in local space) will produce the same hash, enabling deduplication.
    """
    import hashlib
    from ..numpy_mesh_utils import extract_mesh_to_numpy, triangulate_numpy_mesh

    mesh = obj.data
    vertices, faces = extract_mesh_to_numpy(mesh)
    tri = triangulate_numpy_mesh(vertices, faces)
    h = hashlib.sha256()
    h.update(vertices.tobytes())
    h.update(tri.tobytes())
    return h.hexdigest()


def _build_obj_data(context, *, persist_topology_hash: bool) -> list:
    """Construct the list-of-group dicts that ``encode_obj`` pickles.

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
    from ..utils import count_ngon_faces, find_linked_duplicate_siblings
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
                ngon_count = count_ngon_faces(obj)
                if ngon_count > 0:
                    raise ValueError(
                        f"Object '{obj.name}' in group '{group.name}' has "
                        f"{ngon_count} N-gon face(s) (polygon with > 4 "
                        f"vertices). The solver supports only triangles "
                        f"and quads. Apply a Triangulate modifier or use "
                        f"Mesh > Faces > Triangulate Faces before "
                        f"transferring."
                    )

    # Compute and (optionally) store mesh topology summary.
    mesh_hash = compute_mesh_hash(context)
    if persist_topology_hash:
        state.set_mesh_hash(mesh_hash)

    data = []
    curr_frame = context.scene.frame_current
    try:
        return _encode_obj_inner(context, scene, state, data, curr_frame)
    finally:
        context.scene.frame_set(curr_frame)


def _encode_obj_inner(context, scene, state, data, curr_frame):
    context.scene.frame_set(1)
    fps = (
        bpy.context.scene.render.fps
        if state.use_frame_rate_in_output
        else int(state.frame_rate)
    )

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
                else:
                    # Deduplication: check if this mesh is a duplicate
                    content_hash = _local_mesh_hash(obj)
                    if content_hash in canonical_meshes:
                        # Duplicate: reference the canonical mesh, send only transform
                        mesh_ref = canonical_meshes[content_hash]
                        vert = None
                        tri = None
                        uv = []
                        stitch_data = None
                    else:
                        # Canonical: extract local-space mesh + transform
                        from ..uuid_registry import get_or_create_object_uuid
                        canonical_meshes[content_hash] = get_or_create_object_uuid(obj)
                        from ..numpy_mesh_utils import (
                            extract_mesh_to_numpy,
                            triangulate_numpy_mesh,
                            triangulate_uv_data,
                        )
                        local_verts, faces = extract_mesh_to_numpy(mesh)
                        tri = triangulate_numpy_mesh(local_verts, faces)
                        vert = local_verts
                        uv = triangulate_uv_data(mesh, tri)
                        stitch_data = detect_stitch_edges(mesh)

            info = {}
            if group.object_type == "STATIC":
                transform_kf = get_transform_keyframes(obj, context, fps)
                if transform_kf is not None:
                    info["transform_animation"] = transform_kf
                # UI-assigned static ops: find the AssignedObject entry
                # for this obj and serialize its ops list. Mutually
                # exclusive with Blender fcurve animation.
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
                ):
                    # Skip if transform_kf is already set — fcurves take
                    # precedence; the UI warns the user the ops are ignored.
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
                        # Spin/scale always pivot around the object origin —
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
                else:
                    info["face"] = tri
                    if len(uv) > 0:
                        info["uv"] = uv
                    if stitch_data is not None:
                        info["stitch"] = stitch_data

            objects_info.append(info)

        data.append(
            {
                "object": objects_info,
                "type": str(group.object_type),
            }
        )

    return data


def encode_obj(context) -> bytes:
    return pickle.dumps(_build_obj_data(context, persist_topology_hash=True))


def compute_data_hash(context) -> str:
    """SHA-256 fingerprint of the encoded scene data.

    Pickle-derived because the data tree carries large numpy vertex
    arrays; canonicalizing those through the json route the param hash
    uses would dominate the runtime. Pickle output is deterministic for
    fixed dict-insertion order and stable numpy versions, which the
    encoder maintains.

    Side-effect-free: does NOT stamp ``state.mesh_hash_json`` (only the
    upload path should do that). Raises ``ValueError`` for the same
    duplicate-assignment / encoding errors ``encode_obj`` raises.
    """
    data = _build_obj_data(context, persist_topology_hash=False)
    return hashlib.sha256(pickle.dumps(data, protocol=4)).hexdigest()
