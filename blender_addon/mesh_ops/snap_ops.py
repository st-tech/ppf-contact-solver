# File: snap_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore
from mathutils import Vector, kdtree  # pyright: ignore

from ..models.groups import get_addon_data, iterate_active_object_groups, pair_supports_cross_stitch
from ..ui.state import get_snap_objects


def _get_group_type_by_obj(scene, obj):
    from ..core.uuid_registry import get_object_uuid
    uid = get_object_uuid(obj)
    if not uid:
        return None
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            if assigned.uuid == uid:
                return group.object_type
    return None


def _get_snap_points(scene, obj):
    if obj.type == "MESH":
        return [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    if obj.type == "CURVE" and _get_group_type_by_obj(scene, obj) == "ROD":
        from ..core.curve_rod import sample_curve

        vert, _, _ = sample_curve(obj, obj.matrix_world)
        return [Vector(v) for v in vert]
    raise ValueError(
        f"Object '{obj.name}' is not supported for snapping. "
        "Only meshes and curve rods are supported."
    )


def _get_pair_contact_gaps(scene, obj_a, obj_b):
    """Return the per-object contact keep-out distance = contact_gap +
    contact_offset.

    The solver maps ghat=contact_gap, offset=contact_offset (scene.rs). The
    contact barrier between two elements activates at ``ghat + offset`` and
    asserts on a pair starting inside ``offset`` (the minimum separation), where
    both terms SUM the two objects' values; the build contact-offset gate also
    rejects pairs closer than the SUM of the two objects' offsets. So the soft
    stitch must start the closest pair beyond ``(offset_a+offset_b)`` to clear
    both gates, and beyond ``(ghat_a+ghat_b)+(offset_a+offset_b)`` to start
    OUTSIDE the barrier's activation band. Returning ``gap+offset`` per object
    and summing the pair gives that safe separation. The value also seeds the
    cross-stitch search radius."""
    from ..core.uuid_registry import get_object_uuid
    uid_a = get_object_uuid(obj_a)
    uid_b = get_object_uuid(obj_b)
    gap_a = gap_b = 0.0
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            if uid_a and assigned.uuid == uid_a:
                gap_a = group.computed_contact_gap + group.computed_contact_offset
            elif uid_b and assigned.uuid == uid_b:
                gap_b = group.computed_contact_gap + group.computed_contact_offset
    return gap_a, gap_b


def _apply_world_translation(obj, world_translation):
    if obj.parent is None:
        obj.location += world_translation
        return

    parent_inv = obj.parent.matrix_world.inverted_safe().to_3x3()
    obj.location += parent_inv @ world_translation


def _closest_point_on_segment(point, a, b):
    """Return the world-space point on segment [a, b] closest to ``point``."""
    ab = b - a
    denom = ab.dot(ab)
    if denom < 1e-20:
        return Vector(a)
    t = (point - a).dot(ab) / denom
    t = max(0.0, min(1.0, t))
    return a + ab * t


def _iter_rod_segments(scene, obj):
    """Yield world-space (p0, p1) endpoint pairs for each rod edge of a
    ROD-typed curve target, so the closest point on a target edge can be
    computed."""
    if obj.type == "CURVE" and _get_group_type_by_obj(scene, obj) == "ROD":
        from ..core.curve_rod import sample_curve

        vert, edges, _ = sample_curve(obj, obj.matrix_world)
        for e in edges:
            yield Vector(vert[int(e[0])]), Vector(vert[int(e[1])])
        return
    # Fallback: treat consecutive snap points as a polyline.
    pts = _get_snap_points(scene, obj)
    for i in range(len(pts) - 1):
        yield pts[i], pts[i + 1]


def _closest_source_target_pair(scene, source_obj, target_obj, target_type):
    """Return (source_point, target_point, dist): the globally-closest pair
    between a SOURCE object's vertices and the closest point on the TARGET
    object's nearest feature (edge for ROD targets, triangle for MESH/SOLID
    targets). This is feature-based (closest point on edge/triangle), NOT
    nearest vertex-to-vertex. Returns (None, None, inf) if no pair is found."""
    source_points = _get_snap_points(scene, source_obj)
    if not source_points:
        return None, None, float("inf")

    best_src = best_tgt = None
    best_dist = float("inf")

    if target_type == "ROD":
        segments = list(_iter_rod_segments(scene, target_obj))
        if not segments:
            return None, None, float("inf")
        for src in source_points:
            for a, b in segments:
                proj = _closest_point_on_segment(src, a, b)
                dist = (src - proj).length
                if dist < best_dist:
                    best_dist = dist
                    best_src = src
                    best_tgt = proj
        return best_src, best_tgt, best_dist

    # MESH / SOLID target: closest point on the nearest triangle.
    if target_obj.type != "MESH":
        return None, None, float("inf")
    triangles = list(_iter_mesh_triangles(target_obj))
    if not triangles:
        return None, None, float("inf")
    for src in source_points:
        for _tri, p0, p1, p2 in triangles:
            result = _closest_point_on_triangle(src, p0, p1, p2)
            if result is None:
                continue
            proj, _bary = result
            dist = (src - proj).length
            if dist < best_dist:
                best_dist = dist
                best_src = src
                best_tgt = proj
    return best_src, best_tgt, best_dist


def _resolve_stitch_source_target(scene, obj_a, obj_b):
    """Mirror the source/target-type determination in
    :func:`_build_explicit_cross_stitch`: pick which of the pair is the stitch
    SOURCE (its vertices are projected) and which is the TARGET, plus the
    target type. Returns (source_obj, target_obj, target_type) or
    (None, None, None) when the pair does not support cross-stitch."""
    type_a = _get_group_type_by_obj(scene, obj_a)
    type_b = _get_group_type_by_obj(scene, obj_b)
    if not pair_supports_cross_stitch(type_a, type_b):
        return None, None, None
    if {type_a, type_b} == {"SHELL", "SOLID"}:
        source_obj = obj_a if type_a == "SHELL" else obj_b
        target_obj = obj_b if type_a == "SHELL" else obj_a
        return source_obj, target_obj, "SOLID"
    if {type_a, type_b} == {"SHELL"}:
        return obj_a, obj_b, "SHELL"
    if {type_a, type_b} == {"ROD", "SHELL"}:
        source_obj = obj_a if type_a == "ROD" else obj_b
        target_obj = obj_b if type_a == "ROD" else obj_a
        return source_obj, target_obj, "SHELL"
    if {type_a, type_b} == {"ROD", "SOLID"}:
        source_obj = obj_a if type_a == "ROD" else obj_b
        target_obj = obj_b if type_a == "ROD" else obj_a
        return source_obj, target_obj, "SOLID"
    if {type_a, type_b} == {"SOLID"}:
        # Solid-solid: project obj_a's surface vertices onto obj_b's surface
        # triangles. Interior vertices are excluded by the distance threshold
        # in the builder (they sit far from the other solid's surface).
        return obj_a, obj_b, "SOLID"
    if "STATIC" in {type_a, type_b}:
        # Dynamic-to-static: the STATIC is always the TARGET and is treated
        # like a SHELL target (plain triangle mesh, 1:1 indices, no
        # re-projection). The dynamic object is the SOURCE; a SOLID source is
        # re-projected onto its tet surface downstream, a SHELL/ROD source
        # passes through. The STATIC must never be the moved object at snap.
        source_obj = obj_b if type_a == "STATIC" else obj_a
        target_obj = obj_a if type_a == "STATIC" else obj_b
        return source_obj, target_obj, "SHELL"
    # ROD-ROD
    return obj_a, obj_b, "ROD"


def _iter_mesh_triangles(obj):
    # Triangulate via Blender's loop_triangles, the same tessellation the
    # encoder ships to the solver (see numpy_mesh_utils.loop_triangle_indices),
    # so a cross-stitch anchor lands on a triangle the solver actually has.
    # Identical to the old fan split for tris/quads; only concave / non-planar
    # N-gons differ, where the fan could place anchors on triangles outside
    # the real face.
    from ..core.numpy_mesh_utils import loop_triangle_indices
    mesh = obj.data
    mat = obj.matrix_world
    for row in loop_triangle_indices(mesh):
        tri = [int(row[0]), int(row[1]), int(row[2])]
        p0 = mat @ mesh.vertices[tri[0]].co
        p1 = mat @ mesh.vertices[tri[1]].co
        p2 = mat @ mesh.vertices[tri[2]].co
        yield tri, p0, p1, p2


def _closest_point_on_triangle(point, p0, p1, p2):
    e1 = p1 - p0
    e2 = p2 - p0
    v0 = point - p0
    d00 = e1.dot(e1)
    d01 = e1.dot(e2)
    d11 = e2.dot(e2)
    d20 = v0.dot(e1)
    d21 = v0.dot(e2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return None
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    alpha = max(0.0, min(1.0, alpha))
    beta = max(0.0, min(1.0, beta))
    gamma = max(0.0, min(1.0, gamma))
    s = alpha + beta + gamma
    if s > 0.0:
        alpha /= s
        beta /= s
        gamma /= s
    proj = alpha * p0 + beta * p1 + gamma * p2
    return proj, [alpha, beta, gamma]


def _build_explicit_cross_stitch(scene, obj_a, obj_b, threshold):
    type_a = _get_group_type_by_obj(scene, obj_a)
    type_b = _get_group_type_by_obj(scene, obj_b)
    if not pair_supports_cross_stitch(type_a, type_b):
        return None

    if {type_a, type_b} == {"SHELL", "SOLID"}:
        source_obj = obj_a if type_a == "SHELL" else obj_b
        target_obj = obj_b if type_a == "SHELL" else obj_a
        target_type = "SOLID"
    elif {type_a, type_b} == {"SHELL"}:
        source_obj = obj_a
        target_obj = obj_b
        target_type = "SHELL"
    elif {type_a, type_b} == {"ROD", "SHELL"}:
        source_obj = obj_a if type_a == "ROD" else obj_b
        target_obj = obj_b if type_a == "ROD" else obj_a
        target_type = "SHELL"
    elif {type_a, type_b} == {"ROD", "SOLID"}:
        source_obj = obj_a if type_a == "ROD" else obj_b
        target_obj = obj_b if type_a == "ROD" else obj_a
        target_type = "SOLID"
    elif {type_a, type_b} == {"SOLID"}:
        source_obj = obj_a
        target_obj = obj_b
        target_type = "SOLID"
    elif "STATIC" in {type_a, type_b}:
        # Dynamic-to-static: STATIC is the target, treated like a SHELL
        # target (triangle path, 1:1 indices, no re-projection). Mirror of
        # the STATIC branch in _resolve_stitch_source_target.
        source_obj = obj_b if type_a == "STATIC" else obj_a
        target_obj = obj_a if type_a == "STATIC" else obj_b
        target_type = "SHELL"
    else:
        source_obj = obj_a
        target_obj = obj_b
        target_type = "ROD"

    source_points = _get_snap_points(scene, source_obj)
    if target_type == "ROD":
        target_points = _get_snap_points(scene, target_obj)
        if not source_points or not target_points:
            return None
        kd = kdtree.KDTree(len(target_points))
        for i, point in enumerate(target_points):
            kd.insert(point, i)
        kd.balance()
        ind = []
        w = []
        for si, source_point in enumerate(source_points):
            _, ti, dist = kd.find(source_point)
            if dist > threshold:
                continue
            # 6-wide baseline: degenerate source [si, si, si] / [1, 0, 0]
            # and a single target rod vertex ti -> degenerate target bary
            # [ti, ti, ti] / [1, 0, 0]. A SOLID source/target side is
            # re-projected onto its tet surface in the PyO3 decoder.
            ind.append([si, si, si, ti, ti, ti])
            w.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        if not ind:
            return None
        from ..core.uuid_registry import get_or_create_object_uuid
        src_uid = get_or_create_object_uuid(source_obj)
        tgt_uid = get_or_create_object_uuid(target_obj)
        if not src_uid or not tgt_uid:
            return None
        return {
            "source_uuid": src_uid,
            "target_uuid": tgt_uid,
            "ind": ind,
            "w": w,
            "source_points": [
                [float(source_points[row[0]].x), float(source_points[row[0]].y), float(source_points[row[0]].z)]
                for row in ind
            ],
            "target_points": [
                [float(target_points[row[3]].x), float(target_points[row[3]].y), float(target_points[row[3]].z)]
                for row in ind
            ],
        }

    if target_obj.type != "MESH":
        return None

    ind = []
    w = []
    target_positions = []
    for si, source_point in enumerate(source_points):
        best_dist = float("inf")
        best_tri = None
        best_bary = None
        best_proj = None
        for tri, p0, p1, p2 in _iter_mesh_triangles(target_obj):
            result = _closest_point_on_triangle(source_point, p0, p1, p2)
            if result is None:
                continue
            proj, bary = result
            dist = (source_point - proj).length
            if dist < best_dist:
                best_dist = dist
                best_tri = tri
                best_bary = bary
                best_proj = proj
        if best_tri is None or best_bary is None or best_dist > threshold:
            continue
        # 6-wide baseline: degenerate source [si, si, si] / [1, 0, 0] and
        # the snap-time target barycentric on the Blender mesh. A SOLID
        # source/target side is re-projected onto its tet surface in the
        # PyO3 decoder (using source_points / target_points below).
        ind.append([si, si, si, int(best_tri[0]), int(best_tri[1]), int(best_tri[2])])
        w.append([1.0, 0.0, 0.0, float(best_bary[0]), float(best_bary[1]), float(best_bary[2])])
        target_positions.append(
            [float(best_proj.x), float(best_proj.y), float(best_proj.z)]
        )

    if not ind:
        return None
    from ..core.uuid_registry import get_or_create_object_uuid
    src_uid = get_or_create_object_uuid(source_obj)
    tgt_uid = get_or_create_object_uuid(target_obj)
    if not src_uid or not tgt_uid:
        return None
    return {
        "source_uuid": src_uid,
        "target_uuid": tgt_uid,
        "ind": ind,
        "w": w,
        "source_points": [
            [float(source_points[row[0]].x), float(source_points[row[0]].y), float(source_points[row[0]].z)]
            for row in ind
        ],
        "target_points": target_positions,
    }


def _snap_pair(operator, context, obj_a, obj_b):
    """Snap ``obj_a`` to ``obj_b`` and (re)build the merge pair entry.

    Shared by :class:`OBJECT_OT_SnapToVertices` (driven by the snap-A /
    snap-B dropdowns) and :class:`OBJECT_OT_ResnapMergePair` (driven by
    the active merge-pairs UIList entry). Returns the Blender operator
    return set; reports go through ``operator``.
    """
    state = get_addon_data(context.scene).state

    if not obj_a or not obj_b:
        operator.report({"ERROR"}, "One or both objects not found")
        return {"CANCELLED"}

    try:
        points_a = _get_snap_points(context.scene, obj_a)
        points_b = _get_snap_points(context.scene, obj_b)
    except ValueError as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}

    if not points_a or not points_b:
        operator.report({"ERROR"}, "Could not find snap points on one or both objects")
        return {"CANCELLED"}

    # Soft stitch (NOT a weld): find the SOURCE object's vertex whose closest
    # point on the TARGET object's nearest edge/triangle is globally minimal
    # (feature-based, NOT nearest vertex-to-vertex), then position obj_a so that
    # closest source-vertex / target-feature pair starts a small GAP apart,
    # roughly equal to the contact offset. The objects must NOT end coincident
    # or overlapping: a coincident/overlapping start trips a contact-offset
    # assert in the solver, and the soft stitch plus the contact barrier are
    # meant to equilibrate across this small gap rather than be merged.
    source_obj, target_obj, target_type = _resolve_stitch_source_target(
        context.scene, obj_a, obj_b,
    )
    if source_obj is None:
        operator.report(
            {"ERROR"}, "This object pair does not support stitching",
        )
        return {"CANCELLED"}

    source_point, target_point, min_distance = _closest_source_target_pair(
        context.scene, source_obj, target_obj, target_type,
    )
    if source_point is None or target_point is None:
        operator.report({"ERROR"}, "Could not find a closest source/target feature")
        return {"CANCELLED"}

    gap_a, gap_b = _get_pair_contact_gaps(context.scene, obj_a, obj_b)
    # SUM the two objects' (gap+offset): the build gate and the runtime barrier
    # both compare against the SUM of the two contact offsets, so the closest
    # pair must start beyond gap_a + gap_b. The 1.1 factor keeps it strictly
    # above the barrier activation band even when contact_gap is ~0 (so it never
    # trips the dist > offset assert), and the 1e-5 floor avoids exact
    # coincidence when an object has no contact params.
    keep = (gap_a + gap_b) * 1.1 + 1e-5

    # Decide which object to translate. Normally obj_a (the "Object A moves"
    # convention), but a STATIC collider must NEVER be moved: the encoder
    # ships the STATIC's rest-pose geometry and the stitch indices are keyed
    # to it, so translating the STATIC would silently desync its geometry from
    # the stitch. When obj_a is the STATIC, move the other (dynamic) object
    # instead. STATIC-STATIC pairs are rejected by pair_supports_cross_stitch,
    # so the non-STATIC side is always the one to move.
    move_obj = obj_b if _get_group_type_by_obj(context.scene, obj_a) == "STATIC" else obj_a

    # Move ``move_obj`` so the globally-closest source-vertex / target-feature
    # pair ends at distance ``keep`` (a small positive separation), not
    # coincident. The approach vector points from source to target; moving the
    # source toward the target leaves the gap, moving the target toward the
    # source needs the inverse.
    approach = target_point - source_point
    approach_len = approach.length
    if approach_len > 1e-9:
        translation = approach - approach.normalized() * keep
        if move_obj is target_obj:
            # The moved object is the target: invert so the target is pulled
            # toward the source instead of the source toward the target.
            translation = -translation
        _apply_world_translation(move_obj, translation)
        context.view_layer.update()

    # Threshold for selecting which source vertices participate in the
    # cross-stitch. Now that the closest pair sits ``keep`` apart, a vertex
    # belongs to the seam only if it is within a couple of contact bands of
    # the target. Key the radius to ``keep`` (the contact-derived gap), NOT
    # the object's size: a mesh-scale radius (a fraction of the bounding-box
    # diagonal) balloons on large sheets and wrongly stitches far corners
    # whose gap is many times offset+gap. The small absolute floor only
    # guards the degenerate case where contact params are ~0 (so ``keep``
    # collapses to ~1e-5); it is far below any real seam spacing.
    threshold = max(2.5 * keep, 1e-4)
    explicit_cross_stitch = _build_explicit_cross_stitch(
        context.scene, obj_a, obj_b, threshold=threshold,
    )

    from ..core.uuid_registry import get_or_create_object_uuid
    uuid_a = get_or_create_object_uuid(obj_a)
    uuid_b = get_or_create_object_uuid(obj_b)
    if not uuid_a or not uuid_b:
        operator.report(
            {"ERROR"},
            "Cannot snap: one or both objects are library-linked (unwritable)",
        )
        return {"CANCELLED"}
    pair_item = None
    for pair in state.merge_pairs:
        if (
            (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b)
            or (pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a)
        ):
            pair_item = pair
            break
    if pair_item is None:
        pair_item = state.merge_pairs.add()
        pair_item.object_a = obj_a.name
        pair_item.object_b = obj_b.name
        pair_item.object_a_uuid = uuid_a
        pair_item.object_b_uuid = uuid_b
        state.merge_pairs_index = len(state.merge_pairs) - 1
    if explicit_cross_stitch:
        # Stamp the source-mesh vertex counts into the JSON so
        # cleanup_stale_merge_pairs can detect a topology edit after
        # snap (changing vertex count) and clear the stale indices.
        if obj_a.type == "MESH":
            explicit_cross_stitch["a_vert_count"] = len(obj_a.data.vertices)
        if obj_b.type == "MESH":
            explicit_cross_stitch["b_vert_count"] = len(obj_b.data.vertices)
        pair_item.cross_stitch_json = json.dumps(
            explicit_cross_stitch, separators=(",", ":"),
        )
    else:
        pair_item.cross_stitch_json = ""

    from ..ui.dynamics.overlay import apply_object_overlays

    apply_object_overlays()

    operator.report(
        {"INFO"},
        f"Stitched {obj_a.name} to {obj_b.name} "
        f"(was {min_distance:.4f} apart, positioned at gap {keep:.4g})",
    )
    return {"FINISHED"}


class OBJECT_OT_SnapToVertices(Operator):
    """Snap object A to object B based on nearest vertices"""

    bl_idname = "object.snap_to_vertices"
    bl_label = "Snap A to B"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        state = get_addon_data(context.scene).state
        return (
            state.snap_object_a != "NONE"
            and state.snap_object_b != "NONE"
            and state.snap_object_a != state.snap_object_b
        )

    def execute(self, context):
        from ..core.uuid_registry import get_object_by_uuid

        state = get_addon_data(context.scene).state
        # Dropdown identifiers are UUIDs (see ui/state.py::get_snap_objects).
        obj_a = get_object_by_uuid(state.snap_object_a)
        obj_b = get_object_by_uuid(state.snap_object_b)
        return _snap_pair(self, context, obj_a, obj_b)


class OBJECT_OT_ResnapMergePair(Operator):
    """Re-run snap on the active merge pair to rebuild its stitch
    indices (useful after the addon's curve sampling rule changes or
    after edits that invalidated the cached cross-stitch JSON)."""

    bl_idname = "object.resnap_merge_pair"
    bl_label = "Re-snap"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        state = get_addon_data(context.scene).state
        idx = state.merge_pairs_index
        if not (0 <= idx < len(state.merge_pairs)):
            return False
        pair = state.merge_pairs[idx]
        return bool(pair.object_a_uuid and pair.object_b_uuid)

    def execute(self, context):
        from ..core.uuid_registry import get_object_by_uuid

        state = get_addon_data(context.scene).state
        idx = state.merge_pairs_index
        if not (0 <= idx < len(state.merge_pairs)):
            self.report({"ERROR"}, "No merge pair selected")
            return {"CANCELLED"}
        pair = state.merge_pairs[idx]
        obj_a = get_object_by_uuid(pair.object_a_uuid)
        obj_b = get_object_by_uuid(pair.object_b_uuid)
        return _snap_pair(self, context, obj_a, obj_b)


class OBJECT_OT_PickSnapObject(Operator):
    """Set snap object from the active selection"""

    bl_idname = "object.pick_snap_object"
    bl_label = "Pick from Selection"

    target: bpy.props.StringProperty(options={'HIDDEN'})  # pyright: ignore  # "A" or "B"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        from ..core.uuid_registry import get_or_create_object_uuid

        obj = context.active_object
        state = get_addon_data(context.scene).state
        # The snap enum identifies items by UUID (see ui/state.py::get_snap_objects).
        valid = {item[0] for item in get_snap_objects(state, context)}
        identifier = get_or_create_object_uuid(obj)
        if not identifier:
            self.report({"WARNING"}, f"'{obj.name}' is not writable (library-linked)")
            return {"CANCELLED"}
        if identifier not in valid:
            self.report({"WARNING"}, f"'{obj.name}' is not a snap-eligible object")
            return {"CANCELLED"}
        if self.target == "A":
            state.snap_object_a = identifier
        elif self.target == "B":
            state.snap_object_b = identifier
        return {"FINISHED"}


classes = (
    OBJECT_OT_SnapToVertices,
    OBJECT_OT_ResnapMergePair,
    OBJECT_OT_PickSnapObject,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
