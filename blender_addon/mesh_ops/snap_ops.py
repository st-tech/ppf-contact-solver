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


def _compute_target_normal(scene, obj_b, closest_a, closest_b):
    approach_dir = closest_a - closest_b
    if approach_dir.length > 1e-6:
        return approach_dir.normalized()
    return Vector((0, 0, 1))


def _should_apply_gap(scene, obj_a, obj_b):
    type_a = _get_group_type_by_obj(scene, obj_a)
    type_b = _get_group_type_by_obj(scene, obj_b)
    if type_a == "ROD" and type_b == "ROD":
        return False
    if type_a == "SHELL" and type_b == "SHELL":
        return False
    return True


def _get_pair_gap_and_offset(scene, obj_a, obj_b):
    from ..core.uuid_registry import get_object_uuid
    uid_a = get_object_uuid(obj_a)
    uid_b = get_object_uuid(obj_b)
    gap_a = offset_a = gap_b = offset_b = 0.0
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            if uid_a and assigned.uuid == uid_a:
                gap_a = group.computed_contact_gap
                offset_a = group.computed_contact_offset
            elif uid_b and assigned.uuid == uid_b:
                gap_b = group.computed_contact_gap
                offset_b = group.computed_contact_offset
    return gap_a, offset_a, gap_b, offset_b


def _apply_world_translation(obj, world_translation):
    if obj.parent is None:
        obj.location += world_translation
        return

    parent_inv = obj.parent.matrix_world.inverted_safe().to_3x3()
    obj.location += parent_inv @ world_translation


def _iter_mesh_triangles(obj):
    mesh = obj.data
    mat = obj.matrix_world
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) < 3:
            continue
        v0 = verts[0]
        for i in range(1, len(verts) - 1):
            tri = [v0, verts[i], verts[i + 1]]
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
            ind.append([si, ti, ti, ti])
            w.append([1.0, 1.0, 0.0, 0.0])
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
            "target_points": [
                [float(target_points[row[1]].x), float(target_points[row[1]].y), float(target_points[row[1]].z)]
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
        ind.append([si, int(best_tri[0]), int(best_tri[1]), int(best_tri[2])])
        w.append([1.0, float(best_bary[0]), float(best_bary[1]), float(best_bary[2])])
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
        "target_points": target_positions,
    }


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

        if not obj_a or not obj_b:
            self.report({"ERROR"}, "One or both objects not found")
            return {"CANCELLED"}

        try:
            points_a = _get_snap_points(context.scene, obj_a)
            points_b = _get_snap_points(context.scene, obj_b)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if not points_a or not points_b:
            self.report({"ERROR"}, "Could not find snap points on one or both objects")
            return {"CANCELLED"}

        kd = kdtree.KDTree(len(points_b))
        for i, world_co in enumerate(points_b):
            kd.insert(world_co, i)
        kd.balance()

        min_distance = float("inf")
        closest_vert_a = None
        closest_vert_b = None

        for world_co_a in points_a:
            co, _, dist = kd.find(world_co_a)

            if dist < min_distance:
                min_distance = dist
                closest_vert_a = world_co_a
                closest_vert_b = co

        if closest_vert_a is None or closest_vert_b is None:
            self.report({"ERROR"}, "Could not find closest vertices")
            return {"CANCELLED"}

        translation = closest_vert_b - closest_vert_a

        face_normal = _compute_target_normal(
            context.scene, obj_b, closest_vert_a, closest_vert_b
        )

        gap_a, offset_a, gap_b, offset_b = _get_pair_gap_and_offset(
            context.scene, obj_a, obj_b
        )
        total_gap = 0.0
        if _should_apply_gap(context.scene, obj_a, obj_b):
            import numpy as np
            base_gap = max(gap_a, gap_b) + offset_a + offset_b
            coord_mag = max(abs(c) for c in closest_vert_b)
            float32_margin = np.spacing(np.float32(coord_mag)) * 2048
            total_gap = base_gap + max(base_gap, float32_margin)
            translation += face_normal * total_gap

        # Apply translation in world space so parenting does not break snap motion.
        _apply_world_translation(obj_a, translation)
        context.view_layer.update()

        # h = post-snap separation for the closest vertex pair.
        # Gap cases: h = total_gap (the applied separation).
        # No-gap cases: h = max contact gap (mesh-scale tolerance).
        h = total_gap if total_gap > 0 else max(gap_a, gap_b)
        explicit_cross_stitch = _build_explicit_cross_stitch(
            context.scene, obj_a, obj_b, threshold=2.0 * h,
        )

        from ..core.uuid_registry import get_or_create_object_uuid
        uuid_a = get_or_create_object_uuid(obj_a)
        uuid_b = get_or_create_object_uuid(obj_b)
        if not uuid_a or not uuid_b:
            self.report({"ERROR"}, "Cannot snap: one or both objects are library-linked (unwritable)")
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
            pair_item.object_a_uuid = get_or_create_object_uuid(obj_a)
            pair_item.object_b_uuid = get_or_create_object_uuid(obj_b)
            state.merge_pairs_index = len(state.merge_pairs) - 1
        if explicit_cross_stitch:
            # Stamp the source-mesh vertex counts into the JSON so
            # cleanup_stale_merge_pairs can detect a topology edit after
            # snap (changing vertex count) and clear the stale indices.
            explicit_cross_stitch["a_vert_count"] = len(obj_a.data.vertices)
            explicit_cross_stitch["b_vert_count"] = len(obj_b.data.vertices)
            pair_item.cross_stitch_json = json.dumps(
                explicit_cross_stitch, separators=(",", ":"),
            )
        else:
            pair_item.cross_stitch_json = ""

        from ..ui.dynamics.overlay import apply_object_overlays

        apply_object_overlays()

        self.report(
            {"INFO"},
            f"Snapped {obj_a.name} to {obj_b.name} (distance: {min_distance:.4f})",
        )
        return {"FINISHED"}


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


classes = (OBJECT_OT_SnapToVertices, OBJECT_OT_PickSnapObject)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
