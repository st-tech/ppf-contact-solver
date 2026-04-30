# File: encoder/pin.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json

import bpy  # pyright: ignore
import numpy as np
from mathutils import Vector  # pyright: ignore

from ...models.groups import decode_vertex_group_identifier, get_addon_data
from ..utils import get_vertices_in_group, parse_vertex_index, world_matrix
from . import _swap_axes, _to_solver


def _get_point_co(obj, idx):
    """Get control point position by flat index (works for MESH and CURVE)."""
    if obj.type == "MESH":
        return obj.data.vertices[idx].co
    elif obj.type == "CURVE":
        i = 0
        for s in obj.data.splines:
            if s.type == "BEZIER":
                for bp in s.bezier_points:
                    if i == idx:
                        return bp.co
                    i += 1
            elif s.type in ("NURBS", "POLY"):
                for p in s.points:
                    if i == idx:
                        return Vector((p.co[0], p.co[1], p.co[2]))
                    i += 1
    return Vector((0, 0, 0))


def _get_pin_indices(obj, vg_name):
    """Get pin vertex indices for both MESH and CURVE objects."""
    if obj.type == "CURVE":
        raw = obj.get(f"_pin_{vg_name}")
        return json.loads(raw) if raw else []
    vg = obj.vertex_groups.get(vg_name)
    if vg:
        return get_vertices_in_group(obj, vg)
    return []


def _max_towards_center(obj, vg_name, state, direction, frame=None, eps=1e-3):
    """Compute center from pin vertices furthest towards a direction (grab-style).

    The direction is in Blender world space. Vertex selection is done in
    Blender world space, then the centroid is transformed to solver space
    via world_matrix (axis swap only).

    If frame is given, evaluates vertex positions at that frame.
    """
    pin_indices = _get_pin_indices(obj, vg_name)
    if not pin_indices:
        return [0.0, 0.0, 0.0]
    saved_frame = bpy.context.scene.frame_current
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    # Select vertices in Blender world space (direction is Blender space)
    blender_mat = obj.matrix_world
    positions = np.array([list(blender_mat @ _get_point_co(obj, i)) for i in pin_indices])
    d = np.array(direction, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        if frame is not None:
            bpy.context.scene.frame_set(saved_frame)
        return [0.0, 0.0, 0.0]
    d = d / d_norm
    projections = positions @ d
    max_val = projections.max()
    mask = projections > max_val - eps
    selected_indices = [pin_indices[i] for i in range(len(pin_indices)) if mask[i]]
    # Compute centroid in solver space without translation (solver adds
    # displacement separately, so positions must be in untranslated space).
    solver_mat = world_matrix(obj).to_3x3().to_4x4()
    solver_positions = np.array([list(solver_mat @ _get_point_co(obj, i)) for i in selected_indices])
    centroid = solver_positions.mean(axis=0)
    if frame is not None:
        bpy.context.scene.frame_set(saved_frame)
    return centroid.tolist()


def _encode_pin_config(context, groups, state):
    """Encode pin vertex group config and animation."""
    scene = context.scene
    fps = (
        bpy.context.scene.render.fps
        if state.use_frame_rate_in_output
        else int(state.frame_rate)
    )
    # Collect per-object pin config (duration, pull, operations) keyed by vertex index
    # Structure: {obj_name: {vertex_index: {"unpin_time": ..., "pull_strength": ..., "operations": [...]}}}
    pin_config = {}
    for group in groups:
        if group.object_type == "STATIC":
            continue
        for pin_item in group.pin_vertex_groups:
            has_operations = len(pin_item.operations) > 0
            if not pin_item.use_pin_duration and not pin_item.use_pull and not has_operations:
                continue
            from ..uuid_registry import resolve_pin, get_or_create_object_uuid
            obj = resolve_pin(pin_item)
            if not obj or obj.type not in ("MESH", "CURVE"):
                continue
            # Re-read after resolve_pin (handles object + VG renames)
            obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
            if not obj_name or not vg_name:
                continue
            if not _get_pin_indices(obj, vg_name):
                continue
            obj_uuid = get_or_create_object_uuid(obj)
            cfg = {}
            saved_col = get_addon_data(scene).state.saved_pin_keyframes
            if pin_item.use_pin_duration:
                cfg["unpin_time"] = float(pin_item.pin_duration) / fps
            if pin_item.use_pull:
                cfg["pull_strength"] = float(pin_item.pull_strength)
            # Detect embedded move from operations list (synced from F-curves by UI).
            # Reject >1 — each pin gets at most one embedded move because the
            # decoder uses the single slot keyed on embedded_move_index.
            embedded_move_count = sum(
                1 for op in pin_item.operations if op.op_type == "EMBEDDED_MOVE"
            )
            if embedded_move_count > 1:
                raise ValueError(
                    f"Pin '{vg_name}' on '{obj_name}': more than one "
                    f"EMBEDDED_MOVE operation is not supported"
                )
            has_embedded_move = embedded_move_count == 1
            if has_operations or has_embedded_move:
                ops_list = []
                embedded_move_index = -1
                centroid_blender = None
                if has_embedded_move:
                    # Compute centroid from saved_pin_keyframes (frame 1 values)
                    for grp_entry in saved_col:
                        if grp_entry.object_uuid == obj_uuid and grp_entry.vertex_group == vg_name:
                            rest_pos = {}  # {vertex_idx: [x, y, z]}
                            for fc_entry in grp_entry.fcurves:
                                vi = parse_vertex_index(fc_entry.data_path)
                                if vi is not None and fc_entry.points:
                                    if vi not in rest_pos:
                                        rest_pos[vi] = [0.0, 0.0, 0.0]
                                    rest_pos[vi][fc_entry.array_index] = fc_entry.points[0].value
                            if rest_pos:
                                n = len(rest_pos)
                                cx = sum(p[0] for p in rest_pos.values()) / n
                                cy = sum(p[1] for p in rest_pos.values()) / n
                                cz = sum(p[2] for p in rest_pos.values()) / n
                                centroid_blender = _swap_axes([cx, cy, cz])
                            break
                else:
                    # No embedded move -- compute centroid from frame 1 vertex positions
                    # Use no-translation matrix (solver adds displacement separately)
                    pin_indices = _get_pin_indices(obj, vg_name)
                    if pin_indices:
                        mat = world_matrix(obj).to_3x3().to_4x4()
                        positions = [mat @ _get_point_co(obj, i) for i in pin_indices]
                        n = len(positions)
                        centroid_blender = [
                            sum(p[0] for p in positions) / n,
                            sum(p[1] for p in positions) / n,
                            sum(p[2] for p in positions) / n,
                        ]
                # Validate: torque cannot be mixed with kinematic ops
                op_types = {o.op_type for o in pin_item.operations}
                has_torque = "TORQUE" in op_types
                has_kinematic = op_types & {"MOVE_BY", "SPIN", "SCALE"}
                if has_torque and has_kinematic:
                    raise ValueError(
                        f"Pin '{vg_name}' on '{obj_name}': "
                        "torque cannot be mixed with Move/Spin/Scale operations"
                    )
                # CENTROID center mode bakes the frame-1 centroid at
                # encode time; EMBEDDED_MOVE then shifts pins per-frame.
                # The solver doesn't re-compute the centroid as the
                # pins move, so the two combined silently drift apart.
                # Reject up front so the user picks one or the other.
                if has_embedded_move:
                    for op in pin_item.operations:
                        if op.op_type == "SPIN" and op.spin_center_mode == "CENTROID":
                            raise ValueError(
                                f"Pin '{vg_name}' on '{obj_name}': "
                                "SPIN with CENTROID center cannot be combined "
                                "with EMBEDDED_MOVE (centroid is baked at "
                                "frame 1 and would drift from the moving pin)"
                            )
                        if op.op_type == "SCALE" and op.scale_center_mode == "CENTROID":
                            raise ValueError(
                                f"Pin '{vg_name}' on '{obj_name}': "
                                "SCALE with CENTROID center cannot be combined "
                                "with EMBEDDED_MOVE (centroid is baked at "
                                "frame 1 and would drift from the moving pin)"
                            )
                # Object's solver-space translation (displacement).
                # Subtract from absolute centers so they match the solver's
                # untranslated vertex space.
                disp = world_matrix(obj).to_translation()
                if has_embedded_move:
                    embedded_move_index = 0  # always first
                for op_i, op in enumerate(pin_item.operations):
                    if op.op_type == "EMBEDDED_MOVE":
                        continue
                    op_dict = {"type": op.op_type.lower()}
                    if op.op_type == "MOVE_BY":
                        op_dict["delta"] = _to_solver(op.delta)
                    elif op.op_type == "SPIN":
                        if op.spin_center_mode == "CENTROID" and centroid_blender:
                            op_dict["center_mode"] = "centroid"
                            op_dict["center"] = centroid_blender
                        elif op.spin_center_mode == "MAX_TOWARDS":
                            op_dict["center_mode"] = "absolute"
                            op_dict["center"] = _max_towards_center(
                                obj, vg_name, state, list(op.spin_center_direction),
                                frame=op.frame_start,
                            )
                        elif op.spin_center_mode == "VERTEX":
                            if op.spin_center_vertex < 0:
                                raise ValueError(
                                    f"Pin '{vg_name}' on '{obj_name}': "
                                    "Spin center vertex not set — pick a vertex in Edit Mode"
                                )
                            op_dict["center_mode"] = "absolute"
                            mat_nt = world_matrix(obj).to_3x3().to_4x4()
                            op_dict["center"] = list(mat_nt @ _get_point_co(obj, op.spin_center_vertex))
                        else:
                            op_dict["center_mode"] = "absolute"
                            c = Vector(_to_solver(op.spin_center)) - disp
                            op_dict["center"] = list(c)
                        # Normalize the spin axis so the frontend's rotation
                        # formula (which takes it as a unit vector) gets the
                        # angular velocity the user asked for, regardless of
                        # the magnitude they typed in the UI.
                        axis = _swap_axes(op.spin_axis)
                        axis_arr = np.asarray(axis, dtype=np.float64)
                        axis_norm = float(np.linalg.norm(axis_arr))
                        if axis_norm > 1e-9:
                            axis_arr = axis_arr / axis_norm
                        if op.spin_flip:
                            axis_arr = -axis_arr
                        op_dict["axis"] = axis_arr.tolist()
                        op_dict["angular_velocity"] = float(op.spin_angular_velocity)
                    elif op.op_type == "SCALE":
                        if op.scale_center_mode == "CENTROID" and centroid_blender:
                            op_dict["center_mode"] = "centroid"
                            op_dict["center"] = centroid_blender
                        elif op.scale_center_mode == "MAX_TOWARDS":
                            op_dict["center_mode"] = "absolute"
                            op_dict["center"] = _max_towards_center(
                                obj, vg_name, state, list(op.scale_center_direction),
                                frame=op.frame_start,
                            )
                        elif op.scale_center_mode == "VERTEX":
                            if op.scale_center_vertex < 0:
                                raise ValueError(
                                    f"Pin '{vg_name}' on '{obj_name}': "
                                    "Scale center vertex not set — pick a vertex in Edit Mode"
                                )
                            op_dict["center_mode"] = "absolute"
                            mat_nt = world_matrix(obj).to_3x3().to_4x4()
                            op_dict["center"] = list(mat_nt @ _get_point_co(obj, op.scale_center_vertex))
                        else:
                            op_dict["center_mode"] = "absolute"
                            c = Vector(_to_solver(op.scale_center)) - disp
                            op_dict["center"] = list(c)
                        op_dict["factor"] = float(op.scale_factor)
                    elif op.op_type == "TORQUE":
                        comp_idx = {"PC1": 0, "PC2": 1, "PC3": 2}.get(
                            op.torque_axis_component, 2,
                        )
                        op_dict["axis_component"] = comp_idx
                        op_dict["magnitude"] = float(op.torque_magnitude)
                        # Find the vertex with max projection onto PCA axis
                        # as orientation hint (index into the pin group)
                        pin_indices_hint = _get_pin_indices(obj, vg_name)
                        if len(pin_indices_hint) < 3:
                            raise ValueError(
                                f"Pin '{vg_name}' on '{obj_name}': TORQUE "
                                f"requires at least 3 vertices for PCA axis "
                                f"(got {len(pin_indices_hint)})"
                            )
                        mat_hint = world_matrix(obj)
                        pos_hint = np.array([
                            list(mat_hint @ _get_point_co(obj, i))
                            for i in pin_indices_hint
                        ])
                        centroid_hint = pos_hint.mean(axis=0)
                        centered_hint = pos_hint - centroid_hint
                        cov_hint = np.cov(centered_hint, rowvar=False)
                        _, eigvecs_hint = np.linalg.eigh(cov_hint)
                        sort_idx_hint = np.argsort(np.linalg.eigvalsh(cov_hint))[::-1]
                        eigvecs_hint = eigvecs_hint[:, sort_idx_hint]
                        axis_hint = eigvecs_hint[:, comp_idx]
                        projections = centered_hint @ axis_hint
                        if not np.all(np.isfinite(projections)):
                            raise ValueError(
                                f"Pin '{vg_name}' on '{obj_name}': TORQUE "
                                f"PCA produced non-finite axis (are pins collinear?)"
                            )
                        # Store Blender vertex index for axis orientation hint
                        if op.torque_flip:
                            op_dict["hint_vertex"] = int(pin_indices_hint[np.argmin(projections)])
                        else:
                            op_dict["hint_vertex"] = int(pin_indices_hint[np.argmax(projections)])
                    # Clamp inverted ranges to zero duration instead of
                    # emitting a negative interval the solver would reject.
                    start_frame = op.frame_start
                    end_frame = max(op.frame_end, start_frame)
                    op_dict["t_start"] = (start_frame - 1) / fps
                    op_dict["t_end"] = (end_frame - 1) / fps
                    op_dict["transition"] = op.transition.lower()
                    ops_list.append(op_dict)
                if ops_list:
                    cfg["operations"] = ops_list
                if embedded_move_index >= 0:
                    cfg["embedded_move_index"] = embedded_move_index
            # Encode pin animation entirely from saved_pin_keyframes
            # No dependency on mesh fcurves or current frame
            # Use no-translation matrix (solver adds displacement separately)
            mat = world_matrix(obj).to_3x3().to_4x4()
            pin_anim = {}
            for grp_entry in saved_col:
                if grp_entry.object_uuid == obj_uuid and grp_entry.vertex_group == vg_name:
                    keyframes = {}
                    for fc_entry in grp_entry.fcurves:
                        idx = parse_vertex_index(fc_entry.data_path)
                        if idx is None:
                            continue
                        if idx not in keyframes:
                            keyframes[idx] = {}
                        for pt in fc_entry.points:
                            if pt.frame not in keyframes[idx]:
                                keyframes[idx][pt.frame] = {}
                            keyframes[idx][pt.frame][fc_entry.array_index] = pt.value
                    for idx, animation in keyframes.items():
                        min_frame = min(animation.keys())
                        max_frame = max(animation.keys())
                        xyz = []
                        anim_frames = []
                        # Use first keyframe as initial coord (from saved data)
                        coord = [0.0, 0.0, 0.0]
                        if min_frame in animation:
                            for ai, val in animation[min_frame].items():
                                coord[ai] = val
                        for n in range(min_frame, max_frame + 1):
                            if n in animation:
                                for ai, value in animation[n].items():
                                    coord[ai] = value
                                anim_frames.append(n)
                                xyz.append(coord.copy())
                        if xyz:
                            xyz_arr = np.array(
                                [mat @ Vector(v) for v in xyz], dtype=np.float32,
                            )
                            times = [(f - 1) / fps for f in anim_frames]
                            pin_anim[idx] = {"time": times, "position": xyz_arr}
                    break
            if pin_anim:
                cfg["pin_anim"] = pin_anim
            # Key by UUID for rename resilience
            if obj_uuid not in pin_config:
                pin_config[obj_uuid] = {}
            # Tag with group identity so solver can merge torque vertices
            cfg["pin_group_id"] = f"{obj_uuid}:{vg_name}"
            cfg["obj_uuid"] = obj_uuid
            for vi in _get_pin_indices(obj, vg_name):
                pin_config[obj_uuid][vi] = cfg

    return pin_config
