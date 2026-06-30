# File: encoder/pin.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json

import bpy  # pyright: ignore
import numpy as np
from mathutils import Vector  # pyright: ignore

from ...models.groups import decode_vertex_group_identifier
from ..utils import (
    get_vertices_in_group,
    pin_covers_all_vertices,
    world_matrix,
)
from . import _swap_axes, _to_solver, resolve_fps


def _solver_rot_matrix(obj):
    """Translation-stripped solver world matrix (rotation+scale only).

    The solver stores vertices in untranslated space and tracks object
    translation separately as displacement, so vertex-derived positions
    must drop the translation component. See README "Solver space
    convention".
    """
    return world_matrix(obj).to_3x3().to_4x4()


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
    solver_mat = _solver_rot_matrix(obj)
    solver_positions = np.array([list(solver_mat @ _get_point_co(obj, i)) for i in selected_indices])
    centroid = solver_positions.mean(axis=0)
    if frame is not None:
        bpy.context.scene.frame_set(saved_frame)
    return centroid.tolist()


_PIN_VERTEX_CO_RGX = None  # lazy-initialized in the collector below


def _collect_pin_vertex_fcurve_frames(obj, vg_name):
    """Walk ``obj.data.animation_data.action`` for vertex-co fcurves on
    pinned vertices. Returns ``(sorted_unique_frames, lookup)`` where
    ``lookup`` is ``{(vertex_index, axis_index): fcurve}``. Both
    Blender 5.x layered actions and the legacy flat layout are
    supported. Returns ``([], {})`` when no animation data or no
    matching fcurves exist.
    """
    import re

    global _PIN_VERTEX_CO_RGX
    if _PIN_VERTEX_CO_RGX is None:
        _PIN_VERTEX_CO_RGX = re.compile(r"vertices\[(\d+)\]\.co$")
    rgx = _PIN_VERTEX_CO_RGX

    ad = obj.data.animation_data
    action = ad.action if ad is not None else None
    if action is None:
        return [], {}

    pinned = set(_get_pin_indices(obj, vg_name))
    if not pinned:
        return [], {}

    lookup: dict[tuple[int, int], object] = {}
    frames: set[int] = set()

    def consume(fc):
        m = rgx.match(fc.data_path)
        if m is None:
            return
        ai = fc.array_index
        if not (0 <= ai < 3):
            return
        vi = int(m.group(1))
        if vi not in pinned:
            return
        lookup[(vi, ai)] = fc
        for kp in fc.keyframe_points:
            f = int(round(kp.co[0]))
            if f >= 1:
                frames.add(f)

    if hasattr(action, "layers") and len(action.layers) > 0:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    cb = strip.channelbag(slot)
                    if cb is None:
                        continue
                    for fc in cb.fcurves:
                        consume(fc)
    elif hasattr(action, "fcurves"):
        for fc in action.fcurves:
            consume(fc)

    return sorted(frames), lookup


def _encode_pin_config(context, groups, state, fps=None):
    """Encode pin vertex group config and animation.

    ``fps`` is resolved once by ``_build_param_dict`` and threaded in so the
    whole param build shares a single frame rate; callers that omit it (e.g.
    debug scenarios) fall back to resolving it here.
    """
    if fps is None:
        fps = resolve_fps(state)
    # Collect per-object pin config (duration, pull, operations) keyed by vertex index
    # Structure: {obj_name: {vertex_index: {"unpin_time": ..., "pull_strength": ..., "operations": [...]}}}
    pin_config = {}
    for group in groups:
        if group.object_type == "STATIC":
            continue
        for pin_item in group.pin_vertex_groups:
            has_operations = len(pin_item.operations) > 0
            # EMBEDDED_MOVE (the fcurve-keyframed-pin sentinel) lives
            # in ``operations``, so ``has_operations`` already covers
            # the keyframed-pin case alongside Move/Spin/Scale/Torque.
            # SOLID is never skipped on the plain-pin condition: a static
            # (no duration / no pull / no ops) hard-intent SOLID pin must still
            # emit a cfg so it can carry the fix_weight_threshold toggle to the
            # decoder. Non-SOLID keeps the original skip.
            if (group.object_type != "SOLID"
                    and not pin_item.use_pin_duration and not pin_item.use_pull
                    and not has_operations):
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
            has_embedded_move = any(
                op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations
            )
            if has_embedded_move and obj.type != "MESH":
                raise ValueError(
                    f"Pin '{vg_name}' on '{obj_name}': keyframed pin "
                    "animation is only supported for mesh objects"
                )
            # Keyframed pin animation composes cleanly with Move/Spin/
            # Scale via ``embedded_move_index`` (the decoder splices
            # the per-vertex pin_anim track in among the explicit
            # ops). Only Torque is genuinely incompatible, since it
            # shares the per-pin force-vs-kinematic switch the
            # kinematic ops use.
            if has_embedded_move and has_operations:
                if any(op.op_type == "TORQUE" for op in pin_item.operations):
                    raise ValueError(
                        f"Pin '{vg_name}' on '{obj_name}': keyframed pin "
                        "animation cannot be combined with Torque operations"
                    )
            cfg = {}
            if pin_item.use_pin_duration:
                cfg["unpin_time"] = float(pin_item.pin_duration) / fps
            if pin_item.use_pull:
                cfg["pull_strength"] = float(pin_item.pull_strength)
            # A captured deformation on a SOLID group can drive a time-varying
            # rest shape so the dynamic body settles into the motion instead of
            # fighting it. Applies to BOTH pin modes - a pull pin (weak soft
            # pull) and a fixed pin (strong FixPair barrier) track the same rest
            # shape; only the pin constraint differs. This is OPT-IN via the
            # per-pin "Track Rest-Pose Deformation" toggle. Gated on a FULL pin
            # (every vertex in the group): with the whole mesh captured, the
            # rest pose is the captured deformation itself (every sim vertex is
            # driven), so there is no partial-pin reconstruction and no boundary
            # tear. SOLID only.
            if (getattr(pin_item, "track_rest_pose_deformation", False)
                    and getattr(pin_item, "has_captured_anim", False)
                    and group.object_type == "SOLID"
                    and pin_covers_all_vertices(obj, vg_name)):
                cfg["rest_shape_track"] = True
            # Per-pin stiffness scale for the moving (kinematic) pin
            # constraint force. Always emitted; the solver applies it
            # only when the pin is kinematic. Defaults to 1.0 server-side
            # when absent (older payloads).
            cfg["pin_stiffness"] = float(pin_item.pin_stiffness)
            if has_operations:
                ops_list = []
                # Centroid for CENTROID-mode spin/scale: frame-1 vertex
                # positions in the no-translation solver frame.
                centroid_blender = None
                pin_indices = _get_pin_indices(obj, vg_name)
                if pin_indices:
                    mat = _solver_rot_matrix(obj)
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
                # Object's solver-space translation (displacement).
                # Subtract from absolute centers so they match the solver's
                # untranslated vertex space.
                disp = world_matrix(obj).to_translation()
                for op in pin_item.operations:
                    # EMBEDDED_MOVE is the sentinel signaling
                    # keyframe animation; it has no runtime semantics
                    # itself, the per-vertex pin_anim track is what
                    # the solver actually consumes.
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
                                    "Spin center vertex not set: pick a vertex in Edit Mode"
                                )
                            op_dict["center_mode"] = "absolute"
                            mat_nt = _solver_rot_matrix(obj)
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
                                    "Scale center vertex not set: pick a vertex in Edit Mode"
                                )
                            op_dict["center_mode"] = "absolute"
                            mat_nt = _solver_rot_matrix(obj)
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
            # Key by UUID for rename resilience.
            if obj_uuid not in pin_config:
                pin_config[obj_uuid] = {}
            # Keyframed pin animation. Three motion sources can supply
            # ``pin_anim``:
            #
            # 1. Captured depsgraph cache (``_pindeform.pc2``) written
            #    by Capture Deformation. Dense one-sample-per-frame
            #    coverage for armature / lattice / mesh-deform / shape-
            #    key driven pins. PC2 wins over fcurves: when both
            #    exist on the same pin, the cache is the live source.
            #
            # 2. Sparse vertex-co fcurves on the mesh's action, authored
            #    via Make Keyframe. The sentinel is an EMBEDDED_MOVE op
            #    in ``pin_item.operations``; its presence tells the
            #    encoder to gather per-vertex pin_anim from the live
            #    action and splice it in at ``embedded_move_index``.
            #    Pub-main mirrors keyframes through
            #    ``state.saved_pin_keyframes`` first which is an O(N^2)
            #    walk; we read fcurves directly to avoid that.
            #
            # STATIC mesh colliders use a separate dense path
            # (``_staticdeform.pc2`` + ``encoder/mesh.py``) and never
            # show up here.
            vert_tracks = {}
            embedded_move_count = sum(
                1 for op in pin_item.operations
                if op.op_type == "EMBEDDED_MOVE"
            )
            if embedded_move_count > 1:
                raise ValueError(
                    f"Pin '{vg_name}' on '{obj_name}': more than one "
                    "EMBEDDED_MOVE operation is not supported"
                )

            # PC2-wins: check the captured cache first. The
            # ``has_captured_anim`` flag is authoritative when set
            # (capture finalize sets it), but we also probe the cache
            # directly in case a .blend was opened without the
            # load_post reconciler having had a chance to run yet.
            from ..pc2 import get_pin_anim_cache
            pin_cache = (
                get_pin_anim_cache(obj, vg_name)
                if obj.type == "MESH" else None
            )
            if pin_cache is not None and pin_cache.shape[0] >= 2:
                live_pin_indices = _get_pin_indices(obj, vg_name)
                if pin_cache.shape[1] != len(live_pin_indices):
                    raise ValueError(
                        f"Pin '{vg_name}' on '{obj_name}': captured "
                        f"deformation cache has {pin_cache.shape[1]} "
                        f"vertices but the live pin has "
                        f"{len(live_pin_indices)}. The vertex group "
                        "changed since capture; press Clear Deformation "
                        "Cache and Capture Deformation again."
                    )
                n_frames_cache = pin_cache.shape[0]
                cache_frame_start = int(bpy.context.scene.frame_start)
                cache_times = [
                    (cache_frame_start + k - 1) / fps
                    for k in range(n_frames_cache)
                ]
                # Cache stores positions in world solver space
                # (zup_to_yup @ matrix_world @ co_local). The decoder
                # only uses consecutive deltas, so absolute frame and
                # translation cancel; any consistent space works.
                pin_idx_array = np.asarray(live_pin_indices, dtype=np.int64)
                for j, vi in enumerate(pin_idx_array):
                    vert_tracks[int(vi)] = {
                        "time": cache_times,
                        "position": np.ascontiguousarray(
                            pin_cache[:, j, :].astype(np.float32, copy=False)
                        ),
                    }
                cfg["embedded_move_index"] = 0
                # Skip the fcurve scan entirely; PC2 owns the track.
                fcurve_frames, fcurve_lookup = [], {}
            else:
                has_embedded_move_fcurves = (
                    embedded_move_count == 1 and obj.type == "MESH"
                )
                fcurve_frames, fcurve_lookup = (
                    _collect_pin_vertex_fcurve_frames(obj, vg_name)
                    if has_embedded_move_fcurves else ([], {})
                )
            if fcurve_frames:
                mat = world_matrix(obj).to_3x3()
                rot = np.array(
                    [[mat[r][c] for c in range(3)] for r in range(3)],
                    dtype=np.float32,
                )
                # One (time, position) sample per authored fcurve
                # keyframe frame, indexed by vertex. Axes the fcurve
                # doesn't cover sit at the mesh's rest position,
                # mirroring native Blender playback.
                n_verts_total = len(obj.data.vertices)
                rest_co = np.empty(n_verts_total * 3, dtype=np.float32)
                obj.data.vertices.foreach_get("co", rest_co)
                rest_pose = rest_co.reshape(n_verts_total, 3)
                times = [(f - 1) / fps for f in fcurve_frames]
                pose_stack = np.broadcast_to(
                    rest_pose, (len(fcurve_frames), n_verts_total, 3),
                ).copy()
                for (vi, ai), fc in fcurve_lookup.items():
                    if 0 <= vi < n_verts_total:
                        for k, fr in enumerate(fcurve_frames):
                            pose_stack[k, vi, ai] = fc.evaluate(fr)
                transformed = pose_stack @ rot.T
                for vi in _get_pin_indices(obj, vg_name):
                    if 0 <= vi < n_verts_total:
                        vert_tracks[vi] = {
                            "time": times,
                            "position": np.ascontiguousarray(
                                transformed[:, vi, :]
                            ),
                        }
                # ``embedded_move_index = 0``: Make Keyframe pins the
                # EMBEDDED_MOVE op to slot 0 of ``pin_item.operations``,
                # so after the encode loop skips it the pin_anim track
                # wants slot 0 of the emitted ``operations`` list.
                cfg["embedded_move_index"] = 0
            # Tag with group identity so solver can merge torque vertices
            cfg["pin_group_id"] = f"{obj_uuid}:{vg_name}"
            cfg["obj_uuid"] = obj_uuid
            # SOLID hard-pin surface/soft split threshold (per pin). The
            # decoder always splits a hard-intent partial-pin SOLID holder
            # (interior fix pins crash the solver); this scalar only sets how
            # much of the SURFACE is hard vs soft skirt. 0 = whole pinned
            # surface region hard. Irrelevant to pull pins / non-SOLID.
            if group.object_type == "SOLID":
                cfg["fix_weight_threshold"] = float(
                    getattr(pin_item, "fix_weight_threshold", 0.5)
                )
            # For curves we must match the sampled-vertex index space used
            # by ``info["pin"]`` in encoder/mesh.py (the solver's
            # ``pin_holder.index``). Otherwise spin/move_by configs are
            # keyed at control-point indices and the frontend's lookup
            # ``obj_cfg.get(sampled_idx)`` misses them.
            if obj.type == "CURVE":
                from ..curve_rod import map_cp_pins_to_sampled
                config_indices = map_cp_pins_to_sampled(
                    obj, _get_pin_indices(obj, vg_name)
                )
            else:
                config_indices = _get_pin_indices(obj, vg_name)
            for vi in config_indices:
                if vi in vert_tracks:
                    # Shared group fields + this vertex's own track.
                    pin_config[obj_uuid][vi] = {
                        **cfg, "pin_anim": {vi: vert_tracks[vi]},
                    }
                else:
                    pin_config[obj_uuid][vi] = cfg

    return pin_config
