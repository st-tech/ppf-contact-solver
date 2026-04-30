# File: bake_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import time

import bpy  # pyright: ignore
import numpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ...core.client import communicator as com
from ...core.derived import is_server_busy_from_response as is_running
from ...core.pc2 import (
    _curve_cache,
    cleanup_mesh_cache,
    get_pc2_path,
    has_mesh_cache,
    load_curve_cache,
    object_pc2_key,
    read_pc2_frame,
    read_pc2_frame_count,
    read_pc2_n_verts,
)
from ...core.utils import _get_fcurves, redraw_all_areas, set_linear_interpolation
from ...models.collection_utils import safe_update_index
from ..state import iterate_active_object_groups
from .utils import cleanup_pin_vertex_groups_for_object, get_group_from_index, reset_object_display


# ---------------------------------------------------------------------------
# Job state (module-level singleton — read by UI panels, written by modal ops)
# ---------------------------------------------------------------------------

_bake_job: dict = {
    "active": False,
    "aborted": False,
    "kind": "",            # "single" (per-object) or "all"
    "objects": [],         # list of per-object entry dicts
    "object_cursor": 0,    # index into objects of the in-flight entry
    "total_frames_processed": 0,
    "total_frames": 0,
    "status": "",
}


def _reset_job() -> None:
    _bake_job.update({
        "active": False,
        "aborted": False,
        "kind": "",
        "objects": [],
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": 0,
        "status": "",
    })


def is_bake_running() -> bool:
    """True while a modal Bake Animation job is still processing frames."""
    return bool(_bake_job.get("active"))


def bake_progress_snapshot() -> tuple[int, int, str, int]:
    """(done_frames, total_frames, status_line, object_count) for the UI."""
    return (
        int(_bake_job.get("total_frames_processed", 0)),
        int(_bake_job.get("total_frames", 0)),
        str(_bake_job.get("status", "")),
        len(_bake_job.get("objects", [])),
    )


# ---------------------------------------------------------------------------
# Bakeability detection
# ---------------------------------------------------------------------------


def _is_bakeable(obj) -> bool:
    """True when *obj* has PC2 simulation animation that can be baked."""
    return obj is not None and obj.type in ("MESH", "CURVE") and has_mesh_cache(obj)


def _has_animated_objects(context) -> bool:
    """Poll helper: True when at least one dynamic object has animation."""
    if com.busy() or com.animation.frame:
        return False
    response = com.info.response
    if is_running(response):
        return False
    # STATIC groups are included: UI move/spin/scale ops produce PC2 +
    # ContactSolverCache too, and those modifiers must be cleaned up by
    # Bake Animation just like SOLID/SHELL/ROD ones.
    for group in iterate_active_object_groups(context.scene):
        for assigned in group.assigned_objects:
            from ...core.uuid_registry import resolve_assigned
            if _is_bakeable(resolve_assigned(assigned)):
                return True
    return False


# ---------------------------------------------------------------------------
# Curve pose capture/apply (Bake Single Frame path)
# ---------------------------------------------------------------------------


def _capture_curve_pose(curve_data):
    """Snapshot current CV positions for a CURVE's data-block."""
    snapshot = []
    for spline in curve_data.splines:
        if spline.type == "BEZIER":
            snapshot.append((
                "BEZIER",
                [
                    (bp.co.copy(), bp.handle_left.copy(), bp.handle_right.copy())
                    for bp in spline.bezier_points
                ],
            ))
        else:
            snapshot.append((
                "POLY",
                [tuple(pt.co) for pt in spline.points],
            ))
    return snapshot


def _apply_curve_pose(curve_data, snapshot):
    """Write a :func:`_capture_curve_pose` result back onto *curve_data*."""
    for spline, (kind, points) in zip(curve_data.splines, snapshot):
        if kind == "BEZIER" and spline.type == "BEZIER":
            for bp, (co, hl, hr) in zip(spline.bezier_points, points):
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"
                bp.co = co
                bp.handle_left = hl
                bp.handle_right = hr
        elif kind == "POLY" and spline.type != "BEZIER":
            for pt, co in zip(spline.points, points):
                pt.co[0] = co[0]
                pt.co[1] = co[1]
                pt.co[2] = co[2]


# ---------------------------------------------------------------------------
# Pre-bake snapshots for safe abort
# ---------------------------------------------------------------------------


def _snapshot_fcurves(obj) -> set:
    """Set of ``(data_path, array_index)`` for fcurves that exist on
    ``obj.data`` before the bake — used to discriminate newly-inserted
    fcurves during abort."""
    keys = set()
    if obj.data.animation_data and obj.data.animation_data.action:
        for fc in _get_fcurves(obj.data.animation_data.action):
            keys.add((fc.data_path, fc.array_index))
    return keys


def _snapshot_bezier_handles(obj):
    """Per-spline list of ``(handle_left_type, handle_right_type)`` tuples
    for every bezier point. Non-bezier splines yield ``None``. Returns
    ``None`` when *obj* isn't a CURVE."""
    if obj.type != "CURVE":
        return None
    snapshot = []
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            snapshot.append([
                (bp.handle_left_type, bp.handle_right_type)
                for bp in spline.bezier_points
            ])
        else:
            snapshot.append(None)
    return snapshot


def _restore_bezier_handles(obj, snapshot) -> None:
    if obj.type != "CURVE" or snapshot is None:
        return
    for spline, per_spline in zip(obj.data.splines, snapshot):
        if spline.type != "BEZIER" or per_spline is None:
            continue
        for bp, (left, right) in zip(spline.bezier_points, per_spline):
            bp.handle_left_type = left
            bp.handle_right_type = right


def _force_bezier_free(obj) -> None:
    """Pin all bezier handle types to FREE so keyframe_insert captures
    the values we write directly (AUTO handles would be recomputed)."""
    if obj.type != "CURVE":
        return
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"


# ---------------------------------------------------------------------------
# Queue builders
# ---------------------------------------------------------------------------


def _make_entry(obj, group_index: int, assigned_uuid: str, assigned_name: str):
    """Build one bake-queue entry.

    Always returns an entry when the object is :func:`_is_bakeable` — a
    stale modifier with no readable PC2 still needs finalize to strip it.
    When the cache is unreadable or the vertex count doesn't match, the
    entry is queued with ``n_frames == 0`` so ``_tick_job`` skips
    keyframing and ``_finalize_job`` just cleans up the modifier + file.
    """
    from ...core.uuid_registry import get_or_create_object_uuid

    key = object_pc2_key(obj)
    path = get_pc2_path(key)

    entry = {
        "obj_uuid": get_or_create_object_uuid(obj),
        "obj_type": obj.type,
        "obj_name": assigned_name or obj.name,
        "key": key,
        "path": path,
        "n_frames": 0,
        "group_index": group_index,
        "assigned_uuid": assigned_uuid,
        "frames_done": 0,
    }

    pc2_exists = os.path.exists(path)
    pc2_n_frames = read_pc2_frame_count(path) if pc2_exists else 0

    if obj.type == "MESH":
        sk_block = obj.data.shape_keys
        entry["pre_sk_block_existed"] = sk_block is not None
        entry["pre_sk_names"] = (
            {kb.name for kb in sk_block.key_blocks} if sk_block is not None else set()
        )
        entry["added_sk_names"] = []
        entry["sk_prefix"] = "ContactSolverBake"
        if pc2_n_frames > 0:
            n_verts = read_pc2_n_verts(path)
            if len(obj.data.vertices) == n_verts:
                entry["n_verts"] = n_verts
                entry["n_frames"] = pc2_n_frames
    else:
        entry["pre_fcurves"] = _snapshot_fcurves(obj)
        entry["handle_snapshot"] = _snapshot_bezier_handles(obj)
        if pc2_n_frames > 0:
            if key not in _curve_cache:
                load_curve_cache(key)
            cache = _curve_cache.get(key)
            if cache is not None and cache.shape[0] > 0:
                entry["cache"] = cache
                entry["n_frames"] = int(cache.shape[0])
    return entry


def _build_queue(context, *, group_index: int | None = None) -> list:
    """Return job entries for every assigned MESH/CURVE in scope. The
    "all groups" case (``group_index is None``) queues every assigned
    object in every active group — STATIC colliders and fcurve-driven
    statics included — so finalize can purge them all from their groups
    as a terminal Bake step. Entries without a readable cache get
    ``n_frames == 0`` and skip keyframing but still reach finalize for
    modifier + group cleanup."""
    from ...core.uuid_registry import resolve_assigned
    queue: list = []

    if group_index is not None:
        group = get_group_from_index(context.scene, group_index)
        if group is None:
            return queue
        idx = group.assigned_objects_index
        if 0 <= idx < len(group.assigned_objects):
            assigned = group.assigned_objects[idx]
            obj = resolve_assigned(assigned)
            if obj is not None and obj.type in ("MESH", "CURVE"):
                queue.append(_make_entry(obj, group_index, assigned.uuid, assigned.name))
        return queue

    from ...models.groups import get_addon_data
    from ..state import N_MAX_GROUPS
    addon_data = get_addon_data(context.scene)
    for gi in range(N_MAX_GROUPS):
        prop_name = f"object_group_{gi}"
        group = getattr(addon_data, prop_name, None)
        if group is None or not group.active:
            continue
        for assigned in group.assigned_objects:
            obj = resolve_assigned(assigned)
            if obj is None or obj.type not in ("MESH", "CURVE"):
                continue
            queue.append(_make_entry(obj, gi, assigned.uuid, assigned.name))
    return queue


# ---------------------------------------------------------------------------
# Pre-bake validation
# ---------------------------------------------------------------------------


def _check_mesh_shape_keys(queue: list) -> list[str]:
    """Return blocker lines for MESH entries that already carry non-Basis
    shape keys. Baking absolute poses into additional shape keys would
    double-blend with the user's existing keys, so refuse and ask them
    to remove extras manually."""
    from ...core.uuid_registry import get_object_by_uuid
    blockers: list[str] = []
    for entry in queue:
        if entry["obj_type"] != "MESH":
            continue
        # Entries with n_frames == 0 only need a modifier sweep — we
        # aren't adding shape keys for them, so existing non-Basis keys
        # can't double-blend against anything.
        if entry.get("n_frames", 0) <= 0:
            continue
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        sk_block = obj.data.shape_keys
        if sk_block is None:
            continue
        # Index 0 is always the Basis (reference_key) in Blender.
        extras = [kb.name for kb in list(sk_block.key_blocks)[1:]]
        if extras:
            preview = ", ".join(extras[:3])
            if len(extras) > 3:
                preview += f" (+{len(extras) - 3} more)"
            blockers.append(f"{entry['obj_name']} [{preview}]")
    return blockers


# ---------------------------------------------------------------------------
# Job lifecycle: start, tick, finalize, abort
# ---------------------------------------------------------------------------


def _start_job(queue: list, kind: str) -> tuple[bool, str]:
    if _bake_job["active"]:
        return False, "Another bake job is in progress"
    if not queue:
        return False, "No bakeable objects"

    from ...core.uuid_registry import get_object_by_uuid
    for entry in queue:
        if entry["obj_type"] == "CURVE":
            obj = get_object_by_uuid(entry["obj_uuid"])
            if obj is not None:
                _force_bezier_free(obj)

    _bake_job.update({
        "active": True,
        "aborted": False,
        "kind": kind,
        "objects": queue,
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": sum(e["n_frames"] for e in queue),
        "status": "Baking...",
    })
    return True, ""


def _sk_value_fcurve(shape_keys, sk_name):
    """Find the fcurve driving ``key_blocks["<sk_name>"].value`` on the
    shape-keys action, or return ``None``."""
    if shape_keys is None or shape_keys.animation_data is None:
        return None
    action = shape_keys.animation_data.action
    if action is None:
        return None
    target = f'key_blocks["{sk_name}"].value'
    for fc in _get_fcurves(action):
        if fc.data_path == target:
            return fc
    return None


def _process_one_frame_mesh(entry, obj) -> None:
    """MESH: one shape key per PC2 frame. Positions via foreach_set
    (native C loop — orders of magnitude faster than per-vertex
    keyframe_insert). Value is keyed 0→1→0 with CONSTANT interpolation
    so each shape key is "on" for exactly its frame."""
    f = entry["frames_done"]
    scene_frame = f + 1
    n_verts = entry["n_verts"]
    positions = read_pc2_frame(entry["path"], f, n_verts)
    flat = numpy.ascontiguousarray(positions, dtype=numpy.float32).ravel()

    sk_name = f"{entry['sk_prefix']}_{scene_frame:05d}"
    kb = obj.shape_key_add(name=sk_name, from_mix=False)
    kb.data.foreach_set("co", flat)
    entry["added_sk_names"].append(kb.name)

    kb.value = 0.0
    kb.keyframe_insert("value", frame=scene_frame - 1)
    kb.value = 1.0
    kb.keyframe_insert("value", frame=scene_frame)
    kb.value = 0.0
    kb.keyframe_insert("value", frame=scene_frame + 1)

    fc = _sk_value_fcurve(obj.data.shape_keys, kb.name)
    if fc is not None:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def _process_one_frame_curve(entry, obj) -> None:
    """CURVE: per-CV keyframes on bezier_points/points. CV counts are
    small so the per-point keyframe_insert cost is tolerable."""
    f = entry["frames_done"]
    frame = f + 1
    frame_data = entry["cache"][f]
    cv_i = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left = frame_data[cv_i]
                cv_i += 1
                bp.co = frame_data[cv_i]
                cv_i += 1
                bp.handle_right = frame_data[cv_i]
                cv_i += 1
                bp.keyframe_insert("co", frame=frame)
                bp.keyframe_insert("handle_left", frame=frame)
                bp.keyframe_insert("handle_right", frame=frame)
        else:
            for pt in spline.points:
                c = frame_data[cv_i]
                cv_i += 1
                pt.co[0] = float(c[0])
                pt.co[1] = float(c[1])
                pt.co[2] = float(c[2])
                pt.keyframe_insert("co", frame=frame)


def _process_one_frame(entry) -> bool:
    """Process one PC2 frame of *entry*'s object. Returns ``False`` if
    the object is no longer resolvable (skip it)."""
    from ...core.uuid_registry import get_object_by_uuid
    obj = get_object_by_uuid(entry["obj_uuid"])
    if obj is None:
        return False
    if entry["obj_type"] == "MESH":
        _process_one_frame_mesh(entry, obj)
    else:
        _process_one_frame_curve(entry, obj)
    entry["frames_done"] += 1
    return True


def _tick_job(*, budget_ms: int = 40) -> bool:
    """Run work until the per-tick budget expires. Returns ``True`` if
    more frames remain after the tick, ``False`` when keyframing is
    complete (or aborted)."""
    deadline = time.perf_counter() + budget_ms / 1000.0
    objects = _bake_job["objects"]
    while time.perf_counter() < deadline:
        if _bake_job["aborted"]:
            return False
        idx = _bake_job["object_cursor"]
        if idx >= len(objects):
            return False
        entry = objects[idx]
        if entry["frames_done"] >= entry["n_frames"]:
            _bake_job["object_cursor"] += 1
            continue
        _bake_job["status"] = (
            f"Baking {entry['obj_name']} "
            f"[{entry['frames_done'] + 1}/{entry['n_frames']}]"
        )
        if not _process_one_frame(entry):
            _bake_job["object_cursor"] += 1
            continue
        _bake_job["total_frames_processed"] += 1
    return _bake_job["object_cursor"] < len(_bake_job["objects"])


def _finalize_job(context) -> tuple[int, int]:
    """After all keyframes are in, drop PC2 + ContactSolverCache modifier
    and remove baked objects from their groups. Returns
    ``(objects_baked, total_frames_baked)``."""
    from ...core.uuid_registry import get_object_by_uuid
    from ...models.groups import get_addon_data

    baked_frames = _bake_job["total_frames_processed"]
    objects_baked = 0
    uuids_per_group: dict[int, set] = {}
    for entry in _bake_job["objects"]:
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        # Always drop the ContactSolverCache modifier + PC2 file for every
        # queued entry, even if frames_done is 0 (a stale modifier with no
        # readable PC2 file can otherwise survive the bake). cleanup is
        # idempotent — safe if nothing is there.
        try:
            cleanup_mesh_cache(
                obj,
                keep_baked_pose=(entry["obj_type"] == "CURVE"),
            )
            reset_object_display(obj)
            if entry["obj_type"] == "CURVE":
                if obj.data.animation_data and obj.data.animation_data.action:
                    set_linear_interpolation(obj.data.animation_data.action)
        except Exception:
            pass
        # Always remove from the group — Bake Animation is the terminal
        # group step. Colliders and fcurve-driven statics that had nothing
        # to keyframe still get evicted so groups are empty afterwards.
        if entry["frames_done"] > 0:
            objects_baked += 1
        uuids_per_group.setdefault(entry["group_index"], set()).add(entry["assigned_uuid"])

    addon_data = get_addon_data(context.scene)
    for gi, uuids in uuids_per_group.items():
        prop_name = f"object_group_{gi}"
        group = getattr(addon_data, prop_name, None)
        if group is None:
            continue
        i = len(group.assigned_objects) - 1
        while i >= 0:
            if group.assigned_objects[i].uuid in uuids:
                cleanup_pin_vertex_groups_for_object(group, group.assigned_objects[i].uuid)
                group.assigned_objects.remove(i)
            i -= 1
        group.assigned_objects_index = safe_update_index(
            group.assigned_objects_index, len(group.assigned_objects)
        )
    return objects_baked, baked_frames


def _abort_job() -> None:
    """Undo whatever was written so far, leaving PC2 files, MESH_CACHE
    modifiers, and group membership intact so the pre-bake state is
    fully recoverable. For MESH: remove shape keys we added (plus Basis
    if we created the block) and their value fcurves. For CURVE: remove
    inserted per-CV fcurves and restore bezier handle types."""
    from ...core.uuid_registry import get_object_by_uuid
    for entry in _bake_job["objects"]:
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        if entry["obj_type"] == "MESH":
            _abort_mesh_entry(obj, entry)
        else:
            _abort_curve_entry(obj, entry)


def _abort_mesh_entry(obj, entry) -> None:
    sk_block = obj.data.shape_keys
    if sk_block is None:
        return
    # Remove the value-channel fcurves we added, matching by data_path
    # (per-shape-key, so we know exactly which to strip).
    added_paths = {
        f'key_blocks["{name}"].value' for name in entry.get("added_sk_names", [])
    }
    action = sk_block.animation_data.action if sk_block.animation_data else None
    if action is not None and added_paths:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    to_remove = [fc for fc in bag.fcurves if fc.data_path in added_paths]
                    for fc in to_remove:
                        bag.fcurves.remove(fc)
    # Remove the shape keys themselves. Non-Basis first so Basis can be
    # removed cleanly afterwards if we were the ones who created it.
    for name in list(entry.get("added_sk_names", [])):
        kb = sk_block.key_blocks.get(name)
        if kb is not None:
            obj.shape_key_remove(kb)
    if not entry.get("pre_sk_block_existed", False) and obj.data.shape_keys is not None:
        remaining = [kb for kb in obj.data.shape_keys.key_blocks]
        for kb in remaining:
            obj.shape_key_remove(kb)


def _abort_curve_entry(obj, entry) -> None:
    _restore_bezier_handles(obj, entry.get("handle_snapshot"))
    action = obj.data.animation_data.action if obj.data.animation_data else None
    if action is None:
        return
    pre = entry.get("pre_fcurves", set())
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                to_remove = [
                    fc for fc in bag.fcurves
                    if (fc.data_path, fc.array_index) not in pre
                ]
                for fc in to_remove:
                    bag.fcurves.remove(fc)


# ---------------------------------------------------------------------------
# Modal operator mixin (shared between per-object and all-object bakes)
# ---------------------------------------------------------------------------


class _ModalBakeBase:
    """Shared modal loop. Subclasses provide ``_kind`` and
    ``_build_queue_for``; everything else here is reused."""

    _kind = ""

    def _build_queue_for(self, context) -> list:
        raise NotImplementedError

    def execute(self, context):
        queue = self._build_queue_for(context)
        blockers = _check_mesh_shape_keys(queue)
        if blockers:
            joined = "; ".join(blockers)
            self.report(
                {"ERROR"},
                "Remove all shape keys except Basis before baking — "
                f"conflicts on: {joined}",
            )
            return {"CANCELLED"}
        ok, err = _start_job(queue, kind=self._kind)
        if not ok:
            self.report({"WARNING"}, err)
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        redraw_all_areas(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if _bake_job["aborted"]:
            _abort_job()
            self._teardown(context)
            redraw_all_areas(context)
            self.report({"INFO"}, "Bake aborted")
            return {"CANCELLED"}
        more = _tick_job()
        redraw_all_areas(context)
        if not more and not _bake_job["aborted"]:
            n_objs, n_frames = _finalize_job(context)
            self._teardown(context)
            from .overlay import apply_object_overlays
            apply_object_overlays()
            redraw_all_areas(context)
            self.report(
                {"INFO"},
                f"Baked {n_frames} frame(s) for {n_objs} object(s)",
            )
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if _bake_job["active"]:
            _abort_job()
        self._teardown(context)
        redraw_all_areas(context)

    def _teardown(self, context):
        wm = context.window_manager
        timer = getattr(self, "_timer", None)
        if timer is not None:
            wm.event_timer_remove(timer)
            self._timer = None
        _reset_job()


# ---------------------------------------------------------------------------
# Per-object Bake Animation (modal with progress)
# ---------------------------------------------------------------------------


class OBJECT_OT_BakeAnimation(_ModalBakeBase, Operator):
    """Bake animation for the selected object and remove it from the group"""

    bl_idname = "object.bake_animation"
    bl_label = "Bake Animation"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    _kind = "single"

    @classmethod
    def poll(cls, context):
        return not _bake_job["active"]

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def _build_queue_for(self, context) -> list:
        return _build_queue(context, group_index=self.group_index)


# ---------------------------------------------------------------------------
# Per-object Bake Single Frame (stays synchronous — cheap, no hang risk)
# ---------------------------------------------------------------------------


class OBJECT_OT_BakeSingleFrame(Operator):
    """Bake current frame as Frame 1 and remove object from the group"""

    bl_idname = "object.bake_single_frame"
    bl_label = "Bake Single Frame"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _bake_job["active"]

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        index = group.assigned_objects_index
        if not (0 <= index < len(group.assigned_objects)):
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}

        assigned = group.assigned_objects[index]
        from ...core.uuid_registry import resolve_assigned
        obj = resolve_assigned(assigned)
        obj_name = assigned.name
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, f"Object '{obj_name}' not found")
            return {"CANCELLED"}

        if not _is_bakeable(obj):
            self.report({"WARNING"}, f"No animation data on '{obj_name}'")
            return {"CANCELLED"}

        data = obj.data
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)

        if obj.type == "CURVE":
            snapshot = _capture_curve_pose(eval_obj.data)
            cleanup_mesh_cache(obj, keep_baked_pose=True)
            scene.frame_set(1)
            _apply_curve_pose(data, snapshot)
        else:
            current_positions = [v.co.copy() for v in eval_obj.data.vertices]
            cleanup_mesh_cache(obj)
            scene.frame_set(1)
            for i, pos in enumerate(current_positions):
                data.vertices[i].co = pos

        reset_object_display(obj)
        group.assigned_objects.remove(index)
        cleanup_pin_vertex_groups_for_object(group, assigned.uuid)
        group.assigned_objects_index = safe_update_index(index, len(group.assigned_objects))

        apply_object_overlays()
        self.report({"INFO"}, f"Baked frame {scene.frame_current} as Frame 1 for '{obj_name}'")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# All-object Bake Animation (modal with progress)
# ---------------------------------------------------------------------------


class SOLVER_OT_BakeAllAnimation(_ModalBakeBase, Operator):
    """Bake animation for all objects and remove them from their groups"""

    bl_idname = "solver.bake_all_animation"
    bl_label = "Bake Animation"
    bl_options = {"UNDO"}
    _kind = "all"

    @classmethod
    def poll(cls, context):
        if _bake_job["active"]:
            return False
        return _has_animated_objects(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def _build_queue_for(self, context) -> list:
        return _build_queue(context, group_index=None)


# ---------------------------------------------------------------------------
# All-object Bake Single Frame (synchronous — cheap)
# ---------------------------------------------------------------------------


class SOLVER_OT_BakeAllSingleFrame(Operator):
    """Bake current frame as Frame 1 for all objects and remove them from their groups"""

    bl_idname = "solver.bake_all_single_frame"
    bl_label = "Bake Single Frame"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        if _bake_job["active"]:
            return False
        return _has_animated_objects(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        depsgraph = context.evaluated_depsgraph_get()

        # Phase 1: capture current evaluated pose for every bakeable object.
        # STATIC groups are included — UI-op motion produces PC2 caches
        # that must be cleaned up with the same teardown path.
        captured = {}
        failed = []
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                from ...core.uuid_registry import resolve_assigned
                obj = resolve_assigned(assigned)
                uid = assigned.uuid
                if not uid:
                    failed.append(assigned.name or "<unknown>")
                    continue
                if uid in captured:
                    continue
                if not obj:
                    failed.append(assigned.name or uid)
                    continue
                if not _is_bakeable(obj):
                    continue
                eval_obj = obj.evaluated_get(depsgraph)
                if obj.type == "MESH":
                    captured[uid] = ("MESH", [v.co.copy() for v in eval_obj.data.vertices])
                else:
                    captured[uid] = ("CURVE", _capture_curve_pose(eval_obj.data))

        # Phase 2: teardown + apply captured pose at frame 1.
        from ...core.uuid_registry import resolve_assigned as _resolve2
        scene.frame_set(1)
        count = 0
        for group in iterate_active_object_groups(scene):
            i = len(group.assigned_objects) - 1
            while i >= 0:
                assigned_item = group.assigned_objects[i]
                obj = _resolve2(assigned_item)
                uid = assigned_item.uuid
                if obj and uid and uid in captured:
                    cap_type, cap_data = captured[uid]
                    data = obj.data
                    if cap_type == "MESH":
                        cleanup_mesh_cache(obj)
                        for vi, pos in enumerate(cap_data):
                            data.vertices[vi].co = pos
                    else:
                        cleanup_mesh_cache(obj, keep_baked_pose=True)
                        _apply_curve_pose(data, cap_data)
                    reset_object_display(obj)
                    group.assigned_objects.remove(i)
                    cleanup_pin_vertex_groups_for_object(group, assigned_item.uuid)
                    count += 1
                i -= 1
            group.assigned_objects_index = safe_update_index(
                group.assigned_objects_index, len(group.assigned_objects)
            )

        apply_object_overlays()
        if failed:
            self.report({"ERROR"}, f"UUID resolve failed for: {', '.join(failed)}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked single frame for {count} object(s)")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Abort button (reads/writes the module-level job flag)
# ---------------------------------------------------------------------------


class SOLVER_OT_BakeAbort(Operator):
    """Abort the running bake and revert any partially-inserted keyframes"""

    bl_idname = "solver.bake_abort"
    bl_label = "Abort"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(_bake_job.get("active")) and not _bake_job.get("aborted")

    def execute(self, context):
        _bake_job["aborted"] = True
        _bake_job["status"] = "Aborting..."
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    OBJECT_OT_BakeAnimation,
    OBJECT_OT_BakeSingleFrame,
    SOLVER_OT_BakeAllAnimation,
    SOLVER_OT_BakeAllSingleFrame,
    SOLVER_OT_BakeAbort,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
