# File: pin_capture_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Deformation for SHELL/SOLID/ROD pins. Walks the scene frame
# range on a modal timer, samples the depsgraph-evaluated mesh per
# frame, extracts only the pinned vertices, and stores them as a
# ``_pindeform.pc2`` cache. The encoder's pin path consumes this cache
# in preference to the manual Make-Keyframe vertex-co fcurves
# (implicit PC2-wins).
#
# Structurally mirrors ``static_deform_ops.py``; the differences are:
#   - per-pin (not per-object) granularity, so the entry carries the
#     vg_name alongside the object UUID and the pin item index;
#   - vertex storage is the pinned subset, not the full mesh;
#   - finalize sets ``pin_item.has_captured_anim = True`` and ensures
#     the EMBEDDED_MOVE sentinel is on the pin's operations list so
#     the encoder treats this pin as animated.

import time

import bpy  # pyright: ignore
import numpy as np

from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from ...core.pc2 import (
    has_pin_anim_pc2,
    remove_pin_anim_pc2,
    resume_mesh_cache_display,
    suspend_mesh_cache_display,
    write_pin_anim_pc2,
)
from ...core.transform import zup_to_yup
from ...core.utils import has_deforming_modifier_stack, redraw_all_areas
from .utils import get_group_from_index


# ---------------------------------------------------------------------------
# Job state (module-level singleton, mirrors the static-deform pattern)
# ---------------------------------------------------------------------------

_capture_job: dict = {
    "active": False,
    "aborted": False,
    # One entry per pin captured in this run. Each carries:
    # obj_uuid, obj_name, vg_name, group_index, pin_index,
    # pin_indices (snapshot at job start), frame_start, frame_end,
    # range_source, frames (ndarray), frames_done.
    "entries": [],
    "entry_cursor": 0,
    "total_frames_processed": 0,
    "total_frames": 0,
    "saved_frame": 1,
    "status": "",
    "error": "",
    # {obj_uuid: prior show_viewport} for ContactSolverCache modifiers
    # suspended for the duration of the capture (see _start_job).
    "suspended_caches": {},
}


def _reset_job() -> None:
    _capture_job.update({
        "active": False,
        "aborted": False,
        "entries": [],
        "entry_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": 0,
        "saved_frame": 1,
        "status": "",
        "error": "",
        "suspended_caches": {},
    })


def is_pin_capture_running() -> bool:
    """True while a Capture Pin Deformation job is in flight."""
    return bool(_capture_job.get("active"))


def pin_capture_progress_snapshot() -> tuple[int, int, str, int]:
    """(done_frames, total_frames, status_line, pin_count) for the UI."""
    return (
        int(_capture_job.get("total_frames_processed", 0)),
        int(_capture_job.get("total_frames", 0)),
        str(_capture_job.get("status", "")),
        len(_capture_job.get("entries", [])),
    )


# ---------------------------------------------------------------------------
# Per-frame sampling: extract pinned-vertex positions in solver world space
# ---------------------------------------------------------------------------


def _sample_pin_frame_world(eval_obj, pin_indices) -> np.ndarray:
    """Sample one frame's world-space positions for the pin's vertices.

    Returns ``(n_pin_verts, 3)`` float32 in solver world space
    (``zup_to_yup @ eval_obj.matrix_world @ co_local``). Translation is
    included so the captured frame-0 row matches the cloth mesh's
    initial-upload position exactly; the encoder builds MoveBy deltas
    between consecutive frames, so the absolute frame is irrelevant
    beyond that alignment.
    """
    eval_mesh = eval_obj.to_mesh()
    try:
        n_total = len(eval_mesh.vertices)
        all_co = np.empty((n_total, 3), dtype=np.float64)
        eval_mesh.vertices.foreach_get("co", all_co.ravel())
        co = all_co[pin_indices]
        mw = np.array(eval_obj.matrix_world, dtype=np.float64).reshape(4, 4)
        z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
        m = z2y @ mw
        homog = np.concatenate(
            [co, np.ones((co.shape[0], 1), dtype=np.float64)], axis=1,
        )
        world = (homog @ m.T)[:, :3]
        return world.astype(np.float32, copy=False)
    finally:
        eval_obj.to_mesh_clear()


# ---------------------------------------------------------------------------
# Frame range discovery: reuse the static-deform helper unchanged, since
# the question "what frames could change the evaluated mesh of *this*
# object" is the same for STATIC colliders and for cloth meshes whose
# pins ride along with bone deformation.
# ---------------------------------------------------------------------------


def _effective_frame_range(scene, obj) -> tuple[int, int, str]:
    from .static_deform_ops import _effective_frame_range as _sd_range
    return _sd_range(scene, obj)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def _build_entry(scene, group_index, pin_index, error_collector: list):
    """Validate a single pin and return a ready-to-capture entry."""
    from ...core.encoder.pin import _get_pin_indices
    from ...core.uuid_registry import resolve_pin, get_or_create_object_uuid
    from ...models.groups import decode_vertex_group_identifier

    group = get_group_from_index(scene, group_index)
    if group is None or group.object_type == "STATIC":
        error_collector.append(iface_("Pin capture: invalid or STATIC group"))
        return None
    if not (0 <= pin_index < len(group.pin_vertex_groups)):
        error_collector.append(iface_("Pin capture: pin index out of range"))
        return None
    pin_item = group.pin_vertex_groups[pin_index]
    try:
        obj = resolve_pin(pin_item)
    except ValueError as exc:
        error_collector.append(str(exc))
        return None
    if obj is None or obj.type != "MESH":
        error_collector.append(iface_("Pin capture: only mesh pins can be captured"))
        return None
    _, vg_name = decode_vertex_group_identifier(pin_item.name)
    if not vg_name:
        error_collector.append(iface_("Pin capture: pin identifier is missing the vertex group"))
        return None
    if any(op.op_type == "TORQUE" for op in pin_item.operations):
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': torque cannot be combined with "
                "captured embedded animation"
            ).format(pin=vg_name, object=obj.name)
        )
        return None
    pin_indices = _get_pin_indices(obj, vg_name)
    if not pin_indices:
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': no pinned vertices to capture"
            ).format(pin=vg_name, object=obj.name)
        )
        return None
    uid = get_or_create_object_uuid(obj)
    if not uid:
        error_collector.append(
            iface_(
                "'{name}' has no UUID (library-linked?); skipped"
            ).format(name=obj.name)
        )
        return None
    frame_start, frame_end, range_source = _effective_frame_range(scene, obj)
    n_frames = frame_end - frame_start + 1
    if n_frames <= 0:
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': empty frame range "
                "(start={start}, end={end})"
            ).format(pin=vg_name, object=obj.name, start=frame_start, end=frame_end)
        )
        return None
    return {
        "obj_uuid": uid,
        "obj_name": obj.name,
        "vg_name": vg_name,
        "group_index": group_index,
        "pin_index": pin_index,
        "pin_indices": np.asarray(pin_indices, dtype=np.int64),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "range_source": range_source,
        "frames": np.empty((n_frames, len(pin_indices), 3), dtype=np.float32),
        "frames_done": 0,
    }


def _process_one_frame(scene, depsgraph, entry, frame: int) -> tuple[bool, str]:
    """Capture one frame of one pin's deformed positions into ``entry``."""
    from ...core.uuid_registry import get_object_by_uuid

    obj = get_object_by_uuid(entry["obj_uuid"])
    if obj is None:
        return False, iface_("'{name}' disappeared during capture").format(name=entry['obj_name'])
    n_verts_total = len(obj.data.vertices)
    pin_idx = entry["pin_indices"]
    if int(pin_idx.max(initial=-1)) >= n_verts_total:
        return False, (
            iface_(
                "Pin '{pin}' on '{object}': vertex count dropped below "
                "pin's max index at frame {frame} (have {count} verts, "
                "need >{max_index}). A topology-changing modifier moved "
                "above the deformer mid-capture, or the mesh was edited; "
                "re-bind the pin and retry."
            ).format(
                pin=entry['vg_name'],
                object=entry['obj_name'],
                frame=frame,
                count=n_verts_total,
                max_index=int(pin_idx.max()),
            )
        )
    eval_obj = obj.evaluated_get(depsgraph)
    world = _sample_pin_frame_world(eval_obj, pin_idx)
    if world.shape != (len(pin_idx), 3):
        return False, (
            iface_(
                "Pin '{pin}' on '{object}': unexpected sample shape "
                "{shape} at frame {frame}"
            ).format(pin=entry['vg_name'], object=entry['obj_name'], shape=world.shape, frame=frame)
        )
    entry["frames"][entry["frames_done"]] = world
    entry["frames_done"] += 1
    return True, ""


def _start_job(context, entries: list) -> tuple[bool, str]:
    if _capture_job["active"]:
        return False, iface_("Another Capture Deformation job is in progress")
    # Refuse to start if a STATIC capture is in flight: the two share
    # the depsgraph + frame_set side effects and would corrupt each
    # other's running frame cursor.
    from .static_deform_ops import is_capture_running as _static_running
    if _static_running():
        return False, iface_("Static-collider capture is in progress; wait for it to finish")
    if not entries:
        return False, iface_("No eligible pins to capture")
    scene = context.scene
    total = sum(e["frames"].shape[0] for e in entries)
    # Suspend each captured object's ContactSolverCache so the per-frame
    # sample reads the pure deformer output, not the solver's previous
    # (OVERWRITE) result. Restored in _teardown on every exit path.
    from ...core.uuid_registry import get_object_by_uuid
    suspended: dict = {}
    for entry in entries:
        uid = entry["obj_uuid"]
        if uid in suspended:
            continue
        obj = get_object_by_uuid(uid)
        suspended[uid] = suspend_mesh_cache_display(obj)
    _capture_job.update({
        "active": True,
        "aborted": False,
        "entries": entries,
        "entry_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": total,
        "saved_frame": int(scene.frame_current),
        "status": iface_("Capturing pin..."),
        "error": "",
        "suspended_caches": suspended,
    })
    return True, ""


def _tick_job(context, *, budget_ms: int = 40) -> bool:
    """Capture frames until the per-tick budget expires."""
    deadline = time.perf_counter() + budget_ms / 1000.0
    entries = _capture_job["entries"]
    scene = context.scene
    while time.perf_counter() < deadline:
        if _capture_job["aborted"]:
            return False
        idx = _capture_job["entry_cursor"]
        if idx >= len(entries):
            return False
        entry = entries[idx]
        if entry["frames_done"] >= entry["frames"].shape[0]:
            _capture_job["entry_cursor"] += 1
            continue
        frame = entry["frame_start"] + entry["frames_done"]
        scene.frame_set(frame)
        depsgraph = context.evaluated_depsgraph_get()
        ok, reason = _process_one_frame(scene, depsgraph, entry, frame)
        if not ok:
            _capture_job["error"] = reason
            _capture_job["aborted"] = True
            return False
        _capture_job["total_frames_processed"] += 1
        _capture_job["status"] = (
            iface_(
                "Capturing pin '{pin}' on '{object}' "
                "[{done}/{total}] ({source})"
            ).format(
                pin=entry['vg_name'],
                object=entry['obj_name'],
                done=entry['frames_done'],
                total=entry['frames'].shape[0],
                source=entry['range_source'],
            )
        )
    return _capture_job["entry_cursor"] < len(entries)


def _finalize_job(context) -> tuple[int, int]:
    """Write completed entries to PC2 and restore the saved frame."""
    from ...core.uuid_registry import get_object_by_uuid
    from .pin_ops import _ensure_embedded_move_op

    pins_written = 0
    frames_written = 0
    for entry in _capture_job["entries"]:
        if entry["frames_done"] < entry["frames"].shape[0]:
            continue
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        write_pin_anim_pc2(obj, entry["vg_name"], entry["frames"])
        # Re-resolve the live pin item by group + index so we mutate
        # the actual PropertyGroup, not a stale reference.
        group = get_group_from_index(context.scene, entry["group_index"])
        if group is None:
            continue
        pin_index = entry["pin_index"]
        if not (0 <= pin_index < len(group.pin_vertex_groups)):
            continue
        pin_item = group.pin_vertex_groups[pin_index]
        pin_item.has_captured_anim = True
        _ensure_embedded_move_op(pin_item)
        pins_written += 1
        frames_written += entry["frames"].shape[0]
    return pins_written, frames_written


def _restore_frame(context) -> None:
    saved = _capture_job.get("saved_frame")
    if saved is not None:
        context.scene.frame_set(int(saved))


def _cleanup_after_job(context) -> None:
    """Restore the saved frame, resume the suspended ContactSolverCache
    modifiers, and reset the job singleton (the non-timer half of the modal
    teardown, shared with the all-pins coordinator)."""
    from ...core.uuid_registry import get_object_by_uuid

    _restore_frame(context)
    for uid, prior in _capture_job.get("suspended_caches", {}).items():
        resume_mesh_cache_display(get_object_by_uuid(uid), prior)
    _reset_job()


# ---------------------------------------------------------------------------
# Modal capture operator
# ---------------------------------------------------------------------------


class OBJECT_OT_CapturePinDeformation(Operator):
    """Sample the pin's vertices from the depsgraph each frame and
    store them as a captured-animation cache the solver consumes in
    place of manual keyframes"""

    bl_idname = "object.capture_pin_deformation"
    bl_label = "Capture Deformation"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def execute(self, context):
        # Refuse if the pin already carries manual vertex-co fcurves.
        # The user should clear them first so the captured cache owns
        # the EMBEDDED_MOVE sentinel without ambiguity, and so the
        # encoder doesn't have to choose between two animation sources
        # at transfer time.
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None and 0 <= self.pin_index < len(group.pin_vertex_groups):
            pin_item = group.pin_vertex_groups[self.pin_index]
            if _pin_has_vertex_co_fcurves(pin_item):
                self.report(
                    {"WARNING"},
                    iface_(
                        "Pin has manual keyframes (vertex-co fcurves); press "
                        "Delete All Keyframes first, then re-run Capture "
                        "Deformation"
                    ),
                )
                return {"CANCELLED"}
            # Refuse on non-deformable objects up front so EXEC_DEFAULT
            # (panel button, MCP, Python API) never starts a modal job
            # that would just bake the rest pose. The panel greys the
            # button out via the same predicate; this is the same check
            # for callers that bypass the panel.
            from ...core.uuid_registry import resolve_pin as _rp
            try:
                _pin_obj = _rp(pin_item)
            except ValueError:
                _pin_obj = None
            if _pin_obj is None or _pin_obj.type != "MESH" or not has_deforming_modifier_stack(_pin_obj):
                self.report(
                    {"WARNING"},
                    iface_(
                        "Pin object has no deforming modifier stack "
                        "(Armature, Lattice, Mesh Deform, Shape Keys, ...); "
                        "Capture Deformation has nothing to record"
                    ),
                )
                return {"CANCELLED"}

        errors: list = []
        entry = _build_entry(scene, self.group_index, self.pin_index, errors)
        if entry is None:
            self.report({"WARNING"}, errors[0] if errors else iface_("Nothing to capture"))
            return {"CANCELLED"}
        ok, err = _start_job(context, [entry])
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
        # Any exception inside the tick must still run _teardown, or the
        # suspended ContactSolverCache stays disabled and the job flag
        # stays set forever (worse than a visible failure).
        try:
            if _capture_job["aborted"]:
                err = _capture_job.get("error", "")
                self._teardown(context)
                redraw_all_areas(context)
                if err:
                    self.report({"ERROR"}, err)
                else:
                    self.report({"INFO"}, iface_("Pin capture aborted"))
                return {"CANCELLED"}
            more = _tick_job(context)
            redraw_all_areas(context)
            if not more and not _capture_job["aborted"]:
                n_pins, n_frames = _finalize_job(context)
                self._teardown(context)
                redraw_all_areas(context)
                from ...models.groups import invalidate_overlays
                invalidate_overlays()
                self.report(
                    {"INFO"},
                    iface_("Captured {frames} frame(s) across {pins} pin(s)").format(
                        frames=n_frames, pins=n_pins
                    ),
                )
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        except Exception as exc:  # noqa: BLE001 — must restore state
            self._teardown(context)
            redraw_all_areas(context)
            self.report({"ERROR"}, iface_("Pin capture failed: {error}").format(error=exc))
            return {"CANCELLED"}

    def cancel(self, context):
        self._teardown(context)
        redraw_all_areas(context)

    def _teardown(self, context):
        wm = context.window_manager
        timer = getattr(self, "_timer", None)
        if timer is not None:
            wm.event_timer_remove(timer)
            self._timer = None
        _cleanup_after_job(context)


# ---------------------------------------------------------------------------
# Clear operator: drop the cache and (when no manual fcurves exist) the
# EMBEDDED_MOVE sentinel.
# ---------------------------------------------------------------------------


def _pin_has_vertex_co_fcurves(pin_item) -> bool:
    """True if the pin's object has any vertex-co fcurves on its mesh action.

    Used to decide whether Clear should also drop the EMBEDDED_MOVE
    sentinel. If the manual Make-Keyframe path authored fcurves, the
    sentinel still belongs to it.
    """
    from ...core.encoder.pin import _collect_pin_vertex_fcurve_frames
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier

    try:
        obj = resolve_pin(pin_item)
    except ValueError:
        return False
    if obj is None or obj.type != "MESH":
        return False
    _, vg_name = decode_vertex_group_identifier(pin_item.name)
    if not vg_name:
        return False
    frames, _ = _collect_pin_vertex_fcurve_frames(obj, vg_name)
    return bool(frames)


class OBJECT_OT_ClearPinDeformation(Operator):
    """Delete this pin's captured-deformation cache. If the pin has no
    manual keyframes, also remove the embedded-move sentinel."""

    bl_idname = "object.clear_pin_deformation"
    bl_label = "Clear Deformation Cache"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def execute(self, context):
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier
        from .pin_ops import _remove_embedded_move_ops

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}
        if not (0 <= self.pin_index < len(group.pin_vertex_groups)):
            self.report({"ERROR"}, iface_("Pin index out of range"))
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[self.pin_index]
        try:
            obj = resolve_pin(pin_item)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, iface_("Pin object not found"))
            return {"CANCELLED"}
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if not vg_name:
            self.report({"ERROR"}, iface_("Invalid pin identifier"))
            return {"CANCELLED"}
        remove_pin_anim_pc2(obj, vg_name)
        pin_item.has_captured_anim = False
        # Drop the sentinel only if the manual Make Keyframe path
        # didn't author its own fcurves. Otherwise it still owns the
        # EMBEDDED_MOVE slot.
        if not _pin_has_vertex_co_fcurves(pin_item):
            _remove_embedded_move_ops(pin_item)
        redraw_all_areas(context)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        self.report(
            {"INFO"},
            iface_("Cleared captured deformation for pin '{pin}' on '{object}'").format(
                pin=vg_name, object=obj.name
            ),
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Abort button (mirrors the static-deform abort)
# ---------------------------------------------------------------------------


class SOLVER_OT_PinCaptureAbort(Operator):
    """Abort the running Capture Pin Deformation job"""

    bl_idname = "solver.pin_capture_abort"
    bl_label = "Abort"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(_capture_job.get("active")) and not _capture_job.get("aborted")

    def execute(self, context):
        _capture_job["aborted"] = True
        _capture_job["status"] = iface_("Aborting...")
        redraw_all_areas(context)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Public predicates the panel uses to gate the row
# ---------------------------------------------------------------------------


def pin_object_supports_capture(obj) -> bool:
    """True if *obj* has a deforming modifier stack worth capturing.

    Cheap declarative check (matches the STATIC predicate's tier 1).
    The deeper depsgraph-sampling tiers in ``is_deforming_static_object``
    would force per-redraw frame_set calls, which is too expensive for
    a panel poll. Users with driver-only motion can still trigger the
    capture; the predicate exists only to grey-out the button when
    nothing in the modifier stack will produce different positions
    between frames.
    """
    return has_deforming_modifier_stack(obj)


def pin_has_captured_anim(pin_item) -> bool:
    """True if this pin's PropertyGroup flag is set OR a PC2 cache
    exists on disk. The flag is the cheap source of truth; the disk
    fallback catches cases where the flag desynced (rare; the
    load_post reconciler handles the common case)."""
    if pin_item is None:
        return False
    if bool(getattr(pin_item, "has_captured_anim", False)):
        return True
    try:
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier
        obj = resolve_pin(pin_item)
        if obj is None or obj.type != "MESH":
            return False
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if not vg_name:
            return False
        return has_pin_anim_pc2(obj, vg_name)
    except Exception:
        return False


def pin_captured_frame_count(pin_item) -> int:
    """Frame count of this pin's captured cache, or 0 if absent."""
    from ...core.pc2 import get_pin_anim_cache
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier

    if pin_item is None:
        return 0
    try:
        obj = resolve_pin(pin_item)
    except ValueError:
        return 0
    if obj is None or obj.type != "MESH":
        return 0
    _, vg_name = decode_vertex_group_identifier(pin_item.name)
    if not vg_name:
        return 0
    cache = get_pin_anim_cache(obj, vg_name)
    if cache is None:
        return 0
    return int(cache.shape[0])


# ---------------------------------------------------------------------------
# Refresh full-pin coverage (gates the rest-pose tracking toggle)
# ---------------------------------------------------------------------------


class OBJECT_OT_RefreshFullPinState(Operator):
    """Check whether this pin's vertex group covers every vertex of the mesh
    (a full pin) and cache the result on the pin. "Track Rest-Pose
    Deformation" needs a full pin, so its toggle stays disabled until this
    confirms one. The check is an O(N) vertex scan, so it runs only on click,
    not on every panel redraw. Run it again after editing the vertex group to
    update the cached state."""

    bl_idname = "object.refresh_full_pin_state"
    bl_label = "Refresh"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...core.utils import pin_covers_all_vertices
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}
        if not (0 <= self.pin_index < len(group.pin_vertex_groups)):
            self.report({"ERROR"}, iface_("Pin index out of range"))
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[self.pin_index]
        try:
            obj = resolve_pin(pin_item)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        is_full = bool(
            obj is not None
            and obj.type == "MESH"
            and pin_covers_all_vertices(obj, vg_name)
        )
        pin_item.full_pin_checked = True
        pin_item.full_pin_cached = is_full
        redraw_all_areas(context)
        self.report(
            {"INFO"},
            iface_("Full pin: rest-pose tracking available")
            if is_full
            else iface_("Partial pin: rest-pose tracking unavailable"),
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# "Re-capture all" support: enumerate every capturable pin across the active
# non-STATIC groups and drive the shared job from the coordinator in
# solver.py (which owns its own timer). The job machinery already processes
# a list of entries, so "all pins" is just a larger entry set.
# ---------------------------------------------------------------------------


def collect_capturable_pins(context, *, cheap: bool = False) -> list:
    """``(group_index, pin_index)`` for every pin in active non-STATIC groups
    whose object can be captured.

    Mirrors the per-pin button gate: a mesh pin whose object has a deforming
    modifier stack, no TORQUE op, and (when ``cheap`` is False) no manual
    vertex-co fcurves. ``cheap=True`` skips the fcurve scan so the result is
    light enough for a panel poll; ``cheap=False`` is the accurate set used
    when actually starting the job.
    """
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import N_MAX_GROUPS, get_addon_data

    data = get_addon_data(context.scene)
    specs: list = []
    for gi in range(N_MAX_GROUPS):
        group = getattr(data, f"object_group_{gi}", None)
        if not group or not group.active or group.object_type == "STATIC":
            continue
        for pi, pin_item in enumerate(group.pin_vertex_groups):
            try:
                obj = resolve_pin(pin_item)
            except ValueError:
                continue
            if obj is None or obj.type != "MESH":
                continue
            if not pin_object_supports_capture(obj):
                continue
            if any(op.op_type == "TORQUE" for op in pin_item.operations):
                continue
            if not cheap and _pin_has_vertex_co_fcurves(pin_item):
                continue
            specs.append((gi, pi))
    return specs


def start_capture_for_pins(context, specs: list) -> tuple[bool, str]:
    """Build entries for each ``(group_index, pin_index)`` in *specs* and
    start the shared pin-capture job. Returns ``(ok, error_message)``."""
    errors: list = []
    entries = []
    for gi, pi in specs:
        entry = _build_entry(context.scene, gi, pi, errors)
        if entry is not None:
            entries.append(entry)
    if not entries:
        return False, (errors[0] if errors else iface_("No eligible pins to capture"))
    return _start_job(context, entries)


def advance_pin_capture(context) -> tuple[bool, bool, str]:
    """Tick the running pin capture once; returns ``(more, aborted, error)``."""
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    more = _tick_job(context)
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    return more, False, ""


def finalize_pin_capture(context) -> tuple[int, int]:
    """Write completed pin entries to PC2; returns ``(pins, frames)`` written."""
    return _finalize_job(context)


def cleanup_pin_capture(context) -> None:
    """Restore frame, resume suspended caches, and clear the job state."""
    _cleanup_after_job(context)


def collect_pins_with_captured_anim(context) -> list:
    """``(group_index, pin_index)`` for every pin in active non-STATIC groups
    that currently holds a captured-deformation cache. Used by "Clear All
    Deformations" (and its poll)."""
    from ...models.groups import N_MAX_GROUPS, get_addon_data

    data = get_addon_data(context.scene)
    specs: list = []
    for gi in range(N_MAX_GROUPS):
        group = getattr(data, f"object_group_{gi}", None)
        if not group or not group.active or group.object_type == "STATIC":
            continue
        for pi, pin_item in enumerate(group.pin_vertex_groups):
            if pin_has_captured_anim(pin_item):
                specs.append((gi, pi))
    return specs


def clear_captured_pin(context, group_index: int, pin_index: int) -> bool:
    """Drop one pin's captured-deformation cache, clear its
    ``has_captured_anim`` flag, and (when the pin carries no manual vertex-co
    fcurves) remove the EMBEDDED_MOVE sentinel. Returns True if a pin was
    cleared. Mirrors ``OBJECT_OT_ClearPinDeformation.execute`` for the
    batch "Clear All Deformations" path."""
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier
    from .pin_ops import _remove_embedded_move_ops

    group = get_group_from_index(context.scene, group_index)
    if group is None or not (0 <= pin_index < len(group.pin_vertex_groups)):
        return False
    pin_item = group.pin_vertex_groups[pin_index]
    try:
        obj = resolve_pin(pin_item)
    except ValueError:
        return False
    if obj is None or obj.type != "MESH":
        return False
    _, vg_name = decode_vertex_group_identifier(pin_item.name)
    if not vg_name:
        return False
    remove_pin_anim_pc2(obj, vg_name)
    pin_item.has_captured_anim = False
    if not _pin_has_vertex_co_fcurves(pin_item):
        _remove_embedded_move_ops(pin_item)
    return True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


classes = (
    OBJECT_OT_CapturePinDeformation,
    OBJECT_OT_ClearPinDeformation,
    OBJECT_OT_RefreshFullPinState,
    SOLVER_OT_PinCaptureAbort,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
