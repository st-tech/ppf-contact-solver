# File: static_deform_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Deformation: sample Blender's depsgraph for a STATIC mesh
# collider whose modifier stack deforms vertices (Armature, MeshDeform,
# Lattice, shape keys, ...), and store the per-frame absolute vertex
# positions in solver world space as a static-deform PC2 sidecar.
#
# Surfaced as a per-object button on STATIC group rows: deformation is
# only meaningful for STATIC colliders, so there's no top-level
# "Capture All" button — the per-group placement is the source of truth.
#
# The operator runs modally on a timer (matches the bake_ops modal
# pattern so Blender's main thread stays responsive across long
# captures, and the user can cancel mid-job). The capture writes the
# new PC2 namespace in core/pc2.py (``_staticdeform`` keys); the
# encoder picks it up on the next upload.

import time

import bpy  # pyright: ignore
import numpy as np

from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from ...core.pc2 import (
    remove_static_deform_pc2,
    resume_mesh_cache_display,
    static_deform_pc2_key,
    suspend_mesh_cache_display,
    write_static_deform_pc2,
)
from ...core.transform import zup_to_yup
from ...core.utils import is_deforming_static_object, redraw_all_areas
from .utils import get_group_from_index


# ---------------------------------------------------------------------------
# Job state (module-level singleton, mirrors the bake_ops pattern)
# ---------------------------------------------------------------------------

_capture_job: dict = {
    "active": False,
    "aborted": False,
    # Each entry: {"obj_uuid", "obj_name", "n_verts", "frame_start",
    # "frame_end", "range_source", "frames": ndarray, "frames_done"}.
    # Per-entry frame range so two objects with independent deformer
    # chains each capture only the frames their own animation covers.
    "objects": [],
    "object_cursor": 0,    # index of in-flight entry
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
        "objects": [],
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": 0,
        "saved_frame": 1,
        "status": "",
        "error": "",
        "suspended_caches": {},
    })


def is_capture_running() -> bool:
    """True while a Capture Deformation job is in flight."""
    return bool(_capture_job.get("active"))


def capture_progress_snapshot() -> tuple[int, int, str, int]:
    """(done_frames, total_frames, status_line, object_count) for the UI."""
    return (
        int(_capture_job.get("total_frames_processed", 0)),
        int(_capture_job.get("total_frames", 0)),
        str(_capture_job.get("status", "")),
        len(_capture_job.get("objects", [])),
    )


# ---------------------------------------------------------------------------
# Per-object capture: walk frame_start..frame_end on a timer
# ---------------------------------------------------------------------------


def _frame_to_world_solver(eval_obj) -> np.ndarray:
    """Sample one frame's world-space vertex positions in solver space.

    Composes ``zup_to_yup * eval_obj.matrix_world * eval_mesh.vertices[i].co``
    in one numpy matmul, so per-frame cost is O(n_verts) with no Python
    loop. Returns ``(n_verts, 3)`` float32 in solver coordinates.
    """
    eval_mesh = eval_obj.to_mesh()
    try:
        n = len(eval_mesh.vertices)
        co = np.empty((n, 3), dtype=np.float64)
        eval_mesh.vertices.foreach_get("co", co.ravel())
        # Compose world matrix in solver space once per frame.
        mw = np.array(eval_obj.matrix_world, dtype=np.float64).reshape(4, 4)
        z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
        m = z2y @ mw
        homog = np.concatenate([co, np.ones((n, 1), dtype=np.float64)], axis=1)
        world = (homog @ m.T)[:, :3]
        return world.astype(np.float32, copy=False)
    finally:
        eval_obj.to_mesh_clear()


def _max_keyframe_in_action(action) -> int | None:
    """Largest keyframe frame across every fcurve of a layered Action.

    Walks Blender 5.x's action.layers[*].strips[*].channelbags[*].fcurves
    tree; falls back to the legacy flat ``action.fcurves`` on older
    builds. Returns ``None`` when the action has no keyframes.
    """
    max_f = None
    layers = getattr(action, "layers", None) or []
    for layer in layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        f = int(kp.co[0])
                        if max_f is None or f > max_f:
                            max_f = f
    # Pre-Baklava flat fcurves (in case some build still emits them).
    for fc in getattr(action, "fcurves", []) or []:
        for kp in fc.keyframe_points:
            f = int(kp.co[0])
            if max_f is None or f > max_f:
                max_f = f
    return max_f


def _collect_influencing_actions(obj, _seen=None) -> list:
    """Every Action that can change *obj*'s depsgraph-evaluated mesh.

    Walks (1) the object's own animation_data, (2) its shape-keys
    animation_data, (3) every parent in the chain, (4) every modifier's
    ``.object`` target recursively. Cycle-protected via ``_seen``
    (id-keyed). Covers the common deformers: Armature on a parent or
    self, MeshDeform / SurfaceDeform / Lattice / Hook with their own
    animated cages, shape keys driven by an action. Does NOT walk
    drivers, constraints, or geometry-nodes Scene-Time inputs; those
    return no actions and the capture falls back to the solver's global
    frame count (see ``_effective_frame_range``).
    """
    if _seen is None:
        _seen = set()
    out: list = []
    if obj is None or id(obj) in _seen:
        return out
    _seen.add(id(obj))
    ad = getattr(obj, "animation_data", None)
    if ad is not None and ad.action is not None:
        out.append(ad.action)
    data = getattr(obj, "data", None)
    if data is not None:
        sk = getattr(data, "shape_keys", None)
        if sk is not None:
            sk_ad = getattr(sk, "animation_data", None)
            if sk_ad is not None and sk_ad.action is not None:
                out.append(sk_ad.action)
    parent = getattr(obj, "parent", None)
    if parent is not None:
        out.extend(_collect_influencing_actions(parent, _seen))
    for mod in getattr(obj, "modifiers", []) or []:
        target = getattr(mod, "object", None)
        if target is not None:
            out.extend(_collect_influencing_actions(target, _seen))
    return out


def _sim_frame_count(scene) -> int | None:
    """The solver's global simulation frame count for *scene*.

    Reads ``state.frame_count`` (the "Frame Count" / "Number of frames
    for simulation" param). Returns ``None`` when the addon state isn't
    available yet (during register / fresh load_post), so callers can
    fall back.
    """
    try:
        from ...models.groups import get_addon_data
        return int(get_addon_data(scene).state.frame_count)
    except Exception:
        return None


def _effective_frame_range(scene, obj) -> tuple[int, int, str]:
    """Pick the [frame_start, frame_end] to capture for *obj*.

    Lower bound is always ``scene.frame_start`` (so cache index 0
    aligns with solver time 0). The upper bound depends on how the
    motion is authored:

      - Keyframed deformers (Armature, Lattice, shape keys, animated
        Geometry Nodes inputs): the highest keyframe across every
        influencing action. The action settles at its last key, so
        capturing past it would burn frames on a static pose, and the
        viewport timeline being shrunk doesn't truncate the cache.
      - Procedural motion with no detectable keyframes (Geometry Nodes
        driven by Scene Time, drivers, constraints): there's no settle
        point to read, so the capture is bounded by the solver's global
        frame count (``state.frame_count``), exactly the number of
        frames the simulation will consume from the cache. Only when
        the addon state is unavailable do we fall back to
        ``scene.frame_end``.
    """
    frame_start = int(scene.frame_start)
    actions = _collect_influencing_actions(obj)
    max_kf = None
    for act in actions:
        f = _max_keyframe_in_action(act)
        if f is not None and (max_kf is None or f > max_kf):
            max_kf = f
    if max_kf is None:
        # No detectable keyframes: procedural motion (Geometry Nodes
        # on Scene Time, drivers, constraints). Bound the capture by the
        # solver's frame count, not the viewport timeline's frame_end.
        n_sim = _sim_frame_count(scene)
        if n_sim is not None and n_sim > 0:
            frame_end = frame_start + n_sim - 1
            return frame_start, frame_end, iface_("simulation frame count ({count})").format(count=n_sim)
        return frame_start, int(scene.frame_end), iface_("scene.frame_end (state unavailable)")
    frame_end = max(max_kf, frame_start)
    return frame_start, frame_end, iface_("last keyframe ({frame})").format(frame=max_kf)


def _build_entries(scene, objects: list, error_collector: list) -> list:
    """Validate each candidate and return job entries ready to capture.

    Each entry pre-allocates the (n_frames, n_verts, 3) buffer and
    records the bind topology vertex count for the per-frame stability
    check. Per-entry frame ranges so two objects with independent
    deformer chains each capture only the frames their own animation
    needs.
    """
    from ...core.uuid_registry import get_or_create_object_uuid

    entries = []
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        uuid = get_or_create_object_uuid(obj)
        if not uuid:
            error_collector.append(
                iface_("'{name}' has no UUID (library-linked?); skipped").format(name=obj.name)
            )
            continue
        n_verts = len(obj.data.vertices)
        if n_verts == 0:
            error_collector.append(iface_("'{name}' has no vertices; skipped").format(name=obj.name))
            continue
        frame_start, frame_end, range_source = _effective_frame_range(scene, obj)
        n_frames = frame_end - frame_start + 1
        if n_frames <= 0:
            error_collector.append(
                iface_("'{name}' frame range is empty (start={start}, end={end})").format(
                    name=obj.name, start=frame_start, end=frame_end
                )
            )
            continue
        entries.append({
            "obj_uuid": uuid,
            "obj_name": obj.name,
            "n_verts": n_verts,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "range_source": range_source,
            "frames": np.empty((n_frames, n_verts, 3), dtype=np.float32),
            "frames_done": 0,
        })
    return entries


def _process_one_frame(scene, depsgraph, entry, frame: int) -> tuple[bool, str]:
    """Capture one frame of one object's deformed mesh into ``entry``.

    Returns ``(ok, reason)``. ``ok == False`` aborts the whole job: a
    topology mismatch mid-capture is unrecoverable (the PC2 reader
    assumes a fixed vertex count per file).
    """
    from ...core.uuid_registry import get_object_by_uuid

    obj = get_object_by_uuid(entry["obj_uuid"])
    if obj is None:
        return False, iface_("'{name}' disappeared during capture").format(name=entry['obj_name'])
    eval_obj = obj.evaluated_get(depsgraph)
    world = _frame_to_world_solver(eval_obj)
    if world.shape[0] != entry["n_verts"]:
        return False, (
            iface_(
                "'{name}' vertex count changed at frame {frame}: "
                "{actual} vs {expected} at frame_start. "
                "Move any topology-changing modifiers (Subdivision Surface, "
                "Remesh, Decimate) ABOVE the deformer, or apply them, then "
                "retry."
            ).format(
                name=entry['obj_name'],
                frame=frame,
                actual=world.shape[0],
                expected=entry['n_verts'],
            )
        )
    entry["frames"][entry["frames_done"]] = world
    entry["frames_done"] += 1
    return True, ""


def _start_job(context, entries: list) -> tuple[bool, str]:
    if _capture_job["active"]:
        return False, iface_("Another Capture Deformation job is in progress")
    # Refuse to start if a pin-deformation capture is in flight: the two
    # share the depsgraph + frame_set side effects and would corrupt each
    # other's running frame cursor (mirrors the symmetric guard in
    # pin_capture_ops._start_job).
    from .pin_capture_ops import is_pin_capture_running as _pin_running
    if _pin_running():
        return False, iface_("Pin-deformation capture is in progress; wait for it to finish")
    if not entries:
        return False, iface_("No deforming STATIC objects to capture")
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
        suspended[uid] = suspend_mesh_cache_display(get_object_by_uuid(uid))
    _capture_job.update({
        "active": True,
        "aborted": False,
        "objects": entries,
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": total,
        "saved_frame": int(scene.frame_current),
        "status": iface_("Capturing..."),
        "error": "",
        "suspended_caches": suspended,
    })
    return True, ""


def _tick_job(context, *, budget_ms: int = 40) -> bool:
    """Capture frames until the per-tick budget expires. Returns ``True``
    if more frames remain, ``False`` when complete or aborted/failed."""
    deadline = time.perf_counter() + budget_ms / 1000.0
    objects = _capture_job["objects"]
    scene = context.scene
    while time.perf_counter() < deadline:
        if _capture_job["aborted"]:
            return False
        idx = _capture_job["object_cursor"]
        if idx >= len(objects):
            return False
        entry = objects[idx]
        if entry["frames_done"] >= entry["frames"].shape[0]:
            _capture_job["object_cursor"] += 1
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
            iface_("Capturing {name} [{done}/{total}] ({source})").format(
                name=entry['obj_name'],
                done=entry['frames_done'],
                total=entry['frames'].shape[0],
                source=entry['range_source'],
            )
        )
    return _capture_job["object_cursor"] < len(objects)


def _finalize_job(context) -> tuple[int, int]:
    """Write completed entries to PC2 and restore the saved frame.

    Returns ``(objects_written, total_frames_written)``. Entries with
    a partial buffer (job aborted mid-object) are skipped — Capture
    Deformation is atomic per object.
    """
    from ...core.uuid_registry import get_object_by_uuid

    objects_written = 0
    frames_written = 0
    for entry in _capture_job["objects"]:
        if entry["frames_done"] < entry["frames"].shape[0]:
            continue
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        write_static_deform_pc2(obj, entry["frames"])
        objects_written += 1
        frames_written += entry["frames"].shape[0]
    return objects_written, frames_written


def _restore_frame(context) -> None:
    saved = _capture_job.get("saved_frame")
    if saved is not None:
        context.scene.frame_set(int(saved))


def _cleanup_after_job(context) -> None:
    """Restore the saved frame, resume the suspended ContactSolverCache
    modifiers, and reset the job singleton.

    The non-timer half of the modal teardown, factored out so the
    all-objects coordinator (which owns its own timer) can clean up the
    job after each phase without an operator instance.
    """
    from ...core.uuid_registry import get_object_by_uuid

    _restore_frame(context)
    for uid, prior in _capture_job.get("suspended_caches", {}).items():
        resume_mesh_cache_display(get_object_by_uuid(uid), prior)
    _reset_job()


# ---------------------------------------------------------------------------
# Modal base (mirrors _ModalBakeBase, simpler because capture has no
# undo path: a captured PC2 is either fully written or never written)
# ---------------------------------------------------------------------------


class _ModalCaptureBase:
    """Shared modal loop for both per-object and all-object capture."""

    def _build_entries_for(self, context) -> list:
        raise NotImplementedError

    def execute(self, context):
        errors: list = []
        entries = self._build_entries_for(context)
        scene_entries = _build_entries(context.scene, entries, errors)
        if not scene_entries:
            msg = errors[0] if errors else iface_("No deforming STATIC objects to capture")
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        ok, err = _start_job(context, scene_entries)
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
                    self.report({"INFO"}, iface_("Capture aborted"))
                return {"CANCELLED"}
            more = _tick_job(context)
            redraw_all_areas(context)
            if not more and not _capture_job["aborted"]:
                n_objs, n_frames = _finalize_job(context)
                self._teardown(context)
                from .overlay import apply_object_overlays
                apply_object_overlays()
                redraw_all_areas(context)
                self.report(
                    {"INFO"},
                    iface_("Captured {frames} frame(s) for {objects} object(s)").format(
                        frames=n_frames, objects=n_objs
                    ),
                )
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        except Exception as exc:  # noqa: BLE001 — must restore state
            self._teardown(context)
            redraw_all_areas(context)
            self.report({"ERROR"}, iface_("Capture failed: {error}").format(error=exc))
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
# Per-object operator
# ---------------------------------------------------------------------------


class OBJECT_OT_CaptureStaticDeformation(_ModalCaptureBase, Operator):
    """Sample this object's depsgraph-evaluated mesh per frame and
    store it as a static-deform cache the solver consumes as kinematic
    collider geometry"""

    bl_idname = "object.capture_static_deformation"
    bl_label = "Capture Deformation"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def _build_entries_for(self, context) -> list:
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None or group.object_type != "STATIC":
            return []
        idx = group.assigned_objects_index
        if not (0 <= idx < len(group.assigned_objects)):
            return []
        from ...core.uuid_registry import resolve_assigned
        assigned = group.assigned_objects[idx]
        obj = resolve_assigned(assigned)
        if obj is None or obj.type != "MESH":
            return []
        if not is_deforming_static_object(obj, context):
            return []
        return [obj]


# ---------------------------------------------------------------------------
# Cache-clear operator (drops one object's static-deform PC2 + memory)
# ---------------------------------------------------------------------------


class OBJECT_OT_ClearStaticDeformation(Operator):
    """Delete this object's static-deform cache (PC2 file + in-memory copy)"""

    bl_idname = "object.clear_static_deformation"
    bl_label = "Clear Deformation Cache"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def execute(self, context):
        from ...core.uuid_registry import resolve_assigned
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}
        idx = group.assigned_objects_index
        if not (0 <= idx < len(group.assigned_objects)):
            self.report({"ERROR"}, iface_("No object selected"))
            return {"CANCELLED"}
        assigned = group.assigned_objects[idx]
        obj = resolve_assigned(assigned)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, iface_("Object not found"))
            return {"CANCELLED"}
        remove_static_deform_pc2(obj)
        from .overlay import apply_object_overlays
        apply_object_overlays()
        redraw_all_areas(context)
        self.report({"INFO"}, iface_("Cleared deformation cache for '{name}'").format(name=obj.name))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Abort button (matches the bake_ops pattern: flips the job flag and the
# modal loop notices on its next tick)
# ---------------------------------------------------------------------------


class SOLVER_OT_CaptureAbort(Operator):
    """Abort the running Capture Deformation job"""

    bl_idname = "solver.capture_abort"
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
# UI poll helpers (used by panels.py to gate the per-object button)
# ---------------------------------------------------------------------------


def object_needs_deformation_capture(obj, context) -> bool:
    """Public predicate the UI uses to decide whether to show the button.

    Called from panel ``draw()``, so it must stay cheap and write-free: it
    runs the declarative tiers of ``is_deforming_static_object``
    (``allow_eval=False``, no depsgraph sampling / no scene mutation) plus a
    cheap non-fcurve motion check, so the button stays reachable for
    parent/constraint/driver/NLA-driven statics without stepping the scene on
    every redraw. The encoder runs the full depsgraph-backed gate at Transfer.
    """
    from ...core.utils import _has_nonfcurve_motion_source
    return (
        is_deforming_static_object(obj, context, allow_eval=False)
        or _has_nonfcurve_motion_source(obj)
    )


def object_has_deformation_cache(obj) -> bool:
    """True if *obj* already has a static-deform PC2 on disk or in memory."""
    if obj is None:
        return False
    from ...core.pc2 import has_static_deform_animation
    return has_static_deform_animation(obj)


def object_deformation_frame_count(obj) -> int:
    """Number of frames currently in *obj*'s static-deform cache (0 if absent)."""
    if obj is None:
        return 0
    from ...core.pc2 import get_static_deform_cache
    cache = get_static_deform_cache(obj)
    if cache is None:
        return 0
    return int(cache.shape[0])


# ---------------------------------------------------------------------------
# "Re-capture all" support: enumerate every deforming STATIC object across
# the active groups, and drive the shared job from a coordinator that owns
# its own timer (see SOLVER_OT_RecaptureAllDeformations in solver.py). The
# capture machinery already processes a list of entries, so "all" is just a
# different entry set fed through the same _build_entries / _start_job /
# _tick_job / _finalize_job path.
# ---------------------------------------------------------------------------


def collect_capturable_static_objects(context, *, allow_eval: bool = False) -> list:
    """Every deforming STATIC-group object that needs a deformation capture,
    deduplicated, in group/assignment order.

    ``allow_eval=False`` (default) uses the cheap, draw-safe predicate
    (:func:`object_needs_deformation_capture`, no depsgraph sampling), so it
    is callable from a panel poll. ``allow_eval=True`` uses the full
    depsgraph-backed gate (:func:`is_deforming_static_object`), matching the
    per-object Capture button, for the execute path.
    """
    from ...core.uuid_registry import resolve_assigned
    from ...models.groups import iterate_active_object_groups

    out: list = []
    seen: set = set()
    for group in iterate_active_object_groups(context.scene):
        if group.object_type != "STATIC":
            continue
        for assigned in group.assigned_objects:
            try:
                obj = resolve_assigned(assigned)
            except ValueError:
                continue
            if obj is None or obj.type != "MESH" or id(obj) in seen:
                continue
            needs = (
                is_deforming_static_object(obj, context)
                if allow_eval
                else object_needs_deformation_capture(obj, context)
            )
            if needs:
                seen.add(id(obj))
                out.append(obj)
    return out


def scene_has_capturable_static(context) -> bool:
    """Fast, stateless poll signal for "Re-capture All Deformations": True as
    soon as one active STATIC-group MESH object needs a deformation capture.

    Resolving every assigned STATIC object by UUID per redraw was the cost
    (~0.03ms each); the declarative "is this deforming" predicate is ~25x
    cheaper. So this scans ``bpy.data.objects`` with that cheap predicate
    first -- which rejects the rigid-collider majority without resolving
    anything -- and only confirms STATIC-group membership (a UUID-set lookup)
    for the few deforming candidates. Equivalent result to
    :func:`collect_capturable_static_objects` (``allow_eval=False``); the
    full list builder used by ``execute`` still resolves per assignment.
    """
    import bpy
    from ...core.uuid_registry import get_object_uuid
    from ...models.groups import iterate_active_object_groups

    static_uuids = {
        getattr(assigned, "uuid", "")
        for group in iterate_active_object_groups(context.scene)
        if group.object_type == "STATIC"
        for assigned in group.assigned_objects
        if getattr(assigned, "uuid", "")
    }
    if not static_uuids:
        return False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not object_needs_deformation_capture(obj, context):
            continue
        if get_object_uuid(obj) in static_uuids:
            return True
    return False


def start_capture_for_objects(context, objects: list) -> tuple[bool, str]:
    """Build entries for *objects* and start the shared capture job.

    Public entry point for the "Re-capture All Deformations" coordinator;
    returns ``(ok, error_message)`` like :func:`_start_job`.
    """
    errors: list = []
    entries = _build_entries(context.scene, objects, errors)
    if not entries:
        return False, (
            errors[0] if errors else iface_("No deforming STATIC objects to capture")
        )
    return _start_job(context, entries)


def advance_capture(context) -> tuple[bool, bool, str]:
    """Tick the running capture once.

    Returns ``(more_remaining, aborted, error)``. The coordinator calls this
    each timer tick and stops the whole run when ``aborted`` is True (e.g.
    the user hit the Abort button, which sets the job's aborted flag).
    """
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    more = _tick_job(context)
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    return more, False, ""


def finalize_capture(context) -> tuple[int, int]:
    """Write completed entries to PC2; returns ``(objects, frames)`` written."""
    return _finalize_job(context)


def cleanup_capture(context) -> None:
    """Restore frame, resume suspended caches, and clear the job state."""
    _cleanup_after_job(context)


def collect_objects_with_deform_cache(context) -> list:
    """Every STATIC-group object that currently holds a static-deform cache,
    deduplicated. Used by "Clear All Deformations" (and its poll). Draw-safe:
    only checks for an existing cache, no depsgraph sampling."""
    from ...core.uuid_registry import resolve_assigned
    from ...models.groups import iterate_active_object_groups

    out: list = []
    seen: set = set()
    for group in iterate_active_object_groups(context.scene):
        if group.object_type != "STATIC":
            continue
        for assigned in group.assigned_objects:
            try:
                obj = resolve_assigned(assigned)
            except ValueError:
                continue
            if obj is None or obj.type != "MESH" or id(obj) in seen:
                continue
            if object_has_deformation_cache(obj):
                seen.add(id(obj))
                out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


classes = (
    OBJECT_OT_CaptureStaticDeformation,
    OBJECT_OT_ClearStaticDeformation,
    SOLVER_OT_CaptureAbort,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
