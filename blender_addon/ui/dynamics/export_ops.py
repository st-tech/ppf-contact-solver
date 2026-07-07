# File: ui/dynamics/export_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""Export the simulated mesh sequence to USD / Alembic caches.

A lightweight alternative to the shape-key bake (``solver.bake_all_animation``).
The bake freezes every fetched PC2 frame into a per-frame shape key, which is
heavy to manage on the timeline and slow for Blender to play back. Instead,
these operators drive Blender's built-in USD / Alembic exporters over the
simulated frame range and let them sample the ``ContactSolverCache`` MESH_CACHE
deformed mesh at each frame. The result is a standard point cache that Blender
and other DCC tools can play back and retime efficiently.

Design notes (verified against Blender 5.1.2, live):
  * The MESH_CACHE modifier is set up with ``frame_start=1.0`` (core/client.py),
    so PC2 frame ``f`` shows at scene frame ``f + 1``; ``n`` cached frames occupy
    scene frames ``[1, n]``. Exporting that inclusive range samples the deform at
    each PC2 frame (USD stores sparse time samples, so a run of identical poses,
    e.g. a static tail, is written once, spanning the full range via the stage's
    start/end time codes).
  * ``evaluation_mode='VIEWPORT'`` reproduces what the user sees scrubbing the
    timeline. Under VIEWPORT eval the deform is only captured when the cache
    modifier's ``show_viewport`` is on and the object is not hidden, so we force
    both (saved/restored) for the exact target set.
  * ``bpy.ops.wm.usd_export('EXEC_DEFAULT', ...)`` runs synchronously (verified:
    the file is complete on return), so the finally-block state restore is safe.
    Alembic keeps ``as_background_job=False`` for the same reason; do NOT flip it
    on, or the restore would race the still-running export and corrupt the tail.
  * Rods (CURVE objects) deform through a frame-change handler rather than a
    MESH_CACHE modifier, and the built-in exporters do not fire it reliably, so
    they are excluded and reported as a WARNING rather than silently dropped.
"""

from __future__ import annotations

import os
import struct

import bpy  # pyright: ignore
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy_extras.io_utils import ExportHelper  # pyright: ignore

from ...core.client import communicator as com
from ...core.derived import is_server_busy_from_response as is_running
from ...core.pc2 import (
    MODIFIER_NAME,
    get_pc2_path,
    has_mesh_cache,
    object_pc2_key_readonly,
    read_pc2_frame_count,
)
from .bake_ops import _has_unfetched_frames, is_bake_running
from .pin_capture_ops import is_pin_capture_running
from .static_deform_ops import is_capture_running


# ---------------------------------------------------------------------------
# Export-set discovery
# ---------------------------------------------------------------------------


def _busy(context) -> bool:
    """True when another solver activity must lock out the export (draw-safe,
    disk-free).

    Mirrors the cache-mutating modal ops (bake / capture / pin capture) so an
    export cannot race a cache rewrite, and refuses while the server is actively
    simulating (the on-disk cache is still growing). The finer "frames were
    produced but not fetched yet" case is caught by ``_has_unfetched_frames`` in
    ``_preflight``. We deliberately do NOT gate on ``com.busy()``: it stays set
    through transient upload / apply phases and is not reliably cleared outside
    the modal loop, which would grey the button right after a run + fetch,
    exactly when the user wants to export.
    """
    if is_bake_running() or is_capture_running() or is_pin_capture_running():
        return True
    return is_running(com.info.response)


def _has_exportable_mesh(context) -> bool:
    """Cheap, disk-free predicate for ``poll``: is there at least one MESH in the
    view layer carrying a live ContactSolverCache modifier? The exact
    (disk-touching) check lives in ``execute`` / ``_exportable_meshes``."""
    for obj in context.view_layer.objects:
        if obj.type == "MESH" and obj.modifiers.get(MODIFIER_NAME) is not None:
            return True
    return False


def _exportable_meshes(context):
    """``[(obj, n_frames), ...]`` for every MESH in the view layer that has a
    live ContactSolverCache modifier AND a readable PC2 reporting > 0 frames.

    Keyed on the live modifier (not ``has_mesh_cache``, which also returns True
    for a disk-only PC2 with no deformer, or for a modifier whose file has since
    been deleted): the modifier is what actually moves the vertices the exporter
    samples. Every read is existence-guarded so a stale modifier pointing at a
    missing file is skipped instead of raising. Does disk I/O; never call from
    ``poll``.
    """
    out = []
    for obj in context.view_layer.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None:
            continue
        # Read the file the exporter will actually sample (the modifier's own
        # filepath), not the recomputed canonical UUID path: after a project
        # rename or a data/ folder copy the two can diverge, and the modifier
        # path is what drives the deform. Fall back to the canonical UUID path
        # only when the modifier carries no filepath.
        path = bpy.path.abspath(mod.filepath) if mod.filepath else ""
        if not path or not os.path.exists(path):
            key = object_pc2_key_readonly(obj)
            path = get_pc2_path(key) if key else ""
        if not path or not os.path.exists(path):
            continue
        try:
            n_frames = read_pc2_frame_count(path)
        except (OSError, struct.error):
            continue
        if n_frames > 0:
            out.append((obj, n_frames))
    return out


def _excluded_sim_curves(context):
    """Names of CURVE (rod) objects in the view layer that carry a solver cache.

    Rods deform via a frame-change handler the built-in exporters do not fire,
    so they cannot be exported faithfully through this path; we list them so the
    user is warned rather than silently losing them.
    """
    names = []
    for obj in context.view_layer.objects:
        if obj.type == "CURVE" and has_mesh_cache(obj):
            names.append(obj.name)
    return names


# ---------------------------------------------------------------------------
# Shared operator scaffolding
# ---------------------------------------------------------------------------


class _ExportSimCacheBase(ExportHelper):
    """Select / frame-range / visibility save-restore shared by both exporters.

    Subclasses set ``filename_ext`` + ``filter_glob`` and implement
    ``_run_exporter(context, start, end)`` returning the exporter's result set.
    """

    @classmethod
    def poll(cls, context):
        if _busy(context):
            return False
        return _has_exportable_mesh(context)

    def _preflight(self, context):
        """Return a user-facing error string when export can't proceed, else
        None. Mirrors the Bake operators' guards so a truncated or malformed
        cache is never written with a success toast."""
        if _busy(context):
            return "Another solver activity is in progress"
        if context.mode != "OBJECT":
            return "Exit Edit/Sculpt mode before exporting"
        if _has_unfetched_frames(context.scene):
            return (
                "Unfetched animation frames exist. "
                "Fetch all animation frames first."
            )
        if not _exportable_meshes(context):
            return "No simulated mesh sequence to export"
        return None

    def invoke(self, context, event):
        # Fail fast before opening the file browser so the user does not pick a
        # filename only to have the export refuse.
        err = self._preflight(context)
        if err is not None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        err = self._preflight(context)
        if err is not None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        pairs = _exportable_meshes(context)
        objs = [obj for obj, _ in pairs]
        counts = [n for _, n in pairs]
        max_n = max(counts)
        excluded_curves = _excluded_sim_curves(context)

        scene = context.scene
        vl = context.view_layer
        window = context.window

        # --- snapshot every piece of state we are about to touch ---
        saved_start = scene.frame_start
        saved_end = scene.frame_end
        saved_current = scene.frame_current
        saved_step = scene.frame_step
        saved_sel = [(obj, obj.select_get()) for obj in vl.objects]
        saved_active = vl.objects.active
        saved_vis = {}
        for obj in objs:
            mod = obj.modifiers.get(MODIFIER_NAME)
            saved_vis[obj] = (
                obj.hide_viewport,
                obj.hide_get(),
                mod.show_viewport if mod is not None else True,
            )

        if window is not None:
            window.cursor_set("WAIT")

        result = {"CANCELLED"}
        export_error = None
        invisible = []
        try:
            for obj in vl.objects:
                try:
                    obj.select_set(False)
                except RuntimeError:
                    pass
            for obj in objs:
                try:
                    obj.hide_viewport = False
                    obj.hide_set(False)
                    mod = obj.modifiers.get(MODIFIER_NAME)
                    if mod is not None:
                        mod.show_viewport = True
                    obj.select_set(True)
                except RuntimeError:
                    pass
            try:
                vl.objects.active = objs[0]
            except (RuntimeError, ReferenceError):
                pass
            # Under VIEWPORT eval a target still hidden by its collection
            # ("Disable in Viewports", which object-level flags can't override)
            # is dropped by the exporter. Forcing object flags above handles the
            # common case; flag whatever remains invisible so the omission is
            # reported rather than silent.
            invisible = [obj.name for obj in objs if not obj.visible_get()]
            scene.frame_start = 1
            scene.frame_end = max_n
            scene.frame_step = 1
            try:
                result = self._run_exporter(context, 1, max_n)
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean report
                export_error = str(exc)
        finally:
            if window is not None:
                window.cursor_set("DEFAULT")
            scene.frame_start = saved_start
            scene.frame_end = saved_end
            scene.frame_step = saved_step
            # Restore selection while the targets are still visible, then
            # re-apply their original hide flags.
            for obj in vl.objects:
                try:
                    obj.select_set(False)
                except RuntimeError:
                    pass
            for obj, sel in saved_sel:
                try:
                    obj.select_set(sel)
                except (RuntimeError, ReferenceError):
                    pass
            try:
                vl.objects.active = saved_active
            except (RuntimeError, ReferenceError):
                pass
            for obj, (hide_vp, hide_set, show_vp) in saved_vis.items():
                try:
                    obj.hide_viewport = hide_vp
                    obj.hide_set(hide_set)
                    mod = obj.modifiers.get(MODIFIER_NAME)
                    if mod is not None:
                        mod.show_viewport = show_vp
                except (RuntimeError, ReferenceError):
                    pass
            scene.frame_set(saved_current)

        if export_error is not None:
            self.report({"ERROR"}, f"Export failed: {export_error}")
            return {"CANCELLED"}
        if "FINISHED" not in result:
            self.report({"ERROR"}, "Export wrote nothing")
            return {"CANCELLED"}

        n_exported = len(objs) - len(invisible)
        msg = (
            f"Exported {n_exported} mesh(es), frames 1-{max_n} to "
            f"{bpy.path.abspath(self.filepath)}"
        )
        if len(set(counts)) > 1:
            msg += " (shorter caches hold their final pose past their length)"
        self.report({"INFO"}, msg)
        if invisible:
            self.report(
                {"WARNING"},
                f"{len(invisible)} object(s) hidden in the viewport were not "
                f"exported: {', '.join(invisible)}",
            )
        if excluded_curves:
            self.report(
                {"WARNING"},
                f"{len(excluded_curves)} rod/curve object(s) not exported "
                f"(unsupported by this cache export): {', '.join(excluded_curves)}",
            )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

_USD_EXTS = (".usdc", ".usda", ".usd", ".usdz")


class SOLVER_OT_ExportUSD(Operator, _ExportSimCacheBase):
    """Export the simulated mesh sequence as a USD cache.

    A lighter alternative to baking shape keys: the deformation is sampled
    per frame straight from the solver cache into a USD file that Blender and
    other DCC tools can play back and retime.
    """

    bl_idname = "solver.export_usd"
    bl_label = "Export USD"
    bl_options = {"REGISTER"}

    filename_ext = ".usdc"
    filter_glob: StringProperty(
        default="*.usd;*.usdc;*.usda;*.usdz", options={"HIDDEN"}
    )

    def check(self, context):
        # ExportHelper.check would coerce any suffix to ``.usdc`` (the USD
        # exporter picks crate/ascii/zip by extension, so that would make
        # .usda/.usdz unreachable). Keep any of the four known USD suffixes;
        # for anything else, replace the extension with the default (mirroring
        # ExportHelper.check's splitext + empty-basename guard, since
        # bpy.path.ensure_ext only appends and would otherwise double it).
        fp = self.filepath
        if not os.path.basename(fp):
            return False
        if fp.lower().endswith(_USD_EXTS):
            new_fp = fp
        else:
            new_fp = bpy.path.ensure_ext(
                os.path.splitext(fp)[0], self.filename_ext
            )
        if new_fp != self.filepath:
            self.filepath = new_fp
            return True
        return False

    def _run_exporter(self, context, start, end):
        # USD has no start/end kwargs; execute() has set scene.frame_start/end.
        return bpy.ops.wm.usd_export(
            "EXEC_DEFAULT",
            filepath=self.filepath,
            selected_objects_only=True,
            export_animation=True,
            evaluation_mode="VIEWPORT",
            check_existing=False,
            export_materials=False,
            relative_paths=False,
        )


class SOLVER_OT_ExportAlembic(Operator, _ExportSimCacheBase):
    """Export the simulated mesh sequence as an Alembic (ABC) cache.

    A lighter alternative to baking shape keys: the deformation is sampled
    per frame straight from the solver cache into an Alembic file that Blender
    and other DCC tools can play back and retime.
    """

    bl_idname = "solver.export_alembic"
    bl_label = "Export Alembic (ABC)"
    bl_options = {"REGISTER"}

    filename_ext = ".abc"
    filter_glob: StringProperty(default="*.abc", options={"HIDDEN"})

    def _run_exporter(self, context, start, end):
        # as_background_job MUST stay False: the finally-block state restore runs
        # right after this returns and would otherwise race a background job.
        return bpy.ops.wm.alembic_export(
            "EXEC_DEFAULT",
            filepath=self.filepath,
            selected=True,
            start=start,
            end=end,
            evaluation_mode="VIEWPORT",
            as_background_job=False,
            check_existing=False,
        )


classes = (
    SOLVER_OT_ExportUSD,
    SOLVER_OT_ExportAlembic,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
