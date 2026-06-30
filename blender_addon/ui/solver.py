# File: solver.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os

import bpy  # pyright: ignore

from bpy.props import BoolProperty  # pyright: ignore
from bpy.types import (  # pyright: ignore
    Operator,
    Panel,
)

from ..core.animation import (
    clear_animation_data,
    prepare_animation_targets,
)
from ..core import encode_progress
from ..core.async_op import AsyncOperator, StageAbort
from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import is_server_busy_from_response as is_running
from ..core.encoder import prepare_upload
from ..core.module import Cbor2NotInstalledError
from ..core.encoder.mesh import compute_data_hash, encode_obj_with_hash
from ..core.encoder.params import compute_param_hash, encode_param_with_hash
from ..core.pc2 import (
    MODIFIER_NAME,
    get_pc2_dir,
    get_pc2_path,
    has_mesh_cache,
    object_pc2_key_readonly,
)
from ..core.uuid_registry import get_object_uuid
from ..core.utils import (
    get_category_name,
    redraw_all_areas,
)
from ..models.groups import (
    get_addon_data,
    has_addon_data,
    has_simulatable_dynamics,
    iterate_active_object_groups,
)
from .dynamics.bake_ops import bake_progress_snapshot, is_bake_running
from .dynamics.pin_capture_ops import (
    is_pin_capture_running,
    pin_capture_progress_snapshot,
)
from .dynamics.static_deform_ops import (
    capture_progress_snapshot,
    is_capture_running,
)

# Shared wire payload and user-facing message for a remote-data delete.
# Referenced by both TransferRequestMixin.request_delete and
# SOLVER_OT_DeleteRemoteData.execute so the literals stay in one place.
_DELETE_QUERY = {"request": "delete"}
_DELETE_MESSAGE = "Deleting Remote Data..."

# Character width of the ASCII progress-bar fallback used when the Blender
# build lacks the native ``UILayout.progress`` widget. Shared by the bake,
# capture, and pin-capture progress blocks via ``_ascii_bar``.
_ASCII_BAR_WIDTH = 20


def _ascii_bar(factor: float) -> str:
    """Render a ``factor`` in ``[0, 1]`` as a fixed-width ASCII progress bar
    like ``[#####...............]`` for builds without ``progress``."""
    filled = int(round(factor * _ASCII_BAR_WIDTH))
    return "[" + "#" * filled + "." * (_ASCII_BAR_WIDTH - filled) + "]"


def _draw_progress(layout, done, total, status_text, icon, abort_op):
    """Draw one progress block (status label, native ``progress`` bar with an
    ASCII fallback, and an abort operator) into a fresh sub-box of ``layout``.

    Shared by the bake, capture, and pin-capture sites so the divide-by-zero
    guard and the ``progress``-widget fallback live in one place.
    """
    factor = (done / total) if total > 0 else 0.0
    prog_box = layout.box()
    prog_box.label(text=status_text, icon=icon)
    label_text = f"{int(factor * 100)}% ({done}/{total})"
    try:
        prog_box.progress(factor=factor, type="BAR", text=label_text)
    except (AttributeError, TypeError):
        bar = _ascii_bar(factor)
        prog_box.label(text=f"{bar} {label_text}")
    prog_box.operator(abort_op, icon="X")


def _find_missing_pc2_paths(context) -> list[str]:
    """Every referenced-but-missing PC2 path, in deterministic order.

    Two kinds of reference:
      * MESH with ``ContactSolverCache``: ``mod.filepath`` resolved via
        ``bpy.path.abspath`` — the file Blender plays back.
      * CURVE assigned to an active simulation group: the basename-
        derived ``get_pc2_path(key)`` — curves have no modifier, playback
        reads this path directly.

    Gated on ``bpy.data.filepath`` (path resolution is meaningless for
    an unsaved scene) and on at least one mesh modifier existing
    anywhere (proxy for "a sim has been run"), otherwise absent PC2
    files are the expected state for a fresh project.
    """
    if not bpy.data.filepath:
        return []

    # Fast common-case gate: this warning exists for the "data folder
    # moved/gone" case (a rename or OS-copy of the .blend leaves the current
    # project's PC2 directory empty). So if that directory holds ANY cache
    # file, the data is present and nothing is missing -- return immediately
    # without touching a single object. ``os.scandir`` + ``any`` early-exits
    # on the first ``.pc2`` entry, so this is ~one cheap syscall instead of
    # the O(objects) scan below, which now runs ONLY in the rare folder-gone
    # state (exactly when the enumerated list is actually wanted for display).
    try:
        with os.scandir(get_pc2_dir()) as _it:
            if any(_e.name.endswith(".pc2") for _e in _it):
                return []
    except OSError:
        pass  # directory absent -> fall through and enumerate what's missing

    # Collect every referenced PC2 path in deterministic order first, then
    # resolve existence with ONE directory listing per referenced directory
    # rather than an ``os.path.exists`` per file.
    candidates: list[str] = []
    saw_modifier = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or not mod.filepath:
            continue
        saw_modifier = True
        # Modifier's stored filepath — what MESH_CACHE plays back from.
        candidates.append(bpy.path.abspath(mod.filepath))
        # Basename-derived canonical path — what the overlay/heal/bake
        # code paths read. Diverges from mod.filepath after a rename or
        # OS-copy of the .blend (mod.filepath still points at the old
        # data/<basename>/ folder, canonical now targets the new one).
        # Read-only key: draw must never mutate, and the side-effecting
        # variant runs an O(N) duplicate scan per object (O(N^2) overall,
        # ~0.6s on a 1.6k-object scene). An object that owns a cache was
        # already assigned a UUID at capture time, so an empty key here
        # means "no cache" and the canonical check is skipped.
        key = object_pc2_key_readonly(obj)
        if key:
            candidates.append(get_pc2_path(key))

    if not saw_modifier:
        return []

    active_uuids = {
        assigned.uuid
        for group in iterate_active_object_groups(context.scene)
        for assigned in group.assigned_objects
        if assigned.uuid
    }
    for obj in bpy.data.objects:
        if obj.type != "CURVE":
            continue
        uid = get_object_uuid(obj)
        if not uid or uid not in active_uuids:
            continue
        candidates.append(get_pc2_path(uid))

    # Resolve existence per directory: one os.listdir() per unique dir, then
    # membership in that set (an absent dir lists as empty -> all missing,
    # matching the prior os.path.exists semantics). Order/dedup preserved so
    # ``missing[0]`` stays the first referenced path.
    listings: dict[str, set] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        directory = os.path.dirname(path)
        names = listings.get(directory)
        if names is None:
            try:
                names = set(os.listdir(directory))
            except OSError:
                names = set()
            listings[directory] = names
        if os.path.basename(path) not in names:
            missing.append(path)

    return missing


def _detect_pc2_basename_mismatch(context):
    """``(old_name, new_name)`` when every ``ContactSolverCache`` modifier
    points into a single ``data/<old_name>/`` folder that differs from
    the current .blend basename, else ``None``.

    Fires only on the unambiguous "rename / OS-copy of the .blend"
    shape: one old prefix across all modifiers, current basename is
    the new one, old folder still on disk (so a migration can actually
    move or copy it). Ambiguous states (mixed prefixes, modifiers
    pointing outside ``<blend_dir>/data/``) return ``None``.
    """
    if not bpy.data.filepath:
        return None
    blend_dir = os.path.dirname(bpy.data.filepath)
    current = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    expected_root = os.path.realpath(os.path.join(blend_dir, "data"))

    old_names: set[str] = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or not mod.filepath:
            continue
        abs_path = os.path.realpath(bpy.path.abspath(mod.filepath))
        parent = os.path.dirname(abs_path)
        if os.path.dirname(parent) != expected_root:
            continue
        name = os.path.basename(parent)
        if name:
            old_names.add(name)

    if len(old_names) != 1:
        return None
    (old_name,) = old_names
    if old_name == current:
        return None
    if not os.path.isdir(os.path.join(blend_dir, "data", old_name)):
        return None
    return (old_name, current)


def _warn_if_mesh_topology_stale(op, context) -> None:
    """Report a WARNING (and log to the Console) when the mesh hash
    stored at last transfer no longer matches the current scene.

    Doesn't block the operation — transfers must remain in the user's
    control — but makes the divergence visible instead of silently
    running the sim on stale geometry."""
    try:
        state = get_addon_data(context.scene).state
        msg = state.validate_mesh_hash(context)
    except Exception:
        return
    if msg:
        op.report({"WARNING"}, msg)
        try:
            from ..models.console import console
            console.write(f"[Mesh hash] {msg}")
        except Exception:
            pass


def _staged_encode_active() -> bool:
    """True while a Transfer/Run staged encode is running its modal stages.

    ``AsyncOperator.start_stages`` publishes an encode-progress run
    synchronously inside ``execute`` but only dispatches to the engine on its
    final stage, so ``engine.state.busy`` (which the action polls otherwise
    rely on to disable each other) stays False for the whole encode window.
    The action operators add ``not _staged_encode_active()`` to their poll so
    a second Transfer / Run / Update Params / Resume cannot start while an
    encode is still staging.
    """
    return encode_progress.is_active()


def _check_project_name_sync(context) -> str:
    """Detect project-name drift between UI and the active connection.

    The runner captures the project name at Connect. Later writes to
    ``state.project_name`` (e.g. MCP ``set_scene_parameters``) do not resync,
    so a transfer would land under the stale name. Returns an error string,
    or empty string if the two are in sync.
    """
    state = get_addon_data(context.scene).state
    ui_name = (state.project_name or "").strip()
    runner_name = (com.project_name or "").strip()
    if ui_name == runner_name:
        return ""
    return (
        f"Project name out of sync: UI='{ui_name}' but active session='{runner_name}'. "
        "Disconnect and reconnect to resync."
    )


def _check_uuid_consistency(context) -> str:
    """Check UUID consistency for all assigned objects.

    Returns an error message string, or empty string if all OK.
    """
    from ..core.uuid_registry import get_object_uuid, resolve_assigned

    missing_uuid = []
    stale_uuid = []
    for group in iterate_active_object_groups(context.scene):
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is None:
                continue
            obj_uuid = get_object_uuid(obj)
            if not obj_uuid:
                missing_uuid.append(obj.name)
            elif assigned.uuid != obj_uuid:
                stale_uuid.append(obj.name)
    if missing_uuid:
        names = ", ".join(missing_uuid[:3])
        more = f" (+{len(missing_uuid) - 3} more)" if len(missing_uuid) > 3 else ""
        return f"Objects missing UUID: {names}{more}. Run UUID Migration first."
    if stale_uuid:
        names = ", ".join(stale_uuid[:3])
        more = f" (+{len(stale_uuid) - 3} more)" if len(stale_uuid) > 3 else ""
        return f"Stale UUID references: {names}{more}. Run UUID Migration first."
    # Check if broader migration is needed (pins, pairs, keyframes)
    from ..core.migrate import needs_migration
    return needs_migration()


class TransferRequestMixin:
    """Shared request methods for transfer operators.

    Data and param ship together through a single ``upload_atomic``
    transaction so the server cannot observe a mismatched (data, param)
    pair mid-build. Upload sites call ``core.encoder.prepare_upload`` for
    ``(data, param, data_hash, param_hash)`` (which also stamps the local
    cache), then dispatch via ``com.build_pipeline`` or ``com.upload_only``.
    """

    def request_delete(self):
        self._mode = "delete"
        com.query(_DELETE_QUERY, _DELETE_MESSAGE)


class SOLVER_PT_SolverPanel(Panel):
    bl_label = "Solver"
    bl_idname = "SSH_PT_SolverPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        root = get_addon_data(context.scene)
        state = root.state
        layout = self.layout
        has_dynamic = has_simulatable_dynamics(context.scene)
        box = layout.box()
        row = box.row()
        row.operator(SOLVER_OT_Transfer.bl_idname, icon="EXPORT")
        row.operator(SOLVER_OT_UpdateParams.bl_idname, icon="OPTIONS")
        row = box.row()
        row.operator(SOLVER_OT_Run.bl_idname, icon="PLAY")
        row.operator(SOLVER_OT_ResumeFrom.bl_idname, icon="PREV_KEYFRAME")

        row = box.row()
        row.operator(SOLVER_OT_FetchData.bl_idname, icon="IMPORT")
        row.operator(SOLVER_OT_DeleteRemoteData.bl_idname, icon="TRASH")

        row = box.row()
        row.operator("solver.bake_all_animation", icon="ACTION")
        row.operator("solver.bake_all_single_frame", icon="KEYFRAME")

        row = box.row()
        row.operator(SOLVER_OT_ClearAnimation.bl_idname, icon="OBJECT_ORIGIN")

        # Surface missing PC2 files so the user can copy data/<basename>/
        # into place after a rename/duplicate. File-level check (not dir)
        # because an empty data/ directory can still exist.
        missing_pc2 = _find_missing_pc2_paths(context)
        if missing_pc2:
            row = box.row()
            row.label(
                text=f"Data path: {missing_pc2[0]} does not exist.",
                icon="ERROR",
            )
            mismatch = _detect_pc2_basename_mismatch(context)
            if mismatch:
                old_name, new_name = mismatch
                row = box.row()
                row.operator(
                    SOLVER_OT_MigratePC2Folder.bl_idname,
                    icon="FILE_REFRESH",
                    text=f"Migrate data/{old_name}/ \u2192 data/{new_name}/",
                )
            list_box = box.box()
            list_box.label(text=f"Missing files ({len(missing_pc2)}):")
            col = list_box.column(align=True)
            for path in missing_pc2:
                col.label(text=path)

        if is_bake_running():
            done, total, status, _ = bake_progress_snapshot()
            _draw_progress(
                box, done, total, status or "Baking...", "TIME", "solver.bake_abort"
            )

        if is_capture_running():
            done, total, status, _ = capture_progress_snapshot()
            _draw_progress(
                box,
                done,
                total,
                status or "Capturing...",
                "REC",
                "solver.capture_abort",
            )

        if is_pin_capture_running():
            done, total, status, _ = pin_capture_progress_snapshot()
            _draw_progress(
                box,
                done,
                total,
                status or "Capturing pin...",
                "REC",
                "solver.pin_capture_abort",
            )

        # Show missing frames warning (not during simulation)
        response = com.info.response
        remote_frames = int(response.get("frame", 0)) if response else 0
        if remote_frames > 0 and not is_running(response):
            fetched = state.convert_fetched_frames_to_list()
            n_missing = remote_frames - len(fetched)
            if n_missing > 0:
                row = box.row()
                row.label(text=f"{n_missing} frames unfetched. Press \"Fetch All Animation\".", icon="ERROR")

        if not has_dynamic:
            layout.label(text="No dynamics to simulate", icon="ERROR")

        if com.info.status == RemoteStatus.WAITING_FOR_DATA and has_dynamic and SOLVER_OT_Transfer.poll(context):
            row = box.row()
            row.label(text="Click \"Transfer\" to upload data", icon="INFO")

        # Warn when local animation is the only blocker for Run
        if SOLVER_OT_ClearAnimation.poll(context):
            status = com.info.status
            response = com.info.response
            run_ready = (
                not com.busy()
                and status in (
                    RemoteStatus.READY,
                    RemoteStatus.RESUMABLE,
                    RemoteStatus.SIMULATION_FAILED,
                )
                and com.is_connected()
                and com.is_server_running()
                and status.ready()
                and not is_running(response)
            )
            if run_ready:
                row = box.row()
                row.label(text="Clear local animation before running", icon="INFO")

        # Deformation caches: batch re-capture / clear of every STATIC-collider
        # deform cache and every animated-pin capture. Plain (always-expanded)
        # box so both actions are always visible; each button's poll() greys it
        # out when there is nothing to capture / clear.
        box = layout.box()
        box.label(text="Deformations", icon="MOD_SIMPLEDEFORM")
        box.operator(
            SOLVER_OT_RecaptureAllDeformations.bl_idname, icon="FILE_REFRESH"
        )
        box.operator(SOLVER_OT_ClearAllDeformations.bl_idname, icon="TRASH")

        # JupyterLab expandable box
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(state, "show_jupyter", icon="TRIA_DOWN" if state.show_jupyter else "TRIA_RIGHT", emboss=False, text="")
        row.label(text="JupyterLab", icon="FILE_SCRIPT")
        if state.show_jupyter:
            if not com.is_connected():
                box.label(text="Connect to server to export/delete", icon="INFO")
            row = box.row(align=True)
            row.operator("solver.jupyter_export", icon="EXPORT")
            row.operator("solver.jupyter_open", icon="URL")
            row.operator("solver.jupyter_delete", icon="TRASH")
            row = box.row(align=True)
            row.operator("debug.transfer_without_build", icon="ARROW_LEFTRIGHT", text="Transfer without Build")
            row.operator("debug.build", icon="MOD_BUILD")
            row = box.row()
            row.prop(state, "jupyter_port", text="Port")

        # MCP Server section
        from ..mcp.mcp_server import is_mcp_running
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(
            state, "show_mcp",
            icon="TRIA_DOWN" if state.show_mcp else "TRIA_RIGHT",
            emboss=False, text="",
        )
        mcp_label = f"MCP Server (Running :{state.mcp_port})" if is_mcp_running() else "MCP Server (Stopped)"
        row.label(text=mcp_label, icon="NETWORK_DRIVE")
        if state.show_mcp:
            from .mcp_ops import MCP_OT_StartServer, MCP_OT_StopServer
            row = box.row(align=True)
            mcp_running = is_mcp_running()
            if mcp_running:
                row.operator(MCP_OT_StopServer.bl_idname, icon="PAUSE")
            else:
                row.operator(MCP_OT_StartServer.bl_idname, icon="PLAY")
            sub = row.row()
            sub.enabled = not mcp_running
            sub.prop(state, "mcp_port", text="Port")


class SOLVER_OT_Transfer(AsyncOperator):
    """Upload the scene to the solver, overwriting data.pickle and
    param.pickle without touching cached artifacts (tetrahedralization,
    BVH, mesh caches). Use ``Delete Remote Data`` to wipe everything."""

    bl_idname = "solver.transfer"
    bl_label = "Transfer"

    _mode = None
    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        response = com.info.response
        has_dynamic = has_simulatable_dynamics(context.scene)
        # ``status.ready()`` is a protocol-version check only; it
        # returns True for every active operation (BUILDING, FETCHING,
        # STARTING_SOLVER, ...). ``status.in_progress()`` is the
        # explicit "server is doing something the user shouldn't
        # interrupt" set. ``not is_running(response)`` only inspects
        # the cached server response, which lags by one PollTick after
        # a click, so STARTING_SOLVER right after Run would otherwise
        # let Transfer stay clickable for a window.
        return (
            has_dynamic
            and not com.busy()
            and not _staged_encode_active()
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not com.info.status.in_progress()
            and not is_running(response)
            and com.connection.remote_root
        )

    def execute(self, context):
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        error = _check_uuid_consistency(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        # Transfer is upload-only: it overwrites data.pickle and
        # param.pickle atomically and lets the server's build pipeline
        # decide what cached artifacts (tetrahedralization, BVH, mesh
        # caches) remain valid for the new payload. To wipe everything,
        # the user has a dedicated "Delete Remote Data" button.
        #
        # The scene encode runs on the main thread and can take seconds on
        # heavy scenes, so split it across staged modal ticks: the panel
        # shows a labeled progress bar from the moment of the click instead
        # of a frozen cursor, and the geometry / parameter / upload phases
        # are each named as they run. See AsyncOperator.start_stages.
        self._mode = "pipeline"
        self._payload = {}
        self.start_stages(context, [
            ("Encoding scene geometry...", self._stage_encode_geometry),
            ("Encoding parameters...", self._stage_encode_params),
            ("Uploading scene...", self._stage_upload),
        ])
        return {"RUNNING_MODAL"}

    def _stage_encode_geometry(self, context):
        try:
            data, data_hash = encode_obj_with_hash(context)
        except (Cbor2NotInstalledError, ValueError) as e:
            com.set_error(str(e))
            raise StageAbort(str(e))
        self._payload["data"] = data
        self._payload["data_hash"] = data_hash

    def _stage_encode_params(self, context):
        try:
            param, param_hash = encode_param_with_hash(context)
        except (Cbor2NotInstalledError, ValueError) as e:
            com.set_error(str(e))
            raise StageAbort(str(e))
        self._payload["param"] = param
        self._payload["param_hash"] = param_hash

    def _stage_upload(self, context):
        # Local fetched animation is keyed by the previous upload's data;
        # new data invalidates it, so drop it before kicking the pipeline
        # (matches the prior NO_DATA fast-path behavior).
        com.animation.clear()
        com.build_pipeline(
            data=self._payload["data"], param=self._payload["param"],
            data_hash=self._payload["data_hash"],
            param_hash=self._payload["param_hash"],
            message="Uploading scene...",
        )

    def is_complete(self) -> bool:
        if self._mode != "pipeline":
            return False
        if com.busy():
            return False
        return com.info.status != RemoteStatus.BUILDING

    def is_cancelled(self) -> bool:
        return com.is_aborting()

    def on_complete(self, context):
        from ..models.console import console as _console
        _console.write(f"[Transfer] complete: {com.info.status.value}")
        self.report({"INFO"}, "Build completed successfully.")


class SOLVER_OT_Run(AsyncOperator):
    """Run the solver."""

    bl_idname = "solver.run"
    bl_label = "Run"

    timeout: float = 86400.0  # 24 hours for long simulations
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        base = (
            not com.busy()
            and not _staged_encode_active()
            and status in (
                RemoteStatus.READY,
                RemoteStatus.RESUMABLE,
                RemoteStatus.SIMULATION_FAILED,
            )
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and not SOLVER_OT_ClearAnimation.poll(context)
        )
        return base

    def invoke(self, context, event):
        # Guard against running on an emulated (CPU stub, no CUDA) server:
        # that build is for the test rig and produces no real physics.
        # The flag is mirrored from the server's ``hardware.emulated`` onto
        # AppState on every status poll. Only the interactive button path
        # reaches invoke(); MCP / headless callers go straight to execute()
        # (EXEC context), so the rig is never blocked by a dialog.
        from ..core.facade import engine
        if engine.state.emulated:
            return context.window_manager.invoke_props_dialog(
                self, width=420, title="Emulation Mode", confirm_text="Run Anyway",
            )
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="The server is running in EMULATION mode.", icon="ERROR")
        col.label(text="The CPU stub backend (test rig only) produces")
        col.label(text="no real physics. Are you sure you want to run?")

    def execute(self, context):
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        # The click-time drift check re-encodes the scene (compute_data_hash),
        # which is the same heavy main-thread work as Transfer, so stage it
        # behind a labeled progress bar instead of freezing on the click. The
        # final stage starts the solver; after it the modal waits on the sim.
        self.start_stages(context, [
            ("Checking scene geometry...", self._stage_check_geometry),
            ("Checking parameters...", self._stage_check_params),
            ("Starting solver...", self._stage_start_run),
        ])
        return {"RUNNING_MODAL"}

    def _stage_check_geometry(self, context):
        # Refuse the run when the live encoded data no longer matches what
        # the server echoed on its last status response. ``poll`` deliberately
        # ignores hashes so the button is always actionable; the user lands
        # here when their geometry edits haven't been pushed yet.
        try:
            local_data = compute_data_hash(context)
        except ValueError as e:
            raise StageAbort(str(e))
        from ..core.facade import engine
        server_data = engine.state.server_data_hash
        if server_data and local_data != server_data:
            raise StageAbort(
                "Geometry has changed since the last transfer. "
                "Click \"Transfer\" to re-upload before running."
            )

    def _stage_check_params(self, context):
        local_param = compute_param_hash(context)
        from ..core.facade import engine
        server_param = engine.state.server_param_hash
        if server_param and local_param != server_param:
            raise StageAbort(
                "Parameters have changed since the last transfer. "
                "Click \"Update Params\" before running."
            )

    def _stage_start_run(self, context):
        prepare_animation_targets(context, clear_existing=True)
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        com.run(context)

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )


class SOLVER_OT_Resume(AsyncOperator):
    """Resume the solver."""

    bl_idname = "solver.resume"
    bl_label = "Resume"

    timeout: float = 86400.0  # 24 hours for long simulations
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        # Resume stays available not only in the normal RESUMABLE state but also
        # after a failed simulation, as long as the server still holds at least
        # one saved checkpoint to load from. saved_state_frames() reads the
        # cached status response (no network), so this is cheap for poll().
        return (
            not com.busy()
            and not _staged_encode_active()
            and status in (RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and len(com.saved_state_frames()) > 0
        )

    def execute(self, context):
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        com.resume(context)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )


class SOLVER_OT_ResumeFrom(AsyncOperator):
    """Resume the simulation from a chosen saved checkpoint.

    Opens a dialog listing the server's saved checkpoint frames. On
    confirm it guards against geometry/topology drift (a fresh run is
    required when geometry changed), warns when only materials changed
    (those will not apply on resume), re-encodes the parameters with a
    preserve-output build that keeps the already-simulated frames, and
    once the build completes resumes the solver from the chosen frame.
    Frames before the chosen point are kept; the rest are overwritten.
    """

    bl_idname = "solver.resume_from"
    bl_label = "Resume"

    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        # Resume stays available not only in the normal RESUMABLE state but also
        # after a failed simulation, as long as the server still holds at least
        # one saved checkpoint to load from. saved_state_frames() reads the
        # cached status response (no network), so this is cheap for poll().
        return (
            not com.busy()
            and not _staged_encode_active()
            and status in (RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and len(com.saved_state_frames()) > 0
        )

    def invoke(self, context, event):
        # Run the drift guards BEFORE opening the checkpoint picker, so the user
        # is told to Transfer / Update Params up front instead of after choosing
        # a checkpoint. Resume does NOT re-upload or rebuild; it continues the
        # simulation already on the server from the chosen checkpoint (a rebuild
        # happens only on "Transfer" or "Update Params"). So, like Run, refuse
        # when the live encoding has drifted from what the server last echoed: a
        # geometry change invalidates the cached state entirely (a fresh
        # Transfer + Run is required), and a param change must be pushed via
        # "Update Params" (which preserves the checkpoints) before resuming.
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        # The drift check re-encodes the scene on the main thread (the same
        # heavy work as Transfer). It runs before the checkpoint picker dialog
        # opens, so the panel's encode bar is not reachable from here; drive
        # the cursor's progress indicator instead so the click is not a silent
        # freeze. The encode is the fast path now (see the encoder perf fixes),
        # so this is brief.
        wm = context.window_manager
        wm.progress_begin(0.0, 2.0)
        try:
            wm.progress_update(0.0)
            try:
                local_data = compute_data_hash(context)
            except ValueError as e:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            wm.progress_update(1.0)
            local_param = compute_param_hash(context)
            wm.progress_update(2.0)
        finally:
            wm.progress_end()
        from ..core.facade import engine
        server_data = engine.state.server_data_hash
        server_param = engine.state.server_param_hash
        if server_data and local_data != server_data:
            self.report({"ERROR"},
                        "Geometry has changed; resume is not possible. Click "
                        "\"Transfer\" and \"Run\" for a fresh simulation.")
            return {"CANCELLED"}
        if server_param and local_param != server_param:
            self.report({"ERROR"},
                        "Parameters have changed since the last transfer. Click "
                        "\"Update Params\" before resuming.")
            return {"CANCELLED"}
        state = get_addon_data(context.scene).state
        state.checkpoint_frames.clear()
        for frame in com.saved_state_frames():
            item = state.checkpoint_frames.add()
            item.frame = int(frame)
        state.checkpoint_frames_index = (
            len(state.checkpoint_frames) - 1 if len(state.checkpoint_frames) else -1
        )
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        state = get_addon_data(context.scene).state
        layout = self.layout
        if len(state.checkpoint_frames):
            layout.label(text="Resume from saved checkpoint:")
            layout.template_list(
                "SOLVER_UL_CheckpointFrames", "",
                state, "checkpoint_frames",
                state, "checkpoint_frames_index",
                rows=4,
            )
        else:
            layout.label(text="No saved checkpoints available.", icon="ERROR")

    def execute(self, context):
        # Drift guards already ran in invoke() (before the picker opened); the
        # modal dialog blocks scene edits, so nothing can drift in between. Here
        # we only resolve the chosen checkpoint and resume.
        state = get_addon_data(context.scene).state
        frames = state.convert_checkpoint_frames_to_list()
        index = int(state.checkpoint_frames_index)
        if not frames or index < 0 or index >= len(frames):
            self.report({"ERROR"}, "Select a checkpoint frame to resume from.")
            return {"CANCELLED"}
        from_frame = int(frames[index])
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        com.resume(context, from_frame=from_frame)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )

    def is_cancelled(self) -> bool:
        return com.is_aborting()


class SOLVER_OT_UpdateParams(AsyncOperator):
    """Update the parameters of the solver."""

    bl_idname = "solver.update_params"
    bl_label = "Update Params on Remote"

    _mode = None
    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        status = com.info.status
        response = com.info.response
        base = (
            (
                status == RemoteStatus.WAITING_FOR_BUILD
                or status
                in (
                    RemoteStatus.READY,
                    RemoteStatus.RESUMABLE,
                    RemoteStatus.SIMULATION_FAILED,
                )
            )
            and not _staged_encode_active()
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )
        # No drift-time gate here: enable whenever the system can
        # accept a param upload. If the live params already match what
        # the server holds, ``execute`` still uploads -- the result is
        # idempotent (the server mints a new upload_id but the hash is
        # the same), so a redundant click is at worst a no-op round
        # trip. Comparing hashes in ``poll`` requires either a
        # client-side cache (fragile -- PropertyGroup writes do not
        # fire ``depsgraph_update_post``, so the cache silently
        # freezes after a checkbox toggle) or a per-poll fresh hash
        # (stalls the UI on big scenes since Blender hits ``poll``
        # many times per redraw). Skipping the gate sidesteps both.
        return base

    def execute(self, context):
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        remote_root = com.normalized_remote_root()
        if not remote_root:
            self.report({"ERROR"}, "Not connected; cannot send parameters")
            return {"CANCELLED"}
        try:
            _, param_data, _, param_hash = prepare_upload(
                context, want_data=False, want_param=True,
            )
        except (Cbor2NotInstalledError, ValueError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        # Upload param-only atomically + auto-build. data_size=0 tells
        # the server to keep the existing data.pickle untouched while a
        # fresh upload_id is minted against the new param. The param
        # hash rides along so the server's next status response echoes
        # the new fingerprint and ``poll`` re-disables this button. We
        # leave ``data_hash`` empty here: param-only uploads don't
        # touch the server's data fingerprint, so omitting it
        # preserves whatever's already on disk.
        com.build_pipeline(
            data=b"", param=param_data,
            param_hash=param_hash,
            preserve_output=True,
            message="Updating parameters...",
        )
        self._mode = "pipeline"
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        if self._mode != "pipeline":
            return False
        if com.busy():
            return False
        return com.info.status != RemoteStatus.BUILDING

    def is_cancelled(self) -> bool:
        return com.is_aborting()

    def on_complete(self, context):
        self.report({"INFO"}, "Build completed successfully.")


class SOLVER_OT_ClearAnimation(Operator):
    """Reset the position of the object."""

    bl_idname = "solver.clear_animation"
    bl_label = "Clear Local Animation"

    @classmethod
    def poll(cls, context):
        if is_bake_running():
            return False
        if com.busy() or com.animation.frame:
            return False
        response = com.info.response
        if is_running(response):
            return False
        # Recomputed on every redraw (and again via SOLVER_OT_Run.poll, which
        # gates on this result) -- stateless, so it can never go stale the way
        # a memoized flag can. scene_has_solver_cache scans object modifiers
        # directly, which is ~100x cheaper than resolving every assigned
        # object by UUID, so the full scan stays well under a millisecond even
        # on scenes with thousands of objects.
        from ..core.pc2 import scene_has_solver_cache
        return scene_has_solver_cache()

    def execute(self, context):
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        context.scene.frame_set(1)
        clear_animation_data(context)
        return {"FINISHED"}


class SOLVER_OT_RecaptureAllDeformations(Operator):
    """Re-capture deformation for every deforming STATIC collider and every
    animated pin in one pass, so you don't have to capture each one by hand
    before Transfer. Runs the static-collider captures first, then the pin
    captures (the two share the depsgraph and can't run at once); progress
    and an Abort button appear below while it runs."""

    bl_idname = "solver.recapture_all_deformations"
    bl_label = "Re-capture All Deformations"

    _PHASE_STATIC = "STATIC"
    _PHASE_PIN = "PIN"

    @classmethod
    def poll(cls, context):
        # Disabled while any capture/bake is already in flight, and (the
        # "no capturable objects" case) when nothing across the active
        # groups needs a deformation capture. Cheap, draw-safe predicates
        # only: this runs on every panel redraw.
        if is_bake_running() or is_capture_running() or is_pin_capture_running():
            return False
        # Fast stateless poll: early-returns on the first capturable static
        # via the cached uuid resolver (no per-object name reconciliation),
        # instead of building the full list with resolve_assigned per redraw.
        from .dynamics.static_deform_ops import scene_has_capturable_static
        from .dynamics.pin_capture_ops import collect_capturable_pins
        return bool(
            scene_has_capturable_static(context)
            or collect_capturable_pins(context, cheap=True)
        )

    def execute(self, context):
        from .dynamics.static_deform_ops import collect_capturable_static_objects
        from .dynamics.pin_capture_ops import collect_capturable_pins

        static_objs = collect_capturable_static_objects(context, allow_eval=True)
        pin_specs = collect_capturable_pins(context)
        self._queue = []
        if static_objs:
            self._queue.append((self._PHASE_STATIC, static_objs))
        if pin_specs:
            self._queue.append((self._PHASE_PIN, pin_specs))
        if not self._queue:
            self.report({"WARNING"}, "No deformations to capture")
            return {"CANCELLED"}
        self._phase = None
        self._summary = {"objs": 0, "pins": 0}
        if not self._start_next(context):
            self.report({"WARNING"}, "Nothing to capture")
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        redraw_all_areas(context)
        return {"RUNNING_MODAL"}

    def _start_next(self, context) -> bool:
        """Pop the next queued phase and start its job. Returns True once a
        job is running, False when the queue is exhausted (or the remaining
        phases all failed to start, each reported as a warning)."""
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        while self._queue:
            kind, payload = self._queue.pop(0)
            if kind == self._PHASE_STATIC:
                ok, err = sd.start_capture_for_objects(context, payload)
            else:
                ok, err = pc.start_capture_for_pins(context, payload)
            if ok:
                self._phase = kind
                return True
            if err:
                self.report({"WARNING"}, err)
        self._phase = None
        return False

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        try:
            if self._phase == self._PHASE_STATIC:
                more, aborted, err = sd.advance_capture(context)
                if aborted:
                    return self._abort(context, err)
                if not more:
                    n_objs, _ = sd.finalize_capture(context)
                    self._summary["objs"] += n_objs
                    sd.cleanup_capture(context)
                    if not self._start_next(context):
                        return self._finish(context)
                redraw_all_areas(context)
                return {"RUNNING_MODAL"}
            elif self._phase == self._PHASE_PIN:
                more, aborted, err = pc.advance_pin_capture(context)
                if aborted:
                    return self._abort(context, err)
                if not more:
                    n_pins, _ = pc.finalize_pin_capture(context)
                    self._summary["pins"] += n_pins
                    pc.cleanup_pin_capture(context)
                    if not self._start_next(context):
                        return self._finish(context)
                redraw_all_areas(context)
                return {"RUNNING_MODAL"}
            return self._finish(context)
        except Exception as exc:  # noqa: BLE001 — must restore state
            self._cleanup_active(context)
            redraw_all_areas(context)
            self.report({"ERROR"}, f"Re-capture failed: {exc}")
            return self._end_timer(context, {"CANCELLED"})

    def _cleanup_active(self, context):
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        if self._phase == self._PHASE_STATIC:
            sd.cleanup_capture(context)
        elif self._phase == self._PHASE_PIN:
            pc.cleanup_pin_capture(context)
        self._phase = None

    def _abort(self, context, err):
        self._cleanup_active(context)
        redraw_all_areas(context)
        if err:
            self.report({"ERROR"}, err)
        else:
            self.report({"INFO"}, "Re-capture aborted")
        return self._end_timer(context, {"CANCELLED"})

    def _finish(self, context):
        redraw_all_areas(context)
        parts = []
        if self._summary["objs"]:
            parts.append(f"{self._summary['objs']} object(s)")
        if self._summary["pins"]:
            parts.append(f"{self._summary['pins']} pin(s)")
        target = ", ".join(parts) if parts else "nothing"
        self.report({"INFO"}, f"Re-captured deformation for {target}")
        return self._end_timer(context, {"FINISHED"})

    def _end_timer(self, context, retval):
        wm = context.window_manager
        timer = getattr(self, "_timer", None)
        if timer is not None:
            wm.event_timer_remove(timer)
            self._timer = None
        return retval

    def cancel(self, context):
        self._cleanup_active(context)
        self._end_timer(context, {"CANCELLED"})


class SOLVER_OT_ClearAllDeformations(Operator):
    """Delete every captured deformation cache in one pass: all STATIC-collider
    deform caches and all animated-pin captures across the active groups. The
    objects keep their deformers, so Re-capture All Deformations rebuilds the
    caches."""

    bl_idname = "solver.clear_all_deformations"
    bl_label = "Clear All Deformations"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        # Disabled while a capture/bake is in flight, and (the "nothing to
        # clear" case) when no cache exists across the active groups.
        if is_bake_running() or is_capture_running() or is_pin_capture_running():
            return False
        # Stateless and fast: a global static-deform-cache check (in-memory +
        # one directory scan) instead of resolving every STATIC object by
        # UUID and stat-ing its PC2 per redraw. The pin side is already cheap.
        from ..core.pc2 import scene_has_static_deform_cache
        from .dynamics.pin_capture_ops import collect_pins_with_captured_anim
        return bool(
            scene_has_static_deform_cache()
            or collect_pins_with_captured_anim(context)
        )

    def execute(self, context):
        from ..core.pc2 import remove_static_deform_pc2
        from .dynamics.static_deform_ops import collect_objects_with_deform_cache
        from .dynamics.pin_capture_ops import (
            clear_captured_pin,
            collect_pins_with_captured_anim,
        )

        n_objs = 0
        for obj in collect_objects_with_deform_cache(context):
            remove_static_deform_pc2(obj)
            n_objs += 1
        # Snapshot the pin specs before clearing (clearing only flips flags /
        # removes files, so indices stay valid, but snapshotting is clearer).
        n_pins = 0
        for gi, pi in collect_pins_with_captured_anim(context):
            if clear_captured_pin(context, gi, pi):
                n_pins += 1

        # One overlay refresh + invalidation after the whole batch.
        from .dynamics.overlay import apply_object_overlays
        from ..models.groups import invalidate_overlays
        apply_object_overlays()
        invalidate_overlays()
        redraw_all_areas(context)

        parts = []
        if n_objs:
            parts.append(f"{n_objs} object(s)")
        if n_pins:
            parts.append(f"{n_pins} pin(s)")
        target = ", ".join(parts) if parts else "nothing"
        self.report({"INFO"}, f"Cleared deformation cache for {target}")
        return {"FINISHED"}


class SOLVER_OT_MigratePC2Folder(Operator):
    """Rename or copy data/<old>/ to match the current .blend basename
    and rewrite every ContactSolverCache modifier filepath in lockstep,
    so playback and overlay both read from one consistent location."""

    bl_idname = "solver.migrate_pc2_folder"
    bl_label = "Migrate PC2 Cache"
    bl_options = {"REGISTER", "UNDO"}

    keep_copy: BoolProperty(  # pyright: ignore
        name="Keep Copy",
        description="Preserve the old data/<name>/ folder as a backup "
        "(copy instead of rename)",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return _detect_pc2_basename_mismatch(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        mismatch = _detect_pc2_basename_mismatch(context)
        if mismatch:
            old_name, new_name = mismatch
            col = layout.column(align=True)
            col.label(text=f"From:  data/{old_name}/", icon="FILE_FOLDER")
            col.label(text=f"To:    data/{new_name}/", icon="FORWARD")
        layout.prop(self, "keep_copy")

    def execute(self, context):
        import shutil

        mismatch = _detect_pc2_basename_mismatch(context)
        if mismatch is None:
            self.report({"ERROR"}, "No migration target detected.")
            return {"CANCELLED"}
        old_name, new_name = mismatch
        blend_dir = os.path.dirname(bpy.data.filepath)
        old_dir = os.path.join(blend_dir, "data", old_name)
        new_dir = os.path.join(blend_dir, "data", new_name)

        if os.path.exists(new_dir):
            if not os.path.isdir(new_dir) or os.listdir(new_dir):
                self.report(
                    {"ERROR"},
                    f"Target already exists and is non-empty, "
                    f"resolve manually: {new_dir}",
                )
                return {"CANCELLED"}
            # Empty directory — remove it so rename/copytree has a clean
            # destination (both os.rename and shutil.copytree refuse to
            # overwrite an existing dir, even an empty one).
            try:
                os.rmdir(new_dir)
            except OSError as exc:
                self.report(
                    {"ERROR"},
                    f"Could not remove empty target {new_dir}: {exc}",
                )
                return {"CANCELLED"}

        try:
            if self.keep_copy:
                shutil.copytree(old_dir, new_dir)
            else:
                os.rename(old_dir, new_dir)
        except OSError as exc:
            self.report({"ERROR"}, f"Migration failed: {exc}")
            return {"CANCELLED"}

        n_rewritten = 0
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            mod = obj.modifiers.get(MODIFIER_NAME)
            if mod is None or not mod.filepath:
                continue
            abs_path = os.path.realpath(bpy.path.abspath(mod.filepath))
            parent = os.path.dirname(abs_path)
            if os.path.basename(parent) != old_name:
                continue
            filename = os.path.basename(abs_path)
            new_abs = os.path.join(new_dir, filename)
            mod.filepath = bpy.path.relpath(new_abs)
            n_rewritten += 1

        try:
            from ..models.groups import invalidate_overlays
            invalidate_overlays()
        except Exception:
            pass

        action = "copied" if self.keep_copy else "moved"
        self.report(
            {"INFO"},
            f"Migrated {n_rewritten} modifier path(s); "
            f"data/{old_name}/ {action} to data/{new_name}/",
        )
        return {"FINISHED"}


class SOLVER_OT_FetchData(AsyncOperator):
    """Fetch data on the remote server."""

    bl_idname = "solver.fetch_remote_data"
    bl_label = "Fetch All Animation"

    timeout: float = 600.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if is_bake_running():
            return False
        if com.animation.frame:
            return False
        response = com.info.response
        frame = response.get("frame", -1)
        frame = int(frame) if frame is not None else -1
        status = com.info.status
        return (
            not com.busy()
            and frame > 0
            and status in (RemoteStatus.READY, RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )

    def execute(self, context):
        _warn_if_mesh_topology_stale(self, context)
        prepare_animation_targets(context, clear_existing=False)
        com.fetch(context)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not com.busy() and len(com.animation.frame) == 0

    def on_complete(self, context):
        redraw_all_areas(context)
        self.report({"INFO"}, "Animation data fetch finished.")

        state = get_addon_data(context.scene).state
        fetched = state.convert_fetched_frames_to_list()
        frame_count = max(fetched) if fetched else 0
        if frame_count > 0:
            context.scene.frame_end = frame_count + 1
            context.scene.frame_start = 1

        context.scene.frame_set(1)
        bpy.ops.screen.animation_play()


class SOLVER_OT_DeleteRemoteData(AsyncOperator):
    """Delete data on the remote server."""

    bl_idname = "solver.delete_remote_data"
    bl_label = "Delete Remote Data"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        response = com.info.response
        if response is None:
            return False
        data_stat = response.get("data", None)
        # Positive check: enable only when the cached server response
        # affirmatively reports READY. The earlier `!= "NO_DATA"` form
        # also passed when the cache was empty (response cleared on
        # disconnect, no poll back yet) or the field was missing, which
        # left the button live during the reconnect window before the
        # first poll lands. status.in_progress() covers the local-side
        # window right after Run/Resume/Fetch is clicked; is_running
        # only flips once the remote echoes BUSY, which can lag a tick.
        return (
            not com.busy()
            and data_stat == "READY"
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not com.info.status.in_progress()
            and not is_running(response)
        )

    def execute(self, context):
        com.query(_DELETE_QUERY, _DELETE_MESSAGE)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        response = com.info.response
        if response is None:
            return False
        return response.get("data", None) == "NO_DATA"

    def on_complete(self, context):
        redraw_all_areas(context)
        self.report({"INFO"}, "Remote data deleted successfully.")


classes = (
    SOLVER_PT_SolverPanel,
    SOLVER_OT_Transfer,
    SOLVER_OT_Run,
    SOLVER_OT_Resume,
    SOLVER_OT_ResumeFrom,
    SOLVER_OT_ClearAnimation,
    SOLVER_OT_RecaptureAllDeformations,
    SOLVER_OT_ClearAllDeformations,
    SOLVER_OT_MigratePC2Folder,
    SOLVER_OT_UpdateParams,
    SOLVER_OT_DeleteRemoteData,
    SOLVER_OT_FetchData,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
