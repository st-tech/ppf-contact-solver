# File: solver.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import time

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
from ..core.async_op import AsyncOperator
from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import is_server_busy_from_response as is_running
from ..core.encoder import prepare_upload
from ..core.encoder.mesh import compute_data_hash
from ..core.encoder.params import compute_param_hash
from ..core.pc2 import (
    MODIFIER_NAME,
    get_pc2_path,
    has_mesh_cache,
    object_pc2_key,
)
from ..core.uuid_registry import get_object_uuid
from ..core.utils import (
    get_category_name,
    redraw_all_areas,
)
from ..models.groups import (
    get_addon_data,
    has_addon_data,
    iterate_active_object_groups,
)
from .dynamics.bake_ops import bake_progress_snapshot, is_bake_running


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

    missing: list[str] = []
    seen: set[str] = set()

    def _push(path: str) -> None:
        if path and path not in seen:
            seen.add(path)
            missing.append(path)

    saw_modifier = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or not mod.filepath:
            continue
        saw_modifier = True
        # Modifier's stored filepath — what MESH_CACHE plays back from.
        abs_path = bpy.path.abspath(mod.filepath)
        if not os.path.exists(abs_path):
            _push(abs_path)
        # Basename-derived canonical path — what the overlay/heal/bake
        # code paths read. Diverges from mod.filepath after a rename or
        # OS-copy of the .blend (mod.filepath still points at the old
        # data/<basename>/ folder, canonical now targets the new one).
        canonical = get_pc2_path(object_pc2_key(obj))
        if not os.path.exists(canonical):
            _push(canonical)

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
        curve_path = get_pc2_path(object_pc2_key(obj))
        if not os.path.exists(curve_path):
            _push(curve_path)

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

    The old pattern (separate data_send then param_send then build) is
    retired; both files now ship through a single ``upload_atomic``
    transaction so the server can't observe a mismatched (data, param)
    pair mid-build. Upload sites call ``core.encoder.prepare_upload``
    to get ``(data, param, data_hash, param_hash)`` from a single
    canonical compute that also stamps the local cache, then dispatch
    via ``com.build_pipeline`` or ``com.upload_only``.
    """

    def request_delete(self):
        self._mode = "delete"
        com.query({"request": "delete"}, "Deleting Remote Data...")


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
        has_dynamic = any(
            group.object_type in ("SOLID", "SHELL", "ROD")
            and len(group.assigned_objects) > 0
            for group in iterate_active_object_groups(context.scene)
        )
        box = layout.box()
        row = box.row()
        row.operator(SOLVER_OT_Transfer.bl_idname, icon="EXPORT")
        row.operator(SOLVER_OT_UpdateParams.bl_idname, icon="OPTIONS")
        row = box.row()
        row.operator(SOLVER_OT_Run.bl_idname, icon="PLAY")
        row.operator(SOLVER_OT_Resume.bl_idname, icon="PLAY")

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
            done, total, status, n_objs = bake_progress_snapshot()
            factor = (done / total) if total > 0 else 0.0
            prog_box = box.box()
            prog_box.label(text=status or "Baking...", icon="TIME")
            label_text = f"{int(factor * 100)}% ({done}/{total})"
            try:
                prog_box.progress(factor=factor, type="BAR", text=label_text)
            except (AttributeError, TypeError):
                filled = int(round(factor * 20))
                bar = "[" + "#" * filled + "." * (20 - filled) + "]"
                prog_box.label(text=f"{bar} {label_text}")
            prog_box.operator("solver.bake_abort", icon="X")

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


class SOLVER_OT_Transfer(TransferRequestMixin, AsyncOperator):
    """Transfer data to the solver."""

    bl_idname = "solver.transfer"
    bl_label = "Transfer"

    _mode = None
    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        response = com.info.response
        has_dynamic = any(
            group.object_type in ("SOLID", "SHELL", "ROD")
            and len(group.assigned_objects) > 0
            for group in iterate_active_object_groups(context.scene)
        )
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
        status = com.info.status
        # Skip the "Deleting Remote Data..." round-trip when the cached
        # server response already says NO_DATA. The delete-cycle was a
        # blanket precaution for the legacy case where RemoteStatus said
        # NO_DATA but stale data.pickle / app_state still sat on disk —
        # the protocol-0.03 server reports ``data="NO_DATA"`` straight
        # off the filesystem (has_data && has_param in select_project),
        # so trusting it here saves a round-trip plus a misleading
        # status banner. ``remote_root`` must already be populated; if
        # not, fall back to the delete cycle which establishes it.
        response = com.info.response
        if (response and response.get("data") == "NO_DATA"
                and com.connection.remote_root):
            self.report({"INFO"}, "No remote data to delete; uploading scene.")
            try:
                data, param, data_hash, param_hash = prepare_upload(context)
            except ValueError as e:
                self.report({"ERROR"}, str(e))
                com.error = str(e)
                return {"CANCELLED"}
            com.animation.clear()
            com.build_pipeline(
                data=data, param=param,
                data_hash=data_hash, param_hash=param_hash,
                message="Uploading scene...",
            )
            self._mode = "pipeline"
            self.setup_modal(context)
            return {"RUNNING_MODAL"}
        if status in (
            RemoteStatus.READY,
            RemoteStatus.RESUMABLE,
            RemoteStatus.SIMULATION_FAILED,
        ):
            self.report(
                {"INFO"},
                "Remote data already exists. Deleting it before transfer.",
            )
        else:
            self.report({"INFO"}, "Clearing remote project state before transfer.")
        self.request_delete()
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        if self._mode != "pipeline":
            return False
        if com.busy():
            return False
        return com.info.status != RemoteStatus.BUILDING

    def on_complete(self, context):
        self.report({"INFO"}, "Build completed successfully.")

    def modal(self, context, event):
        from ..models.console import console as _console
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if com.is_aborting():
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if time.time() - self._start_time > self.timeout:
            self.cleanup_modal(context)
            self.on_timeout(context)
            redraw_all_areas(context)
            return {"CANCELLED"}

        redraw_all_areas(context)
        if self._mode == "delete":
            # Wait for the delete query to complete before dispatching the
            # upload. If activity is still EXECUTING, the BuildPipelineRequested
            # transition's ``state.can_operate`` guard would silently drop
            # the event and the modal would report "complete" on stale
            # status.
            if com.busy():
                return {"PASS_THROUGH"}
            data_stat = com.info.response.get("data", None)
            if data_stat == "NO_DATA":
                if not com.connection.remote_root:
                    # Root not yet refreshed after delete — wait for next poll
                    return {"PASS_THROUGH"}
                self.report({"INFO"}, "Remote data deleted successfully.")
                try:
                    data, param, data_hash, param_hash = prepare_upload(context)
                except ValueError as e:
                    self.report({"ERROR"}, str(e))
                    com.error = str(e)
                    self.cleanup_modal(context)
                    redraw_all_areas(context)
                    return {"CANCELLED"}
                com.animation.clear()
                _console.write("[Transfer] uploading scene...")
                com.build_pipeline(
                    data=data, param=param,
                    data_hash=data_hash,
                    param_hash=param_hash,
                    message="Uploading scene...",
                )
                self._mode = "pipeline"

        if self.is_complete():
            _console.write(f"[Transfer] complete: {com.info.status.value}")
            self.cleanup_modal(context)
            self.on_complete(context)
            return {"FINISHED"}
        return {"PASS_THROUGH"}


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

    def execute(self, context):
        _warn_if_mesh_topology_stale(self, context)
        # Click-time consistency check: refuse the run when the live
        # encoded data or params no longer match what the server
        # echoed on its last status response. This is the *only*
        # drift gate -- ``poll`` deliberately ignores hashes so the
        # button is always actionable, and the user lands on this
        # report when their edits haven't been pushed yet. "data"
        # diverges -> Transfer; "param" diverges -> Update Params.
        try:
            local_data = compute_data_hash(context)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        local_param = compute_param_hash(context)
        from ..core.facade import engine
        server_data = engine.state.server_data_hash
        server_param = engine.state.server_param_hash
        if server_data and local_data != server_data:
            self.report({"ERROR"},
                        "Geometry has changed since the last transfer. "
                        "Click \"Transfer\" to re-upload before running.")
            return {"CANCELLED"}
        if server_param and local_param != server_param:
            self.report({"ERROR"},
                        "Parameters have changed since the last transfer. "
                        "Click \"Update Params\" before running.")
            return {"CANCELLED"}
        prepare_animation_targets(context, clear_existing=True)
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        com.run(context)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

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
        return (
            not com.busy()
            and status == RemoteStatus.RESUMABLE
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )

    def execute(self, context):
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


class SOLVER_OT_UpdateParams(AsyncOperator):
    """Update the parameters of the solver."""

    bl_idname = "solver.update_params"
    bl_label = "Update Params on Remote"

    _mode = "param"
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
        remote_root = com.connection.remote_root.rstrip("/")
        if not remote_root:
            self.report({"ERROR"}, "Not connected; cannot send parameters")
            return {"CANCELLED"}
        try:
            _, param_data, _, param_hash = prepare_upload(
                context, want_data=False, want_param=True,
            )
        except ValueError as e:
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

    def on_complete(self, context):
        self.report({"INFO"}, "Build completed successfully.")

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if com.is_aborting():
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if time.time() - self._start_time > self.timeout:
            self.cleanup_modal(context)
            self.on_timeout(context)
            redraw_all_areas(context)
            return {"CANCELLED"}

        redraw_all_areas(context)

        if self.is_complete():
            self.cleanup_modal(context)
            self.on_complete(context)
            return {"FINISHED"}

        return {"PASS_THROUGH"}


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
        # Check if any assigned object has a MESH_CACHE modifier (mesh or
        # curve). Includes STATIC groups — UI-op static objects carry a
        # cache too when animated via move/spin/scale.
        for group in iterate_active_object_groups(context.scene):
            for assigned in group.assigned_objects:
                from ..core.uuid_registry import resolve_assigned
                obj = resolve_assigned(assigned)
                if obj and has_mesh_cache(obj):
                    return True
        return False

    def execute(self, context):
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        context.scene.frame_set(1)
        clear_animation_data(context)
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
        # status.in_progress() covers the local-side window right after
        # Run/Resume/Fetch is clicked — is_running(response) only flips
        # once the remote echoes BUSY, which can lag by a tick or two.
        return (
            not com.busy()
            and data_stat != "NO_DATA"
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not com.info.status.in_progress()
            and not is_running(response)
        )

    def execute(self, context):
        com.query({"request": "delete"}, "Deleting Remote Data...")
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
    SOLVER_OT_ClearAnimation,
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
