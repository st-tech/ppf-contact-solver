# File: main_panel.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Main panel (MAIN_PT_RemotePanel), GlobalStateWatcher, and aggregated
# operator registration.  Extracted from ui/client.py.

import os
from dataclasses import dataclass

import bpy  # pyright: ignore
from bpy.types import Panel  # pyright: ignore

from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import (
    is_server_busy_from_response as is_running,
    is_sim_running_from_response as is_simulating,
)
from ..models.groups import get_addon_data, has_addon_data
from ..core.module import cbor2_available, get_install_error_message, get_install_result, get_installing_status, module_exists
from ..core.reload_server import get_reload_server_status
from ..core.utils import (
    WINDOWS_MAX_PATH,
    find_invalid_name_char,
    find_invalid_path_char,
    get_category_name,
    windows_long_paths_enabled,
    windows_path_too_long,
)

from .connection_ops import (
    REMOTE_OT_Abort,
    REMOTE_OT_CancelConnect,
    REMOTE_OT_CancelStartServer,
    REMOTE_OT_Connect,
    REMOTE_OT_Disconnect,
    REMOTE_OT_OpenProfile,
    REMOTE_OT_StartServer,
    REMOTE_OT_StopServer,
    classes as connection_classes,
)
from .install_ops import (
    REMOTE_OT_InstallCbor2,
    REMOTE_OT_InstallDocker,
    REMOTE_OT_InstallParamiko,
    classes as install_classes,
)
from .mcp_ops import classes as mcp_classes
from .debug_ops import (
    DEBUG_OT_BrowseLogPath,
    DEBUG_OT_ClearLogPath,
    DEBUG_OT_Compile,
    DEBUG_OT_DataReceive,
    DEBUG_OT_DataSend,
    DEBUG_OT_DeleteLog,
    DEBUG_OT_ExecuteServer,
    DEBUG_OT_ExecuteShell,
    DEBUG_OT_GitPull,
    DEBUG_OT_GitPullLocal,
    DEBUG_OT_RenderAnimation,
    DEBUG_OT_StopRender,
    is_render_anim_running,
    get_render_anim_progress,
    WM_OT_OpenGitHubLink,
    classes as debug_classes,
)
from .addon_ops import classes as addon_classes
from .jupyter_ops import classes as jupyter_classes
from .solver_control_ops import (
    SOLVER_OT_ForceTerminatePort,
    SOLVER_OT_SaveAndQuit,
    SOLVER_OT_ShowConsole,
    SOLVER_OT_Terminate,
    SOLVER_OT_UpdateStatus,
    classes as solver_control_classes,
)
from .geometry_cleanup_ops import (
    MESH_OT_RemoveIsolatedVertices,
    classes as geometry_cleanup_classes,
)


# Tiny TTL cache for the port-error probe in the panel draw. The panel
# can redraw many times per second; a TCP probe on every paint would be
# wasteful even at sub-millisecond cost. ``_PROBE_TTL_S`` keeps the
# answer fresh enough that the user sees the stale-error suppression
# kick in within a frame or two of attaching.
#
# Only one port is ever in play at a time (the one named in the current
# error), so a single (port, timestamp, ours) tuple caps the cache at one
# entry instead of accumulating a dict key per distinct port hit.
_PROBE_TTL_S = 1.5
_probe_cache: tuple[int, float, bool] | None = None


# The server reports frame indices 0-based; Blender's timeline is 1-based.
# This offset documents the convention in one place for the display-side
# remaps below.
REMOTE_FRAME_OFFSET = 1


def remote_frame_to_blender(value, *, grouped=False) -> str:
    """Convert a remote 0-based frame index into a Blender 1-based frame for
    display. Strips thousands separators before parsing so the grouped
    ``Total Frames`` string is tolerated, adds ``REMOTE_FRAME_OFFSET``, and
    returns the result thousands-grouped when *grouped* is set. A value that
    isn't an integer is returned unchanged (coerced to ``str``).
    """
    try:
        n = int(str(value).replace(",", "")) + REMOTE_FRAME_OFFSET
    except (ValueError, TypeError):
        return str(value)
    return f"{n:,}" if grouped else str(n)


def _our_server_responding_in_error(error_msg: str) -> bool:
    """True when the panel's port-in-use error names a port that now
    answers a ppf-cts-server TCMD ping. Used to suppress the stale
    error + Force Terminate button after the spawn path's attach branch
    takes over.
    """
    import re
    import time

    global _probe_cache
    m = re.search(r"\bPort\s+(\d+)", error_msg)
    if not m:
        return False
    port = int(m.group(1))
    now = time.monotonic()
    if _probe_cache and _probe_cache[0] == port and (now - _probe_cache[1]) < _PROBE_TTL_S:
        return _probe_cache[2]
    from ..core.connection import _probe_ppf_cts_server
    ours = _probe_ppf_cts_server(port, timeout=0.5)
    _probe_cache = (port, now, ours)
    return ours


def _draw_path_warning(layout, path) -> bool:
    """Draw a one-line warning when *path* holds a space or shell-unsafe
    character, and return ``True`` so the caller can skip any follow-up status
    line.  Draws nothing and returns ``False`` for a valid (or blank) path.
    """
    if find_invalid_path_char(path) is None:
        return False
    layout.label(text="Path should not contain spaces or special characters", icon="ERROR")
    return True


def _draw_win_native_status(layout, win_path) -> None:
    """Draw the Windows Native solver-path validity line(s) for *win_path*.

    Resolves *win_path* to the real solver root (walking up from a selected
    subdirectory such as ``target/release``, ``bin``, or the embedded
    ``python`` folder), then draws a CHECKMARK when a root is found, adding a
    second line naming the resolved root when it differs from what the user
    selected, or an ERROR when no ancestor holds ``ppf-cts-server.exe``.
    No-op for a blank path.
    """
    win_path = (win_path or "").strip().rstrip("/\\")
    if not win_path:
        return
    from ..core.connection import resolve_win_native_root
    resolved = resolve_win_native_root(win_path)
    if resolved is None:
        layout.label(text="ppf-cts-server.exe not found", icon="ERROR")
        return
    layout.label(text="Solver path valid", icon="CHECKMARK")
    if os.path.normpath(resolved) != os.path.normpath(win_path):
        layout.label(text=f"Using solver root: {resolved}")


def _draw_long_path_warning(layout, path, project_name) -> bool:
    """Draw a warning when the build pipeline's deepest cache file under
    *path* would reach the Windows ``MAX_PATH`` limit for *project_name*, and
    return ``True``. Draws nothing and returns ``False`` otherwise.

    This is what makes a too-long Windows solver path fail loudly here, at the
    time it is set, instead of as a bare ``FileNotFoundError`` deep inside a
    later Transfer. Stays silent when Windows long-path support is enabled,
    since the limit no longer applies there.
    """
    if windows_long_paths_enabled():
        return False
    projected = windows_path_too_long(path, project_name)
    if projected is None:
        return False
    layout.label(
        text=f"Path too long: cache files reach {projected} chars (Windows limit {WINDOWS_MAX_PATH})",
        icon="ERROR",
    )
    layout.label(text="Use a shorter solver path, or enable Windows long paths")
    return True


def _draw_name_warning(layout, name) -> bool:
    """Draw a one-line warning when the project *name* holds a space or a
    character that isn't filename-safe, and return ``True``. Draws nothing and
    returns ``False`` for a valid (or blank) name.
    """
    if find_invalid_name_char(name) is None:
        return False
    layout.label(text="Project name should not contain spaces or special characters", icon="ERROR")
    return True


def _draw_install_prompt(layout, *, operator_idname, module_label) -> None:
    """Draw the install operator for *module_label* followed by its current
    status: an in-progress notice while installing, the install error message
    (or a generic failure fallback) when the last attempt failed, or a prompt
    that the module still needs to be installed otherwise.
    """
    layout.operator(operator_idname)
    if get_installing_status():
        layout.label(text="Installing...", icon="FILE_REFRESH")
    else:
        install_result = get_install_result()
        if install_result is False:
            error_msg = get_install_error_message()
            if error_msg:
                layout.label(text=error_msg, icon="ERROR")
            else:
                layout.label(text=f"{module_label} installation failed.", icon="ERROR")
        else:
            layout.label(text=f"{module_label} needs to be installed.", icon="ERROR")


class MAIN_OT_ProjectNameFromFile(bpy.types.Operator):
    """Copy the .blend filename into the Project Name field"""

    bl_idname = "main.project_name_from_file"
    bl_label = "Use Filename"

    def execute(self, context):
        filepath = bpy.data.filepath
        name = os.path.splitext(os.path.basename(filepath))[0]
        get_addon_data(context.scene).state.project_name = name
        return {"FINISHED"}


class MAIN_PT_RemotePanel(Panel):
    """Backend Communicator panel: connection settings, server status, transfer, and statistics."""

    bl_label = "Backend Communicator"
    bl_idname = "MAIN_PT_RemotePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout
        root = get_addon_data(context.scene)
        state = root.state
        props = root.ssh_state

        # cbor2 encodes every Transfer regardless of backend, so prompt
        # for it up front whenever the bundled wheel is missing.
        if not cbor2_available():
            _draw_install_prompt(
                layout,
                operator_idname=REMOTE_OT_InstallCbor2.bl_idname,
                module_label="cbor2",
            )

        if (
            props.server_type == "COMMAND"
            or props.server_type == "CUSTOM"
            or "SSH" in props.server_type
        ):
            if not module_exists(["paramiko"]):
                _draw_install_prompt(
                    layout,
                    operator_idname=REMOTE_OT_InstallParamiko.bl_idname,
                    module_label="Paramiko",
                )
        elif props.server_type == "DOCKER" and not module_exists(["docker"]):
            _draw_install_prompt(
                layout,
                operator_idname=REMOTE_OT_InstallDocker.bl_idname,
                module_label="Docker-Py",
            )

        profile_row = layout.row(align=True)
        profile_row.enabled = com.is_connected() is False
        if props.profile_path and os.path.isfile(bpy.path.abspath(props.profile_path)):
            profile_row.prop(props, "profile_selection", text="Profile")
            profile_row.operator(
                REMOTE_OT_OpenProfile.bl_idname, text="", icon="FILEBROWSER"
            )
            profile_row.operator(
                "ssh.clear_profile", text="", icon="X"
            )
            profile_row.operator(
                "ssh.reload_profile", text="", icon="FILE_REFRESH"
            )
            profile_row.operator(
                "ssh.save_profile", text="", icon="FILE_TICK"
            )
        else:
            profile_row.operator(
                REMOTE_OT_OpenProfile.bl_idname, text="Open Profile", icon="FILEBROWSER"
            )
            profile_row.operator(
                "ssh.save_profile", text="", icon="FILE_TICK"
            )

        # Connection section (expandable)
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(
            state, "show_connection",
            icon="TRIA_DOWN" if state.show_connection else "TRIA_RIGHT",
            emboss=False, text="",
        )
        row.label(text="Connection", icon="LINKED" if com.is_connected() else "UNLINKED")

        if state.show_connection:
            col = box.column()
            col.enabled = com.is_connected() is False
            col.prop(props, "server_type")
            if props.server_type == "COMMAND":
                col.prop(props, "command")
            elif props.server_type == "CUSTOM":
                col.prop(props, "host")
                col.prop(props, "port")
                col.prop(props, "username")
                col.prop(props, "key_path")
                _draw_path_warning(col, props.key_path)
            elif props.server_type == "DOCKER":
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH":
                col.prop(props, "host")
                col.prop(props, "port")
                col.prop(props, "username")
                col.prop(props, "key_path")
                _draw_path_warning(col, props.key_path)
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH_COMMAND":
                col.prop(props, "command")
                col.prop(props, "container")
            if props.server_type == "WIN_NATIVE":
                col.prop(props, "win_native_path")
                if not _draw_path_warning(col, props.win_native_path):
                    _draw_win_native_status(col, props.win_native_path)
                    _draw_long_path_warning(col, props.win_native_path, state.project_name)
            elif props.server_type == "LOCAL":
                col.prop(props, "local_path")
                _draw_path_warning(col, props.local_path)
            elif props.server_type in ["CUSTOM", "COMMAND"]:
                col.prop(props, "ssh_remote_path")
                _draw_path_warning(col, props.ssh_remote_path)
            else:
                col.prop(props, "docker_path")
                _draw_path_warning(col, props.docker_path)

            row = box.row(align=True)
            row.enabled = not com.is_server_running() and not com.is_server_launching()
            row.prop(state, "project_name", text="Project Name")
            if bpy.data.filepath and state.project_name.strip() in ("", "unnamed"):
                row.operator("main.project_name_from_file", text="", icon="COPYDOWN")
            _draw_name_warning(box, state.project_name)
            if "DOCKER" in props.server_type:
                box.prop(props, "docker_port")

            row = box.row()
            if com.is_connecting():
                sub = row.row()
                sub.enabled = False
                sub.operator(REMOTE_OT_Connect.bl_idname, text="Connecting...", icon="LINKED")
            else:
                row.operator(REMOTE_OT_Connect.bl_idname, icon="LINKED")
            row.operator(REMOTE_OT_Disconnect.bl_idname, icon="UNLINKED")

            row = box.row()
            if com.is_server_launching():
                sub = row.row()
                sub.enabled = False
                sub.operator(REMOTE_OT_StartServer.bl_idname, text="Server Starting...", icon="PLAY")
            else:
                row.operator(REMOTE_OT_StartServer.bl_idname, icon="PLAY")
            row.operator(REMOTE_OT_StopServer.bl_idname, icon="CANCEL")

            if not com.is_connected() and not com.is_connecting():
                row = box.row()
                row.label(text="Fill entries and click \"Connect\"", icon="INFO")
            elif com.is_connected() and not com.is_server_running() and not com.is_server_launching():
                row = box.row()
                row.label(text="Click \"Start Server on Remote\"", icon="INFO")

        status = com.info.status
        # After a crash the server stays in the failed state but keeps the
        # saved checkpoints, so surface both: "Simulation Failed (Resumable)"
        # rather than a bare "Resumable" (which hides the failure) or a bare
        # "Simulation Failed" (which hides that a resume is still possible).
        status_text = status.value
        if status == RemoteStatus.SIMULATION_FAILED and len(com.saved_state_frames()) > 0:
            status_text = f"{status.value} (Resumable)"
        message = com.message or f"Status: {status_text}"
        if com.is_connecting():
            layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_CancelConnect.bl_idname, text="Cancel", icon="X")
        elif com.is_server_launching():
            layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_CancelStartServer.bl_idname, text="Cancel", icon="X")
        elif status == RemoteStatus.ABORTING:
            layout.label(text="Status: Aborting...", icon="CANCEL")
        elif status in (RemoteStatus.DATA_SENDING, RemoteStatus.BUILDING):
            if com.info.traffic:
                layout.label(
                    text=f"{message} ({com.info.traffic})",
                    icon=status.icon,
                )
            else:
                layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_Abort.bl_idname, text="Cancel", icon="X")
        elif com.info.traffic:
            layout.label(
                text=f"{message} ({com.info.traffic})",
                icon=com.info.status.icon,
            )
        else:
            layout.label(text=message, icon=com.info.status.icon)
        error = com.error
        if error:
            # Recovery hatch for "Port N is in use" — surfaces only when
            # the error wording matches PortInUseByForeignProcess so the
            # button isn't a foot-gun that's always visible. Clicking it
            # walks the process tree and force-kills the squatter.
            #
            # Defense in depth: when the message implicates a port, probe
            # it live before showing. If our own ppf-cts-server is now
            # responding there (e.g. the user clicked Connect a second
            # time and the spawn path's attach branch took over), the
            # error is stale — suppress the label AND the button so we
            # don't tempt the user into killing our own running server.
            err_lower = error.lower()
            is_port_error = "in use" in err_lower and "port" in err_lower
            stale_port_error = is_port_error and _our_server_responding_in_error(error)
            if not stale_port_error:
                layout.label(text=error, icon="ERROR")
                if is_port_error:
                    layout.operator(
                        SOLVER_OT_ForceTerminatePort.bl_idname, icon="X",
                    )
                elif "isolated vert" in err_lower:
                    # Stray faceless vertices on a STATIC collider abort the
                    # build; offer a one-click cleanup (see the encoder's
                    # detect_isolated_vertices check / geometry_cleanup_ops).
                    layout.operator(
                        MESH_OT_RemoveIsolatedVertices.bl_idname, icon="TRASH",
                    )

        server_error = com.server_error
        if server_error:
            layout.label(text=f"Remote: {server_error}", icon="ERROR")

        row = layout.row()
        row.operator(SOLVER_OT_UpdateStatus.bl_idname, icon="FILE_REFRESH")
        row.operator(SOLVER_OT_ShowConsole.bl_idname, icon="CONSOLE")
        row.prop(state, "debug_mode")

        # Remote Hardware info (shown when connected)
        hardware = com.response.get("hardware", {})
        if hardware and com.is_connected():
            # Prominent, always-visible banner when the connected server
            # is an emulated (CPU stub, no CUDA) build: it produces no
            # real physics. `alert` renders the box in the theme's red.
            if hardware.get("emulated"):
                warn = layout.box()
                warn.alert = True
                wcol = warn.column(align=True)
                wcol.label(text="SERVER IN EMULATION MODE (no CUDA)", icon="ERROR")
                wcol.label(text="Solver produces no real physics (test-rig build).")
            hw_box = layout.box()
            row = hw_box.row()
            row.prop(
                state,
                "show_hardware",
                icon="TRIA_DOWN" if state.show_hardware else "TRIA_RIGHT",
                emboss=False,
                icon_only=True,
            )
            row.label(text="Remote Hardware", icon="DESKTOP")
            if state.show_hardware:
                col = hw_box.column(align=True)
                for key, value in hardware.items():
                    row = col.row(align=True)
                    row.label(text=key)
                    row.label(text=str(value))

        # Scene-encode / drift-check progress. This runs on the main thread
        # inside Transfer / Run before any server status exists, so it has its
        # own snapshot (see core.encode_progress); showing it here gives a
        # labeled bar from the moment of the click that flows straight into the
        # server's build/sim bar below.
        from ..core import encode_progress
        if encode_progress.is_active():
            done, total, label = encode_progress.snapshot()
            factor = min(1.0, done / total) if total else 0.0
            layout.progress(
                factor=factor, type="BAR", text=label or "Preparing scene data...",
            )

        response = com.response
        if com.info.status.in_progress() or is_running(response):
            progress_text = com.message or com.info.status.value
            layout.progress(
                factor=com.info.progress, type="BAR", text=progress_text
            )
            if com.info.status.abortable():
                layout.operator(REMOTE_OT_Abort.bl_idname, icon="CANCEL", text="Abort")
            row = layout.row()
            if is_simulating(response):
                row.operator(SOLVER_OT_SaveAndQuit.bl_idname, icon="FILE_TICK")
                row.operator(SOLVER_OT_Terminate.bl_idname, icon="CANCEL")
        live_summary = com.response.get("summary", {})
        average_summary = com.response.get("average_summary", {})
        displayed_summary = live_summary if is_simulating(response) else average_summary
        stats_label = "Realtime Statistics" if is_simulating(response) else "Average Statistics"
        show_stats_box = is_simulating(response) or bool(average_summary)

        if show_stats_box:
            stats_box = layout.box()
            stats_box.prop(
                state,
                "show_statistics",
                icon="TRIA_DOWN" if state.show_statistics else "TRIA_RIGHT",
                emboss=False,
                text=stats_label,
            )

            def add_statistic_row(col, label, value):
                row = col.row(align=True)
                row.label(text=label)
                row.label(text=value)

            if state.show_statistics:
                col = stats_box.column(align=True)
                for key, value in displayed_summary.items():
                    # Remap remote frame index to Blender frame (0-based → 1-based)
                    if key == "frame":
                        value = remote_frame_to_blender(value)
                    add_statistic_row(col, key, value)

        scene_info = com.response.get("scene_info", {})
        if scene_info:
            # Remap remote frame indices to Blender frames (0-based → 1-based)
            scene_info = dict(scene_info)
            if "Total Frames" in scene_info:
                scene_info["Total Frames"] = remote_frame_to_blender(
                    scene_info["Total Frames"], grouped=True
                )
            if "Last Saved" in scene_info and scene_info["Last Saved"] != "None":
                scene_info["Last Saved"] = remote_frame_to_blender(
                    scene_info["Last Saved"], grouped=True
                )
        if scene_info:
            info_box = layout.box()
            info_box.prop(
                state,
                "show_scene_info",
                icon="TRIA_DOWN" if state.show_scene_info else "TRIA_RIGHT",
                emboss=False,
                text="Scene Info",
            )
            if state.show_scene_info:
                col = info_box.column(align=True)
                for key, value in scene_info.items():
                    row = col.row(align=True)
                    row.label(text=key)
                    row.label(text=str(value))

        if state.debug_mode:
            box = layout.box()
            row = box.row()
            row.label(text="Shell Calls", icon="CONSOLE")

            prop_row = box.row()
            prop_row.enabled = com.is_server_running()
            prop_row.prop(state, "server_script")
            box.operator(DEBUG_OT_ExecuteServer.bl_idname)

            prop_row = box.row()
            prop_row.enabled = com.is_connected()
            prop_row.prop(state, "shell_command")
            row = box.row()
            row.operator(DEBUG_OT_ExecuteShell.bl_idname)
            prop_col = row.column()
            prop_col.enabled = com.is_connected()
            prop_col.prop(state, "use_shell")

            label_row = box.row()
            label_row.enabled = com.is_connected()
            label_row.label(text="Data Transfer Tests", icon="ARROW_LEFTRIGHT")
            row = box.row(align=True)
            row.operator(DEBUG_OT_DataSend.bl_idname, icon="EXPORT")
            row.operator(DEBUG_OT_DataReceive.bl_idname, icon="IMPORT")
            prop_row = box.row()
            prop_row.enabled = com.is_connected()
            prop_row.prop(state, "data_size", text="Data Size (MB)")

            col = box.column()
            col.label(text="Options", icon="PREFERENCES")
            col.prop(state, "max_console_lines")

            col = box.column()
            col.label(text="Console Log Export", icon="TEXT")
            row = col.row(align=True)
            row.prop(state, "log_file_path", text="")
            row.operator(DEBUG_OT_BrowseLogPath.bl_idname, icon="FILEBROWSER", text="")
            row.operator(DEBUG_OT_ClearLogPath.bl_idname, icon="X", text="")
            row = col.row(align=True)
            row.enabled = bool(state.log_file_path)
            row.operator(DEBUG_OT_DeleteLog.bl_idname, icon="TRASH", text="Delete Log")

            label_row = box.row()
            label_row.enabled = com.is_connected()
            label_row.label(text="GitHub Repo on Remote", icon="URL")
            row = box.row(align=True)
            row.operator(DEBUG_OT_GitPull.bl_idname, icon="IMPORT")
            row.operator(DEBUG_OT_Compile.bl_idname, icon="FILE_REFRESH")
            box.operator(WM_OT_OpenGitHubLink.bl_idname, icon="URL")

            col = box.column()
            col.label(text="GitHub Repo on Local", icon="URL")
            col.operator(DEBUG_OT_GitPullLocal.bl_idname, icon="IMPORT")

            col = box.column()
            col.label(text="UUID Migration", icon="FILE_REFRESH")
            col.operator("debug.run_uuid_migration", icon="FILE_REFRESH")
            if state.uuid_migration_result:
                col.label(text=state.uuid_migration_result, icon="INFO")

            col = box.column()
            col.label(text="Render", icon="RENDER_ANIMATION")
            row = col.row(align=True)
            running = is_render_anim_running()
            r1 = row.row(align=True)
            r1.enabled = not running
            r1.operator(DEBUG_OT_RenderAnimation.bl_idname,
                        text="Render Animation", icon="RENDER_ANIMATION")
            r2 = row.row(align=True)
            r2.enabled = running
            r2.operator(DEBUG_OT_StopRender.bl_idname,
                        text="Stop", icon="PAUSE")
            if running:
                current, total, current_frame = get_render_anim_progress()
                pct = (current / total) if total > 0 else 0.0
                col.row().progress(
                    factor=pct,
                    type="BAR",
                    text=f"Frame {current_frame}  ({current}/{total}, {pct*100:.0f}%)",
                )

            col = box.column()
            col.label(text="Add-on Local Debug Server", icon="TOOL_SETTINGS")

            row = col.row(align=True)
            debug_running = get_reload_server_status()
            if debug_running:
                row.operator("addon.stop_reload_server", text="Stop", icon="PAUSE")
            else:
                row.operator("addon.start_reload_server", text="Start", icon="PLAY")
            sub = row.row()
            sub.enabled = not debug_running
            sub.prop(state, "reload_port", text="Port")
            reload_row = col.row(align=True)
            reload_row.operator(
                "addon.trigger_reload", text="Reload Add-on Now", icon="FILE_REFRESH"
            )
            reload_row.operator(
                "addon.trigger_full_reload", text="Full Reload", icon="FILE_REFRESH"
            )



@dataclass
class GlobalStateWatcher:
    last_state: RemoteStatus = RemoteStatus.UNKNOWN
    last_install_status: bool = False
    last_install_result: bool | None = None
    last_progress: float = 0.0
    last_is_connected: bool | None = None
    last_is_connecting: bool | None = None
    last_is_server_running: bool | None = None
    last_is_server_launching: bool | None = None
    last_message: str | None = None
    last_traffic: str | None = None

    def has_changed(self):
        return (
            self.last_state != com.info.status
            or self.last_progress != com.info.progress
            or self.last_install_status != get_installing_status()
            or self.last_install_result != get_install_result()
            or self.last_is_connected != com.is_connected()
            or self.last_is_connecting != com.is_connecting()
            or self.last_is_server_running != com.is_server_running()
            or self.last_is_server_launching != com.is_server_launching()
            or self.last_message != com.message
            or self.last_traffic != com.info.traffic
        )

    def reset(self):
        self.last_progress = com.info.progress
        self.last_state = com.info.status
        self.last_install_status = get_installing_status()
        self.last_install_result = get_install_result()
        self.last_is_connected = com.is_connected()
        self.last_is_connecting = com.is_connecting()
        self.last_is_server_running = com.is_server_running()
        self.last_is_server_launching = com.is_server_launching()
        self.last_message = com.message
        self.last_traffic = com.info.traffic


global_state = GlobalStateWatcher()


def refresh_ssh_panel():
    """refresh the ssh connection panel ui."""
    global global_state
    if global_state.has_changed():
        global_state.reset()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()


classes = (
    connection_classes
    + install_classes
    + mcp_classes
    + solver_control_classes
    + list(geometry_cleanup_classes)
    + [MAIN_OT_ProjectNameFromFile, MAIN_PT_RemotePanel]
    + debug_classes
    + addon_classes
    + list(jupyter_classes)
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
