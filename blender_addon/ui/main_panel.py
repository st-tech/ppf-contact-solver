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
from ..core.module import get_install_error_message, get_install_result, get_installing_status, module_exists
from ..core.reload_server import get_reload_server_status
from ..core.utils import get_category_name

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
    WM_OT_OpenGitHubLink,
    classes as debug_classes,
)
from .addon_ops import classes as addon_classes
from .jupyter_ops import classes as jupyter_classes
from .solver_control_ops import (
    SOLVER_OT_SaveAndQuit,
    SOLVER_OT_ShowConsole,
    SOLVER_OT_Terminate,
    SOLVER_OT_UpdateStatus,
    classes as solver_control_classes,
)


class MAIN_OT_ProjectNameFromFile(bpy.types.Operator):
    """Copy the .blend filename into the Project Name field"""

    bl_idname = "main.project_name_from_file"
    bl_label = "Use Filename"

    def execute(self, context):
        import os
        filepath = bpy.data.filepath
        name = os.path.splitext(os.path.basename(filepath))[0]
        get_addon_data(context.scene).state.project_name = name
        return {"FINISHED"}


class MAIN_PT_RemotePanel(Panel):
    """Panel for SSH connection settings."""

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

        if (
            props.server_type == "COMMAND"
            or props.server_type == "CUSTOM"
            or "SSH" in props.server_type
        ):
            if not module_exists(["paramiko"]):
                layout.operator(REMOTE_OT_InstallParamiko.bl_idname)
                if get_installing_status():
                    layout.label(text="Installing...", icon="FILE_REFRESH")
                else:
                    install_result = get_install_result()
                    if install_result is False:
                        error_msg = get_install_error_message()
                        if error_msg:
                            layout.label(text=error_msg, icon="ERROR")
                        else:
                            layout.label(
                                text="Paramiko installation failed.", icon="ERROR"
                            )
                    else:
                        layout.label(
                            text="Paramiko needs to be installed.",
                            icon="ERROR",
                        )
        elif props.server_type == "DOCKER" and not module_exists(["docker"]):
            layout.operator(REMOTE_OT_InstallDocker.bl_idname)
            if get_installing_status():
                layout.label(text="Installing...", icon="FILE_REFRESH")
            else:
                install_result = get_install_result()
                if install_result is False:
                    error_msg = get_install_error_message()
                    if error_msg:
                        layout.label(text=error_msg, icon="ERROR")
                    else:
                        layout.label(
                            text="Docker-Py installation failed.", icon="ERROR"
                        )
                else:
                    layout.label(
                        text="Docker-Py needs to be installed.",
                        icon="ERROR",
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
            elif props.server_type == "DOCKER":
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH":
                col.prop(props, "host")
                col.prop(props, "port")
                col.prop(props, "username")
                col.prop(props, "key_path")
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH_COMMAND":
                col.prop(props, "command")
                col.prop(props, "container")
            if props.server_type == "WIN_NATIVE":
                col.prop(props, "win_native_path")
                win_path = props.win_native_path.strip().rstrip("/\\")
                if win_path:
                    server_py = os.path.join(win_path, "server.py")
                    if os.path.exists(server_py):
                        col.label(text="Solver path valid", icon="CHECKMARK")
                    else:
                        col.label(text="server.py not found", icon="ERROR")
            elif props.server_type == "LOCAL":
                col.prop(props, "local_path")
            elif props.server_type in ["CUSTOM", "COMMAND"]:
                col.prop(props, "ssh_remote_path")
            else:
                col.prop(props, "docker_path")

            row = box.row(align=True)
            row.enabled = not com.is_server_running() and not com.is_server_launching()
            row.prop(state, "project_name", text="Project Name")
            if bpy.data.filepath and state.project_name.strip() in ("", "unnamed"):
                row.operator("main.project_name_from_file", text="", icon="COPYDOWN")
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

        message = com.message or f"Status: {com.info.status.value}"
        status = com.info.status
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
            layout.label(text=error, icon="ERROR")

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
                        try:
                            value = str(int(value) + 1)
                        except (ValueError, TypeError):
                            pass
                    add_statistic_row(col, key, value)

        scene_info = com.response.get("scene_info", {})
        if scene_info:
            # Remap remote frame indices to Blender frames (0-based → 1-based)
            scene_info = dict(scene_info)
            if "Total Frames" in scene_info:
                try:
                    n = int(scene_info["Total Frames"].replace(",", ""))
                    scene_info["Total Frames"] = f"{n + 1:,}"
                except (ValueError, TypeError):
                    pass
            if "Last Saved" in scene_info and scene_info["Last Saved"] != "None":
                try:
                    n = int(scene_info["Last Saved"])
                    scene_info["Last Saved"] = str(n + 1)
                except (ValueError, TypeError):
                    pass
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
