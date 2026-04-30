# File: connection_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# SSH/Docker/Local connection operators.

import shlex
import time

import bpy  # pyright: ignore
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..core.async_op import AsyncOperator
from ..core.client import communicator as com
from ..core.module import module_exists
from ..core.utils import get_timer_wait_time, redraw_all_areas
from ..models.groups import get_addon_data


def refresh_ssh_panel():
    """Refresh the ssh connection panel UI.

    This wrapper uses a lazy import to break a circular dependency:
    main_panel.py imports operator classes from connection_ops.py, so
    connection_ops.py cannot import from main_panel.py at module level.
    """
    from .main_panel import refresh_ssh_panel as _refresh
    _refresh()


class REMOTE_OT_Connect(Operator):
    """Establish an SSH connection and execute a command asynchronously.

    Note: This operator intentionally does NOT use AsyncOperator because it
    stays alive for the entire connection lifetime (refreshing the SSH
    panel each tick) and only finishes when disconnected.
    """

    bl_idname = "ssh.run_command"
    bl_label = "Connect"

    _timer = None
    _connection_established = False
    _start_time: float = 0.0
    timeout: float = 60.0

    def get_remote_path(self, props):
        if props.server_type == "LOCAL":
            return props.local_path
        elif props.server_type in ["CUSTOM", "COMMAND"]:
            return props.ssh_remote_path
        else:
            return props.docker_path

    @classmethod
    def poll(cls, context):
        root = get_addon_data(context.scene)
        props = root.ssh_state
        state = root.state
        project_name_valid = state.project_name.strip() != ""

        if props.server_type == "COMMAND":
            return (
                not com.is_connected()
                and props.command.strip() != ""
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "CUSTOM":
            return (
                not com.is_connected()
                and props.host.strip() != ""
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER":
            return (
                not com.is_connected()
                and props.container.strip() != ""
                and props.key_path.strip() != ""
                and module_exists(["docker"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER_SSH":
            return (
                not com.is_connected()
                and props.host.strip() != ""
                and props.container.strip() != ""
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER_SSH_COMMAND":
            return (
                not com.is_connected()
                and props.command.strip() != ""
                and props.container.strip() != ""
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "WIN_NATIVE":
            return (
                not com.is_connected()
                and props.win_native_path.strip() != ""
                and project_name_valid
            )
        elif props.server_type == "LOCAL":
            return not com.is_connected() and project_name_valid

    def execute(self, context):
        root = get_addon_data(context.scene)
        props = root.ssh_state
        com.set_project_name(root.state.project_name)
        if props.server_type == "COMMAND" or props.server_type == "DOCKER_SSH_COMMAND":
            command_parts = shlex.split(props.command)
            host, port, username, key_path = None, 22, None, None
            for i, part in enumerate(command_parts):
                if part == "-p" and i + 1 < len(command_parts):
                    port = int(command_parts[i + 1])
                elif part.startswith("-i") and i + 1 < len(command_parts):
                    key_path = command_parts[i + 1]
                elif "@" in part:
                    username, host = part.split("@")
                elif not host and not part.startswith("-") and part != "ssh":
                    host = part
            if not host:
                self.report(
                    {"ERROR"},
                    "Failed to parse command. Ensure it includes host.",
                )
                return {"CANCELLED"}
            container = props.container if "DOCKER" in props.server_type else None
            com.connect_ssh(
                host=host,
                port=port,
                username=username,
                key_path=key_path,
                path=self.get_remote_path(props),
                container=container,
                server_port=props.docker_port,
            )
        elif props.server_type == "CUSTOM" or props.server_type == "DOCKER_SSH":
            container = props.container if "DOCKER" in props.server_type else None
            com.connect_ssh(
                host=props.host,
                port=props.port,
                username=props.username,
                key_path=props.key_path,
                path=self.get_remote_path(props),
                container=container,
                server_port=props.docker_port,
            )
        elif props.server_type == "DOCKER":
            com.connect_docker(
                props.container,
                self.get_remote_path(props),
                server_port=props.docker_port,
            )
        elif props.server_type == "WIN_NATIVE":
            win_path = props.win_native_path.strip()
            if not win_path:
                self.report({"ERROR"}, "Solver path is not set")
                return {"CANCELLED"}
            from ..models.defaults import DEFAULT_SERVER_PORT
            com.connect_win_native(win_path, props.docker_port or DEFAULT_SERVER_PORT)
        elif props.server_type == "LOCAL":
            com.connect_local(
                self.get_remote_path(props),
                server_port=props.docker_port,
            )

        self._connection_established = False
        self._start_time = time.time()
        self._timer = context.window_manager.event_timer_add(
            get_timer_wait_time(), window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # Check is_connected() before the cancel/timeout branches: a fast connect
        # (e.g., LOCAL) can reach ONLINE before the first timer tick, which would
        # otherwise be misread as a cancellation since is_connecting() is False.
        if com.is_connected() and not self._connection_established:
            self._connection_established = True
            redraw_all_areas(context)
        # Detect cancellation (user canceled or connection failed)
        if not self._connection_established and not com.is_connecting():
            if self._timer:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            return {"CANCELLED"}
        # Timeout only applies before connection is established
        if (
            not self._connection_established
            and time.time() - self._start_time > self.timeout
        ):
            if self._timer:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            self.report({"ERROR"}, "Connection timed out")
            return {"CANCELLED"}
        refresh_ssh_panel()
        if self._connection_established and not com.is_connected():
            if self._timer:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            return {"FINISHED"}
        return {"PASS_THROUGH"}


class REMOTE_OT_CancelConnect(Operator):
    """Cancel a pending connection attempt."""

    bl_idname = "ssh.cancel_connect"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, _):
        return com.is_connecting()

    def execute(self, context):
        com.disconnect()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_Disconnect(Operator):
    """Disconnect the SSH connection."""

    bl_idname = "ssh.disconnect"
    bl_label = "Disconnect"

    @classmethod
    def poll(cls, _):
        return com.is_connected() and not com.info.status.abortable()

    def execute(self, context):
        com.disconnect()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_Abort(Operator):
    """Abort the current in-progress operation."""

    bl_idname = "ssh.abort"
    bl_label = "Abort"

    def execute(self, context):
        com.abort()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_StartServer(AsyncOperator):
    """Start the remote server process."""

    bl_idname = "ssh.start_server"
    bl_label = "Start Server on Remote"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return (
            com.is_connected() and com.is_server_running() is False and not com.busy()
        )

    def execute(self, context):
        com.start_server()
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return com.is_server_running()

    def is_cancelled(self) -> bool:
        return not com.is_server_launching() and not com.is_server_running()

    def on_complete(self, context):
        redraw_all_areas(context)


class REMOTE_OT_CancelStartServer(Operator):
    """Cancel a pending server start."""

    bl_idname = "ssh.cancel_start_server"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, _):
        return com.is_server_launching()

    def execute(self, context):
        com.stop_server()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_StopServer(AsyncOperator):
    """Stop the remote server process."""

    bl_idname = "ssh.stop_server"
    bl_label = "Stop Server on Remote"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return (
            com.is_connected()
            and com.is_server_running()
            and not com.info.status.abortable()
        )

    def execute(self, context):
        com.stop_server()
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not com.is_server_running()

    def on_complete(self, context):
        redraw_all_areas(context)


class REMOTE_OT_OpenProfile(Operator):
    """Open a TOML connection profile file."""

    bl_idname = "ssh.open_profile"
    bl_label = "Open Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "connection_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ..core.profile import get_profile_names

        root = get_addon_data(context.scene)
        props = root.ssh_state
        props.profile_path = self.filepath
        abs_path = bpy.path.abspath(self.filepath)
        names = get_profile_names(abs_path)
        if names:
            props.profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_ClearProfile(Operator):
    """Clear the loaded connection profile."""

    bl_idname = "ssh.clear_profile"
    bl_label = "Clear Profile"

    def execute(self, context):
        root = get_addon_data(context.scene)
        root.ssh_state.profile_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_ReloadProfile(Operator):
    """Re-apply the currently selected connection profile."""

    bl_idname = "ssh.reload_profile"
    bl_label = "Reload Profile"

    def execute(self, context):
        from ..core.profile import apply_profile, load_profiles

        root = get_addon_data(context.scene)
        props = root.ssh_state
        if not props.profile_path or props.profile_selection == "NONE":
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(props.profile_path)
        profiles = load_profiles(abs_path)
        profile = profiles.get(props.profile_selection)
        if profile is None:
            return {"CANCELLED"}
        apply_profile(profile, props)
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_SaveProfile(Operator):
    """Save current connection settings to a profile."""

    bl_idname = "ssh.save_profile"
    bl_label = "Save Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore

    def invoke(self, context, event):
        root = get_addon_data(context.scene)
        props = root.ssh_state
        if props.profile_path and props.profile_selection != "NONE":
            return self.execute(context)
        if not self.filepath:
            self.filepath = "connection_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ..core.profile import read_connection_profile, save_profile_entry

        root = get_addon_data(context.scene)
        props = root.ssh_state
        data = read_connection_profile(props)
        if props.profile_path and props.profile_selection != "NONE":
            abs_path = bpy.path.abspath(props.profile_path)
            save_profile_entry(abs_path, props.profile_selection, data)
            self.report({"INFO"}, f"Saved to '{props.profile_selection}'")
        else:
            save_profile_entry(self.filepath, self.entry_name, data)
            props.profile_path = self.filepath
            props.profile_selection = self.entry_name
            self.report({"INFO"}, f"Saved to '{self.entry_name}'")
        redraw_all_areas(context)
        return {"FINISHED"}


classes = [
    REMOTE_OT_Connect,
    REMOTE_OT_CancelConnect,
    REMOTE_OT_Disconnect,
    REMOTE_OT_Abort,
    REMOTE_OT_StartServer,
    REMOTE_OT_CancelStartServer,
    REMOTE_OT_StopServer,
    REMOTE_OT_OpenProfile,
    REMOTE_OT_ClearProfile,
    REMOTE_OT_ReloadProfile,
    REMOTE_OT_SaveProfile,
]
