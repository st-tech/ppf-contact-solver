# File: debug_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# All DEBUG_OT_* operators and WM_OT_OpenGitHubLink.

import os
import webbrowser

from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import is_server_busy_from_response as is_running
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..core.async_op import AsyncOperator
from ..core.utils import redraw_all_areas
from ..models.console import console
from ..models.groups import get_addon_data, iterate_active_object_groups
from ..ui.solver import TransferRequestMixin

test_data = None


class DEBUG_OT_ExecuteServer(AsyncOperator):
    """Execute the server.py script."""

    bl_idname = "debug.execute_server"
    bl_label = "Exec Command via Server"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return com.is_connected() and com.is_server_running() and not com.busy()

    def execute(self, context):
        state = get_addon_data(context.scene).state
        raw_test_script = state.server_script.strip()
        key_values = {}
        key = None
        for word in raw_test_script.split():
            if word.startswith("--"):
                key = word[2:]
            elif key is not None:
                key_values[key] = word
        com.query(key_values, "Communicating with Server...")
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return com.response is not None

    def on_complete(self, context):
        result = com.response
        console.write("------")
        for key, value in result.items():
            console.write(f"{key}: {value}\n")
        console.write("------")
        self.report({"INFO"}, "Python script result ready.")


class DEBUG_OT_ExecuteShell(Operator):
    """Execute a shell command."""

    bl_idname = "debug.execute_shell"
    bl_label = "Execute Shell Command on Remote"

    @classmethod
    def poll(cls, context):
        state = get_addon_data(context.scene).state
        return (
            com.is_connected() and not com.busy() and state.shell_command.strip() != ""
        )

    def execute(self, context):
        state = get_addon_data(context.scene).state
        command = state.shell_command.strip()
        if command:
            com.exec(command, shell=state.use_shell)
        else:
            self.report({"ERROR"}, "Shell command is empty.")
        return {"FINISHED"}


class DEBUG_OT_DataSend(AsyncOperator):
    """Send data to the remote server."""

    bl_idname = "debug.data_send"
    bl_label = "Data Send"

    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return com.is_server_running() and not com.busy()

    def execute(self, context):
        global test_data
        state = get_addon_data(context.scene).state
        test_data = os.urandom(state.data_size * 1024 * 1024)
        # Remote is always Linux; build a POSIX path regardless of local OS.
        remote_root = com.connection.remote_root.rstrip("/")
        if not remote_root:
            self.report({"ERROR"}, "Not connected; cannot send test data")
            return {"CANCELLED"}
        com.data_send(
            f"{remote_root}/dummy_data.pickle",
            test_data,
            "Uploading Test Data...",
        )
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        if not com.is_server_running():
            return True
        return com.info.progress >= 0.99

    def on_complete(self, context):
        if not com.is_server_running():
            self.report({"ERROR"}, "Connection lost during data transfer")
        redraw_all_areas(context)


class DEBUG_OT_DataReceive(AsyncOperator):
    """Receive data from the remote server."""

    bl_idname = "debug.data_receive"
    bl_label = "Data Receive"

    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        global test_data
        return not com.busy() and test_data is not None and com.is_server_running()

    def execute(self, context):
        com.data_receive(
            os.path.join(
                com.connection.remote_root,
                "dummy_data.pickle",
            ),
            "Downloading Test Data...",
        )
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return com.info.progress >= 0.99 and com.data is not None

    def on_complete(self, context):
        global test_data
        received_data = com.data
        if received_data == test_data:
            self.report({"INFO"}, "Data received matches test data.")
        else:
            self.report({"ERROR"}, "Data received does not match test data.")


class DEBUG_OT_TransferWithoutBuild(TransferRequestMixin, AsyncOperator):
    """Transfer data and parameters without triggering a remote build."""

    bl_idname = "debug.transfer_without_build"
    bl_label = "Transfer without Build"

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
        return (
            has_dynamic
            and not com.busy()
            and com.connection.remote_root
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )

    def execute(self, context):
        status = com.info.status
        if status in (RemoteStatus.READY, RemoteStatus.RESUMABLE):
            self.report(
                {"INFO"},
                "Remote data already exists. Deleting first.",
            )
            self.request_delete()
        else:
            try:
                from ..core.encoder import prepare_upload
                data, param, data_hash, param_hash = prepare_upload(context)
            except ValueError as e:
                self.report({"ERROR"}, str(e))
                com.error = str(e)
                redraw_all_areas(context)
                return {"CANCELLED"}
            com.upload_only(
                data=data, param=param,
                data_hash=data_hash,
                param_hash=param_hash,
                message="Uploading scene (no build)...",
            )
            self._mode = "uploading"
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return self._mode == "uploading" and not com.busy() and not com.error

    def on_complete(self, context):
        redraw_all_areas(context)
        self.report({"INFO"}, "Transfer completed successfully.")

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self.timeout and self._start_time is not None:
            import time

            if time.time() - self._start_time > self.timeout:
                self.cleanup_modal(context)
                self.on_timeout(context)
                redraw_all_areas(context)
                return {"CANCELLED"}

        redraw_all_areas(context)
        if self._mode == "delete":
            # Wait for the delete query to finish so com.upload_only
            # isn't silently dropped by the can_operate guard.
            if com.busy():
                return {"PASS_THROUGH"}
            data_stat = com.info.response.get("data", None)
            if data_stat == "NO_DATA":
                try:
                    from ..core.encoder import prepare_upload
                    data, param, data_hash, param_hash = prepare_upload(context)
                except ValueError as e:
                    self.report({"ERROR"}, str(e))
                    com.error = str(e)
                    self.cleanup_modal(context)
                    redraw_all_areas(context)
                    return {"CANCELLED"}
                com.upload_only(
                    data=data, param=param,
                    data_hash=data_hash,
                    param_hash=param_hash,
                    message="Uploading scene (no build)...",
                )
                self._mode = "uploading"
                self.report({"INFO"}, "Remote data deleted successfully.")

        if self.is_complete():
            self.cleanup_modal(context)
            self.on_complete(context)
            return {"FINISHED"}
        return {"PASS_THROUGH"}


class DEBUG_OT_Build(AsyncOperator):
    """Trigger a remote build without retransferring data."""

    bl_idname = "debug.build"
    bl_label = "Build"

    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        status = com.info.status
        response = com.info.response
        return (
            status in (
                RemoteStatus.WAITING_FOR_BUILD,
                RemoteStatus.READY,
                RemoteStatus.RESUMABLE,
            )
            and not com.busy()
            and com.connection.remote_root
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )

    def execute(self, context):
        com.build()
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return com.info.status != RemoteStatus.BUILDING

    def on_complete(self, context):
        redraw_all_areas(context)
        self.report({"INFO"}, "Build completed successfully.")


class DEBUG_OT_GitPull(Operator):
    """Pull the latest changes from the Git repository."""

    bl_idname = "debug.git_pull"
    bl_label = "Git Pull"

    @classmethod
    def poll(cls, _):
        return com.is_connected() and not com.busy()

    def execute(self, _):
        com.exec("git pull")
        self.report({"INFO"}, "Git pull executed.")
        return {"FINISHED"}


class DEBUG_OT_Compile(Operator):
    """Compile the project."""

    bl_idname = "debug.compile"
    bl_label = "Compile"

    @classmethod
    def poll(cls, _):
        return com.is_connected() and not com.busy()

    def execute(self, _):
        com.exec("/root/.cargo/bin/cargo build --release")
        self.report({"INFO"}, "Project compiled.")
        return {"FINISHED"}


class DEBUG_OT_BrowseLogPath(Operator):
    """Pick a destination for the console log file (enables file logging)."""

    bl_idname = "debug.browse_log_path"
    bl_label = "Browse Log Path"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.txt;*.log", options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        state = get_addon_data(context.scene).state
        # Pre-fill with the current selection if any, else suggest log.txt
        # in the user's home dir so the dialog opens somewhere sensible
        # without writing into a per-bundle path the addon shouldn't touch.
        if state.log_file_path:
            self.filepath = state.log_file_path
        else:
            self.filepath = os.path.join(os.path.expanduser("~"), "log.txt")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        state.log_file_path = self.filepath
        redraw_all_areas(context)
        return {"FINISHED"}


class DEBUG_OT_ClearLogPath(Operator):
    """Clear the log path (disables file logging)."""

    bl_idname = "debug.clear_log_path"
    bl_label = "Clear Log Path"

    @classmethod
    def poll(cls, context):
        return bool(get_addon_data(context.scene).state.log_file_path)

    def execute(self, context):
        get_addon_data(context.scene).state.log_file_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class DEBUG_OT_DeleteLog(Operator):
    """Delete the log file specified in the export log path."""

    bl_idname = "debug.delete_log"
    bl_label = "Delete Log"

    @classmethod
    def poll(cls, context):
        state = get_addon_data(context.scene).state
        log_path = state.log_file_path
        return bool(log_path) and os.path.exists(log_path)

    def execute(self, context):
        state = get_addon_data(context.scene).state
        log_path = state.log_file_path
        if log_path and os.path.exists(log_path):
            try:
                os.remove(log_path)
                self.report({"INFO"}, f"Deleted log file: {log_path}")
            except Exception as e:
                self.report({"ERROR"}, f"Failed to delete log: {e}")
        else:
            self.report({"WARNING"}, "Log file does not exist.")
        return {"FINISHED"}


class DEBUG_OT_GitPullLocal(AsyncOperator):
    """Pull the latest changes from the local Git repository."""

    bl_idname = "debug.git_pull_local"
    bl_label = "Git Pull (Local)"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return (
            not hasattr(cls, "_process")
            or cls._process is None
            or cls._process.poll() is not None
        )

    def execute(self, context):
        import subprocess

        try:
            type(self)._process = subprocess.Popen(
                ["git", "pull"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
            )
            self.setup_modal(context)
            self.report({"INFO"}, "Local git pull started...")
            return {"RUNNING_MODAL"}
        except Exception as e:
            self.report({"ERROR"}, f"Exception: {e}")
            type(self)._process = None
            return {"FINISHED"}

    def is_complete(self) -> bool:
        proc = type(self)._process
        if proc is None:
            return True
        return proc.poll() is not None

    def on_complete(self, context):
        proc = type(self)._process
        if proc is None:
            return
        retcode = proc.poll()
        _, stderr = proc.communicate()
        if retcode == 0:
            self.report({"INFO"}, "Local git pull succeeded.")
        else:
            self.report({"ERROR"}, f"Local git pull failed: {stderr.strip()}")
        type(self)._process = None
        redraw_all_areas(context)

    def on_timeout(self, context):
        proc = type(self)._process
        if proc is not None:
            proc.kill()
            type(self)._process = None
        self.report({"ERROR"}, "Local git pull timed out")


class WM_OT_OpenGitHubLink(Operator):
    """Open the GitHub repository link."""

    bl_idname = "wm.open_github_link"
    bl_label = "Open GitHub Link"

    def execute(self, _):
        webbrowser.open("https://github.com/st-tech/ppf-contact-solver/")
        return {"FINISHED"}


class DEBUG_OT_RunUUIDMigration(Operator):
    """Migrate legacy name-based data to UUID-based identification"""

    bl_idname = "debug.run_uuid_migration"
    bl_label = "Run UUID Migration"

    def execute(self, context):
        from ..core.migrate import migrate_legacy_data
        result = migrate_legacy_data()
        from ..models.groups import get_addon_data
        get_addon_data(context.scene).state.uuid_migration_result = result
        self.report({"INFO"}, result)
        return {"FINISHED"}


classes = [
    WM_OT_OpenGitHubLink,
    DEBUG_OT_RunUUIDMigration,
    DEBUG_OT_ExecuteServer,
    DEBUG_OT_ExecuteShell,
    DEBUG_OT_DataSend,
    DEBUG_OT_DataReceive,
    DEBUG_OT_TransferWithoutBuild,
    DEBUG_OT_Build,
    DEBUG_OT_GitPull,
    DEBUG_OT_Compile,
    DEBUG_OT_BrowseLogPath,
    DEBUG_OT_ClearLogPath,
    DEBUG_OT_DeleteLog,
    DEBUG_OT_GitPullLocal,
]
