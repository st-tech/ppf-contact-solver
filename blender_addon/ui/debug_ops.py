# File: debug_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# All DEBUG_OT_* operators and WM_OT_OpenGitHubLink.

import os
import webbrowser

import bpy  # pyright: ignore

from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import is_server_busy_from_response as is_running
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..core.async_op import AsyncOperator
from ..core.utils import redraw_all_areas
from ..models.console import console
from ..models.groups import (
    get_addon_data,
    has_simulatable_dynamics,
)
from ..ui.solver import TransferRequestMixin

test_data = None


class DEBUG_OT_ExecuteServer(AsyncOperator):
    """Send a query through the solver server."""

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
        remote_root = com.normalized_remote_root()
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
        has_dynamic = has_simulatable_dynamics(context.scene)
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
                com.set_error(str(e))
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
                    com.set_error(str(e))
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


# ---------------------------------------------------------------------------
# Debug Render: per-frame manual animation render.
#
# Blender's built-in Render Animation has been observed to capture the
# depsgraph eval between ``render_pre`` and ``frame_change_pre``, which
# leaves the curve PC2 cache half-applied — about half the splines render
# at their rest pose because their bevel mesh wasn't re-evaluated yet.
# This manual loop drives ``scene.frame_set(frame)`` first, which fires
# our curve playback handler synchronously and finishes mutating every
# bezier_point before we ever call ``bpy.ops.render.render(write_still=
# True)``. Output goes to the same paths Blender's animation render
# would write (``scene.render.frame_path(frame=N)`` per frame).
# ---------------------------------------------------------------------------

# Module-level handoff between the operator, the render_complete /
# render_cancel handlers that chain frames, the Stop button, and the
# panel that draws the progress bar. Reset between runs.
_render_anim_state = {
    "running": False,
    "cancel": False,
    "current": 0,         # 1-based count of frames completed
    "total": 0,           # total frames in this run
    "current_frame": 0,   # the scene-frame number being rendered
    "next_frame": 0,      # the next scene-frame to render
    "step": 1,
    "end": 0,
    "saved_filepath": "",
    "saved_frame": 0,
}


def _render_anim_tick():
    """One frame per timer fire.

    Driven by ``bpy.app.timers.register``: returning a float from this
    function re-arms it for the next event-loop iteration.  The actual
    render is synchronous so the current frame blocks the event loop,
    but the timer yields *between* frames -- enough for the Stop button
    click to be processed before the next frame is submitted.
    """
    s = _render_anim_state
    if not s["running"]:
        return None
    if s["cancel"]:
        _cleanup_render_anim(cancelled=True)
        return None
    fr = s["next_frame"]
    if fr > s["end"]:
        _cleanup_render_anim(cancelled=False)
        return None

    scene = bpy.context.scene
    # Drive the curve playback handler synchronously before any
    # render-related work touches the scene.
    scene.frame_set(fr)
    s["current_frame"] = fr
    # Compose the per-frame output path against the *original*
    # render.filepath so frame_path() doesn't accumulate frame
    # numbers ("0001", "00010002", ...). frame_path() returns the
    # numeric stem without the format extension; append the
    # configured extension explicitly via scene.render.file_extension
    # ("." + "png" etc.). Disable use_file_extension while we render
    # so Blender does not double-append.
    scene.render.filepath = s["saved_filepath"]
    target = scene.render.frame_path(frame=fr)
    ext = scene.render.file_extension
    if ext and not target.endswith(ext):
        target = target + ext
    saved_use_ext = scene.render.use_file_extension
    scene.render.filepath = target
    scene.render.use_file_extension = False
    redraw_all_areas(bpy.context)
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        console.write(f"[render anim] frame {fr} failed: {e}")
        scene.render.use_file_extension = saved_use_ext
        _cleanup_render_anim(cancelled=True)
        return None
    scene.render.use_file_extension = saved_use_ext

    s["current"] += 1
    s["next_frame"] += s["step"]
    redraw_all_areas(bpy.context)
    # Returning a small interval (instead of 0.0) yields the event
    # loop a tick of breathing room so the Stop button click can be
    # dispatched and the panel can repaint before the next frame.
    return 0.05


def _cleanup_render_anim(cancelled):
    scene = bpy.context.scene
    scene.render.filepath = _render_anim_state["saved_filepath"]
    try:
        scene.frame_set(_render_anim_state["saved_frame"])
    except Exception:
        pass
    n_done = _render_anim_state["current"]
    n_total = _render_anim_state["total"]
    _render_anim_state["running"] = False
    _render_anim_state["cancel"] = False
    _render_anim_state["current"] = 0
    _render_anim_state["total"] = 0
    _render_anim_state["current_frame"] = 0
    _render_anim_state["next_frame"] = 0
    verb = "cancelled" if cancelled else "finished"
    console.write(f"[render anim] {verb}: {n_done}/{n_total} frames written")
    redraw_all_areas(bpy.context)


def _expected_frame_paths(context):
    """Return the absolute file paths this render would write."""
    scene = context.scene
    start = scene.frame_start
    end = scene.frame_end
    step = max(1, scene.frame_step)
    ext = scene.render.file_extension
    paths = []
    for fr in range(start, end + 1, step):
        target = scene.render.frame_path(frame=fr)
        if ext and not target.endswith(ext):
            target = target + ext
        paths.append(bpy.path.abspath(target))
    return paths


class DEBUG_OT_RenderAnimation(Operator):
    """Render the active frame range one frame at a time.

    Drives ``scene.frame_set(N)`` -> synchronous single-frame render
    via a ``bpy.app.timers`` chain. Output paths match the built-in
    Render Animation (``scene.render.frame_path(frame=N)`` per frame).
    Between frames the event loop runs, so the Stop button click is
    processed before the next frame is submitted.
    """

    bl_idname = "debug.render_animation"
    bl_label = "Render Animation"
    bl_options = {"REGISTER"}

    # Filled in on invoke when previous files are detected.
    _existing_count: int = 0

    @classmethod
    def poll(cls, _):
        return not _render_anim_state["running"]

    def invoke(self, context, _event):
        # Skip the dialog when no prior frames would be overwritten.
        existing = [p for p in _expected_frame_paths(context) if os.path.exists(p)]
        if not existing:
            return self.execute(context)
        self._existing_count = len(existing)
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(
            text=f"Output directory already contains {self._existing_count} "
                 f"matching frame file(s).",
            icon="ERROR",
        )
        layout.label(text="They will be deleted before rendering starts.")
        layout.label(text="Click OK to proceed, or Cancel to abort.")

    def execute(self, context):
        scene = context.scene
        # Clean any previous frame files that would be overwritten.
        deleted = 0
        for p in _expected_frame_paths(context):
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted += 1
                except OSError as e:
                    console.write(f"[render anim] could not delete {p}: {e}")

        start = scene.frame_start
        end = scene.frame_end
        step = max(1, scene.frame_step)
        n_total = max(1, (end - start) // step + 1)

        _render_anim_state["running"] = True
        _render_anim_state["cancel"] = False
        _render_anim_state["current"] = 0
        _render_anim_state["total"] = n_total
        _render_anim_state["current_frame"] = start
        _render_anim_state["next_frame"] = start
        _render_anim_state["step"] = step
        _render_anim_state["end"] = end
        _render_anim_state["saved_filepath"] = scene.render.filepath
        _render_anim_state["saved_frame"] = scene.frame_current

        suffix = f" ({deleted} previous file(s) removed)" if deleted else ""
        self.report({"INFO"},
                    f"Render Animation: frames {start}..{end} step {step} "
                    f"-> {n_total} frames{suffix}")
        bpy.app.timers.register(_render_anim_tick, first_interval=0.05)
        return {"FINISHED"}


class DEBUG_OT_StopRender(Operator):
    """Cancel the in-progress manual Render Animation.

    Flips the cancel flag; the timer-driven chain halts before the
    next frame is submitted. The in-progress frame (if any) keeps
    rendering to completion because Blender does not expose a clean
    mid-frame cancel from Python.
    """

    bl_idname = "debug.stop_render_animation"
    bl_label = "Stop"

    @classmethod
    def poll(cls, _):
        return _render_anim_state["running"]

    def execute(self, _):
        _render_anim_state["cancel"] = True
        self.report({"INFO"},
                    "Stop requested; in-flight frame will finish, then halt")
        return {"FINISHED"}


def is_render_anim_running() -> bool:
    """Helper for UI code to know whether to show the progress bar."""
    return _render_anim_state["running"]


def get_render_anim_progress() -> tuple[int, int, int]:
    """Return ``(current, total, current_frame)`` for the panel.

    ``current`` and ``total`` are frame *counts*; ``current_frame`` is the
    scene-frame number being rendered.  All zero when no render is in
    progress.
    """
    s = _render_anim_state
    return s["current"], s["total"], s["current_frame"]


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
    DEBUG_OT_RenderAnimation,
    DEBUG_OT_StopRender,
]
