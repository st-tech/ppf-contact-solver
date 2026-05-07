# File: solver_control_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Solver console/terminate/status operators.

import os
import subprocess

from bpy.types import Operator  # pyright: ignore

from ..core.client import RemoteStatus, communicator as com
from ..models.console import console
from ..models.defaults import DEFAULT_SERVER_PORT


class SOLVER_OT_ShowConsole(Operator):
    """Show a console window with read-only text."""

    bl_idname = "solver.show_console"
    bl_label = "Show Console"

    def execute(self, _):
        console.show()
        return {"FINISHED"}


class SOLVER_OT_Terminate(Operator):
    """Terminate the solver."""

    bl_idname = "solver.terminate"
    bl_label = "Terminate"

    def execute(self, _):
        com.terminate()
        return {"FINISHED"}


class SOLVER_OT_SaveAndQuit(Operator):
    """Save the current state and quit the solver."""

    bl_idname = "solver.save_quit"
    bl_label = "Save and Quit"

    @classmethod
    def poll(cls, _):
        # Gate on connected + not busy so clicking this mid-Transfer doesn't
        # inject a quit request while a modal is still waiting on the same
        # state transition.
        return (
            com.is_connected()
            and com.is_server_running()
            and not com.busy()
        )

    def execute(self, _):
        com.save_and_quit()
        self.report({"INFO"}, "Save and Quit executed.")
        return {"FINISHED"}


class SOLVER_OT_UpdateStatus(Operator):
    """Update server status."""

    bl_idname = "solver.update_status"
    bl_label = "Update Stat"

    @classmethod
    def poll(cls, _):
        return com.is_connected() and com.is_server_running() and (
            not com.busy() or com.info.status == RemoteStatus.BUILDING
        )

    def execute(self, context):
        if com.info.status == RemoteStatus.BUILDING:
            # During build, the polling loop already refreshes data every 0.25s.
            # Just force a UI redraw instead of queuing a blocked query.
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
        else:
            com.query(message="Updating Status...")
        return {"FINISHED"}


def _find_pid_holding_port(port: int) -> int | None:
    """Return the PID of the LISTENING process on loopback `port`, or None.

    Cross-platform helpers: netstat -ano on Windows, lsof -ti tcp:N on
    POSIX. Both ship with the OS by default. Errors (parse failure,
    binary missing, no match) all coalesce to None.
    """
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return None
            suffix = f":{port}"
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3].upper() == "LISTENING":
                    if parts[1].endswith(suffix):
                        try:
                            return int(parts[4])
                        except ValueError:
                            continue
            return None
        else:
            r = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode not in (0, 1):
                return None
            for tok in r.stdout.split():
                try:
                    return int(tok)
                except ValueError:
                    continue
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _kill_pid(pid: int) -> None:
    """Force-kill `pid` and any child processes. Raises on failure."""
    if os.name == "nt":
        # /T walks the process tree, /F forces termination — needed
        # because ppf-cts-server.exe spawns child solver subprocesses
        # the user wants gone too.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=True, timeout=10, capture_output=True,
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)


def _resolved_server_port() -> int:
    """Best-effort lookup of the solver server port the addon is using.

    Falls back to DEFAULT_SERVER_PORT if nothing else is available.
    """
    info_port = getattr(com.info, "server_port", None)
    if isinstance(info_port, int) and info_port > 0:
        return info_port
    return DEFAULT_SERVER_PORT


class SOLVER_OT_ForceTerminatePort(Operator):
    """Find and force-kill any process listening on the solver server port.

    Used as a recovery hatch when a previous Stop/Disconnect cycle left
    a ppf-cts-server.exe (or some other squatter) bound to the port,
    surfacing as ``Port N is in use`` on the next Connect attempt.
    Walks the process tree on Windows so child processes the server
    spawned (e.g. the CUDA solver subprocess) also go away.
    """

    bl_idname = "solver.force_terminate_port"
    bl_label = "Force Terminate Process"
    bl_description = (
        "Find and force-kill the process listening on the solver "
        "server port (recovery hatch for stuck ppf-cts-server)"
    )

    def execute(self, context):
        port = _resolved_server_port()
        pid = _find_pid_holding_port(port)
        if pid is None:
            self.report(
                {"WARNING"},
                f"No LISTENING process found on port {port}.",
            )
            return {"CANCELLED"}
        try:
            _kill_pid(pid)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            self.report(
                {"ERROR"},
                f"Failed to kill PID {pid} on port {port}: {stderr or exc}",
            )
            return {"CANCELLED"}
        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Failed to kill PID {pid} on port {port}: {exc}",
            )
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Killed PID {pid} (was holding port {port}).",
        )
        return {"FINISHED"}


classes = [
    SOLVER_OT_ShowConsole,
    SOLVER_OT_Terminate,
    SOLVER_OT_SaveAndQuit,
    SOLVER_OT_UpdateStatus,
    SOLVER_OT_ForceTerminatePort,
]
