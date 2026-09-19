# File: solver_control_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Solver console/terminate/status operators.

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_

from ..core.client import RemoteStatus, communicator as com
from ..models.console import console


class SOLVER_OT_ShowConsole(Operator):
    """Show a console window with read-only text."""

    bl_idname = "solver.show_console"
    bl_label = "Show Console"

    def execute(self, _):
        console.show()
        return {"FINISHED"}


class SOLVER_OT_OpenSessionFolder(Operator):
    """Open the folder holding the solver logs and status record for this run."""

    bl_idname = "solver.open_session_folder"
    bl_label = "Open Session Folder"

    @classmethod
    def session_path(cls) -> str:
        """Filesystem path this operator opens, or ``""`` when unavailable.

        The remote root names a path on the machine the SERVER runs on, so
        this reaches real files only when that machine is this one. The
        operator opens what it is given and lets the OS report a path it
        cannot open, rather than deciding for the user which roots are local.
        """
        root = com.normalized_remote_root()
        return f"{root}/session" if root else ""

    @classmethod
    def poll(cls, _):
        return (
            com.info.status == RemoteStatus.SIMULATION_FAILED
            and bool(cls.session_path())
        )

    def execute(self, _):
        path = self.session_path()
        if not path:
            self.report({"WARNING"}, iface_("No session folder to open."))
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
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
        self.report({"INFO"}, iface_("Save and Quit executed."))
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


classes = [
    SOLVER_OT_ShowConsole,
    SOLVER_OT_OpenSessionFolder,
    SOLVER_OT_Terminate,
    SOLVER_OT_SaveAndQuit,
    SOLVER_OT_UpdateStatus,
]
