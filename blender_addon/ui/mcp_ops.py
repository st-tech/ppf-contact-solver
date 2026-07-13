# File: mcp_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP server start/stop operators.

from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_
from bpy.types import Operator  # pyright: ignore

from ..core.utils import redraw_all_areas
from ..models.groups import get_addon_data


class MCP_OT_StartServer(Operator):
    """Start the MCP server."""

    bl_idname = "mcp.start_server"
    bl_label = "Start MCP Server on Local"

    def execute(self, context):
        from ..mcp.mcp_server import get_mcp_server, is_mcp_running, start_mcp_server

        try:
            state = get_addon_data(context.scene).state
            original_port = state.mcp_port

            start_mcp_server(original_port)

            if is_mcp_running():
                server = get_mcp_server()
                actual_port = getattr(server, "port", original_port)

                if actual_port != original_port:
                    self.report(
                        {"WARNING"},
                        iface_(
                            "MCP server started on alternative port {actual_port} "
                            "(requested port {original_port} was unavailable)"
                        ).format(actual_port=actual_port, original_port=original_port),
                    )
                    state.mcp_port = actual_port
                else:
                    self.report(
                        {"INFO"},
                        iface_("MCP server started successfully on port {actual_port}").format(
                            actual_port=actual_port
                        ),
                    )
            else:
                self.report({"ERROR"}, iface_("Failed to start MCP server - unknown error"))
                return {"CANCELLED"}

            redraw_all_areas(context)
            return {"FINISHED"}

        except Exception as e:
            error_msg = str(e)
            if "port" in error_msg.lower():
                self.report(
                    {"ERROR"},
                    iface_(
                        "Port conflict: {error}. Try reloading the addon or using a different port."
                    ).format(error=error_msg),
                )
            else:
                self.report(
                    {"ERROR"},
                    iface_("Failed to start MCP server: {error}").format(error=error_msg),
                )
            return {"CANCELLED"}


class MCP_OT_StopServer(Operator):
    """Stop the MCP server."""

    bl_idname = "mcp.stop_server"
    bl_label = "Stop MCP Server"

    def execute(self, context):
        from ..mcp.mcp_server import is_mcp_running, stop_mcp_server

        try:
            if not is_mcp_running():
                self.report({"INFO"}, iface_("MCP server is not running"))
                return {"FINISHED"}

            stop_mcp_server()

            if not is_mcp_running():
                self.report({"INFO"}, iface_("MCP server stopped successfully"))
            else:
                self.report(
                    {"WARNING"},
                    iface_("MCP server may still be running - check console for details"),
                )

            redraw_all_areas(context)
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, iface_("Error stopping MCP server: {error}").format(error=e))
            return {"CANCELLED"}


classes = [
    MCP_OT_StartServer,
    MCP_OT_StopServer,
]
