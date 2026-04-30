# File: addon_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reload server lifecycle operators.

from bpy.types import Operator  # pyright: ignore

from ..models.groups import get_addon_data


class ADDON_OT_StartReloadServer(Operator):
    """Start the addon reload server."""

    bl_idname = "addon.start_reload_server"
    bl_label = "Start Reload Server"

    def execute(self, context):
        from ..core.reload_server import start_reload_server

        state = get_addon_data(context.scene).state

        try:
            start_reload_server(state.reload_port)
            self.report({"INFO"}, f"Reload server started on port {state.reload_port}")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to start reload server: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


class ADDON_OT_StopReloadServer(Operator):
    """Stop the addon reload server."""

    bl_idname = "addon.stop_reload_server"
    bl_label = "Stop Reload Server"

    def execute(self, context):
        from ..core.reload_server import stop_reload_server

        try:
            stop_reload_server()
            self.report({"INFO"}, "Reload server stopped")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to stop reload server: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


class ADDON_OT_TriggerReload(Operator):
    """Trigger addon reload immediately."""

    bl_idname = "addon.trigger_reload"
    bl_label = "Trigger Reload"

    def execute(self, context):
        from ..core.reload_server import trigger_reload_now

        try:
            trigger_reload_now()
            self.report({"INFO"}, "Add-on reload triggered")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to trigger reload: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


class ADDON_OT_TriggerFullReload(Operator):
    """Full addon reload — splits disable and enable across event-loop
    ticks so Blender rebuilds RNA for classes with nested CollectionProperty
    bindings. Use this when a PropertyGroup schema change doesn't appear
    after the regular reload."""

    bl_idname = "addon.trigger_full_reload"
    bl_label = "Trigger Full Reload"

    def execute(self, context):
        from ..core.reload_server import trigger_full_reload_now

        try:
            trigger_full_reload_now()
            self.report({"INFO"}, "Add-on full reload triggered")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to trigger full reload: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


classes = [
    ADDON_OT_StartReloadServer,
    ADDON_OT_StopReloadServer,
    ADDON_OT_TriggerReload,
    ADDON_OT_TriggerFullReload,
]
