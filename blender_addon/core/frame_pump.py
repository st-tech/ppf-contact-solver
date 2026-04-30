# File: frame_pump.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A dedicated modal operator that drives apply_animation + MESH_CACHE
# self-heal from a modal-operator timer context. This is the only context
# where Blender 5.x permits the ID writes these involve (writes to the
# State PropertyGroup, modifier.cache_format, scene.frame_start, etc.).
# The addon's persistent bpy.app.timers tick is NOT permissive and
# therefore cannot drive these writes.

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

_pump_stop_requested = False


def _request_stop() -> None:
    global _pump_stop_requested
    _pump_stop_requested = True


def _clear_stop() -> None:
    global _pump_stop_requested
    _pump_stop_requested = False


class PPF_OT_FramePump(Operator):
    """Internal modal that applies simulation frames and heals broken
    MESH_CACHE modifiers. Runs for the entire lifetime of the addon."""

    bl_idname = "ppf.frame_pump"
    bl_label = "PPF Frame Pump (internal)"
    bl_options = {"INTERNAL"}

    _timer = None

    def execute(self, context):
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _detach_timer(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

    def cancel(self, context):
        """Called by Blender when the operator is being torn down, e.g.
        when its class is unregistered during addon reload. Must clean
        up the event timer so Blender doesn't keep dispatching TIMER
        events into a handler whose class is gone."""
        self._detach_timer(context)

    def modal(self, context, event):
        if _pump_stop_requested:
            self._detach_timer(context)
            return {"CANCELLED"}
        # If the addon was reloaded, our class is stale. Check whether
        # the registry still points at the class we're an instance of;
        # if not, bail out so we don't read from a freed operator type.
        current = getattr(bpy.types, "PPF_OT_frame_pump", None)
        if current is None or current is not self.__class__:
            self._detach_timer(context)
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            from .client import apply_animation, heal_mesh_caches_if_stale
            heal_mesh_caches_if_stale()
            apply_animation()
        except Exception as e:
            print(f"frame pump error: {e}")
        return {"PASS_THROUGH"}


def ensure_modal_running():
    """Spawn the frame-pump modal if no live instance exists. Called
    both from the register-time kickoff timer and from the persistent
    Blender tick, so the modal self-heals after file-open, scene change
    or reload events that cancel it. Cheap early-out when one is
    already running."""
    import gc
    if _pump_stop_requested:
        return
    if getattr(bpy.types, "PPF_OT_frame_pump", None) is None:
        return
    for obj in gc.get_objects():
        if type(obj).__name__ == "PPF_OT_FramePump" and getattr(obj, "_timer", None) is not None:
            return
    try:
        bpy.ops.ppf.frame_pump("INVOKE_DEFAULT")
    except Exception as e:
        print(f"frame pump ensure: {e}")


def _kickoff_modal():
    _clear_stop()
    ensure_modal_running()
    return None  # one-shot


def register() -> None:
    try:
        bpy.utils.register_class(PPF_OT_FramePump)
    except ValueError:
        # The class is still registered from a previous session (e.g.
        # reload that skipped unregister). Reuse it.
        pass
    # Deferred start: bpy.ops must not be called during addon register.
    # Deregister any stale kickoff timer still in the queue from a
    # previous generation so two kickoffs don't race.
    if bpy.app.timers.is_registered(_kickoff_modal):
        try:
            bpy.app.timers.unregister(_kickoff_modal)
        except ValueError:
            pass
    bpy.app.timers.register(_kickoff_modal, first_interval=1.0)


def unregister() -> None:
    # Ask the running modal to stop on its next tick. Blender also
    # invokes our cancel() method when the class is unregistered, which
    # detaches the event timer. The modal additionally self-cancels if
    # it detects its class object is no longer the registered one, so
    # any stale instance left over from an earlier reload will bail out
    # cleanly on its next modal invocation instead of reading a freed
    # operator type.
    _request_stop()
    # Remove the deferred kickoff timer if it hasn't fired yet so it
    # can't re-invoke the modal after the class has been unregistered.
    if bpy.app.timers.is_registered(_kickoff_modal):
        try:
            bpy.app.timers.unregister(_kickoff_modal)
        except ValueError:
            pass
    try:
        bpy.utils.unregister_class(PPF_OT_FramePump)
    except RuntimeError:
        pass
