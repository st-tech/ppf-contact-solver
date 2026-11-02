# File: frame_pump.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A dedicated modal operator that drives apply_animation + MESH_CACHE
# self-heal from a modal-operator timer context. This is the only context
# where Blender 5.x permits the ID writes these involve (writes to the
# State PropertyGroup, modifier.cache_format, scene.frame_start, etc.).
# The addon's persistent bpy.app.timers tick is NOT permissive and
# therefore cannot drive these writes.
#
# The operator runs only while it has work, and MUST NOT become
# resident. Blender refuses to auto-save while any modal operator handler
# is attached to a window: its auto-save timer re-arms itself for another
# 10 ms instead of writing, for as long as such a handler exists. A pump
# that stays up for the addon's lifetime therefore defers auto-save
# forever, and the user gets no recovery file for the whole session
# while the addon is merely enabled. work_pending() is the single
# predicate deciding both when to start and when to finish.

import time

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

_pump_stop_requested = False

_pump_error_reported: set[str] = set()

# The modal TIMER fires every 0.1s. Run the (O(N)) mesh-cache self-heal
# only every Nth tick (~once per second) so a large scene doesn't spend
# the main thread re-scanning every object 10x/second. Starts at 0 so the
# first tick heals immediately.
_heal_tick_counter = 0
_HEAL_EVERY_N_TICKS = 10

# A heal pass owed to the scene: requested when the addon registers and
# whenever a .blend is loaded, since Blender cancels every modal
# operator on file load and the pump is not resident to notice.
_heal_requested = False

# How long the modal keeps running after its work is gone. Without it a
# momentary idle between two stages of one solve would tear the modal
# down and rebuild it on the next 0.25s tick. Auto-save pays nothing for
# this: its retry cadence is 10 ms against an interval of minutes.
_IDLE_LINGER_S = 1.0


def _log_pump_error(msg: str) -> None:
    if msg in _pump_error_reported:
        return
    _pump_error_reported.add(msg)
    try:
        from ..models.console import console
        console.write(f"[frame pump] {msg}")
    except Exception:
        pass


def _request_stop() -> None:
    global _pump_stop_requested
    _pump_stop_requested = True


def _clear_stop() -> None:
    global _pump_stop_requested
    _pump_stop_requested = False


def request_heal() -> None:
    """Ask for one MESH_CACHE heal pass, starting the pump if needed.

    The heal is what rebinds a ContactSolverCache the scene lost, so it
    has to run once after the addon registers and once after every file
    load. Both are moments when no solve is in flight, which is exactly
    when the pump is otherwise absent.
    """
    global _heal_requested
    _heal_requested = True


def work_pending() -> bool:
    """True when something needs a modal-operator context.

    Three sources: a requested heal, an engine that is not idle (a solve
    building, running, fetching or applying), and frames still queued for
    apply_animation. The last one is separate from the second because a
    solve can return the state machine to idle while its final frames are
    still waiting in the runner's buffer.

    Both ensure_modal_running and the modal itself read this, so the
    start and stop conditions cannot drift apart.
    """
    if _heal_requested:
        return True
    try:
        from .facade import _engine_is_idle, communicator
        if not _engine_is_idle():
            return True
        return communicator.has_pending_animation_frames()
    except Exception as e:
        # Half-built facade during register or teardown. Report it and
        # keep the pump running: a spurious tick costs a deferred
        # auto-save, while a pump that refuses to start drops frames.
        _log_pump_error(f"work_pending failed, assuming work: {e}")
        return True


class PPF_OT_FramePump(Operator):
    """Internal modal that applies simulation frames and heals broken
    MESH_CACHE modifiers.

    Started on demand and finished as soon as work_pending() has been
    false for _IDLE_LINGER_S, so that an idle scene carries no modal
    handler and Blender can auto-save."""

    bl_idname = "ppf.frame_pump"
    bl_label = "PPF Frame Pump (internal)"
    bl_options = {"INTERNAL"}

    _timer = None
    _idle_since = 0.0

    def execute(self, context):
        # Reset so the first tick after a (re)start always heals once,
        # catching a post-load/reload MESH_CACHE teardown even when the
        # engine is idle (periodic heal is otherwise skipped at rest).
        global _heal_tick_counter
        _heal_tick_counter = 0
        self._idle_since = 0.0
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
            # apply_animation is the latency-sensitive path: it applies
            # simulation frames as they stream in from the solver (frame
            # arrival is async with no Blender event, hence the poll). It
            # early-outs to ~0ms when no frames are pending, so polling it
            # every tick is essentially free at rest.
            #
            # heal is pure self-repair (recreates a broken/missing
            # MESH_CACHE modifier). Scanning every assigned object is O(N)
            # in bpy API calls, so it must NOT run 10x/second on a large
            # scene. It only matters while caches may be changing (frames
            # arriving) or once right after the modal (re)starts to catch a
            # post-load/reload teardown. On a fully idle scene it is
            # skipped entirely, so an at-rest scene costs ~0 here.
            global _heal_tick_counter, _heal_requested
            run_heal = _heal_tick_counter == 0
            if not run_heal and _heal_tick_counter % _HEAL_EVERY_N_TICKS == 0:
                from .facade import _engine_is_idle
                run_heal = not _engine_is_idle()
            if run_heal:
                heal_mesh_caches_if_stale()
                # Settled by the pass above. Cleared after the call
                # rather than before so the debt cannot go missing
                # between the request and the work it asks for.
                _heal_requested = False
            _heal_tick_counter += 1
            apply_animation()
        except Exception as e:
            _log_pump_error(str(e))
        if self._idle_expired():
            self._detach_timer(context)
            return {"FINISHED"}
        return {"PASS_THROUGH"}

    def _idle_expired(self) -> bool:
        """True once work_pending() has been false for _IDLE_LINGER_S.

        Finishing is what lets Blender auto-save again, so this is the
        whole point of the operator being non-resident; see the module
        comment.
        """
        if work_pending():
            self._idle_since = 0.0
            return False
        now = time.monotonic()
        if self._idle_since == 0.0:
            self._idle_since = now
            return False
        return now - self._idle_since >= _IDLE_LINGER_S


def ensure_modal_running() -> str:
    """Spawn the frame-pump modal if work is pending and no live
    instance exists. Called both from the register-time kickoff timer
    and from the persistent Blender tick, so the modal starts whenever a
    solve, a queued frame or a requested heal needs a modal-operator
    context, and is absent the rest of the time so Blender can auto-save.

    The work_pending() gate sits before the heap walk below on purpose:
    at rest that walk would otherwise scan every Python object four
    times a second for a pump that has nothing to do.

    Returns the decision it reached, one of ``stopped``, ``unregistered``,
    ``no-work``, ``already-running``, ``started`` or ``error``. Callers
    ignore it; it is what makes the gate observable to a test, since
    whether a spawn happened is otherwise only visible through the event
    loop, which no synchronous caller can advance.
    """
    import gc
    if _pump_stop_requested:
        return "stopped"
    if getattr(bpy.types, "PPF_OT_frame_pump", None) is None:
        return "unregistered"
    if not work_pending():
        return "no-work"
    for obj in gc.get_objects():
        if type(obj).__name__ == "PPF_OT_FramePump" and getattr(obj, "_timer", None) is not None:
            return "already-running"
    try:
        bpy.ops.ppf.frame_pump("INVOKE_DEFAULT")
    except Exception as e:
        print(f"frame pump ensure: {e}")
        return "error"
    return "started"


def _kickoff_modal():
    _clear_stop()
    # Owed unconditionally: this is the addon's one chance to rebind a
    # ContactSolverCache in a .blend that was already open when the
    # addon was enabled, and it is also what starts the pump at all.
    request_heal()
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
