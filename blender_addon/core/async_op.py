# File: async_op.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import time

from bpy.types import Operator  # pyright: ignore

from . import encode_progress
from .utils import get_timer_wait_time, redraw_all_areas


class StageAbort(Exception):
    """Raised by a staged-work callback to cancel the run with a user-facing
    message. :meth:`AsyncOperator.modal` catches it, reports it through
    ``self.report`` at ``level`` (default ``ERROR``), and ends the modal."""

    def __init__(self, message: str, level: str = "ERROR"):
        super().__init__(message)
        self.level = level


class AsyncOperator(Operator):
    """Base class for modal operators with timer-based polling.

    Subclasses must implement:
        is_complete() -> bool: Return True when the operation is done.

    Optionally override:
        on_complete(context): Called when is_complete() returns True.
        on_timeout(context): Called when timeout is exceeded.
        timeout: float = 60.0: Maximum seconds before timeout.

    Optionally drive blocking pre-work with :meth:`start_stages`: a list of
    ``(label, fn)`` stages is run one per timer tick (so the panel paints a
    labeled progress bar between them) before the modal falls through to the
    ``is_complete()`` wait. Used by Transfer / Run to surface the otherwise
    silent scene-encode and drift-check phases.
    """

    _timer = None
    _start_time: float = 0.0
    timeout: float = 60.0
    auto_redraw: bool = False
    _stages = None
    _stage_index: int = 0

    def is_complete(self) -> bool:
        raise NotImplementedError

    def is_cancelled(self) -> bool:
        return False

    def on_complete(self, context):
        pass

    def on_timeout(self, context):
        self.report({"ERROR"}, "Operation timed out")

    def setup_modal(self, context):
        self._start_time = time.time()
        self._timer = context.window_manager.event_timer_add(
            get_timer_wait_time(), window=context.window
        )
        context.window_manager.modal_handler_add(self)

    def start_stages(self, context, stages):
        """Begin a staged modal: run each ``(label, fn)`` in *stages* one per
        timer tick, publishing an ``encode_progress`` snapshot between them so
        the panel paints a labeled bar from the moment the operator returns
        ``RUNNING_MODAL``. ``fn(context)`` does one chunk of blocking work and
        may raise :class:`StageAbort` to cancel with a message. The final
        stage typically dispatches the async request; once every stage has
        run, the modal proceeds to the normal ``is_complete()`` wait.

        Call this from ``execute`` instead of ``setup_modal`` and return
        ``{"RUNNING_MODAL"}``.
        """
        self._stages = list(stages)
        self._stage_index = 0
        first_label = self._stages[0][0] if self._stages else ""
        encode_progress.begin(len(self._stages), first_label)
        # Paint the bar with the first stage's label now, during the ~one-tick
        # gap before the first timer fires, so that label is on screen while
        # the first (blocking) stage runs rather than appearing only after it.
        redraw_all_areas(context)
        self.setup_modal(context)

    def _end_stages(self):
        # Only clear the shared snapshot when THIS operator owns an active
        # staged run, so a non-staging operator's teardown can't wipe a bar
        # another operator is publishing.
        if self._stages is not None:
            encode_progress.end()
        self._stages = None
        self._stage_index = 0

    def _run_stage_tick(self, context):
        """Run the current stage. Returns a modal result dict to return now,
        or ``None`` when every stage has finished (fall through to the
        completion poll)."""
        _label, fn = self._stages[self._stage_index]
        try:
            fn(context)
        except StageAbort as exc:
            self._end_stages()
            self.cleanup_modal(context)
            self.report({exc.level}, str(exc))
            redraw_all_areas(context)
            return {"CANCELLED"}
        except Exception as exc:  # noqa: BLE001 - surface any encode failure
            self._end_stages()
            self.cleanup_modal(context)
            self.report({"ERROR"}, f"Encoding failed: {exc}")
            redraw_all_areas(context)
            return {"CANCELLED"}
        self._stage_index += 1
        if self._stage_index >= len(self._stages):
            # All stages done; the dispatching stage (if any) has set the
            # server status, so let the completion poll take over this tick.
            self._end_stages()
            return None
        # Publish the next (now-current) stage so the upcoming redraw shows
        # its label and an advanced bar before it blocks the main thread.
        encode_progress.update(
            self._stage_index, self._stages[self._stage_index][0]
        )
        redraw_all_areas(context)
        return {"PASS_THROUGH"}

    def cleanup_modal(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except (RuntimeError, ReferenceError):
                pass
            self._timer = None

    def cancel(self, context):
        """Called by Blender when the operator class is torn down (e.g.
        addon reload). Drop the event timer so it can't fire against a
        freed handler, and clear any in-flight encode-progress bar."""
        self._end_stages()
        self.cleanup_modal(context)

    def _is_stale_class(self):
        """Detect an instance whose class has been replaced by a module
        reload: ``bpy.types`` now points at a different class object
        bearing the same ``bl_idname``. Touching ``self`` properties on
        that instance risks reading a freed operator type."""
        import bpy  # pyright: ignore
        bl_idname = getattr(self.__class__, "bl_idname", "")
        if not bl_idname:
            return False
        # bl_idname like "ssh.run_command" → bpy.types.SSH_OT_run_command
        try:
            area, name = bl_idname.split(".", 1)
        except ValueError:
            return False
        registry_name = f"{area.upper()}_OT_{name}"
        current = getattr(bpy.types, registry_name, None)
        return current is not None and current is not self.__class__

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # If the addon was reloaded while this modal was alive, our
        # class object is no longer the one registered under our
        # bl_idname. Bail out before touching attributes that would
        # read from a freed operator type.
        if self._is_stale_class():
            self._end_stages()
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self.auto_redraw:
            redraw_all_areas(context)
        if self.is_cancelled():
            self._end_stages()
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        # Run any staged pre-work one stage per tick, before the completion
        # poll, so the panel paints the labeled encode bar between stages.
        if self._stages is not None:
            staged = self._run_stage_tick(context)
            if staged is not None:
                return staged
        if time.time() - self._start_time > self.timeout:
            self.cleanup_modal(context)
            self.on_timeout(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self.is_complete():
            self.cleanup_modal(context)
            self.on_complete(context)
            return {"FINISHED"}
        return {"PASS_THROUGH"}
