# File: async_op.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import time

from bpy.types import Operator  # pyright: ignore

from .utils import get_timer_wait_time, redraw_all_areas


class AsyncOperator(Operator):
    """Base class for modal operators with timer-based polling.

    Subclasses must implement:
        is_complete() -> bool: Return True when the operation is done.

    Optionally override:
        on_complete(context): Called when is_complete() returns True.
        on_timeout(context): Called when timeout is exceeded.
        timeout: float = 60.0: Maximum seconds before timeout.
    """

    _timer = None
    _start_time: float = 0.0
    timeout: float = 60.0
    auto_redraw: bool = False

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
        freed handler."""
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
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self.auto_redraw:
            redraw_all_areas(context)
        if self.is_cancelled():
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
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


class Pipeline:
    """Simple stage-based pipeline for multi-stage modal operators.

    Usage:
        pipeline = Pipeline([
            ("stage_name", start_fn, complete_fn),
            ...
        ])

    Each stage is a tuple of (name, start_callable, is_complete_callable).
    - start_callable(context): Called when the stage begins.
    - is_complete_callable(): Returns True when the stage is done.

    The pipeline auto-advances through stages. Use with AsyncOperator by
    calling pipeline.start(context) in execute() and checking
    pipeline.is_complete() in the operator's is_complete().
    """

    def __init__(self, stages):
        self._stages = stages
        self._current = 0

    @property
    def stage_name(self):
        if self._current < len(self._stages):
            return self._stages[self._current][0]
        return None

    def start(self, context):
        self._current = 0
        if self._stages:
            self._stages[0][1](context)

    def advance(self, context):
        """Check current stage completion and advance if ready. Returns True when all done."""
        if self._current >= len(self._stages):
            return True
        _, _, is_done = self._stages[self._current]
        if is_done():
            self._current += 1
            if self._current < len(self._stages):
                self._stages[self._current][1](context)
            else:
                return True
        return self._current >= len(self._stages)

    def is_complete(self, context):
        """Single call that checks and advances. Returns True when pipeline is done."""
        return self.advance(context)
