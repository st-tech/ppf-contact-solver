# File: engine.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The event loop that drives the pure state machine.
#
# Events arrive from any thread via ``dispatch()`` into a thread-safe queue.
# On every Blender main-thread timer tick, ``tick()`` drains the queue,
# runs each event through the pure ``transition()`` function, and hands the
# resulting ``Effect`` list to the ``EffectRunner`` for execution.
#
# State is only mutated inside ``tick()`` (main thread), so no locks are
# needed for the business logic.  A single ``threading.Lock`` protects the
# ``_state`` field for thread-safe reads from other threads (e.g. UI panels
# calling ``engine.state`` from ``draw()``).

from __future__ import annotations

import collections
import queue
import sys
import threading
import time
import traceback
from typing import Callable

from .events import Event
from .state import AppState
from .transitions import transition


class _NonReentrantLock:
    """``threading.Lock`` that crashes loudly if the same thread tries to
    re-acquire it while already holding it.

    Why: we had a hung debug scenario where ``with engine._lock: engine.state``
    re-entered the lock through the ``state`` property and silently
    deadlocked. A plain ``threading.Lock`` makes that bug invisible. This
    wrapper turns it into an immediate crash with a stack trace so the
    offending call site is obvious.
    """

    __slots__ = ("_lock", "_owner")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            sys.stderr.write(
                "FATAL: non-reentrant engine lock re-acquired by the same "
                f"thread (ident={me}). This is a deadlock bug; the second "
                "acquire would block forever. Most common cause: calling a "
                "property or helper that re-enters the lock from inside "
                "`with engine._lock:`.\nStack:\n"
            )
            traceback.print_stack(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError("engine lock re-acquired by same thread")
        if blocking and timeout < 0:
            ok = self._lock.acquire()
        else:
            ok = self._lock.acquire(blocking=blocking, timeout=timeout)
        if ok:
            self._owner = me
        return ok

    def release(self) -> None:
        self._owner = None
        self._lock.release()

    def __enter__(self) -> "_NonReentrantLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# Type alias for state-change listeners.
# ``fn(old_state, new_state)`` -- called on the main thread.
StateListener = Callable[[AppState, AppState], None]


# Capacity of the dispatch ring buffer used for debug/observability.
# Large enough to capture a full fetch cycle (hundreds of ProgressUpdated
# events) but bounded so it never grows without limit.
_RECENT_EVENTS_CAPACITY = 500


class Engine:
    """Event-driven state machine engine.

    Usage::

        engine = Engine()
        runner = EffectRunner(engine)

        # From any thread:
        engine.dispatch(SomeEvent(...))

        # From Blender main-thread timer (every 0.25 s):
        engine.tick(runner)

        # Read current state (thread-safe):
        s = engine.state
    """

    def __init__(self) -> None:
        self._state = AppState()
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: list[StateListener] = []
        self._lock = _NonReentrantLock()
        # Bounded ring buffer of recently-dispatched events for
        # post-mortem debugging. deque.append is thread-safe; the
        # snapshot-returning property copies under the engine lock so a
        # concurrent dispatch can't tear the read.
        self._recent_events: collections.deque[tuple[float, str, str]] = (
            collections.deque(maxlen=_RECENT_EVENTS_CAPACITY)
        )

    # -- state access --

    @property
    def state(self) -> AppState:
        """Thread-safe snapshot of current state."""
        with self._lock:
            return self._state

    # -- event dispatch --

    def dispatch(self, event: Event) -> None:
        """Submit an event from any thread.  Non-blocking."""
        self._recent_events.append(
            (time.monotonic(), type(event).__name__, repr(event)[:200])
        )
        self._queue.put(event)

    @property
    def recent_events(self) -> list[tuple[float, str, str]]:
        """Thread-safe snapshot of the most recent dispatched events."""
        with self._lock:
            return list(self._recent_events)

    def has_pending(self) -> bool:
        """Best-effort check for queued events. Safe to race — a false
        negative just delays processing by one tick."""
        return not self._queue.empty()

    # -- main-thread processing --

    def tick(self, runner: "EffectRunner") -> None:  # noqa: F821
        """Drain the event queue and process all pending events.

        Must be called from Blender's main thread (e.g. inside a timer
        callback or a modal operator's ``modal()``).

        For each event:
        1. Run ``transition(state, event)`` to get ``(new_state, effects)``.
        2. Update ``self._state``.
        3. Notify listeners if state changed.
        4. Execute each effect via the runner.
        """
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break

            old_state = self._state
            new_state, effects = transition(self._state, event)

            with self._lock:
                self._state = new_state

            if old_state != new_state:
                for listener in self._listeners:
                    listener(old_state, new_state)

            for effect in effects:
                runner.execute(effect)

    # -- listeners --

    def on_change(self, fn: StateListener) -> None:
        """Register a listener called whenever state changes.

        Listeners run on the main thread (inside ``tick()``).
        """
        self._listeners.append(fn)

    def remove_listener(self, fn: StateListener) -> None:
        """Remove a previously registered listener."""
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    # -- reset --

    def reset(self) -> None:
        """Reset engine to initial state and clear the queue."""
        with self._lock:
            self._state = AppState()
            self._recent_events.clear()
        # Drain any pending events
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
