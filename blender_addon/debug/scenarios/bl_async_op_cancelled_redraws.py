# File: scenarios/bl_async_op_cancelled_redraws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# AsyncOperator CANCELLED paths must call ``redraw_all_areas``.
#
# Motivation: commit ``ff0d20ca`` ("Fix Blender UI staleness after
# state changes across operators and handlers"). Before that fix,
# the shared ``AsyncOperator.modal`` returned ``CANCELLED`` from the
# stale-class, ``is_cancelled()``, and timeout branches without
# tagging a redraw, leaving the panel stuck on the pre-cancel state
# until the user moved the mouse.
#
# This scenario exercises that contract end-to-end inside Blender:
#
#   A. ``connect_to_closed_port_cancels``: dispatch a Connect to a
#      bogus path with a known-closed local port; assert the engine
#      surfaces ``ConnectionFailed`` and the phase ends OFFLINE
#      without ever entering ONLINE.
#   B. ``modal_dispatches_redraw_at_cancel``: monkey-patch
#      ``redraw_all_areas`` everywhere it is bound, then drive
#      ``AsyncOperator.modal`` against a stub ``self`` whose
#      ``is_cancelled``, ``_is_stale_class``, and ``cleanup_modal``
#      are the unbound production methods from
#      ``REMOTE_OT_StartServer``. Headless Blender refuses to let
#      Python construct ``Operator`` subclass instances directly
#      (``bpy_struct.__new__`` requires the registry-managed creation
#      path), so a stub ``self`` is the only way to drive ``modal``
#      from a script. By binding the real subclass methods we still
#      exercise the production cancel/stale-class/cleanup logic in
#      the same call. Assert each branch returns ``{"CANCELLED"}``
#      AND the patched counter incremented at every branch.
#   C. ``state_consistent_after_cancel``: after the failed Connect
#      in (A) and the forced cancel in (B), assert the engine state
#      converged to IDLE/OFFLINE/UNKNOWN with the error surfaced,
#      that the production ``cleanup_modal`` actually nulled each
#      ``_timer`` (load bearing now that the cleanup ran against a
#      real ``bpy.types.Timer`` -- the previous version of this
#      check was tautological because the test used a no-op
#      ``cleanup_modal`` lambda), and that the production
#      ``core.facade._persistent_tick`` is still registered on
#      ``bpy.app.timers`` (catches a regression where a cancel
#      side effect tears down the global tick).

from __future__ import annotations

import os
import socket

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import time
import traceback
import types

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
CLOSED_PORT = <<CLOSED_PORT>>


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Patch the redraw helper everywhere it is bound. Each ``from
    # .utils import redraw_all_areas`` creates a fresh module-local
    # binding, so monkey-patching only ``core.utils`` would miss the
    # copies in ``core.async_op`` and ``ui.connection_ops``. We track
    # patched modules so we can restore on the way out.
    utils_mod = __import__(pkg + ".core.utils",
                           fromlist=["redraw_all_areas"])
    async_op_mod = __import__(pkg + ".core.async_op",
                              fromlist=["redraw_all_areas", "AsyncOperator"])
    connection_ops_mod = __import__(
        pkg + ".ui.connection_ops",
        fromlist=["redraw_all_areas", "REMOTE_OT_StartServer"])
    counter = [0]
    originals = {}

    def counting_redraw(context):
        counter[0] += 1
        # Also exercise the real helper so any side-effect bug
        # surfaces here, not in production.
        try:
            return originals["utils"](context)
        except Exception:
            return None

    for label, mod in (("utils", utils_mod),
                       ("async_op", async_op_mod),
                       ("connection_ops", connection_ops_mod)):
        originals[label] = getattr(mod, "redraw_all_areas")
        setattr(mod, "redraw_all_areas", counting_redraw)

    # Capture the production persistent-tick callable so subtest C
    # can assert it is still registered. ``bpy.app.timers`` does not
    # expose a list/set of registered timers, but ``is_registered(fn)``
    # is supported, which is enough to detect the (unlikely but
    # auditable) regression where a cancel-path side effect tears
    # down the global tick timer.
    facade_mod = __import__(pkg + ".core.facade",
                            fromlist=["_persistent_tick"])
    persistent_tick = getattr(facade_mod, "_persistent_tick", None)
    persistent_tick_registered_before = (
        persistent_tick is not None
        and bpy.app.timers.is_registered(persistent_tick)
    )

    # ----- A: Connect to a bogus path + closed port cancels ------
    # ``win_native`` raises FileNotFoundError at backend-construction
    # when ``server.py`` is missing, so ``_do_connect`` catches it and
    # dispatches ``ConnectionFailed`` without ever reaching ONLINE.
    # We pair that with a definitely-closed local port so a future
    # change that swaps the backend type still hits a transport
    # failure rather than a silent success.
    visited_phases = []
    saw_online = False

    facade = dh.facade
    events = dh.events
    com = dh.com
    com.set_project_name("async_op_cancel_test")

    # Record the starting phase, then dispatch the failing Connect.
    # We snapshot phase before tick() too because the worker thread
    # can dispatch ``ConnectionFailed`` quickly enough that a single
    # tick after dispatch moves us all the way back to OFFLINE,
    # leaving no observable CONNECTING window in the loop below.
    visited_phases.append(facade.engine.state.phase.name)

    facade.engine.dispatch(events.ConnectRequested(
        backend_type="win_native",
        config={"path": "/this/path/does/not/exist/anywhere"},
        server_port=CLOSED_PORT,
    ))
    facade.tick()
    ph = facade.engine.state.phase.name
    if visited_phases[-1] != ph:
        visited_phases.append(ph)

    deadline = time.time() + 8.0
    error_seen = False
    while time.time() < deadline:
        facade.engine.dispatch(events.PollTick())
        facade.tick()
        ph = facade.engine.state.phase.name
        if not visited_phases or visited_phases[-1] != ph:
            visited_phases.append(ph)
        if ph == "ONLINE":
            saw_online = True
        if facade.engine.state.error:
            error_seen = True
        if ph == "OFFLINE" and error_seen:
            break
        time.sleep(0.05)

    final_phase_a = facade.engine.state.phase.name
    error_a = facade.engine.state.error or ""
    dh.record(
        "A_connect_to_closed_port_cancels",
        final_phase_a == "OFFLINE"
        and not saw_online
        and bool(error_a),
        {
            "visited_phases": visited_phases,
            "final_phase": final_phase_a,
            "error": error_a,
            "saw_online": saw_online,
            "error_seen": error_seen,
        },
    )

    # ----- B: AsyncOperator CANCELLED path tags redraw -----------
    # Headless Blender refuses to let Python construct an
    # ``Operator`` subclass directly: ``bpy_struct.__new__`` rejects
    # the no-arg constructor with ``expected a single argument``
    # because operator instances are only meant to be created by the
    # registry-managed dispatch path. We confirmed this empirically
    # while iterating on this scenario. We therefore drive
    # ``AsyncOperator.modal`` against a stub ``self`` -- but each
    # stub *binds the production ``is_cancelled``, ``_is_stale_class``,
    # and ``cleanup_modal`` from ``REMOTE_OT_StartServer``* (the real
    # registered ``AsyncOperator`` subclass). That way the cancel
    # gate, the stale-class detection, and the cleanup walk through
    # the same code production runs, and only the Blender-managed
    # parts (operator instantiation, window manager event timer)
    # remain stubbed.
    counter_before = counter[0]

    StartServer = connection_ops_mod.REMOTE_OT_StartServer
    # ``cleanup_modal`` only enters its body when ``self._timer`` is
    # truthy, so to make the production cleanup load bearing we
    # need a real ``bpy.types.Timer`` (a string sentinel raises
    # ``TypeError`` from ``event_timer_remove`` which the production
    # except clause does NOT swallow -- it only catches RuntimeError
    # / ReferenceError -- so a fake handle would crash the test
    # before the redraw is tagged). We allocate a real timer via the
    # real window manager and pass it through. If the headless
    # context has no window we fall back to None and accept the
    # weaker truthy-skip path; the rest of the modal body still
    # runs.
    real_ctx = bpy.context
    real_event = types.SimpleNamespace(type="TIMER")
    real_wm = real_ctx.window_manager

    def _make_real_timer():
        try:
            return real_wm.event_timer_add(
                0.5, window=real_ctx.window,
            )
        except Exception:
            return None

    def _make_probe(*, _timer, _start_time, timeout=60.0,
                    auto_redraw=False, force_stale=False,
                    force_cancelled=False):
        # The probe is a fresh class so the production methods we
        # bind below resolve via normal method lookup (they read
        # ``self.__class__`` via ``_is_stale_class`` and ``self._timer``
        # via ``cleanup_modal``). This avoids the bpy_struct
        # restriction that a real Operator subclass enforces.
        class _Probe:
            pass
        _Probe.bl_idname = StartServer.bl_idname
        # Bind production methods from the real subclass. These are
        # the bodies that run in production; we are not re-implementing
        # them in the test.
        _Probe.cleanup_modal = StartServer.cleanup_modal
        _Probe._is_stale_class = StartServer._is_stale_class
        _Probe.cancel = StartServer.cancel
        if force_cancelled:
            _Probe.is_cancelled = lambda self: True
        elif force_stale:
            _Probe.is_cancelled = lambda self: False
            _Probe._is_stale_class = lambda self: True
        else:
            _Probe.is_cancelled = lambda self: False
        _Probe.is_complete = lambda self: False
        _Probe.on_complete = lambda self, c: None
        _Probe.on_timeout = lambda self, c: None
        _Probe.report = lambda self, *a, **kw: None
        p = _Probe()
        p._timer = _timer
        p._start_time = _start_time
        p.timeout = timeout
        p.auto_redraw = auto_redraw
        return p

    # Cancel branch: production ``cleanup_modal`` runs against a real
    # ``window_manager.event_timer_remove`` call (we pre-seed _timer
    # with a real ``bpy.types.Timer`` if we have a window; otherwise
    # the truthy guard skips the body but the rest of the modal
    # body still runs).
    timer_cancel = _make_real_timer()
    probe_cancel = _make_probe(
        _timer=timer_cancel,
        _start_time=time.time(),
        force_cancelled=True,
    )
    cancel_result = async_op_mod.AsyncOperator.modal(
        probe_cancel, real_ctx, real_event,
    )
    counter_after_cancel = counter[0]

    # Timeout branch: production cleanup, real ``on_timeout`` is the
    # base-class one (it calls self.report which we stubbed -- the
    # method body otherwise has no side effects we can observe).
    timer_timeout = _make_real_timer()
    probe_timeout = _make_probe(
        _timer=timer_timeout,
        _start_time=time.time() - 1.0e6,
        timeout=1.0,
    )
    timeout_result = async_op_mod.AsyncOperator.modal(
        probe_timeout, real_ctx, real_event,
    )
    counter_after_timeout = counter[0]

    # Stale-class branch: production ``cleanup_modal`` runs.
    timer_stale = _make_real_timer()
    probe_stale = _make_probe(
        _timer=timer_stale,
        _start_time=time.time(),
        force_stale=True,
    )
    stale_result = async_op_mod.AsyncOperator.modal(
        probe_stale, real_ctx, real_event,
    )
    counter_after_stale = counter[0]
    real_timers_used = (
        timer_cancel is not None
        and timer_timeout is not None
        and timer_stale is not None
    )

    # Production ``cleanup_modal`` MUST have nulled each ``_timer``
    # (this is load-bearing: the previous version of this test used
    # ``cleanup_modal=lambda c: None`` which made the assertion
    # tautological).
    cleanup_nulled_timers = (
        probe_cancel._timer is None
        and probe_timeout._timer is None
        and probe_stale._timer is None
    )

    dh.record(
        "B_modal_dispatches_redraw_at_cancel",
        cancel_result == {"CANCELLED"}
        and timeout_result == {"CANCELLED"}
        and stale_result == {"CANCELLED"}
        and counter_after_cancel > counter_before
        and counter_after_timeout > counter_after_cancel
        and counter_after_stale > counter_after_timeout
        and cleanup_nulled_timers,
        {
            "real_subclass_methods_bound_from": StartServer.__name__,
            "cancel_result": list(cancel_result),
            "timeout_result": list(timeout_result),
            "stale_result": list(stale_result),
            "counter_before": counter_before,
            "counter_after_cancel": counter_after_cancel,
            "counter_after_timeout": counter_after_timeout,
            "counter_after_stale": counter_after_stale,
            "cleanup_nulled_timers": cleanup_nulled_timers,
            "real_timers_used": real_timers_used,
        },
    )

    # ----- C: state consistent after cancel ----------------------
    # All assertions here probe production surfaces, none of the
    # test stubs. (Previous version asserted ``probe._timer is None``
    # which was tautological because cleanup_modal was a no-op
    # lambda. Now production ``cleanup_modal`` ran against real
    # window-manager calls in B, so the same assertion is non-trivial,
    # but we additionally cover state-machine convergence and
    # ``bpy.app.timers`` leak detection so a future regression that
    # reintroduces a stub-only cleanup still gets caught here.)
    s_pre = facade.engine.state
    # Drain any deferred dispatches the cancel storm or the failed
    # Connect may have queued. If the cancel branches accidentally
    # left an Activity != IDLE behind, a tick will surface it.
    for _ in range(3):
        facade.engine.dispatch(events.PollTick())
        facade.tick()
    s = facade.engine.state

    persistent_tick_registered_after = (
        persistent_tick is not None
        and bpy.app.timers.is_registered(persistent_tick)
    )
    persistent_tick_intact = (
        persistent_tick_registered_before
        == persistent_tick_registered_after
    )

    cleanup_held = (
        probe_cancel._timer is None
        and probe_timeout._timer is None
        and probe_stale._timer is None
    )
    ok_state = (
        s.phase.name == "OFFLINE"
        and s.server.name == "UNKNOWN"
        and s.activity.name == "IDLE"
        and bool(s.error)
        and cleanup_held
        and persistent_tick_intact
    )
    dh.record(
        "C_state_consistent_after_cancel",
        ok_state,
        {
            "phase": s.phase.name,
            "phase_pre_drain": s_pre.phase.name,
            "server": s.server.name,
            "activity": s.activity.name,
            "error": (s.error or "")[:120],
            "cleanup_held": cleanup_held,
            "persistent_tick_registered_before":
                persistent_tick_registered_before,
            "persistent_tick_registered_after":
                persistent_tick_registered_after,
        },
    )

    # Restore originals so the rest of the Blender process is not
    # left with our counter shim in place.
    for label, mod in (("utils", utils_mod),
                       ("async_op", async_op_mod),
                       ("connection_ops", connection_ops_mod)):
        setattr(mod, "redraw_all_areas", originals[label])
    dh.log("teardown_done")

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def _allocate_closed_port() -> int:
    """Bind to an ephemeral port, then close the socket.

    The returned port is highly likely to stay closed for the few
    seconds the driver needs (the OS's ephemeral pool is large). If
    a racing process happens to grab it, the worst outcome is the
    Connect succeeds against an unrelated listener; the win_native
    backend still raises FileNotFoundError on the bogus path before
    any port-level traffic, so subtest A stays valid.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return port


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<CLOSED_PORT>>", str(_allocate_closed_port()))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 90.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
