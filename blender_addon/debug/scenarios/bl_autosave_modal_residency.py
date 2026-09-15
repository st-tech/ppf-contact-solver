# File: scenarios/bl_autosave_modal_residency.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for issue #145: with the addon merely enabled, Blender
# never wrote an auto-save file, so the user had nothing under
# File > Recover > Auto Save for the whole session.
#
# Blender skips its auto-save for as long as ANY modal operator handler is
# attached to a window: the auto-save timer re-arms itself for another
# 10 ms and returns without writing. The addon held two modal operators
# open indefinitely. PPF_OT_FramePump ran for the addon's lifetime, which
# is why enabling the addon was enough to reproduce, and REMOTE_OT_Connect
# ran for the whole lifetime of a server connection.
#
# Neither may be resident. This scenario pins the predicates that decide
# it, which is what the event loop consults; the loop itself is invisible
# to the rig, so a scenario that waited for a real auto-save would certify
# nothing.
#
#   A. idle_scene_has_no_work: on an idle scene with no solve and no
#      queued frames, work_pending() is False. This is the state the
#      reporter was in, and the only state in which an auto-save is due.
#   B. ensure_is_noop_at_rest: ensure_modal_running() starts NOTHING while
#      work_pending() is False, reporting "no-work". The pump used to be
#      spawned unconditionally from the persistent tick, which is what
#      made it resident. The reason string is what makes this observable:
#      whether a spawn happened otherwise shows up only through the event
#      loop, which the rig driver cannot advance.
#   C. heal_request_is_work: request_heal() counts as work, so a heal owed
#      after register or a file load still starts the pump.
#   D. queued_frame_is_work: a frame waiting in the runner's buffer counts
#      as work even with the engine idle. A solve can return the state
#      machine to idle with its last frames still queued, and those frames
#      need the modal context to be applied.
#   E. linger_before_finish: the modal does not finish the instant work
#      disappears. _idle_expired() arms a clock on the first workless tick
#      and only reports True once _IDLE_LINGER_S has passed, so a momentary
#      idle inside one solve cannot tear the pump down and rebuild it.
#   F. work_resets_the_linger: work reappearing during the linger disarms
#      the clock.
#   G. connect_modal_finishes_once_connected: REMOTE_OT_Connect ends at the
#      handshake instead of holding a handler open for the connection's
#      lifetime, and releases its event timer on the way out.
#   H. load_post_requests_heal: the load_post handler that replaces the
#      revive-on-file-open behavior is registered and does owe a heal.
#      Blender cancels every modal operator on file load, and a
#      non-resident pump is not there to notice.
#
# Assertion-only: no server connection or solve.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, types, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def live_pumps():
    import gc
    return [o for o in gc.get_objects()
            if type(o).__name__ == "PPF_OT_FramePump"
            and getattr(o, "_timer", None) is not None]


def mock_context():
    ctx = types.SimpleNamespace()
    ctx.window_manager = types.SimpleNamespace(
        event_timer_remove=lambda t: None,
        event_timer_add=lambda **kw: None,
        modal_handler_add=lambda op: None,
    )
    ctx.screen = types.SimpleNamespace(areas=[])
    ctx.window = None
    return ctx


try:
    fp = __import__(pkg + ".core.frame_pump",
                    fromlist=["work_pending", "request_heal",
                              "ensure_modal_running", "PPF_OT_FramePump"])
    facade = __import__(pkg + ".core.facade", fromlist=["communicator"])
    conn = __import__(pkg + ".ui.connection_ops",
                      fromlist=["REMOTE_OT_Connect", "com"])
    root = __import__(pkg, fromlist=["_request_cache_heal_on_load"])
    ctx = bpy.context

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    ctx.view_layer.update()

    # Settle any heal owed by addon register so the scene starts at rest.
    # The pump clears the debt on its own first tick, which is a modal
    # body the rig cannot drive, so clear it here instead. Check H ends
    # the scenario owing one again, which is the correct state to leave:
    # the pump serves it on the next tick and retires.
    fp._heal_requested = False

    # A: the reporter's state. Nothing to do, so nothing may be running.
    a_work = fp.work_pending()
    record("A_idle_scene_has_no_work", a_work is False, {"work_pending": a_work})

    # B: the spawn is gated, not unconditional. Asserted through the
    #    reason string rather than a pump count, because the pump the
    #    addon's own register kicked off may still be lingering when this
    #    scenario runs, and a count cannot tell "refused to start" from
    #    "one was already up".
    b_reason = fp.ensure_modal_running()
    record("B_ensure_is_noop_at_rest", b_reason == "no-work",
           {"reason": b_reason, "live_pumps": len(live_pumps())})

    # C: an owed heal is work.
    fp.request_heal()
    c_work = fp.work_pending()
    fp._heal_requested = False
    record("C_heal_request_is_work", c_work is True, {"work_pending": c_work})

    # D: a queued frame is work on its own, with the engine idle. The
    #    driver holds the main thread for its whole body, so no timer can
    #    consume the sentinel between the push and the pop.
    runner = facade.communicator._runner
    with runner._anim_lock:
        runner._anim_frames.append((0, None, None, None))
    d_pending = facade.communicator.has_pending_animation_frames()
    d_work = fp.work_pending()
    with runner._anim_lock:
        runner._anim_frames.clear()
    d_after = fp.work_pending()
    record("D_queued_frame_is_work",
           d_pending is True and d_work is True and d_after is False,
           {"has_pending": d_pending, "work_with_frame": d_work,
            "work_without_frame": d_after})

    # E: the modal lingers rather than finishing on the first idle tick.
    #    A Blender Operator cannot be constructed from Python, so the
    #    predicate runs against a stand-in carrying the one field it
    #    reads, the way the addon's own unit tests drive modal().
    op = types.SimpleNamespace(_idle_since=0.0)
    op._idle_expired = types.MethodType(
        fp.PPF_OT_FramePump._idle_expired, op)
    first = op._idle_expired()          # arms the clock
    armed = op._idle_since
    op._idle_since = time.monotonic() - fp._IDLE_LINGER_S - 1.0
    expired = op._idle_expired()
    record("E_linger_before_finish",
           first is False and armed != 0.0 and expired is True,
           {"first_call": first, "clock_armed": armed != 0.0,
            "expired_after_linger": expired,
            "linger_s": fp._IDLE_LINGER_S})

    # F: work reappearing during the linger disarms the clock.
    op._idle_since = time.monotonic() - fp._IDLE_LINGER_S - 1.0
    fp.request_heal()
    f_expired = op._idle_expired()
    f_clock = op._idle_since
    fp._heal_requested = False
    record("F_work_resets_the_linger",
           f_expired is False and f_clock == 0.0,
           {"expired_while_working": f_expired, "clock": f_clock})

    # G: the connect operator ends at the handshake. It must not hold a
    #    handler open for the connection's lifetime.
    com = conn.com
    fake = types.SimpleNamespace(
        _connection_established=False,
        _timer=object(),
        _start_time=time.time(),
        timeout=60.0,
        report=lambda *a, **kw: None,
    )
    fake._detach_timer = types.MethodType(
        conn.REMOTE_OT_Connect._detach_timer, fake)
    com.is_connected = lambda: True
    com.is_connecting = lambda: False
    try:
        verdict = conn.REMOTE_OT_Connect.modal(
            fake, mock_context(), types.SimpleNamespace(type="TIMER"))
    finally:
        del com.is_connected
        del com.is_connecting
    record("G_connect_modal_finishes_once_connected",
           verdict == {"FINISHED"}
           and fake._timer is None
           and fake._connection_established is True,
           {"verdict": sorted(verdict), "timer_released": fake._timer is None,
            "established": fake._connection_established})

    # H: file load still owes a heal, since Blender cancels every modal
    #    operator on load and the pump is not resident to notice.
    names = [getattr(h, "__name__", "") for h in bpy.app.handlers.load_post]
    fp._heal_requested = False
    root._request_cache_heal_on_load()
    h_owed = fp._heal_requested
    record("H_load_post_requests_heal",
           "_request_cache_heal_on_load" in names and h_owed is True,
           {"handlers": names, "heal_owed": h_owed})

    result["phases"].append((round(time.time(), 3),
                             "checks=" + str(len(result["checks"]))))
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
