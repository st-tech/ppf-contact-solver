# File: scenarios/bl_probe_sampler_phase_change.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The rig's probe can sample a phase change without deadlocking Blender.
#
# The probe (debug/probe.py) samples the add-on's state from a timer. Its
# sampler used to write the `phase_change` event while still holding its own
# lock, and the writer takes that lock again; the lock is not reentrant, so the
# first phase change froze Blender's main thread for good. No scenario saw it,
# because every driver held the main thread for the whole run and the
# sampler's timer never fired mid-scenario. It surfaced only when
# `bl_servers_answer_outside_calls` handed the main thread back, as a scenario
# timing out with nothing in its logs.
#
#   A_phase_change_sampled  with the last phase forced stale, one sample
#                           returns within 10 s and records a `phase_change`
#                           event. The sample runs on a worker thread joined
#                           with a timeout, so a deadlock is reported here by
#                           name instead of hanging the scenario.
#   B_lock_released         afterwards the probe's lock can be taken, so the
#                           sampler left nothing held.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import json
import os
import threading
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    probe = __import__(pkg + ".debug.probe", fromlist=["_singleton", "start"])
    p = probe._singleton
    if not p.running:
        probe.start(os.path.join(PROBE_DIR or os.path.dirname(RESULT_PATH),
                                 "probe_regression"))
    events_path = p.events_path

    def phase_changes():
        try:
            with open(events_path, encoding="utf-8") as f:
                return sum(1 for line in f if '"phase_change"' in line)
        except OSError:
            return 0

    before = phase_changes()
    p.last_phase = ("stale", "stale", "stale")
    done = threading.Event()
    failure = []

    def sample():
        try:
            p._sample()
        except Exception as exc:
            failure.append(repr(exc))
        done.set()

    threading.Thread(target=sample, daemon=True).start()
    returned = done.wait(timeout=10.0)
    if not returned and p.lock.locked():
        # A deadlocked sampler is stuck on the lock it already holds. A plain
        # Lock can be released from another thread, so free it here: the
        # sampler then finishes, and Blender can still shut down, which keeps
        # the failure a named check rather than a scenario timeout.
        p.lock.release()
        done.wait(timeout=5.0)
    after = phase_changes()
    dh.record("A_phase_change_sampled",
              returned and not failure and after == before + 1,
              {"returned_within_10s": returned, "error": failure,
               "phase_change_events": [before, after]})

    released = p.lock.acquire(timeout=2.0)
    if released:
        p.lock.release()
    dh.record("B_lock_released", released, {"acquired": released})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
