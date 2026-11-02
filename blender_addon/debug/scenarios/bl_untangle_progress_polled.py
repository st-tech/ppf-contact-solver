# File: scenarios/bl_untangle_progress_polled.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The shell untangler's child process finishes while Blender reads its
# progress without pause (issue #159).
#
# Resolve All Intersections runs `core/shell_untangle.Run`, a child process
# that writes `progress.json` by writing a sibling file and replacing it, while
# the add-on's modal handler opens `progress.json` to read it at every tick.
# On Windows a file Python opened cannot be replaced until it is closed, so a
# replace that lands during a read is refused with WinError 5, and that one
# refusal ended the whole run. At the modal's own pace the two meet only now
# and then (`bl_resolve_all_intersections` part H never saw it); here Blender
# reads the progress in a loop for the whole run, which fails every run within
# a second on Windows when the progress file is written in place.
#
#   `A_runs_finish_while_progress_is_read_in_a_loop`: three runs, each read in
#       a loop until it ends, all finish, converge, and give what the same
#       untangle gives in this process.
#   `B_the_reads_saw_the_progress_move`: the loop read several distinct
#       progress values in every run, so the reads overlapped the child's
#       writes and the race was exercised rather than missed.
#
# The child is Blender's own Python, as it is for an artist, which is the
# interpreter the issue was reported on.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r

NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it. It never connects to the
# solver, so nothing in it is backend-specific.
BACKENDS = ("real",)

# POSIX replaces a file another process has open, so the race this checks
# exists only on Windows.
PLATFORMS = ("win32",)


_DRIVER_TEMPLATE = r"""
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

RUNS = 3
RUN_TIMEOUT = 240.0


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


def sheet(su, name, n, phase, lift, fx, fy):
    u, v = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    z = lift + 0.02 * np.sin(fx * u + phase) * np.cos(fy * v)
    vert = np.stack([u, v, z], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tri = np.concatenate([np.stack([a.ravel(), b.ravel(), c.ravel()], 1),
                          np.stack([a.ravel(), c.ravel(), d.ravel()], 1)])
    return su.Surface(name, vert, tri.astype(np.int64))


try:
    su = __import__(pkg + ".core.shell_untangle", fromlist=["Run"])
    # Two wavy sheets laid through each other.
    surfaces = [sheet(su, "Under", 12, 0.0, 0.0, 5.0, 4.0),
                sheet(su, "Over", 12, 0.6, 0.004, 4.0, 5.0)]
    expected, expected_report = su.untangle(surfaces, su.Settings())
    log("reference_done")

    runs = []
    for index in range(RUNS):
        run = su.Run(surfaces)
        run.start()
        seen, polls = set(), 0
        started = time.monotonic()
        while not run.finished and time.monotonic() - started < RUN_TIMEOUT:
            # What the modal reads at every tick (`_publish`), read in a loop.
            seen.add((run.stage, run.iteration, run.pairs, tuple(sorted(run.status.items()))))
            polls += 1
        entry = {"finished": run.finished, "polls": polls, "distinct": len(seen),
                 "seconds": round(time.monotonic() - started, 2)}
        if not run.finished:
            run.kill()
            entry["error"] = "still running after %.0f s" % RUN_TIMEOUT
        else:
            try:
                out, report = run.result()
                entry["converged"] = bool(report.converged)
                entry["same"] = all(np.array_equal(a, b) for a, b in zip(out, expected))
            except Exception as exc:
                entry["error"] = "%s: %s" % (type(exc).__name__, exc)
        runs.append(entry)
        log("run_%d_done" % index)

    record("A_runs_finish_while_progress_is_read_in_a_loop",
           expected_report.converged
           and all(e["finished"] and "error" not in e and e.get("converged") and e.get("same")
                   for e in runs),
           {"runs": runs, "reference_converged": bool(expected_report.converged)})
    record("B_the_reads_saw_the_progress_move",
           all(e["distinct"] >= 3 for e in runs),
           {"distinct": [e["distinct"] for e in runs], "polls": [e["polls"] for e in runs]})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    # No DriverHelpers here, so the build-failure helpers are prepended.
    return dl.BUILD_FAILURE_LIB + _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
