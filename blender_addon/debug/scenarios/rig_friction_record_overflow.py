# File: scenarios/rig_friction_record_overflow.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A frictional sheet that LANDS during a step must not stop the run.
#
# The friction memory records one entry per frictional pair, into a capacity
# that grows to twice any count past half of it, and the step commits the
# record its LAST Newton iteration wrote. When a sheet arrives on another
# within a step, the trailing error-reduction iteration can count more than
# twice the pairs of the iteration before it, so its record overflows. The run
# used to stop there with "the friction memory's last record counted N
# frictional pair(s) against a capacity of M"; the Newton loop now runs that
# iteration again into the grown capacity (`driver/step.rs`, STEP B33).
#
# THE SCENE is the add-on's default drop at 100 fps and a 0.01 s step: a
# 0.5 sheet 4 mm above a held 1.0 sheet, both at friction 0.5 and a 1 mm
# contact gap, the held sheet pinned in place (a STATIC collision mesh) in one
# case and pinned with a slow move (in the solved namespace) in the other.
# Measured before the fix: the static case stopped at its first contact (251
# pairs against a capacity of 144); the moving one happened to fit.
#
# Checks, per case:
#   <case>_runs_to_the_last_frame  the run finishes, with no FATAL in its log.
#   <case>_sheet_rests_on_held     the sheet ends on the held one, not
#                                  through it (contact still holds it).
# And once: repeat_path_was_taken, that some case's log shows the
# error-reduction iteration repeated, so the scenario exercised the overflow
# rather than a scene whose pair count happened to fit.
#
# NO BLENDER, and a SUBPROCESS, as `_existing_intersection.py` runs its probe.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


PROBE = r'''
import json
import os
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

FRAMES = 20
GAP = 0.004
REPEAT = "repeating the error reduction step"
cases = {}
repeated = False
repeat_counts = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}
    print("PPFCASE %s %s" % (name, "pass" if ok else "fail"), flush=True)


for case, moving in (("static_held", False), ("pinned_held", True)):
    app = App.create("rig_friction_overflow_" + case)
    V, F = app.mesh.square(res=10, size=1.0, ex=[1, 0, 0], ey=[0, 0, 1])
    app.asset.add.tri("held", np.asarray(V)[:, :3], np.asarray(F))
    V, F = app.mesh.square(res=6, size=0.5, ex=[1, 0, 0], ey=[0, 0, 1])
    app.asset.add.tri("sheet", np.asarray(V)[:, :3], np.asarray(F))
    scene = app.scene.create()
    held = scene.add("held").at(0.0, 0.0, 0.0)
    holder = held.pin()
    if moving:
        holder.move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
    sheet = scene.add("sheet").at(0.0, GAP, 0.0)
    for obj in (held, sheet):
        obj.param.set("friction", 0.5)
        obj.param.set("contact-gap", 0.001)
    log_text = ""
    try:
        fixed = scene.build(quiet=True)
        session = app.session.create(fixed)
        session.param.set("frames", FRAMES)
        session.param.set("fps", 100.0)
        session.param.set("dt", 0.01)
        session = session.build()
        session.start(blocking=True)
        finished = bool(session.finished())
        first, _ = session.get.vertex(0)
        last, _ = session.get.vertex(FRAMES - 1)
        first = np.asarray(first, dtype=np.float64)[:, :3]
        last = np.asarray(last, dtype=np.float64)[:, :3]
        # The session's own log, read through the session so the path
        # resolves the same way on every OS (Windows keeps its data root
        # under the tree, not under ~/.local/share).
        lines = session.get.log.stdout()
        logs = [session.info.path]
        log_text = "\n".join(lines)
        error = None
    except Exception as exc:
        finished, error = False, repr(exc)[:400]
    fatal = [l for l in log_text.splitlines() if "FATAL" in l][:2]
    n_repeat = log_text.count(REPEAT)
    repeated = repeated or n_repeat > 0
    repeat_counts[case] = {"repeats": n_repeat, "log_lines": len(log_text.splitlines())}
    check(case + "_runs_to_the_last_frame",
          finished and not fatal and error is None,
          {"finished": finished, "fatal": fatal, "error": error,
           "logs": len(logs) if error is None else 0, "repeats": n_repeat})
    if error is None:
        mine = np.abs(first[:, 1] - GAP) < 1e-4
        rest = float(last[mine, 1].mean()) if mine.any() else None
        check(case + "_sheet_rests_on_held",
              finished and rest is not None and 0.0 < rest < GAP,
              {"rest_y": rest, "vertices": int(mine.sum())})
    else:
        check(case + "_sheet_rests_on_held", False, {"error": error})

check("repeat_path_was_taken", repeated, repeat_counts)
print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, REPO_ROOT_POSIX],
            capture_output=True, text=True, env=env,
            timeout=max(ctx.timeout, 900.0),
        )
    except subprocess.TimeoutExpired as expired:
        return r.failed([f"probe timed out after {expired.timeout:.0f}s"])
    marker = [l for l in proc.stdout.splitlines() if l.startswith("PPFRESULT")]
    if not marker:
        return r.failed([
            "probe produced no result marker; "
            f"rc={proc.returncode} stderr={proc.stderr[-800:]!r}"
        ])
    cases = json.loads(marker[-1][len("PPFRESULT"):])
    return r.report_named_checks(cases, label="friction record overflow cases")
