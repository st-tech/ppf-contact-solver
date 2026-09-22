# File: scenarios/rig_coincident_contact_pair.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: two sheets at bit-identical coordinates,
# one of them flagged `allow-inter-object-intersection`, build and run to the
# last frame with a clean `finished` outcome.
#
# Coincident sheets are the hardest shape an allowed pair can take: every
# closest point collapses to zero, so the contact normal would be a 0/0
# normalize and the barrier singular. An allowed pair is not a contact pair,
# so neither the barrier nor the CCD line search ever meets it, and the run
# has nothing to jam on. A regression that let the allowed pair back into
# contact ends this run with `overlapping_start` (or a device assert) instead,
# which is what B names.
#
# The control, A, is the same geometry with no allowance: the build gate must
# refuse it, which is what establishes that the sheets really do overlap.
#
# The `overlapping_start` report itself stays covered end to end by
# `rig_collider_coincident_pair`, which reaches it with no allowance at all (a
# dynamic sheet inside a static floor's contact offset). The shared-embed
# branch, a DYNAMIC pair collapsed to its offset, has no authored scene: a
# coincident dynamic pair is refused at build unless it is allowed, and an
# allowed one is out of contact.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. unallowed_overlap_is_refused
#         the same two sheets with no allowance fail `scene.build()`.
#   B. allowed_overlap_runs_to_the_end
#         with the allowance, the build succeeds, the run reaches its last
#         frame, and `status.cbor` records `kind == "finished"`.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the contact filter in the
# neutral kernels, which every backend renders from the same body.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import sys

import cbor2

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def tail(path, limit=700):
    """Last `limit` characters of a solver log, or a marker when absent."""
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read()[-limit:]
    except OSError:
        return "<absent>"


FRAMES = 4


def author(allow):
    app = App.create("rig_coincident_contact_pair_" + ("allowed" if allow else "control"))
    # `mesh.square` returns 5 columns: xyz then uv, so only [:, :3] is a position.
    V, F = app.mesh.square(res=4, ex=[1, 0, 0], ey=[0, 1, 0])
    app.asset.add.tri("sheet", V, F)
    scene = app.scene.create()
    a = scene.add("sheet").at(0.0, 0.0, 0.0)
    b = scene.add("sheet").at(0.0, 0.0, 0.0)
    if allow:
        a.param.set("allow-inter-object-intersection", 1.0)
    # One edge each, so the sheets hold the overlap instead of free-falling
    # out of it.
    a.pin(a.grab([0, 1, 0]))
    b.pin(b.grab([0, 1, 0]))
    return app, scene


def build(scene):
    try:
        return scene.build(quiet=True), ""
    except Exception as error:
        return None, "{}: {}".format(
            type(error).__name__, str(error).splitlines()[0][:200])


_, control_scene = author(False)
control_fixed, control_error = build(control_scene)
record("A_unallowed_overlap_is_refused", control_fixed is None,
       {"build_error": control_error})

app, scene = author(True)
fixed, build_error = build(scene)
details = {"build_error": build_error}
finished = False
kind = ""
if fixed is not None:
    session = app.session.create(fixed)
    session.param.set("dt", 0.01).set("frames", FRAMES)
    session = session.build()
    try:
        session.start(blocking=True)
    except Exception as error:
        details["start_raised"] = "{}: {}".format(
            type(error).__name__, str(error))[:300]
    finished = bool(session.finished())
    status_path = os.path.join(session.info.path, "output", "status.cbor")
    if os.path.isfile(status_path):
        with open(status_path, "rb") as handle:
            status_record = cbor2.load(handle)
        outcome = (status_record.get("payload") or {}).get("outcome") or {}
        kind = str(outcome.get("kind", ""))
        details["sub_kind"] = str(outcome.get("sub_kind", ""))
        details["detail"] = str(outcome.get("detail", ""))[:300]
    else:
        details["status_record"] = "absent at " + status_path
        details["error_log"] = tail(os.path.join(session.info.path, "error.log"))
        details["stdout_log"] = tail(os.path.join(session.info.path, "stdout.log"))
details.update({"finished": finished, "kind": kind})
record("B_allowed_overlap_runs_to_the_end",
       fixed is not None and finished and kind == "finished", details)

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX],
        capture_output=True,
        text=True,
        env=env,
        # A real solve pays a CUDA context init and a solver startup before
        # its first frame, so the budget covers startup rather than simulation.
        timeout=max(ctx.timeout, 600.0),
    )
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("PPFRESULT")
    ]
    if not marker:
        return r.failed([
            "probe produced no result marker; "
            f"rc={proc.returncode} stderr={proc.stderr[-800:]!r}"
        ])
    cases = json.loads(marker[-1][len("PPFRESULT"):])
    return r.report_named_checks(cases, label="coincident-pair cases")
