# File: scenarios/rig_coincident_contact_pair.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: a contact pair whose separation has
# collapsed to the contact offset ends the advance as
# `CrashKind::OverlappingStart`, and the cause it names is
# `contact_separated = false`.
#
# Two sheets at bit-identical coordinates are the shape that produces it. An
# overlapping scene normally leaves a small but non-zero separation, which the
# barrier can still take a direction from; coincident sheets drive a pair's two
# closest points to the same point, so the contact normal is a 0/0 normalize
# and the barrier is singular there. The assembly therefore contributes nothing
# for that pair and reports it instead, and the host turns the report into a
# terminal crash record naming the flag.
#
# WHY THE CONDITION MUST REPORT RATHER THAN ASSERT. NDEBUG is absent from the
# cargo CUDA build and present in the Windows one (the flags are in
# `crates/ppf-cts-solver/build.rs` and `build-win-native/`),
# so an assert on this condition is a live device trap on one shipped build and
# no code at all on the other, leaving that one to normalize a zero vector and
# assemble a NaN into the system. Neither outcome names a cause. The
# reachability is not hypothetical: contact never consults an intersection
# allowance, so an allowance admits exactly the scenes that can put an authored
# pair into this state, and a supported feature can reach it.
#
# WHY NOT A CHEAPER TEST.
#   A device unit test sees one hop.
#   `test_accd_degenerate.cu` drives `ccd_helper` against synthetic globals and
#   checks the flag it sets, which is the first of five: the assembly branch,
#   the host readback after the line search, `StepResult::contact_separated`,
#   `crash_kind_from_step`, and the status record the addon panel reads. A
#   regression at any later hop leaves that test green.
#   A scene-build test cannot reach it either. `rig_intersection_allowances`
#   settles which scenes the allowance ADMITS, and admitting this one is only
#   the precondition here; what the solver then does with it is a separate
#   question and needs the solver to run.
#   So the check needs a solver that assembles contact and runs a line
#   search, which is `BACKENDS = ("real",)`, and on CI that is the GPU jobs.
#
# WHY `status.cbor` AND NOT THE CRASH DUMP. Not every build of the solver
# writes the dump, so a check written against it reports a missing file rather
# than a wrong crash kind wherever the dump is absent, which is green where it
# was written and vacuous everywhere else. `output/status.cbor`
# carries the same verdict on every tree: `payload.outcome.sub_kind` is the
# spelling `CrashKind::tag` emits, and `payload.outcome.detail` carries the
# `StepResult` booleans the sub-kind was derived from.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. scene_is_admitted_by_the_allowance
#         `allow-inter-object-intersection` on one sheet is what gets the scene
#         past the build gate. Without it the run under test never happens, so
#         a failure here means B and C prove nothing, and says so.
#   B. advance_reports_overlapping_start
#         `outcome.sub_kind == "overlapping_start"`. A `device_assert` here
#         means the assembly is trapping on the degenerate pair instead of
#         reporting it; an absent record means the process died before it could
#         write one, and the details carry the solver's own log tails.
#   C. cause_is_contact_separated
#         `outcome.detail` carries `contact_separated=false`. The sub-kind
#         alone does not establish which boolean produced it, and
#         `crash_kind_from_step` has six inputs.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# Real backend only: the reported condition is evaluated in device contact
# assembly, which the solver does not compile. Selected by the AWS
# GPU jobs via ``runtests --backend real``.
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


app = App.create("rig_coincident_contact_pair")
# `mesh.square` returns 5 columns: xyz then uv, so only [:, :3] is a position.
V, F = app.mesh.square(res=4, ex=[1, 0, 0], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)

scene = app.scene.create()
a = scene.add("sheet").at(0.0, 0.0, 0.0)
b = scene.add("sheet").at(0.0, 0.0, 0.0)
# The allowance is what admits the scene at all; the build gate would otherwise
# refuse it and the run under test would never happen. Nothing downstream
# consults the allowance, which is the point: contact meets this pair either
# way, and what the solver reports for it is the subject of B and C.
a.param.set("allow-inter-object-intersection", 1.0)
# One edge each, so the sheets hold the overlap instead of free-falling out of
# it before a contact pair is ever assembled.
a.pin(a.grab([0, 1, 0]))
b.pin(b.grab([0, 1, 0]))

fixed = None
build_error = ""
try:
    fixed = scene.build(quiet=True)
except Exception as error:
    build_error = "{}: {}".format(
        type(error).__name__, str(error).splitlines()[0][:200])

record("A_scene_is_admitted_by_the_allowance", fixed is not None,
       {"build_error": build_error})

outcome = {}
sub_kind = ""
detail = ""
notes = {}

if fixed is not None:
    session = app.session.create(fixed)
    # Three frames is more than the run needs: the pair is coincident at t=0,
    # so the very first advance is the one that has to report. The extra frames
    # exist so a run that DOES advance is visibly different from one that
    # reports on entry.
    session.param.set("dt", 0.01).set("frames", 3)
    session = session.build()
    try:
        session.start(blocking=True)
    except Exception as error:
        # The advance is EXPECTED to fail, so the exception carries no verdict.
        # Which cause the run RECORDED is the assertion, and that is read from
        # the status record below.
        notes["start_raised"] = "{}: {}".format(
            type(error).__name__, str(error))[:300]

    status_path = os.path.join(session.info.path, "output", "status.cbor")
    if os.path.isfile(status_path):
        with open(status_path, "rb") as handle:
            status_record = cbor2.load(handle)
        outcome = (status_record.get("payload") or {}).get("outcome") or {}
        sub_kind = str(outcome.get("sub_kind", ""))
        detail = str(outcome.get("detail", ""))
    else:
        # No terminal record at all. A device assert kills the process on the
        # trap, so the solver's logs are the only evidence left of what stopped
        # it.
        notes["status_record"] = "absent at " + status_path
        notes["error_log"] = tail(os.path.join(session.info.path, "error.log"))
        notes["stdout_log"] = tail(os.path.join(session.info.path, "stdout.log"))

b_details = {"kind": str(outcome.get("kind", "")), "sub_kind": sub_kind}
b_details.update(notes)
record("B_advance_reports_overlapping_start",
       sub_kind == "overlapping_start", b_details)

# Spaces are stripped so the check reads the token itself and not the wrapping
# of the host's format string.
record("C_cause_is_contact_separated",
       "contact_separated=false" in detail.replace(" ", ""),
       {"detail": detail[:300]})

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
        # A real solve pays a CUDA context init and a solver startup before it
        # can report anything, and this run is expected to end on its first
        # advance, so the budget covers startup rather than simulation.
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
