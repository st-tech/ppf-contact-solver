# File: scenarios/rig_collider_coincident_pair.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: when a DYNAMIC vertex starts inside a
# STATIC collider's contact offset, the advance ends as
# `CrashKind::OverlappingStart` AND the solver names the pair through one of the
# three COLLISION-MESH assembly paths, never through the shared embed.
#
# WHY THE STATIC COLLIDER IS THE POINT. The contact assembly has four places
# that can report a separation already collapsed to the contact offset, and they
# are not interchangeable. The shared embed in `contact/contact.cu` reports kind
# 6: it funnels every dynamic-vs-dynamic pair type through one call, so it can
# name no type, and both of its indices are in the one dynamic vertex space. The
# collision-mesh vertex-face, face-vertex and edge-edge paths do their own
# barrier math, report kinds 7, 8 and 9, and report a MIXED pair whose first
# index is a dynamic vertex and whose second is in the static collision-mesh
# vertex space. `main/main.cu` maps those three to
# "vertex-face (collision mesh, assembly)", "face-vertex (collision mesh,
# assembly)" and "edge-edge (collision mesh, assembly)", and prints the
# index-space note as a further line for exactly kinds 7 to 9.
#
# A scene built from two DYNAMIC bodies reaches kind 6 and nothing else, so the
# three collision-mesh kinds, the strings they map to, and the index-space note
# ride along unexercised: swap the reported index pair, or hand a
# collision-mesh branch a neighbor's kind number, and the run still crashes with
# the same sub-kind and still names two numbers. This scene has exactly one
# dynamic body and one static collider, so the pair the assembly finds can only
# come from a collision-mesh path, and the string it prints is what says which.
#
# HOW THE SCENE REACHES THE SOLVER. It does not intersect and needs no
# intersection allowance. The dynamic sheet hovers `GAP` above the static floor
# and carries `contact-offset = OFFSET` with `GAP < OFFSET`, so the two surfaces
# are apart in space and inside the summed contact offset at t=0, which is the
# state the assembly reports. The frontend admits it because its build-time
# contact-offset scan measures dynamic elements against each other only
# (`kernels/fixed_scene_assemble.rs` runs that step over `dyn_verts` and the
# dynamic triangle and rod arrays), so dynamic-vs-static clearance is not a
# quantity it computes. Subtest A pins that: should the frontend ever measure
# it, A fails and says the rest of the run proves nothing.
#
# WHY NOT A CHEAPER TEST.
#   A device unit test sees one hop. It can drive a
#   collision-mesh branch against synthetic globals and read the latch, which is
#   the first of six: the branch, the first-writer CAS that decides which pair
#   is named, the host readback, the kind-to-string table, `crash_kind_from_step`
#   over `StepResult::contact_separated`, and the status record. A regression at
#   any later hop leaves that test green.
#   A scene-build test cannot reach it. The frontend never measures the
#   clearance this scene depends on, so it has nothing to say about the scene,
#   let alone about what the solver then reports for it.
#   `rig_coincident_contact_pair` cannot reach it either. Two dynamic sheets
#   settle the shared embed, kind 6, which is the one branch these three are
#   distinguished FROM.
#   So the check needs a solver that assembles contact, which is
#   `BACKENDS = ("real",)`, and on CI that is the GPU jobs.
#
# WHY THE SOLVER LOG AND NOT ONLY `status.cbor`. The status record carries the
# verdict (`payload.outcome.sub_kind`) and the `StepResult` booleans behind it
# (`payload.outcome.detail`), and both are read here. Neither carries the KIND:
# the kind is resolved to a string on the host and printed, so the solver's own
# `stdout.log` under the session directory is the only place the assembly path
# is named. The crash dump is not an option for any of it, because not every
# build of the solver writes one; a check written against it reports a missing
# file rather than a wrong kind wherever the dump is absent.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. scene_is_admitted_without_a_clearance_gate
#         The build has to accept a dynamic vertex inside a static collider's
#         contact offset, or the run under test never happens.
#   B. index_spaces_are_separable
#         Checked numerically against the authored arrays, with no solver: every
#         static vertex whose index is small enough to pass for a dynamic one
#         sits far from the sheet. This is what makes F able to see a swapped
#         pair, and it fails loudly if the mesh generator's vertex ordering
#         moves rather than letting F pass vacuously.
#   C. advance_reports_overlapping_start
#         `outcome.sub_kind == "overlapping_start"`. A `device_assert` here
#         means a collision-mesh branch is trapping instead of reporting; an
#         absent record means the process died before writing one.
#   D. cause_is_contact_separated
#         `outcome.detail` carries `contact_separated=false`. The sub-kind alone
#         does not establish which boolean produced it.
#   E. kind_is_a_collision_mesh_assembly_path
#         The report line names exactly one of the three collision-mesh assembly
#         strings. Naming "contact assembly" instead means a collision-mesh pair
#         is being reported under the shared embed's kind.
#   F. reported_pair_lands_in_its_own_index_space
#         First index in the dynamic space, second in the static one, with B's
#         separation making a swap visible as an out-of-range first index.
#   G. second_index_is_named_as_collision_mesh_space
#         The extra line that tells a reader which space the second index is in
#         is printed. It is keyed on the kind, so a kind that drifts out of the
#         7 to 9 window takes this line with it.

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
import re
import sys

import cbor2
import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

# The sheet's contact offset, and the clearance it starts with. A collision-mesh
# branch reports when the pair's squared separation has fallen to the SUMMED
# offset of its two sides, so GAP < OFFSET is the whole authoring requirement.
# GAP > 0 keeps the surfaces apart, so nothing here is an intersection and the
# solver's t=0 intersection scan (an edge-versus-face crossing test) has nothing
# to find.
OFFSET = 0.02
GAP = 0.01

# Every static vertex whose index is small enough to pass for a dynamic one has
# to be far enough from the sheet that no primitive carrying it can be the pair
# the assembly reports. The static grid's cell is about 0.29 across and the
# reported pair is separated by less than OFFSET, so this covers both with room.
SEPARATION = 0.5

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


app = App.create("rig_collider_coincident_pair")

# `mesh.square` returns 5 columns: xyz then uv, so only [:, :3] is a position.
# The sheet is deliberately tiny: 4 vertices hold the dynamic index space to
# 0..3, and its two triangles share an edge, so the build-time dynamic proximity
# scan (which skips vertex-sharing pairs) has no pair of its own to measure even
# with a contact offset this large.
sheet_V, sheet_F = app.mesh.square(res=2, size=0.2, ex=[1, 0, 0], ey=[0, 0, 1])
app.asset.add.tri("sheet", sheet_V, sheet_F)

# The collider is a wide 8x8 grid. The sheet sits over its middle, so every
# static vertex the assembly can reach carries a high index while the low static
# indices stay out at the rim. Subtest B checks that against the arrays rather
# than trusting the generator's ordering.
floor_V, floor_F = app.mesh.square(res=8, size=2.0, ex=[1, 0, 0], ey=[0, 0, 1])
app.asset.add.tri("floor", floor_V, floor_F)

n_dyn_vert = int(np.asarray(sheet_V).shape[0])
n_static_vert = int(np.asarray(floor_V).shape[0])

scene = app.scene.create()
sheet = scene.add("sheet").at(0.0, GAP, 0.0)
sheet.param.set("contact-offset", OFFSET)
# Pinning every vertex with no operation, pull or unpin time is what classifies
# an object static, and a static object is routed to the collision-mesh pool the
# three paths under test read. Nothing else in the frontend addresses that pool.
scene.add("floor").at(0.0, 0.0, 0.0).pin()

sheet_world = (
    np.asarray(sheet_V, dtype=np.float64)[:, :3] + np.array([0.0, GAP, 0.0])
)
floor_world = np.asarray(floor_V, dtype=np.float64)[:, :3]
nearest_low_static = float(
    np.min(
        np.linalg.norm(
            floor_world[:n_dyn_vert, None, :] - sheet_world[None, :, :], axis=2
        )
    )
)
record("B_index_spaces_are_separable", nearest_low_static > SEPARATION,
       {"n_dyn_vert": n_dyn_vert, "n_static_vert": n_static_vert,
        "nearest_low_static_to_sheet": nearest_low_static,
        "required": SEPARATION})

fixed = None
build_error = ""
try:
    fixed = scene.build(quiet=True)
except Exception as error:
    build_error = "{}: {}".format(
        type(error).__name__, str(error).splitlines()[0][:200])

record("A_scene_is_admitted_without_a_clearance_gate", fixed is not None,
       {"build_error": build_error, "offset": OFFSET, "gap": GAP})

outcome = {}
sub_kind = ""
detail = ""
log_text = ""
notes = {}

if fixed is not None:
    session = app.session.create(fixed)
    # Two frames is more than the run needs: the pair is inside the offset at
    # t=0, so the very first advance is the one that has to report. The second
    # frame exists so a run that DOES advance is visibly different from one that
    # reports on entry.
    session.param.set("dt", 0.01).set("frames", 2)
    session = session.build()
    try:
        session.start(blocking=True)
    except Exception as error:
        # The advance is EXPECTED to fail, so the exception carries no verdict.
        # What the run RECORDED is the assertion, and that is read below.
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
        # trap, so the logs are the only evidence left of what stopped it.
        notes["status_record"] = "absent at " + status_path
        notes["error_log"] = tail(os.path.join(session.info.path, "error.log"))

    stdout_path = os.path.join(session.info.path, "stdout.log")
    try:
        with open(stdout_path, "r", errors="replace") as handle:
            log_text = handle.read()
    except OSError:
        notes["stdout_log"] = "absent at " + stdout_path

c_details = {"kind": str(outcome.get("kind", "")), "sub_kind": sub_kind}
c_details.update(notes)
record("C_advance_reports_overlapping_start",
       sub_kind == "overlapping_start", c_details)

# Spaces are stripped so the check reads the token itself and not the wrapping
# of the host's format string.
record("D_cause_is_contact_separated",
       "contact_separated=false" in detail.replace(" ", ""),
       {"detail": detail[:300]})

# The three strings `main.cu` maps kinds 7, 8 and 9 to. Any of them is correct
# here: which collision-mesh branch wins the first-writer CAS depends on the
# order the assembly's kernels retire, and all three are collision-mesh paths.
# "contact assembly" (kind 6) is the shared embed and is the wrong answer for a
# pair one side of which is in the collision mesh.
ASSEMBLY_KINDS = (
    "vertex-face (collision mesh, assembly)",
    "face-vertex (collision mesh, assembly)",
    "edge-edge (collision mesh, assembly)",
)
report_lines = [
    line for line in log_text.splitlines()
    if "contact starts overlapping" in line
]
report = report_lines[0] if report_lines else ""
named = [kind for kind in ASSEMBLY_KINDS if kind in report]
record("E_kind_is_a_collision_mesh_assembly_path", len(named) == 1,
       {"named": named, "n_report_lines": len(report_lines),
        "report": report[:400]})

# The host prints the pair as "offending pair: vertices %u and %u".
pair = re.search(r"vertices (\d+) and (\d+)", report)
first = int(pair.group(1)) if pair else -1
second = int(pair.group(2)) if pair else -1
record("F_reported_pair_lands_in_its_own_index_space",
       pair is not None
       and 0 <= first < n_dyn_vert
       and n_dyn_vert <= second < n_static_vert,
       {"first": first, "second": second, "n_dyn_vert": n_dyn_vert,
        "n_static_vert": n_static_vert, "report": report[:400]})

SPACE_NOTE = "second index is in the static collision-mesh vertex space"
record("G_second_index_is_named_as_collision_mesh_space",
       SPACE_NOTE in log_text,
       {"note": SPACE_NOTE, "log_bytes": len(log_text)})

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
    return r.report_named_checks(cases, label="collider-pair cases")
