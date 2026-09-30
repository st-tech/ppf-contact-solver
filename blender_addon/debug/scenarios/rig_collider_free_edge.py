# File: scenarios/rig_collider_free_edge.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: a cloth dropped astride the FREE EDGE of a
# static collision mesh is caught by that edge and hangs on it.
#
# The collider is an open sheet standing on edge, so its top is a boundary edge
# with nothing above it. The cloth falls onto that edge with vertices on both
# sides of the collider's plane, further from the plane than the contact gap,
# and the edge's two end points lie well outside the cloth's footprint. That
# leaves the three collision-mesh passes with exactly one pair kind to find:
#
#   dynamic vertex  against collider face    nothing, no vertex reaches a face
#   collider vertex against dynamic face     nothing, no collider vertex is
#                                            under the cloth
#   dynamic edge    against collider edge    the cloth edges crossing the rim
#
# So the edge-edge pass alone holds the cloth, and the edges it has to act on
# are SHELL edges, which carry no mass of their own: the build puts a shell's
# inertia on its vertices and leaves `EdgeProp::mass` at zero for every edge
# that is not a rod's. A pass that reads that field to decide whether an edge
# can move skips every one of them. The cloth then falls through the collider
# on whole steps, and `check_intersection` stays silent, because its
# collision-mesh scan asks the same question of the same edge. Nothing in the
# run reports a failure, so the verdict here is read off the OUTPUT.
#
# A is the control: the same cloth with no collider reaches free fall, which is
# what makes B evidence that the rim held it rather than that the run was too
# short for it to leave.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. control_reaches_free_fall
#         with no collider the cloth's highest vertex ends below half the free
#         fall distance.
#   B. premises_isolate_the_edge_edge_pass
#         no cloth vertex starts within twice the contact gap of the collider's
#         plane, and the rim's end points are outside the cloth's footprint.
#   C. collider_left_the_solved_namespace
#         the output carries the cloth's vertices alone, so the blade is a
#         static collision mesh and not a pinned shell caught by self-contact.
#   D. run_with_the_collider_finishes
#         `status.cbor` records `kind == "finished"`.
#   E. rim_holds_the_cloth
#         the cloth's lowest vertex ends above the depth a cloth hanging on the
#         rim can reach.
#   F. cloth_is_still_astride_the_rim
#         vertices remain on both sides of the collider's plane.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the collision-mesh sweep in the
# neutral kernels, which every backend renders from the same body.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import sys

import cbor2
import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

SHEET_HALF = 0.15
# EVEN, so that no column of cloth vertices lies in the blade's plane.
RESOLUTION = 4
START_HEIGHT = 0.05
BLADE_HALF_LENGTH = 1.0
BLADE_DEPTH = 1.0
FRAMES = 30
FPS = 60.0
GRAVITY = -9.8
DT = 0.5 / FPS
CONTACT_GAP = 0.01

ELAPSED = FRAMES / FPS
FREE_FALL = 0.5 * abs(GRAVITY) * ELAPSED * ELAPSED
# A cloth hanging on the rim cannot reach lower than its own half width below
# it, plus what it stretches. Twice the half width is a quarter of the free
# fall and well below anything a caught cloth does.
HANG_LIMIT = 2.0 * SHEET_HALF

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def grid(resolution, half, height):
    axis = np.linspace(-half, half, resolution)
    points = np.array(
        [[x, height, z] for x in axis for z in axis], dtype=np.float64)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            c = a + resolution
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return points, np.array(faces, dtype=np.uint32)


sheet_V, sheet_F = grid(RESOLUTION, SHEET_HALF, START_HEIGHT)
blade_V = np.array(
    [
        [0.0, 0.0, -BLADE_HALF_LENGTH],
        [0.0, 0.0, BLADE_HALF_LENGTH],
        [0.0, -BLADE_DEPTH, -BLADE_HALF_LENGTH],
        [0.0, -BLADE_DEPTH, BLADE_HALF_LENGTH],
    ],
    dtype=np.float64,
)
blade_F = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.uint32)
n_sheet = int(sheet_V.shape[0])

nearest_to_plane = float(np.abs(sheet_V[:, 0]).min())
record("B_premises_isolate_the_edge_edge_pass",
       nearest_to_plane > 2.0 * CONTACT_GAP
       and BLADE_HALF_LENGTH > 3.0 * SHEET_HALF,
       {"nearest_cloth_vertex_to_plane": nearest_to_plane,
        "contact_gap": CONTACT_GAP,
        "rim_half_length": BLADE_HALF_LENGTH,
        "cloth_half_width": SHEET_HALF})


def simulate(name, with_blade):
    app = App.create(name)
    app.asset.add.tri("sheet", sheet_V, sheet_F)
    if with_blade:
        app.asset.add.tri("blade", blade_V, blade_F)
    scene = app.scene.create()
    sheet = scene.add("sheet")
    sheet.param.set("young-mod", 2000.0).set("bend", 20.0).set(
        "contact-gap", CONTACT_GAP)
    if with_blade:
        # Pinning every vertex with no operation is what classifies an object
        # static, and a static object is routed to the collision-mesh pool.
        blade = scene.add("blade")
        blade.param.set("contact-gap", CONTACT_GAP)
        blade.pin()
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("dt", DT).set("fps", FPS).set(
        "gravity", [0.0, GRAVITY, 0.0]).set(
        "precond", "block-jacobi").set("frames", FRAMES)
    session = session.build()
    notes = {}
    try:
        session.start(blocking=True)
    except Exception as error:
        notes["start_raised"] = "{}: {}".format(
            type(error).__name__, str(error))[:300]

    kind = ""
    status_path = os.path.join(session.info.path, "output", "status.cbor")
    if os.path.isfile(status_path):
        with open(status_path, "rb") as handle:
            status_record = cbor2.load(handle)
        outcome = (status_record.get("payload") or {}).get("outcome") or {}
        kind = str(outcome.get("kind", ""))
    else:
        notes["status_record"] = "absent at " + status_path

    def read_frame(index):
        path = os.path.join(
            session.info.path, "output", "vert_{}.bin".format(index))
        if not os.path.isfile(path):
            return None
        return np.fromfile(path, dtype=np.float32).reshape(-1, 3)

    return {"kind": kind, "notes": notes,
            "rest": read_frame(0), "last": read_frame(FRAMES)}


control = simulate("rig_collider_free_edge_control", False)
held = simulate("rig_collider_free_edge", True)

control_last = control["last"]
record("A_control_reaches_free_fall",
       control_last is not None
       and float(control_last[:, 1].max()) < -0.5 * FREE_FALL,
       {"highest": None if control_last is None
        else float(control_last[:, 1].max()),
        "free_fall": FREE_FALL, "kind": control["kind"],
        "notes": control["notes"]})

rest = held["rest"]
record("C_collider_left_the_solved_namespace",
       rest is not None and int(rest.shape[0]) == n_sheet,
       {"output_vertices": None if rest is None else int(rest.shape[0]),
        "cloth_vertices": n_sheet})

record("D_run_with_the_collider_finishes", held["kind"] == "finished",
       {"kind": held["kind"], "notes": held["notes"]})

last = held["last"]
lowest = None if last is None else float(last[:, 1].min())
record("E_rim_holds_the_cloth",
       lowest is not None and lowest > -HANG_LIMIT,
       {"lowest": lowest, "limit": -HANG_LIMIT, "free_fall": FREE_FALL})

left = 0 if last is None else int((last[:, 0] < 0.0).sum())
right = 0 if last is None else int((last[:, 0] > 0.0).sum())
record("F_cloth_is_still_astride_the_rim", left > 0 and right > 0,
       {"on_one_side": left, "on_the_other": right})

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
        # Two real solves, each paying a device context and a solver startup
        # before its thirty frames.
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
    return r.report_named_checks(cases, label="free-edge cases")
