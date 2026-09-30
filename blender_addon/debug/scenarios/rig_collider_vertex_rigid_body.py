# File: scenarios/rig_collider_vertex_rigid_body.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: a rigid body dropped flat onto the APEX of
# a static collision mesh is held by that vertex.
#
# The collider is one standing triangle, so its top is a single vertex, and the
# body is a slab wide enough that its own corners and edges stay half a slab
# away from it. That leaves the three collision-mesh passes with exactly one
# pair kind to find:
#
#   dynamic vertex  against collider face    nothing, the slab's corners are far
#   collider vertex against dynamic face     the apex under the slab's base
#   dynamic edge    against collider edge    nothing, the slab's edges are far
#
# So the collider-vertex pass alone holds the body, and the faces it has to act
# on are a PDRD rigid body's, which carry no mass of their own: a body's mass
# is volumetric and the build hands it to the vertices, zeroing the per-face
# figure. A pass that reads that field to decide whether a face can move skips
# every face of every rigid body. The spike then goes through the slab on whole
# steps, and `check_intersection` stays silent, because its collision-mesh scan
# walks dynamic edges against collider faces and the crossing here is a
# collider edge through a dynamic face. Nothing in the run reports a failure,
# so the verdict is read off the OUTPUT.
#
# `rig_collider_free_edge` is the same contract for the edge-edge pass and a
# shell, whose EDGES carry no mass of their own.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. control_reaches_free_fall
#         with no collider the slab's highest vertex ends below half the free
#         fall distance.
#   B. premises_isolate_the_collider_vertex_pass
#         the slab is more than five times as wide as the spike.
#   C. collider_left_the_solved_namespace
#         the output carries the slab's vertices alone.
#   D. run_with_the_collider_finishes
#         `status.cbor` records `kind == "finished"`.
#   E. apex_has_not_gone_through_the_body
#         the slab's highest vertex ends above the apex. The slab is rigid and
#         starts above it, so a slab the spike passed through has every vertex
#         below.
#   F. body_rests_on_the_apex
#         the slab's lowest vertex ends above the depth a slab tilted on the
#         apex can reach.

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

SLAB_HALF = 0.5
SLAB_THICKNESS = 0.05
START_HEIGHT = 0.05
SPIKE_HALF_WIDTH = 0.05
SPIKE_DEPTH = 0.5
FRAMES = 30
FPS = 60.0
GRAVITY = -9.8
DT = 0.5 / FPS
CONTACT_GAP = 0.01

ELAPSED = FRAMES / FPS
FREE_FALL = 0.5 * abs(GRAVITY) * ELAPSED * ELAPSED
# A slab resting on the apex may tilt about it, and a tilted slab's low side
# reaches at most its half width below the apex.
REST_LIMIT = SLAB_HALF

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


low, high = START_HEIGHT, START_HEIGHT + SLAB_THICKNESS
slab_V = np.array(
    [
        [-SLAB_HALF, low, -SLAB_HALF],
        [SLAB_HALF, low, -SLAB_HALF],
        [SLAB_HALF, low, SLAB_HALF],
        [-SLAB_HALF, low, SLAB_HALF],
        [-SLAB_HALF, high, -SLAB_HALF],
        [SLAB_HALF, high, -SLAB_HALF],
        [SLAB_HALF, high, SLAB_HALF],
        [-SLAB_HALF, high, SLAB_HALF],
    ],
    dtype=np.float64,
)
slab_F = np.array(
    [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ],
    dtype=np.uint32,
)
spike_V = np.array(
    [
        [0.0, 0.0, 0.0],
        [-SPIKE_HALF_WIDTH, -SPIKE_DEPTH, 0.0],
        [SPIKE_HALF_WIDTH, -SPIKE_DEPTH, 0.0],
    ],
    dtype=np.float64,
)
spike_F = np.array([[0, 1, 2]], dtype=np.uint32)
n_slab = int(slab_V.shape[0])

record("B_premises_isolate_the_collider_vertex_pass",
       SLAB_HALF > 5.0 * SPIKE_HALF_WIDTH,
       {"slab_half_width": SLAB_HALF, "spike_half_width": SPIKE_HALF_WIDTH})


def simulate(name, with_spike):
    app = App.create(name)
    app.asset.add.tri("slab", slab_V, slab_F)
    if with_spike:
        app.asset.add.tri("spike", spike_V, spike_F)
    scene = app.scene.create()
    slab = scene.add("slab").as_pdrd()
    slab.param.set("contact-gap", CONTACT_GAP)
    if with_spike:
        # Pinning every vertex with no operation is what classifies an object
        # static, and a static object is routed to the collision-mesh pool.
        spike = scene.add("spike")
        spike.param.set("contact-gap", CONTACT_GAP)
        spike.pin()
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


control = simulate("rig_collider_vertex_rigid_body_control", False)
held = simulate("rig_collider_vertex_rigid_body", True)

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
       rest is not None and int(rest.shape[0]) == n_slab,
       {"output_vertices": None if rest is None else int(rest.shape[0]),
        "slab_vertices": n_slab})

record("D_run_with_the_collider_finishes", held["kind"] == "finished",
       {"kind": held["kind"], "notes": held["notes"]})

last = held["last"]
highest = None if last is None else float(last[:, 1].max())
lowest = None if last is None else float(last[:, 1].min())
record("E_apex_has_not_gone_through_the_body",
       highest is not None and highest > 0.0,
       {"highest": highest, "apex": 0.0, "free_fall": FREE_FALL})
record("F_body_rests_on_the_apex",
       lowest is not None and lowest > -REST_LIMIT,
       {"lowest": lowest, "limit": -REST_LIMIT})

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
    return r.report_named_checks(cases, label="collider-vertex cases")
