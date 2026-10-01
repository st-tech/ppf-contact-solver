# File: scenarios/rig_sand_rod_point_edge.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: SAND grains dropped onto the INTERIOR of a
# rod segment come to rest on it, and the run finishes (public issue #154).
#
# A grain is a vertex with no edge and a rod edge has no face, which leaves the
# line search's self-contact sweeps with one way to see the pair:
#
#   point-face   nothing, the rod has no face
#   edge-edge    nothing, the grain owns no edge
#   point-point  the rod's end points only, which lie far from every landing
#   point-edge   the grain against the rod edge's interior
#
# Contact assembly forms the point-edge pair whether or not the line search
# bounded it, so a step the sweeps let through carries a grain into the rod's
# contact offset, and the next assembly stops the run with "contact starts
# overlapping". That is the failure the issue reported, and the one this scene
# reproduces in its first second without the point-edge sweep.
#
# The rod has three vertices with only its two ends pinned, so it stays a
# DYNAMIC rod and its edges go through the self-contact sweeps; pinning all of
# it would make it a static collider, which other passes handle.
#
# A is the control: the same grains with no rod fall past the rod's height,
# which is what makes D evidence that the rod caught them rather than that the
# run ended before they arrived.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. control_grains_fall_past_the_rod
#         with no rod every grain ends well below the rod's height.
#   B. premises_leave_only_the_point_edge_sweep
#         every grain lands at least LANDING_CLEARANCE from every rod vertex,
#         and the grains start further apart than two grain radii.
#   C. run_with_the_rod_finishes
#         `status.cbor` records `kind == "finished"`.
#   D. grains_rest_on_the_rod
#         every grain ends above the rod, within reach of it, and outside the
#         pair's contact offset.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the CCD sweep in the neutral
# kernels, which every backend renders from the same body.
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

GRAIN_RADIUS = 0.01
ROD_OFFSET = 0.001
GAP = 0.001
ROD_HALF = 0.3
DROP_HEIGHT = 0.6
GRAIN_X = [-0.22, -0.15, -0.08, 0.08, 0.15, 0.22]
LANDING_CLEARANCE = 0.05
FRAMES = 40
FPS = 60.0
DT = 0.002
GRAVITY = 9.8

OFFSET = GRAIN_RADIUS + ROD_OFFSET
# A grain resting on the rod sits at the offset plus at most the gap; this is
# generous on top of that, and far below the free fall the control reaches.
REST_REACH = OFFSET + 10.0 * GAP
# Float32 output of a separation that is held at the offset plus a gap.
SEPARATION_TOLERANCE = 1.0e-4

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


rod_V = np.array(
    [[-ROD_HALF, 0.0, 0.0], [0.0, 0.0, 0.0], [ROD_HALF, 0.0, 0.0]],
    dtype=np.float64,
)
rod_E = np.array([[0, 1], [1, 2]], dtype=np.uint32)
grains = np.array([[x, DROP_HEIGHT, 0.0] for x in GRAIN_X], dtype=np.float32)

nearest_landing = min(abs(x - v[0]) for x in GRAIN_X for v in rod_V)
nearest_pair = min(
    abs(a - b) for i, a in enumerate(GRAIN_X) for b in GRAIN_X[i + 1:])
record("B_premises_leave_only_the_point_edge_sweep",
       nearest_landing >= LANDING_CLEARANCE
       and nearest_pair > 2.0 * GRAIN_RADIUS,
       {"nearest_landing_to_rod_vertex": nearest_landing,
        "landing_clearance": LANDING_CLEARANCE,
        "nearest_grain_pair": nearest_pair,
        "grain_radius": GRAIN_RADIUS})


def simulate(name, with_rod):
    app = App.create(name)
    if with_rod:
        app.asset.add.rod("rod", rod_V, rod_E)
    app.asset.add.points("grains", grains)
    scene = app.scene.create()
    if with_rod:
        rod = scene.add("rod")
        rod.param.set("contact-offset", ROD_OFFSET).set("contact-gap", GAP)
        rod.pin([0, 2])
    g = scene.add("grains")
    g.param.set("contact-offset", GRAIN_RADIUS)
    g.param.set("sand-particle-mass", 1e-3)
    g.param.set("contact-gap", GAP)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("dt", DT).set("fps", FPS).set("frames", FRAMES).set(
        "gravity", [0.0, -GRAVITY, 0.0]).set("precond", "block-jacobi")
    session = session.build()
    notes = {}
    try:
        session.start(blocking=True)
    except Exception as error:
        notes["start_raised"] = "{}: {}".format(
            type(error).__name__, str(error))[:600]

    kind = ""
    status_path = os.path.join(session.info.path, "output", "status.cbor")
    if os.path.isfile(status_path):
        with open(status_path, "rb") as handle:
            status_record = cbor2.load(handle)
        outcome = (status_record.get("payload") or {}).get("outcome") or {}
        kind = str(outcome.get("kind", ""))
    else:
        notes["status_record"] = "absent at " + status_path

    path = os.path.join(
        session.info.path, "output", "vert_{}.bin".format(FRAMES))
    last = None
    if os.path.isfile(path):
        last = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
    grain_index = np.array(fixed._map_by_name["grains"], dtype=int)
    rod_index = (np.array(fixed._map_by_name["rod"], dtype=int)
                 if with_rod else None)
    return {"kind": kind, "notes": notes, "last": last,
            "grain_index": grain_index, "rod_index": rod_index}


def point_segment(p, a, b):
    e = b - a
    s = float(np.clip(np.dot(p - a, e) / np.dot(e, e), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * e))), a + s * e


control = simulate("rig_sand_rod_point_edge_control", False)
caught = simulate("rig_sand_rod_point_edge", True)

control_last = control["last"]
record("A_control_grains_fall_past_the_rod",
       control_last is not None
       and float(control_last[control["grain_index"], 1].max()) < -0.5,
       {"highest_grain": None if control_last is None
        else float(control_last[control["grain_index"], 1].max()),
        "kind": control["kind"], "notes": control["notes"]})

record("C_run_with_the_rod_finishes", caught["kind"] == "finished",
       {"kind": caught["kind"], "notes": caught["notes"]})

last = caught["last"]
per_grain = []
rests = last is not None
if last is not None:
    rod_now = last[caught["rod_index"]].astype(np.float64)
    for gi in caught["grain_index"]:
        p = last[gi].astype(np.float64)
        best = None
        for a, b in ((rod_now[0], rod_now[1]), (rod_now[1], rod_now[2])):
            d, foot = point_segment(p, a, b)
            if best is None or d < best[0]:
                best = (d, foot)
        distance, foot = best
        above = bool(p[1] > foot[1])
        ok = (above and distance < REST_REACH
              and distance > OFFSET - SEPARATION_TOLERANCE)
        rests = rests and ok
        per_grain.append({"grain": int(gi), "distance": distance,
                          "above": above, "ok": ok})
record("D_grains_rest_on_the_rod", rests,
       {"offset": OFFSET, "rest_reach": REST_REACH, "grains": per_grain})

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
        # Two real solves, each paying a solver startup before its frames.
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
    return r.report_named_checks(cases, label="sand-rod point-edge cases")
