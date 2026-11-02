# File: scenarios/rig_stitch_rest_length.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# One contract, asserted end to end: a stitch that carries its own rest length
# holds its two points at that distance, with no preload.
#
# A stitch without one takes half the mean contact gap of its vertices as its
# rest length, which closes a seam to contact distance. Between two bodies that
# rest apart (a strand couched onto a cloth, two yarn center lines a yarn
# diameter apart) that rest length is far below the distance the stitch spans,
# so every stitch starts stretched to its length cap and pulls the bodies into
# each other. An authored rest length is the distance the stitch is meant to
# hold, and `Scene.cross_stitch(..., rest_length=...)` carries it to the solver.
#
# THE SCENE IS TWO STRAIGHT RODS, parallel and REST apart, stitched vertex to
# vertex along their whole length, with no gravity, no pin and no collider, so
# the stitches are the only force in the scene.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS, for the reason `rig_rod_fixed_clamp` states.
#
# Subtests:
#   A. runs_finish
#         every run records `kind == "finished"`.
#   B. initial_rest_length_is_measured
#         `rest_length="initial"` measures every stitch at REST, to round-off.
#   C. at_rest_no_force
#         the rods stitched at their initial distance do not move: every vertex
#         stays within AT_REST_DRIFT of where it started, over every frame.
#   D. derived_rest_length_preloads
#         the control: the same rods stitched with the gap-derived rest length
#         are pulled together, closing at least PRELOAD_CLOSE of the distance.
#         Without it, (C) could pass on a solve that assembles no stitch.
#   E. stretched_restores
#         rods started STRETCHED apart with the rest length REST authored as a
#         number move toward REST: the distance leaves the start by at least
#         MOVED, and the second half of the run, several periods of a swing
#         about the rest length, averages to REST within RESTORE_TOLERANCE of
#         the initial stretch.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the rest-length selection in the
# neutral stitch kernel, which every backend renders from the same body, and
# the frontend plumbing that feeds it.
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

LENGTH = 1.0               # m
SEGMENT = 0.05             # m
REST = 0.4                 # m, the distance the rods rest apart
STRETCHED = 0.6            # m, where the restore case starts
DENSITY = 1e-3             # kg/m
YOUNG = 1.0e4              # rod stretch, stiff enough that a rod stays its length
BEND = 1.0
CONTACT_GAP = 1e-3         # m, derives a rest length of 5e-4, far below REST
STIFFNESS = 0.1            # the stitch's raw force factor: the restore case swings with a
                           # period of about 0.75 s, several periods inside the run
DT = 1e-2
FPS = 60.0
FRAMES = 300
AT_REST_DRIFT = 1e-5       # m, solve round-off, not a motion budget
PRELOAD_CLOSE = 0.5        # fraction of REST the control must close
MOVED = 0.05               # m the restore case must move off its start
RESTORE_TOLERANCE = 0.25   # of the initial stretch, for the second-half mean

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


x = np.arange(0.0, LENGTH + 0.5 * SEGMENT, SEGMENT)
E = np.array([[a, a + 1] for a in range(len(x) - 1)], dtype=np.uint32)


def rod(y):
    return np.c_[x, np.full_like(x, y), np.zeros_like(x)]


def simulate(name, spread, rest_length):
    app = App.create(name)
    app.asset.add.rod("rod-a", rod(0.0), E)
    app.asset.add.rod("rod-b", rod(spread), E)
    scene = app.scene.create()
    for key in ("rod-a", "rod-b"):
        (scene.add(key).param.set("density", DENSITY).set("young-mod", YOUNG)
            .set("bend", BEND).set("contact-offset", 0.0).set("contact-gap", CONTACT_GAP))
    count = len(x)
    ind = [[i, i, i, i, i, i] for i in range(count)]
    w = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * count
    scene.cross_stitch("rod-a", "rod-b", ind, w, stiffness=STIFFNESS, rest_length=rest_length)
    fixed = scene.build(quiet=True)
    measured = np.asarray(fixed._stitch_rest_length, dtype=np.float64)
    given = np.asarray(fixed._stitch_has_rest_length, dtype=bool)
    session = app.session.create(fixed)
    session.param.set("dt", DT).set("fps", FPS).set("frames", FRAMES).set(
        "gravity", [0.0, 0.0, 0.0])
    session = session.build()
    notes = {}
    try:
        session.start(blocking=True)
    except Exception as error:
        notes["start_raised"] = "{}: {}".format(type(error).__name__, str(error))[:600]
    kind = ""
    status_path = os.path.join(session.info.path, "output", "status.cbor")
    if os.path.isfile(status_path):
        with open(status_path, "rb") as handle:
            status_record = cbor2.load(handle)
        outcome = (status_record.get("payload") or {}).get("outcome") or {}
        kind = str(outcome.get("kind", ""))
    else:
        notes["status_record"] = "absent at " + status_path
    a = np.array(fixed._map_by_name["rod-a"], dtype=int)
    b = np.array(fixed._map_by_name["rod-b"], dtype=int)
    frames = []
    for frame in range(FRAMES + 1):
        path = os.path.join(session.info.path, "output", "vert_{}.bin".format(frame))
        if not os.path.isfile(path):
            notes["missing_frame"] = frame
            break
        frames.append(np.fromfile(path, dtype=np.float32).reshape(-1, 3).astype(np.float64))
    distance = [float(np.mean(np.linalg.norm(P[b] - P[a], axis=1))) for P in frames]
    drift = (max(float(np.abs(P - frames[0]).max()) for P in frames) if frames else None)
    return {
        "kind": kind, "notes": notes, "frames": len(frames), "distance": distance,
        "drift": drift, "rest_length": measured.tolist(), "given": given.tolist(),
    }


at_rest = simulate("rig_stitch_rest_length_at_rest", REST, "initial")
derived = simulate("rig_stitch_rest_length_derived", REST, None)
restore = simulate("rig_stitch_rest_length_restore", STRETCHED, REST)
runs = {"at_rest": at_rest, "derived": derived, "restore": restore}

record("A_runs_finish",
       all(run["kind"] == "finished" and run["frames"] == FRAMES + 1 for run in runs.values()),
       {name: {"kind": run["kind"], "frames": run["frames"], "notes": run["notes"]}
        for name, run in runs.items()})
measured = np.asarray(at_rest["rest_length"])
record("B_initial_rest_length_is_measured",
       len(measured) == len(x) and all(at_rest["given"])
       and float(np.abs(measured - REST).max()) < 1e-6,
       {"rest_length": at_rest["rest_length"][:4], "given": at_rest["given"][:4],
        "derived_given": derived["given"][:4]})
record("C_at_rest_no_force",
       at_rest["drift"] is not None and at_rest["drift"] < AT_REST_DRIFT,
       {"drift": at_rest["drift"], "tolerance": AT_REST_DRIFT,
        "distance_first_last": at_rest["distance"][:1] + at_rest["distance"][-1:]})
closed = (None if not derived["distance"] else REST - min(derived["distance"]))
record("D_derived_rest_length_preloads",
       closed is not None and closed > PRELOAD_CLOSE * REST and not any(derived["given"]),
       {"closed": closed, "required": PRELOAD_CLOSE * REST,
        "distance_first_last": derived["distance"][:1] + derived["distance"][-1:]})
distance = restore["distance"]
moved = None if not distance else STRETCHED - min(distance)
tail = distance[len(distance) // 2:]
mean = float(np.mean(tail)) if tail else None
record("E_stretched_restores",
       moved is not None and moved > MOVED and mean is not None
       and abs(mean - REST) < RESTORE_TOLERANCE * (STRETCHED - REST),
       {"moved": moved, "second_half_mean": mean, "rest": REST,
        "tolerance": RESTORE_TOLERANCE * (STRETCHED - REST),
        "distance_every_10": distance[::10]})

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
        # Three real solves, each paying a solver startup before its frames.
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
    return r.report_named_checks(cases, label="stitch rest-length cases")
