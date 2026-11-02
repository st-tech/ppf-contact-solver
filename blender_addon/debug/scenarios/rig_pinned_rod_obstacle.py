# File: scenarios/rig_pinned_rod_obstacle.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# One contract, asserted end to end: a ROD PINNED AT EVERY VERTEX is an
# obstacle. A horizontal dowel, every vertex held by a fix pin with no pin
# operation, lies across the path of a plain rod, a Cosserat rod, a shell and a
# solid dropped onto it, and each comes to rest on it.
#
# WHAT IT GUARDS. Pinning every vertex with no operation classifies a SURFACE
# object static, which moves it to the contact-only collision mesh. That mesh
# holds triangles only, so a rod classified the same way would contribute
# nothing to it and leave the scene: whatever falls onto it passes through, the
# run finishes, and nothing reports a failure (a yarn strip hung from a dowel
# pinned whole). `Object.update_static` keeps a faceless object solved, every
# vertex held exactly by its pins, so the solver's pair filter admits every
# pair it forms with a free element. The verdict is read off the OUTPUT.
#
# Each falling kind runs against a control with no dowel, which reaches free
# fall; that is what makes the caught run evidence that the dowel held it
# rather than that the run was too short for it to leave.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS, for the reason `rig_collider_free_edge` gives.
#
# Subtests, for each falling kind K in rod, cosserat, shell, solid:
#   K_control_reaches_free_fall
#         with no dowel the object's highest vertex ends below half the free
#         fall distance.
#   K_dowel_is_in_the_solve
#         the output carries the dowel's vertices beside the object's, so the
#         dowel was not built as a (triangle-only) collision mesh.
#   K_dowel_holds_still
#         every dowel vertex ends where it started.
#   K_dowel_holds_the_object
#         the object's lowest vertex ends above the depth an object caught on
#         the dowel can reach.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. The contact it asserts is the dynamic-namespace
# pair filter in the neutral kernels, which every backend renders from one body.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

FRAMES = 30
FPS = 60.0
GRAVITY = -9.8
DT = 0.5 / FPS
CONTACT_GAP = 0.005
DOWEL_HALF_LENGTH = 0.5
DOWEL_VERTICES = 41
START_HEIGHT = 0.05
SPAN_HALF = 0.15

ELAPSED = FRAMES / FPS
FREE_FALL = 0.5 * abs(GRAVITY) * ELAPSED * ELAPSED
# An object caught on the dowel reaches no lower than its own half span below
# it plus its stretch; twice the half span is a quarter of the free fall.
HANG_LIMIT = 2.0 * SPAN_HALF

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def line(p0, p1, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    V = (1.0 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
    E = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32)
    return V, E


def grid(resolution, half, height):
    axis = np.linspace(-half, half, resolution)
    points = np.array(
        [[x, height, z] for x in axis for z in axis], dtype=np.float64)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b, c = a + 1, a + resolution
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return points, np.array(faces, dtype=np.uint32)


def simulate(kind, with_dowel):
    app = App.create("rig_pinned_rod_obstacle_{}_{}".format(
        kind, "dowel" if with_dowel else "free"))
    scene = app.scene.create()
    if kind in ("rod", "cosserat"):
        V, E = line([-SPAN_HALF, START_HEIGHT, 0.0],
                    [SPAN_HALF, START_HEIGHT, 0.0], 16)
        app.asset.add.rod("falling", V, E)
        obj = scene.add("falling")
        if kind == "cosserat":
            obj.as_cosserat()
    elif kind == "shell":
        V, F = grid(6, SPAN_HALF, START_HEIGHT)
        app.asset.add.tri("falling", V, F)
        obj = scene.add("falling")
        obj.param.set("young-mod", 2000.0).set("bend", 20.0)
    else:
        V, F, T = app.mesh.tet_box(2.0 * SPAN_HALF, 0.05, 2.0 * SPAN_HALF)
        V = np.asarray(V, dtype=np.float64)
        V[:, 1] += START_HEIGHT + 0.025 - V[:, 1].mean()
        app.asset.add.tet("falling", V, np.asarray(F), np.asarray(T))
        obj = scene.add("falling")
    obj.param.set("contact-gap", CONTACT_GAP)
    n_falling = int(len(V))
    if with_dowel:
        DV, DE = line([0.0, 0.0, -DOWEL_HALF_LENGTH],
                      [0.0, 0.0, DOWEL_HALF_LENGTH], DOWEL_VERTICES)
        app.asset.add.rod("dowel", DV, DE)
        dowel = scene.add("dowel")
        if kind == "cosserat":
            dowel.as_cosserat()
        dowel.param.set("contact-gap", CONTACT_GAP)
        dowel.pin()
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

    def read_frame(index):
        path = os.path.join(
            session.info.path, "output", "vert_{}.bin".format(index))
        if not os.path.isfile(path):
            return None
        return np.fromfile(path, dtype=np.float32).reshape(-1, 3)

    return {"n_falling": n_falling, "notes": notes,
            "rest": read_frame(0), "last": read_frame(FRAMES)}


for kind in ("rod", "cosserat", "shell", "solid"):
    control = simulate(kind, False)
    last = control["last"]
    record(kind + "_control_reaches_free_fall",
           last is not None and float(last[:, 1].max()) < -0.5 * FREE_FALL,
           {"highest": None if last is None else float(last[:, 1].max()),
            "free_fall": FREE_FALL, "notes": control["notes"]})

    held = simulate(kind, True)
    rest, last = held["rest"], held["last"]
    n_falling = held["n_falling"]
    expected = n_falling + DOWEL_VERTICES
    count = None if rest is None else int(rest.shape[0])
    record(kind + "_dowel_is_in_the_solve", count == expected,
           {"output_vertices": count, "expected": expected,
            "notes": held["notes"]})
    if rest is None or last is None or count != expected:
        record(kind + "_dowel_holds_still", False, {"reason": "no output"})
        record(kind + "_dowel_holds_the_object", False, {"reason": "no output"})
        continue
    # The output orders vertices by element kind, so the dowel is found by its
    # rest positions on the z axis.
    on_axis = np.abs(rest[:, :2]).max(axis=1) < 1.0e-6
    drift = float(np.abs(last[on_axis] - rest[on_axis]).max())
    record(kind + "_dowel_holds_still",
           int(on_axis.sum()) == DOWEL_VERTICES and drift < 1.0e-5,
           {"dowel_vertices": int(on_axis.sum()), "drift": drift})
    lowest = float(last[~on_axis][:, 1].min())
    record(kind + "_dowel_holds_the_object", lowest > -HANG_LIMIT,
           {"lowest": lowest, "limit": -HANG_LIMIT, "free_fall": FREE_FALL})

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
        # Eight short real solves, each paying a device context and a solver
        # startup before its thirty frames.
        timeout=max(ctx.timeout, 900.0),
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
    return r.report_named_checks(cases, label="pinned-rod obstacle cases")
