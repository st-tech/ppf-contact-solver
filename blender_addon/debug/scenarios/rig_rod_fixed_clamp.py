# File: scenarios/rig_rod_fixed_clamp.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One contract, asserted end to end: a rod clamped by FIXED pins holds the
# clamp's moment, so a cantilever of it droops as the elastica says and not as a
# hinge lets it.
#
# A fixed pin is a Dirichlet condition: its vertex leaves the Newton system and
# the Dirichlet pass lifts its couplings into the free rows. The rod bending
# term is a three-vertex stencil about an interior vertex, and its energy
# depends on the two neighbors' positions whether or not the interior vertex is
# held. A site skipped because its INTERIOR vertex is fixed drops the bending
# between the clamp's last segment and the first free one, so the rod turns
# freely about the clamp's last vertex and hangs straight down whatever its
# stiffness. A pull pin is a spring and leaves the vertex in the system, so a
# clamp of pull pins never had the defect: it is the control.
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. runs_finish
#         both runs record `kind == "finished"`.
#   B. pull_clamp_matches_the_elastica
#         the control: the pull-pin clamp's tip drop is the elastica's for the
#         rod's EI, weight and overhang, to DISCRETE_TOLERANCE.
#   C. fixed_clamp_matches_the_elastica
#         the fixed-pin clamp's tip drop is the same, to the same tolerance.
#   D. fixed_clamp_is_no_hinge
#         the fixed-pin clamp's tip slope stays below HINGE_SLOPE degrees; a
#         hinge leaves the rod hanging at about 90.
#   E. clamps_agree
#         the two clamps' tip drops agree to CLAMP_AGREEMENT, which neither the
#         discretization nor the elastica's idealization enters.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the rod bending site test in
# the neutral kernel, which every backend renders from the same body.
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

LENGTH = 0.25              # m
SEGMENT = 1.0 / 300.0      # m
CLAMP = 0.03               # m held at the root
DENSITY = 1e-3             # kg/m
EI = 1.8e-5                # N m^2: bend = EI / (density * (1 cm)^2), rod_bend_stiffness
EA = 2.0                   # N: young-mod = EA / (density * segment), rod_force
GRAVITY = 9.8
PULL = 1.0e4               # N/m, the control clamp's spring: stiff enough that it barely turns
DT = 1e-2
FPS = 60.0
FRAMES = 150
AIR = 0.2                  # times a vertex's mass: about critical damping for this rod at DT
DISCRETE_TOLERANCE = 0.03  # relative, of the tip drop, against the elastica: the discrete rod droops about 2 %
                           # more, its clamp's edge being uncertain by about half a segment
CLAMP_AGREEMENT = 0.01     # relative, of the tip drop, between the two clamps
HINGE_SLOPE = 70.0         # degrees; the elastica's is about 45

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


x = np.arange(0.0, LENGTH + 0.5 * SEGMENT, SEGMENT)
V = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
E = np.array([[a, a + 1] for a in range(len(V) - 1)], dtype=np.uint32)
held = np.nonzero(x < CLAMP - 1e-9)[0]
overhang = LENGTH - x[held[-1]]


def elastica_drop(alpha):
    """The tip drop over the length of a uniformly loaded cantilever, alpha =
    q L^3 / EI: theta'' = -alpha (1 - s) cos(theta), theta(0) = 0,
    theta'(1) = 0, shot on theta'(0) by bisection with RK4."""
    def shoot(m, n=1000):
        h = 1.0 / n
        y = np.array([0.0, m, 0.0])            # theta, theta', drop
        f = lambda s, y: np.array([y[1], -alpha * (1.0 - s) * np.cos(y[0]), np.sin(y[0])])
        s = 0.0
        for _ in range(n):
            k1 = f(s, y); k2 = f(s + h / 2, y + h / 2 * k1)
            k3 = f(s + h / 2, y + h / 2 * k2); k4 = f(s + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s += h
        return y
    lo, hi = 0.0, alpha
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if shoot(mid)[1] > 0.0:
            hi = mid
        else:
            lo = mid
    return float(shoot(0.5 * (lo + hi))[2])


alpha = DENSITY * GRAVITY * overhang ** 3 / EI
expected = elastica_drop(alpha) * overhang


def simulate(name, pull):
    app = App.create(name)
    app.asset.add.rod("rod", V, E)
    scene = app.scene.create()
    rod = scene.add("rod")
    (rod.param.set("density", DENSITY).set("young-mod", EA / (DENSITY * SEGMENT))
        .set("bend", EI / (DENSITY * 1e-4)).set("contact-offset", 1e-4).set("contact-gap", 1e-4))
    pin = rod.pin(held.tolist())
    if pull:
        pin.pull(PULL)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("dt", DT).set("fps", FPS).set("frames", FRAMES).set(
        "gravity", [0.0, -GRAVITY, 0.0]).set("isotropic-air-friction", AIR * DENSITY * SEGMENT)
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
    path = os.path.join(session.info.path, "output", "vert_{}.bin".format(FRAMES))
    drop = slope = None
    if os.path.isfile(path):
        P = np.fromfile(path, dtype=np.float32).reshape(-1, 3)[np.array(fixed._map_by_name["rod"], dtype=int)]
        drop = float(-P[-1, 1])
        slope = float(np.degrees(np.arctan2(P[-2, 1] - P[-1, 1], P[-1, 0] - P[-2, 0])))
    return {"kind": kind, "notes": notes, "drop": drop, "slope": slope}


pulled = simulate("rig_rod_fixed_clamp_pull", True)
fixed_pins = simulate("rig_rod_fixed_clamp", False)

record("A_runs_finish", pulled["kind"] == "finished" and fixed_pins["kind"] == "finished",
       {"pull": pulled, "fixed": fixed_pins})
for name, run in (("B_pull_clamp_matches_the_elastica", pulled), ("C_fixed_clamp_matches_the_elastica", fixed_pins)):
    error = None if run["drop"] is None else abs(run["drop"] - expected) / expected
    record(name, error is not None and error < DISCRETE_TOLERANCE,
           {"drop": run["drop"], "elastica": expected, "relative_error": error,
            "tolerance": DISCRETE_TOLERANCE, "alpha": alpha, "overhang": overhang})
record("D_fixed_clamp_is_no_hinge", fixed_pins["slope"] is not None and fixed_pins["slope"] < HINGE_SLOPE,
       {"slope": fixed_pins["slope"], "hinge_slope": HINGE_SLOPE})
agreement = (None if pulled["drop"] is None or fixed_pins["drop"] is None
             else abs(fixed_pins["drop"] - pulled["drop"]) / pulled["drop"])
record("E_clamps_agree", agreement is not None and agreement < CLAMP_AGREEMENT,
       {"fixed": fixed_pins["drop"], "pull": pulled["drop"], "relative_difference": agreement,
        "tolerance": CLAMP_AGREEMENT})

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
    return r.report_named_checks(cases, label="rod fixed-clamp cases")
