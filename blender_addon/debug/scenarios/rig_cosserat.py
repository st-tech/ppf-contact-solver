# File: scenarios/rig_cosserat.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cosserat rods, asserted end to end on the real backend: their bending has the
# rigidity its parameter states at any segment length, their rest curvature has
# a sign a plain rod's lacks, and their twist degree of freedom can store and
# release energy no position-only rod has.
#
# NO BLENDER. The scenes are authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# Subtests:
#   A. runs_finish
#         every run records `kind == "finished"`.
#   B. cantilever_matches_the_elastica
#         a Cosserat rod clamped by fixed pins over whole edges droops under its
#         weight as the elastica says for EI = cosserat-bend * density *
#         (1 cm)^2, to CANTILEVER_TOLERANCE, at two segment lengths. The
#         elastica is taken over the EFFECTIVE overhang, the free length plus
#         half a segment: the bend site at the clamp's last node integrates
#         half a segment inside the clamp, which is the discretization's own
#         O(segment) offset and not a defect.
#   C. one_value_is_one_rod
#         the two segment lengths' errors agree to RESOLUTION_AGREEMENT. A
#         rigidity that scaled with the node's mass alone would halve with the
#         segment length and fail this by tens of percent.
#   D. kink_returns_to_its_own_side
#         a rod with one rest kink, its two ends fixed, is pushed straight by a
#         pin on its apex and released: the Cosserat rod's apex returns to the
#         side it was bent to, within KINK_TOLERANCE of its rest height.
#   E. plain_kink_has_no_side
#         the same scene with a plain rod: its rest bend is a turning angle
#         without a sign, so straight is a balanced state and the apex stays on
#         the chord. This is the control that makes D a test of the sign.
#   F. held_twist_buckles_out_of_plane
#         a slender rod with a supercritical rest twist, its far end brought a
#         fifth of its length toward the near one: with both end frames held
#         the twist converts to writhe, and the rod's non-planarity (the RMS
#         distance of its nodes from their best-fit plane) grows past
#         PERVERSION_GROWTH times its seed's.
#   G. free_ends_buckle_flat
#         the same rod with free end frames: its twist unwinds, it meets the
#         end shortening as a flat Euler arc, and its non-planarity ends below
#         FLAT_FRACTION of the held rod's. This is the control that makes F a
#         test of the twist degree of freedom rather than of buckling.
#         The rod is SLENDER, its stretch and shear stiffness times a segment
#         squared well above its bending rigidity: with both ends held at the
#         arc length's distance an inextensible rod has no way out of line, and
#         a rod softer than its bending at the segment scale can shed a whole
#         turn of twist through one edge collapsing and its frame flipping.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the neutral Cosserat kernels and
# the augmented solve, which every backend renders from the same bodies.
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

GRAVITY = 9.8
DENSITY = 1e-3              # kg/m
EI = 1.8e-5                 # N m^2
EA = 2.0                    # N
GA = 1.0                    # N: cosserat-shear = 2 GA / density
REF = 1e-4                  # (1 cm)^2, the bend parameter's reference area
LENGTH = 0.24               # m: 36 and 72 segments are 1/150 m and 1/300 m
CLAMP = 0.03                # m held at the root
CANTILEVER_TOLERANCE = 0.01  # relative, of the tip drop; the shear deflection q L^2 / (2 GA) is 0.2 %
RESOLUTION_AGREEMENT = 0.005
KINK_TOLERANCE = 0.1        # of the rest height
PERVERSION_GROWTH = 10.0
FLAT_FRACTION = 0.1

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def outcome(session):
    path = os.path.join(session.info.path, "output", "status.cbor")
    if not os.path.isfile(path):
        return "absent"
    with open(path, "rb") as handle:
        status = cbor2.load(handle)
    return str(((status.get("payload") or {}).get("outcome") or {}).get("kind", ""))


def run(session, frames, name, fixed):
    notes = {}
    try:
        session.start(blocking=True)
    except Exception as error:
        notes["start_raised"] = "{}: {}".format(type(error).__name__, str(error))[:600]
    path = os.path.join(session.info.path, "output", "vert_{}.bin".format(frames))
    P = None
    if os.path.isfile(path):
        P = np.fromfile(path, dtype=np.float32).reshape(-1, 3)[np.array(fixed._map_by_name[name], dtype=int)]
    return outcome(session), notes, P


def elastica_drop(alpha):
    """The tip drop over the length of a uniformly loaded cantilever, alpha =
    q L^3 / EI, by shooting on theta'(0) with RK4."""
    def shoot(m, n=1000):
        h = 1.0 / n
        y = np.array([0.0, m, 0.0])
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


def cantilever(segments, frames):
    x = np.linspace(0.0, LENGTH, segments + 1)
    segment = LENGTH / segments
    V = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    E = np.array([[a, a + 1] for a in range(len(V) - 1)], dtype=np.uint32)
    held = np.nonzero(x < CLAMP - 1e-9)[0]
    app = App.create("rig_cosserat_cantilever_{}".format(segments))
    app.asset.add.rod("rod", V, E)
    scene = app.scene.create()
    rod = scene.add("rod").as_cosserat()
    (rod.param.set("density", DENSITY).set("young-mod", EA / (DENSITY * segment))
        .set("cosserat-shear", 2.0 * GA / DENSITY).set("cosserat-bend", EI / (DENSITY * REF))
        .set("cosserat-twist", EI / (DENSITY * REF))
        .set("contact-offset", 1e-4).set("contact-gap", 1e-4))
    rod.pin(held.tolist())
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    (session.param.set("dt", 1e-2).set("fps", 60.0).set("frames", frames)
        .set("gravity", [0.0, -GRAVITY, 0.0])
        .set("isotropic-air-friction", 0.2 * DENSITY * segment))
    session = session.build()
    kind, notes, P = run(session, frames, "rod", fixed)
    overhang = LENGTH - x[held[-1]] + 0.5 * segment
    expected = elastica_drop(DENSITY * GRAVITY * overhang ** 3 / EI) * overhang
    drop = None if P is None else float(-P[-1, 1])
    error = None if drop is None else (drop - expected) / expected
    return {"kind": kind, "notes": notes, "drop": drop, "elastica": expected,
            "effective_overhang": overhang, "relative_error": error}


def kink(cosserat, a=0.05, h=0.01, frames=90):
    V = np.array([[-a, 0.0, 0.0], [0.0, h, 0.0], [a, 0.0, 0.0]])
    E = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    segment = float(np.hypot(a, h))
    app = App.create("rig_cosserat_kink_{}".format("cosserat" if cosserat else "plain"))
    app.asset.add.rod("rod", V, E)
    scene = app.scene.create()
    rod = scene.add("rod")
    if cosserat:
        rod.as_cosserat()
        (rod.param.set("cosserat-shear", 2.0 * GA / DENSITY).set("cosserat-bend", EI / (DENSITY * REF))
            .set("cosserat-twist", EI / (DENSITY * REF)))
    else:
        rod.param.set("bend", EI / (DENSITY * REF)).set("bend-rest-from-geometry", 1.0)
    (rod.param.set("density", DENSITY).set("young-mod", EA / (DENSITY * segment))
        .set("contact-offset", 1e-4).set("contact-gap", 1e-4))
    rod.pin([0, 2])
    rod.pin([1]).move_by([0.0, -h, 0.0], 0.0, 0.25).unpin(0.5)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    (session.param.set("dt", 1e-2).set("fps", 60.0).set("frames", frames)
        .set("gravity", [0.0, 0.0, 0.0])
        .set("isotropic-air-friction", 0.5 * DENSITY * segment))
    session = session.build()
    kind, notes, P = run(session, frames, "rod", fixed)
    return {"kind": kind, "notes": notes, "apex": None if P is None else float(P[1, 1]), "rest": h}


def perversion(fix_ends, frames=80, slack=0.2):
    n, length, seed = 40, 1.5, 5e-3
    s = np.linspace(0.0, 1.0, n)
    V = np.stack([s * length, seed * np.sin(np.pi * s), 0.4 * seed * np.sin(2.0 * np.pi * s)], axis=1)
    E = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32)
    segment = float(np.linalg.norm(V[1] - V[0]))
    app = App.create("rig_cosserat_perversion_{}".format("held" if fix_ends else "free"))
    app.asset.add.rod("coil", V, E)
    scene = app.scene.create()
    rod = scene.add("coil").as_cosserat()
    # EA seg^2 and GA seg^2 at about 15 and 19 times EI: the slender rod the
    # header names.
    (rod.param.set("cosserat-bend", 100 * segment / REF).set("cosserat-twist", 100 * segment / REF)
        .set("cosserat-rest-twist", 10.0).set("cosserat-fix-ends", 1.0 if fix_ends else 0.0)
        .set("young-mod", 1e6).set("cosserat-shear", 1e5))
    rod.pin(rod.grab([-1, 0, 0]))
    rod.pin(rod.grab([1, 0, 0])).move_by([-slack * length, 0.0, 0.0], 0.0, 0.3)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    (session.param.set("dt", 0.01).set("min-newton-steps", 4)
        .set("gravity", [0.0, 0.0, 0.0]).set("frames", frames))
    session = session.build()
    kind, notes, P = run(session, frames, "coil", fixed)

    def nonplanarity(v):
        c = v - v.mean(axis=0)
        return float(np.linalg.svd(c, compute_uv=False)[-1] / np.sqrt(len(v)))

    start = nonplanarity(V)
    end = None if P is None else nonplanarity(P)
    return {"kind": kind, "notes": notes, "nonplanarity_start": start, "nonplanarity_end": end,
            "growth": None if end is None else end / start}


coarse = cantilever(36, 200)
fine = cantilever(72, 300)
kinked = kink(True)
plain = kink(False)
held = perversion(True)
free = perversion(False)

runs = {"coarse": coarse, "fine": fine, "kink": kinked, "plain": plain,
        "perversion_held": held, "perversion_free": free}
record("A_runs_finish", all(run["kind"] == "finished" for run in runs.values()),
       {name: {"kind": run["kind"], "notes": run["notes"]} for name, run in runs.items()})
errors = [run["relative_error"] for run in (coarse, fine)]
record("B_cantilever_matches_the_elastica",
       all(e is not None and abs(e) < CANTILEVER_TOLERANCE for e in errors),
       {"coarse": coarse, "fine": fine, "tolerance": CANTILEVER_TOLERANCE})
record("C_one_value_is_one_rod",
       None not in errors and abs(errors[0] - errors[1]) < RESOLUTION_AGREEMENT,
       {"errors": errors, "tolerance": RESOLUTION_AGREEMENT})
record("D_kink_returns_to_its_own_side",
       kinked["apex"] is not None and abs(kinked["apex"] - kinked["rest"]) < KINK_TOLERANCE * kinked["rest"],
       {"apex": kinked["apex"], "rest": kinked["rest"], "tolerance": KINK_TOLERANCE})
record("E_plain_kink_has_no_side",
       plain["apex"] is not None and abs(plain["apex"]) < KINK_TOLERANCE * plain["rest"],
       {"apex": plain["apex"], "rest": plain["rest"], "tolerance": KINK_TOLERANCE})
record("F_held_twist_buckles_out_of_plane",
       held["growth"] is not None and held["growth"] > PERVERSION_GROWTH,
       {"held": held, "threshold": PERVERSION_GROWTH})
record("G_free_ends_buckle_flat",
       None not in (held["nonplanarity_end"], free["nonplanarity_end"])
       and free["nonplanarity_end"] < FLAT_FRACTION * held["nonplanarity_end"],
       {"held": held, "free": free, "fraction": FLAT_FRACTION})

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
        # Six real solves, the two cantilevers the longest.
        timeout=max(ctx.timeout, 1800.0),
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
    return r.report_named_checks(cases, label="Cosserat rod cases")
