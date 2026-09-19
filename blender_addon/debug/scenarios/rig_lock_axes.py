# File: scenarios/rig_lock_axes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Truth table for every Lock Translation / Lock Rotation mode, at the
# AUTHORING and SCENE-BUILD gate.
#
# Five modes reach the solver, and each pins a different number of constraint
# rows into the shared projector:
#
#   translation, axis       2 rows   the center of mass stays on a line
#   translation, all axes   3 rows   the center of mass stays at a point
#   rotation, allow-only    2 rows   only rotation about the axis survives
#   rotation, prohibit-axis 1 row    rotation about the axis is forbidden
#   rotation, all axes      3 rows   there is no net rotation at all
#
# What makes this worth a scenario rather than a unit test is the WIRE. The
# mode carries the enable bit and the axis does not, which is the opposite of
# what every other axis-shaped field in this project does, so the failure it
# guards against is silent: an all-axes lock whose enablement is read off its
# (necessarily zero) axis is simply dropped, and a dropped lock produces no
# wrong number to notice. Nothing downstream can recover it either, since by
# then the record looks exactly like a disabled one. Every case here therefore
# asserts the EXPORTED BYTES, not merely that the scene built.
#
# The negative half matters as much. A gate that accepts too much is invisible
# in a happy-path test, so each accepted case is paired with a control that
# must still be REFUSED, including the two ways to spell a record
# non-canonically.
#
# `rig_lock_axes_projector` covers the same modes at the other gate, the
# solver's own live projector.
#
# The probe runs in a SUBPROCESS. It imports `frontend`, which loads the
# per-tree cdylib and installs the solver's debug patches, and the
# orchestrator imports every scenario into one long-lived process that must
# not inherit either.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# No Blender and no solver process: this is the host-side authoring and build
# check, so it holds on the real-GPU jobs too.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
WORKSPACE = sys.argv[2]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

import frontend
from frontend import App

app = App.create("rig_lock_axes")

V, F = app.mesh.square(res=4, ex=[1, 0, 0], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)
# A PDRD body needs four non-coplanar vertices for a non-singular rest-shape
# Gram, so the flat sheet above cannot serve as one. That validation is
# correct and predates this feature; the box is the fixture that satisfies it.
Vb, Fb = app.mesh.box(1.0, 1.0, 1.0)
app.asset.add.tri("brick", Vb, Fb)

TRANSLATION_LOCK_AXIS = 0
TRANSLATION_LOCK_ALL = 1
ROTATION_LOCK_ALLOW_ONLY = 0
ROTATION_LOCK_PROHIBIT_AXIS = 1
ROTATION_LOCK_ALL = 2

cases = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def export(setup, tag):
    """Build a one-object scene and return its exported lock tables.

    Reads the BYTES the solver will read rather than the Python state that
    produced them, so an authoring call that never reaches the wire fails here.
    """
    scene = app.scene.create()
    setup(scene.add("sheet").at(0.0, 0.0, 0.0))
    fixed = scene.build(quiet=True)
    path = os.path.join(WORKSPACE, "export_" + tag)
    fixed.export_fixed(path, delete_exist=True)
    binp = os.path.join(path, "bin")

    def axes(name):
        p = os.path.join(binp, name)
        if not os.path.exists(p):
            return None
        return np.fromfile(p, dtype=np.float32).reshape(-1, 3)

    def modes(name):
        p = os.path.join(binp, name)
        if not os.path.exists(p):
            return None
        return np.fromfile(p, dtype=np.uint32)

    return {
        "taxis": axes("translation_lock.bin"),
        "tmode": modes("translation_lock_mode.bin"),
        "raxis": axes("rotation_lock.bin"),
        "rmode": modes("rotation_lock_mode.bin"),
    }


def row(table, i=0):
    return None if table is None else table[i].tolist()


def refuses(setup):
    """True when the authoring call or the build rejects the scene."""
    try:
        scene = app.scene.create()
        setup(scene.add("sheet").at(0.0, 0.0, 0.0))
        scene.build(quiet=True)
    except Exception:
        return True
    return False


# --- the five modes reach the wire, each with its own mode value ----------
e = export(lambda o: o.lock_translation(1.0, 0.0, 0.0), "t_axis")
check("translation_axis_exports_axis_mode",
      row(e["taxis"]) == [1.0, 0.0, 0.0]
      and e["tmode"] is not None
      and int(e["tmode"][0]) == TRANSLATION_LOCK_AXIS,
      {"axis": row(e["taxis"]), "mode": None if e["tmode"] is None else int(e["tmode"][0])})

e = export(lambda o: o.lock_all_translations(), "t_all")
check("translation_all_exports_all_mode_and_zero_axis",
      row(e["taxis"]) == [0.0, 0.0, 0.0]
      and e["tmode"] is not None
      and int(e["tmode"][0]) == TRANSLATION_LOCK_ALL,
      {"axis": row(e["taxis"]), "mode": None if e["tmode"] is None else int(e["tmode"][0])})

e = export(lambda o: o.lock_rotation(0.0, 0.0, 1.0), "r_allow")
check("rotation_allow_only_exports_mode_zero",
      row(e["raxis"]) == [0.0, 0.0, 1.0]
      and int(e["rmode"][0]) == ROTATION_LOCK_ALLOW_ONLY,
      {"axis": row(e["raxis"]), "mode": None if e["rmode"] is None else int(e["rmode"][0])})

e = export(lambda o: o.lock_rotation(0.0, 0.0, 1.0).lock_rotation_prohibit_axis(True),
           "r_prohibit")
check("rotation_prohibit_exports_mode_one",
      row(e["raxis"]) == [0.0, 0.0, 1.0]
      and int(e["rmode"][0]) == ROTATION_LOCK_PROHIBIT_AXIS,
      {"axis": row(e["raxis"]), "mode": None if e["rmode"] is None else int(e["rmode"][0])})

e = export(lambda o: o.lock_all_rotations(), "r_all")
check("rotation_all_exports_mode_two_and_zero_axis",
      row(e["raxis"]) == [0.0, 0.0, 0.0]
      and int(e["rmode"][0]) == ROTATION_LOCK_ALL,
      {"axis": row(e["raxis"]), "mode": None if e["rmode"] is None else int(e["rmode"][0])})

# --- the two locks stay independent --------------------------------------
e = export(lambda o: o.lock_all_translations().lock_all_rotations(), "both_all")
check("both_all_axes_coexist",
      int(e["tmode"][0]) == TRANSLATION_LOCK_ALL
      and int(e["rmode"][0]) == ROTATION_LOCK_ALL
      and row(e["taxis"]) == [0.0, 0.0, 0.0]
      and row(e["raxis"]) == [0.0, 0.0, 0.0],
      {"tmode": int(e["tmode"][0]), "rmode": int(e["rmode"][0])})

e = export(lambda o: o.lock_all_translations().lock_rotation(0.0, 1.0, 0.0), "mixed")
check("all_translation_with_axis_rotation",
      int(e["tmode"][0]) == TRANSLATION_LOCK_ALL
      and row(e["taxis"]) == [0.0, 0.0, 0.0]
      and int(e["rmode"][0]) == ROTATION_LOCK_ALLOW_ONLY
      and row(e["raxis"]) == [0.0, 1.0, 0.0],
      {"tmode": int(e["tmode"][0]), "raxis": row(e["raxis"])})

# --- THE EMPTINESS RULE ---------------------------------------------------
# The lock files are written when SOME object is locked in ANY mode, not when
# some object carries an axis. An all-axes object's axis row is exactly zero,
# so an axis-shaped emptiness test writes no lock file at all for a scene
# whose only locked objects use an all-axes mode, and the solver then reads a
# completely unlocked scene. Nothing about such a scene looks wrong: it
# builds, it runs, and the object is simply not locked.
e = export(lambda o: o.lock_all_rotations(), "all_only")
check("all_axes_only_scene_still_writes_lock_bins",
      e["raxis"] is not None and e["rmode"] is not None,
      {"rotation_lock.bin": e["raxis"] is not None,
       "rotation_lock_mode.bin": e["rmode"] is not None})

# A scene with no lock at all must still write NEITHER file, or "absent means
# unlocked" stops being readable from the session directory.
e = export(lambda o: o, "unlocked")
check("unlocked_scene_writes_no_lock_bins",
      e["taxis"] is None and e["tmode"] is None
      and e["raxis"] is None and e["rmode"] is None,
      {k: v is not None for k, v in e.items()})

# --- last call wins, in both directions ----------------------------------
e = export(lambda o: o.lock_translation(1.0, 0.0, 0.0).lock_all_translations(),
           "t_axis_then_all")
check("all_after_axis_wins_and_clears_the_axis",
      int(e["tmode"][0]) == TRANSLATION_LOCK_ALL
      and row(e["taxis"]) == [0.0, 0.0, 0.0],
      {"mode": int(e["tmode"][0]), "axis": row(e["taxis"])})

e = export(lambda o: o.lock_all_translations().lock_translation(0.0, 1.0, 0.0),
           "t_all_then_axis")
check("axis_after_all_wins_and_restores_the_axis",
      int(e["tmode"][0]) == TRANSLATION_LOCK_AXIS
      and row(e["taxis"]) == [0.0, 1.0, 0.0],
      {"mode": int(e["tmode"][0]), "axis": row(e["taxis"])})

# An all-axes rotation must also drop a previously requested prohibit-axis
# mode, or the exported mode would be 1 while the artist asked for all three.
e = export(
    lambda o: o.lock_rotation(0.0, 0.0, 1.0)
               .lock_rotation_prohibit_axis(True)
               .lock_all_rotations(),
    "r_prohibit_then_all")
check("all_rotations_supersedes_prohibit_axis",
      int(e["rmode"][0]) == ROTATION_LOCK_ALL
      and row(e["raxis"]) == [0.0, 0.0, 0.0],
      {"mode": int(e["rmode"][0]), "axis": row(e["raxis"])})

# --- controls that must still be refused ---------------------------------
check("zero_translation_axis_refused",
      refuses(lambda o: o.lock_translation(0.0, 0.0, 0.0)), {})
check("zero_rotation_axis_refused",
      refuses(lambda o: o.lock_rotation(0.0, 0.0, 0.0)), {})
check("non_finite_axis_refused",
      refuses(lambda o: o.lock_translation(float("nan"), 0.0, 0.0)), {})
# prohibit-axis is a MODE on an axis, so it is meaningless without one. This
# stays refused after the all-axes mode exists: "all" is a different request.
check("prohibit_axis_without_axis_refused",
      refuses(lambda o: o.lock_rotation_prohibit_axis(True)), {})

# --- PDRD: the same modes on a rigid body --------------------------------
def pdrd(setup, tag):
    scene = app.scene.create()
    obj = scene.add("brick").at(0.0, 0.0, 0.0).as_pdrd()
    setup(obj)
    fixed = scene.build(quiet=True)
    path = os.path.join(WORKSPACE, "pdrd_" + tag)
    fixed.export_fixed(path, delete_exist=True)
    binp = os.path.join(path, "bin")
    tm = os.path.join(binp, "translation_lock_mode.bin")
    rm = os.path.join(binp, "rotation_lock_mode.bin")
    return (
        np.fromfile(tm, dtype=np.uint32) if os.path.exists(tm) else None,
        np.fromfile(rm, dtype=np.uint32) if os.path.exists(rm) else None,
    )


try:
    tm, rm = pdrd(lambda o: o.lock_all_rotations(), "r_all")
    check("pdrd_all_rotations_builds",
          rm is not None and int(rm[0]) == ROTATION_LOCK_ALL,
          {"rmode": None if rm is None else int(rm[0])})
except Exception as exc:
    check("pdrd_all_rotations_builds", False, {"raised": repr(exc)[:300]})

# Six rows saturate a body's reduced six-vector, freezing it completely. That
# is a meaningful request and is ALLOWED: the reduced orthonormalizer drops
# algebraically dependent rows before its count assert, and a seventh
# independent row cannot exist in R^6.
try:
    tm, rm = pdrd(lambda o: o.lock_all_translations().lock_all_rotations(),
                  "frozen")
    check("pdrd_fully_frozen_builds",
          tm is not None and rm is not None
          and int(tm[0]) == TRANSLATION_LOCK_ALL
          and int(rm[0]) == ROTATION_LOCK_ALL,
          {"tmode": None if tm is None else int(tm[0]),
           "rmode": None if rm is None else int(rm[0])})
except Exception as exc:
    check("pdrd_fully_frozen_builds", False, {"raised": repr(exc)[:300]})

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX, ctx.workspace],
        capture_output=True,
        text=True,
        env=env,
        timeout=max(ctx.timeout, 300.0),
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
    return r.report_named_checks(cases, label="lock mode cases")
