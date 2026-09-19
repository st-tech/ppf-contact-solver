# File: scenarios/rig_lock_axes_projector.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Every Lock Translation / Lock Rotation mode driven through the
# solver, asserting the invariant each one names.
#
# `rig_lock_axes` covers the authoring and scene-build gate: it proves the
# right bytes reach the session directory. This covers the other gate, the
# projector that reads them. Both have to agree, and the interesting failure
# is one-sided: a mode that exports correctly and is then dropped by the
# projector leaves a scene that builds, runs, and is simply not locked.
#
# EVERY LOCKED CASE IS PAIRED WITH A CONTROL THAT MUST MOVE. A test that only
# asserts "the locked object did not move" passes just as well when the
# solver is not integrating at all, when gravity is zero, or when the
# excitation never reached the solver. The controls are what make the locked
# assertions mean something, so a failing control fails the scenario rather
# than being treated as a setup detail.
#
# The rotation excitation is an initial angular velocity, baked in at scene
# build. Gravity cannot serve: a uniform field exerts no torque about the
# center of mass, so a free sheet under gravity alone does not rotate and
# every rotation case would pass for the wrong reason.
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


# The solver is the point: it is the only CUDA-free implementation of the
# projector, so this is the lock gate that runs on a host with no GPU.
# RUNS ON THE REAL BACKEND. What this asserts is the PROJECTOR, which is
# backend-neutral: every locked case is paired with a control that must
# move, so a backend that integrates nothing fails it rather than passing.
BACKENDS = ("real",)
# Runs a solver process per case, so it should not share a worker with another
# solver.
NOT_PARALLELIZABLE = True


_PROBE = r'''
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

import frontend
from frontend import App

FRAMES = 24


def drive(project, setup, gravity, spin):
    """Run one sheet through the solver and return its motion.

    Returns the center-of-mass displacement and the best-fit rigid rotation
    angle between the first and last output frame. The sheet is uniform, so
    the plain vertex mean is its mass-weighted center of mass.
    """
    app = App.create(project)
    V, F = app.mesh.square(res=6, ex=[1, 0, 0], ey=[0, 1, 0])
    app.asset.add.tri("sheet", V, F)

    scene = app.scene.create()
    obj = scene.add("sheet").at(0.0, 0.0, 0.0)
    if spin is not None:
        obj.angular_velocity(*spin)
    setup(obj)
    fixed = scene.build(quiet=True)

    session = app.session.create(fixed)
    session.param.set("dt", 0.01).set("frames", FRAMES)
    session.param.set("gravity", gravity)
    session = session.build()
    session.start(blocking=True)
    if not session.finished():
        raise RuntimeError("run did not finish")

    first, _ = session.get.vertex(0)
    last, _ = session.get.vertex(FRAMES - 1)
    first = np.asarray(first, dtype=np.float64)[:, :3]
    last = np.asarray(last, dtype=np.float64)[:, :3]

    com = last.mean(axis=0) - first.mean(axis=0)

    # Kabsch about each frame's own centroid, so the angle measures rotation
    # alone and is not contaminated by whatever the translation lock allowed.
    a = first - first.mean(axis=0)
    b = last - last.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    cos = max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0))
    return {"com": com.tolist(), "angle_deg": math.degrees(math.acos(cos))}


cases = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


# EVERY COMPONENT OF BOTH EXCITATIONS IS NON-ZERO, and that is the point.
# There are three translation rows and three angular rows, and a row is only
# tested if something pushes along it: a gravity lying in the XZ plane never
# excites the Y translation row, and a spin about Z alone never excites the X
# or Y angular rows, so a projector that dropped those rows would pass every
# case below. This is not hypothetical, it is the exact shape of a defect this
# feature already shipped once, where a constraint-space accumulator written
# one line per component silently stopped at four of six.
#
# The gravity also gives an axis-mode translation lock something to permit
# along its axis AND something to forbid across it.
G = (4.0, 2.5, -9.8)
SPIN = (3.0, -2.0, 4.0)

# --- controls: an unlocked sheet must translate AND rotate ----------------
free = drive("rig_emul_lock_free", lambda o: o, G, SPIN)
check("control_unlocked_sheet_translates",
      min(abs(v) for v in free["com"]) > 1e-3, free)
check("control_unlocked_sheet_rotates",
      free["angle_deg"] > 1.0, free)

# --- translation, axis mode ----------------------------------------------
# The axis is X, so the center of mass may still slide along X and must not
# move across it. Asserting only the second half would pass on a sheet that
# was frozen outright, which is a different mode.
tx = drive("rig_emul_lock_t_axis",
           lambda o: o.lock_translation(1.0, 0.0, 0.0), G, None)
# Y and Z are both pushed by gravity and both must be held; only X is free.
check("translation_axis_holds_the_perpendicular_plane",
      abs(tx["com"][1]) < 1e-4 and abs(tx["com"][2]) < 1e-4, tx)
check("translation_axis_leaves_its_own_axis_free",
      abs(tx["com"][0]) > 1e-3, tx)

# --- translation, all axes -----------------------------------------------
ta = drive("rig_emul_lock_t_all", lambda o: o.lock_all_translations(), G, None)
check("translation_all_holds_every_direction",
      max(abs(v) for v in ta["com"]) < 1e-4, ta)

# --- rotation, all axes --------------------------------------------------
ra = drive("rig_emul_lock_r_all", lambda o: o.lock_all_rotations(), G, SPIN)
check("rotation_all_removes_the_net_rotation",
      ra["angle_deg"] < 0.5, ra)
# Rotation and translation are independent, so an all-axes ROTATION lock must
# leave the center of mass free. Without this, a rotation lock that
# accidentally froze translation too would still pass.
check("rotation_all_leaves_translation_free",
      abs(ra["com"][2]) > 1e-3, ra)

# --- rotation, allow-only, still behaves ---------------------------------
# Allow-only about Z permits exactly the spin the excitation asks for, so the
# sheet must still turn. This is the case an over-eager all-axes branch would
# break by treating every rotation lock as total.
rz = drive("rig_emul_lock_r_allow",
           lambda o: o.lock_rotation(0.0, 0.0, 1.0), G, SPIN)
check("rotation_allow_only_keeps_its_permitted_axis",
      rz["angle_deg"] > 1.0, rz)

# --- both all-axes locks together ----------------------------------------
both = drive("rig_emul_lock_both",
             lambda o: o.lock_all_translations().lock_all_rotations(), G, SPIN)
check("both_all_axes_locks_hold_together",
      max(abs(v) for v in both["com"]) < 1e-4 and both["angle_deg"] < 0.5,
      both)

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    # NO KNOBS. Both of the two this used to set only neutralized stub
    # behavior: one asked the solver to compute elasticity at all, and one
    # turned off its per-step sleep. A real backend always computes real
    # elasticity and has no sleep, so the controls move for the right reason.
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX],
        capture_output=True,
        text=True,
        env=env,
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
    return r.report_named_checks(cases, label="lock cases")
