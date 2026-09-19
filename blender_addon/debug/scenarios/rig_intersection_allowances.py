# File: scenarios/rig_intersection_allowances.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Truth table for the three intersection allowances of issue #138, at the
# SCENE-BUILD gate.
#
# The allowances are opt-ins that stop an intersecting pair from being
# reported: `allow-self-intersection` and `allow-inter-object-intersection`
# per object, and a per-pin flag that exempts an element all of whose vertices
# are pinned by pins that set it.
#
# What makes this worth a scenario rather than a unit test is the NEGATIVE
# half. An allowance that suppresses too much is invisible in a happy-path
# test: a scene that was supposed to build still builds. So every "allowed"
# case here is paired with a control that must still be REFUSED, including
# cases where the wrong allowance is set on the right object. A change that
# collapses the three rules into one, or that lets either flag stand in for
# the other, fails on the controls and not on the allowances.
#
# The solver's own live scan covers the same rules at the other gate, the
# solver's own live scan. Both gates have to grant the same set: a scene that
# builds here and aborts there is a broken feature, and the reverse is a
# silent tolerance.
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


# No Blender and no solver process: this is the host-side build check, so it
# holds on the real-GPU jobs too.
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

import frontend
from frontend import App
from frontend._scene_ import ValidationError

app = App.create("rig_intersection_allowances")

# `mesh.square` returns 5 columns: xyz then uv, so only [:, :3] is a position.
V, F = app.mesh.square(res=4, ex=[1, 0, 0], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)

# ONE asset holding two overlapping sheets, so the tangle lives entirely
# inside a single object and there is no second object in the scene at all.
# Built by concatenation rather than by folding a sheet: a fold has to be
# tuned until it genuinely crosses, and one that merely comes close builds
# clean, which would make the allowed case pass for the wrong reason.
angle = np.deg2rad(12.0)
rot = np.array([
    [np.cos(angle), 0.0, np.sin(angle)],
    [0.0, 1.0, 0.0],
    [-np.sin(angle), 0.0, np.cos(angle)],
])
V2 = np.array(V, dtype=np.float64, copy=True)
V2[:, :3] = (V2[:, :3] @ rot.T) + np.array([0.3, 0.0, 0.0])
app.asset.add.tri("tangled", np.vstack([V, V2]), np.vstack([F, F + len(V)]))


def builds(setup):
    """True when scene.build() accepts the scene."""
    scene = app.scene.create()
    setup(scene)
    try:
        scene.build(quiet=True)
    except ValidationError:
        return False
    return True


def pair(scene, a_flags=(), b_flags=(), pin_a=False, pin_a_allows=False):
    a = scene.add("sheet").at(0.0, 0.0, 0.0)
    b = scene.add("sheet").at(0.3, 0.0, 0.0).rotate(12.0, "y")
    for key in a_flags:
        a.param.set(key, 1.0)
    for key in b_flags:
        b.param.set(key, 1.0)
    if pin_a:
        # Every vertex of `a`, so every one of its elements is fully pinned
        # and the all-N-vertices rule can fire at all. The op is what keeps
        # `a` DYNAMIC: `Object.update_static` promotes a fully pinned object
        # with no operations to a rest-pose STATIC collider, which leaves the
        # solved namespace entirely and takes its pins with it, so the pin
        # allowance could never apply and the case would pass or fail for a
        # reason that has nothing to do with it. The move starts at t=0, so
        # the pose this build check reads is unchanged.
        a.pin(allow_intersection=pin_a_allows).move_by(
            [0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    return a, b


def tangled(scene, flags=(), pin=False, pin_allows=False):
    obj = scene.add("tangled").at(0.0, 0.0, 0.0)
    for key in flags:
        obj.param.set(key, 1.0)
    if pin:
        # See `pair`: the op is what keeps the object out of the static
        # collision-mesh pool, where `both_collider` would skip the pair and
        # the case would pass without the allowance doing anything.
        obj.pin(allow_intersection=pin_allows).move_by(
            [0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    return obj


cases = {}


def check(name, expect_build, setup):
    got = builds(setup)
    cases[name] = {
        "ok": got == expect_build,
        "details": {"expected": "build" if expect_build else "refuse",
                    "got": "build" if got else "refuse"},
    }


# --- inter-object -------------------------------------------------------
check("inter_control_refused", False, lambda s: pair(s))
check("inter_allowed_one_side", True,
      lambda s: pair(s, a_flags=("allow-inter-object-intersection",)))
check("inter_allowed_other_side", True,
      lambda s: pair(s, b_flags=("allow-inter-object-intersection",)))
check("inter_not_covered_by_self_flag", False,
      lambda s: pair(s, a_flags=("allow-self-intersection",)))

# --- self ---------------------------------------------------------------
check("self_control_refused", False, lambda s: tangled(s))
check("self_allowed", True, lambda s: tangled(s, ("allow-self-intersection",)))
check("self_not_covered_by_inter_flag", False,
      lambda s: tangled(s, ("allow-inter-object-intersection",)))

# --- per-pin ------------------------------------------------------------
# A pin that does NOT ask for the allowance must not grant one, which is what
# separates "pinned" from "pinned and allowed": the exemption is opt-in per
# pin, not a property of being pinned.
check("pin_without_flag_refused", False,
      lambda s: pair(s, pin_a=True, pin_a_allows=False))
check("pin_with_flag_allowed", True,
      lambda s: pair(s, pin_a=True, pin_a_allows=True))
check("pin_with_flag_allowed_self", True,
      lambda s: tangled(s, pin=True, pin_allows=True))

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
        timeout=max(ctx.timeout, 180.0),
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
    return r.report_named_checks(cases, label="allowance cases")
