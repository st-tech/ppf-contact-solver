# File: scenarios/_allowance_contact.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The probe the `rig_intersection_allowance_contact*` scenarios share, and the
# runner that drives one section of it.
#
# An allowed pair is NOT A CONTACT PAIR, asserted by simulating one.
#
# The four intersection allowances (`allow-self-intersection`,
# `allow-inter-object-intersection`, `allow-inter-group-intersection` and a
# pin's `allow_intersection`) take the pairs they name out of every pass:
# contact assembles no barrier for them, the CCD line search does not filter
# the step against them, and the intersection scan does not report them. `rig_intersection_allowances` settles which scenes
# each allowance ADMITS at the build check; this settles what the solver then
# DOES with an allowed pair, which only a run can answer.
#
# THE SHAPE: a small sheet falls onto a larger sheet that is held flat 0.1
# below it, with an invisible floor 0.3 below that. Where the pair is allowed,
# the small sheet falls THROUGH the held one and comes to rest on the floor;
# where it is not, it comes to rest ON the held sheet. The two outcomes are
# 0.3 apart, so the verdict needs no tolerance tuned to a backend. The floor is
# an analytic collider, which no allowance reaches, so every pass-through case
# also proves the allowance stayed inside the pairs it names.
#
# Every allowance is paired with a control that must NOT pass through: the
# same scene with no flag, and the same scene with the OTHER flag, since
# over-reaching (self covering inter-object, or the reverse) is invisible in a
# happy-path test.
#
# The drape section is the report that prompted these scenarios: a cloth
# hung vertically over a floor and let fall. With `allow-self-intersection` its
# folds cross one another as it crumples; without it no sampled frame has a
# self-intersecting triangle pair, which is the penetration-free guarantee every
# pair the user did not allow still gets. The allowed case reads the PEAK over
# the sampled frames, not the last frame: once the cloth lies on the floor its
# layers can settle flat and uncrossed, and whether they do varies run to run
# (the solver is not deterministic), while the crumple on the way down crossed
# itself at every resolution measured (a peak of 28 pairs at res 16, 156 at 20
# and 358 at 24).
#
# NO BLENDER. The scene is authored through `frontend` directly, in a
# SUBPROCESS: importing `frontend` loads the per-tree cdylib and installs the
# rig's debug patches, and the orchestrator imports every scenario into one
# long-lived process that must not inherit either.
#
# ONE SCENE PER SECTION. On a GPU a solve pays a process start and a device
# start that dwarf these tiny scenes, so a solve per case (21 of them) took
# 452 s on Linux CUDA and outran a 900 s budget on the Windows leg. Each
# section instead lays all of its cases out side by side in one scene, 2 m
# apart, and solves once; every case keeps its own objects, pins and group
# labels, and is judged by the vertices that started in its own slot. The
# three sections stay separate scenarios so a failure in one does not hide the
# others and the orchestrator can place them on different shards. The probe
# prints a line before its solve and after each verdict, so a probe the runner
# has to kill still names where it stopped.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


PROBE = r'''
import json
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
SECTION = sys.argv[2]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App
from frontend._intersection_ import check_self_intersection

SELF = "allow-self-intersection"
INTER = "allow-inter-object-intersection"
GROUP = "allow-inter-group-intersection"

# Where the falling sheet starts, where the held sheet lies, and where the
# floor is. Resting on the held sheet reads about 0.0 (a few centimeters lower
# on a pull pin, which sags under the load); resting on the floor reads about
# FLOOR. The two bands below meet at the midpoint between the two.
DROP = 0.1
FLOOR = -0.3
FRAMES = 40
# Every case of a section shares ONE scene and ONE solve, each laid out at its
# own x. A held sheet is 1 m wide, so 2 m apart leaves a 1 m gap no contact
# range comes near, and each case's objects, pins and group labels are its own.
# The row is centered on the origin, so the farthest case stays within about
# 9 m of it, inside the coordinate range every solver build supports.
SPACING = 2.0


def case_x(i, n):
    return (i - 0.5 * (n - 1)) * SPACING

cases = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}
    print("PPFCASE %s %s" % (name, "pass" if ok else "fail"), flush=True)


def begin(label):
    # A line before each solve, flushed, so a probe the scenario has to kill
    # still says which solve it was inside and which cases had finished.
    print("PPFSTART " + label, flush=True)


def sheet(app, size, res):
    V, F = app.mesh.square(res=res, size=size, ex=[1, 0, 0], ey=[0, 0, 1])
    return np.asarray(V, dtype=np.float64)[:, :3], np.asarray(F)


def run_session(app, scene, frames):
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", frames)
    session = session.build()
    session.start(blocking=True)

    def frame(n):
        got, _ = session.get.vertex(n)
        return np.asarray(got, dtype=np.float64)[:, :3]

    return fixed, bool(session.finished()), frame


def passed_through(got):
    # Came to rest on the floor, and not below it: the floor is an analytic
    # collider no allowance reaches.
    return (got["finished"] and got["falling_vertices"] > 0
            and FLOOR - 0.01 < got["rest_y"] < FLOOR + 0.05)


def held(got):
    return (got["finished"] and got["falling_vertices"] > 0
            and FLOOR / 2.0 < got["rest_y"] < DROP)


def add_case(scene, held_x, x, name, layout, falling=(), held_flags=(),
             pin_allow=False, groups=(None, None)):
    """Lay one drop case out at *x*: a small sheet above a held one.

    `layout` is "self" (both sheets in ONE object, the larger one pinned),
    "pinned" (two objects, the larger one fully pinned with a slow move so it
    stays in the solved namespace), "pull" (two objects, the larger one held
    by a PULL pin), "static" (two objects, the larger one fully pinned with
    no move, which makes it a STATIC collision mesh) or "split" (the held
    sheet's left half pinned by an allowing pin and its right half by a plain
    one, with a small sheet dropped over each half). `groups` names the held
    and the falling object's groups, None leaving one in the default group;
    labels are prefixed with the case name so no two cases share a group.
    `held_x` is the held sheet's own vertex x coordinates, which name its
    vertices for the pins.
    """
    if layout == "self":
        obj = scene.add("both").at(x, 0.0, 0.0)
        for key in falling:
            obj.param.set(key, 1.0)
        obj.pin(list(range(len(held_x))), allow_intersection=pin_allow)
        return
    h = scene.add("held").at(x, 0.0, 0.0)
    if layout == "split":
        left = [i for i in range(len(held_x)) if held_x[i] < -1e-6]
        right = [i for i in range(len(held_x)) if held_x[i] >= -1e-6]
        h.pin(left, allow_intersection=pin_allow).move_by(
            [0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
        h.pin(right).move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
        scene.add("small").at(x - 0.25, DROP, 0.0)
        scene.add("small").at(x + 0.25, DROP, 0.0)
        return
    f = scene.add("falling").at(x, DROP, 0.0)
    if groups[0] is not None:
        h.group(name + ":" + groups[0])
    if groups[1] is not None:
        f.group(name + ":" + groups[1])
    for key in held_flags:
        h.param.set(key, 1.0)
    for key in falling:
        f.param.set(key, 1.0)
    holder = h.pin(allow_intersection=pin_allow)
    if layout == "pinned":
        holder.move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
    elif layout == "pull":
        holder.pull(10.0)


def rest_of(first, last, finished, lo, hi):
    # The falling vertices are every vertex that STARTED at the drop height
    # inside [lo, hi) in x. The output buffer's order is the fixed scene's, not
    # the assets', so vertices are picked by where they were, not by index.
    mask = ((np.abs(first[:, 1] - DROP) < 1e-4)
            & (first[:, 0] >= lo) & (first[:, 0] < hi))
    return {"finished": finished, "falling_vertices": int(mask.sum()),
            "rest_y": float(last[mask, 1].mean()) if mask.any()
            else float("nan")}


def run_drops(project, specs):
    """Solve every drop case in *specs* in one scene and judge each."""
    app = App.create(project)
    Vh, Fh = sheet(app, 1.0, 10)
    Vf, Ff = sheet(app, 0.5, 6)
    Vs, Fs = sheet(app, 0.3, 5)
    app.asset.add.tri("held", Vh, Fh)
    app.asset.add.tri("falling", Vf, Ff)
    app.asset.add.tri("small", Vs, Fs)
    app.asset.add.tri("both", np.vstack([Vh, Vf + np.array([0.0, DROP, 0.0])]),
                      np.vstack([Fh, Ff + len(Vh)]))
    scene = app.scene.create()
    for i, (name, _expect, kw) in enumerate(specs):
        add_case(scene, Vh[:, 0], case_x(i, len(specs)), name, **kw)
    scene.add.invisible.wall([0, FLOOR, 0], [0, 1, 0])
    begin(SECTION)
    try:
        _, finished, frame = run_session(app, scene, FRAMES)
        first, last = frame(0), frame(FRAMES - 1)
    except Exception as error:
        for name, _expect, _kw in specs:
            check(name, False, {"error": repr(error)[:400]})
        return
    for i, (name, expect, kw) in enumerate(specs):
        x = case_x(i, len(specs))
        if kw.get("layout") == "split":
            left = rest_of(first, last, finished, x - 0.5, x)
            right = rest_of(first, last, finished, x, x + 0.5)
            left_ok = passed_through(left) if expect else held(left)
            check(name, left_ok and held(right),
                  {"left": left, "right": right})
            continue
        got = rest_of(first, last, finished, x - 0.3, x + 0.3)
        check(name, passed_through(got) if expect else held(got), got)


if SECTION == "pairs":
    run_drops("rig_allow_contact_pairs", [
        ("self_allowed_passes_through", True,
         dict(layout="self", falling=(SELF,))),
        ("self_unflagged_is_held", False, dict(layout="self")),
        ("self_inter_object_flag_does_not_cover", False,
         dict(layout="self", falling=(INTER,))),
        ("inter_object_falling_side_passes", True,
         dict(layout="pinned", falling=(INTER,))),
        ("inter_object_held_side_passes", True,
         dict(layout="pinned", held_flags=(INTER,))),
        ("inter_object_unflagged_is_held", False, dict(layout="pinned")),
        ("inter_object_self_flags_do_not_cover", False,
         dict(layout="pinned", falling=(SELF,), held_flags=(SELF,))),
        ("pin_allowance_passes", True, dict(layout="pinned", pin_allow=True)),
        ("static_collider_flagged_passes", True,
         dict(layout="static", falling=(INTER,))),
        ("static_collider_unflagged_is_held", False, dict(layout="static")),
    ])

if SECTION == "pins_groups":
    run_drops("rig_allow_contact_pins_groups", [
        ("pin_allowance_same_object_passes", True,
         dict(layout="self", pin_allow=True)),
        ("pull_pin_allowance_passes", True, dict(layout="pull", pin_allow=True)),
        ("pull_pin_unflagged_is_held", False, dict(layout="pull")),
        ("inter_group_falling_side_passes", True,
         dict(layout="pinned", falling=(GROUP,), groups=("floor", "cloth"))),
        ("inter_group_held_side_passes", True,
         dict(layout="pinned", held_flags=(GROUP,), groups=("floor", "cloth"))),
        ("inter_group_same_group_is_held", False,
         dict(layout="pinned", falling=(GROUP,), held_flags=(GROUP,),
              groups=("cloth", "cloth"))),
        ("inter_group_static_collider_passes", True,
         dict(layout="static", falling=(GROUP,), groups=("cloth", "cloth"))),
        ("pin_allowance_covers_only_held_faces", True,
         dict(layout="split", pin_allow=True)),
        ("pin_split_control_is_held", False, dict(layout="split")),
    ])


if SECTION == "drape":
    # Both cloths in one scene, 3 m apart: each hangs 1 m wide over the same
    # invisible floor, and only the first carries the allowance.
    app = App.create("rig_allow_contact_drape")
    V, F = app.mesh.square(res=20, size=1.0, ex=[1, 0, 0], ey=[0, 1, 0])
    app.asset.add.tri("cloth", np.asarray(V, dtype=np.float64)[:, :3],
                      np.asarray(F))
    scene = app.scene.create()
    drapes = (("drape_allowed_crosses_itself", True, 0.0),
              ("drape_unflagged_never_intersects", False, 3.0))
    for _name, allow, x in drapes:
        obj = scene.add("cloth").at(x, 0.6, 0.0).rotate(15.0, "x")
        if allow:
            obj.param.set(SELF, 1.0)
    scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
    frames = 80
    begin(SECTION)
    try:
        fixed, finished, frame = run_session(app, scene, frames)
        # The triangles as the fixed scene numbered them, which is the
        # numbering the output buffer is in, split by which cloth each
        # vertex started in.
        tri = np.asarray(fixed._tri)
        first = frame(0)
        samples = [(n, frame(n)) for n in list(range(0, frames, 4))
                   + [frames - 1]]
    except Exception as error:
        for name, _allow, _x in drapes:
            check(name, False, {"error": repr(error)[:400]})
        samples = None
    if samples is not None:
        for name, allow, x in drapes:
            mine = np.abs(first[:, 0] - x) < 1.0
            own = tri[mine[tri].all(axis=1)]
            pairs = [len(check_self_intersection(vert, own, None))
                     for _n, vert in samples]
            min_y = min(float(vert[mine, 1].min()) for _n, vert in samples)
            got = {"finished": finished, "min_y": min_y,
                   "pairs_per_sample": pairs, "triangles": int(len(own))}
            ok = finished and pairs[0] == 0 and min_y > -1.0e-3
            ok = ok and (max(pairs) > 0 if allow else max(pairs) == 0)
            check(name, ok, got)

print("PPFRESULT" + json.dumps(cases))
'''


def run_section(ctx: r.ScenarioContext, section: str, label: str) -> dict:
    """Run one section of the probe and report its cases."""
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, REPO_ROOT_POSIX, section],
            capture_output=True,
            text=True,
            env=env,
            timeout=max(ctx.timeout, 900.0),
        )
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        done = [l[len("PPFCASE "):] for l in out.splitlines()
                if l.startswith("PPFCASE ")]
        begun = [l[len("PPFSTART "):] for l in out.splitlines()
                 if l.startswith("PPFSTART ")]
        # One PPFSTART per solve, then one PPFCASE per verdict it produced.
        inside = (f"the {begun[-1]!r} solve" if begun and not done
                  else "no solve")
        return r.failed([
            f"probe section {section!r} timed out after {expired.timeout:.0f}s "
            f"inside {inside}; finished: {done}"
        ])
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("PPFRESULT")
    ]
    if not marker:
        return r.failed([
            "probe produced no result marker; "
            f"rc={proc.returncode} stderr={proc.stderr[-800:]!r}"
        ])
    cases = json.loads(marker[-1][len("PPFRESULT"):])
    return r.report_named_checks(cases, label=label)
