# File: scenarios/_existing_intersection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The probe the `rig_existing_intersection*` scenarios share, and the runner
# that drives one section of it.
#
# ALLOW EXISTING INTERSECTIONS, asserted end to end through the frontend. The
# option lets a scene START tangled: the scene-build check links, vertex by
# vertex, the pairs it finds crossing (or closer than their contact offsets)
# in a pair an opted-in object belongs to, and the solver treats a linked pair
# as a neighbor for the whole run. Every other pair keeps full contact.
#
# TWO SECTIONS, each its own scenario so a failure in one does not hide the
# other:
#
#   build  which scenes build and which are refused, with no solve. A tangled
#          start is refused without the flag and builds with it; either side
#          opting in is enough; another allowance does not stand in for it; a
#          pair closer than its contact offsets is linked like a crossing; a
#          crossing against a STATIC collision mesh links into the collider's
#          pool; a clean flagged scene links nothing; sand is refused by name;
#          and the record the overlay draws (`start_link_exemptions`) names
#          every exempted pair with each element's positions.
#
#   run    what the solver then does, in ONE solve with the cases side by
#          side. A small sheet starts tilted through a held sheet, so its
#          lower half hangs through and its upper half sits above. It must run
#          to its last frame, which the solver's own per-step scan already
#          requires, and at every sampled frame every intersecting triangle
#          pair must be EXEMPT by the links the build made: that is the
#          guarantee statement, "no NEW penetration", checked independently of
#          the solver. And the upper half must come to rest ON the held sheet,
#          0.3 above the floor it would reach if the exemption leaked past the
#          linked region. The same case against a STATIC collider runs beside
#          it. A third case starts a sheet 4 mm above a held one with contact
#          offsets summing to 10 mm: every pair is inside the barrier's reach
#          and every pair is linked (decision D2), so the sheet must fall
#          THROUGH to the floor. That is the case that sees the barrier
#          ASSEMBLY honor the links; in the tilted cases the linked pairs sit
#          outside the contact gap, so a barrier assembled for them would
#          change nothing measurable.
#
# NO BLENDER, and a SUBPROCESS, for the reason `_allowance_contact.py` states.

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

EXISTING = "allow-existing-intersection"
SELF = "allow-self-intersection"

# The tilted sheet: 0.5 wide, turned 30 degrees about x, so its local z maps
# to a height of z / 2 and it crosses the held sheet's plane along a line. The
# lift keeps that line between two rows of its vertices rather than on one,
# so no vertex starts exactly in the held plane.
TILT = 30.0
LIFT = 0.013
FLOOR = -0.3
FRAMES = 40
SPACING = 2.0

cases = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}
    print("PPFCASE %s %s" % (name, "pass" if ok else "fail"), flush=True)


def begin(label):
    print("PPFSTART " + label, flush=True)


def sheet(app, size, res):
    V, F = app.mesh.square(res=res, size=size, ex=[1, 0, 0], ey=[0, 0, 1])
    return np.asarray(V, dtype=np.float64)[:, :3], np.asarray(F)


def assets(app):
    Vh, Fh = sheet(app, 1.0, 10)
    Vf, Ff = sheet(app, 0.5, 6)
    app.asset.add.tri("held", Vh, Fh)
    app.asset.add.tri("tilted", Vf, Ff)
    app.asset.add.tri("near", Vf, Ff)


def add_tangle(scene, x, layout="pinned", tilted_flags=(EXISTING,),
               held_flags=()):
    """A held sheet at *x* and a tilted sheet crossing it.

    `layout` "pinned" holds the larger sheet with a slow move, which keeps it
    in the solved namespace; "static" pins it without one, which makes it a
    STATIC collision mesh.
    """
    h = scene.add("held").at(x, 0.0, 0.0)
    t = scene.add("tilted").at(x, LIFT, 0.0).rotate(TILT, "x")
    for key in held_flags:
        h.param.set(key, 1.0)
    for key in tilted_flags:
        t.param.set(key, 1.0)
    holder = h.pin()
    if layout == "pinned":
        holder.move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
    return h, t


def try_build(label, populate):
    app = App.create("rig_existing_" + label)
    assets(app)
    scene = app.scene.create()
    populate(app, scene)
    try:
        fixed = scene.build(quiet=True)
    except Exception as error:
        return None, type(error).__name__, str(error)[:300]
    return fixed, None, ""


if SECTION == "build":
    def expect_refused(name, populate, error_type="ValidationError",
                       says=None):
        fixed, kind, text = try_build(name, populate)
        ok = kind == error_type and (says is None or says in text)
        check(name, ok, {"error": kind, "message": text})

    def expect_linked(name, populate, want_static=False):
        fixed, kind, text = try_build(name, populate)
        if fixed is None:
            check(name, False, {"error": kind, "message": text})
            return
        links = np.asarray(fixed.start_links)
        n_dyn = len(fixed._vert[1])
        got = {"links": int(len(links)), "dynamic_vertices": int(n_dyn),
               "static_links": int((links >= n_dyn).any(axis=1).sum())
               if len(links) else 0}
        ok = len(links) > 0 and bool((links[:, 0] < links[:, 1]).all())
        if want_static:
            ok = ok and got["static_links"] > 0
        check(name, ok, got)

    expect_refused("crossing_unflagged_is_refused",
                   lambda app, s: add_tangle(s, 0.0, tilted_flags=()))
    expect_linked("crossing_flagged_is_linked",
                  lambda app, s: add_tangle(s, 0.0))
    expect_linked("crossing_held_side_flag_is_enough",
                  lambda app, s: add_tangle(s, 0.0, tilted_flags=(),
                                            held_flags=(EXISTING,)))
    # Two objects, so the self allowance names a pair this is not.
    expect_refused("crossing_self_allowance_does_not_stand_in",
                   lambda app, s: add_tangle(s, 0.0, tilted_flags=(SELF,)))
    expect_linked("crossing_static_collider_links_into_its_pool",
                  lambda app, s: add_tangle(s, 0.0, layout="static"),
                  want_static=True)

    def parallel(flag):
        def populate(app, s):
            h = s.add("held").at(0.0, 0.0, 0.0)
            h.pin().move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
            n = s.add("near").at(0.0, 0.004, 0.0)
            h.param.set("contact-offset", 0.005)
            n.param.set("contact-offset", 0.005)
            if flag:
                n.param.set(EXISTING, 1.0)
        return populate

    # Decision D2: a pair closer than its contact offsets is refused exactly
    # as a crossing is, and linked exactly as one.
    expect_refused("offset_violation_unflagged_is_refused", parallel(False))
    expect_linked("offset_violation_flagged_is_linked", parallel(True))

    def clean(app, s):
        t = s.add("tilted").at(0.0, 1.0, 0.0)
        t.param.set(EXISTING, 1.0)
        s.add("held").at(0.0, 0.0, 0.0).pin()

    fixed, kind, text = try_build("clean", clean)
    check("clean_flagged_scene_links_nothing",
          fixed is not None and len(fixed.start_links) == 0
          and fixed.start_link_exemptions() == [],
          {"error": kind, "message": text,
           "links": None if fixed is None else int(len(fixed.start_links))})

    # The overlay record: one record, typed, counting every exempted pair, and
    # every drawn pair two elements of three (triangle) or two (rod edge)
    # three-coordinate positions. The crossing's pairs are triangle pairs,
    # and every drawn position lies on one of the two sheets' extents.
    fixed, kind, text = try_build("record", lambda app, s: add_tangle(s, 0.0))
    records = [] if fixed is None else fixed.start_link_exemptions()
    got = {"error": kind, "records": len(records)}
    ok = fixed is not None and len(records) == 1
    if ok:
        rec = records[0]
        pairs = rec.get("pairs", [])
        got.update({"type": rec.get("type"), "count": rec.get("count"),
                    "pairs": len(pairs)})
        shapes = {(len(p["a"]), len(p["b"])) for p in pairs}
        coords = [c for p in pairs for side in ("a", "b") for c in p[side]]
        ok = (rec.get("type") == "existing_intersection"
              and rec.get("count") == len(pairs) and len(pairs) > 0
              and shapes == {(3, 3)}
              and all(len(c) == 3 and max(abs(v) for v in c) <= 0.51
                      for c in coords))
        got["shapes"] = sorted(shapes)
    check("overlay_record_names_every_exempted_pair", ok, got)

    def sand(app, s):
        rng = np.random.default_rng(0)
        app.asset.add.points("grains", rng.uniform(-0.1, 0.1, (20, 3)))
        s.add("grains").at(0.0, 1.0, 0.0).param.set(EXISTING, 1.0)

    expect_refused("sand_flagged_is_refused_by_name", sand,
                   error_type="ValueError", says="sand")


if SECTION == "run":
    app = App.create("rig_existing_run")
    assets(app)
    scene = app.scene.create()
    layouts = (("pinned_tangle", "pinned", -SPACING),
               ("static_tangle", "static", 0.0))
    for _name, layout, x in layouts:
        add_tangle(scene, x, layout=layout)
    OFFSET_X = SPACING
    NEAR_Y = 0.004
    h = scene.add("held").at(OFFSET_X, 0.0, 0.0)
    h.pin().move_by([0.0, 0.0, 0.001], t_start=0.0, t_end=10.0)
    h.param.set("contact-offset", 0.005)
    n = scene.add("near").at(OFFSET_X, NEAR_Y, 0.0)
    n.param.set("contact-offset", 0.005)
    n.param.set(EXISTING, 1.0)
    scene.add.invisible.wall([0, FLOOR, 0], [0, 1, 0])
    begin(SECTION)
    try:
        fixed = scene.build(quiet=True)
        session = app.session.create(fixed)
        session.param.set("frames", FRAMES)
        session = session.build()
        session.start(blocking=True)
        finished = bool(session.finished())

        def frame(n):
            got, _ = session.get.vertex(n)
            return np.asarray(got, dtype=np.float64)[:, :3]

        first = frame(0)
        samples = [(n, frame(n)) for n in list(range(0, FRAMES, 5))
                   + [FRAMES - 1]]
        tri = np.asarray(fixed._tri)
        links = np.asarray(fixed.start_links)
    except Exception as error:
        for name, _layout, _x in layouts:
            check(name + "_runs", False, {"error": repr(error)[:400]})
        check("offset_linked_sheet_falls_through", False,
              {"error": repr(error)[:400]})
        samples = None

    if samples is not None:
        n_dyn = len(first)
        linked = {}
        for u, v in links:
            linked.setdefault(int(u), set()).add(int(v))
            linked.setdefault(int(v), set()).add(int(u))

        def exempt(ta, tb):
            return any(int(w) in linked.get(int(u), ())
                       for u in ta for w in tb)

        for name, layout, x in layouts:
            mine = np.abs(first[:, 0] - x) < 0.75
            own = tri[mine[tri].all(axis=1)]
            upper = mine & (first[:, 1] > 0.05)
            unexempt = []
            pairs_per_sample = []
            for n, vert in samples:
                pairs = check_self_intersection(vert, own, None)
                pairs_per_sample.append(len(pairs))
                for a, b in pairs:
                    if a < 0 or b < 0:
                        continue
                    if not exempt(own[a], own[b]):
                        unexempt.append((n, own[a].tolist(), own[b].tolist()))
            rest_upper = float(samples[-1][1][upper, 1].mean())
            got = {"finished": finished, "links": int(len(links)),
                   "pairs_per_sample": pairs_per_sample,
                   "unexempt": unexempt[:5], "upper_rest_y": rest_upper,
                   "upper_vertices": int(upper.sum())}
            # A STATIC collider is not in the output buffer, so its pairs are
            # the solver's own scan to judge, which the finished run passed.
            if layout == "pinned":
                check(name + "_has_no_new_penetration",
                      finished and not unexempt and pairs_per_sample[0] > 0,
                      got)
            else:
                check(name + "_runs", finished, got)
            check(name + "_unlinked_part_still_collides",
                  finished and upper.sum() > 0 and rest_upper > FLOOR / 2.0,
                  got)

        near = ((np.abs(first[:, 1] - NEAR_Y) < 1e-4)
                & (np.abs(first[:, 0] - OFFSET_X) < 0.3))
        rest_near = float(samples[-1][1][near, 1].mean())
        check("offset_linked_sheet_falls_through",
              finished and near.sum() > 0
              and FLOOR - 0.01 < rest_near < FLOOR + 0.05,
              {"finished": finished, "near_vertices": int(near.sum()),
               "rest_y": rest_near, "floor": FLOOR})

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
        return r.failed([
            f"probe section {section!r} timed out after {expired.timeout:.0f}s; "
            f"finished: {done}"
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
