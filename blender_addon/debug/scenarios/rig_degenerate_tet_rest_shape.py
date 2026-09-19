# File: scenarios/rig_degenerate_tet_rest_shape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The TET arm of the solver's build-time rest-shape gate,
# `builder::invert_rest_or_panic3`.
#
# A tet's rest edge matrix is inverted once at scene build and the elastic
# Hessian is quadratic in that inverse, so a coplanar or near-coplanar tet
# reaches the device carrying entries fp32 cannot represent. The linear solve
# then reports a non-finite quantity and names no geometry at all, which is the
# same failure the shell arm was reported for. The gate refuses the build
# instead, naming the tet, its four vertices and the measured conditioning.
#
# WHY THE CHEAPER TESTS DO NOT COVER THIS.
#
# `builder.rs` already has a unit test that calls `invert_rest_or_panic3` with
# a flat 3x3 matrix and reads its panic payload. That fixes the WORDING and
# nothing else. Three properties of the refusal live outside the function and
# only a scene can settle them:
#
#   * REACHABILITY. `compute_inv_rest` runs the tet arm only over
#     `mesh.mesh.mesh.tet`, and the shell scenario's geometry has no tets at
#     all, so `rig_degenerate_rest_shape` leaves this arm untouched. A marker
#     placed in this panic branch was never written across a full
#     sweep, so `ascii_matrix3` and the tet wording were reached by no rig
#     scenario at all.
#   * WHICH GATE OWNS THE CASE. `main.rs` asserts `triutils::tet_volumes` over
#     the CURRENT positions before the build, so an exactly coplanar tet is
#     refused there, with a different message, and never reaches this one. Only
#     a tet that is near-coplanar but of nonzero volume arrives here. A unit
#     test on the function cannot see that split, and a change that moves the
#     boundary would go unnoticed.
#   * WHAT THE ARTIST ACTUALLY READS. The panic string is not the message the
#     frontend surfaces. `analyze_solver_error` extracts a seven-line context
#     block around the `panicked at` line and joins it, and the refusal spans
#     two of those lines because the rest edge matrix is printed after a
#     newline. So the matrix line's survival is a property of that window, not
#     of the panic. The same path is where a non-ASCII matrix rendering costs
#     the whole diagnostic: box-drawing characters have no encoding in the ANSI
#     codepage a Windows console and a default-encoded Python reader use, and a
#     consumer there fails ON the text rather than reporting it. The ASCII
#     assertion below is checkable only end to end, on the message a reader
#     receives.
#
# THE SCENE is the smallest thing that reproduces it: a single tetrahedron,
# built through the frontend's own tet path (`app.asset.add.tet`), which hands
# the builder a tet directly and needs no SOLID asset and no tetrahedralizer.
# Its base is the triangle (0,0,0), (1,0,0), (0,1,0) and its apex sits at
# (1/3, 1/3, h), over the base centroid. Then:
#
#   * The rest edge matrix has columns (1,0,0), (0,1,0), (1/3,1/3,h), whose
#     singular values are 1.105542, 1 and 0.904534 * h. Its conditioning is
#     therefore 0.818182 * h, so h alone moves a case across the threshold with
#     nothing else about the mesh changing. That is what makes the refused and
#     accepted cases comparable: they differ in one number.
#   * All four surface triangles keep an area of at least 1/6 for every h,
#     h = 0 included, so `triutils::face_areas` never fires and the case that
#     reaches a gate is the tet one.
#   * Every pair of the four faces shares two vertices, so the scene-build
#     self-intersection and contact-offset scans skip every pair by their
#     share-a-vertex filter. A pancake tet is not rejected as tangled geometry
#     on its way to the solver.
#
# NO BLENDER AND NO GPU. The gate is host-side Rust in the solver driver,
# shared by both backends, so this runs anywhere the server runs and
# holds on the real-GPU jobs unchanged. `rig_degenerate_rest_shape` is the
# shell arm of the same gate; the two arms build the same quantity for the same
# energy out of the same arithmetic and must refuse on the same grounds.
#
# Subtests:
#   A. near_coplanar_tet_refused
#         h = 1e-05, a conditioning of 8.18e-06, about 42x below the floor.
#         The run must be refused.
#   B. refusal_names_the_tet_and_its_four_vertices
#         The message has to carry the tet index, all four vertices and the
#         word coplanar. Without them the artist is left with the symptom and
#         no geometry, which is what the gate exists to prevent. The four
#         indices are compared as a SET: the solver orders its vertex buffer
#         itself, so their order is not this scenario's to assert.
#   C. refusal_reports_the_conditioning_and_the_floor
#         The measured ratio, the floor it failed, and the rest edge matrix
#         itself. The ratio is parsed and checked against the geometry, so a
#         message that prints a constant instead of a measurement fails.
#   D. refusal_message_is_pure_ascii
#         Every character of the refusal is ASCII, so a Windows console and a
#         default-encoded reader can print it.
#   E. thin_but_sound_tet_still_builds
#         h = 1e-02, a conditioning of 8.18e-03, about 24x above the floor,
#         an aspect ratio of 100:1. Flat geometry is legitimate and must still
#         run; this is the cost of setting the gate too tight, so it is
#         asserted rather than assumed.
#   F. exactly_coplanar_tet_refused_by_the_earlier_gate
#         h = 0. This one never reaches the rest-shape gate: `main.rs` asserts
#         `triutils::tet_volumes` over the CURRENT positions first and aborts
#         with `volume is zero`. Recorded so the division of labor between the
#         two aborts is visible, and so a change that moves it is noticed.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# No Blender and no GPU: this is the host-side build check in the solver
# driver, so it holds on the real-GPU jobs too.
BACKENDS = ("real",)


_PROBE = r'''
import json
import re
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App


# `builder::REST_SHAPE_MIN_CONDITION`, which is `sqrt(f32::EPSILON)`, and the
# text the refusal prints it as (Rust `{:.3e}`).
FLOOR = 3.4526698e-04
FLOOR_TEXT = "3.453e-4"

# Conditioning of `pancake_tet(h)` in units of h: the rest edge matrix has
# singular values 1.105542, 1 and 0.904534 * h.
CONDITIONING_PER_H = 0.818182


def pancake_tet(h):
    """One tetrahedron whose apex sits `h` above the centroid of its base.

    The base is the triangle (0,0,0), (1,0,0), (0,1,0) and the apex is
    (1/3, 1/3, h), so the rest edge matrix has columns (1,0,0), (0,1,0),
    (1/3,1/3,h) and its conditioning is 0.818182 * h. Every surface triangle
    keeps an area of at least 1/6 for any h, so an earlier area gate has
    nothing to fire on and the tet gate is the one under test. Faces are wound
    outward: the base normal points down, the three side faces point up.
    """
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0 / 3.0, 1.0 / 3.0, h]], dtype=np.float64)
    F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.uint32)
    T = np.array([[0, 1, 2, 3]], dtype=np.uint32)
    return V, F, T


def attempt(tag, h):
    """Run one frame and return (finished, message)."""
    app = App.create("rig_degenerate_tet_rest_shape_" + tag)
    V, F, T = pancake_tet(h)
    app.asset.add.tet("wedge", V, F, T)
    scene = app.scene.create()
    scene.add("wedge")
    try:
        scene = scene.build(quiet=True)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    session = app.session.create(scene)
    session.param.set("dt", 0.01).set("frames", 1)
    session = session.build()
    try:
        session.start(blocking=True)
    except Exception as exc:
        return False, str(exc)
    return bool(session.finished()), ""


cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


# ----- A to D: the refused case ---------------------------------------
H_REFUSED = 1.0e-05
finished_a, msg_a = attempt("near_coplanar", H_REFUSED)
record("A_near_coplanar_tet_refused", not finished_a,
       {"finished": finished_a, "message": msg_a[-400:]})

named = re.search(
    r"Degenerate tetrahedron (\d+) on vertices (\d+), (\d+), (\d+), (\d+)",
    msg_a,
)
# The tet index is 0 because the scene holds exactly one tet. The four vertex
# indices are compared as a set: the solver orders its own vertex buffer.
vertices = {int(named.group(i)) for i in (2, 3, 4, 5)} if named else set()
record(
    "B_refusal_names_the_tet_and_its_four_vertices",
    (not finished_a)
    and named is not None
    and named.group(1) == "0"
    and vertices == {0, 1, 2, 3}
    and "coplanar" in msg_a,
    {
        "tet": named.group(1) if named else None,
        "vertices": sorted(vertices),
        "message": msg_a[-700:],
    },
)

measured = re.search(r"is conditioned at ([0-9.]+e-?[0-9]+)", msg_a)
ratio = float(measured.group(1)) if measured else None
expected = CONDITIONING_PER_H * H_REFUSED
record(
    "C_refusal_reports_the_conditioning_and_the_floor",
    (not finished_a)
    and ratio is not None
    and ratio < FLOOR
    # A factor of ten either way. The ratio is scale invariant, so only the
    # rounding of the stored apex height moves it, and this window is
    # wide enough to survive any vertex ordering the solver picks.
    and 0.1 * expected < ratio < 10.0 * expected
    and ("below the " + FLOOR_TEXT) in msg_a
    and "rest edge matrix (row-major): [" in msg_a,
    {"ratio": ratio, "expected": expected, "message": msg_a[-700:]},
)

# Only the refusal itself is checked for ASCII: the lines around it carry a
# workspace path this scenario does not own.
marker = "Degenerate tetrahedron"
tail = msg_a[msg_a.index(marker):] if marker in msg_a else ""
record(
    "D_refusal_message_is_pure_ascii",
    bool(tail) and tail.isascii(),
    {"non_ascii": [hex(ord(c)) for c in tail if ord(c) > 127][:8],
     "message": tail[:700]},
)

# ----- E: flat geometry is not degenerate geometry ---------------------
finished_e, msg_e = attempt("thin_sound", 1.0e-02)
record("E_thin_but_sound_tet_still_builds", finished_e,
       {"finished": finished_e, "message": msg_e[-400:]})

# ----- F: the earlier gate still owns the exactly-zero case ------------
finished_f, msg_f = attempt("exactly_coplanar", 0.0)
record(
    "F_exactly_coplanar_tet_refused_by_the_earlier_gate",
    (not finished_f) and "volume is zero" in msg_f,
    {"finished": finished_f, "message": msg_f[-400:]},
)

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    # No `PPF_STEP_DELAY_MS`. Only one case here runs frames, none of them
    # measures time, and nothing watches a run in progress, so a per-step delay
    # would be pure sweep cost. The knob defaults to zero, so the absence is
    # also the default.
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX],
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
    return r.report_named_checks(cases, label="tet rest-shape cases")
