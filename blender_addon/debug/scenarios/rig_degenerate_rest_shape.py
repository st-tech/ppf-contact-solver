# File: scenarios/rig_degenerate_rest_shape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The solver's build-time rest-shape gate, `builder::invert_rest_or_panic2`,
# and the two properties that decide whether its refusal reaches a reader.
#
# A shell face's rest matrix is inverted once at scene build and the elastic
# Hessian is quadratic in that inverse, so a near-collinear face reaches the
# device carrying entries fp32 cannot represent. The linear solve then reports
# `p^T A p is not-a-number at iter 0` and names no geometry at all, which is
# issue #144: the reporter could say only that a build failed, and the mesh
# ran once they triangulated it by hand.
#
# A gate that tests exact singularity cannot see this at all. A near-collinear
# rest triangle is entirely finite,
# `try_inverse` succeeds on it, and the inverse it returns is finite too, so
# neither a finiteness test nor a `None` from `try_inverse` can see it. The test
# is the singular-value ratio against `REST_SHAPE_MIN_CONDITION`, which is
# `sqrt(f32::EPSILON)`: the point where squaring the inverse consumes every
# digit fp32 has.
#
# The scene is the smallest thing that reproduces it, and it is the shape the
# report came from: a quad split along its 0-2 diagonal, with vertex 1 sitting a
# height `h` above the straight line from vertex 0 to vertex 2. Face (0, 1, 2)
# is then a sliver whose conditioning is `0.4 * h`, so `h` alone moves a case
# across the threshold with nothing else about the mesh changing. That is what
# makes the refused and accepted cases comparable: they differ in one number.
#
# HOW THE REFUSAL TRAVELS, and why two of the subtests are about the text.
# `builder::build` runs inside the solver binary, so the refusal is a Rust
# panic on that subprocess's stderr. `frontend/_session_.py` reads the file
# back and hands it to `analyze_solver_error`
# (`crates/ppf-cts-core/src/datamodel/session/log.rs`), which returns the line
# holding `panicked at` plus three lines either side, blanks dropped. That is
# the whole diagnostic the artist ever sees, and it imposes two constraints
# neither a Rust unit test nor a UTF-8 Linux terminal can observe:
#
#   * PURE ASCII. `builder::ascii_matrix2` spells the matrix out as
#     `[a, b; c, d]` rather than deferring to nalgebra's `Display`, which draws
#     the matrix inside box-drawing characters (U+250C and its family). The
#     ANSI codepage a Windows console and a default-encoded Python reader use
#     has no encoding for those, so a non-ASCII diagnostic makes the consumer
#     fail ON the panic text: what surfaces is a `'charmap' codec can't
#     encode character` failure naming U+250C, and the geometry that caused
#     the refusal is never named. The platform that loses the message is the
#     one where reproducing the artist's mesh is hardest, and a Linux run of
#     this scenario is green either way, so the property is asserted rather
#     than read off a passing run.
#   * ONE LINE. The default panic hook spends the three lines after
#     `panicked at` on the message's first line, the message's second line, and
#     the `RUST_BACKTRACE` note. The matrix is that second line and sits at the
#     edge of the window, so it survives only while the whole matrix renders on
#     a single line. Any multi-line rendering is cut mid-matrix and the label
#     arrives with nothing behind it.
#
# NO BLENDER AND NO GPU. The gate is host-side Rust in the solver driver, shared
# by both backends, so this runs anywhere the server runs and holds on
# the real-GPU jobs unchanged. The addon-side gate that refuses the same
# geometry one step earlier, where the object still has a name, is
# `bl_degenerate_tessellation_rejection`. Both gates have to grant the same set.
#
# Subtests:
#   A. near_collinear_face_refused
#         h = 3.75e-07, the conditioning of the worst face of the reporter's
#         own mesh. The run must be refused.
#   B. refusal_names_the_face_and_the_conditioning
#         The message has to carry the face index, the three vertices and the
#         measured ratio. Without them the artist is back to the symptom the
#         issue was filed on, which named nothing.
#   C. thin_but_sound_face_still_builds
#         h = 2.5e-03, a ratio of 1e-03, an aspect ratio of about 800:1. Thin
#         geometry is legitimate and must still run; this is the cost of
#         setting the gate too tight, so it is asserted rather than assumed.
#   D. exactly_collinear_face_refused_by_the_earlier_gate
#         h = 0. This one never reaches the rest-shape gate: `main.rs` asserts
#         `triutils::face_areas` over the CURRENT positions first and aborts
#         with `area is zero`. Recorded so the division of labor between the
#         two aborts is visible, and so a change that moves it is noticed.
#   E. refusal_message_is_pure_ascii
#         `str.isascii()` over the refusal A produced. One box-drawing
#         character anywhere in it costs the whole diagnostic on the consumer
#         that most needs it.
#   F. matrix_survives_the_analysis_context_window
#         The `rest tangent matrix (row-major)` label and a bracketed two-row
#         matrix, on the SAME line. Both halves are needed: the label alone
#         arrives even from a rendering the context window cut, and the row is
#         what proves the numbers came with it.

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


# The label `invert_rest_or_panic2` prints ahead of the matrix, and the shape
# `ascii_matrix2` renders it in: two comma-separated rows, semicolon between
# them, every entry in scientific notation. The character classes exclude the
# newline, so a match proves the whole matrix is on ONE line.
MATRIX_LABEL = "rest tangent matrix (row-major)"
MATRIX_ROW = re.compile(r"\[[-+0-9.eE, ]+;[-+0-9.eE, ]+\]")


def quad_split_on_the_bad_diagonal(h):
    """A quad split along its 0-2 diagonal, vertex 1 at height `h` over the
    line from vertex 0 to vertex 2.

    Face (0, 1, 2) has singular values sqrt(5) and 2h/sqrt(5), so its
    conditioning is 0.4 * h and h is the only thing that moves it. Face
    (0, 2, 3) is the other half of the same quad and stays well formed
    throughout, so a refusal can only have come from the first.
    """
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, h, 0.0],
                  [2.0, 0.0, 0.0],
                  [1.0, -1.0, 0.0]], dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return V, F


def attempt(tag, h):
    """Run one frame and return (finished, message)."""
    app = App.create("rig_degenerate_rest_shape_" + tag)
    V, F = quad_split_on_the_bad_diagonal(h)
    app.asset.add.tri("patch", V, F)
    scene = app.scene.create()
    scene.add("patch")
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


def labeled_line(msg):
    """The one line of `msg` carrying the matrix label, or an empty string."""
    for line in msg.splitlines():
        if MATRIX_LABEL in line:
            return line
    return ""


cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


# ----- A / B: the reported case ---------------------------------------
finished_a, msg_a = attempt("near_collinear", 3.75e-07)
record("A_near_collinear_face_refused", not finished_a,
       {"finished": finished_a, "message": msg_a[-400:]})
record(
    "B_refusal_names_the_face_and_the_conditioning",
    (not finished_a)
    and "Degenerate shell face 0" in msg_a
    and "0, 1, 2" in msg_a
    and "conditioned at" in msg_a
    and "Triangulate" in msg_a,
    {"message": msg_a[-600:]},
)

# ----- C: thin geometry is not degenerate geometry ---------------------
finished_c, msg_c = attempt("thin_sound", 2.5e-03)
record("C_thin_but_sound_face_still_builds", finished_c,
       {"finished": finished_c, "message": msg_c[-400:]})

# ----- D: the earlier gate still owns the exactly-zero case ------------
finished_d, msg_d = attempt("exactly_collinear", 0.0)
record(
    "D_exactly_collinear_face_refused_by_the_earlier_gate",
    (not finished_d) and "area is zero" in msg_d,
    {"finished": finished_d, "message": msg_d[-400:]},
)

# ----- E / F: the refusal has to survive the trip to a reader ----------
# Both read the message A already produced. They are about how that one
# message renders, so neither needs a second run.
non_ascii = sorted({c for c in msg_a if not c.isascii()})
record(
    "E_refusal_message_is_pure_ascii",
    (not finished_a) and bool(msg_a) and msg_a.isascii(),
    {
        "non_ascii": [f"U+{ord(c):04X}" for c in non_ascii[:8]],
        "message": msg_a[-600:],
    },
)

label_line = labeled_line(msg_a)
row_match = MATRIX_ROW.search(label_line.split(MATRIX_LABEL, 1)[-1])
record(
    "F_matrix_survives_the_analysis_context_window",
    bool(label_line) and row_match is not None,
    {
        "label_present": bool(label_line),
        "label_line": label_line[-300:],
        "row": row_match.group(0) if row_match is not None else None,
        "message": msg_a[-600:],
    },
)

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
    return r.report_named_checks(cases, label="rest-shape cases")
