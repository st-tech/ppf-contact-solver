# File: scenarios/rig_solver_log_format.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The shape of the solver's STREAMED log.
#
# `crates/ppf-cts-solver/src/kernels/simplelog/SimpleLog.cpp` spells the four
# output shapes that are the whole vocabulary a reader sees go by during a
# solve:
#
#   `message(fmt, ...)`   the text, with NO timestamp
#   `mark(name, value)`   `* <name>: %d` when `fmodf(value, 1.0) == 0`,
#                         `* <name>: %.3e` otherwise
#   `pop()`               `> <name>...<N> msec`
#   ctor / dtor           `====== <name> ======`, `===== <name>: <N> msec =====`
#
# `driver/log.rs` does both halves of that from one call: `mark` appends a row
# to its `.out` stream AND prints the line. Only the printing half is visible to
# someone watching a run, and a `mark` that wrote its file and returned would
# leave the `.out` streams complete and the transcript down to three lines a
# step, two of them content-free banners. Measured on `drape`, a step prints
# about twenty-eight mark and phase lines. That is why the checks below read
# stdout rather than the streams.
#
# WHY A SCENARIO AND NOT A UNIT TEST. `log.rs`'s own tests pin the two number
# formats, and they cannot see any of what is checked here: whether a call site
# reaches the printer at all, whether the phases close in the order the step
# runs them, whether a line that should carry no timestamp acquired one from the
# `log4rs` pattern, or whether a demoted diagnostic is still reachable. Those
# are properties of a RUN, and only a run shows them.
#
# THE TWO RUNS ARE THE POINT OF THE DEMOTION CHECK. Four diagnostics sit at
# `debug!` rather than `info!`, so that the default transcript carries the four
# shapes above and nothing else. That is a demotion only if something can
# still ask for them: with the level hardcoded the call would be unreachable in
# the shipped binary and the fixtures asserting on those lines would have no
# way to switch them back on. So this runs the same scene twice, once at the
# default level and once with `RUST_LOG=debug`, and requires the lines to be
# absent from the first and present in the second.

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# No Blender. It needs a solver that actually runs a step, so it is real-backend
# only: the log shape is a property of the driver, which every backend shares,
# so whichever one the host has answers the question.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend import App


# A hinge of two triangles with three corners pinned, which is the smallest
# scene that runs a real Newton step: it assembles, solves, line-searches and
# commits, so every phase the solve times is entered.
ARM, WIDTH, TOP = 0.1, 0.1, 1.0
VERTS = np.array(
    [
        [-ARM, TOP, 0.5 * WIDTH],
        [0.0, TOP, 0.0],
        [0.0, TOP, WIDTH],
        [ARM, TOP, 0.5 * WIDTH],
    ],
    dtype=np.float64,
)
FACES = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)


def solve(tag):
    """Run two frames and return the solver's own stdout as a list of lines."""
    app = App.create(f"rig-log-format-{tag}")
    app.asset.add.tri("strip", VERTS, FACES)
    scene = app.scene.create(f"rig-log-format-{tag}")
    strip = scene.add("strip")
    strip.param.set("model", "arap").set("young-mod", 1000.0).set("bend", 2500.0)
    strip.pin([0, 1, 2]).move_by([0.0, 0.0, 0.0], t_start=0.0, t_end=1.0)
    session = app.session.create(scene.build())
    session.param.set("dt", 0.5 / 60.0).set("fps", 60.0).set("frames", 2)
    session = session.build()
    session.start(blocking=True)
    return list(session.get.log.stdout(n_lines=100000)), session


lines, session = solve("default")
cases = {}


def record(name, ok, detail):
    cases[name] = {"ok": bool(ok), **detail}


# ---- the four shapes ---------------------------------------------------
# `> name...N msec`, exactly as `SimpleLog::pop` spells it.
PHASE = re.compile(r"^> ([a-z_]+)\.\.\.(\d+) msec$")
# `* name: value`. The two value forms are checked apart, below.
MARK = re.compile(r"^\* ([A-Za-z_][A-Za-z0-9_ -]*): (.+)$")
# C's `%e` with a signed exponent of at least two digits. Rust's own `{:.3e}`
# writes `5.220e-9` and would fail this.
#
# TWO MANTISSA WIDTHS ARE CORRECT, AND ONE RUN EMITS BOTH. `mark` spells a
# value `%.3e`, while the handful of message lines that spell a `toi` use
# `%.2e`. A single transcript can carry `* toi_advanced: 1.00e+00` from the
# second and `* toi_advanced: 1` from the first three lines apart, so a pattern
# admitting only `%.3e` reports a correct log as malformed.
C_EXPONENTIAL = re.compile(r"^-?\d\.\d{2,3}e[-+]\d{2,}$")
INTEGER = re.compile(r"^-?\d+$")
HEADER = re.compile(r"^====== ([a-z_]+) ======$")
FOOTER = re.compile(r"^===== ([a-z_]+): (\d+) msec =====$")

phases = [PHASE.match(x) for x in lines]
phases = [m for m in phases if m]
marks = [MARK.match(x) for x in lines]
marks = [m for m in marks if m]

record(
    "A_phase_timings_are_printed",
    len(phases) > 0,
    {"count": len(phases), "sample": [m.group(0) for m in phases[:6]]},
)
record(
    "B_marks_are_printed",
    len(marks) > 0,
    {"count": len(marks), "sample": [m.group(0) for m in marks[:6]]},
)

# A whole number takes the integer form and a fractional one takes C's `%.3e`.
# Both must occur, or one of the two branches is untested by this run.
integer_marks = [m.group(0) for m in marks if INTEGER.match(m.group(2))]
exponential_marks = [m.group(0) for m in marks if C_EXPONENTIAL.match(m.group(2))]
malformed = [
    m.group(0)
    for m in marks
    if not INTEGER.match(m.group(2)) and not C_EXPONENTIAL.match(m.group(2))
]
record(
    "C_a_whole_mark_takes_the_integer_form",
    len(integer_marks) > 0,
    {"count": len(integer_marks), "sample": integer_marks[:4]},
)
record(
    "D_a_fractional_mark_takes_c_s_exponential_form",
    len(exponential_marks) > 0,
    {"count": len(exponential_marks), "sample": exponential_marks[:4]},
)
record(
    "E_no_mark_takes_a_third_form",
    not malformed,
    {"malformed": malformed[:6]},
)

# Every step is framed, and the footer names the same scope as the header.
headers = [HEADER.match(x) for x in lines]
headers = [m for m in headers if m]
footers = [FOOTER.match(x) for x in lines]
footers = [m for m in footers if m]
record(
    "F_each_scope_is_framed_by_a_header_and_a_footer",
    len(headers) > 0 and len(headers) == len(footers)
    and [m.group(1) for m in headers] == [m.group(1) for m in footers],
    {
        "headers": [m.group(1) for m in headers],
        "footers": [m.group(1) for m in footers],
    },
)
record(
    "G_the_step_scope_is_named_advance",
    "advance" in [m.group(1) for m in headers],
    {"scopes": sorted({m.group(1) for m in headers})},
)

# THE TIMESTAMP CONVENTION. The SimpleLog-shaped lines print unprefixed, while
# the Rust `info!` lines go through `log4rs` under `[%Y-%m-%d %H:%M:%S]`. The
# two conventions share one stdout stream, so what is checked is that the
# SimpleLog-shaped lines stayed unprefixed rather than acquiring the pattern.
TIMESTAMPED_SHAPE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] (\*|>|=====|======) ")
timestamped = [x for x in lines if TIMESTAMPED_SHAPE.match(x)]
record(
    "H_marks_and_phases_carry_no_timestamp",
    not timestamped,
    {"offenders": timestamped[:6]},
)

# THE ORDER WITHIN A STEP. A Newton step closes contact assembly, then matrix
# assembly, then the linear solve, then the line search, and a reader takes that
# order off the transcript as the shape of the step.
order = [m.group(1) for m in phases]
def first(name):
    return order.index(name) if name in order else -1
ok_order = (
    first("asm_contact") >= 0
    and first("matrix_assembly") > first("asm_contact")
    and first("linsolve") > first("matrix_assembly")
    and first("line_search") > first("linsolve")
)
record(
    "I_phases_close_in_the_reference_order",
    ok_order,
    {"order": order[:12]},
)

# THE PORT-ONLY LINES ARE ABSENT AT THE DEFAULT LEVEL.
PORT_ONLY = (
    "solver driver: log path",
    "face assembly dispatch passes",
    "collision-mesh broad phase has",
    "committed pose carries no intersecting pair",
)
text = "\n".join(lines)
present_by_default = [s for s in PORT_ONLY if s in text]
record(
    "J_port_only_lines_are_absent_at_the_default_level",
    not present_by_default,
    {"present": present_by_default},
)

# ...AND REACHABLE WITH RUST_LOG=debug, which is what makes it a demotion.
os.environ["RUST_LOG"] = "debug"
debug_lines, _ = solve("debug")
debug_text = "\n".join(debug_lines)
reachable = [s for s in PORT_ONLY if s in debug_text]
record(
    "K_a_demoted_line_is_reachable_with_rust_log_debug",
    len(reachable) > 0,
    {"reachable": reachable, "checked": list(PORT_ONLY)},
)

# THE LOG'S OWN .out FILES STILL CARRY TWO NUMBERS PER ROW. The printing sits
# beside the write rather than replacing it, and every reader of these streams
# takes a time column and a value column.
#
# ONLY THE STREAMS THE LOG WRITES ARE CHECKED, which is what the `scope.channel`
# shape selects. `data/` also holds files written directly by
# `crates/ppf-cts-solver/src/main.rs` that are not `mark` output and have their
# own shapes: `total_mass.out` carries the rod, area and volume totals as THREE
# columns. Requiring two columns everywhere reports that file as malformed.
data_dir = os.path.join(session.output.path, "data")
bad = []
seen = 0
skipped = []
if os.path.isdir(data_dir):
    for entry in sorted(os.listdir(data_dir)):
        if not entry.endswith(".out"):
            continue
        # `scope.channel.out` is a mark stream; `scope.out` is a section total.
        # Anything else in here belongs to a direct writer, not to the log.
        stem = entry[: -len(".out")]
        if stem.count(".") == 0 and stem not in ("advance", "initialize", "clock"):
            skipped.append(entry)
            continue
        seen += 1
        with open(os.path.join(data_dir, entry)) as handle:
            for row in handle:
                row = row.strip()
                if not row:
                    continue
                parts = row.split()
                if len(parts) != 2:
                    bad.append(f"{entry}: {row!r}")
                    break
                try:
                    float(parts[0]); float(parts[1])
                except ValueError:
                    bad.append(f"{entry}: {row!r}")
                    break
record(
    "L_the_log_streams_still_carry_two_numbers_per_row",
    seen > 0 and not bad,
    {"files": seen, "bad": bad[:6], "not_log_streams": skipped[:6]},
)

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    # The probe sets RUST_LOG itself for its second run, so it must not start
    # from an inherited one: a developer with RUST_LOG=debug in their shell
    # would otherwise see subtest J fail for a reason that is theirs and not
    # the code's.
    env.pop("RUST_LOG", None)
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
    return r.report_named_checks(cases, label="solver log format")
