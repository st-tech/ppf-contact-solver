# File: tools/run_suite.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Run the example notebooks headlessly on whichever backend the host carries,
# and judge each run by the frames it produced and the motion in them.
#
# WHY THE VERDICT IS NOT THE EXIT CODE. A notebook's exit status is not evidence
# that it simulated. `plastic.ipynb` has no `assert session.finished()` and no
# blocking start, so a NewtonStall leaves it exiting 0; `friction.ipynb` has the
# assertion and still exited 0 after the solver aborted partway with a vertex
# moving 32 units in one frame. A stale solver process anywhere on the box
# produces the same false success, because `Utils.busy()` scans EVERY process
# for "ppf-contact" and the frontend then refuses to start while the script
# exits 0. So this harness reads the OUTPUT: how many frames landed, against how
# many the notebook asked for, and how far the mesh actually moved between them.
#
# The frozen-animation test is the reason the motion column exists. A backend
# that writes a full frame set without moving anything passes every check that
# only counts files.
#
# THE SECOND VERDICT: EXECUTION SHAPE. Frames and displacement are VALUE
# instruments, and a value instrument can be green while a backend computes the
# right answer the wrong way. The Metal backend accepted `precond =
# "block-jacobi"`, the default every example runs with, and applied nothing: its
# unreduced PCG ran plain conjugate gradient. Over a complete `drape` run that
# cost 2.9x the linear-solve iterations, and the trajectory still landed within
# 0.25% of CUDA's, so the displacement columns above reported a clean pass. The
# shape gate below reads the two per-step indicator streams the solver already
# writes and compares HOW MUCH WORK the run did against a measured CUDA
# reference. See `shape_profile` for the streams and `compare_shape` for the
# band and its justification.

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# TWO ROOTS, BECAUSE THE HARNESS AND THE TREE IT JUDGES ARE NOT ALWAYS THE SAME
# TREE. `HERE` is this harness's own data, the execution-shape reference and the
# self-test fixtures, which travel with the script. `ROOT` is the tree under
# test: the one holding `frontend/` and `examples/`, which the notebooks are read
# from and which every notebook process is given as its PYTHONPATH and working
# directory. In a checkout both are the same repository. Against a release
# distribution they are not, because no distribution ships this harness: it then
# runs from a repository against an unpacked archive, and `--root` is what says
# which tree the verdict is about.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PY = Path.home() / ".local/share/ppf-cts/venv/bin/python"


def notebooks(skip_large: bool = True):
    """The suite, in a stable order. `large-*` is excluded by the goal.

    THE SUITE IS WHAT `examples/` HOLDS, AND THAT IS THE WHOLE RULE. A scene
    too big for a first-step check that the `large-` prefix does not name is
    not filtered here: it lives in the private benchmark tree, which this glob
    does not reach, beside a README saying what each one measures.
    """
    out = sorted(p for p in (ROOT / "examples").glob("*.ipynb"))
    if skip_large:
        out = [p for p in out if not p.name.startswith("large-")]
    return out


def declared_frames(script: Path) -> int | None:
    """The `frames` the notebook asks for, read off the converted script.

    Best effort and allowed to fail: it only sharpens the verdict from "wrote
    N frames" to "wrote N of M". A run is never failed for want of this number.
    """
    try:
        text = script.read_text()
    except OSError:
        return None
    hits = re.findall(r'\bframes\s*=\s*(\d+)', text)
    if not hits:
        hits = re.findall(r'["\']frames["\']\s*[:,]\s*(\d+)', text)
    return max(int(h) for h in hits) if hits else None


def convert(nb: Path, script: Path) -> str | None:
    """Write the notebook's code cells to a plain script.

    Deliberately NOT `jupyter nbconvert`. The Mac has no outbound network, so a
    venv missing jupyter cannot gain it, and the suite would be unrunnable on
    the one host that can exercise Metal. What nbconvert adds over concatenating
    the code cells is cell markers, magics and shell escapes; measured across
    all 24 notebooks in this suite there are ZERO magics and ZERO shell escapes,
    so for these inputs the two are equivalent.

    Re-check that if the suite ever gains a notebook: a `%` or `!` line would
    reach the interpreter as a syntax error rather than being translated, so it
    is refused here by name instead of failing three frames later.
    """
    try:
        doc = json.loads(nb.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable notebook: {exc}"
    out = []
    for i, cell in enumerate(doc.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                return (f"cell {i} uses a notebook magic or shell escape "
                        f"({stripped[:40]!r}); this converter handles neither")
        out.append(f"# ---- cell {i} ----\n{src}\n")
    if not out:
        return "no code cells"
    script.write_text("\n".join(out))
    return None


def make_fast_check(script: Path) -> bool:
    """Rewrite the script to run ONE frame, the way `warmup.py fast_check` does.

    `App.set_fast_check()` forces `frames = 1` into the exported `param.toml`
    through the Rust `to_toml_string` override, so the solver runs a single
    step. The injection point is the `App.create(...)` or `App.load(...)` call,
    matching `warmup.py`'s own substitution exactly rather than inventing a
    second spelling of the same idea.

    WHAT THIS DOES AND DOES NOT BUY. It does not shrink what a backend must
    IMPLEMENT: one frame still needs the BVH, contact detection and assembly,
    the linear solve, the CCD line search, the commit and the intersection gate,
    which is the whole driver. What it buys is RUNTIME, and on a backend far
    slower than CUDA that is the difference between a suite that can be run and
    one that cannot: 24 notebooks at one frame each instead of at 100 to 850.

    It is also a weaker check, and the weakness is worth naming rather than
    forgetting. A single frame exercises scene build, initialize and one step;
    it cannot show a drift that accumulates, a contact set that degrades as a
    scene settles, or the frozen-animation failure this suite exists to catch,
    since one step has no second frame to compare against. Use it to establish
    that every example RUNS, and a full run to establish that one is RIGHT.
    """
    try:
        text = script.read_text()
    except OSError:
        return False
    patched, n = re.subn(r"(app = App\.(?:create|load)\([^)]+\))",
                         r"\1; App.set_fast_check()", text)
    if n == 0:
        return False
    script.write_text(patched)
    return True


def make_frame_cap(script: Path, frames: int) -> bool:
    """Rewrite the script to run exactly `frames` frames, keeping every other parameter.

    SEPARATE FROM `make_fast_check` BECAUSE IT ANSWERS A DIFFERENT QUESTION, and
    the difference is the whole reason this exists. Fast check forces `frames =
    1` through the solver's own export override and is a liveness probe: it says
    a notebook RUNS. This one leaves the export alone and edits the notebook's
    own `frames` parameter, so the run is an ordinary run of a shorter scene,
    which is what a MEASUREMENT needs.

    WHY A SHORT MEASURED RUN IS WORTH HAVING AT ALL. A chaotic scene amplifies
    any difference until its own run-to-run spread swamps the difference under
    test, so a full-length statistic on one cannot separate two builds. Early
    frames are before the amplification: `domino` repeats to about two parts per
    million at frame 5, while by frames 10, 25 and 50 one build's own spread is
    13.9%, 368% and 1115%. So a cross-build comparison at a low frame count is
    decisive where tens of whole-run outcomes are not.

    The substitution targets the `frames` parameter as the examples set it,
    `.set("frames", N)`. Returns False when the script has none, so a caller
    that asked for a bounded run is told rather than silently given the full
    scene, which is the same failure mode `make_fast_check` refuses.
    """
    try:
        text = script.read_text()
    except OSError:
        return False
    patched, n = re.subn(r'(\.set\(\s*"frames"\s*,\s*)\d+(\s*\))',
                         lambda m: f'{m.group(1)}{frames}{m.group(2)}', text)
    if n == 0:
        return False
    script.write_text(patched)
    return True


def simulates(script: Path) -> bool:
    """Does this notebook actually run the solver?

    Not every example is a simulation. `walkthrough.ipynb` builds meshes and
    plots them and never creates a session, so it writes no frames and no
    `assert session.finished()` applies to it. Judging it by frame count would
    report a correct notebook as a failed one, which is the kind of false red
    that teaches a reader to ignore the suite.
    """
    try:
        text = script.read_text()
    except OSError:
        return True  # cannot tell; judge it as a simulation rather than excuse it
    # A frame-stepped notebook starts its solver through the stepping calls,
    # which launch it when it is not running.
    return any(call in text for call in
               ("session.start(", "session.run_until_frame(", "session.step_frame("))


def app_name(script: Path) -> str | None:
    """The name the notebook passes to `App.create`, read off the script.

    This is the session's directory leaf, and reading it is what makes the
    lookup EXACT. It cannot be derived from the notebook filename:
    `trapped-919539a.ipynb` creates `trapped2`.
    """
    try:
        text = script.read_text()
    except OSError:
        return None
    m = re.search(r'App\.create\(\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else None


def _freshest_frame_mtime(output: Path) -> float:
    """The newest `vert_*.bin` mtime in this directory, or 0.0 if it holds none.

    THE DIRECTORY'S OWN MTIME IS THE WRONG CLOCK and reading it produced a false
    negative. A directory's mtime moves when an entry is added or removed, not
    when an existing entry is overwritten, so a run that lands on a session
    directory already holding a full frame set of the same names rewrites every
    file in place and leaves the directory stamped with the earlier run's time.
    The frames are what the verdict is taken on, so the frames are what dates it.
    """
    newest = 0.0
    try:
        for f in output.glob("vert_*.bin"):
            try:
                newest = max(newest, f.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return 0.0
    return newest


def find_session(data_root: Path, newer_than: float, name: str | None,
                 claimed: dict, stem: str) -> Path | None:
    """This notebook's `session/output`, resolved by NAME and then by RECENCY.

    The verdict is taken on the output rather than on the exit code, so
    attributing the wrong directory to a run is not a reporting nuisance: it
    decides the result. Both directions have been observed here and the guards
    below are one per direction.

    A FALSE PASS came from resolving by "newest directory" alone: a notebook that
    failed before creating its session was credited with the previous scene's
    output and reported as a clean run belonging to a different scene. The app
    name from `App.create` is therefore read off the script and the lookup is
    exact, and a directory already attributed to an earlier notebook in this
    suite is refused outright.

    A FALSE FAILURE came from taking the FIRST name match. The data root holds
    one subdirectory per branch (`git-<branch>`, plus `git-unknown` for a tree
    with no git), so one app name can exist under several of them, and an old
    branch's copy sorts ahead of the live one. Taking the first match found a
    stale directory, judged it older than this run, and reported a run that had
    just written a full frame set as producing nothing. Every name match is
    therefore considered and the freshest wins.

    `claimed` maps a directory to the notebook stem that took it, rather than
    merely holding it. A repeat of the SAME notebook (`--repeat`, which measures
    a backend's own run-to-run envelope) legitimately lands on the directory its
    previous repeat used, and refusing it there would report the second repeat
    as producing nothing.
    """
    if not data_root.is_dir():
        return None

    pattern = f"*/{name}/session/output" if name else "*/*/session/output"

    best, best_mtime = None, newer_than
    try:
        candidates = list(data_root.glob(pattern))
    except OSError:
        return None
    for out_dir in candidates:
        if claimed.get(out_dir) not in (None, stem):
            continue
        mtime = _freshest_frame_mtime(out_dir)
        if mtime >= best_mtime:
            best, best_mtime = out_dir, mtime
    # A named lookup that matched nothing fresh is a real miss: this run produced
    # no frames of its own, and saying so is the point.
    return best


def read_frames(output: Path):
    """Load every `vert_N.bin` in frame order.

    float32, 3 per vertex. Read as float64 the count halves and the coordinates
    come out huge, which resembles an explosion rather than a parse error, so
    the dtype is not negotiable. Frames are sorted NUMERICALLY: lexicographic
    order puts vert_10 before vert_2 and fabricates motion that is really just
    frames out of sequence.
    """
    files = []
    for p in output.glob("vert_*.bin"):
        m = re.match(r"vert_(\d+)\.bin$", p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    frames = []
    for _, p in files:
        a = np.fromfile(p, dtype=np.float32)
        if a.size % 3:
            continue
        frames.append(a.reshape(-1, 3))
    return frames


def motion(frames):
    """Per-frame mean vertex displacement, and the summary the verdict uses.

    `disp_mean` over every vertex is the statistic to compare, not `disp_max`:
    the max is one vertex and moves several percent between runs of one build,
    while a real change moves both the same way.
    """
    if len(frames) < 2:
        return {"steps": 0, "disp_mean": 0.0, "disp_max": 0.0, "moving_frames": 0}
    per = []
    for a, b in zip(frames, frames[1:]):
        if a.shape != b.shape:
            break
        per.append(float(np.linalg.norm(b - a, axis=1).mean()))
    if not per:
        return {"steps": 0, "disp_mean": 0.0, "disp_max": 0.0, "moving_frames": 0}
    per = np.array(per)
    # Cumulative displacement from frame 0, sampled at FIXED indices. Two
    # properties make this the number to compare across backends, and the
    # whole-run mean has neither.
    #
    # It survives a TRUNCATED run: a backend that got 164 of 200 frames still
    # produced frame 25, and comparing its frame 25 against the reference's
    # frame 25 is like for like, where two means over different frame counts are
    # not. And it is taken BEFORE the amplification on a chaotic scene, where an
    # outcome statistic over a whole run is nearly useless: `domino` repeats to
    # about 2 parts per million at frame 5 while one build's own spread reaches
    # 368% by frame 25 and 1115% by frame 50.
    early = {}
    for k in (5, 10, 25, 50):
        if len(frames) > k:
            d = np.linalg.norm(frames[k] - frames[0], axis=1)
            early[f"disp_at_{k}"] = float(d.mean())
    return {
        "steps": len(per),
        "disp_mean": float(per.mean()),
        "disp_max": float(per.max()),
        # A frozen run is not "mean is small": a scene can settle and legitimately
        # stop. It is "almost no frame moved at all", so the count is reported and
        # the verdict is taken on it.
        "moving_frames": int((per > 1e-9).sum()),
        **early,
    }


# ===========================================================================
# The execution-shape gate
# ===========================================================================
#
# WHAT IT COMPARES. Two per-step indicator streams that every backend already
# writes, into `<session>/output/data/` and NOT into `<session>/output/` beside
# the frames:
#
#   advance.newton_steps.out   one record per advance() call, so one per
#                              TOI-limited substep. The value is the Newton
#                              iteration count that substep consumed.
#   advance.iter.out           one record per PCG solve, so one per Newton
#                              iteration. The value is the iteration count that
#                              linear solve consumed.
#
# One driver marks both, for every backend
# (crates/ppf-cts-solver/src/driver/step.rs), so the two streams have the same
# shape whichever backend produced them. Both files are
# `<simulated time> <value>` per line, appended a row at a time by `log::mark`
# (crates/ppf-cts-solver/src/driver/log.rs). One advance() writes all of
# its `iter` records at once, so they all carry that substep's time stamp, and
# the count of distinct contiguous time stamps in `iter` equals the record count
# of `newton_steps`. That pairing is reported as a consistency figure.
#
# WHY THE TWO STREAMS ARE JUDGED DIFFERENTLY. `newton_steps` is an integer the
# shared termination rule decides: `step` starts at 1 and the loop cannot exit
# below `min_newton_steps`, a field of the shared `ParamSet`, both in
# `crates/ppf-cts-solver/src/driver/step.rs`. Its FLOOR is therefore exact
# across backends and no fp32 difference can move it. How often
# a substep needs more than the floor is decided by `toi_advanced`, which is
# fp32, so the rest of that stream is spread and is banded. `iter` is banded
# throughout: it moves with fp32 association order and with the per-process mesh
# edge permutation (crates/ppf-cts-solver/src/mesh.rs:188).

ITER_STREAM = "advance.iter.out"
NEWTON_STREAM = "advance.newton_steps.out"

# Bin width for the windowed comparison a truncated run falls back to. 0.25 s is
# the width a hand comparison of the `plastic` cost already localizes it to, so
# a hand comparison and this gate bin the same way.
SHAPE_WINDOW_S = 0.25

# A run counts as covering the reference when its last recorded time reaches
# this fraction of the reference's. A complete run OVERSHOOTS the last frame
# time by up to one substep: measured on three complete `drape` runs the last
# stamps are 1.660189, 1.663067 and 1.665546 s, a spread of 0.32%. 0.99 leaves
# 3x that.
SHAPE_COVERAGE_FULL = 0.99

# And the mirror of it. The whole-run COUNT checks compare two totals, so they
# mean something only when both sides cover the same simulated time: a run
# twice as long as its reference honestly does about twice the iterations, and
# reading that as a finding is a statement about the two run LENGTHS rather
# than about the solver. The reference file is not guaranteed to hold a
# complete run of every scene, so this direction is reachable in ordinary use
# and was measured: against references recorded at 31 frames, full runs of
# `belt`, `cards` and `domino` reported `iter.count` ratios of 6.5, 5.8 and 8.1,
# each equal to that scene's own frames-run over frames-recorded, while the two
# scenes whose references were full length passed. Symmetric by construction,
# so the window either side may be short by is the same 1%.
SHAPE_COVERAGE_OVER = 1.0 / SHAPE_COVERAGE_FULL

# Two complete shared bins is the least a windowed comparison may speak on.
SHAPE_MIN_WINDOWS = 2

# THE BAND, AND EVERY DIGIT OF IT.
#
# A banded statistic passes when `got / want` lies in [1/BAND, BAND]. BAND is
# 2.0, constructed as the geometric midpoint of two measured numbers:
#
#   NOISE, the widest run-to-run spread this project has recorded for cg
#   iterations: 40%, a ratio of 1.40 (single runs swing 10 to 40%). Measured
#   on `drape` specifically it is far narrower. Four complete runs of correct
#   builds, on two backends and two hosts, read mean
#   318.5 / 320.6 / 322.8 / 323.08 (ratio 1.014) and median 369 / 370 / 371 /
#   372 (ratio 1.008); `examples/headless.py` is recorded stable to about 1%
#   over 18 runs. So 1.40 is a ceiling drawn from the whole project rather than
#   from this scene.
#
#   DEFECT, the weakest statistic of the divergence this gate exists to catch.
#   A CUDA build with the block-Jacobi `invert()` forced to the identity
#   degenerates PCG to the plain conjugate gradient Metal was running, and on
#   `drape` reads mean 936.36 against a clean 323.08 (2.8983x) and median 1008.5
#   against 371 (2.7183x). The unpreconditioned Metal run reads 2.9152x and
#   2.8208x on the same two. 2.7183 is the weakest of the four.
#
#   sqrt(1.40 * 2.7183) = 1.9508, rounded to 2.0. That leaves 1.4286x of margin
#   above the widest recorded noise and 1.3592x below the weakest statistic of
#   the known defect.
#
# WHAT IT CAN AND CANNOT SEE. It sees any change that doubles or halves the
# solver work a scene needs, which is the size of the defect above. It is BLIND
# to anything smaller: a backend needing 1.5x the PCG iterations passes, and so
# does one needing 1.9x. Changing the band is a measurement job, not a judgement
# call: run `--repeat` on the scene, which records that backend's own envelope
# beside the reference, then set `band` on that scene's entry from what the
# envelope measured, together with a `band_source` naming the measurement.
# `compare_shape` prefers a per-scene band over this default and REFUSES one
# that arrives without a `band_source`, in either direction, because a band
# widened with no stated evidence is how a gate stops seeing.
SHAPE_BAND = 2.0

SHAPE_REFERENCE = HERE / "execution_shape_reference.json"

# Exact: an inequality is a defect, not spread.
SHAPE_EXACT_CHECKS = (("newton_steps", "min"),)

# Banded at SHAPE_BAND. The two COUNT entries are the noisiest of the five and
# set the floor on any tightening: over four complete correct `drape` runs
# `iter.count` reads 768 / 790 / 793 / 802, a spread of 1.0443, against 1.014 on
# `iter.mean` and 1.008 on `iter.median`. So a per-scene band below about 1.05
# would fire on the counts while the means were still quiet, and one band cannot
# be set from the mean's envelope alone.
SHAPE_BANDED_CHECKS = (
    ("iter", "mean"),
    ("iter", "median"),
    ("iter", "count"),
    ("newton_steps", "mean"),
    ("newton_steps", "count"),
)

# Reported, never gated, because their measured run-to-run spread is wide enough
# that gating them would fire on noise: `iter.max` spans 502 to 584 over four
# complete correct `drape` runs (16%), and section 8.3 records `plastic`'s p90
# moving 3.4x and its max 2x between two runs of ONE build.
SHAPE_REPORTED_ONLY = (
    ("iter", "min"),
    ("iter", "p90"),
    ("iter", "max"),
    ("newton_steps", "max"),
)


class ShapeError(RuntimeError):
    """A stream that exists but cannot be read as one.

    Raised rather than returned, and never swallowed into a default: a stream
    this harness cannot parse is a stream whose verdict it must not render.
    """


def read_stream(path: Path):
    """Return the LAST run's `(time, value)` rows, and how many runs are stacked.

    THE APPEND TRAP. Every stream is opened in APPEND mode
    (`crates/ppf-cts-solver/src/driver/log.rs`), so a session directory reused by
    a second run holds both runs concatenated, and a naive read averages them
    together. Simulated
    time is non-decreasing within one run and restarts at the next, so a
    decrease is exactly a run boundary. The last segment is this run's.

    The one case that escapes it is a resume whose first time is at or above the
    previous run's last, which appends with no decrease to find. The stacked
    count is reported for every stream so that case is visible as a record count
    that does not match the scene.
    """
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ShapeError(
                f"{path}:{lineno}: expected '<time> <value>', got {line!r}")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ShapeError(f"{path}:{lineno}: {exc}") from None
    segments, prev = [[]], None
    for t, v in rows:
        if prev is not None and t < prev:
            segments.append([])
        segments[-1].append((t, v))
        prev = t
    return segments[-1], len(segments)


def _quantile(ordered, frac):
    """The `frac` quantile of an already-sorted list, by index."""
    return ordered[min(len(ordered) - 1, int(frac * len(ordered)))]


def stream_stats(rows, window: float = SHAPE_WINDOW_S):
    """Summarize one stream, plus its per-window means.

    The windows exist so a TRUNCATED run is still comparable. A whole-run mean
    averages whichever substeps happened to land, and a run that stopped early
    covers a different interval of the scene, not a noisier sample of the same
    one: section 8.3 records `belt` reading a displacement ratio of 0.91 cut off
    at 164 of 201 frames and 1.036 run to completion. Binning both sides on a
    fixed grid of simulated time compares like for like.
    """
    values = [v for _, v in rows]
    times = [t for t, _ in rows]
    ordered = sorted(values)
    integral = all(float(v).is_integer() for v in values)
    counts = Counter(int(v) for v in values) if integral else None
    bins: dict = {}
    for t, v in rows:
        bins.setdefault(int(math.floor(t / window)), []).append(v)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(ordered, 0.9),
        "min": ordered[0],
        "max": ordered[-1],
        "sum": sum(values),
        "t_first": times[0],
        "t_last": times[-1],
        "integral": integral,
        # Only for a stream that is integral and narrow. A wide histogram is
        # noise in a diff, and `iter` is wide by nature.
        "histogram": ({str(k): counts[k] for k in sorted(counts)}
                      if counts is not None and len(counts) <= 16 else None),
        "window_s": window,
        "windows": [{"bin": k, "t_start": round(k * window, 6), "n": len(vs),
                     "mean": statistics.fmean(vs)}
                    for k, vs in sorted(bins.items())],
    }


def _time_groups(rows) -> int:
    """Runs of equal time stamps, which is one per advance() call."""
    groups, prev = 0, None
    for t, _ in rows:
        if prev is None or t != prev:
            groups += 1
        prev = t
    return groups


def shape_profile(data_dir: Path, vertices: int = 0, frames: int = 0) -> dict:
    """Read both indicator streams out of `<session>/output/data/`.

    `vertices` and `frames` travel with the profile because a reference number
    is only comparable against a run of the SAME PROBLEM. fTetWild is not
    deterministic, so a cached tetrahedralization and a fresh one are different
    meshes and a solid scene's vertex count differs between hosts with unequal
    asset caches (measured: `roller` 6300 against 6238, `plastic` 5465 against
    5654). A vertex-count mismatch is reported so a mesh difference is never
    read as a solver difference. `drape` has no tetrahedralization, so its count
    must match exactly.
    """
    profile: dict = {"data_dir": str(data_dir), "vertices": int(vertices),
                     "frames": int(frames), "streams": {}, "missing": [],
                     "error": None}
    for key, fname in (("iter", ITER_STREAM), ("newton_steps", NEWTON_STREAM)):
        path = data_dir / fname
        if not path.is_file():
            profile["missing"].append(fname)
            continue
        try:
            rows, stacked = read_stream(path)
        except (OSError, ShapeError) as exc:
            profile["error"] = str(exc)
            return profile
        if not rows:
            profile["missing"].append(f"{fname} (empty)")
            continue
        stats = stream_stats(rows)
        stats["appended_runs"] = stacked
        if key == "iter":
            stats["time_groups"] = _time_groups(rows)
        profile["streams"][key] = stats
    return _pair_streams(profile)


def _pair_streams(profile: dict) -> dict:
    """Cross the two streams against each other, which they should agree.

    One `newton_steps` record per advance(), and that advance's `iter` records
    all carry its time stamp, so these two counts are equal on a healthy run.
    They differ by one when the solve that never returned is the last thing the
    run did, which is where a `cg-max-iter` abort lands: that difference of one
    places such an abort inside the linear solve rather than in assembly.
    Reported, not gated, since the run that produces the difference has already
    failed loudly.
    """
    both = profile["streams"]
    if "iter" in both and "newton_steps" in both and both["newton_steps"]["count"]:
        profile["advances_from_iter"] = both["iter"]["time_groups"]
        profile["advances_from_newton"] = both["newton_steps"]["count"]
        profile["solves_per_advance"] = (
            both["iter"]["count"] / both["newton_steps"]["count"])
    return profile


def _band_check(name, got, want, band):
    ratio = None
    ok = False
    if want in (None, 0) or got is None:
        detail = "reference value is missing or zero"
    else:
        ratio = got / want
        ok = (1.0 / band) <= ratio <= band
        detail = ""
    return {"name": name, "kind": "band", "got": got, "want": want,
            "ratio": ratio, "band": band, "ok": ok, "detail": detail}


def _exact_check(name, got, want):
    return {"name": name, "kind": "exact", "got": got, "want": want,
            "ratio": None, "ok": got == want, "detail": ""}


def compare_shape(profile: dict, reference: dict, band: float = SHAPE_BAND) -> dict:
    """Render a verdict for one run against one measured reference entry.

    Verdicts, all of them explicit and none of them a fallback:

      pass                every gated check inside its rule, over a run that
                          covered the reference's simulated time.
      out-of-band         at least one gated check outside its rule. This is the
                          finding the gate exists for.
      partial             the run and the reference cover different simulated
                          time, in either direction. The comparison then runs on
                          the fixed time windows both sides cover, and `partial`
                          is reported whether or not those windows agree,
                          because a comparison over part of a scene is not
                          evidence that the whole scene matches. `checks` still
                          says which windows disagreed.
      insufficient-window the run stopped so early that fewer than
                          SHAPE_MIN_WINDOWS complete bins are shared.
      no-reference        nobody has measured this scene on CUDA yet.
      no-streams          the run wrote no indicator streams.
      unreadable          a stream exists and could not be parsed.
    """
    out = {"verdict": None, "band": band, "checks": [], "notes": [],
           "reported": {}}
    if profile.get("error"):
        out["verdict"] = "unreadable"
        out["notes"].append(profile["error"])
        return out
    if reference is None:
        out["verdict"] = "no-reference"
        out["notes"].append(
            "no measured CUDA reference for this scene; produce one on a CUDA "
            "host with: run_suite.py --backend cuda --only <scene> "
            "--record-reference")
        return out
    if "iter" not in profile["streams"] or "newton_steps" not in profile["streams"]:
        out["verdict"] = "no-streams"
        out["notes"].append(
            "missing " + ", ".join(profile["missing"] or ["both streams"])
            + f" under {profile['data_dir']}")
        return out

    ref_streams = reference.get("streams", {})
    if "iter" not in ref_streams or "newton_steps" not in ref_streams:
        out["verdict"] = "no-reference"
        out["notes"].append("the reference entry carries no stream statistics")
        return out

    # A scene may carry its own band, set from an envelope measured on that
    # scene. It must also carry the measurement it came from: a band is the one
    # number in this gate that decides what counts as a finding, so one arriving
    # with no stated evidence is refused and the default is used instead. That
    # is a failing check rather than a note, in both directions, because a band
    # widened without evidence is how a gate stops seeing.
    scene_band = reference.get("band")
    if scene_band is not None:
        justified = bool(str(reference.get("band_source", "")).strip())
        out["checks"].append({
            "name": "band.justified", "kind": "exact",
            "got": f"band {scene_band}, band_source "
                   f"{'present' if justified else 'MISSING'}",
            "want": "band_source present", "ratio": None, "ok": justified,
            "detail": "a per-scene band must record the measurement that set "
                      "it, for instance the --repeat envelope on that scene",
        })
        if justified:
            band = float(scene_band)
            out["band"] = band
            out["notes"].append(
                f"per-scene band {band}: {reference['band_source']}")

    # AND A SCENE MAY BAND ONE STATISTIC APART FROM THE REST, keyed
    # `stream.stat`, for the case a single band cannot express: a scene whose
    # per-SOLVE iteration statistics are unusable while its run TOTALS are
    # stable to a percent. Measured on `domino`, whose `iter.mean` spans 6.79x
    # over four runs of one tree while its `iter.count` holds to 1.0072x;
    # widening the whole scene to cover the mean would have thrown the count
    # check away with it. Same evidence rule as the scene band, and the same
    # refusal: `band_source` justifies both, because an unjustified per-statistic
    # band is the more dangerous of the two, being narrower and easier to slip in.
    per_stat = reference.get("band_by_statistic") or {}
    if per_stat and not bool(str(reference.get("band_source", "")).strip()):
        out["checks"].append({
            "name": "band_by_statistic.justified", "kind": "exact",
            "got": f"{len(per_stat)} per-statistic band(s), band_source MISSING",
            "want": "band_source present", "ratio": None, "ok": False,
            "detail": "a per-statistic band must record the measurement that "
                      "set it, exactly as a per-scene band must",
        })
        per_stat = {}
    if per_stat:
        out["band_by_statistic"] = dict(per_stat)
        out["notes"].append(
            "per-statistic bands: "
            + ", ".join(f"{k} {float(v)}" for k, v in sorted(per_stat.items())))

    # The statistics whose measured run-to-run spread is too wide to gate. They
    # travel in the verdict so a reader can see them move without the gate
    # claiming they mean anything on their own.
    for stream, stat in SHAPE_REPORTED_ONLY:
        got = profile["streams"][stream].get(stat)
        want = ref_streams[stream].get(stat)
        out["reported"][f"{stream}.{stat}"] = {
            "got": got, "want": want,
            "ratio": (got / want) if (want not in (None, 0)
                                      and got is not None) else None,
        }

    # The oracle must be CUDA. A reference recorded on any other backend would
    # let a backend certify itself, so it is a failing check rather than a note.
    ref_backend = reference.get("backend", "unknown")
    out["checks"].append({
        "name": "reference.backend", "kind": "exact", "got": ref_backend,
        "want": "cuda", "ratio": None, "ok": ref_backend == "cuda",
        "detail": "CUDA is the reference implementation and the only oracle "
                  "every parity gate is defined against",
    })

    ref_verts = reference.get("vertices", 0)
    got_verts = profile.get("vertices", 0)
    if ref_verts and got_verts:
        out["checks"].append({
            "name": "vertices", "kind": "exact", "got": got_verts,
            "want": ref_verts, "ratio": None, "ok": got_verts == ref_verts,
            "detail": "a different vertex count is a different problem; on a "
                      "tetrahedralized scene copy the reference host's "
                      "~/.cache/ppf-cts rather than reading this as a solver "
                      "difference",
        })
    elif ref_verts:
        out["notes"].append(
            f"vertex count not observed, so this comparison cannot exclude a "
            f"mesh difference against the reference's {ref_verts}")

    for stream, stat in SHAPE_EXACT_CHECKS:
        out["checks"].append(_exact_check(
            f"{stream}.{stat}",
            profile["streams"][stream][stat], ref_streams[stream].get(stat)))

    ref_last = ref_streams["iter"].get("t_last") or 0.0
    got_last = profile["streams"]["iter"]["t_last"]
    coverage = (got_last / ref_last) if ref_last else 0.0
    out["coverage"] = coverage

    if SHAPE_COVERAGE_FULL <= coverage <= SHAPE_COVERAGE_OVER:
        for stream, stat in SHAPE_BANDED_CHECKS:
            name = f"{stream}.{stat}"
            out["checks"].append(_band_check(
                name, profile["streams"][stream][stat],
                ref_streams[stream].get(stat),
                float(per_stat.get(name, band))))
        out["verdict"] = "pass" if all(c["ok"] for c in out["checks"]) \
            else "out-of-band"
        return out

    # The two cover different simulated time, in EITHER direction. Compare
    # per-window means over the bins both sides COMPLETE, so a bin either side
    # stopped inside is dropped rather than averaged.
    #
    # THE WINDOWED PATH DISCRIMINATES AS WELL AS THE WHOLE-RUN ONE, measured
    # across the three checked-in `drape` fixtures: every one of the seven
    # 0.25 s bins separates the clean run from both defective ones by 2.44x to
    # 3.64x on the `iter` mean, so a run truncated anywhere past two bins still
    # reads the defect.
    #
    # The per-window RECORD COUNTS are carried in the profile and are
    # deliberately not checked. They are the noisiest quantity in the set: over
    # those same three runs they span 1.000 to 1.259 bin by bin, on bins holding
    # as few as 17 records, against 1.014 for the whole-run `iter.mean`. The
    # count dimension is covered by the whole-run `iter.count` and
    # `newton_steps.count` checks, and the discriminating signal in a truncated
    # run is the window means.
    shorter = "the run" if coverage < 1.0 else "the reference"
    out["notes"].append(
        f"run covered {coverage:.3f} of the reference's simulated time "
        f"({got_last:.4f} s against {ref_last:.4f} s), so {shorter} is the "
        f"shorter; the whole-run count checks would compare two totals taken "
        f"over different simulated time, and the comparison is made on the "
        f"{SHAPE_WINDOW_S} s windows both sides complete instead")
    for stream in ("iter", "newton_steps"):
        ref_bins = {w["bin"]: w for w in ref_streams[stream].get("windows", [])}
        got_bins = {w["bin"]: w for w in profile["streams"][stream]["windows"]}
        shared = sorted(set(ref_bins) & set(got_bins))
        window_s = ref_streams[stream].get("window_s", SHAPE_WINDOW_S)
        # COMPLETE ON BOTH SIDES. `got_last` alone is the right test only
        # when the reference is the longer of the two; when it is the shorter
        # its own last bin is the partial one.
        covered = min(got_last, ref_last)
        complete = [b for b in shared if (b + 1) * window_s <= covered]
        if len(complete) < SHAPE_MIN_WINDOWS:
            out["verdict"] = "insufficient-window"
            out["notes"].append(
                f"{stream}: {len(complete)} complete shared windows, "
                f"{SHAPE_MIN_WINDOWS} needed")
            return out
        for b in complete:
            out["checks"].append(_band_check(
                f"{stream}.window[{ref_bins[b]['t_start']:.2f}].mean",
                got_bins[b]["mean"], ref_bins[b]["mean"], band))
    out["verdict"] = "partial"
    out["failing_windows"] = [c["name"] for c in out["checks"] if not c["ok"]]
    return out


# How a verdict maps onto the exit code, by gate mode.
#
#   off       every verdict is printed and none of them decides anything.
#   measured  (default) a verdict about a scene the gate COULD judge decides the
#             exit code. `no-reference` and `insufficient-window` do not,
#             because they are this repository's debt rather than a finding
#             about the backend under test, and failing on them would make the
#             suite unrunnable until all 24 scenes are measured. They are
#             counted and named in the summary instead.
#   strict    everything but `pass` fails. This is the mode for a CI leg on a
#             host whose scenes are all measured, where an unmeasured scene is
#             itself the thing to fix.
#
# `no-streams` fails in `measured` as well as in `strict`. A backend that ran a
# scene and wrote no indicator stream is not an unmeasured scene, it is a
# backend with no per-step telemetry, which is a divergence in its own right.
SHAPE_GATE_MODES = ("off", "measured", "strict")


def shape_fails(verdict: dict, mode: str) -> bool:
    """Does this verdict fail the run, under this gate mode?"""
    if mode == "off":
        return False
    name = verdict["verdict"]
    if name == "pass":
        return False
    if mode == "strict":
        return True
    if name in ("no-reference", "insufficient-window"):
        return False
    if name == "partial":
        # The run stopped early, which the frame verdict already reports. The
        # shape gate speaks only about the windows both sides completed.
        return bool(verdict.get("failing_windows"))
    return True


def load_reference(path: Path) -> dict:
    """Load the reference file, or fail by name.

    A missing or malformed reference is not a reason to skip the gate quietly:
    the caller asked for a comparison and there is nothing to compare against,
    which is a state worth naming.
    """
    if not path.is_file():
        raise ShapeError(f"reference file not found: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ShapeError(f"reference file {path}: {exc}") from None
    if doc.get("schema") != 1:
        raise ShapeError(
            f"reference file {path}: schema {doc.get('schema')!r}, expected 1")
    return doc


def record_reference(path: Path, scene: str, backend: str, profile: dict,
                     meta: dict) -> None:
    """Write this run's shape into the reference file, under `scene`.

    Only ever called for a run this harness judged OK. A run that truncated,
    froze or crashed must not become the oracle, and the caller enforces that.
    """
    doc = load_reference(path) if path.is_file() else {
        "schema": 1, "band": SHAPE_BAND, "window_s": SHAPE_WINDOW_S,
        "scenes": {},
    }
    entry = {"backend": backend, "status": "measured"}
    entry.update(meta)
    entry["vertices"] = profile.get("vertices", 0)
    entry["frames"] = profile.get("frames", 0)
    entry["streams"] = profile["streams"]
    for key in ("advances_from_iter", "advances_from_newton",
                "solves_per_advance"):
        if key in profile:
            entry[key] = profile[key]
    doc.setdefault("scenes", {})[scene] = entry
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")


def envelope(profiles) -> dict:
    """This backend's OWN run-to-run spread, over repeats of one scene.

    A single run proves nothing about this solver: fp32 atomics and tree
    reductions make it non-deterministic run to run, and the recorded
    cg-iteration swing reaches 40%. So a band is only honest once the backend's
    own envelope has been measured on the scene in question, and `--repeat` is
    how this harness measures it. The ratio column
    is max/min, which is directly comparable against SHAPE_BAND.
    """
    out: dict = {"runs": len(profiles), "statistics": {}}
    for stream in ("iter", "newton_steps"):
        for stat in ("count", "mean", "median", "p90", "min", "max"):
            vals = [p["streams"][stream][stat] for p in profiles
                    if stream in p.get("streams", {})]
            if len(vals) < 2:
                continue
            lo, hi = min(vals), max(vals)
            out["statistics"][f"{stream}.{stat}"] = {
                "values": vals, "min": lo, "max": hi,
                "ratio": (hi / lo) if lo else None,
            }
    return out


def format_shape(verdict: dict) -> str:
    """One line for the console: the verdict plus whatever failed.

    A failed EXACT check is printed ahead of any banded one however small its
    numbers look. An inequality there is a statement about shared code, not
    about spread, so it is the one a reader should see first.
    """
    bad = [c for c in verdict["checks"] if not c["ok"]]
    head = f"shape={verdict['verdict']}"
    if not bad:
        return head
    ordered = sorted(bad, key=lambda c: (c["kind"] != "exact",
                                         -(c["ratio"] or 0.0)))[:3]
    parts = []
    for c in ordered:
        if c["kind"] == "exact" or c["ratio"] is None:
            parts.append(f"{c['name']} {c['got']}!={c['want']}")
        else:
            parts.append(f"{c['name']} {c['ratio']:.2f}x")
    return head + " [" + ", ".join(parts) + "]"


SHAPE_FIXTURES = HERE / "execution_shape_fixtures"


def profile_from_files(iter_path: Path, newton_path: Path, vertices: int = 0,
                       frames: int = 0) -> dict:
    """A profile built from two named stream files rather than a session.

    This is what lets the gate be exercised on a machine with no GPU: the two
    streams are the whole input, and where they came from does not matter to the
    comparator.
    """
    profile: dict = {"data_dir": str(iter_path.parent), "vertices": vertices,
                     "frames": frames, "streams": {}, "missing": [],
                     "error": None}
    for key, path in (("iter", iter_path), ("newton_steps", newton_path)):
        if not path.is_file():
            profile["missing"].append(path.name)
            continue
        rows, stacked = read_stream(path)
        stats = stream_stats(rows)
        stats["appended_runs"] = stacked
        if key == "iter":
            stats["time_groups"] = _time_groups(rows)
        profile["streams"][key] = stats
    return _pair_streams(profile)


def _summary_profile(count, mean, median, t_last, newton) -> dict:
    """A profile carrying only the statistics a band check reads.

    For a published run whose raw streams this repository does not hold. The
    exact and reported-only checks have nothing to read here, so this is a band
    check and nothing more, which is exactly what it claims to be.
    """
    return {
        "data_dir": "(published summary)", "vertices": 0, "frames": 0,
        "missing": [], "error": None,
        "streams": {
            "iter": {"count": count, "mean": mean, "median": median,
                     "p90": None, "min": None, "max": None, "sum": None,
                     "t_first": 0.0, "t_last": t_last, "integral": True,
                     "histogram": None, "window_s": SHAPE_WINDOW_S,
                     "windows": []},
            "newton_steps": newton,
        },
    }


def self_test() -> int:
    """THE GATE THAT PROVES THE GATE. Runs anywhere: no GPU, no Metal device,
    no built solver, no frontend venv. Measured at 0.06 s on a machine with
    neither backend, which is what makes it affordable on every build.

    Every case below is measured data, not a construction.
    `execution_shape_fixtures/` beside this file holds three complete 101-frame `drape` runs,
    captured during the divergence audit and checked in beside this harness:

      drape.cuda-clean               the reference build
      drape.cuda-identity-precond    the same build with the block-Jacobi
                                     `invert()` forced to return the identity,
                                     which degenerates PCG to plain conjugate
                                     gradient
      drape.metal-unpreconditioned   the Metal backend before its preconditioner
                                     was wired, which ran that same recurrence

    Their whole-run `advance.iter.out` means are 323.08, 936.36 and 941.84, so
    a degenerate preconditioner reads about 2.9x the clean build on this scene.

    HOW TO REPRODUCE THE FAULT INJECTION. The inversion is a NEUTRAL BODY,
    `block_jacobi_invert` in
    `crates/ppf-cts-solver/src/kernels/solver/block_jacobi.kernel.cpp`, so one
    edit reaches CUDA, Metal and the CPU backend alike rather than one driver's
    copy. Its successful path ends in exactly one `    result.inverse = minv;`
    followed by `    return result;` (the other assignment of that field is the
    invalid path above it, which writes a zero block and returns early). In a
    SCRATCH tree and never in a live one:

        S=crates/ppf-cts-solver/src/kernels/solver/block_jacobi.kernel.cpp
        cp $S $S.orig
        python3 -c "p='$S'; s=open(p).read(); \\
          old='    minv(2, 1) = minv(1, 2);\\n    result.inverse = minv;'; \\
          assert s.count(old)==1; \\
          open(p,'w').write(s.replace(old, '    minv(2, 1) = minv(1, 2);\\n' \\
            '    result.inverse = Mat3x3f::Identity();  // FAULT INJECTION'))"
        cargo build --release
        python tools/run_suite.py --backend cuda-identity --only drape \\
            --shape-gate strict

    That run must report `shape=out-of-band` with `iter.mean` near 2.9x and exit
    1. Restore with `cp $S.orig $S && rm $S.orig` and rebuild, then re-run
    without the patch: it must report `shape=pass` and exit 0.

    THE STALE-OBJECT TRAP IS NOT GONE, IT MOVED. This edit is under a tree both
    build scripts watch, so cargo re-runs them and the rendering is regenerated;
    what a CUDA build still skips is a `.cu` whose object is newer than its
    source. If a build finishes in about 20 s rather than the usual minute and a
    half, nothing recompiled, and the binary is still running the old code.

    The same replay works without rebuilding anything, from the streams the two
    builds already produced:

        python tools/run_suite.py --only drape --shape-from \\
            <a directory holding advance.iter.out and advance.newton_steps.out>

    Case K below runs exactly that over all three fixtures and asserts the exit
    codes 0, 1, 1.

    LEAVING A FAULT-INJECTED SOLVER IN `target/release/` IS A LIVE HAZARD, which
    is why the injection belongs in a scratch tree.
    """
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'ok  ' if condition else 'FAIL'} {name}"
              + (f"   {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    def fixture(label):
        return profile_from_files(
            SHAPE_FIXTURES / f"drape.{label}.advance.iter.out",
            SHAPE_FIXTURES / f"drape.{label}.advance.newton_steps.out",
            vertices=81920, frames=101)

    print("execution-shape self-test")
    print(f"  fixtures: {SHAPE_FIXTURES}")
    print(f"  reference: {SHAPE_REFERENCE}")

    doc = load_reference(SHAPE_REFERENCE)
    ref = doc.get("scenes", {}).get("drape")
    if ref is None:
        print("  FAIL the reference file carries no `drape` entry")
        return 1

    clean = fixture("cuda-clean")
    identity = fixture("cuda-identity-precond")
    metal = fixture("metal-unpreconditioned")

    print("\nA. the checked-in reference is the checked-in evidence")
    # Without this the reference could drift away from the streams it claims to
    # summarize, and every verdict below would be measured against a number
    # nobody can re-derive.
    for stream in ("iter", "newton_steps"):
        for stat in ("count", "mean", "median", "min", "max", "t_last"):
            got = clean["streams"][stream][stat]
            want = ref["streams"][stream][stat]
            same = (abs(got - want) <= 1e-9 * max(1.0, abs(want))
                    if isinstance(got, float) else got == want)
            check(f"reference.{stream}.{stat}", same, f"{got} vs {want}")

    print("\nB. a correct build passes, on runs the reference did not come from")
    # Two further CUDA runs and one Metal run, all with the preconditioner
    # wired, recorded after the fix. Only their whole-run count, mean and median
    # were recorded, so the newton_steps half of each verdict here is handed the
    # reference's own numbers and is vacuous. What case B tests is the `iter` band against three
    # correct runs the reference was NOT derived from, which is the separation
    # case A cannot supply.
    published = {
        "cuda run 1 (published)": (790, 320.6, 372.0),
        "cuda run 2 (published)": (802, 318.5, 369.0),
        "metal preconditioned (published)": (793, 322.8, 370.0),
    }
    for label, (count, mean, median) in published.items():
        prof = _summary_profile(count, mean, median,
                                ref["streams"]["iter"]["t_last"],
                                clean["streams"]["newton_steps"])
        verdict = compare_shape(prof, ref)
        check(label, verdict["verdict"] == "pass", format_shape(verdict))

    print("\nC. the defect the gate exists for fails it")
    for label, prof, want_ratio in (
            ("cuda, invert() -> identity", identity, 2.8983),
            ("metal, unpreconditioned", metal, 2.9152)):
        verdict = compare_shape(prof, ref)
        check(f"{label}: verdict", verdict["verdict"] == "out-of-band",
              format_shape(verdict))
        mean_check = next(c for c in verdict["checks"]
                          if c["name"] == "iter.mean")
        check(f"{label}: iter.mean ratio",
              abs(mean_check["ratio"] - want_ratio) < 0.01,
              f"{mean_check['ratio']:.4f}x against a band of {SHAPE_BAND}")

    print("\nD. newton_steps alone would NOT have caught it")
    # Worth asserting rather than assuming. The defect sits inside the linear
    # solve, and the Newton loop terminates on `toi_advanced`, so both defective
    # runs agree with the reference on every newton_steps check: floor 1 on all
    # three, and means 1.5772 / 1.5993 / 1.5770. The two streams answer
    # different questions and the gate needs both.
    for label, prof in (("identity", identity), ("metal", metal)):
        verdict = compare_shape(prof, ref)
        newton = [c for c in verdict["checks"]
                  if c["name"].startswith("newton_steps.")]
        check(f"{label}: every newton_steps check passes",
              bool(newton) and all(c["ok"] for c in newton),
              ", ".join(f"{c['name']}={c['got']}" for c in newton))

    print("\nE. a stacked stream is split, not averaged")
    # The streams are appended to, so a reused session directory holds both
    # runs.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        stacked = Path(tmp) / "advance.iter.out"
        stacked.write_text(
            (SHAPE_FIXTURES / "drape.cuda-clean.advance.iter.out").read_text()
            + (SHAPE_FIXTURES
               / "drape.cuda-identity-precond.advance.iter.out").read_text())
        rows, segments = read_stream(stacked)
        stats = stream_stats(rows)
        check("two runs are seen as two", segments == 2, f"segments={segments}")
        check("the last run is the one read",
              abs(stats["mean"] - identity["streams"]["iter"]["mean"]) < 1e-9,
              f"mean={stats['mean']:.4f}")
        naive = statistics.fmean(
            [float(l.split()[1]) for l in stacked.read_text().splitlines() if l.strip()])
        check("the naive read would have been wrong",
              abs(naive - stats["mean"]) > 100.0,
              f"naive={naive:.1f} against {stats['mean']:.1f}")

    print("\nF. a run with no streams is named, never passed")
    empty = {"data_dir": "(none)", "vertices": 0, "frames": 0, "streams": {},
             "missing": [ITER_STREAM, NEWTON_STREAM], "error": None}
    verdict = compare_shape(empty, ref)
    check("no-streams verdict", verdict["verdict"] == "no-streams",
          verdict["verdict"])
    verdict = compare_shape(clean, None)
    check("no-reference verdict", verdict["verdict"] == "no-reference",
          verdict["verdict"])

    print("\nG. a truncated run is compared on the windows it completed")
    def truncate(profile_src, fraction):
        label = profile_src
        rows_i, _ = read_stream(
            SHAPE_FIXTURES / f"drape.{label}.advance.iter.out")
        rows_n, _ = read_stream(
            SHAPE_FIXTURES / f"drape.{label}.advance.newton_steps.out")
        cut = rows_i[-1][0] * fraction
        keep_i = [r for r in rows_i if r[0] <= cut]
        keep_n = [r for r in rows_n if r[0] <= cut]
        prof = {"data_dir": "(truncated)", "vertices": 81920,
                "frames": 0, "streams": {}, "missing": [], "error": None}
        for key, rows in (("iter", keep_i), ("newton_steps", keep_n)):
            st = stream_stats(rows)
            st["appended_runs"] = 1
            if key == "iter":
                st["time_groups"] = _time_groups(rows)
            prof["streams"][key] = st
        return prof

    cut_clean = truncate("cuda-clean", 0.6)
    verdict = compare_shape(cut_clean, ref)
    check("a correct truncated run reports partial",
          verdict["verdict"] == "partial", format_shape(verdict))
    check("and its shared windows all agree",
          not verdict.get("failing_windows"),
          str(verdict.get("failing_windows"))[:120])

    cut_identity = truncate("cuda-identity-precond", 0.6)
    verdict = compare_shape(cut_identity, ref)
    check("a defective truncated run still names its windows",
          bool(verdict.get("failing_windows")),
          f"{len(verdict.get('failing_windows') or [])} windows out of band")

    tiny = truncate("cuda-clean", 0.15)
    verdict = compare_shape(tiny, ref)
    check("too little coverage refuses to speak",
          verdict["verdict"] == "insufficient-window", verdict["verdict"])

    # THE MIRROR OF G, and the direction the reference file actually reaches:
    # a complete run against a reference recorded over less time. Before the
    # coverage test was made symmetric this produced a confident `out-of-band`
    # naming `iter.count`, on two identical builds, because the totals were
    # taken over different simulated time.
    print("\nG2. a truncated REFERENCE is compared the same way")
    short_ref = truncate("cuda-clean", 0.4)
    short_ref["backend"] = "cuda"
    short_ref["streams"]["iter"]["window_s"] = SHAPE_WINDOW_S
    short_ref["streams"]["newton_steps"]["window_s"] = SHAPE_WINDOW_S
    full_clean = fixture("cuda-clean")
    verdict = compare_shape(full_clean, short_ref)
    check("a correct run against a short reference is not a finding",
          verdict["verdict"] != "out-of-band", format_shape(verdict))
    check("it is reported as partial",
          verdict["verdict"] == "partial", verdict["verdict"])
    check("and the note names the reference as the shorter side",
          any("the reference is the shorter" in n for n in verdict["notes"]),
          "; ".join(verdict["notes"])[:160])
    check("its shared windows agree",
          not verdict.get("failing_windows"),
          str(verdict.get("failing_windows"))[:120])
    verdict = compare_shape(fixture("cuda-identity-precond"), short_ref)
    check("a DEFECTIVE run against a short reference still names its windows",
          bool(verdict.get("failing_windows")),
          f"{len(verdict.get('failing_windows') or [])} windows out of band")

    print("\nG3. a per-statistic band overrides the scene band, with evidence")
    # A scene whose per-solve statistics are unusable while its totals are
    # stable cannot be expressed by one number. This is the case `domino`
    # measured: iter.mean 6.79x over four runs of one tree, iter.count 1.0072x.
    wide = dict(ref)
    wide["band"] = 2.0
    wide["band_source"] = "self-test"
    wide["band_by_statistic"] = {"iter.mean": 12.0}
    inflated = fixture("cuda-clean")
    inflated["streams"] = json.loads(json.dumps(inflated["streams"]))
    inflated["streams"]["iter"]["mean"] = ref["streams"]["iter"]["mean"] * 5.0
    verdict = compare_shape(inflated, wide)
    check("a 5x mean passes under a 12x per-statistic band",
          all(c["ok"] for c in verdict["checks"] if c["name"] == "iter.mean"),
          format_shape(verdict))
    # AND THE REST OF THE SCENE STAYS AT ITS OWN BAND, which is the whole point:
    # widening the scene to cover the mean would have taken the count with it.
    inflated2 = fixture("cuda-clean")
    inflated2["streams"] = json.loads(json.dumps(inflated2["streams"]))
    inflated2["streams"]["iter"]["count"] = int(ref["streams"]["iter"]["count"] * 5)
    verdict = compare_shape(inflated2, wide)
    check("a 5x count still FAILS at the scene's 2.0 band",
          any(not c["ok"] for c in verdict["checks"] if c["name"] == "iter.count"),
          format_shape(verdict))
    # UNJUSTIFIED IS REFUSED, the same rule the scene band carries.
    naked = {k: v for k, v in ref.items()}
    naked.pop("band", None)
    naked.pop("band_source", None)
    naked["band_by_statistic"] = {"iter.mean": 12.0}
    verdict = compare_shape(fixture("cuda-identity-precond"), naked)
    check("a per-statistic band with no band_source is refused",
          any(c["name"] == "band_by_statistic.justified" and not c["ok"]
              for c in verdict["checks"]),
          format_shape(verdict))

    print("\nH. the band is the band")
    base = ref["streams"]["iter"]["mean"]
    check("a ratio of exactly the band passes",
          _band_check("x", base * SHAPE_BAND, base, SHAPE_BAND)["ok"])
    check("just past it fails",
          not _band_check("x", base * SHAPE_BAND * 1.0001, base,
                          SHAPE_BAND)["ok"])
    check("and the band is two-sided",
          not _band_check("x", base / (SHAPE_BAND * 1.0001), base,
                          SHAPE_BAND)["ok"])
    # A band is what decides whether anything is a finding, so one arriving in
    # the reference with no measurement behind it must not take effect quietly.
    unjustified = dict(ref)
    unjustified["band"] = 100.0
    unjustified.pop("band_source", None)
    verdict = compare_shape(identity, unjustified)
    check("an unjustified per-scene band is refused",
          verdict["band"] == SHAPE_BAND
          and any(c["name"] == "band.justified" and not c["ok"]
                  for c in verdict["checks"]), format_shape(verdict))
    check("and the defect it would have hidden still fails",
          verdict["verdict"] == "out-of-band", verdict["verdict"])
    justified = dict(unjustified)
    justified["band_source"] = "for the self-test only, not a measurement"
    verdict = compare_shape(identity, justified)
    check("a justified per-scene band takes effect", verdict["band"] == 100.0,
          format_shape(verdict))

    print("\nI. a non-CUDA reference cannot certify a backend")
    metal_ref = dict(ref)
    metal_ref["backend"] = "metal"
    verdict = compare_shape(clean, metal_ref)
    check("reference.backend is a failing check",
          verdict["verdict"] == "out-of-band"
          and any(c["name"] == "reference.backend" and not c["ok"]
                  for c in verdict["checks"]), format_shape(verdict))

    print("\nJ. a mesh difference is named as one")
    other_mesh = dict(clean)
    other_mesh["vertices"] = 81919
    verdict = compare_shape(other_mesh, ref)
    check("vertex count mismatch fails",
          any(c["name"] == "vertices" and not c["ok"]
              for c in verdict["checks"]), format_shape(verdict))

    print("\nK. end to end, through the reader and out to an exit code")
    # Everything above calls compare_shape directly. This runs the path a real
    # session takes: locate the streams under a `data` directory, read them off
    # disk, profile, compare, and return the status the shell sees.
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        for label, want_rc in (("cuda-clean", 0),
                               ("cuda-identity-precond", 1),
                               ("metal-unpreconditioned", 1)):
            data = Path(tmp) / label / "data"
            data.mkdir(parents=True)
            for stream in (ITER_STREAM, NEWTON_STREAM):
                (data / stream).write_bytes(
                    (SHAPE_FIXTURES / f"drape.{label}.{stream}").read_bytes())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = shape_from_dir(data, "drape", doc, SHAPE_BAND)
            check(f"--shape-from {label} exits {want_rc}", rc == want_rc,
                  f"exit {rc}")

    print("\nL. the envelope reports the spread it is handed")
    # `--repeat` is the only honest route to a tighter band, so the arithmetic
    # it rests on is checked here. These two profiles are different BUILDS, not
    # repeats of one, so this asserts the function's arithmetic and claims
    # nothing physical: max/min over the two iter means is the same 2.8983 the
    # ratio checks above read.
    env = envelope([clean, identity])
    check("envelope pairs both runs", env["runs"] == 2, str(env["runs"]))
    got = env["statistics"]["iter.mean"]["ratio"]
    check("envelope ratio is max/min", abs(got - 2.8983) < 1e-3, f"{got:.4f}x")

    print()
    if failures:
        print(f"self-test: {len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("self-test: all checks passed")
    return 0


def shape_from_dir(directory: Path, scene: str, reference: dict,
                   band: float) -> int:
    """Render a shape verdict for a session that already ran, and exit on it.

    A run leaves its streams on disk, so re-reading them costs nothing and does
    not need the scene re-simulated. Two uses: checking a session captured
    before this gate existed, and exercising the gate itself on a machine with
    no GPU, where the fixture directories under
    `tools/execution_shape_fixtures/` are not laid out as sessions but the
    comparator does not care where two streams came from.

    `directory` may be the session's `output/data`, or its `output`, or any
    directory holding the two `advance.*.out` files.
    """
    if (directory / ITER_STREAM).is_file():
        data_dir = directory
    elif (directory / "data" / ITER_STREAM).is_file():
        data_dir = directory / "data"
    else:
        print(f"no {ITER_STREAM} under {directory} or {directory / 'data'}")
        return 1

    # A session's frames sit one level above its streams, so the mesh identity
    # is recoverable when the directory is a real session and not otherwise.
    frames = read_frames(data_dir.parent) if data_dir.name == "data" else []
    profile = shape_profile(data_dir,
                            vertices=int(frames[0].shape[0]) if frames else 0,
                            frames=len(frames))
    verdict = compare_shape(profile, reference.get("scenes", {}).get(scene),
                            band)
    print(f"scene={scene}  {data_dir}")
    for stream in ("iter", "newton_steps"):
        st = profile["streams"].get(stream)
        if st is None:
            continue
        print(f"  {stream:13s} n={st['count']:5d} mean={st['mean']:9.4f} "
              f"median={st['median']:8.1f} p90={st['p90']:8.1f} "
              f"min={st['min']:.6g} max={st['max']:.6g} "
              f"t_last={st['t_last']:.6f} runs_stacked={st['appended_runs']}")
    if "solves_per_advance" in profile:
        print(f"  advances: {profile['advances_from_newton']} from newton_steps, "
              f"{profile['advances_from_iter']} from iter time stamps, "
              f"{profile['solves_per_advance']:.3f} solves each")
    print(f"  {format_shape(verdict)}")
    for note in verdict["notes"]:
        print(f"  note: {note}")
    for c in verdict["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        ratio = f"  ratio {c['ratio']:.4f}x" if c["ratio"] is not None else ""
        print(f"  {mark} {c['name']:36s} got {c['got']} want {c['want']}{ratio}")
    return 0 if verdict["verdict"] == "pass" else 1


def run_one(nb: Path, backend: str, data_root: Path, timeout: int, claimed: dict,
            fast: bool = False, frame_cap: int = 0):
    """Convert, run, then judge by output rather than by exit code."""
    work = Path(os.environ.get("PPF_SUITE_WORK", "/tmp/ppf-suite"))
    work.mkdir(parents=True, exist_ok=True)
    script = work / f"{nb.stem}.py"

    err = convert(nb, script)
    if err is not None:
        return {"name": nb.stem, "backend": backend, "stage": "convert",
                "ok": False, "error": err}

    if fast and not make_fast_check(script):
        return {"name": nb.stem, "backend": backend, "stage": "convert",
                "ok": False,
                "error": "--fast-check asked for, but no App.create/App.load call to "
                           "attach it to; running the full scene would silently ignore "
                           "the flag"}

    if frame_cap and not make_frame_cap(script, frame_cap):
        return {"name": nb.stem, "backend": backend, "stage": "convert",
                "ok": False,
                "error": f"--frames {frame_cap} asked for, but this notebook sets no "
                         f"frames parameter to bound; running the full scene would "
                         f"silently ignore the flag"}

    sim = simulates(script)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("PPF_CTS_HEADLESS", "1")

    started = time.time()
    # THE NOTEBOOK IS SPAWNED AS ITS OWN PROCESS GROUP, AND A TIMEOUT KILLS THE
    # GROUP. `subprocess.run(timeout=...)` kills only the notebook, and the
    # solver it launched is a GRANDCHILD that survives, keeps the GPU, and keeps
    # writing frames into the session this run was judged on. Measured on a Mac:
    # `cards` timed out at 3600 s, its solver ran on for another 90 minutes, and
    # the next scene's notebook waited behind it the whole time, so every later
    # row on that box was a row about the orphan. On a platform with no process
    # groups the kill falls back to the notebook alone.
    popen_kw = {"start_new_session": True} if hasattr(os, "killpg") else {}
    proc = subprocess.Popen([str(VENV_PY), str(script)], cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, **popen_kw)
    try:
        out, errs = proc.communicate(timeout=timeout)
        rc, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        out, errs = proc.communicate()
    tail = (out or "")[-1500:] + (errs or "")[-1500:]
    wall = time.time() - started

    session = find_session(data_root, started - 5.0, app_name(script), claimed,
                           nb.stem)
    if session is not None:
        claimed[session] = nb.stem
    frames = read_frames(session) if session else []
    stats = motion(frames)
    want = declared_frames(script)
    # A SOLVER ABORT IS A FAILURE WHATEVER THE FRAME COUNT SAYS. The
    # completeness clause below admits a run that reached 90 percent of its
    # declared frames, and `declared_frames` is best effort, so a run that
    # aborted in its last tenth, or one whose count could not be parsed, reads
    # as a pass on frames alone. Measured: `friction` aborted with
    # `### ccd failed` at frame 826 of 850 and still reported ok. The witness
    # that needs no declared count is the panic the driver prints into the
    # session's error log, so that is the verdict.
    aborted = False
    abort_reason = ""
    if session is not None:
        error_log = session.parent / "error.log"
        if error_log.exists() and "panicked" in error_log.read_text(errors="replace"):
            aborted, abort_reason = True, "driver panicked (see error.log)"
        if aborted:
            first = next((ln.strip() for ln in reversed(tail.splitlines())
                          if ln.strip().startswith("###") or "### " in ln), "")
            if first:
                abort_reason += ": " + first.split("### ", 1)[-1][:120]

    # The indicator streams live one directory below the frames. Reading them is
    # unconditional and cheap: the columns are informative even when no
    # reference exists to compare them against, and a scene with no streams is
    # something the report should say out loud rather than omit.
    profile = None
    if session is not None:
        profile = shape_profile(
            session / "data",
            vertices=int(frames[0].shape[0]) if frames else 0,
            frames=len(frames))

    # A backend that REFUSES a scene by name is not a failure of this suite. Both
    # the Metal and CPU backends are designed to refuse what they cannot solve at
    # `initialize()`, naming the feature and its count, precisely so a run never
    # produces a plausible trajectory computed for a different problem. Reporting
    # that as FAIL would make the suite unreadable on those backends, and worse,
    # would train a reader to skim past real failures next to expected ones.
    refused = any(
        marker in tail
        for marker in (
            "cannot solve this scene yet",
            "refuses it rather than writing frames",
            "the Metal backend implements only the single-level",
            "which the Metal backend cannot solve yet",
            "a preconditioner the Metal backend does not implement",
        )
    )
    if refused:
        reason = next(
            (ln.strip() for ln in tail.splitlines() if "refus" in ln or "cannot solve" in ln),
            "",
        )
        return {
            "name": nb.stem, "backend": backend, "stage": "refused", "ok": True,
            "rc": rc, "wall_s": round(wall, 1), "frames": len(frames),
            "reason": reason[:200],
        }

    # The verdict. Every clause is about the OUTPUT; the exit code is recorded
    # and deliberately not the criterion. A non-simulating notebook is judged on
    # the one thing that CAN fail for it: whether it ran to completion.
    #
    # A TIMEOUT IS NEVER OK, ON ANY BRANCH. The frames a killed run left behind
    # are a PREFIX and not a result, and the sim branch below cannot catch that
    # on its own: `declared_frames` is best effort by design, so a notebook
    # whose count it cannot parse leaves `want` at None, the completeness clause
    # short-circuits to True, and a run killed at 54 of 301 frames reports as a
    # pass. Measured on `curtain` and `trapped`, which both came back
    # `ok=True, timed_out=True, rc=None, frames=54/None`.
    if timed_out:
        ok = False
    elif not sim:
        ok = (rc == 0) and not timed_out
    else:
        if fast:
            # One frame means frame 0 and frame 1, so two files and one step,
            # and that step must have MOVED. The declared frame count is not
            # demanded, because the flag deliberately overrode it.
            ok = len(frames) >= 2 and stats["moving_frames"] >= 1
        else:
            ok = (
                len(frames) >= 2
                and stats["moving_frames"] >= max(1, stats["steps"] // 10)
                and (want is None or len(frames) >= 0.9 * want)
                and not aborted
            )
    return {
        "name": nb.stem, "backend": backend,
        "stage": "run" if sim else "run-nosim", "ok": ok,
        "rc": rc, "timed_out": timed_out, "wall_s": round(wall, 1),
        "aborted": aborted, "abort_reason": abort_reason,
        "frames": len(frames), "frames_wanted": want,
        "vertices": int(frames[0].shape[0]) if frames else 0,
        "session": str(session) if session else None,
        "shape_profile": profile,
        **stats,
        "tail": tail[-800:] if not ok else "",
    }


def main():
    # --python rebinds the interpreter for this run; run_one and the check below
    # read the module-level name, so it is replaced once, right after parsing.
    global VENV_PY, ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", help="label for the report; the key a recorded reference is stored under")
    ap.add_argument("--only", nargs="*", help="notebook stems; default is the whole suite")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first scene that does not pass, instead of "
                         "running the whole suite. A refusal by name is not a "
                         "failure and does not stop the run")
    ap.add_argument("--fast-check", action="store_true",
                    help="one frame per notebook, as warmup.py fast_check does: establishes that every example RUNS, not that it is right")
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="run each notebook for exactly N frames by editing its own frames "
                         "parameter: an ordinary run of a shorter scene, for a measurement "
                         "taken before a chaotic scene amplifies its own spread")
    ap.add_argument("--out", default="suite_results.json")
    ap.add_argument("--data-root", default=str(Path.home() / ".local/share/ppf-cts"))
    ap.add_argument("--python", default=str(VENV_PY), metavar="PATH",
                    help="the interpreter each notebook runs under (default: the "
                         "developer environment). A distribution passes its own "
                         "python/bin/python3, with --data-root naming its "
                         "local/share/ppf-cts")
    ap.add_argument("--root", default=str(ROOT), metavar="PATH",
                    help="the tree whose examples/ is run and whose frontend/ the "
                         "notebooks import (default: this checkout). A "
                         "distribution names its own unpacked directory")
    ap.add_argument("--reference", default=str(SHAPE_REFERENCE),
                    help="measured CUDA execution-shape reference to compare against")
    ap.add_argument("--shape-band", type=float, default=SHAPE_BAND,
                    help=f"ratio window for a banded shape check (default {SHAPE_BAND}); "
                         "tighten it only from a measured --repeat envelope")
    ap.add_argument("--shape-gate", choices=SHAPE_GATE_MODES, default="measured",
                    help="off: report the shape and let nothing decide. "
                         "measured (default): a scene with a measured reference "
                         "decides the exit code, an unmeasured one is counted and "
                         "named. strict: an unmeasured scene fails too.")
    ap.add_argument("--record-reference", action="store_true",
                    help="write this run's shape into the reference file (CUDA host only, "
                         "and only for a run this harness judged OK)")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="run each notebook N times and report this backend's OWN "
                         "run-to-run envelope; a band is only honest against one")
    ap.add_argument("--self-test", action="store_true",
                    help="replay the checked-in drape fixtures through the comparator; "
                         "needs no GPU, no Metal device, no built solver and no "
                         "frontend venv (numpy, which this file already imports, "
                         "is the only requirement)")
    ap.add_argument("--shape-from", metavar="DIR",
                    help="skip running anything: read the two indicator streams "
                         "already in DIR and render the shape verdict. Needs "
                         "--only <one scene> to say which reference to use.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.shape_from:
        if not args.only or len(args.only) != 1:
            sys.exit("--shape-from needs --only <scene>, exactly one, to name "
                     "the reference entry to compare against")
        return shape_from_dir(Path(args.shape_from), args.only[0],
                              load_reference(Path(args.reference)),
                              args.shape_band)

    if not args.backend:
        sys.exit("--backend is required (it labels the report and keys a recorded reference)")
    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    VENV_PY = Path(args.python)
    if not VENV_PY.exists():
        sys.exit(f"no interpreter at {VENV_PY}")
    # BOTH HALVES ARE NAMED, BECAUSE EITHER ONE MISSING PRODUCES A RUN THAT
    # LOOKS LIKE A SOLVER RESULT. A root with no `examples/` sweeps nothing and
    # reports a clean zero-scene pass; a root with no `frontend/` lets every
    # notebook fail its import, which reads as the backend refusing the scene.
    ROOT = Path(args.root).resolve()
    for needed in ("frontend", "examples"):
        if not (ROOT / needed).is_dir():
            sys.exit(f"--root {ROOT} has no {needed}/; it is not a solver tree")

    todo = notebooks()
    if args.only:
        want = set(args.only)
        todo = [p for p in todo if p.stem in want]

    # A missing or malformed reference is named here, before any scene runs, so
    # a four-hour suite is not spent to discover it at the summary.
    reference = None
    ref_path = Path(args.reference)
    try:
        reference = load_reference(ref_path)
    except ShapeError as exc:
        if args.shape_gate != "off" or args.record_reference:
            sys.exit(f"execution-shape reference unusable: {exc}")
        print(f"note: no usable execution-shape reference ({exc}); "
              f"shape statistics will be reported without a verdict")
    ref_scenes = (reference or {}).get("scenes", {})

    if args.fast_check and args.frames:
        sys.exit("--fast-check and --frames are two different measurements and cannot be "
                 "combined: fast check forces frames=1 through the solver's export "
                 "override, which would overwrite whatever --frames wrote.")
    mode = ("fast-check (1 frame each)" if args.fast_check
            else f"bounded runs ({args.frames} frames each)" if args.frames
            else "full runs")
    repeat = f"  repeat={args.repeat}" if args.repeat > 1 else ""
    print(f"backend={args.backend}  notebooks={len(todo)}  {mode}{repeat}  "
          f"data_root={args.data_root}")
    print(f"execution-shape gate={args.shape_gate}  band={args.shape_band}  "
          f"reference={ref_path}")
    if args.frames and args.shape_gate != "off":
        # Same argument as the fast-check case below: a handful of frames is too
        # few substeps for the reference's whole-scene statistics to be the same
        # measurement.
        print(f"--frames {args.frames} is too few substeps for a shape comparison; "
              f"forcing --shape-gate off")
        args.shape_gate = "off"
    if args.fast_check and args.shape_gate != "off":
        # One frame is one substep, so both streams hold a couple of records
        # and every statistic over them is a sample of size one. The reference
        # summarizes hundreds of substeps over a whole scene, so the two are not
        # the same measurement and comparing them would be arithmetic without
        # meaning.
        print("--fast-check runs one frame, which is too few substeps for a "
              "shape comparison; forcing --shape-gate off")
        args.shape_gate = "off"
    # Session directories already attributed, so one scene's output can never
    # be credited to a later scene that produced none of its own. A repeat of
    # the same notebook may reuse its own directory, which is why this maps to
    # the stem rather than merely holding the path.
    claimed: dict = {}
    results = []
    for i, nb in enumerate(todo, 1):
        runs = []
        for k in range(1, args.repeat + 1):
            tag = f"[{i}/{len(todo)}] {nb.stem}"
            if args.repeat > 1:
                tag += f" ({k}/{args.repeat})"
            print(f"{tag} ... ", end="", flush=True)
            try:
                r = run_one(nb, args.backend, Path(args.data_root), args.timeout,
                            claimed, fast=args.fast_check, frame_cap=args.frames)
            except Exception as exc:  # a harness fault must not be read as a scene failure
                r = {"name": nb.stem, "backend": args.backend, "stage": "harness",
                     "ok": False, "error": repr(exc)}

            # The shape verdict. Rendered for every run that actually ran,
            # whether or not it decides the exit code, because the columns are
            # what a reader needs to see a divergence the value columns cannot
            # show. A scene the backend REFUSED by name gets none: it wrote no
            # frames by design, so there is no execution to have a shape, and
            # the refusal line above is the whole report on it.
            profile = r.get("shape_profile")
            if r.get("stage") == "run" and profile is not None:
                r["shape"] = compare_shape(profile, ref_scenes.get(nb.stem),
                                           args.shape_band)
            runs.append(r)
            results.append(r)

            if r.get("stage") == "run":
                print(f"{'ok ' if r['ok'] else 'FAIL'} frames={r['frames']}/{r['frames_wanted']} "
                      f"disp_mean={r['disp_mean']:.3e} moving={r['moving_frames']}/{r['steps']} "
                      f"{r['wall_s']}s"
                      + (f"  {format_shape(r['shape'])}" if "shape" in r else "")
                      + (f"  aborted: {r['abort_reason']}" if r.get("aborted") else ""))
            elif r.get("stage") == "refused":
                print(f"refused (by design, not a failure) {r.get('reason','')[:90]}")
            elif r.get("stage") == "run-nosim":
                print(f"{'ok ' if r['ok'] else 'FAIL'} (no session; not a simulation) "
                      f"rc={r['rc']} {r['wall_s']}s")
            else:
                print(f"FAIL ({r.get('stage')}) {r.get('error', '')[:120]}")
            Path(args.out).write_text(json.dumps(results, indent=2))

            # --fail-fast stops at the first scene that did not pass, so a run
            # whose failure is going to repeat on every scene reports it in one
            # scene's time rather than the whole suite's. A REFUSAL is not a
            # failure and does not stop anything: the backend declined a
            # capability by name and wrote no frames by design. `run-nosim` is
            # judged on its own `ok`, since a non-simulating example still has
            # an exit code worth honoring.
            if args.fail_fast and r.get("stage") != "refused" and not r.get("ok"):
                print(f"\n--fail-fast: stopping at {nb.stem}, which did not pass.")
                print(f"  stage={r.get('stage')} error={str(r.get('error',''))[:200]}")
                tail = r.get("tail")
                if tail:
                    print("  tail:")
                    for line in str(tail).splitlines()[-12:]:
                        print(f"    {line}")
                Path(args.out).write_text(json.dumps(results, indent=2))
                return 1

        if args.repeat > 1:
            profiles = [x["shape_profile"] for x in runs
                        if x.get("shape_profile")
                        and "iter" in x["shape_profile"].get("streams", {})]
            if len(profiles) >= 2:
                env = envelope(profiles)
                runs[-1]["envelope"] = env
                print(f"    envelope over {env['runs']} runs of {nb.stem} "
                      f"on {args.backend}:")
                for name, e in env["statistics"].items():
                    ratio = f"{e['ratio']:.4f}x" if e["ratio"] else "n/a"
                    print(f"      {name:24s} {e['min']:.4g} .. {e['max']:.4g}  "
                          f"spread {ratio}")
                Path(args.out).write_text(json.dumps(results, indent=2))

        if args.record_reference:
            last = runs[-1]
            if not last.get("ok") or last.get("stage") != "run":
                print(f"    NOT recording a reference for {nb.stem}: the run did "
                      f"not pass, and a run that truncated, froze or crashed must "
                      f"not become the oracle")
            elif not last.get("shape_profile", {}).get("streams"):
                print(f"    NOT recording a reference for {nb.stem}: it wrote no "
                      f"indicator streams")
            else:
                meta = {"recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                        "host": platform.node(),
                        "wall_s": last["wall_s"]}
                # A measured envelope travels with the reference it was taken
                # beside. It is what a later tightening of the band must be
                # argued from, and an envelope recorded on a different day
                # against a different reference argues for nothing.
                if "envelope" in last:
                    meta["envelope"] = last["envelope"]
                record_reference(ref_path, nb.stem, args.backend,
                                 last["shape_profile"], meta)
                print(f"    recorded reference for {nb.stem} in {ref_path}")

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{args.backend}: {ok}/{len(results)} produced moving output")
    for r in results:
        if not r.get("ok"):
            print(f"  FAIL {r['name']}: frames={r.get('frames')} "
                  f"moving={r.get('moving_frames')} rc={r.get('rc')}")
            # The captured tail is the only place the scene's own diagnosis
            # survives: the notebook runs in a subprocess whose output this
            # harness swallows, so without this a solver abort reads as a bare
            # `rc=1 frames=0` and the next step is to re-run the converted
            # script by hand to see what it already said.
            # Progress bars redraw with carriage returns and would otherwise
            # crowd out the end of stderr, which is where the diagnosis is.
            noise = ("it/s]", "s/it]", "?it/s")
            lines = [ln.split("\r")[-1].rstrip()
                     for ln in (r.get("tail") or "").splitlines()]
            lines = [ln for ln in lines
                     if ln.strip() and not any(n in ln for n in noise)]
            for line in lines[-12:]:
                print(f"      | {line}")

    # The shape summary is printed whether or not it gates, and the categories
    # stay apart on purpose: "this backend diverged" and "nobody has measured
    # this scene yet" call for different actions and must not share a number.
    shaped = [r for r in results if "shape" in r]
    ran = [r for r in results if r.get("stage") == "run"]
    if shaped:
        by_verdict: dict = {}
        for r in shaped:
            by_verdict.setdefault(r["shape"]["verdict"], []).append(r["name"])
        print(f"{args.backend}: execution shape over {len(shaped)} runs")
        for verdict in sorted(by_verdict):
            names = by_verdict[verdict]
            print(f"  {verdict:20s} {len(names):3d}  {', '.join(sorted(set(names)))[:100]}")
        for r in shaped:
            for c in r["shape"]["checks"]:
                if not c["ok"]:
                    ratio = (f"  ratio {c['ratio']:.4f}x"
                             if c["ratio"] is not None else "")
                    print(f"  {r['name']}: {c['name']} got {c['got']} "
                          f"want {c['want']}{ratio}")
        # The unmeasured scenes are the gate's own debt, and a debt nobody
        # prints is a debt nobody pays. Name them every run, with the command.
        unmeasured = sorted(set(by_verdict.get("no-reference", [])))
        if unmeasured:
            noun = "scene carries" if len(unmeasured) == 1 else "scenes carry"
            print(f"  {len(unmeasured)} {noun} no measured CUDA reference, so "
                  f"the gate said NOTHING about them: {', '.join(unmeasured)}")
            print(f"  measure them on a CUDA host with: run_suite.py --backend "
                  f"cuda --only <scene> --record-reference")

    # AN EMPTY SET OF VERDICTS IS NOT A PASS. `all([])` is True, and a gate that
    # green-lights a run in which it rendered no verdict at all is the failure
    # mode this whole detector exists to close: the coverage vanishes and the
    # summary still reads clean. So the gate demands a verdict for every scene
    # that ran, and at least one of them.
    unshaped = len(ran) - len(shaped)
    failed = [r for r in shaped if shape_fails(r["shape"], args.shape_gate)]
    shape_ok = not failed and (args.shape_gate == "off"
                               or (bool(shaped) and unshaped == 0))
    if not shape_ok:
        if not shaped:
            print("execution-shape gate: FAILED, it rendered no verdict at all "
                  "(nothing ran, or no run reached the stream reader). A gate "
                  "that measured nothing has not passed.")
        elif unshaped:
            print(f"execution-shape gate: FAILED, {unshaped} of {len(ran)} runs "
                  f"carry no verdict")
        else:
            print(f"execution-shape gate: FAILED on "
                  f"{', '.join(sorted({r['name'] for r in failed}))}")
    return 0 if (ok == len(results) and shape_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
