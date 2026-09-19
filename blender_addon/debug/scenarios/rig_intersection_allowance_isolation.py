# File: scenarios/rig_intersection_allowance_isolation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Does an issue-#138 intersection allowance stay INSIDE the pairs that asked
# for it?
#
# The other gates on the feature all ask whether the rule is right for one
# pair at a time: `rig_intersection_allowances` walks it at the scene-build
# gate. None of them
# can see a defect in how the flags are DISTRIBUTED over a scene, which is
# what this scenario measures. A policy byte written to the wrong slice of the
# vertex buffer, two objects handed the same object id, and a pin bit reaching
# an element that is not fully pinned all satisfy the truth table exactly, and
# all of them suppress pairs that never asked to be suppressed.
#
# The method is arithmetic rather than a judgment call. Two crossed sheets
# report a fixed number of tri-tri pairs, and two such tangles placed far
# apart in one scene share no element pair, so the scene reports exactly twice
# that. For each allowance:
#
#     one unflagged tangle           ->  solo
#     two unflagged tangles          ->  2 * solo
#     one flagged, one unflagged     ->  solo        (nothing leaked)
#     both flagged                   ->  0
#
# `solo` is MEASURED at run time rather than written down here, so changing
# the fixture resolution moves all four expectations together instead of
# turning into a stale constant.
#
# All three allowances are covered that way: `allow-inter-object-intersection`
# on one of a crossed pair, `allow-self-intersection` on a single object built
# by concatenating two crossed sheets, and the per-pin flag. The pin case
# carries one extra control, a pinned object whose pin does NOT ask for the
# allowance, which separates "pinned" from "pinned and allowed": the exemption
# is opt-in per pin, not a property of being pinned.
#
# Then two things the build gate alone cannot answer.
#
# A RUN. A scene whose only tangle is tolerated must build and clear the
# solver's own check at `initialize`, which is the DEVICE half of the allowance
# gate and has to grant the same set the host scene-build gate did. Reaching a
# Newton step is the evidence: an initialize that rejected the tangle would
# never get there. What the run does AFTER initialize is a separate question and
# is non-deterministic on a real GPU. The fixture is two crossed sheets with
# EVERY vertex of both pinned by a pin that sets the flag, so it is authored
# with coincident elements; the ACCD line search may find a pair whose start
# separation is zero and refuse to advance into it (`### ccd failed` /
# `contact starts overlapping`). That refusal is the penetration-free guarantee
# working, not a defect (the intersection-allowance smoke scene states it), so
# this case accepts a completed run and a clean post-initialize stop alike, and
# it is the reason the run assertion cannot be a bare `finished()`. A device
# assert is still caught: it takes the probe down and the scenario fails on a
# missing result marker. Each pin carries a move op, which is what keeps its
# object DYNAMIC: a fully pinned object with no operations is promoted to a
# rest-pose STATIC collider, which leaves the solved namespace and is then
# skipped by `both_collider`, so the run would pass without the allowance
# doing anything.
#
# The WIRING, which shows that the right bytes reach the GPU rather than only
# that the rule over them is right. The two per-vertex files the solver reads
# are `bin/object_vert.bin` (u32 source-object identity, the only thing that
# tells a self-intersection from an inter-object one) and
# `bin/intersect_policy.bin` (u8, bit 0 allow self, bit 1 allow inter-object).
# Both are read back with numpy off a built session, and the policy bits must
# sit on exactly the flagged object's vertices. The two objects come from
# assets of DIFFERENT resolutions, so the flagged one is identified by its
# vertex count and the check assumes nothing about the order in which the
# frontend numbers objects. The control is the same pair of objects with no
# allowance anywhere: `intersect_policy.bin` must then be absent, since the
# frontend writes it only when some object asks.
#
# BOTH BACKENDS. The counting cases are the host-side scene-build check and
# the wiring cases read exported files, so both run identically on either. The
# run case is the one that differs and it is a real gate on each: on the
# solver it exercises the live edge-triangle scan, and on CUDA it exercises
# `check_intersection` at `initialize` and after every step.
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


# See the header: the build-gate and wiring halves are backend-agnostic, and
# the run case asserts something true of both solvers.
BACKENDS = ("real",)
# Drives a solver run, so it should not share a worker with another one.
NOT_PARALLELIZABLE = True


_PROBE = r'''
import json
import os
import sys

import numpy as np

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App
from frontend._scene_ import ValidationError

ALLOW_SELF = "allow-self-intersection"
ALLOW_INTER = "allow-inter-object-intersection"

# `VertexProp::intersect_policy` is a bit field: bit 0 allows self, bit 1
# allows inter-object. Mirrored from `crates/ppf-cts-solver/src/data.rs` and
# `frontend/_scene_.py`, because the wiring case reads the exported byte and
# so has to name the bit the solver reads. A wiring that sent the two group
# allowances to the same bit, or swapped them, writes a policy file of the
# right length with non-zero bytes in it and differs from this one value.
BIT_INTER = 1 << 1

# Distance between the two tangles. Each one spans about 1.3 units, so this
# leaves them without a single element pair in common, which is what makes the
# reported counts additive.
SPACING = 6.0

app = App.create("rig_isect_allowance_isolation")

# `mesh.square` returns 5 columns: xyz then uv, so only [:, :3] is a position.
V, F = app.mesh.square(res=4, ex=[1, 0, 0], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)

# A second sheet at a different resolution. Only the wiring case uses it, and
# it uses it for its vertex COUNT: two objects of different sizes can be told
# apart in `object_vert.bin` without assuming an ordering.
VFINE, FFINE = app.mesh.square(res=5, ex=[1, 0, 0], ey=[0, 1, 0])
app.asset.add.tri("sheet_fine", VFINE, FFINE)
N_SHEET = len(V)
N_SHEET_FINE = len(VFINE)

# ONE asset holding two overlapping sheets, so a self-intersection fixture
# keeps its tangle inside a single object and needs no second object at all.
# Built by concatenation rather than by folding a sheet: a fold has to be
# tuned until it genuinely crosses, and one that merely comes close builds
# clean, which would make the allowed case pass for the wrong reason.
ANGLE = np.deg2rad(12.0)
ROT = np.array([
    [np.cos(ANGLE), 0.0, np.sin(ANGLE)],
    [0.0, 1.0, 0.0],
    [-np.sin(ANGLE), 0.0, np.cos(ANGLE)],
])
VROT = np.array(V, dtype=np.float64, copy=True)
VROT[:, :3] = (VROT[:, :3] @ ROT.T) + np.array([0.3, 0.0, 0.0])
app.asset.add.tri("tangled", np.vstack([V, VROT]), np.vstack([F, F + len(V)]))


cases = {}
measurements = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def measure(setup):
    """(reported self-intersection pairs, other violation types).

    The number is the `self_intersection` violation's own `count`, which is
    the total pair count rather than the truncated list of triangle positions
    beside it. No fixture here carries a rod, so every counted pair is
    tri-tri. A build that raises for some OTHER reason would otherwise read as
    zero intersections, so the types are returned alongside and asserted
    empty by `no_unrelated_violations`.
    """
    scene = app.scene.create()
    setup(scene)
    try:
        scene.build(quiet=True)
    except ValidationError as error:
        reported = 0
        other = []
        for violation in error.violations:
            if violation.get("type") == "self_intersection":
                reported = int(violation["count"])
            else:
                other.append(str(violation.get("type")))
        return reported, sorted(other)
    return 0, []


def take(name, setup):
    reported, other = measure(setup)
    measurements[name] = {"reported": reported, "other": other}
    return reported


def crossed_pair(scene, x, flags=(), pin=False, pin_allows=False):
    """An INTER-OBJECT tangle near `x`: two objects crossing each other."""
    a = scene.add("sheet").at(x, 0.0, 0.0)
    scene.add("sheet").at(x + 0.3, 0.0, 0.0).rotate(12.0, "y")
    for key in flags:
        a.param.set(key, 1.0)
    if pin:
        # Every vertex of `a`, so every one of its elements is fully pinned
        # and the all-N-vertices rule can fire at all. The op keeps `a`
        # DYNAMIC (see the header), and it starts at t=0, so the pose this
        # build check reads is the authored one.
        a.pin(allow_intersection=pin_allows).move_by(
            [0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)


def self_tangle(scene, x, flags=()):
    """A SELF tangle near `x`: one object whose two sheets cross."""
    obj = scene.add("tangled").at(x, 0.0, 0.0)
    for key in flags:
        obj.param.set(key, 1.0)


# --- allow-inter-object-intersection ------------------------------------
solo_pair = take("solo_pair", lambda s: crossed_pair(s, 0.0))
two_pairs = take(
    "two_pairs",
    lambda s: (crossed_pair(s, 0.0), crossed_pair(s, SPACING)))
inter_mixed = take(
    "inter_mixed",
    lambda s: (crossed_pair(s, 0.0, flags=(ALLOW_INTER,)),
               crossed_pair(s, SPACING)))
inter_both = take(
    "inter_both",
    lambda s: (crossed_pair(s, 0.0, flags=(ALLOW_INTER,)),
               crossed_pair(s, SPACING, flags=(ALLOW_INTER,))))

record("pair_tangle_is_reported", solo_pair > 0,
       {"solo": solo_pair})
record("pair_tangles_count_additively", two_pairs == 2 * solo_pair,
       {"solo": solo_pair, "two": two_pairs, "expected": 2 * solo_pair})
record("inter_object_flag_spares_only_its_own_tangle",
       inter_mixed == solo_pair,
       {"mixed": inter_mixed, "expected": solo_pair, "two": two_pairs})
record("inter_object_flag_on_both_reports_nothing", inter_both == 0,
       {"both": inter_both})

# --- allow-self-intersection --------------------------------------------
solo_self = take("solo_self", lambda s: self_tangle(s, 0.0))
two_selves = take(
    "two_selves",
    lambda s: (self_tangle(s, 0.0), self_tangle(s, SPACING)))
self_mixed = take(
    "self_mixed",
    lambda s: (self_tangle(s, 0.0, (ALLOW_SELF,)), self_tangle(s, SPACING)))
self_both = take(
    "self_both",
    lambda s: (self_tangle(s, 0.0, (ALLOW_SELF,)),
               self_tangle(s, SPACING, (ALLOW_SELF,))))

record("self_tangle_is_reported", solo_self > 0, {"solo": solo_self})
record("self_tangles_count_additively", two_selves == 2 * solo_self,
       {"solo": solo_self, "two": two_selves, "expected": 2 * solo_self})
record("self_flag_spares_only_its_own_tangle", self_mixed == solo_self,
       {"mixed": self_mixed, "expected": solo_self, "two": two_selves})
record("self_flag_on_both_reports_nothing", self_both == 0,
       {"both": self_both})

# --- the per-pin flag ----------------------------------------------------
pin_plain = take(
    "pin_plain",
    lambda s: (crossed_pair(s, 0.0, pin=True), crossed_pair(s, SPACING)))
pin_mixed = take(
    "pin_mixed",
    lambda s: (crossed_pair(s, 0.0, pin=True, pin_allows=True),
               crossed_pair(s, SPACING)))
pin_both = take(
    "pin_both",
    lambda s: (crossed_pair(s, 0.0, pin=True, pin_allows=True),
               crossed_pair(s, SPACING, pin=True, pin_allows=True)))

record("pin_without_the_flag_spares_nothing", pin_plain == 2 * solo_pair,
       {"pinned_unflagged": pin_plain, "expected": 2 * solo_pair})
record("pin_flag_spares_only_its_own_tangle", pin_mixed == solo_pair,
       {"mixed": pin_mixed, "expected": solo_pair,
        "pinned_unflagged": pin_plain})
record("pin_flag_on_both_reports_nothing", pin_both == 0, {"both": pin_both})

record("no_unrelated_violations",
       all(not m["other"] for m in measurements.values()),
       {name: m for name, m in measurements.items() if m["other"]}
       or {"note": "every refused build was refused for intersections only"})


# --- the wiring: what actually lands in the session ----------------------

def build_session(name, setup):
    """(session, first line of the refusal), the session being None if refused.

    A refused build is a RESULT here rather than a crash: every case below
    wants the scene it names to build, so a refusal has to reach the report as
    that case failing rather than as a traceback that takes the whole probe
    down before any case is printed.
    """
    scene = app.scene.create()
    setup(scene)
    try:
        fixed = scene.build(quiet=True)
    except ValidationError as error:
        return None, str(error).splitlines()[0][:120]
    session = app.session.create(fixed, name=name)
    session.param.set("dt", 0.01).set("frames", 8)
    return session.build(), ""


def read_bin(session, name, dtype):
    """None when the file is absent, which is the answer the control needs."""
    if session is None:
        return None
    path = os.path.join(session.info.path, "bin", name)
    if not os.path.isfile(path):
        return None
    return np.fromfile(path, dtype=dtype)


def wiring_scene(scene):
    flagged = scene.add("sheet").at(0.0, 0.0, 0.0)
    flagged.param.set(ALLOW_INTER, 1.0)
    scene.add("sheet_fine").at(0.3, 0.0, 0.0).rotate(12.0, "y")


def control_scene(scene):
    # The same two objects with no allowance anywhere. They are pulled apart
    # because an unflagged tangle does not build at all, and the export writes
    # these two files the same way whether or not the scene intersects.
    scene.add("sheet").at(0.0, 0.0, 0.0)
    scene.add("sheet_fine").at(SPACING, 0.0, 0.0)


wiring, wiring_error = build_session("wiring", wiring_scene)
object_id = read_bin(wiring, "object_vert.bin", np.uint32)
policy = read_bin(wiring, "intersect_policy.bin", np.uint8)

sizes = {}
if object_id is not None:
    ids, counts = np.unique(object_id, return_counts=True)
    sizes = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())}
flagged_ids = [i for i, count in sizes.items() if count == N_SHEET]
record(
    "object_ids_are_distinct_per_object",
    N_SHEET != N_SHEET_FINE
    and sorted(sizes.values()) == sorted([N_SHEET, N_SHEET_FINE]),
    {"vertices_per_object_id": sizes, "n_sheet": N_SHEET,
     "n_sheet_fine": N_SHEET_FINE, "build_error": wiring_error},
)

policy_ok = (
    policy is not None
    and object_id is not None
    and len(flagged_ids) == 1
    and policy.size == object_id.size
)
on_flagged = []
elsewhere = []
if policy_ok:
    mine = object_id == flagged_ids[0]
    on_flagged = sorted({int(v) for v in policy[mine].tolist()})
    elsewhere = sorted({int(v) for v in policy[~mine].tolist()})
    policy_ok = on_flagged == [BIT_INTER] and elsewhere == [0]
record(
    "policy_bits_reach_only_the_flagged_object",
    policy_ok,
    {"policy_bytes": None if policy is None else int(policy.size),
     "object_bytes": None if object_id is None else int(object_id.size),
     "flagged_object_ids": flagged_ids,
     "bytes_on_flagged": on_flagged, "expected_on_flagged": [BIT_INTER],
     "bytes_elsewhere": elsewhere, "build_error": wiring_error},
)

control, control_error = build_session("control", control_scene)
control_policy = read_bin(control, "intersect_policy.bin", np.uint8)
control_object = read_bin(control, "object_vert.bin", np.uint32)
record(
    "unflagged_build_writes_no_policy_file",
    control_policy is None and control_object is not None,
    {"intersect_policy_bin": None if control_policy is None
     else int(control_policy.size),
     "object_vert_bin": None if control_object is None
     else int(control_object.size),
     "build_error": control_error,
     "note": "object_vert.bin is written unconditionally, so its presence is "
             "what makes the absence of intersect_policy.bin a statement "
             "about the feature rather than about no bins being written"},
)


# --- the run: the solver's own check has to grant the same set -----------

def pinned_tangle_scene(allow):
    """One side fully pinned and asking for the allowance, one side FREE.

    Only ONE side may be fully pinned. A pair whose BOTH sides are prescribed
    is discarded by `either_dyn` before the allowance rule is ever consulted
    (`FaceProp::fixed` is set when every vertex of the element is fix-pinned,
    builder.rs), so such a fixture would build and run whatever the flag said
    and would certify nothing. The free side is held by a grab pin on one edge
    so it stays in the domain; a partial pin leaves its elements unfixed, which
    is what keeps `either_dyn` true and the rule live.

    The move op on the pinned side is what keeps THAT object dynamic:
    `Object.update_static` promotes a fully pinned object with no operations to
    a rest-pose STATIC collider, which leaves the solved namespace and is then
    skipped by `both_collider`, another way to certify nothing.
    """
    def setup(scene):
        a = scene.add("sheet").at(0.0, 0.0, 0.0)
        b = scene.add("sheet").at(0.3, 0.0, 0.0).rotate(12.0, "y")
        a.pin(allow_intersection=allow).move_by(
            [0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
        b.pin(b.grab([0, 1, 0]))
    return setup


# The control comes first and is what makes the run case non-vacuous: the SAME
# fixture with the flag off must be REFUSED. If a future change lets the pair
# be discarded before the rule runs, this control starts building and fails,
# which is the failure mode a run-only case cannot see.
ctl_session, ctl_error = build_session("pinned_tangle_control",
                                       pinned_tangle_scene(False))
record("untolerated_tangle_is_refused", ctl_session is None,
       {"built": ctl_session is not None, "build_error": ctl_error})

run_session, run_error = build_session("pinned_tangle_run",
                                       pinned_tangle_scene(True))
finished = False
reached_advance = False
stop_cause = ""
if run_session is not None:
    try:
        run_session.start(blocking=True)
    except Exception:
        # A penetration-free stop surfaces as a solver abort the frontend
        # re-raises. Which stop it was is read from the solver's own log
        # below, not from the exception.
        pass
    finished = bool(run_session.finished())
    try:
        with open(os.path.join(run_session.info.path, "stdout.log")) as _log:
            solver_log = _log.read()
    except OSError:
        solver_log = ""
    # The tolerated tangle has to clear the solver's own initialize scan, which
    # is the DEVICE half of the allowance gate and must grant the same set the
    # host scene-build gate did. A run that reaches a Newton step proves
    # initialize honored the allowance rather than rejecting the tangle. What
    # happens after initialize is separate and non-deterministic on a real GPU:
    # the tangle is authored with coincident elements, so the line search may
    # find a pair whose start separation is zero and refuse to advance into it.
    # That refusal is the penetration-free guarantee working, not a defect
    # (the intersection-allowance smoke scene states it directly), so a clean
    # `overlapping_start` or `ccd` stop AFTER initialize counts the same as a
    # completed run. A device assert never reaches this line: it takes the whole
    # probe down and the scenario fails on a missing result marker instead.
    reached_advance = "newton step" in solver_log
    if not finished:
        if "contact starts overlapping" in solver_log:
            stop_cause = "overlapping_start"
        elif "ccd failed" in solver_log:
            stop_cause = "ccd"
        else:
            stop_cause = "other"
run_ok = (run_session is not None) and (
    finished or (reached_advance and stop_cause in ("overlapping_start", "ccd")))
record("tolerated_tangle_runs_past_initialize", run_ok,
       {"built": run_session is not None, "build_error": run_error,
        "finished": finished, "reached_advance": reached_advance,
        "stop_cause": stop_cause})

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    # No `PPF_STEP_DELAY_MS`. Nothing here watches a run in progress: each
    # case is judged by what its completed run reported, so a per-step delay
    # would only slow the sweep.
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
    return r.report_named_checks(cases, label="allowance isolation cases",
                                 max_violations=8)
