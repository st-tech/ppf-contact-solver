# File: scenarios/rig_autosave_resume.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Auto-save and resume, asserted end to end on the real backend for every scene
# kind whose run state a checkpoint has to carry: a resume must START, and the
# frames it produces must be the frames the uninterrupted run produced.
#
# WHY THIS SCENARIO EXISTS. The other resume scenarios (`bl_save_resume`,
# `bl_resume_from_frame`, `bl_chain_save_resume`, `bl_world_scaling_resume`)
# drive a FULLY PINNED plane whose every position is a function of time, so a
# resume that dropped all run state would still reproduce it, and none of them
# runs a Cosserat rod, a friction contact, a plastic rest shape or a solid. A
# solver that refused every Cosserat resume outright passed all of them.
#
# NO BLENDER. The scenes are authored through `frontend` directly, in a
# SUBPROCESS, for the reason `rig_cosserat` gives: importing `frontend` loads
# the per-tree cdylib, and the orchestrator's long-lived process must not.
#
# Each case runs the same built scene in three sessions:
#   uninterrupted  N frames from the start with auto-save on,
#   control        the same N frames from the start again, auto-save on,
#   resumed        the uninterrupted session's own directory, resumed from its
#                  checkpoint K (the OLDEST one kept, so an explicit frame and
#                  not the latest-checkpoint sentinel), run to N.
#
# THE BOUND IS MEASURED, NOT CHOSEN. The resumed run is compared with the
# uninterrupted one over frames K+1..N. Two runs of one scene on one build may
# differ by the backend's own nondeterminism (parallel reductions in another
# order), and that is exactly what the control measures: it diverges from the
# uninterrupted run over all N frames, while a correct resume diverges over
# only N-K, so the control's largest difference over the compared frames bounds
# what a correct resume can show. On a deterministic backend the control
# agrees to the bit, and a correct resume must too. A resume that loses a piece
# of run state (a Cosserat frame, a friction memory, a crept rest shape, a
# velocity) restarts that piece and differs by a physical amount.
#
# THE COMPARISON MUST BE ABLE TO FAIL. Each case also requires its own motion
# over frames K..N to exceed that bound: a scene at rest by frame K would match
# any resume that loaded positions at all, which is the trap the pinned-plane
# scenarios fell into.
#
# Subtests, per case `<c>` in cloth_friction, cosserat_rod, plastic_shell,
# tet_solid:
#   <c>_A_uninterrupted_finishes   both from-start runs record `finished`.
#   <c>_B_checkpoints_at_cadence   the `state_<N>.bin.gz` set on disk is
#                                  exactly the auto-save cadence plus the
#                                  final frame, pruned to `keep-states`, and a
#                                  scene with friction, plasticity or Cosserat
#                                  rods carries a `friction_<N>` /
#                                  `plastic_<N>` / `runstate_<N>` beside every
#                                  one of them and nowhere else.
#   <c>_C_resume_starts            the resumed launch is a NEW launch that
#                                  records `finished` (a refused resume records
#                                  its panic, which is reported verbatim), kept
#                                  frames 0..K byte for byte, and wrote
#                                  K+1..N.
#   <c>_D_resume_matches           the resumed frames K+1..N agree with the
#                                  uninterrupted ones within the control's
#                                  largest difference over the same frames,
#                                  or one float32 step at the scene's largest
#                                  coordinate if that is larger and the control
#                                  is not zero,
#                                  and the scene moved by more than that bound
#                                  over K..N.
#   <c>_E_resume_prunes            after the resume the checkpoint set is the
#                                  same kept set again: checkpoints above K
#                                  were dropped and rewritten, not stacked.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# RUNS ON THE REAL BACKEND. Checkpoint writing and loading is neutral host code
# every backend shares; what differs per backend is only how deterministic a
# run is, which the control session measures rather than assumes.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import sys

import cbor2
import numpy as np

REPO_ROOT = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App

# Every session of this probe lives under the rig's own workspace rather than
# the user's data directory, so two checkouts or two workers running this
# scenario at once never share a session directory. `get_data_dirpath` is the
# one resolver `App` reads it through, and it is patchable for this purpose.
DATA_DIR = os.path.join(os.environ["PPF_CTS_DATA_ROOT"], "autosave_resume")
App.get_data_dirpath = staticmethod(lambda: DATA_DIR)

# The run, the auto-save cadence and the retention. With these the solver
# writes state_<f> for solver frames f > 0 with (f + 1) % AUTO_SAVE == 0, plus
# the final frame on finishing: 4, 9, 14, 19, 24. KEEP_STATES = 3 keeps 14, 19
# and 24, and the resume starts from 14, ten frames before the end.
FRAMES = 24
AUTO_SAVE = 5
KEEP_STATES = 3
FPS = 60.0

cases = {}


def record(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}


def status(session):
    path = os.path.join(session.info.path, "output", "status.cbor")
    if not os.path.isfile(path):
        return {"launch_id": None, "kind": "absent", "outcome": None}
    with open(path, "rb") as handle:
        record_ = cbor2.load(handle)
    payload = record_.get("payload") or {}
    outcome = payload.get("outcome") or {}
    return {
        "launch_id": payload.get("launch_id") or record_.get("launch_id"),
        "kind": str(outcome.get("kind", "")),
        "outcome": json.loads(json.dumps(outcome, default=str)),
    }


def output_dir(session):
    return os.path.join(session.info.path, "output")


def listed(session, prefix, suffix):
    out = output_dir(session)
    found = []
    for entry in os.listdir(out):
        if entry.startswith(prefix) and entry.endswith(suffix):
            middle = entry[len(prefix):-len(suffix)]
            if middle.isdigit():
                found.append(int(middle))
    return sorted(found)


def frames_of(session, frames):
    out = output_dir(session)
    result = {}
    for n in frames:
        path = os.path.join(out, "vert_{}.bin".format(n))
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                result[n] = handle.read()
    return result


def positions(raw):
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).astype(np.float64)


def start(session, **kwargs):
    try:
        session.start(force=True, blocking=True, **kwargs)
        return None
    except Exception as error:
        return "{}: {}".format(type(error).__name__, str(error))[:600]


def log_tail(session, n=12):
    lines = []
    for channel in ("stderr", "stdout"):
        try:
            lines += ["[{}] {}".format(channel, line) for line in getattr(session.get.log, channel)(n)]
        except Exception as error:
            lines.append("[{}] unreadable: {}".format(channel, error))
    return lines


def expected_states():
    saved = [f for f in range(1, FRAMES + 1) if (f + 1) % AUTO_SAVE == 0]
    if FRAMES not in saved:
        saved.append(FRAMES)
    return sorted(saved)[-KEEP_STATES:]


# ---------------------------------------------------------------- the scenes


def cloth_friction(app):
    """Two sheets with friction 0.5 on a 30 degree wall: tan(30) = 0.58 is
    above it, so the lower sheet keeps sliding down the wall (an analytic
    contact) through the resume, carrying the upper one on mesh contacts.
    Where each contact stuck is the friction memory the checkpoint must
    carry, and the slide is the motion that makes the comparison mean
    something."""
    tilt = np.radians(30.0)
    normal = np.array([np.sin(tilt), np.cos(tilt), 0.0])
    downhill = np.array([np.cos(tilt), -np.sin(tilt), 0.0])
    across = [0.0, 0.0, 1.0]
    V, F = app.mesh.square(res=8, size=0.3, ex=list(downhill), ey=across)
    app.asset.add.tri("lower", V, F)
    V, F = app.mesh.square(res=6, size=0.15, ex=list(downhill), ey=across)
    app.asset.add.tri("upper", V, F)
    scene = app.scene.create()
    lower = scene.add("lower").at(*(0.01 * normal))
    upper = scene.add("upper").at(*(0.03 * normal))
    for sheet in (lower, upper):
        sheet.param.set("friction", 0.5).set("young-mod", 1000).set("bend", 5.0)
    scene.add.invisible.wall([0.0, 0.0, 0.0], list(normal)).param.set("friction", 0.5)
    return scene, {}


def cosserat_rod(app):
    """A Cosserat rod clamped over its first two nodes, horizontal, with a rest
    twist, swinging down under gravity: its material frames turn every step,
    and they are run state a resume must restore."""
    density, ei, ga, ref = 1e-3, 1.8e-5, 1.0, 1e-4
    length, segments = 0.16, 16
    x = np.linspace(0.0, length, segments + 1)
    segment = length / segments
    V = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    E = np.array([[a, a + 1] for a in range(segments)], dtype=np.uint32)
    app.asset.add.rod("rod", V, E)
    scene = app.scene.create()
    rod = scene.add("rod").as_cosserat()
    (rod.param.set("density", density).set("young-mod", 2.0 / (density * segment))
        .set("cosserat-shear", 2.0 * ga / density).set("cosserat-bend", ei / (density * ref))
        .set("cosserat-twist", ei / (density * ref)).set("cosserat-rest-twist", 5.0)
        .set("contact-offset", 1e-4).set("contact-gap", 1e-4))
    rod.pin([0, 1])
    return scene, {"dt": 1e-2}


def plastic_shell(app):
    """A patch held on its left column and stretched by its right one, which is
    let go at t = 0.3 s (frame 18). Plasticity creeps the rest shape on the
    device while it is held, so at the checkpoint (frame 14) the rest shape is
    no longer the built one, and the released patch springs back toward
    whichever rest shape the resumed run holds."""
    size, res = 0.2, 5
    xs = np.linspace(0.0, size, res)
    zs = np.linspace(-0.5 * size, 0.5 * size, res)
    V = np.array([[x, 0.0, z] for x in xs for z in zs], dtype=np.float64)
    F = []
    for i in range(res - 1):
        for j in range(res - 1):
            a = i * res + j
            F += [[a, a + res, a + 1], [a + 1, a + res, a + res + 1]]
    app.asset.add.tri("patch", V, np.array(F, dtype=np.uint32))
    scene = app.scene.create()
    patch = scene.add("patch")
    (patch.param.set("young-mod", 500.0).set("bend", 1.0).set("deformation-damping", 0.5)
        .set("plasticity", 20.0).set("plasticity-threshold", 0.01))
    patch.pin(list(range(res)))
    patch.pin([(res - 1) * res + j for j in range(res)]).move_by(
        [0.25 * size, 0.0, 0.0], t_start=0.0, t_end=0.1).unpin(0.3)
    return scene, {"gravity": [0.0, 0.0, 0.0], "precond": "block-jacobi"}


def tet_solid(app):
    """A tetrahedral box sliding down a 20 degree wall with friction 0.2, below
    tan(20) = 0.36, so it keeps sliding: a solid's elastic state and a
    dynamic friction contact both cross the resume."""
    tilt = np.radians(20.0)
    normal = np.array([np.sin(tilt), np.cos(tilt), 0.0])
    V, F, T = app.mesh.tet_box(0.05, 0.05, 0.05)
    app.asset.add.tet("box", V, F, T)
    scene = app.scene.create()
    box = scene.add("box").rotate(-20.0, "z").at(*(0.03 * normal))
    box.param.set("friction", 0.2).set("young-mod", 1e4)
    scene.add.invisible.wall([0.0, 0.0, 0.0], list(normal)).param.set("friction", 0.2)
    return scene, {}


SCENES = {
    "cloth_friction": (cloth_friction, {"friction"}),
    "cosserat_rod": (cosserat_rod, {"runstate"}),
    "plastic_shell": (plastic_shell, {"plastic"}),
    "tet_solid": (tet_solid, {"friction"}),
}


def session_for(app, fixed, name, extra):
    session = app.session.create(fixed, name)
    param = (session.param.set("fps", FPS).set("frames", FRAMES)
             .set("auto-save", AUTO_SAVE).set("keep-states", KEEP_STATES))
    for key, value in extra.items():
        param.set(key, value)
    return session.build()


def run_case(case):
    builder, sidecars = SCENES[case]
    app = App.create("rig_autosave_resume_{}".format(case))
    scene, extra = builder(app)
    fixed = scene.build(quiet=True)
    first = session_for(app, fixed, "uninterrupted", extra)
    control = session_for(app, fixed, "control", extra)

    raised = {"uninterrupted": start(first), "control": start(control)}
    first_status, control_status = status(first), status(control)
    record(case + "_A_uninterrupted_finishes",
           first_status["kind"] == "finished" and control_status["kind"] == "finished",
           {"uninterrupted": first_status, "control": control_status, "raised": raised,
            "log": [] if first_status["kind"] == "finished" else log_tail(first)})
    if first_status["kind"] != "finished" or control_status["kind"] != "finished":
        return

    kept = expected_states()
    states = listed(first, "state_", ".bin.gz")
    side = {kind: listed(first, kind + "_", ".bin.gz") for kind in ("friction", "plastic", "runstate")}
    side_ok = all((side[kind] == kept) if kind in sidecars else (side[kind] == [])
                  for kind in side)
    record(case + "_B_checkpoints_at_cadence", states == kept and side_ok,
           {"state_frames": states, "expected": kept, "sidecars": side,
            "sidecars_expected": sorted(sidecars), "auto_save": AUTO_SAVE,
            "keep_states": KEEP_STATES, "frames": FRAMES})

    every = list(range(FRAMES + 1))
    uninterrupted = frames_of(first, every)
    reference = frames_of(control, every)
    if sorted(uninterrupted) != every or sorted(reference) != every:
        record(case + "_C_resume_starts", False,
               {"reason": "a from-start run is missing frames",
                "uninterrupted": sorted(uninterrupted), "control": sorted(reference)})
        return

    # The oldest checkpoint kept: an explicit frame, not the latest one.
    resume_from = kept[0]
    saved = first.get.saved()
    raised_resume = None
    if resume_from not in saved:
        raised_resume = "frame {} is not among the saved states {}".format(resume_from, saved)
    else:
        try:
            # `resume` returns without launching anything when it cannot
            # resolve the frame, which is why the frame is checked above.
            first.resume(frame=resume_from, blocking=True)
        except Exception as error:
            raised_resume = "{}: {}".format(type(error).__name__, str(error))[:600]
    resumed_status = status(first)
    resumed = frames_of(first, every)
    kept_prefix = all(resumed.get(n) == uninterrupted[n] for n in range(resume_from + 1))
    produced = sorted(resumed) == every
    new_launch = resumed_status["launch_id"] not in (None, first_status["launch_id"])
    started = (raised_resume is None and new_launch and resumed_status["kind"] == "finished"
               and kept_prefix and produced)
    record(case + "_C_resume_starts", started,
           {"resume_from": resume_from, "raised": raised_resume, "status": resumed_status,
            "new_launch": new_launch, "frames_0_to_K_unchanged": kept_prefix,
            "frames_present": sorted(resumed) == every,
            # The first run's `finished.txt` outlives a resume refused before
            # the solver's run loop, so `session.finished()` says True here;
            # this is why C reads the launch's own status record instead.
            "finished_marker_present": os.path.isfile(
                os.path.join(output_dir(first), "finished.txt")),
            "log": [] if started else log_tail(first)})
    if not started:
        return

    window = range(resume_from + 1, FRAMES + 1)
    resume_diff = max(float(np.abs(positions(resumed[n]) - positions(uninterrupted[n])).max())
                      for n in window)
    bound = max(float(np.abs(positions(reference[n]) - positions(uninterrupted[n])).max())
                for n in window)
    motion = float(np.abs(positions(uninterrupted[FRAMES])
                          - positions(uninterrupted[resume_from])).max())
    # ONE ROUNDING STEP AT THE SCENE'S LARGEST COORDINATE is the least two
    # stored poses can differ by. Where the control shows the backend is not
    # deterministic, a single control is one sample of that noise: both it and
    # a correct resume can sit at a one-step flip, and which coordinate flips
    # decides which is larger. So the bound is never below that step there. A
    # deterministic backend (control zero) must still match to the bit.
    extent = max(float(np.abs(positions(uninterrupted[n])).max()) for n in window)
    step = float(np.spacing(np.float32(extent)))
    allowed = max(bound, step) if bound > 0.0 else 0.0
    record(case + "_D_resume_matches", resume_diff <= allowed and motion > allowed,
           {"resume_from": resume_from, "largest_resume_difference_m": resume_diff,
            "control_bound_m": bound, "rounding_step_m": step, "allowed_m": allowed,
            "motion_over_resumed_frames_m": motion,
            "worst_frame": max(window, key=lambda n: float(np.abs(
                positions(resumed[n]) - positions(uninterrupted[n])).max()))})

    after = listed(first, "state_", ".bin.gz")
    record(case + "_E_resume_prunes", after == kept, {"state_frames": after, "expected": kept})


SUBTESTS = ("A_uninterrupted_finishes", "B_checkpoints_at_cadence", "C_resume_starts",
            "D_resume_matches", "E_resume_prunes")

for case in SCENES:
    if ONLY is not None and case not in ONLY:
        continue
    try:
        run_case(case)
    except Exception as error:
        import traceback
        record(case + "_A_uninterrupted_finishes", False,
               {"probe_raised": traceback.format_exc()[-1500:]})
    # A subtest a failure upstream kept from running is a failure too, so every
    # case reports all five and the count does not shrink with the breakage.
    failed_first = next((name for name in SUBTESTS
                         if case + "_" + name in cases and not cases[case + "_" + name]["ok"]), None)
    for name in SUBTESTS:
        cases.setdefault(case + "_" + name, {
            "ok": False, "details": {"not_reached": "{}_{} failed".format(case, failed_first)}})

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
        encoding="utf-8",
        errors="replace",
        env=env,
        # Twelve real solves of tiny scenes, 24 or 10 frames each.
        timeout=max(ctx.timeout, 1800.0),
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
    return r.report_named_checks(cases, label="auto-save and resume cases",
                                 max_violations=20)
