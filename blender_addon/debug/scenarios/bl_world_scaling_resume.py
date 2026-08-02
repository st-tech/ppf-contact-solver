# File: scenarios/bl_world_scaling_resume.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling across a SAVE / RESUME checkpoint. A world-scaled sim is
# run two ways on one build: once start-to-finish (the reference), and
# once interrupted mid-run by save_and_quit and then resumed to
# completion. We assert the resumed run reproduces the uninterrupted run
# exactly -- i.e. the checkpoint does NOT double-scale.
#
# Why this matters: the solver saves its checkpoint state at SIM scale
# (geometry x world_scaling) and, on resume, re-reads world_scaling
# fresh from param.toml and divides the per-frame output back by it. If
# resume re-applied the scale-in to the already-scaled checkpoint (or
# assumed world_scaling = 1.0 because it forgot to re-read it), the
# frames after the checkpoint would land at the wrong scale. The fully
# pinned, kinematic scene makes the per-frame target a deterministic
# function of time, so a correct resume reproduces the reference to
# round-off regardless of when the checkpoint happened to land.
#
# Interruption timing: subtest B asserts the interrupted run reaches a
# checkpoint and subtest E asserts that checkpoint landed mid-run, so the
# interruption is engineered rather than left to chance. The sim is paced
# to span ~7.2 s of wall clock (KNOBS below), and the driver waits for a
# real frame advance (state.frame >= 1) before it dispatches
# SaveAndQuitRequested. The solver reads that marker at the top of a loop
# iteration and checkpoints the frame it has reached, so the dispatch has
# to arrive while that frame is still short of the last one. Two ways it
# can arrive too late, and each has its own assertion: once the solver is
# out of its loop there is no checkpoint at all (it logs "simulation
# finished, not saving state...", the addon settles at READY, and B fails
# for a reason that has nothing to do with world scaling), and in the
# window just before that the checkpoint names the FINAL frame, which
# leaves nothing for the resumed solver to produce and reduces C and D to
# comparing two complete runs. The pacing widens the usable window and
# the gate puts the dispatch inside it; E is what proves it landed there,
# since the release frame is a function of the host's status-observation
# latency and not of anything in this file. The gate also refuses to
# dispatch into a solver that has already left the running set, so that
# case is named instead of arriving as a failed B.
#
# What C and D read: the capture after ``resume_and_wait`` only drains
# frames that are already on disk. Starting a run there would relaunch
# the solver with ``--load 0`` (executor/solver.rs maps resume_from=None
# to that) and rewrite session/output from frame 0, so C would compare
# two uninterrupted runs, which agree whatever resume does with
# world_scaling, and D would read that third run's positions.
#
# Subtests (the report sorts them by name, so E prints last even though
# it is asserted right after B):
#   A. reference_all_frames  - the uninterrupted run produced every frame.
#   B. checkpoint_reached    - the interrupted run hit RESUMABLE.
#   C. resume_matches_run    - resumed frames == uninterrupted frames
#                              (no double-scaling across the checkpoint).
#   D. authored_scale        - resumed positions are at the authored
#                              scale, not the (x world_scaling) sim scale.
#   E. checkpoint_mid_run    - the checkpoint's own frame is strictly
#                              inside the run, so frames C and D read
#                              were produced by the resumed solver.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Cross-cycle solver/checkpoint state; keep it off the parallel batch.
NOT_PARALLELIZABLE = True
# Wall-clock ms the emulated backend sleeps per solver step (the sleep at
# the end of cpp_emul advance()). Step count, derived from the state the
# driver configures below:
#   - FRAME_COUNT 12 is the Blender-side count; core/encoder/params.py
#     sends "frames" = frame_count - 1, so the solver's output frames are
#     0 through 11.
#   - step_size 0.01 with frame_rate 100 gives a substep dt of
#     min(0.01, 0.9999/100) = 0.009999 (clamp_substep_dt holds dt
#     strictly under one frame).
#   - backend.rs writes frame 0 before the loop, then after each step
#     emits every frame up to floor(time * fps). Step n sits at time
#     n * 0.009999, so frame f first appears on step f + 1.
#   - The loop stops once curr_frame reaches 11, i.e. after 12 steps:
#     ~7.2 s at this pacing.
# The solver reads the save_and_quit marker at the TOP of a loop
# iteration and checkpoints whatever frame it has reached, and it reads
# that marker BEFORE the frames-done test. After step n the run stands at
# frame n - 1, so the top of the 12th step carries frame 10, the highest
# value that is still short of the last frame, and it is reached after 11
# sleeps (~6.6 s). Past that the loop's next visit carries frame 11 and a
# marker found there writes a last-frame checkpoint (subtest E rejects
# it); past THAT the loop has exited and no checkpoint is written at all
# (subtest B rejects it).
#
# The pacing is set from the driver's OBSERVATION latency, not from the
# step arithmetic alone, and that is what makes it this large. The gate
# below waits for the addon to report frame >= 1, and the first report
# carrying a non-zero frame lands about 2 s after the run starts however
# fast the solver is going: the driver polls the server's status through
# the same command queue it uses for frame fetches, so the first status
# it sees already reflects several steps. Nothing can be dispatched
# before that report, so the pacing has to hold the solver several frames
# short of its last one at the 2 s mark AND keep the 12th step's top well
# beyond it. 600 ms buys both: the run spans 7.2 s, the gate releases
# around frame 1 or 2, and the ~4.5 s that remain before the 12th step's
# top absorb the request's trip through the server to the output
# directory. That budget is an argument for the pacing, not evidence
# about a given host, which is why E asserts the outcome on disk.
KNOBS = {"PPF_EMULATED_STEP_MS": "600"}

WORLD_SCALING = 10.0


_DRIVER_BODY = r"""
import glob
import os
import time
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 12
# The solver's last frame INDEX: the encoder sends "frames" =
# frame_count - 1, so the run ends at index 11 and that is the number
# state.frame carries when the run is done.
LAST_FRAME = FRAME_COUNT - 1
WORLD_SCALING = <<WORLD_SCALING>>


def _capture(dh, plane):
    # Read back the frames the solver has ALREADY written, without
    # starting anything: query until state.frame reaches the run's last
    # frame index, settle, fetch, and snapshot the PC2. fetch_and_drain
    # clears the fetched set first, so every frame is re-downloaded and
    # written over its own PC2 index (core/client.py
    # _write_mesh_frame_to_pc2 takes the overwrite branch for an index the
    # file already holds). The snapshot is therefore the whole span as it
    # stands on disk right now, and it carries FRAME_COUNT samples: PC2
    # index 0 is gap-filled when the file is created, because a fetch
    # covers solver frames 1..N and never downloads frame 0.
    #
    # Capturing must stay separable from running: after a resume the
    # frames under test are the ones the resumed solver produced, and any
    # com.run() in this path would relaunch the solver with --load 0 and
    # rewrite session/output from frame 0.
    #
    # LAST_FRAME, not FRAME_COUNT: state.frame carries the solver's last
    # frame INDEX, so a target of FRAME_COUNT would spend the whole
    # timeout waiting for a number the server never reports.
    dh.force_frame_query(expected_frames=LAST_FRAME, timeout=60.0)
    dh.settle_idle(timeout=10.0)
    dh.fetch_and_drain()
    pc2 = dh.find_pc2_for(plane)
    if not pc2 or not os.path.isfile(pc2):
        raise RuntimeError(f"no PC2 (path={pc2!r})")
    return dh.read_pc2(pc2).copy()


def _run_and_capture(dh, plane):
    # Reference form: run the whole sim from frame 0, then capture it.
    dh.run_and_wait(timeout=180.0)
    return _capture(dh, plane)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="WsResumeMesh")
    root = dh.configure_state(project_name="ws_resume",
                              frame_count=FRAME_COUNT, frame_rate=100)
    root.state.world_scaling = WORLD_SCALING
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.4, 0.0, 0.0), frame_start=1, frame_end=10,
                transition="LINEAR")

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _d, _p = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "ws-resume:build", timeout=180.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # ---- Reference: uninterrupted run ----
    ref = _run_and_capture(dh, plane)
    dh.record("A_reference_all_frames", ref.shape[0] == FRAME_COUNT,
              {"samples": int(ref.shape[0]), "expected": FRAME_COUNT})

    # ---- Interrupted run: save_and_quit mid-run, then resume ----
    # Gate on a real frame advance before asking for the checkpoint, so
    # the request reaches a solver that still has frames to run, and
    # after the spawn that deletes any save_and_quit sentinel already in
    # the output dir (see executor/solver.rs launch_solver).
    # RunRequested resets state.frame to 0 and that same spawn scrubs the
    # prior run's status.cbor, so frame >= 1 is evidence THIS run
    # advanced, not a tail from the reference run above. The scenario
    # declares no BACKENDS, i.e. emulated-only, so the gate only has to
    # cover the CPU emulator: its first frame lands two steps in
    # (~1.2 s at the pacing above), with no CUDA cold start to absorb.
    GATE_TIMEOUT_S = 30.0
    dh.com.run()
    # RunRequested transitions the solver to STARTING in the same tick
    # (core/transitions.py), so anything else here means the request was
    # dropped and no run is coming. Naming that separately is what lets
    # the loop below read a terminal solver state as "the run ended".
    started = dh.facade.engine.state
    if started.solver.name not in ("STARTING", "RUNNING"):
        raise RuntimeError(
            f"ws-resume gate: RunRequested did not take, so the "
            f"interrupted run never started "
            f"(solver={started.solver.name}, phase={started.phase.name}, "
            f"activity={started.activity.name})"
        )
    gate_deadline = time.time() + GATE_TIMEOUT_S
    while time.time() < gate_deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        # A terminal solver state means this run is over. The solver reads
        # the save_and_quit marker at the top of a step, so a request sent
        # now produces no checkpoint and B would fail carrying a
        # checkpoint message for a pacing cause. Tested BEFORE the frame
        # test because the poll that reports the finish carries the final
        # frame count, which satisfies frame >= 1. The label is
        # trustworthy here: at most one status poll is in flight (the
        # effect runner has a single I/O worker and serves commands ahead
        # of polls), against a 3-deep starting_poll_guard that rewrites a
        # pre-start READY back to STARTING with frame 0.
        if s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            raise RuntimeError(
                f"ws-resume gate: the run reached {s.solver.name} at "
                f"frame={s.frame} before SaveAndQuitRequested could be "
                f"dispatched, so no checkpoint can come out of it; widen "
                f"the window with a larger PPF_EMULATED_STEP_MS "
                f"(error={(s.error or s.server_error or '')!r})"
            )
        if s.frame >= 1:
            break
        time.sleep(0.05)
    if dh.facade.engine.state.frame < 1:
        raise RuntimeError(
            f"ws-resume gate: solver never advanced past frame 0 within "
            f"{GATE_TIMEOUT_S:.0f}s "
            f"(frame={dh.facade.engine.state.frame}, "
            f"solver={dh.facade.engine.state.solver.name})"
        )
    dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
    dh.log(f"save_and_quit dispatched at frame={dh.facade.engine.state.frame}")
    saw_resumable = False
    deadline = time.time() + 120.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RESUMABLE":
            saw_resumable = True
            break
        if s.solver.name == "FAILED":
            break
        time.sleep(0.15)
    dh.record("B_checkpoint_reached", saw_resumable,
              {"solver": dh.facade.engine.state.solver.name})

    # E: the checkpoint is MEANINGFUL, i.e. it names a frame strictly
    # inside the run, so the resumed solver has frames left to produce and
    # C and D are reading them. B cannot show this: it passes on any
    # checkpoint, wherever it landed.
    #
    # The witness is the solver's own state_<N>.bin.gz. backend.rs
    # save_state names that file from state.curr_frame, so N is where the
    # checkpoint actually landed, whereas state.frame is the run progress
    # the driver already polled and reports the same number whether or not
    # a checkpoint exists. Read it BEFORE the resume: main.rs resolves the
    # --load -1 sentinel to the highest N on disk, so this is also the
    # frame the resumed solver starts from.
    #
    # Both ends of the range matter, for the same reason:
    #   * at LAST_FRAME the resumed solver is already done at its first
    #     loop visit, so every frame under test comes from the interrupted
    #     run and C compares the reference against itself.
    #   * at 0 the resume is a fresh start, because setup() reads load == 0
    #     as "wipe the output directory and run from the beginning", so C
    #     again compares two uninterrupted runs.
    # The gate above (frame >= 1) holds the low end off 0 and the pacing
    # holds the high end off LAST_FRAME, and this is what checks both.
    output_dir = os.path.join(
        dh.facade.engine.state.remote_root, "session", "output")
    state_files = sorted(glob.glob(os.path.join(output_dir, "state_*.bin.gz")))
    checkpoint_frames = []
    for path in state_files:
        stem = os.path.basename(path)[len("state_"):-len(".bin.gz")]
        if stem.isdigit():
            checkpoint_frames.append(int(stem))
    checkpoint_frame = max(checkpoint_frames) if checkpoint_frames else None
    dh.record(
        "E_checkpoint_mid_run",
        checkpoint_frame is not None and 0 < checkpoint_frame < LAST_FRAME,
        {"checkpoint_frame": checkpoint_frame,
         "last_frame": LAST_FRAME,
         "frames_after_checkpoint": (
             None if checkpoint_frame is None
             else LAST_FRAME - checkpoint_frame),
         "state_files": [os.path.basename(p) for p in state_files],
         "output_dir": output_dir},
    )

    dh.settle_idle(timeout=10.0)
    resumed_running = dh.resume_and_wait(timeout=180.0)
    # The resumed run has to have FINISHED before anything is captured.
    # resume_and_wait returns the same way on solver=FAILED and on its own
    # timeout as it does on success, and the capture below only reads what
    # is on disk, so a resume that died at some frame k would leave the
    # interrupted run's own frames in place and C would report a match for
    # a run that never happened.
    rs = dh.facade.engine.state
    if not (resumed_running and rs.solver.name in ("READY", "RESUMABLE")):
        raise RuntimeError(
            f"ws-resume resume: the resumed run did not reach a healthy "
            f"terminal state, so there is nothing to capture "
            f"(solver={rs.solver.name}, activity={rs.activity.name}, "
            f"frame={rs.frame}, saw_running={resumed_running}, "
            f"error={(rs.error or rs.server_error or '')!r})"
        )
    # Capture only: session/output now holds the interrupted run's frames
    # up to the checkpoint and the resumed solver's frames after it, which
    # is the span C and D are about.
    resumed = _capture(dh, plane)

    # C: resumed run reproduces the uninterrupted run (no double-scaling).
    same_shape = resumed.shape == ref.shape
    if same_shape:
        max_diff = float(np.max(np.abs(resumed - ref)))
    else:
        max_diff = float("inf")
    dh.record("C_resume_matches_run", same_shape and max_diff < 1e-3,
              {"max_abs_diff": round(max_diff, 8) if same_shape else None,
               "shape_ref": list(ref.shape), "shape_resumed": list(resumed.shape)})

    # D: resumed positions are at the AUTHORED scale (plane size 1 moved by
    # 0.4 -> |coords| ~ 1.0), not the x10 sim scale that a double-scale
    # would produce (|coords| ~ 10).
    max_abs = float(np.max(np.abs(resumed)))
    dh.record("D_authored_scale", max_abs < 3.0,
              {"max_abs_pos": round(max_abs, 5),
               "sim_scale_would_be": round(max_abs * WORLD_SCALING, 3)})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<WORLD_SCALING>>", repr(WORLD_SCALING))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
