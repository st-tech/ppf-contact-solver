# File: scenarios/bl_fetch_frame_discovery.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What the fetch asks the session directory for, and whether the answer
# reaches the fetch: ``EffectRunner._count_remote_frames``.
#
# The method lists ``vert_<N>.bin`` under ``<remote_root>/session/output`` and
# returns the highest N. That is a max frame INDEX, not a file count: the
# solver writes the rest pose as ``vert_0.bin`` through the same path as every
# other frame, so the directory always holds one more file than the number the
# method returns. ``_do_fetch_frames`` consumes it twice. Fetch All calls it
# unconditionally and takes its answer over the caller's ``frame_count``
# whenever it comes back positive, because the server's status ``frame`` field
# can lag the files already on disk. The live fetch calls it only when the
# caller has no frame number at all, and downloads the frame it names.
#
# A zero from it is indistinguishable at the caller from a solve that produced
# nothing. Fetch All then dispatches ``FetchComplete(total_frames=0)``, no
# frame is applied, no PC2 is written, no MESH_CACHE modifier is attached, and
# the artist sees a finished run with no animation and no error anywhere near
# the cause.
#
# THE OBVIOUS CHEAPER TEST CANNOT SEE THIS, which is the whole reason this
# scenario exists. An end-to-end fetch never observes the return value: the
# caller uses it only when it is greater than zero, and every rig driver walks
# ``force_frame_query`` until ``state.frame`` matches the run, so the
# ``frame_count`` already handed in is correct and the override changes
# nothing. Replacing the body with ``return 0`` therefore leaves the full-fetch
# scenarios green (measured on ``bl_fetch_failed_watchdog`` and
# ``bl_world_scaling_shell_drape``, each of which builds, runs, fetches and
# drains). This scenario removes what masks it: it hands ``_do_fetch_frames`` a
# ``frame_count`` of 0, which leaves the discovery as the only thing that can
# size the fetch.
#
# WHY IT CALLS PRIVATE METHODS. Subtests B, C and F call
# ``_count_remote_frames`` directly because the assertion needs the exact
# integer and the value has no public accessor. Subtests D and E call
# ``_do_fetch_frames`` directly because ``EffectRunner.execute`` only hands the
# same call to the I/O worker, and neither branch emits a signal that separates
# "discovery answered zero" from "the worker has not started yet", so a
# synchronous call is what makes a red run red at once rather than after a
# timeout. The public route (Fetch button, transitions, ``execute``) runs in
# this same scenario as the ``fetch_and_drain`` that sets these checks up, and
# ``bl_fetch_clear_refetch`` and ``bl_progress_fetching`` cover it directly.
#
# THE COUNTER RESET BEFORE D AND E CARRIES WEIGHT. ``_do_fetch_frames``
# returns before it touches ``_anim_total`` when the frame count lands below 1,
# so the total left behind by the setup fetch would answer for the probe and
# report a pass. Each probe starts from ``DoResetAnimationBuffer`` plus
# ``clear_fetched_frames`` so the numbers it reads can only have come from it.
#
# Subtests:
#   A. frames_present_on_disk
#         The setup fetch worked and the session directory holds a contiguous
#         ``vert_0..vert_N``. Everything below compares against N, so a broken
#         pipeline names itself here instead of failing five checks at once.
#   B. discovery_returns_the_true_max_frame_index
#         ``_count_remote_frames`` equals the N this driver read off the same
#         directory itself.
#   C. discovery_is_the_max_index_not_the_file_count
#         The directory holds N+1 files and the answer is N. An implementation
#         that returns the file count is off by exactly the rest pose, which
#         makes Fetch All ask for a ``vert_<N+1>.bin`` that does not exist.
#   D. fetch_all_sizes_itself_from_discovery
#         Fetch All with ``frame_count=0`` still queues frames 1..N and sets
#         ``_anim_total`` to N.
#   E. live_fetch_falls_back_to_discovery
#         The live-fetch branch with ``frame_count=0`` queues exactly frame N.
#   F. local_discovery_does_not_shell_out
#         The branch taken for ``local`` and ``win_native`` reads the output
#         directory off this machine, because both backends are co-located
#         with the server. A ``ls -1 vert_*.bin`` shell-out returns the same
#         numbers on Linux and macOS, so a result comparison alone cannot tell
#         the two implementations apart there. This subtest asks the question
#         that does: with the backend's ``exec_command`` replaced by one
#         answering the way cmd.exe answers for ``ls`` (exit code 9009, empty
#         stdout), a filesystem discovery still returns N and never calls it,
#         while a shell-based one returns 0. That reproduces the Windows
#         failure on any host.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND. The discovery reads a directory the solver filled
# and nothing it touches is backend-specific, so it holds for any backend that
# writes those frames. ``dh.connect`` picks LOCAL off Windows and WIN_NATIVE on
# it; both take the same filesystem branch.
BACKENDS = ("real",)

# NO KNOBS, and no pacing is wanted. This scenario does not observe a run in
# progress: the eight frames a solve of this sheet takes arrive well inside the
# waits below, and the discovery under test reads a directory listing. A
# `PPF_STEP_DELAY_MS` here would buy nothing and slow the sweep.
KNOBS = {}

_FRAME_COUNT = 8


_DRIVER_BODY = r"""
import glob
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>


def vert_indices(output_dir):
    # Every N for which vert_<N>.bin exists, read off the session directory by
    # this driver rather than by the addon, so the addon's answer has an
    # independent number to be compared against. An in-progress write is named
    # vert_<N>.bin.tmp and does not end in .bin, so it cannot appear here.
    found = []
    for path in glob.glob(os.path.join(output_dir, "vert_*.bin")):
        name = os.path.basename(path)
        try:
            found.append(int(name[len("vert_"):-len(".bin")]))
        except ValueError:
            continue
    return sorted(found)


try:
    dh = DriverHelpers(pkg, result)
    effects = __import__(pkg + ".core.effects",
                         fromlist=["DoResetAnimationBuffer"])

    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=4, y_subdivisions=4, size=1.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "DiscoverySheet"
    # A pinned top edge and a free remainder: enough for the real solver to
    # have DOFs to solve, small enough that eight frames cost nothing. No
    # subtest reads a position, so the motion itself carries no assertion.
    pinned_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.49]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned_idx, 1.0, "REPLACE")
    dh.save_blend(PROBE_DIR, "fetch_frame_discovery.blend")

    root = dh.configure_state(
        project_name="fetch_frame_discovery",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="fetch_frame_discovery:build", timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)

    runner = dh.facade.runner
    remote_root = dh.facade.engine.state.remote_root
    output_dir = os.path.join(remote_root, "session", "output")

    # The solver is terminal by now, so the listing settles and stays settled.
    # Wait for it anyway: a rename that has not landed yet would make every
    # number below one short, which reads exactly like a discovery defect.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        seen = vert_indices(output_dir)
        if seen and seen[-1] >= FRAME_COUNT - 1:
            break
        time.sleep(0.2)

    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total} solver={solver_state}")
    # Drain the live-fetch side effects the polling above may have queued, so
    # the I/O worker is idle while the probes below run on the main thread.
    dh.settle_idle(timeout=15.0)

    on_disk = vert_indices(output_dir)
    max_index = on_disk[-1] if on_disk else 0
    contiguous = on_disk == list(range(0, max_index + 1))
    pc2_path = dh.find_pc2_for(sheet)

    # ----- A: the ground the rest of the scenario stands on --------------
    dh.record(
        "A_frames_present_on_disk",
        max_index >= 1
        and contiguous
        and pc2_path is not None
        and solver_state != "FAILED",
        {
            "output_dir": output_dir,
            "indices": on_disk,
            "max_index": max_index,
            "contiguous": contiguous,
            "pc2_path": pc2_path,
            "solver_state": solver_state,
            "applied": applied,
            "total": total,
            "state_frame": int(dh.facade.engine.state.frame),
        },
    )

    # ----- B / C: the value itself ---------------------------------------
    discovered = runner._count_remote_frames(remote_root)
    dh.record(
        "B_discovery_returns_the_true_max_frame_index",
        max_index >= 1 and discovered == max_index,
        {"discovered": discovered, "max_index": max_index,
         "backend_type": runner.backend.backend_type},
    )
    dh.record(
        "C_discovery_is_the_max_index_not_the_file_count",
        discovered == max_index and len(on_disk) == max_index + 1,
        {"discovered": discovered, "file_count": len(on_disk),
         "max_index": max_index},
    )

    # ----- D: Fetch All sized by the discovery alone ----------------------
    # frame_count=0 is the case the caller's own comment describes: the status
    # number lags or is missing, and the range downloaded has to come from the
    # files on disk. The reset is what makes the reading honest; without it the
    # setup fetch's _anim_total would still be standing.
    runner.execute(effects.DoResetAnimationBuffer())
    runner.clear_fetched_frames()
    runner._do_fetch_frames(remote_root, 0, runner._fetched, False)
    with runner._anim_lock:
        queued_all = sorted(frame[0] for frame in runner._anim_frames)
        total_all = runner._anim_total
    dh.record(
        "D_fetch_all_sizes_itself_from_discovery",
        max_index >= 1
        and total_all == max_index
        and queued_all == list(range(1, max_index + 1)),
        {"anim_total": total_all, "queued": queued_all,
         "expected": list(range(1, max_index + 1))},
    )

    # ----- E: the live-fetch fallback -------------------------------------
    runner.execute(effects.DoResetAnimationBuffer())
    runner.clear_fetched_frames()
    runner._do_fetch_frames(remote_root, 0, runner._fetched, True)
    with runner._anim_lock:
        queued_latest = sorted(frame[0] for frame in runner._anim_frames)
    dh.record(
        "E_live_fetch_falls_back_to_discovery",
        max_index >= 1 and queued_latest == [max_index],
        {"queued": queued_latest, "max_index": max_index},
    )

    # ----- F: a co-located backend reads the disk, not a shell -------------
    shell_calls = []

    def refuse_shell(command, *, shell=False, cwd=None, timeout=None):
        # What cmd.exe answers when asked to run `ls`: nothing on stdout and
        # 9009, its code for a command it cannot find.
        shell_calls.append(command)
        return {
            "exit_code": 9009,
            "stdout": [],
            "stderr": ["'ls' is not recognized as an internal or external "
                       "command, operable program or batch file."],
        }

    backend = runner.backend
    backend.exec_command = refuse_shell
    try:
        discovered_no_shell = runner._count_remote_frames(remote_root)
    finally:
        del backend.exec_command
    dh.record(
        "F_local_discovery_does_not_shell_out",
        max_index >= 1
        and discovered_no_shell == max_index
        and shell_calls == [],
        {"discovered": discovered_no_shell, "max_index": max_index,
         "shell_calls": shell_calls,
         "backend_type": backend.backend_type},
    )

    # Hand the runner back the way the setup left it: the probes above queued
    # frames that nothing is going to apply.
    runner.execute(effects.DoResetAnimationBuffer())
    runner.clear_fetched_frames()

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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
