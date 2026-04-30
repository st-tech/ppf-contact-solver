# File: scenarios/bl_transition_chains.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Long chain of state transitions in one Blender process. Each link
# in the chain records its own pass/fail, so the suite surfaces the
# exact transition that broke. Catches the bug class where individual
# operations are correct in isolation but combine into a stuck or
# stale state across rebuilds, reconnects, and aborts.
#
# Steps:
#   01_connect               connect_local -> ONLINE/RUNNING
#   02_initial_transfer      first BuildPipeline; data+param hashes
#                            land on the server, build settles READY.
#   03_initial_run           run -> RUNNING -> READY; frames produced.
#   04_initial_fetch         fetch+drain; PC2 + MESH_CACHE present.
#   05_clear_animation       clear_animation_data wipes PC2 + state.
#   06_rerun_after_clear     run again on the same build, frames
#                            produced; state.frame > 0.
#   07_fetch_after_rerun     fetch+drain again; PC2 size != 0.
#   08_save_and_quit_midrun  start a fresh run, dispatch
#                            SaveAndQuitRequested mid-run, settle
#                            on RESUMABLE with state_*.bin.gz on disk.
#   09_resume_to_completion  resume, run completes, fetch all frames.
#   10_param_retransfer      mutate a parameter, rebuild, run,
#                            assert the simulation reflects the
#                            new param.
#   11_abort_midbuild        kick off a new build, dispatch
#                            AbortRequested, settle to IDLE,
#                            assert pending_build cleared.
#   12_retransfer_after_abort  fresh build pipeline succeeds.
#   13_disconnect            disconnect; phase=OFFLINE,
#                            runner._backend=None, project_name="".
#   14_reconnect_restores    reconnect; server echoes the persisted
#                            hashes; engine.state.server_data_hash
#                            and server_param_hash are non-empty
#                            and match what the client uploaded.
#   15_run_after_reconnect   final run; sim still works end-to-end
#                            against the reconnected server.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


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


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])

    plane = dh.reset_scene_to_pinned_plane(name="ChainMesh")
    dh.save_blend(PROBE_DIR, "chain.blend")
    root = dh.configure_state(project_name="transition_chains",
                              frame_count=10)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=8,
                transition="LINEAR")

    state = root.state

    def encode_payload():
        return (encoder_mesh.compute_data_hash(bpy.context),
                encoder_params.compute_param_hash(bpy.context),
                dh.encoder_mesh.encode_obj(bpy.context),
                dh.encoder_param.encode_param(bpy.context))

    def build_pipeline_dispatch(message):
        dh_data, dh_param, data_bytes, param_bytes = encode_payload()
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes,
            data_hash=dh_data, param_hash=dh_param,
            message=message,
        ))
        return dh_data, dh_param

    def wait_until(predicate, *, timeout, label):
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            last_state = dh.facade.engine.state
            if predicate(last_state):
                return last_state
            time.sleep(0.2)
        raise RuntimeError(
            f"{label} timed out after {timeout}s; activity="
            f"{last_state.activity.name if last_state else '?'} "
            f"solver={last_state.solver.name if last_state else '?'}"
        )

    def step(name, fn):
        if result.get("errors"):
            dh.record(name, False, {"skipped": "earlier step raised"})
            return
        try:
            details = fn() or {}
            dh.record(name, True, details)
            dh.log(f"PASS {name}")
        except Exception as exc:
            dh.record(name, False, {
                "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc()[-500:],
            })
            dh.log(f"FAIL {name}: {exc}")

    def output_state_files():
        return sorted(glob.glob(os.path.join(
            dh.facade.engine.state.remote_root, "session", "output",
            "state_*.bin.gz",
        )))

    # ------------------------------------------------------------------
    def s01_connect():
        dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                         project_name=root.state.project_name)
        s = dh.facade.engine.state
        assert s.phase.name == "ONLINE"
        assert s.server.name == "RUNNING"
        return {"phase": s.phase.name, "server": s.server.name}

    def s02_initial_transfer():
        dh_data, dh_param = build_pipeline_dispatch("chain:initial")
        s = wait_until(
            lambda s: s.activity.name == "IDLE"
                      and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
            timeout=90.0, label="initial transfer",
        )
        assert s.solver.name in ("READY", "RESUMABLE")
        assert s.server_data_hash == dh_data, (
            f"server_data_hash {s.server_data_hash[:8]} != client "
            f"{dh_data[:8]}"
        )
        assert s.server_param_hash == dh_param
        assert s.server_upload_id, "missing upload_id after upload"
        return {"upload_id": s.server_upload_id[:12],
                "data": dh_data[:12], "param": dh_param[:12]}

    def s03_initial_run():
        dh.run_and_wait(timeout=120.0)
        dh.force_frame_query(expected_frames=1, timeout=10.0)
        s = dh.facade.engine.state
        assert s.solver.name in ("READY", "RESUMABLE"), s.solver.name
        assert s.frame >= 1
        return {"frame": s.frame, "solver": s.solver.name}

    def s04_initial_fetch():
        dh.settle_idle(timeout=15.0)
        applied, total = dh.fetch_and_drain()
        assert total > 0 and applied == total, (applied, total)
        pc2 = dh.find_pc2_for(plane)
        assert pc2 and os.path.isfile(pc2)
        assert dh.has_mesh_cache(plane)
        return {"applied": applied, "total": total,
                "pc2_size": os.path.getsize(pc2)}

    def s05_clear_animation():
        animation_mod.clear_animation_data(bpy.context)
        assert not dh.has_mesh_cache(plane), "MESH_CACHE still attached"
        assert len(state.fetched_frame) == 0
        with dh.facade.runner._anim_lock:
            assert len(dh.facade.runner._anim_frames) == 0
        return {"fetched_frame_count": len(state.fetched_frame)}

    def s06_rerun_after_clear():
        dh.run_and_wait(timeout=120.0)
        dh.force_frame_query(expected_frames=1, timeout=10.0)
        s = dh.facade.engine.state
        assert s.solver.name in ("READY", "RESUMABLE")
        assert s.frame >= 1
        return {"frame": s.frame}

    def s07_fetch_after_rerun():
        dh.settle_idle(timeout=15.0)
        applied, total = dh.fetch_and_drain()
        assert total > 0 and applied == total
        return {"applied": applied, "total": total}

    def s08_save_and_quit_midrun():
        dh.com.run()
        saw_running = False
        save_dispatched = False
        deadline = time.time() + 90.0
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            s = dh.facade.engine.state
            if s.solver.name == "RUNNING":
                saw_running = True
            if saw_running and not save_dispatched and s.frame >= 1:
                dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
                save_dispatched = True
            if save_dispatched and s.solver.name in ("RESUMABLE", "FAILED"):
                break
            time.sleep(0.2)
        assert save_dispatched, "save_and_quit was never dispatched"
        s = dh.facade.engine.state
        assert s.solver.name == "RESUMABLE", s.solver.name
        states = output_state_files()
        assert states, "no state_*.bin.gz on disk"
        return {"solver": s.solver.name, "saves": len(states)}

    def s09_resume_to_completion():
        dh.settle_idle(timeout=10.0)
        saw = dh.resume_and_wait(timeout=120.0)
        assert saw, "resume never saw RUNNING"
        dh.force_frame_query(expected_frames=root.state.frame_count - 1,
                             timeout=20.0)
        dh.settle_idle(timeout=15.0)
        applied, total = dh.fetch_and_drain()
        assert total > 0 and applied == total
        return {"applied": applied, "total": total}

    def s10_param_retransfer():
        # Mutate a parameter, re-encode, push, re-run, fetch, assert
        # the new value actually round-tripped.
        original_delta = tuple(
            root.object_group_0.pin_vertex_groups[0].operations[0].delta
        )
        new_delta = (0.5, 0.0, 0.0)
        root.object_group_0.pin_vertex_groups[0].operations[0].delta = new_delta
        dh_data, dh_param = build_pipeline_dispatch("chain:param-retransfer")
        wait_until(
            lambda s: s.activity.name == "IDLE"
                      and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
            timeout=90.0, label="param retransfer build",
        )
        s = dh.facade.engine.state
        assert s.solver.name in ("READY", "RESUMABLE"), (
            f"solver={s.solver.name!r} after retransfer build "
            f"(error={s.error!r}, server_error={s.server_error!r})"
        )
        assert s.server_param_hash == dh_param, (
            f"param_hash mismatch: server={s.server_param_hash[:12]!r} "
            f"vs client={dh_param[:12]!r}"
        )
        dh.run_and_wait(timeout=120.0)
        dh.force_frame_query(expected_frames=root.state.frame_count - 1,
                             timeout=20.0)
        dh.settle_idle(timeout=15.0)
        applied, total = dh.fetch_and_drain()
        pc2 = dh.find_pc2_for(plane)
        assert pc2 and os.path.isfile(pc2), (
            f"pc2 not on disk after run+fetch (pc2={pc2!r}, "
            f"applied={applied}/{total})"
        )
        arr = dh.read_pc2(pc2)
        last_shift = float(arr[-1][0][0] - arr[0][0][0])
        # Allow a small tolerance for solver-step rounding.
        assert abs(last_shift - new_delta[0]) < 1e-2, (
            f"last_shift={last_shift:.4f}, expected ~{new_delta[0]}"
        )
        return {"original_delta": original_delta, "new_delta": new_delta,
                "last_shift": last_shift, "applied": applied, "total": total}

    def s11_abort_midbuild():
        # Fire build then abort in the same tick. AbortRequested has
        # no busy guard so it should always set ABORTING and clear
        # pending_build.
        dh_data, dh_param, data_bytes, param_bytes = encode_payload()
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes,
            data_hash=dh_data, param_hash=dh_param,
            message="chain:abort",
        ))
        dh.facade.engine.dispatch(dh.events.AbortRequested())
        dh.facade.tick()
        s = dh.facade.engine.state
        assert s.activity.name == "ABORTING"
        assert s.pending_build is False
        # Drain to IDLE so the next step starts clean.
        wait_until(
            lambda s: s.activity.name == "IDLE",
            timeout=30.0, label="abort settle",
        )
        return {"post_abort_solver": dh.facade.engine.state.solver.name}

    def s12_retransfer_after_abort():
        dh_data, dh_param = build_pipeline_dispatch("chain:after-abort")
        wait_until(
            lambda s: s.activity.name == "IDLE"
                      and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
            timeout=90.0, label="post-abort transfer",
        )
        s = dh.facade.engine.state
        assert s.solver.name in ("READY", "RESUMABLE")
        assert s.server_data_hash == dh_data
        assert s.server_param_hash == dh_param
        return {"upload_id": s.server_upload_id[:12]}

    def s13_disconnect():
        dh.com.disconnect()
        # Drain any pending events.
        for _ in range(5):
            dh.facade.tick()
            time.sleep(0.05)
        s = dh.facade.engine.state
        assert s.phase.name == "OFFLINE"
        assert dh.facade.runner._backend is None
        assert not dh.facade.runner.project_name
        return {"phase": s.phase.name}

    def s14_reconnect_restores():
        dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                         project_name=root.state.project_name)
        # Force a query so the response carries the persisted
        # data_hash / param_hash from disk.
        wait_until(
            lambda s: bool(s.server_data_hash) and bool(s.server_param_hash),
            timeout=15.0, label="reconnect hash sync",
        )
        s = dh.facade.engine.state
        # Compare the live scene's hashes against what the server
        # persisted across the reconnect.
        fresh_data = encoder_mesh.compute_data_hash(bpy.context)
        fresh_param = encoder_params.compute_param_hash(bpy.context)
        assert s.server_data_hash == fresh_data, (
            s.server_data_hash[:12], fresh_data[:12]
        )
        assert s.server_param_hash == fresh_param
        return {"data": s.server_data_hash[:12],
                "param": s.server_param_hash[:12]}

    def s15_run_after_reconnect():
        dh.run_and_wait(timeout=120.0)
        dh.force_frame_query(expected_frames=root.state.frame_count - 1,
                             timeout=20.0)
        dh.settle_idle(timeout=15.0)
        applied, total = dh.fetch_and_drain()
        assert total > 0 and applied == total
        return {"applied": applied, "total": total}

    # ------------------------------------------------------------------
    step("01_connect", s01_connect)
    step("02_initial_transfer", s02_initial_transfer)
    step("03_initial_run", s03_initial_run)
    step("04_initial_fetch", s04_initial_fetch)
    step("05_clear_animation", s05_clear_animation)
    step("06_rerun_after_clear", s06_rerun_after_clear)
    step("07_fetch_after_rerun", s07_fetch_after_rerun)
    step("08_save_and_quit_midrun", s08_save_and_quit_midrun)
    step("09_resume_to_completion", s09_resume_to_completion)
    step("10_param_retransfer", s10_param_retransfer)
    step("11_abort_midbuild", s11_abort_midbuild)
    step("12_retransfer_after_abort", s12_retransfer_after_abort)
    step("13_disconnect", s13_disconnect)
    step("14_reconnect_restores", s14_reconnect_restores)
    step("15_run_after_reconnect", s15_run_after_reconnect)

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
