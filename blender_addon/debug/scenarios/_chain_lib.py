# File: scenarios/_chain_lib.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared step library for the ``bl_chain_*`` scenarios.
#
# Each chain scenario specifies a sequence of step names from
# ``STEP_CATALOG`` and uses :data:`CHAIN_LIB` (a Python source
# fragment) inside its driver to set up the harness:
#
#   * ``ChainHarness`` — wraps DriverHelpers and the addon imports the
#     steps need (``encoder.mesh.compute_data_hash``,
#     ``encoder.params.compute_param_hash``, ``core.animation.
#     clear_animation_data``).
#   * ``run_steps(harness, sequence)`` — invokes each named step and
#     records pass/fail on ``result["checks"]`` keyed by the position
#     in the sequence (so a chain that visits ``initial_run`` twice
#     records both visits independently).
#
# Adding a new building block: append to ``STEP_CATALOG`` here and any
# chain that lists it picks it up. The orderings live in
# ``bl_chain_*.py``; this module owns the steps themselves so the
# scenarios stay short and the test-coverage matrix stays
# enumerable in one place.

from __future__ import annotations


# Step names referenced by ``bl_chain_*.py``. The actual step bodies
# are defined in ``CHAIN_LIB`` below; this list documents the
# vocabulary at a glance.
STEP_NAMES = (
    "connect",
    "transfer",                  # initial transfer (data + param)
    "param_retransfer",          # mutate a param, re-encode, re-build
    "data_retransfer",           # add a vertex, re-encode, re-build
    "run",                       # full sim run (start → finish)
    "fetch",                     # explicit fetch + drain modal
    "clear_animation",           # animation_mod.clear_animation_data
    "save_and_quit",             # mid-run save_and_quit
    "resume",                    # resume from saved state
    "abort_midbuild",            # build + abort same tick
    "abort_during_run",          # run, then abort while RUNNING
    "disconnect",                # com.disconnect()
    "reconnect",                 # connect_local again, hashes echo
    "stop_server",               # com.stop_server() — kills the proc
    "start_server",              # com.start_server() — relaunches
    "verify_idle",               # asserts activity=IDLE, no work in flight
    "verify_pc2",                # asserts find_pc2_for(plane) is non-empty
    "verify_no_pc2",             # asserts no PC2 file (post-clear)
    "verify_resumable",          # asserts solver=RESUMABLE
)


_CHAIN_DRIVER_TEMPLATE = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
MESH_NAME = "<<MESH_NAME>>"
SEQUENCE = <<SEQUENCE>>


try:
    dh = DriverHelpers(pkg, result)
    plane = dh.reset_scene_to_pinned_plane(name=MESH_NAME)
    dh.save_blend(PROBE_DIR, PROJECT_NAME + ".blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=10)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=8,
                transition="LINEAR")
    harness = ChainHarness(dh, plane, root, LOCAL_PATH, SERVER_PORT)
    run_steps(harness, SEQUENCE)
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_chain_driver(ctx, *, project_name, mesh_name, sequence):
    """Return the Python source the bootstrap will exec for a chain
    scenario. The chain author only specifies the project name, the
    mesh name (so concurrent workers don't collide on object names),
    and the sequence of step keys."""
    import json
    import os
    from . import _driver_lib as _dl

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    body = _CHAIN_DRIVER_TEMPLATE
    body = body.replace("<<LOCAL_PATH>>", repo_root)
    body = body.replace("<<SERVER_PORT>>", str(ctx.server_port))
    body = body.replace("<<PROJECT_NAME>>", project_name)
    body = body.replace("<<MESH_NAME>>", mesh_name)
    body = body.replace("<<SEQUENCE>>", json.dumps(list(sequence)))
    return _dl.DRIVER_LIB + CHAIN_LIB + body


CHAIN_LIB = r"""
import glob
import os
import time
import traceback


class ChainHarness:
    # Wraps DriverHelpers + the auxiliary modules every step needs,
    # plus helpers that several steps share (encode_payload,
    # build_pipeline_dispatch, wait_until).

    def __init__(self, dh, plane, root, local_path, server_port):
        self.dh = dh
        self.plane = plane
        self.root = root
        self.local_path = local_path
        self.server_port = server_port
        self.encoder_mesh = __import__(
            dh.pkg + ".core.encoder.mesh",
            fromlist=["compute_data_hash", "encode_obj"],
        )
        self.encoder_params = __import__(
            dh.pkg + ".core.encoder.params",
            fromlist=["compute_param_hash", "encode_param"],
        )
        self.animation_mod = __import__(
            dh.pkg + ".core.animation",
            fromlist=["clear_animation_data"],
        )
        # The first step that mutates the pin op flips this so
        # ``data_retransfer`` knows to add a vertex instead of
        # re-using the original. Avoids cross-talk between repeated
        # data_retransfer calls in the same chain.
        self._param_delta_step = 0
        self._data_vertex_step = 0

    def encode_payload(self):
        ctx = bpy.context
        return (
            self.encoder_mesh.compute_data_hash(ctx),
            self.encoder_params.compute_param_hash(ctx),
            self.encoder_mesh.encode_obj(ctx),
            self.encoder_params.encode_param(ctx),
        )

    def build_pipeline_dispatch(self, message):
        data_hash, param_hash, data_bytes, param_bytes = self.encode_payload()
        self.dh.facade.engine.dispatch(
            self.dh.events.BuildPipelineRequested(
                data=data_bytes, param=param_bytes,
                data_hash=data_hash, param_hash=param_hash,
                message=message,
            )
        )
        return data_hash, param_hash

    def wait_until(self, predicate, *, timeout, label):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            self.dh.facade.engine.dispatch(self.dh.events.PollTick())
            self.dh.facade.tick()
            last = self.dh.facade.engine.state
            if predicate(last):
                return last
            time.sleep(0.2)
        raise RuntimeError(
            f"{label} timed out after {timeout}s; activity="
            f"{last.activity.name if last else '?'} "
            f"solver={last.solver.name if last else '?'}"
        )

    def output_state_files(self):
        root = self.dh.facade.engine.state.remote_root
        if not root:
            return []
        return sorted(glob.glob(os.path.join(
            root, "session", "output", "state_*.bin.gz",
        )))


# ---- step bodies --------------------------------------------------

def step_connect(h):
    h.dh.connect_local(local_path=h.local_path, server_port=h.server_port,
                       project_name=h.root.state.project_name)
    s = h.dh.facade.engine.state
    assert s.phase.name == "ONLINE"
    assert s.server.name == "RUNNING"
    return {"phase": s.phase.name, "server": s.server.name}


def step_transfer(h):
    dh_data, dh_param = h.build_pipeline_dispatch("chain:transfer")
    s = h.wait_until(
        lambda s: s.activity.name == "IDLE"
                  and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
        timeout=90.0, label="transfer",
    )
    assert s.solver.name in ("READY", "RESUMABLE")
    assert s.server_data_hash == dh_data
    assert s.server_param_hash == dh_param
    assert s.server_upload_id
    return {"upload_id": s.server_upload_id[:12]}


def step_param_retransfer(h):
    h._param_delta_step += 1
    pin_op = h.root.object_group_0.pin_vertex_groups[0].operations[0]
    # Sweep through unique deltas so repeated invocations actually
    # change the param hash.
    pin_op.delta = (0.05 + 0.10 * h._param_delta_step, 0.0, 0.0)
    dh_data, dh_param = h.build_pipeline_dispatch("chain:param_retransfer")
    s = h.wait_until(
        lambda s: s.activity.name == "IDLE"
                  and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
        timeout=90.0, label="param_retransfer",
    )
    assert s.solver.name in ("READY", "RESUMABLE"), (
        f"solver={s.solver.name!r} after retransfer (error={s.error!r}, "
        f"server_error={s.server_error!r})"
    )
    assert s.server_param_hash == dh_param, (
        f"server_param_hash mismatch: server={s.server_param_hash[:12]!r} "
        f"vs dispatched={dh_param[:12]!r} (solver={s.solver.name})"
    )
    return {"new_delta": tuple(pin_op.delta), "param": dh_param[:12]}


def step_data_retransfer(h):
    # Subdivide the existing plane so the topology actually changes
    # (more verts, more edges, more faces) but the result is a
    # well-formed shell the solver can build and simulate. A bare
    # ``bm.verts.new`` would create an unreferenced vertex and the
    # solver would panic on the malformed mesh — we want the test to
    # exercise legitimate "user kept editing the scene then re-
    # transferred" flows, not invalid-input crashes.
    import bmesh
    h._data_vertex_step += 1
    bm = bmesh.new()
    bm.from_mesh(h.plane.data)
    bmesh.ops.subdivide_edges(
        bm, edges=list(bm.edges), cuts=1, use_grid_fill=True,
    )
    bm.to_mesh(h.plane.data)
    bm.free()
    h.plane.data.update()
    # Reassign the AllPin vertex group across every vertex of the
    # subdivided mesh so the sim still has something to pin.
    vg = h.plane.vertex_groups.get("AllPin")
    if vg is None:
        vg = h.plane.vertex_groups.new(name="AllPin")
    n_verts = len(h.plane.data.vertices)
    vg.add(list(range(n_verts)), 1.0, "REPLACE")
    dh_data, dh_param = h.build_pipeline_dispatch("chain:data_retransfer")
    s = h.wait_until(
        lambda s: s.activity.name == "IDLE"
                  and s.solver.name in ("READY", "RESUMABLE", "FAILED"),
        timeout=120.0, label="data_retransfer",
    )
    if s.solver.name not in ("READY", "RESUMABLE"):
        raise AssertionError(
            f"solver={s.solver.name} after data_retransfer "
            f"(error={s.error!r}, server_error={s.server_error!r})"
        )
    assert s.server_data_hash == dh_data
    return {"verts": len(h.plane.data.vertices), "data": dh_data[:12]}


def step_run(h):
    # Budgets sized for slow CI runners. Free GitHub Linux runners can
    # take 200+ seconds for a chain run that the local dev box clears in
    # 30s; without this padding the test fires "saw_running=False" even
    # though the solver did reach RUNNING, just past the previous window.
    saw_running = h.dh.run_and_wait(timeout=240.0)
    h.dh.force_frame_query(expected_frames=1, timeout=30.0)
    s = h.dh.facade.engine.state
    if s.solver.name not in ("READY", "RESUMABLE"):
        raise AssertionError(
            f"solver={s.solver.name} (saw_running={saw_running}, "
            f"error={s.error!r}, server_error={s.server_error!r}, "
            f"frame={s.frame})"
        )
    if s.frame < 1:
        raise AssertionError(
            f"state.frame={s.frame} < 1 (solver={s.solver.name}, "
            f"saw_running={saw_running})"
        )
    return {"frame": s.frame, "solver": s.solver.name,
            "saw_running": saw_running}


def step_fetch(h):
    h.dh.settle_idle(timeout=15.0)
    applied, total = h.dh.fetch_and_drain()
    assert total > 0 and applied == total, (applied, total)
    pc2 = h.dh.find_pc2_for(h.plane)
    assert pc2 and os.path.isfile(pc2)
    return {"applied": applied, "total": total,
            "pc2_size": os.path.getsize(pc2)}


def step_clear_animation(h):
    h.animation_mod.clear_animation_data(bpy.context)
    assert not h.dh.has_mesh_cache(h.plane)
    assert len(h.root.state.fetched_frame) == 0
    return {"fetched_frame_count": len(h.root.state.fetched_frame)}


def step_save_and_quit(h):
    h.dh.com.run()
    saw_running = False
    save_dispatched = False
    deadline = time.time() + 90.0
    while time.time() < deadline:
        h.dh.facade.engine.dispatch(h.dh.events.PollTick())
        h.dh.facade.tick()
        s = h.dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and not save_dispatched and s.frame >= 1:
            h.dh.facade.engine.dispatch(h.dh.events.SaveAndQuitRequested())
            save_dispatched = True
        if save_dispatched and s.solver.name in ("RESUMABLE", "FAILED"):
            break
        time.sleep(0.2)
    assert save_dispatched
    s = h.dh.facade.engine.state
    assert s.solver.name == "RESUMABLE"
    saves = h.output_state_files()
    assert saves
    return {"solver": s.solver.name, "saves": len(saves)}


def step_resume(h):
    h.dh.settle_idle(timeout=10.0)
    saw = h.dh.resume_and_wait(timeout=120.0)
    s = h.dh.facade.engine.state
    assert saw, (
        f"resume never saw RUNNING (state.frame={s.frame}, "
        f"solver={s.solver.name}, error={s.error!r})"
    )
    h.dh.force_frame_query(expected_frames=h.root.state.frame_count - 1,
                           timeout=20.0)
    s = h.dh.facade.engine.state
    assert s.solver.name in ("READY", "RESUMABLE"), (
        f"solver={s.solver.name!r} after resume (frame={s.frame}, "
        f"error={s.error!r})"
    )
    return {"solver": s.solver.name}


def step_abort_midbuild(h):
    dh_data, dh_param, data_bytes, param_bytes = h.encode_payload()
    h.dh.facade.engine.dispatch(h.dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=dh_data, param_hash=dh_param,
        message="chain:abort_midbuild",
    ))
    h.dh.facade.engine.dispatch(h.dh.events.AbortRequested())
    h.dh.facade.tick()
    s = h.dh.facade.engine.state
    assert s.activity.name == "ABORTING"
    assert s.pending_build is False
    h.wait_until(
        lambda s: s.activity.name == "IDLE",
        timeout=30.0, label="abort settle",
    )
    return {"post_abort_solver": h.dh.facade.engine.state.solver.name}


def step_abort_during_run(h):
    # Run-time abort: kick a run, dispatch AbortRequested while solver
    # is RUNNING, settle to IDLE.
    h.dh.com.run()
    saw_running = False
    aborted = False
    deadline = time.time() + 60.0
    while time.time() < deadline:
        h.dh.facade.engine.dispatch(h.dh.events.PollTick())
        h.dh.facade.tick()
        s = h.dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and not aborted:
            h.dh.facade.engine.dispatch(h.dh.events.AbortRequested())
            aborted = True
        if aborted and s.activity.name == "IDLE":
            break
        time.sleep(0.1)
    assert aborted
    s = h.dh.facade.engine.state
    assert s.activity.name == "IDLE"
    return {"saw_running": saw_running, "solver": s.solver.name}


def step_disconnect(h):
    h.dh.com.disconnect()
    for _ in range(5):
        h.dh.facade.tick()
        time.sleep(0.05)
    s = h.dh.facade.engine.state
    assert s.phase.name == "OFFLINE"
    assert h.dh.facade.runner._backend is None
    assert not h.dh.facade.runner.project_name
    return {"phase": s.phase.name}


def step_reconnect(h):
    h.dh.connect_local(local_path=h.local_path, server_port=h.server_port,
                       project_name=h.root.state.project_name)
    h.wait_until(
        lambda s: bool(s.server_data_hash) and bool(s.server_param_hash),
        timeout=15.0, label="reconnect hash sync",
    )
    s = h.dh.facade.engine.state
    fresh_data = h.encoder_mesh.compute_data_hash(bpy.context)
    fresh_param = h.encoder_params.compute_param_hash(bpy.context)
    assert s.server_data_hash == fresh_data
    assert s.server_param_hash == fresh_param
    return {"data": s.server_data_hash[:12],
            "param": s.server_param_hash[:12]}


def step_stop_server(h):
    h.dh.com.stop_server()
    h.wait_until(
        lambda s: s.server.name in ("STOPPED", "STOPPING", "UNKNOWN"),
        timeout=15.0, label="stop_server",
    )
    s = h.dh.facade.engine.state
    return {"server": s.server.name}


def step_start_server(h):
    h.dh.com.start_server()
    h.wait_until(
        lambda s: s.server.name == "RUNNING",
        timeout=30.0, label="start_server",
    )
    return {"server": h.dh.facade.engine.state.server.name}


def step_verify_idle(h):
    s = h.dh.facade.engine.state
    assert s.activity.name == "IDLE", s.activity.name
    return {"activity": s.activity.name}


def step_verify_pc2(h):
    pc2 = h.dh.find_pc2_for(h.plane)
    assert pc2 and os.path.isfile(pc2)
    return {"pc2_size": os.path.getsize(pc2)}


def step_verify_no_pc2(h):
    pc2 = h.dh.find_pc2_for(h.plane)
    assert pc2 is None or not os.path.isfile(pc2)
    return {"pc2": pc2}


def step_verify_resumable(h):
    s = h.dh.facade.engine.state
    assert s.solver.name == "RESUMABLE", s.solver.name
    return {"solver": s.solver.name}


STEP_CATALOG = {
    "connect": step_connect,
    "transfer": step_transfer,
    "param_retransfer": step_param_retransfer,
    "data_retransfer": step_data_retransfer,
    "run": step_run,
    "fetch": step_fetch,
    "clear_animation": step_clear_animation,
    "save_and_quit": step_save_and_quit,
    "resume": step_resume,
    "abort_midbuild": step_abort_midbuild,
    "abort_during_run": step_abort_during_run,
    "disconnect": step_disconnect,
    "reconnect": step_reconnect,
    "stop_server": step_stop_server,
    "start_server": step_start_server,
    "verify_idle": step_verify_idle,
    "verify_pc2": step_verify_pc2,
    "verify_no_pc2": step_verify_no_pc2,
    "verify_resumable": step_verify_resumable,
}


def run_steps(harness, sequence):
    # Loops the chain, recording one named check per visit. Position
    # prefix in the key disambiguates repeated visits of the same
    # step name (e.g. ``run`` showing up at index 02 and again at 06).
    halted = False
    for idx, step_name in enumerate(sequence, start=1):
        check_name = f"{idx:02d}_{step_name}"
        if halted:
            harness.dh.record(check_name, False, {"skipped": "earlier step failed"})
            continue
        fn = STEP_CATALOG.get(step_name)
        if fn is None:
            harness.dh.record(check_name, False,
                              {"error": f"unknown step {step_name!r}"})
            halted = True
            continue
        try:
            details = fn(harness) or {}
            harness.dh.record(check_name, True, details)
            harness.dh.log(f"PASS {check_name}")
        except Exception as exc:
            harness.dh.record(check_name, False, {
                "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc()[-500:],
            })
            harness.dh.log(f"FAIL {check_name}: {exc}")
            halted = True
"""
