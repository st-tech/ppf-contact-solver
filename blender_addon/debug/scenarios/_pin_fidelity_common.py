# File: scenarios/_pin_fidelity_common.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared driver + host-side diff for the per-pin-op fidelity scenarios.
#
# Each per-op scenario (bl_pin_move_by, bl_pin_spin_centroid, ...)
# specifies a CASE dict and delegates to the helpers in this module.
#
# CASE schema:
#   {
#     "name": "spin_centroid_z",          -- used as the project name
#     "frame_count": 10,
#     "frame_rate": 100,
#     "step_size": 0.01,
#     "ops": [                            -- one or more pin ops
#         {
#           "type": "MOVE_BY"|"SPIN"|"SCALE"|"TORQUE",
#           "frame_start": 1, "frame_end": 4,
#           "transition": "LINEAR"|"SMOOTH",
#           # MOVE_BY:
#           "delta": (dx, dy, dz),
#           # SPIN:
#           "axis": (ax, ay, az),
#           "angular_velocity": deg_per_sec,
#           "flip": False,
#           "center_mode": "CENTROID"|"ABSOLUTE"|"MAX_TOWARDS"|"VERTEX",
#           "center": (cx, cy, cz),       -- ABSOLUTE
#           "center_direction": (dx, dy, dz),  -- MAX_TOWARDS
#           "center_vertex": idx,         -- VERTEX
#           # SCALE:
#           "factor": float,
#           # TORQUE:
#           "magnitude": Nm,
#           "axis_component": "PC1"|"PC2"|"PC3",
#         },
#         ...
#     ],
#   }
#
# Ops are applied in the order listed (the helper handles the addon's
# insert-at-zero quirk so the visible execution order matches the list).
#
# The diff uses ``frontend.FixedScene.time(t)`` (the source of truth
# behind ``frontend.preview()``) for the reference trajectory. The
# trajectory is mapped from solver vertex order + Y-up to Blender vertex
# order + Z-up so it can be diffed against the PC2 the addon writes.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import _runner as r


_DRIVER_TEMPLATE = """
import bpy, time, traceback, json, os
result.setdefault("phases", [])
result.setdefault("errors", [])
def log(msg):
    result["phases"].append((round(time.time(), 3), msg))

CASE = <<CASE_JSON>>
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups_mod = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh", fromlist=["encode_obj"])
    encoder_param = __import__(pkg + ".core.encoder.params", fromlist=["encode_param"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    events_mod = __import__(pkg + ".core.events", fromlist=["PollTick"])
    com = client.communicator
    solver_api = api_mod.solver

    log(f"setup_start_case={CASE['name']}")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "FidelityMesh"
    rest_world = [tuple(plane.matrix_world @ v.co) for v in plane.data.vertices]
    result["rest_world"] = rest_world
    n_verts = len(plane.data.vertices)

    vg = plane.vertex_groups.new(name="AllPin")
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "fidelity.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    log(f"saved_blend filepath={bpy.data.filepath!r}")

    root = groups_mod.get_addon_data(bpy.context.scene)
    root.state.project_name = CASE["name"]
    root.state.frame_count = CASE.get("frame_count", 10)
    root.state.frame_rate = CASE.get("frame_rate", 100)
    root.state.step_size = CASE.get("step_size", 0.01)
    root.state.disable_contact = True
    root.state.gravity_3d = (0.0, 0.0, 0.0)
    root.state.air_density = 0.0
    root.state.wind_strength = 0.0

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")

    # Apply ops in the order they appear in CASE["ops"]. The addon's
    # ``add()`` builders move the freshly-added op to position 0, so
    # iterating in REVERSE here puts the first listed op at position 0
    # (the position the solver evaluates first).
    for op in reversed(CASE["ops"]):
        op_t = op["type"]
        common = dict(
            frame_start=op.get("frame_start", 1),
            frame_end=op.get("frame_end", 4),
            transition=op.get("transition", "LINEAR"),
        )
        if op_t == "MOVE_BY":
            pin.move_by(delta=tuple(op["delta"]), **common)
        elif op_t == "SPIN":
            kwargs = dict(common)
            kwargs["axis"] = tuple(op.get("axis", (0, 0, 1)))
            kwargs["angular_velocity"] = op.get("angular_velocity", 90.0)
            kwargs["flip"] = op.get("flip", False)
            cmode = op.get("center_mode")
            if cmode == "ABSOLUTE":
                kwargs["center"] = tuple(op["center"])
            elif cmode == "MAX_TOWARDS":
                kwargs["center_direction"] = tuple(op["center_direction"])
            elif cmode == "VERTEX":
                kwargs["center_vertex"] = int(op["center_vertex"])
            elif cmode is not None:
                kwargs["center_mode"] = cmode
            pin.spin(**kwargs)
        elif op_t == "SCALE":
            kwargs = dict(common)
            kwargs["factor"] = op.get("factor", 1.0)
            cmode = op.get("center_mode")
            if cmode == "ABSOLUTE":
                kwargs["center"] = tuple(op["center"])
            elif cmode == "MAX_TOWARDS":
                kwargs["center_direction"] = tuple(op["center_direction"])
            elif cmode == "VERTEX":
                kwargs["center_vertex"] = int(op["center_vertex"])
            elif cmode is not None:
                kwargs["center_mode"] = cmode
            pin.scale(**kwargs)
        elif op_t == "TORQUE":
            pin.torque(
                magnitude=op.get("magnitude", 1.0),
                axis_component=op.get("axis_component", "PC3"),
                flip=op.get("flip", False),
                frame_start=common["frame_start"],
                frame_end=common["frame_end"],
            )
        else:
            raise ValueError(f"unknown op type: {op_t}")

    data_bytes = encoder_mesh.encode_obj(bpy.context)
    param_bytes = encoder_param.encode_param(bpy.context)
    result["data_size"] = len(data_bytes)
    result["param_size"] = len(param_bytes)

    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = LOCAL_PATH
    root.ssh_state.docker_port = SERVER_PORT
    com.set_project_name(CASE["name"])
    com.connect_local(root.ssh_state.local_path,
                      server_port=root.ssh_state.docker_port)

    deadline = time.time() + 30.0
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
            break
        time.sleep(0.2)
    if facade.engine.state.server.name != "RUNNING":
        raise RuntimeError("server never reached RUNNING")
    log("connected")

    com.build_pipeline(data=data_bytes, param=param_bytes,
                       message=f"fidelity-build:{CASE['name']}")
    deadline = time.time() + 300.0
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            break
        time.sleep(0.5)
    if facade.engine.state.solver.name == "FAILED":
        raise RuntimeError(f"build failed: {facade.engine.state.error!r}")
    log("built")

    com.run()
    saw_running = False
    deadline = time.time() + 90.0
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and s.solver.name in ("READY", "RESUMABLE"):
            break
        if s.solver.name == "FAILED":
            break
        time.sleep(0.3)
    log(f"ran_solver={facade.engine.state.solver.name}")

    # ---- 4) Force a query so state.frame reflects the COMPLETED run ----
    # The PollTick handler in the addon only emits DoQuery while solver
    # is in {RUNNING, STARTING, SAVING, BUILDING}. Once the run finishes
    # solver flips to READY, polling stops, and state.frame can be stale
    # at 0 (in which case com.fetch's DoFetchFrames sizes ``range(1,
    # state.frame+1)`` to an empty list and downloads nothing). Pinging
    # via QueryRequested -- guarded only by ``not state.busy`` -- forces
    # a fresh response that updates state.frame to the final count.
    expected_frames = root.state.frame_count - 1
    # Generous timeout: under parallel=4 the host can be heavily
    # loaded and a single query round-trip may queue behind other
    # workers' I/O, so a short deadline trips even when nothing is
    # actually wrong.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if not facade.engine.state.busy:
            facade.engine.dispatch(
                events_mod.QueryRequested(request={}, message="")
            )
        facade.tick()
        if facade.engine.state.frame >= expected_frames:
            break
        time.sleep(0.2)
    log(f"final_frame_query state.frame={facade.engine.state.frame}/{expected_frames}")
    if facade.engine.state.frame < 1:
        raise RuntimeError(
            f"state.frame=0 after run completed; fetch would be empty"
        )

    # ---- 5) Fetch ALL frames into _anim_frames ----
    # Live fetches during the run download frames as they're produced,
    # but the modal that drains the queue cannot fire while this driver
    # holds the main thread (time.sleep blocks Blender's event loop).
    # FetchRequested then dispatches DoResetAnimationBuffer which CLEARS
    # the queue, dropping every live-fetched frame; the explicit fetch
    # would otherwise skip those indices because ``_fetched`` still
    # says they're downloaded.
    #
    # Plan: drain any in-flight background fetches so they don't re-add
    # to ``_fetched`` after we clear it (race between our clear and the
    # bg io worker's _fetched.append). Settle on activity=IDLE for one
    # full second before clearing ``_fetched`` and dispatching the
    # explicit fetch.
    settle_deadline = time.time() + 15.0
    stable_since = None
    while time.time() < settle_deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.activity.name == "IDLE":
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= 1.0:
                break
        else:
            stable_since = None
        time.sleep(0.1)
    log(f"settled activity={facade.engine.state.activity.name}")
    facade.runner.clear_fetched_frames()
    com.fetch()
    deadline = time.time() + 60.0
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        # FetchRequested -> FETCHING -> FetchComplete -> APPLYING. We're
        # done with the network half the moment we hit APPLYING; the
        # actual PC2 write happens when we yield the event loop and the
        # addon's frame_pump modal drains _anim_frames.
        if s.activity.name == "APPLYING":
            break
        if s.activity.name == "IDLE" and s.solver.name in ("READY", "RESUMABLE"):
            # Fetch finished synchronously with 0 frames downloaded
            # (none new in to_fetch); nothing to do.
            break
        time.sleep(0.3)
    runner = facade.runner
    with runner._anim_lock:
        queued = len(runner._anim_frames)
        total = runner._anim_total
    log(f"fetched activity={facade.engine.state.activity.name} "
        f"queued={queued} total={total}")

    # NOTE: PC2 writes happen in PPF_OT_FramePump.modal AFTER this exec
    # returns. The bootstrap's drain-then-quit timer (in blender_harness
    # _BOOTSTRAP_TEMPLATE) waits for the modal to fully drain
    # _anim_frames before quitting Blender. Calling apply_animation
    # directly from this thread is unreliable because the modal-context
    # ID-write check denies non-deterministically under contention.

    s = facade.engine.state
    result["project_name"] = CASE["name"]
    result["remote_root"] = s.remote_root
    result["frames_total"] = root.state.frame_count - 1
    result["fps"] = root.state.frame_rate
    result["dt"] = root.state.step_size
    result["pin_indices"] = list(range(n_verts))
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_HOST_DIFF_SCRIPT = r"""
# Runs in the project .venv. Reads the addon's fetched PC2 (the
# downstream artifact of the Rust binary's vert_*.bin) and compares
# pinned vertex positions per frame against frontend.FixedScene.time(t)
# -- the same source of truth that ``frontend.preview()`` uses.
#
# To make the comparison apples-to-apples, we apply the addon's
# solver->blender remapping (vertex permutation + Y-up to Z-up axis
# swap) to fixed.time's output before diffing against PC2.

import json, os, pickle, struct, sys, traceback
import numpy as np

req = json.loads(sys.stdin.read())
shadow_root = req["shadow_root"]
addon_data_dir = req["addon_data_dir"]
project_name = req["project_name"]
fps = float(req["fps"])
frames_total = int(req["frames_total"])
pin_indices = req["pin_indices"]
tolerance = float(req.get("tolerance", 1e-4))

result = {"per_frame": [], "max_error": 0.0, "errors": [], "project_name": project_name}


def read_pc2(path):
    # Layout: 12B sig, 4B u32 version, 4B u32 n_verts, 4B float
    # start_frame, 4B float sample_rate, 4B u32 n_samples, body.
    with open(path, "rb") as f:
        sig = f.read(12)
        if not sig.startswith(b"POINTCACHE2"):
            raise ValueError(f"not a PC2 file: {sig!r}")
        version, n_verts = struct.unpack("<II", f.read(8))
        start_frame, sample_rate = struct.unpack("<ff", f.read(8))
        (n_samples,) = struct.unpack("<I", f.read(4))
        body = f.read(n_samples * n_verts * 3 * 4)
    arr = np.frombuffer(body, dtype="<f4").reshape(n_samples, n_verts, 3)
    return arr, int(start_frame), float(sample_rate), int(n_samples)


def solver_to_blender_axes(arr):
    # Inverse of encoder's (x, y, z) -> (x, z, -y) axis swap. Applied
    # vertex-wise; preserves vertex order.
    return np.column_stack([arr[:, 0], -arr[:, 2], arr[:, 1]])


try:
    os.environ["PPF_CTS_DATA_ROOT"] = shadow_root
    sys.path.insert(0, req["repo_root"])

    # Read Rust's per-frame timing log BEFORE we touch ``frontend``.
    # ``frontend.BlenderApp(name).populate().make()`` reconstructs
    # ``FixedSession``, whose __init__ calls ``self.delete()`` and
    # wipes the entire session dir -- including
    # ``output/data/frame_to_time.out`` which we depend on. Capture it
    # up-front, then proceed.
    f2t_path = os.path.join(shadow_root, "git-debug", project_name,
                            "session", "output", "data",
                            "frame_to_time.out")
    frame_times: dict[int, float] = {}
    if os.path.exists(f2t_path):
        with open(f2t_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        frame_times[int(parts[0])] = float(parts[1])
                    except ValueError:
                        continue

    from server.emulator import install as install_debug_emulator
    install_debug_emulator()
    import frontend  # type: ignore

    app = frontend.BlenderApp(project_name).populate().make()
    fixed = app.scene

    # Locate the PC2 the addon wrote on fetch. Path is
    # <addon_data_dir>/<basename>/<uuid>.pc2.
    pc2_path = None
    for sub in os.listdir(addon_data_dir):
        sub_path = os.path.join(addon_data_dir, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.endswith(".pc2"):
                    pc2_path = os.path.join(sub_path, f)
                    break
        if pc2_path:
            break
    if pc2_path is None:
        raise FileNotFoundError(f"no .pc2 file under {addon_data_dir}")

    arr, start_frame, sample_rate, n_samples = read_pc2(pc2_path)
    result["pc2_shape"] = list(arr.shape)
    n_pc2_verts = arr.shape[1]

    # Load the solver->Blender vertex mapping. Frontend's BlenderApp
    # writes ``map.pickle`` per session. Schema: dict[uuid_str, ndarray].
    map_path = os.path.join(shadow_root, "git-debug", project_name,
                            "session", "map.pickle")
    with open(map_path, "rb") as f:
        anim_map = pickle.load(f)
    if not anim_map:
        raise RuntimeError("empty anim_map")
    vmap = list(anim_map.values())[0]
    if vmap is None or len(vmap) < n_pc2_verts:
        raise RuntimeError(
            f"vertex map too short: have {len(vmap)} need {n_pc2_verts}"
        )
    vmap = np.asarray(vmap[:n_pc2_verts], dtype=np.int64)

    # Per-frame diff. PC2 sample N == vert_N.bin (the addon's
    # ``_apply_single_frame`` maps Rust frame n to Blender frame n+1
    # to PC2 frame_idx n). Sample 0 is the rest pose; we skip it.
    for n in range(1, min(frames_total + 1, n_samples)):
        actual = arr[n]
        # Use the exact (frame, time) Rust recorded for this vert_N.bin.
        # Fall back to N/fps if frame_to_time.out is unavailable.
        t = frame_times.get(n, n / fps)
        solver_pos = np.asarray(fixed.time(t), dtype=np.float32)
        blender_order = solver_pos[vmap]
        expected = solver_to_blender_axes(blender_order).astype(np.float32)

        diff = np.abs(actual[pin_indices] - expected[pin_indices])
        per_frame_max = float(diff.max()) if len(diff) else 0.0
        result["per_frame"].append({
            "frame": n, "t": round(t, 6), "max_error": per_frame_max,
            "actual": actual[pin_indices].tolist(),
            "expected": expected[pin_indices].tolist(),
        })
        result["max_error"] = max(result["max_error"], per_frame_max)

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())

print("__FIDELITY_RESULT__" + json.dumps(result))
"""


def _read_payload(resp: dict) -> dict | None:
    output = (resp.get("output") or "") + (resp.get("error") or "")
    for line in reversed(output.splitlines()):
        if line.startswith("__FIDELITY_RESULT__"):
            try:
                return json.loads(line[len("__FIDELITY_RESULT__"):])
            except json.JSONDecodeError:
                continue
    return None


def build_driver(case: dict, ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<CASE_JSON>>", json.dumps(case))
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext, case: dict) -> dict:
    import blender_harness as bh

    bspec = ctx.artifacts.get("blender_spec")
    proc = ctx.artifacts.get("blender_proc")
    if bspec is None or proc is None:
        return r.failed(["no Blender process attached"])

    try:
        blender_result = bh.wait_for_result(bspec, proc, timeout=max(ctx.timeout, 240.0))
    except TimeoutError as e:
        return r.failed([str(e)])

    if blender_result.get("errors"):
        return r.failed(
            blender_result["errors"][:2],
            notes=[f"Blender phases: {blender_result.get('phases', [])[-5:]}"],
        )

    shadow_root = os.path.join(ctx.workspace, "project")
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    # Windows venvs use Scripts\python.exe; POSIX uses bin/python.
    if os.name == "nt":
        venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.isfile(venv_python):
        venv_python = sys.executable

    diff_input = {
        "project_name": blender_result["project_name"],
        "fps": blender_result["fps"],
        "dt": blender_result["dt"],
        "frames_total": blender_result["frames_total"],
        "pin_indices": blender_result["pin_indices"],
        "shadow_root": shadow_root,
        "repo_root": repo_root,
        "addon_data_dir": os.path.join(ctx.workspace, "data"),
        "tolerance": case.get("tolerance", 1e-4),
    }

    proc2 = subprocess.run(
        [venv_python, "-c", _HOST_DIFF_SCRIPT],
        input=json.dumps(diff_input).encode(),
        capture_output=True,
        timeout=120.0,
    )
    output = proc2.stdout.decode() + proc2.stderr.decode()
    payload = _read_payload({"output": output})
    if payload is None:
        return r.failed(
            ["no fidelity payload from .venv comparison"],
            notes=[f"diff output (tail): {output[-1500:]!r}"],
        )

    violations: list[str] = list(payload.get("errors") or [])
    tolerance = case.get("tolerance", 1e-4)
    if payload.get("max_error", float("inf")) > tolerance:
        violations.append(
            f"pin trajectory diverged from frontend.time(): "
            f"max_error={payload['max_error']:.3e} (tolerance {tolerance:.0e})"
        )

    notes = [
        f"case: {case['name']}",
        f"frames compared: {len(payload.get('per_frame', []))}",
        f"max error: {payload.get('max_error', 0):.3e}",
    ]
    if violations:
        return r.failed(violations, notes=notes)
    return r.passed(notes=notes)
