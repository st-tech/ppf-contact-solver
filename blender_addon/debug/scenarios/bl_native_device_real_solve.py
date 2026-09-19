# File: scenarios/bl_native_device_real_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A real solve through the local native backend once per Compute Device, with
# the ADD-ON launching the server as it does for an artist, and proof of which
# solver binary each run executed.
#
# WHY. The device selection picks which build directory the server comes
# from, and three things must agree for a run to use that build: the server
# binary, the cdylib the build worker loads, and the solver the session
# launcher names. They come from three different places, and each binary
# answers `--backend` honestly about itself, so a split between them reports
# nothing: an earlier revision spawned the CPU server while the solve ran the
# GPU solver out of `target/release`. This scenario asks the launcher the
# server actually ran: `command.sh` (or `command.bat`) passes its arguments
# through to the solver, and the solver answers `--backend` before anything
# else, so the answer comes from the exact binary the run used.
#
# The rig owns a server on this worker's port and sets the NO_SPAWN variable
# so the add-on attaches to it rather than launching one. This scenario clears
# that for itself, launches on ports of its own, and restores it afterwards.
#
# Subtests (the per-device ones are prefixed cpu_ / gpu_):
#   A. both_builds_present: the root holds a GPU build and a CPU build; the
#      scenario fails here by name, with the command that builds one, rather
#      than proving half of what it claims.
#   B. <device>_server_launched: Connect + Start Server reached RUNNING on a
#      server the add-on spawned, and that server reports the selected build's
#      target directory as the one its runs use.
#   C. <device>_solve_completed: build, run and fetch finished, and the free
#      region of a sheet pinned along one edge sagged under gravity, so the
#      run computed physics rather than returning a frozen mesh.
#   D. <device>_launcher_names_the_selected_solver: the session launcher's
#      solver sits in the selected build's release directory, and running the
#      launcher with --backend answers cpu for CPU and cuda or metal for GPU.
#   E. the_two_runs_used_different_solvers: the CPU run and the GPU run did not
#      execute the same binary.
#   F. <device>_server_stopped: Stop Server ended the server the add-on
#      spawned, measured by its port going quiet. A native server survives
#      Disconnect on purpose, so a stop that does not stop leaves exactly the
#      server the next Connect would attach to.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# THE THREE LOCAL NATIVE BACKENDS, one per platform. Linux joined them with the
# LINUX_NATIVE connection, and it is the platform this proof matters most on:
# the GPU legs of Blender CI are Linux, so this is where a real GPU run and a
# real CPU run can be compared on one machine.
PLATFORMS = ("darwin", "win32", "linux")
# Real-only: both runs are genuine gravity dynamics, one per backend.
BACKENDS = ("real",)
# It launches two servers and two solves of its own beside the rig's
# per-worker server, so it runs in the serial tail rather than beside other
# workers' solvers.
NOT_PARALLELIZABLE = True


_FRAME_COUNT = 24


_DRIVER_BODY = r"""
import os
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
REPO = <<REPO_REPR>>
FRAME_COUNT = <<FRAME_COUNT>>

windows = sys.platform.startswith("win")
# (server_type, path property, backend type, no-spawn variable) for this
# machine. One table rather than a chain of branches, so a platform cannot be
# added to three of the four places and missed in the fourth.
NATIVE = {
    "win32": ("WIN_NATIVE", "win_native_path", "win_native",
              "PPF_WIN_NATIVE_NO_SPAWN"),
    "darwin": ("MAC_NATIVE", "mac_native_path", "mac_native",
               "PPF_MAC_NATIVE_NO_SPAWN"),
}.get(sys.platform, ("LINUX_NATIVE", "linux_native_path", "linux_native",
                     "PPF_LINUX_NATIVE_NO_SPAWN"))
SERVER_TYPE, PATH_FIELD, BACKEND_TYPE, NO_SPAWN = NATIVE
saved_env = {key: os.environ.get(key) for key in (NO_SPAWN, "PPF_CTS_DATA_ROOT")}
dh = None


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def pump(done, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if done(dh.facade.engine.state):
            return True
        time.sleep(0.2)
    return False


def read_launcher(project_root):
    # The solver the session launcher names, and what the launcher answers
    # to --backend. Both come from the file the server executed.
    launcher = os.path.join(
        project_root, "session", "command.bat" if windows else "command.sh"
    )
    if not os.path.isfile(launcher):
        return launcher, "", ""
    solver_path = ""
    with open(launcher, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if windows and line.startswith("set SOLVER_PATH="):
                solver_path = line[len("set SOLVER_PATH="):]
            elif not windows and line.startswith("SOLVER_PATH="):
                solver_path = line[len("SOLVER_PATH="):].strip('"')
    command = (["cmd", "/c", launcher, "--backend"] if windows
               else ["bash", launcher, "--backend"])
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    lines = done.stdout.strip().splitlines()
    return launcher, solver_path, (lines[-1].strip() if lines else "")


def stop_and_disconnect(port):
    # Stop Server moves the state to STOPPING at once and runs the stop as an
    # effect afterwards, so waiting only for "not RUNNING" returns before the
    # server has been touched, and a disconnect then tears the backend down
    # under the queued stop, which leaves the server running. Wait for the
    # stop to COMPLETE, and for the port to stop answering, which is the
    # only thing that proves the process is gone.
    dh.com.stop_server()
    pump(lambda s: s.server.name not in ("RUNNING", "STOPPING", "LAUNCHING"), 60.0)
    deadline = time.time() + 30.0
    while time.time() < deadline and conn._probe_ppf_cts_server(port, timeout=0.5):
        time.sleep(0.5)
    stopped = not conn._probe_ppf_cts_server(port, timeout=0.5)
    dh.com.disconnect()
    for _ in range(10):
        dh.facade.tick()
        time.sleep(0.1)
    return stopped


try:
    dh = DriverHelpers(pkg, result)
    conn = __import__(pkg + ".core.connection",
                      fromlist=["check_running_server"])
    resolver, _, _, resolve_root = conn.native_resolvers(BACKEND_TYPE)
    root = resolve_root(REPO) or REPO
    connect = getattr(dh.com, f"connect_{BACKEND_TYPE}")
    server_type = SERVER_TYPE

    # ----- A: both builds exist ----------------------------------------
    gpu_bin = resolver(root, conn.DEVICE_GPU)
    cpu_bin = resolver(root, conn.DEVICE_CPU)
    dh.record(
        "A_both_builds_present",
        gpu_bin is not None and cpu_bin is not None,
        {
            "root": root,
            "gpu_server": gpu_bin,
            "cpu_server": cpu_bin,
            "build_cpu_with": "CARGO_TARGET_DIR=target/cpu cargo build "
                              "--release --features cpu",
        },
    )
    if gpu_bin is None or cpu_bin is None:
        raise RuntimeError(
            "this root does not hold both a GPU and a CPU build; see check A"
        )

    # The add-on launches its own server here, as it does for an artist. Its
    # sessions go under this scenario's probe directory rather than the
    # user's data root: the spawn copies this process's environment.
    os.environ.pop(NO_SPAWN, None)
    os.environ["PPF_CTS_DATA_ROOT"] = os.path.join(PROBE_DIR, "native_data")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=10, y_subdivisions=10, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "DeviceDrapeSheet"
    # SHELL meshes are not remeshed, so PC2 vertex order matches the input.
    pinned_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    group = sheet.vertex_groups.new(name="TopEdge")
    group.add(pinned_idx, 1.0, "REPLACE")

    dh.save_blend(PROBE_DIR, "native_device_real_solve.blend")
    addon = dh.configure_state(
        project_name="native_device_real_solve",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")
    data_bytes, param_bytes = dh.encode_payload()

    ssh_state = addon.ssh_state
    ssh_state.server_type = server_type
    setattr(ssh_state, PATH_FIELD, REPO)

    solvers = {}
    for device, server_bin, answers in (
        ("CPU", cpu_bin, ("cpu",)),
        ("GPU", gpu_bin, ("cuda", "metal", "rocm")),
    ):
        tag = device.lower()
        expected_dir = conn._expected_target_dir(root, server_bin)
        expected_release = os.path.join(expected_dir, "release")
        port = free_port()
        ssh_state.docker_port = port
        ssh_state.native_device = device
        dh.com.set_project_name(addon.state.project_name)
        dh.log(f"{tag}: connect + start on port {port}")

        # ----- B: the add-on's own server, reporting the selected build
        connect(REPO, port, device)
        pump(lambda s: s.phase.name == "ONLINE"
             or (s.phase.name == "OFFLINE" and bool(s.error)), 30.0)
        dh.com.start_server()
        pump(lambda s: s.server.name == "RUNNING" or bool(s.error), 180.0)
        response = conn._query_ppf_cts_server(port) or {}
        reported = str(response.get("solver_target_dir") or "")
        s = dh.facade.engine.state
        dh.record(
            f"B_{tag}_server_launched",
            s.server.name == "RUNNING"
            and conn._same_directory(reported, expected_dir),
            {
                "phase": s.phase.name,
                "server": s.server.name,
                "error": s.error,
                "reported_dir": reported,
                "expected_dir": expected_dir,
                "reported_backend": response.get("solver_backend"),
            },
        )
        if s.server.name != "RUNNING":
            raise RuntimeError(
                f"{device}: the add-on's server never reached RUNNING: {s.error}"
            )

        # ----- C: a real solve ------------------------------------------
        dh.build_and_wait(data_bytes, param_bytes,
                          message=f"native_device_real_solve:{tag}",
                          timeout=600.0)
        dh.run_and_wait(timeout=900.0)
        solver_state = dh.facade.engine.state.solver.name
        # The project directory the server reported to the add-on's OWN polls.
        # A status query from here would name a project of its own and be
        # answered with that project's directory instead.
        project_root = dh.facade.engine.state.remote_root
        dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
        dh.settle_idle(timeout=15.0)
        dh.fetch_and_drain()
        pc2_path = dh.find_pc2_for(sheet)
        arr = dh.read_pc2(pc2_path) if pc2_path and os.path.isfile(pc2_path) else None
        samples = int(arr.shape[0]) if arr is not None else 0
        finite = bool(arr is not None and np.all(np.isfinite(arr)))
        pin_disp = -1.0
        mean_free_dz = 1.0
        if arr is not None and samples >= 2 and finite:
            rest, last = arr[0], arr[-1]
            pinned = np.asarray(pinned_idx, dtype=np.int64)
            free = np.asarray(free_idx, dtype=np.int64)
            pin_disp = float(np.max(np.linalg.norm(last[pinned] - rest[pinned], axis=1)))
            mean_free_dz = float(np.mean((last[free] - rest[free])[:, 2]))
        dh.record(
            f"C_{tag}_solve_completed",
            solver_state != "FAILED"
            and samples >= FRAME_COUNT - 1
            and finite
            and 0.0 <= pin_disp < 0.1
            and mean_free_dz < -0.05,
            {
                "solver_state": solver_state,
                "samples": samples,
                "all_finite": finite,
                "max_pinned_disp": round(pin_disp, 5),
                "mean_free_dz": round(mean_free_dz, 5),
                "error": dh.facade.engine.state.error,
            },
        )

        # ----- D: the launcher the server ran names the selected solver --
        launcher, solver_path, answer = read_launcher(project_root)
        solvers[device] = (solver_path, answer)
        dh.record(
            f"D_{tag}_launcher_names_the_selected_solver",
            bool(solver_path)
            and conn._same_directory(os.path.dirname(solver_path), expected_release)
            and answer in answers,
            {
                "launcher": launcher,
                "solver_path": solver_path,
                "expected_release_dir": expected_release,
                "backend_answer": answer,
                "accepted_answers": list(answers),
            },
        )

        # ----- F: Stop Server really ends the server the add-on spawned ---
        # A server left running would be what the next device's Connect
        # attaches to, and it is the step the refusal message tells the user
        # to take, so a stop that does not stop is a failure here too.
        dh.record(f"F_{tag}_server_stopped", stop_and_disconnect(port),
                  {"port": port})
        # The next device's fetch must write its own frames, so the check on
        # them cannot pass by reading this run's file.
        if pc2_path and os.path.isfile(pc2_path):
            os.remove(pc2_path)

    # ----- E: the two runs executed different binaries -----------------
    cpu_path, cpu_answer = solvers.get("CPU", ("", ""))
    gpu_path, gpu_answer = solvers.get("GPU", ("", ""))
    different = False
    try:
        different = not os.path.samefile(cpu_path, gpu_path)
    except OSError:
        different = False
    dh.record(
        "E_the_two_runs_used_different_solvers",
        different and cpu_answer != gpu_answer,
        {"cpu": [cpu_path, cpu_answer], "gpu": [gpu_path, gpu_answer]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    # Leave nothing running and the rig's contract as it was found.
    try:
        if dh is not None and dh.facade.engine.state.server.name == "RUNNING":
            stop_and_disconnect(globals().get("port"))
    except Exception as exc:
        result["errors"].append(f"cleanup: {type(exc).__name__}: {exc}")
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<REPO_REPR>>", repr(REPO_ROOT_POSIX))
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 1500.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
