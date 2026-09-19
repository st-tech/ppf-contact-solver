# File: scenarios/bl_native_attach_build_check.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A local native backend refuses to attach to a server running a different
# build from the one its Compute Device selection names.
#
# WHY. Disconnect leaves a native server running on purpose, and Connect then
# attaches to whatever answers on the port. The device selection chooses
# which server to LAUNCH, so on that path it reached nothing: with CPU
# selected, connecting to the GPU server an earlier connection had left ran
# every solve on the GPU build while the panel read CPU. The server now
# reports the target directory its runs use (`solver_target_dir`), and Connect
# refuses one that is not the selected build.
#
# The rig owns this worker's server, launched from `target/release`, which is
# the GPU build on the hosts this runs on. So no second server is needed:
# selecting CPU against it is the mismatch, and selecting GPU is the control
# showing that the check does not refuse a server that matches.
#
# Subtests:
#   A. both_builds_present: the root holds a GPU build and a CPU build.
#      Without a CPU build a CPU selection is refused as "not found" before the
#      running server is consulted at all, so the rest would prove nothing; the
#      scenario fails here by name, with the command that builds one.
#   B. server_reports_its_build: the rig's server reports the GPU target
#      directory, and a GPU backend asked of its own solver binary.
#   C. gpu_selection_attaches: Connect with GPU reaches ONLINE with the server
#      RUNNING.
#   D. cpu_selection_is_refused: Connect with CPU does not go ONLINE, and the
#      error on the panel names the running server's directory, the selected
#      build's directory, and the way out.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# THE THREE LOCAL NATIVE BACKENDS, one per platform, since Linux gained the
# LINUX_NATIVE connection.
PLATFORMS = ("darwin", "win32", "linux")
# The rig's own server must be the GPU build for a CPU selection to be the
# mismatch, which a real-backend host provides. Nothing here runs physics.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import sys
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
REPO = <<REPO_REPR>>
SERVER_PORT = <<SERVER_PORT>>

try:
    dh = DriverHelpers(pkg, result)
    conn = __import__(pkg + ".core.connection",
                      fromlist=["check_running_server"])
    # (server_type, path property, backend type) for this machine, in one
    # place rather than a chain of branches.
    server_type, path_field, backend_type = {
        "win32": ("WIN_NATIVE", "win_native_path", "win_native"),
        "darwin": ("MAC_NATIVE", "mac_native_path", "mac_native"),
    }.get(sys.platform, ("LINUX_NATIVE", "linux_native_path", "linux_native"))
    resolver, _, _, resolve_root = conn.native_resolvers(backend_type)
    root = resolve_root(REPO) or REPO
    connect = getattr(dh.com, f"connect_{backend_type}")

    # ----- A: both builds exist, or nothing below proves anything -----
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
    gpu_dir = conn._expected_target_dir(root, gpu_bin)
    cpu_dir = conn._expected_target_dir(root, cpu_bin)

    # ----- B: the rig's server says which build its runs use ----------
    response = conn._query_ppf_cts_server(SERVER_PORT) or {}
    reported = str(response.get("solver_target_dir") or "")
    backend = str(response.get("solver_backend") or "")
    dh.record(
        "B_server_reports_its_build",
        conn._same_directory(reported, gpu_dir)
        and backend in ("cuda", "metal", "rocm"),
        {
            "reported_dir": reported,
            "expected_dir": gpu_dir,
            "reported_backend": backend,
        },
    )

    state = dh.groups.get_addon_data(bpy.context.scene).ssh_state
    state.server_type = server_type
    setattr(state, path_field, REPO)
    state.docker_port = SERVER_PORT
    dh.com.set_project_name("native_attach_build_check")

    def attempt(device, timeout=30.0):
        state.native_device = device
        connect(REPO, SERVER_PORT, device)
        deadline = time.time() + timeout
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            s = dh.facade.engine.state
            if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
                break
            if s.phase.name == "OFFLINE" and s.error:
                break
            time.sleep(0.2)
        s = dh.facade.engine.state
        return {"phase": s.phase.name, "server": s.server.name,
                "error": s.error}

    # ----- C: the matching selection attaches ---------------------------
    got = attempt("GPU")
    dh.record(
        "C_gpu_selection_attaches",
        got["phase"] == "ONLINE" and got["server"] == "RUNNING",
        got,
    )
    dh.com.disconnect()
    for _ in range(10):
        dh.facade.tick()
        time.sleep(0.1)

    # ----- D: the other selection is refused, and says why -------------
    got = attempt("CPU")
    message = got["error"] or ""
    dh.record(
        "D_cpu_selection_is_refused",
        got["phase"] != "ONLINE"
        and "Compute Device is set to CPU" in message
        and reported in message
        and cpu_dir in message
        and "set Compute Device to GPU" in message,
        dict(got, expected_named=[reported, cpu_dir]),
    )
    state.native_device = "GPU"

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<REPO_REPR>>", repr(REPO_ROOT_POSIX))
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
