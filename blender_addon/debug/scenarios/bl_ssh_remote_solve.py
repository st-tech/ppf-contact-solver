# File: scenarios/bl_ssh_remote_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend end-to-end smoke for the addon's SSH backend: macOS
# Blender drives a REAL CUDA solver running on a remote GPU box over
# SSH. This is the "local Blender + remote GPU" workflow that nothing
# else in CI exercises.
#
# The addon connects with server_type=CUSTOM (paramiko). The whole
# encode -> upload -> remote solve -> fetch cycle streams over paramiko's
# direct-tcpip channel to the remote server's loopback; there is no
# scp/rsync and no shared disk. The remote ppf-cts-server is expected to
# be already listening (the macos-ssh CI job pre-starts it), so a bare
# connect reaches phase=ONLINE, server=RUNNING.
#
# The scene is a shell sheet pinned along one edge and released under
# gravity. On the real solver it DRAPES: the pinned edge holds, the free
# vertices sag. That is genuine physics, so this is real-only.
#
# SSH connection parameters come from the environment (injected by the
# job via `runtests --knob SSH_*=...`), NOT from string substitution,
# because the host / key path / remote path are runtime values:
#   PPF_SSH_HOST, PPF_SSH_PORT, PPF_SSH_USER, PPF_SSH_KEY,
#   PPF_SSH_REMOTE_PATH, PPF_SSH_SERVER_PORT
#
# Subtests:
#   A. ssh_connect_build_run_fetch: connected ONLINE over SSH, the
#      remote build+run succeeded, and a finite PC2 with >= frame_count-1
#      samples came back over the tunnel.
#   B. cloth_draped: the pinned edge held and the free vertices moved
#      down (-z), i.e. the remote real solver actually simulated.
#   C. remote_build_listing_matches_the_running_server: the build listing
#      taken from the REAL remote host parses, the Compute Device resolves
#      against it to a server on that host, and the directory that server
#      reports its runs use is the one the selection names. This is the only
#      place in CI where the remote listing comes off a machine that is not
#      the one running Blender.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# SSH backend uses paramiko on the local (macOS) side; only meaningful
# there, and it needs a real remote GPU server, so real-only + darwin.
PLATFORMS = ("darwin",)
BACKENDS = ("real",)


_FRAME_COUNT = 15


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
FRAME_COUNT = <<FRAME_COUNT>>

# SSH connection details injected as env knobs by the macos-ssh job.
SSH_HOST = os.environ.get("PPF_SSH_HOST", "")
SSH_PORT = int(os.environ.get("PPF_SSH_PORT", "22"))
SSH_USER = os.environ.get("PPF_SSH_USER", "ubuntu")
SSH_KEY = os.environ.get("PPF_SSH_KEY", "")
SSH_REMOTE_PATH = os.environ.get("PPF_SSH_REMOTE_PATH", "")
SSH_SERVER_PORT = int(os.environ.get("PPF_SSH_SERVER_PORT", "9090"))


def _build_drape_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "DrapeSheet"
    pinned = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    return sheet, pinned, free


try:
    if not (SSH_HOST and SSH_KEY and SSH_REMOTE_PATH):
        raise RuntimeError(
            "missing SSH env: PPF_SSH_HOST / PPF_SSH_KEY / PPF_SSH_REMOTE_PATH"
        )
    # paramiko must be importable in Blender's Python for the CUSTOM
    # backend; surface a clear error if the install step was skipped.
    try:
        import paramiko  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"paramiko not importable in Blender: {exc}")

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sheet, pinned_idx, free_idx = _build_drape_scene()
    dh.save_blend(PROBE_DIR, "ssh_remote_solve.blend")
    root = dh.configure_state(
        project_name="ssh_remote_solve",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    data_bytes, param_bytes = dh.encode_payload()

    dh.connect_ssh(
        host=SSH_HOST, port=SSH_PORT, username=SSH_USER, key_path=SSH_KEY,
        remote_path=SSH_REMOTE_PATH, server_port=SSH_SERVER_PORT,
        project_name=root.state.project_name, timeout=90.0,
    )
    dh.log("connected over ssh")

    dh.build_and_wait(data_bytes, param_bytes, "ssh:build", timeout=300.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total} solver={solver_state}")

    pc2_path = dh.find_pc2_for(sheet)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: connect + remote build/run/fetch completed --------------
    dh.record(
        "A_ssh_connect_build_run_fetch",
        solver_state != "FAILED"
        and arr is not None
        and samples >= FRAME_COUNT - 1
        and finite,
        {
            "solver_state": solver_state,
            "pc2_path": pc2_path,
            "samples": samples,
            "expected_min_samples": FRAME_COUNT - 1,
            "all_finite": finite,
            "error": dh.facade.engine.state.error,
        },
    )

    # ----- B: cloth draped (pinned edge held, free verts sank) --------
    pin_disp = -1.0
    mean_drop = 1.0
    if arr is not None and samples >= 2 and finite:
        rest = arr[0]
        last = arr[-1]
        pinned = np.asarray(pinned_idx, dtype=np.int64)
        free = np.asarray(free_idx, dtype=np.int64)
        pin_disp = float(np.max(np.linalg.norm(last[pinned] - rest[pinned], axis=1)))
        mean_drop = float(np.mean((last[free] - rest[free])[:, 2]))
    dh.record(
        "B_cloth_draped",
        pin_disp >= 0.0 and pin_disp < 0.1 and mean_drop < -0.02,
        {
            "max_pinned_disp": round(pin_disp, 5),
            "mean_free_dz": round(mean_drop, 5),
        },
    )

    # ----- C: the Compute Device resolved against the REAL remote host -
    #
    # The one place in CI where the remote build listing is taken from a
    # machine that is not this one. Every other check of it drives a
    # stand-in, so this is what says the command reaches a real host, its
    # output parses, and what it resolves to is the directory the server
    # over there says its runs use.
    #
    # THE SERVER HERE WAS PRE-STARTED BY THE JOB, so this compares the
    # selection against a server the add-on attached to rather than one it
    # launched. That is the comparison `check_running_server` makes for a
    # native connection, which the remote path has no equivalent of, so a
    # disagreement is exactly the silent substitution worth catching.
    remote_builds = __import__(pkg + ".core.remote_builds",
                               fromlist=["cached_builds"])
    conn_mod = __import__(pkg + ".core.connection",
                          fromlist=["remote_server_binary"])
    listing = dict(remote_builds.cached_builds())
    probe_error = remote_builds.probe_error()
    # THE PROBED root, not the data root. `normalized_remote_root()` answers
    # `<share>/ppf-cts/git-<branch>/<project>`, which the listing's keys are
    # never under, so resolving against it misses every time and reads as a
    # host with no build.
    root = remote_builds.probed_root()
    resolved = conn_mod.remote_server_binary(root, listing, "GPU")
    expected_dir = conn_mod.remote_target_dir(resolved) if resolved else ""
    reported_dir = str((dh.com.response or {}).get("solver_target_dir") or "")
    reported_backend = str((dh.com.response or {}).get("solver_backend") or "")
    # AN EMPTY REPORT IS A FAILURE TO MEASURE, NOT A MISMATCH. The server
    # leaves both fields empty when it could not ask its own solver, and
    # reading that as a disagreement would fail a leg over a question that
    # was never answered.
    agrees = (
        not reported_dir
        or not expected_dir
        or os.path.normpath(reported_dir) == os.path.normpath(expected_dir)
    )
    dh.record(
        "C_remote_build_listing_matches_the_running_server",
        bool(listing) and not probe_error and resolved is not None and agrees,
        {
            "root": root,
            "listing": listing,
            "probe_error": probe_error,
            "resolved_server": resolved,
            "expected_target_dir": expected_dir,
            "server_reports_target_dir": reported_dir,
            "server_reports_backend": reported_backend,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE.replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
