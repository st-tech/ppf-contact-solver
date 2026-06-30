# File: scenarios/bl_sand_emulated_roundtrip.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end emulated round-trip for the SAND granular type on a no-GPU
# host (macOS). A closed mesh is Converted into a particle mesh (N loose
# verts seeded inside it), assigned to a SAND group, transferred, run,
# and fetched. With PPF_EMULATED_SAND=1 the emulator's phony mover
# (crates/ppf-cts-solver/src/cpp_emul/sand.hpp) drifts every free grain
# under gravity, so the cloud FALLS. Asserts:
#   A. pc2_has_all_frames    - every output frame produced.
#   B. grains_have_N_rows    - each frame carries all N grains (the
#                              backend.rs write-gate fix for faceless clouds).
#   C. grains_fell           - mean -z displacement clearly negative.
#   D. simulation_stable     - all positions finite and bounded.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX

NEEDS_BLENDER = True
# The emulated backend auto-drifts a faceless point cloud, so no PPF_EMULATED_SAND
# opt-in is needed; STEP_MS=0 just paces the run fast for the unit test.
KNOBS = {"PPF_EMULATED_STEP_MS": "0"}

_DRIVER_BODY = r"""
import os
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 20
GRAIN_RADIUS = 0.04   # fills the r=1 icosphere with a few thousand grains
EXTRA_SPACING = 0.0   # densest non-overlapping packing

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # A closed solid (icosphere) to fill with sand.
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(0, 0, 0))
    src = bpy.context.object
    src.name = "SandBall"

    # Convert to a particle mesh (N loose verts seeded inside) via the
    # addon's headless-callable seeder.
    sand_ops = __import__(pkg + ".ui.dynamics.sand_ops", fromlist=["build_and_commit_particle_mesh"])
    n = sand_ops.build_and_commit_particle_mesh(src, GRAIN_RADIUS, EXTRA_SPACING, rng_seed=0)
    dh.log(f"converted particles n={n} verts={len(src.data.vertices)} polys={len(src.data.polygons)}")

    dh.save_blend(PROBE_DIR, "sand_roundtrip.blend")
    root = dh.configure_state(
        project_name="sand_emulated_roundtrip",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    sand = dh.api.solver.create_group("Sand", "SAND")
    sand.add(src.name)

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _pre_d, _pre_p = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, "sand:build", timeout=120.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")
    saw_running = dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.log(f"ran saw_running={saw_running} frame={dh.facade.engine.state.frame}")
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(src)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path)  # (n_samples, n_verts, 3)
    dh.log(f"pc2 shape={arr.shape}")
    try:
        np.save("/tmp/sand_rt_pc2.npy", arr)  # absolute path, survives worker cleanup
    except Exception:
        pass

    rest, last = arr[0], arr[-1]

    dh.record("A_pc2_has_all_frames", arr.shape[0] == FRAME_COUNT,
              {"pc2_samples": int(arr.shape[0]), "expected": FRAME_COUNT})

    # B: the faceless cloud's N grains all reach the bin file (write-gate fix).
    dh.record("B_grains_have_N_rows", arr.shape[1] == n,
              {"pc2_verts": int(arr.shape[1]), "expected_grains": int(n)})

    # C: grains fell under phony gravity (mean -z displacement negative).
    dz = (last - rest)[:, 2]
    mean_drop = float(np.mean(dz))
    dh.record("C_grains_fell", mean_drop < -0.02,
              {"mean_dz": round(mean_drop, 5), "min_dz": round(float(np.min(dz)), 5)})

    # D: stable.
    finite = bool(np.all(np.isfinite(arr)))
    bounded = float(np.max(np.abs(arr))) < 1000.0
    dh.record("D_simulation_stable", finite and bounded,
              {"finite": finite, "max_abs": round(float(np.max(np.abs(arr))), 4)})

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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
