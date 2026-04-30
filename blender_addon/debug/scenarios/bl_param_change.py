# File: scenarios/bl_param_change.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the parameter-transfer pipeline. Verifies that editing
# a scene parameter and re-transferring actually produces a different
# simulation result, i.e. the new ``param.pickle`` reaches the solver
# and is honored.
#
# Sequence:
#   1. Build a scene with a pinned plane + MOVE_BY(delta=DELTA_A).
#   2. Encode + upload + build + run + fetch + drain.
#      Snapshot vertex 0's final-frame position; expect rest + DELTA_A.
#   3. Mutate ``operations[0].delta`` to DELTA_B.
#   4. Re-encode + re-upload + re-build + re-run + re-fetch + drain.
#      Snapshot again; expect rest + DELTA_B.
#   5. Assert the two positions differ by approximately DELTA_B - DELTA_A.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def run_pipeline(dh, plane, message, expected_frames):
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes, message=message)
    dh.run_and_wait(timeout=60.0)
    dh.force_frame_query(expected_frames=expected_frames, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    return dh.fetch_and_drain()


try:
    DELTA_A = 0.1
    DELTA_B = 0.5

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ParamMesh")
    dh.save_blend(PROBE_DIR, "param.blend")
    root = dh.configure_state(project_name="param_change", frame_count=6)
    expected = root.state.frame_count - 1

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(DELTA_A, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    run_pipeline(dh, plane, message="param-A", expected_frames=expected)
    dh.log("pass_A.done")

    pc2_path = dh.find_pc2_for(plane)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 after pass A: {pc2_path}")
    arr_a = dh.read_pc2(pc2_path)
    rest = arr_a[0]
    last_a = arr_a[-1]
    delta_a_seen = float(last_a[0][0] - rest[0][0])
    dh.record(
        "pass_A_applied_delta_A",
        abs(delta_a_seen - DELTA_A) < 1e-3,
        {"delta_a_seen": delta_a_seen, "expected": DELTA_A,
         "rest_v0": rest[0].tolist(), "last_v0": last_a[0].tolist()},
    )
    dh.log(f"pass_A delta_seen={delta_a_seen:.4f}")

    # Mutate the pin op's delta and run the pipeline again.
    pin_item = root.object_group_0.pin_vertex_groups[0]
    move_op = pin_item.operations[0]
    move_op.delta = (DELTA_B, 0.0, 0.0)
    dh.log(f"mutated delta -> {tuple(move_op.delta)}")

    run_pipeline(dh, plane, message="param-B", expected_frames=expected)
    dh.log("pass_B.done")

    pc2_path_b = dh.find_pc2_for(plane)
    if not pc2_path_b or not os.path.isfile(pc2_path_b):
        raise RuntimeError(f"no PC2 after pass B: {pc2_path_b}")
    arr_b = dh.read_pc2(pc2_path_b)
    rest_b = arr_b[0]
    last_b = arr_b[-1]
    delta_b_seen = float(last_b[0][0] - rest_b[0][0])
    dh.record(
        "pass_B_applied_delta_B",
        abs(delta_b_seen - DELTA_B) < 1e-3,
        {"delta_b_seen": delta_b_seen, "expected": DELTA_B,
         "rest_v0": rest_b[0].tolist(), "last_v0": last_b[0].tolist()},
    )
    dh.record(
        "pass_A_and_B_differ",
        abs(delta_b_seen - delta_a_seen - (DELTA_B - DELTA_A)) < 1e-3,
        {"delta_a_seen": delta_a_seen, "delta_b_seen": delta_b_seen},
    )
    dh.log(f"pass_B delta_seen={delta_b_seen:.4f}")

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
