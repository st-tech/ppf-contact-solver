# File: scenarios/bl_pdrd_driven_translate.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PDRD Driven Motion: a Translate step, encoding round-trip.
#
# In the specialized rigid-body panel a Driven pin carries a list of motion
# steps. "Translate" is the MOVE_BY operation (the only translate kind the
# curated UI offers). The params encoder emits it as an op dict
#   {"type": "move_by", "delta": <solver-swapped>, "t_start":, "t_end":, ...}
# inside the pin's per-vertex cfg.
#
# This adds a Driven pin via the Add-operation operator (op_type="MOVE_BY",
# the exact call the panel's "Translate" button makes), encodes, decodes,
# and asserts the move_by op round-trips with the right delta and timing.
#
# Encoding-only (the emulated backend has no rigid physics); the prescribed
# motion is exercised on a real CUDA host.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

FRAME_START = 1
FRAME_END = 10
DELTA = (0.0, 0.0, 0.5)

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Rigid"
    # A handle face drives the whole rigid body.
    vg = cube.vertex_groups.new(name="handle")
    vg.add([4, 5, 6, 7], 1.0, "REPLACE")

    fps = 100
    root = dh.configure_state(
        project_name="pdrd_driven_translate", frame_count=20, frame_rate=fps
    )
    grp_api = dh.api.solver.create_group("Rigid", "PDRD")
    grp_api.add("Rigid")
    grp_api.create_pin("Rigid", "handle")

    group = root.object_group_0
    pin = group.pin_vertex_groups[0]

    # Add a Translate step exactly as the panel's "Translate" button does.
    group.pin_vertex_groups_index = 0
    bpy.ops.object.add_pin_operation(group_index=0, op_type="MOVE_BY")
    op = pin.operations[0]
    if op.op_type != "MOVE_BY":
        raise RuntimeError(f"expected MOVE_BY op, got {op.op_type}")
    op.delta = DELTA
    op.frame_start = FRAME_START
    op.frame_end = FRAME_END
    op.transition = "LINEAR"

    enc_pin = __import__(pkg + ".core.encoder.pin", fromlist=["_to_solver"])
    expected_delta = list(enc_pin._to_solver(DELTA))

    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    pin_config = decoded.get("pin_config", {})
    if not pin_config:
        raise RuntimeError("pin_config empty; driven pin was not encoded")
    cfg = list(list(pin_config.values())[0].values())[0]
    ops = cfg.get("operations") or []
    dh.log(f"ops={ops}")
    if not ops:
        raise RuntimeError("no operations in cfg")
    op_d = ops[0]

    # ----- A: the step encodes as a move_by ---------------------------
    dh.record(
        "A_step_is_move_by",
        op_d.get("type") == "move_by",
        {"type": op_d.get("type")},
    )

    # ----- B: delta round-trips through the solver axis swap ----------
    got = [float(x) for x in (op_d.get("delta") or [])]
    delta_ok = (
        len(got) == 3
        and all(abs(got[i] - expected_delta[i]) < 1e-6 for i in range(3))
    )
    dh.record(
        "B_delta_axis_swapped",
        delta_ok,
        {"delta": got, "expected": expected_delta},
    )

    # ----- C: frame range encodes as t_start / t_end ------------------
    expect_t0 = (FRAME_START - 1) / fps
    expect_t1 = (FRAME_END - 1) / fps
    t_ok = (
        abs(float(op_d.get("t_start", -1)) - expect_t0) < 1e-6
        and abs(float(op_d.get("t_end", -1)) - expect_t1) < 1e-6
    )
    dh.record(
        "C_timing_encoded",
        t_ok,
        {"t_start": op_d.get("t_start"), "t_end": op_d.get("t_end"),
         "expected": [expect_t0, expect_t1]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
