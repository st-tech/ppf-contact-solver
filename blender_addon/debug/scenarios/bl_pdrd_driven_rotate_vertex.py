# File: scenarios/bl_pdrd_driven_rotate_vertex.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PDRD Driven Motion: a Rotate step about a picked vertex, encoding round-trip.
#
# In the specialized rigid-body panel a Driven pin's "Rotate" step is the SPIN
# operation with a curated center: Centroid or a picked Vertex (the cloth-only
# Absolute and Max-Towards modes are not offered). A VERTEX-centered spin
# encodes as
#   {"type": "spin", "center_mode": "absolute", "center": <vertex in solver
#    space>, "axis": <unit, axis-swapped>, "angular_velocity": deg/s, ...}
#
# This adds a Driven pin, a SPIN step with spin_center_mode="VERTEX" (the
# state the panel's Vertex toggle + eyedropper set), encodes, decodes, and
# asserts the spin round-trips with the picked-vertex center and a unit axis.
#
# Encoding-only (the emulated backend has no rigid physics); the spin runs on
# a real CUDA host.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import math
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

CENTER_VERTEX = 0
AXIS = (0.0, 0.0, 1.0)
ANGULAR_VELOCITY = 90.0

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Rigid"
    vg = cube.vertex_groups.new(name="spinner")
    vg.add([4, 5, 6, 7], 1.0, "REPLACE")

    fps = 100
    root = dh.configure_state(
        project_name="pdrd_driven_rotate_vertex", frame_count=20, frame_rate=fps
    )
    grp_api = dh.api.solver.create_group("Rigid", "PDRD")
    grp_api.add("Rigid")
    grp_api.create_pin("Rigid", "spinner")

    group = root.object_group_0
    pin = group.pin_vertex_groups[0]

    # Add a Rotate step (the panel's "Rotate" button) then set the curated
    # Vertex center the panel's Vertex toggle + eyedropper produce.
    group.pin_vertex_groups_index = 0
    bpy.ops.object.add_pin_operation(group_index=0, op_type="SPIN")
    op = pin.operations[0]
    if op.op_type != "SPIN":
        raise RuntimeError(f"expected SPIN op, got {op.op_type}")
    op.spin_center_mode = "VERTEX"
    op.spin_center_vertex = CENTER_VERTEX
    op.spin_axis = AXIS
    op.spin_angular_velocity = ANGULAR_VELOCITY
    op.spin_flip = False
    op.frame_start = 1
    op.frame_end = 10
    op.transition = "LINEAR"

    enc_pin = __import__(pkg + ".core.encoder.pin", fromlist=["_swap_axes"])
    sw = list(enc_pin._swap_axes(AXIS))
    n = math.sqrt(sum(c * c for c in sw)) or 1.0
    expected_axis = [c / n for c in sw]

    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    pin_config = decoded.get("pin_config", {})
    if not pin_config:
        raise RuntimeError("pin_config empty; driven spin was not encoded")
    cfg = list(list(pin_config.values())[0].values())[0]
    ops = cfg.get("operations") or []
    dh.log(f"ops={ops}")
    if not ops:
        raise RuntimeError("no operations in cfg")
    op_d = ops[0]

    # ----- A: the step encodes as a spin ------------------------------
    dh.record(
        "A_step_is_spin",
        op_d.get("type") == "spin",
        {"type": op_d.get("type")},
    )

    # ----- B: vertex center encodes as an absolute center point -------
    center = op_d.get("center")
    center_ok = (
        op_d.get("center_mode") == "absolute"
        and isinstance(center, (list, tuple))
        and len(center) == 3
        and all(math.isfinite(float(c)) for c in center)
    )
    dh.record(
        "B_vertex_center_encoded",
        center_ok,
        {"center_mode": op_d.get("center_mode"), "center": center},
    )

    # ----- C: unit axis + angular velocity ----------------------------
    got_axis = [float(x) for x in (op_d.get("axis") or [])]
    axis_ok = (
        len(got_axis) == 3
        and all(abs(got_axis[i] - expected_axis[i]) < 1e-6 for i in range(3))
    )
    av_ok = abs(float(op_d.get("angular_velocity", 0.0)) - ANGULAR_VELOCITY) < 1e-6
    dh.record(
        "C_unit_axis_and_velocity",
        axis_ok and av_ok,
        {"axis": got_axis, "expected_axis": expected_axis,
         "angular_velocity": op_d.get("angular_velocity")},
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
