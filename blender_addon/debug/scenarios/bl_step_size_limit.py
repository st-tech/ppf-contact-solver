# File: scenarios/bl_step_size_limit.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The step-size limit at the add-on's two doors.
#
# The solver refuses a step size past `MAX_STEP_SIZE` (the core's parameter
# registry), and the add-on holds a copy of the limit in `core/limits.py`. A
# step size reaches the encoder two ways: the Step Size field, whose maximum
# is the limit, and an F-curve on that field, which is evaluated directly and
# so is not bound by the field's maximum. Both must stay inside the limit.
#
# Subtests:
#   A. field_max_is_the_limit:  the field's hard and soft maxima are the limit.
#   B. a_larger_value_clamps:   setting the field past the limit lands on it.
#   C. at_the_limit_encodes:    a step size at the limit encodes cleanly.
#   D. keyframed_overshoot_refused: an F-curve keyed past the limit is refused
#         at encode, with the sentence naming the frame.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0, x_subdivisions=3,
                                    y_subdivisions=3, location=(0.0, 0.0, 0.0))
    cloth = bpy.context.active_object
    cloth.name = "StepCloth"
    dh.save_blend(PROBE_DIR, "step_size_limit.blend")
    root = dh.configure_state(project_name="step_size_limit", frame_count=6,
                              frame_rate=60, step_size=0.01)
    state = root.state
    group = dh.api.solver.create_group("Cloth", "SHELL")
    group.add(cloth.name)

    limits = __import__(pkg + ".core.limits", fromlist=["MAX_STEP_SIZE"])
    limit = float(limits.MAX_STEP_SIZE)

    # ---- A: the field's maxima are the solver's limit -----------------
    prop = state.bl_rna.properties["step_size"]
    dh.record(
        "A_field_max_is_the_limit",
        abs(prop.hard_max - limit) < 1e-9 and abs(prop.soft_max - limit) < 1e-9,
        {"hard_max": prop.hard_max, "soft_max": prop.soft_max, "limit": limit},
    )

    # ---- B: a value past the limit lands on it -------------------------
    state.step_size = 2.5 * limit
    dh.record(
        "B_a_larger_value_clamps",
        abs(state.step_size - limit) < 1e-6,
        {"after_setting": state.step_size, "limit": limit},
    )

    # ---- C: the limit itself encodes -----------------------------------
    state.step_size = limit
    encoded = True
    detail = ""
    try:
        dh.encode_payload()
    except Exception as exc:  # noqa: BLE001 - the verdict records what was raised
        encoded = False
        detail = f"{type(exc).__name__}: {exc}"
    dh.record("C_at_the_limit_encodes", encoded, {"error": detail})

    # ---- D: a keyframed value past the limit is refused ----------------
    # The keys are inserted at the limit, then one is moved past it on the
    # curve itself, which is what a Bezier overshoot or an edited key does.
    state.step_size = limit
    state.keyframe_insert("step_size", frame=1)
    state.keyframe_insert("step_size", frame=6)
    utils = __import__(pkg + ".core.utils", fromlist=["get_id_fcurves"])
    curves = [fc for fc in utils.get_id_fcurves(bpy.context.scene)
              if fc.data_path.endswith("state.step_size")]
    if curves:
        curves[0].keyframe_points[1].co[1] = 3.0 * limit
        curves[0].update()
    refused = False
    message = ""
    try:
        dh.encode_payload()
    except ValueError as exc:
        refused = True
        message = str(exc)
    dh.record(
        "D_keyframed_overshoot_refused",
        bool(curves) and refused and "keyframed Step Size" in message,
        {"curves": len(curves), "refused": refused, "message": message},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(dict(result.get("checks", {})))
