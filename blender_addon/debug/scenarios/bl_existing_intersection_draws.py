# File: scenarios/bl_existing_intersection_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The viewport overlay turns an Allow Existing Intersections record into a GPU
# batch and a label, the way it draws a build's exemptions on the start frame.
#
# A SCENARIO OF ITS OWN BECAUSE IT NEEDS A GPU CONTEXT. Windows
# runs the rig headless, and there `gpu.shader.from_builtin` raises
# "GPU functions for drawing requires the gpu module to be initialized".
# `bl_existing_intersection` checks that a real build's record reaches the
# add-on in this shape, on every platform; this one takes a record of that
# shape and needs no build.
#
# Subtests:
#   A. record_becomes_one_batch_and_label: a record with one triangle pair and
#      one rod-edge pair gives one batch and one label naming the count.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS AND SO HAS NO GPU CONTEXT to
# build a batch in. Linux gives the rig its own Xvfb and macOS has a real
# window server.
PLATFORMS = ("linux", "darwin")

_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    geo = __import__(pkg + ".ui.dynamics.overlay_geometry",
                     fromlist=["_build_violation_batches"])
    record = {
        "type": "existing_intersection",
        "count": 2,
        "pairs": [
            {"a": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
             "b": [[0.2, 0.2, -0.5], [0.2, 0.2, 0.5], [0.8, 0.2, 0.0]]},
            {"a": [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
             "b": [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]},
        ],
    }
    try:
        batches, labels = geo._build_violation_batches(
            bpy.context.scene, bpy.context.evaluated_depsgraph_get(), [record])
    except SystemError as e:
        # Named here too: a hand-written scenario list skips the PLATFORMS
        # gate, and the bare text reads as a defect rather than a platform
        # that cannot draw.
        raise SystemError(f"this Blender has no GPU context to build batches in: {e}")
    texts = [label.get("text", "") for label in labels]
    result["checks"]["A_record_becomes_one_batch_and_label"] = {
        "ok": len(batches) == 1 and texts == ["2 Existing Intersections Allowed"],
        "details": {"batches": len(batches), "labels": texts},
    }

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
