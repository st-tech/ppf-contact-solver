# File: scenarios/bl_force_field_visualize_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's Visualize overlay builds its GPU batches: one per kind of
# arrow drawn (field objects' pushes and the script's).
#
# A SCENARIO OF ITS OWN BECAUSE IT NEEDS A GPU CONTEXT. Windows
# runs the rig headless, and there `gpu.shader.from_builtin` raises
# "GPU functions for drawing requires the gpu module to be initialized".
# `bl_force_field_visualize` holds every check that needs no GPU, so Windows
# still runs those.
#
# Subtests:
#   A. batches_build: with a Force field and a script, the overlay builds one
#      batch for the pushes and one for the script.

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
    dh = DriverHelpers(pkg, result)
    viz = __import__(pkg + ".ui.dynamics.overlay_geometry.force_field",
                     fromlist=["build_force_field_batches"])
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    dh.save_blend(PROBE_DIR, "force_field_visualize_draws.blend")
    root = dh.configure_state(project_name="force_field_visualize_draws", frame_count=11)
    state = root.state
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)
    bpy.ops.object.effector_add(type="FORCE", location=(0.0, 0.0, -0.4))
    bpy.context.object.field.strength = 5.0
    text = bpy.data.texts.new("ff_draws.py")
    text.from_string("def eval(x, y, z, t):\n    return (y, -x, 0.5)\n")
    state.force_field_script = text
    state.force_field_visualize = True
    state.force_field_preview_resolution = (4, 3, 2)

    try:
        batches = viz.build_force_field_batches(scene, state, 10.0)
    except SystemError as e:
        # Named here too: a hand-written scenario list skips the PLATFORMS
        # gate, and the bare text reads as a defect rather than a platform
        # that cannot draw.
        raise SystemError(f"this Blender has no GPU context to build batches in: {e}")
    dh.record("A_batches_build", len(batches) == 2, {"batches": len(batches)})

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
    return r.report_named_checks(result.get("checks", {}))
