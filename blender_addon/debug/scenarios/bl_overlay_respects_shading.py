# File: scenarios/bl_overlay_respects_shading.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: the addon's overlay system must not stomp the user's
# viewport Solid-shading color_type. A community report on Blender 5.1
# noted that picking Material / Random / Vertex etc. would snap back
# to OBJECT on the next redraw or undo/redo because the overlay code
# unconditionally wrote ``space.shading.color_type = "OBJECT"`` when a
# group had ``show_overlay_color`` on.
#
# Subtests:
#   A. apply_object_overlays_preserves_random:
#         With an active group whose show_overlay_color is on, the
#         viewport's color_type set to RANDOM stays RANDOM across a
#         call to apply_object_overlays().
#   B. apply_object_overlays_preserves_material:
#         Same as A but with color_type = "MATERIAL".

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _first_view3d_space():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        return sp
    return None


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    overlay_mod = __import__(
        pkg + ".ui.dynamics.overlay",
        fromlist=["apply_object_overlays"],
    )
    apply_overlays = overlay_mod.apply_object_overlays

    # Fresh scene with a SHELL group + assigned plane; show_overlay_color
    # defaults to on so the bug condition is reproduced.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "OverlayShadingPlane"

    api = dh.api.solver
    cloth = api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    space = _first_view3d_space()
    if space is None:
        raise RuntimeError("no VIEW_3D space available")

    # Force Solid shading; this is where the bug fires.
    space.shading.type = "SOLID"

    # ---- A: RANDOM stays RANDOM ---------------------------------
    space.shading.color_type = "RANDOM"
    apply_overlays()
    after_random = space.shading.color_type
    dh.record(
        "A_apply_object_overlays_preserves_random",
        after_random == "RANDOM",
        {"color_type_after_apply": after_random},
    )

    # ---- B: MATERIAL stays MATERIAL -----------------------------
    space.shading.color_type = "MATERIAL"
    apply_overlays()
    after_material = space.shading.color_type
    dh.record(
        "B_apply_object_overlays_preserves_material",
        after_material == "MATERIAL",
        {"color_type_after_apply": after_material},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 60.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
