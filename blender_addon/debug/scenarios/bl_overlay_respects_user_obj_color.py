# File: scenarios/bl_overlay_respects_user_obj_color.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: the addon's apply_object_overlays() must not touch
# obj.color on meshes that are not in any active addon group. A prior
# version walked every MESH / CURVE in bpy.data.objects and
# unconditionally reset obj.color = (1, 1, 1, 1), then re-applied
# group tints on top. The reset clobbered colors the user had set on
# unrelated meshes (reference assets, third-party tool color hints)
# every undo/redo and every group op.
#
# Subtests:
#   A. user_color_on_non_group_object_is_preserved:
#         A fresh cube with obj.color = red and no group membership
#         keeps its color after apply_object_overlays().
#   B. group_member_still_receives_tint:
#         A plane in a SHELL group with show_overlay_color on gets
#         obj.color = group.color after apply_object_overlays() (the
#         existing positive behavior is not regressed by the narrowing).
#   C. excluded_group_member_is_reset:
#         A plane in a group whose assignment has included=False is
#         reset to white (no addon-applied tint remains).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    overlay_mod = __import__(
        pkg + ".ui.dynamics.overlay",
        fromlist=["apply_object_overlays"],
    )
    apply_overlays = overlay_mod.apply_object_overlays
    groups_mod = __import__(
        pkg + ".models.groups",
        fromlist=["get_addon_data"],
    )

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Bystander mesh: NOT added to any addon group, colored red.
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(3.0, 3.0, 0.0))
    bystander = bpy.context.active_object
    bystander.name = "Bystander"
    bystander.color = (1.0, 0.0, 0.0, 1.0)

    # Member mesh: added to a SHELL group with show_overlay_color on.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    member = bpy.context.active_object
    member.name = "MemberPlane"

    api = dh.api.solver
    cloth = api.create_group("Cloth", "SHELL")
    cloth.add(member.name)
    # Ensure the overlay-color toggle is on (default), and set a known
    # group color we can check against.
    root = groups_mod.get_addon_data(bpy.context.scene)
    group_pg = root.object_group_0
    group_pg.show_overlay_color = True
    group_pg.color = (0.2, 0.7, 0.3, 1.0)
    expected_tint = (0.2, 0.7, 0.3, 1.0)

    # ---- A: bystander's color is preserved -----------------------
    apply_overlays()
    bystander_after = tuple(bystander.color)
    dh.record(
        "A_user_color_on_non_group_object_is_preserved",
        # Allow tiny float rounding (Blender clamps to float32 on store).
        all(abs(bystander_after[i] - (1.0, 0.0, 0.0, 1.0)[i]) < 1e-5 for i in range(4)),
        {
            "bystander_color_before": [1.0, 0.0, 0.0, 1.0],
            "bystander_color_after": list(bystander_after),
        },
    )

    # ---- B: member receives the group tint -----------------------
    member_after = tuple(member.color)
    dh.record(
        "B_group_member_still_receives_tint",
        all(abs(member_after[i] - expected_tint[i]) < 1e-5 for i in range(4)),
        {
            "member_color_after": list(member_after),
            "expected_tint": list(expected_tint),
        },
    )

    # ---- C: included=False resets the member to white ------------
    group_pg.assigned_objects[0].included = False
    apply_overlays()
    member_excluded = tuple(member.color)
    dh.record(
        "C_excluded_group_member_is_reset",
        all(abs(member_excluded[i] - 1.0) < 1e-5 for i in range(4)),
        {"member_color_after_excluded": list(member_excluded)},
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
