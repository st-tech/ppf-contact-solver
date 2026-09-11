# File: scenarios/bl_mcp_curve_authoring.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Curve authoring over MCP, and the checkpoint frame list.
#
# A curve is built through a pending BUILDER rather than written straight into
# the scene: create_curve opens one, add_curve_spline and set_curve_material
# fill it, and finalize_curve turns it into the object. Nothing exists in the
# scene until the finalize, which is why every step names the builder and
# refuses a name no builder holds.
#
# Assertions:
#   A. ``builder_opens_and_takes_a_spline`` -- create_curve reports the
#      builder's settings, and a spline of three points is accepted.
#   B. ``a_spline_needs_two_points`` -- a one-point spline is refused with the
#      minimum named.
#   C. ``unknown_builder_is_refused`` -- add_curve_spline, set_curve_material
#      and finalize_curve each refuse a name no builder holds, and each says
#      to call create_curve first.
#   D. ``material_binds_to_a_spline`` -- set_curve_material with
#      create_if_missing binds the material to spline 0.
#   E. ``finalize_makes_a_curve_object`` -- finalize_curve produces a CURVE
#      object carrying the spline count, and the object is in the scene.
#   F. ``curve_joins_a_rod_group`` -- the finalized curve assigns to a ROD
#      group, which is the object type a curve is authored for.
#   G. ``checkpoint_frames_are_normalized`` -- set_save_checkpoint_frames
#      sorts, de-duplicates and lifts a frame below 1, and the listing reports
#      the normalized set; clear empties it.
#   H. ``remote_checkpoint_list_is_separate`` -- list_checkpoint_frames asks a
#      different question (what a run has saved) and answers empty here,
#      rather than reporting the frames the scene merely requested.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [600]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    call("clear_solver")

    # ----- A --------------------------------------------------------
    builder = call("create_curve", {"name": "Strand", "bevel_depth": 0.01})
    spline = call("add_curve_spline",
                  {"name": "Strand", "points": [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]})
    mcp_check(result, "A_builder_opens_and_takes_a_spline",
              builder.get("status") == "success"
              and builder.get("curve_name") == "Strand"
              and abs(builder.get("bevel_depth", 0) - 0.01) < 1e-9
              and builder.get("dimensions") == "3D"
              and spline.get("status") == "success"
              and spline.get("spline_index") == 0 and spline.get("point_count") == 3
              # Nothing is in the scene until the finalize.
              and "Strand" not in bpy.data.objects,
              {"builder": builder, "spline": spline,
               "in_scene_yet": "Strand" in bpy.data.objects})

    # ----- B --------------------------------------------------------
    short = call("add_curve_spline", {"name": "Strand", "points": [[0, 0, 0]]})
    mcp_check(result, "B_a_spline_needs_two_points",
              short.get("status") == "error"
              and "at least 2 points" in short.get("message", ""),
              {"reply": short})

    # ----- C --------------------------------------------------------
    unknown = {
        "add_curve_spline": call("add_curve_spline",
                                 {"name": "NoSuch", "points": [[0, 0, 0], [1, 0, 0]]}),
        "set_curve_material": call("set_curve_material",
                                   {"name": "NoSuch", "spline_index": 0,
                                    "material_name": "M", "create_if_missing": True}),
        "finalize_curve": call("finalize_curve", {"name": "NoSuch"}),
    }
    mcp_check(result, "C_unknown_builder_is_refused",
              all(v.get("status") == "error"
                  and "NoSuch" in v.get("message", "")
                  and "create_curve" in v.get("message", "")
                  for v in unknown.values()),
              {k: v.get("message") for k, v in unknown.items()})

    # ----- D --------------------------------------------------------
    material = call("set_curve_material",
                    {"name": "Strand", "spline_index": 0,
                     "material_name": "RodMat", "create_if_missing": True})
    mcp_check(result, "D_material_binds_to_a_spline",
              material.get("status") == "success"
              and material.get("spline_index") == 0
              and material.get("material_name") == "RodMat",
              {"reply": material})

    # ----- E --------------------------------------------------------
    finalized = call("finalize_curve", {"name": "Strand"})
    obj = bpy.data.objects.get("Strand")
    mcp_check(result, "E_finalize_makes_a_curve_object",
              finalized.get("status") == "success"
              and finalized.get("object_type") == "CURVE"
              and finalized.get("spline_count") == 1
              and obj is not None and obj.type == "CURVE"
              and len(obj.data.splines) == 1,
              {"reply": finalized,
               "in_scene": obj is not None,
               "splines": len(obj.data.splines) if obj else None})

    # ----- F --------------------------------------------------------
    rod = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": rod, "type": "ROD"})
    joined = call("add_objects_to_group",
                  {"group_uuid": rod, "object_names": ["Strand"]})
    members = [o.get("name") for o in
               ((call("get_group", {"group_uuid": rod}).get("group") or {})
                .get("assigned_objects") or [])]
    mcp_check(result, "F_curve_joins_a_rod_group",
              joined.get("status") == "success"
              and [o.get("name") for o in (joined.get("added_objects") or [])] == ["Strand"]
              and members == ["Strand"],
              {"joined": joined, "members": members})

    # ----- G --------------------------------------------------------
    # 5 twice, 2 out of order, and 0 below the floor.
    setf = call("set_save_checkpoint_frames", {"frames": [5, 2, 5, 0]})
    listed = call("list_save_checkpoint_frames")
    cleared = call("clear_save_checkpoint_frames")
    after = call("list_save_checkpoint_frames")
    mcp_check(result, "G_checkpoint_frames_are_normalized",
              setf.get("status") == "success" and setf.get("frames") == [1, 2, 5]
              and listed.get("frames") == [1, 2, 5] and listed.get("count") == 3
              and cleared.get("status") == "success"
              and after.get("frames") == [] and after.get("count") == 0,
              {"set": setf, "listed": listed, "after_clear": after})

    # ----- H --------------------------------------------------------
    call("set_save_checkpoint_frames", {"frames": [3, 7]})
    remote = call("list_checkpoint_frames")
    requested = call("list_save_checkpoint_frames")
    mcp_check(result, "H_remote_checkpoint_list_is_separate",
              remote.get("status") == "success"
              and remote.get("checkpoint_frames") == [] and remote.get("count") == 0
              # The scene asked for two, and no run has saved any.
              and requested.get("frames") == [3, 7],
              {"remote": remote, "requested": requested})

    call("clear_save_checkpoint_frames")
    call("clear_solver")
    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
