# File: scenarios/bl_force_field_objects_only.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Blender force field OBJECTS alone move simulated cloth end to end, with NO
# script in the scene (issue #114): the sampled grids are the only force
# field the solver receives. The other solve scenarios always carry a script
# beside the fields, so a payload of grids and an empty script list is
# exercised here and nowhere else.
#
# Five sheets side by side and one far away, gravity off, Air Density 1. Each
# of the first five has one field under or around it, shape Point and limited
# by a maximum distance so it reaches no neighbor:
#
#   * FORCE (Point) below its sheet, pushing away: the sheet rises.
#   * WIND blowing up (+Z): the drag lifts the sheet.
#   * VORTEX around the vertical axis through its sheet: the sheet turns
#     about its center, so the motion is tangential and the height stays.
#   * TURBULENCE at its sheet: the sheet moves, by a pattern with no net lift
#     to predict, so only that it moved is checked.
#   * A Force field KEYFRAMED from far away to under its sheet over the run:
#     the sheet rises only once the field arrives, which only time samples of
#     a moving field can produce.
#
# The far sheet has no field near it and must not move at all, which is also
# what separates "the fields act" from "something moved".
#
# Subtests:
#   A. payload_carries_grids_and_no_script: an acceleration and an
#      air-velocity grid, and an empty script list.
#   B. force_field_lifts_its_sheet
#   C. wind_field_lifts_through_the_drag
#   D. vortex_turns_its_sheet
#   E. turbulence_moves_its_sheet
#   F. moving_field_acts_once_it_arrives: the keyframed Force's sheet has not
#      moved by the middle of the run and has risen by the end.
#   G. untouched_sheet_stays_put

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 30

_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def sheet(name, x):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                        size=1.0, location=(x, 0.0, 0.0))
        o = bpy.context.object
        o.name = name
        return o

    xs = {"FFOForce": -3.2, "FFOWind": -1.6, "FFOVortex": 0.0,
          "FFOTurb": 1.6, "FFOMoving": 3.2, "FFOFar": 9.0}
    sheets = {name: sheet(name, x) for name, x in xs.items()}
    dh.save_blend(PROBE_DIR, "force_field_objects_only.blend")
    root = dh.configure_state(project_name="force_field_objects_only",
                              frame_count=FRAME_COUNT)
    state = root.state
    state.air_density = 1.0
    state.gravity_3d = (0.0, 0.0, 0.0)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    for o in sheets.values():
        cloth.add(o.name)

    def field(kind, name, loc, strength, reach=0.7):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        o.field.strength = strength
        # Shape Point, so the maximum distance is from the field's center.
        # Blender's default for Wind and Vortex is Plane, where it is the
        # distance from the field's XY plane and the field reaches the whole
        # slab around it, every sheet here included.
        o.field.shape = "POINT"
        o.field.use_max_distance = True
        o.field.distance_max = reach
        return o

    field("FORCE", "FieldForce", (xs["FFOForce"], 0.0, -0.3), 6.0)
    field("WIND", "FieldWind", (xs["FFOWind"], 0.0, -0.3), 6.0)
    field("VORTEX", "FieldVortex", (xs["FFOVortex"], 0.0, -0.2), 6.0, reach=0.8)
    turb = field("TURBULENCE", "FieldTurb", (xs["FFOTurb"], 0.0, 0.0), 8.0)
    turb.field.size = 0.3
    moving = field("FORCE", "FieldMoving", (xs["FFOMoving"], 3.0, -0.3), 6.0)
    moving.keyframe_insert("location", frame=1)
    moving.location = (xs["FFOMoving"], 0.0, -0.3)
    moving.keyframe_insert("location", frame=FRAME_COUNT)
    state.force_field_script = None
    state.force_field_padding = 0.5
    state.force_field_spacing = 0.1
    state.force_field_time_samples = 8

    # ----- A --------------------------------------------------------------
    ffp = params_mod._build_param_dict(bpy.context).get("force_field") or {}
    kinds = sorted(g["kind"] for g in ffp.get("grids", []))
    dh.record("A_payload_carries_grids_and_no_script",
              kinds == ["acceleration", "air-velocity"] and ffp.get("scripts") == [],
              {"kinds": kinds, "scripts": ffp.get("scripts")})

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes,
                      message="force_field_objects_only:build", timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def frames(obj):
        path = dh.find_pc2_for(obj)
        arr = dh.read_pc2(path) if path else None
        if arr is None or arr.shape[0] < 3 or not np.all(np.isfinite(arr)):
            return None
        return arr

    got = {name: frames(o) for name, o in sheets.items()}
    # Row 0 is the display gap-fill, so travel is measured from the first
    # solver row. PC2 rows are in object space: each sheet's origin is its
    # center, which is where its Vortex's axis runs.
    def travel(name, row=-1):
        arr = got[name]
        return None if arr is None else arr[row] - arr[1]

    detail = {"error": dh.facade.engine.state.error,
              "frames": {k: (None if v is None else int(v.shape[0])) for k, v in got.items()}}

    t = travel("FFOForce")
    detail["force_mean"] = None if t is None else [float(v) for v in t.mean(axis=0)]
    dh.record("B_force_field_lifts_its_sheet",
              t is not None and t[:, 2].mean() > 0.01, detail)

    t = travel("FFOWind")
    detail["wind_mean"] = None if t is None else [float(v) for v in t.mean(axis=0)]
    dh.record("C_wind_field_lifts_through_the_drag",
              t is not None and t[:, 2].mean() > 0.005, detail)

    t = travel("FFOVortex")
    tangential = None
    if t is not None:
        rel = got["FFOVortex"][1][:, :2]
        r_ = np.linalg.norm(rel, axis=1)
        keep = r_ > 0.1
        tangent = np.stack([-rel[:, 1], rel[:, 0]], axis=1) / np.maximum(r_, 1e-9)[:, None]
        tangential = float((t[keep, :2] * tangent[keep]).sum(axis=1).mean())
        detail["vortex_tangential"] = tangential
        detail["vortex_lift"] = float(t[:, 2].mean())
    dh.record("D_vortex_turns_its_sheet",
              tangential is not None and abs(tangential) > 0.005
              and abs(detail["vortex_lift"]) < 0.2 * abs(tangential), detail)

    t = travel("FFOTurb")
    detail["turb_max"] = None if t is None else float(np.abs(t).max())
    dh.record("E_turbulence_moves_its_sheet",
              t is not None and float(np.abs(t).max()) > 0.005, detail)

    arr = got["FFOMoving"]
    mid = travel("FFOMoving", row=(arr.shape[0] // 2) - 4) if arr is not None else None
    end = travel("FFOMoving")
    detail["moving_mid"] = None if mid is None else float(mid[:, 2].mean())
    detail["moving_end"] = None if end is None else float(end[:, 2].mean())
    dh.record("F_moving_field_acts_once_it_arrives",
              mid is not None and end is not None
              and abs(mid[:, 2].mean()) < 1e-4 and end[:, 2].mean() > 0.002, detail)

    t = travel("FFOFar")
    detail["far_max"] = None if t is None else float(np.abs(t).max())
    dh.record("G_untouched_sheet_stays_put",
              t is not None and float(np.abs(t).max()) < 1e-6, detail)

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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
