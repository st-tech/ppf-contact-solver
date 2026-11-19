# File: scenarios/bl_scene_unit_scale.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Take Unit Scale from Scene (`use_scene_unit_scale`, off by default, like
# Take FPS from Scene) uses the Blender scene's Unit Scale (Scene > Units) as
# World Scaling. The two say the same thing, one scene unit is that many
# meters, and every reader of the factor goes through
# `encoder.resolve_world_scaling`, so the whole encode switches source at once.
#
#   A_toggle_off_uses_the_field: with the toggle off, the World Scaling field
#      is the factor, whatever the scene's Unit Scale.
#   B_toggle_on_uses_the_unit_scale: with the toggle on, the solver's
#      world-scaling is the scene's Unit Scale, the field keeps its value.
#   C_every_scaled_length_follows: the lengths World Scaling multiplies switch
#      with it: the constraint gap and an absolute group contact gap.
#   D_profile_leaves_it_out: a scene profile stores neither World Scaling nor
#      where it comes from, so a profile does not resize a scene it is loaded
#      into.
#
# Then into the solver: the world_scaling in the session's param.toml, the
# file the solver parses, which the build writes.
#
#   E_field_reaches_the_solver: toggle off, param.toml holds the field's 1.0.
#   F_unit_scale_reaches_the_solver: toggle on at Unit Scale 0.1 (one unit is
#      10 cm), param.toml holds 0.1.
#   G_length_unit_is_display_only: Unit Scale 1000 with Length in kilometers,
#      param.toml holds 1000: the Length unit, which only chooses how Blender
#      displays lengths, adds nothing to the factor.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import _scene_source_solve as ss
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 4

try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["_build_param_dict"])
    profile = __import__(pkg + ".core.profile", fromlist=["_SCENE_PARAM_FIELDS"])
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4,
                                    size=1.0, location=(0, 0, 0))
    sheet = bpy.context.object
    root = dh.configure_state(project_name="scene_unit_scale",
                              frame_count=FRAME_COUNT, gravity=(0.0, 0.0, -9.8))
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    group = next(g for g in dh.groups.iterate_object_groups(scene)
                 if sheet.name in [a.name for a in g.assigned_objects])
    group.use_group_bounding_box_diagonal = False
    group.contact_gap = 0.002
    state.constraint_ghat = 0.001

    def encode():
        built = params_mod._build_param_dict(bpy.context)
        sp = built["scene"]
        gp = next(p for p, _n, uuids in built["group"] if uuids)
        gap = gp["contact-gap"]
        gap = float(next(iter(gap.values()))) if isinstance(gap, dict) else float(gap)
        return (float(sp["world-scaling"]), float(sp["constraint-ghat"]), gap)

    state.world_scaling = 2.0
    scene.unit_settings.scale_length = 0.1

    state.use_scene_unit_scale = False
    ws, ghat, gap = encode()
    dh.record("A_toggle_off_uses_the_field", np.isclose(ws, 2.0),
              {"world_scaling": ws})

    state.use_scene_unit_scale = True
    ws_on, ghat_on, gap_on = encode()
    dh.record("B_toggle_on_uses_the_unit_scale",
              np.isclose(ws_on, 0.1) and np.isclose(state.world_scaling, 2.0),
              {"world_scaling": ws_on, "field_kept": float(state.world_scaling)})

    dh.record("C_every_scaled_length_follows",
              np.isclose(ghat, 0.001 * 2.0) and np.isclose(gap, 0.002 * 2.0)
              and np.isclose(ghat_on, 0.001 * 0.1) and np.isclose(gap_on, 0.002 * 0.1),
              {"field": {"ghat": ghat, "gap": gap},
               "unit_scale": {"ghat": ghat_on, "gap": gap_on}})

    fields = profile._SCENE_PARAM_FIELDS
    dh.record("D_profile_leaves_it_out",
              "use_scene_unit_scale" not in fields.values()
              and "world_scaling" not in fields.values(),
              {"profile_properties": sorted(fields.values())})

    # Into the solver: the world_scaling in the session's param.toml, which
    # the build writes and the solver parses. What the solver then does with
    # World Scaling is the bl_world_scaling_* scenarios' subject.
    project = state.project_name
    scene.unit_settings.length_unit = "METERS"
    state.use_scene_unit_scale = False
    state.world_scaling = 1.0
    e = build_and_read_params(dh, project=project, connected=False,
                              label="scale-field")
    dh.record("E_field_reaches_the_solver",
              np.isclose(e["param_world_scaling"], 1.0), {"param_toml": e})

    state.use_scene_unit_scale = True
    scene.unit_settings.scale_length = 0.1
    f = build_and_read_params(dh, project=project, connected=True,
                              label="scale-tenth")
    dh.record("F_unit_scale_reaches_the_solver",
              np.isclose(f["param_world_scaling"], 0.1), {"param_toml": f})

    scene.unit_settings.scale_length = 1000.0
    scene.unit_settings.length_unit = "KILOMETERS"
    g = build_and_read_params(dh, project=project, connected=True,
                              label="scale-km")
    dh.record("G_length_unit_is_display_only",
              np.isclose(g["param_world_scaling"], 1000.0), {"param_toml": g})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + ss.SOLVE_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
