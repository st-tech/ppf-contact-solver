# File: scenarios/bl_scene_gravity_checkbox.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Take Gravity from Scene (`use_scene_gravity`, off by default, like Take FPS
# from Scene) makes the simulation use Blender's own Scene > Gravity: its X, Y
# and Z, and none while its checkbox is off. The panel shows that vector read
# only in place of the add-on's field (`scene_gravity_display`), and the field
# keeps its own value for when the toggle is turned off again. Before this, the
# add-on ignored Blender's checkbox, so a scene with it unticked still fell.
#
#   A_toggle_off_keeps_the_field: with the toggle off the add-on's field
#      applies, whatever Blender's settings say, as it always did.
#   B_toggle_on_takes_the_scene_vector: with the toggle on, the solver gets
#      Blender's Scene > Gravity vector, the panel's read-only vector shows
#      the same, and the add-on's field keeps its own value.
#   C_toggle_on_unticked_sends_zero: with Blender's checkbox off, the solver
#      gets zero and the panel's vector reads zero.
#   D_field_keyframes_not_in_use: a keyframed add-on Gravity sends no samples
#      while the scene's gravity is taken, and its samples again once it is not.
#   E_profile_carries_the_toggle: a scene profile saves and loads the toggle.
#
# Then into the solver: the gravity in the session's param.toml, the file the
# solver parses, which the build writes.
#
#   F_field_reaches_the_solver: toggle off, param.toml holds the field's -9.8.
#   H_unticked_reaches_the_solver_as_none: toggle on with Blender's checkbox
#      off, param.toml holds zero gravity.
#   G_scene_vector_reaches_the_solver: toggle on, param.toml holds Blender's
#      (2.45, 0, -4.9), and a short solve of a free cloth drifts sideways half
#      as far as it falls, as that vector says.

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
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 6
GRAVITY_PATH = "zozo_contact_solver.state.gravity_3d"

try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["_build_param_dict"])
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8,
                                    size=1.0, location=(0, 0, 1.0))
    sheet = bpy.context.object
    sheet.name = "FreeSheet"
    root = dh.configure_state(project_name="scene_gravity",
                              frame_count=FRAME_COUNT,
                              gravity=(0.0, 0.0, -9.8))
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)

    enc = __import__(pkg + ".core.encoder", fromlist=["solver_gravity"])
    profile = __import__(pkg + ".core.profile", fromlist=["_SCENE_PARAM_FIELDS"])
    state = root.state

    def sent():
        return [float(x) for x in params_mod._build_param_dict(bpy.context)["scene"]["gravity"]]

    field = [float(x) for x in enc.solver_gravity((0.0, 0.0, -9.8))]

    state.use_scene_gravity = False
    scene.use_gravity = False
    scene.gravity = (0.0, 0.0, -9.81)
    a = sent()
    dh.record("A_toggle_off_keeps_the_field", np.allclose(a, field),
              {"sent": a, "field": field})

    state.use_scene_gravity = True
    scene.use_gravity = True
    scene.gravity = (1.5, 0.0, -4.0)
    want = [float(x) for x in enc.solver_gravity((1.5, 0.0, -4.0))]
    b = sent()
    shown = [float(x) for x in state.scene_gravity_display]
    kept = [float(x) for x in state.gravity_3d]
    dh.record("B_toggle_on_takes_the_scene_vector",
              np.allclose(b, want) and np.allclose(shown, [1.5, 0.0, -4.0])
              and np.allclose(kept, [0.0, 0.0, -9.8]),
              {"sent": b, "want": want, "panel_shows": shown, "field_kept": kept})

    scene.use_gravity = False
    c = sent()
    shown_off = [float(x) for x in state.scene_gravity_display]
    dh.record("C_toggle_on_unticked_sends_zero",
              c == [0.0, 0.0, 0.0] and shown_off == [0.0, 0.0, 0.0],
              {"sent": c, "panel_shows": shown_off})

    state.gravity_3d = (0.0, 0.0, -9.8)
    scene.keyframe_insert(data_path=GRAVITY_PATH, index=2, frame=1)
    state.gravity_3d = (0.0, 0.0, -4.9)
    scene.keyframe_insert(data_path=GRAVITY_PATH, index=2, frame=FRAME_COUNT)
    scene.frame_set(1)
    keyed = params_mod._build_param_dict(bpy.context).get("dyn_param", {})
    state.use_scene_gravity = False
    keyed_off = params_mod._build_param_dict(bpy.context).get("dyn_param", {})
    samples_off = [list(map(float, e[1])) for e in keyed_off.get("gravity", [])]
    dh.record("D_field_keyframes_not_in_use",
              "gravity" not in keyed and len(samples_off) >= 2
              and any(abs(s[1]) > 1.0 for s in samples_off),
              {"taken_from_scene_keys": sorted(keyed),
               "field_samples": samples_off[:4]})

    fields = profile._SCENE_PARAM_FIELDS
    dh.record("E_profile_carries_the_toggle",
              fields.get("use_scene_gravity") == "use_scene_gravity",
              {"profile_key": fields.get("use_scene_gravity")})
    # The keyframes are only for D; what follows runs on static values.
    scene.animation_data_clear()
    state.gravity_3d = (0.0, 0.0, -9.8)
    project = state.project_name

    # F: toggle off, the field is what the solver reads.
    state.use_scene_gravity = False
    scene.use_gravity = True
    scene.gravity = (2.45, 0.0, -4.9)
    f = build_and_read_params(dh, project=project, connected=False,
                              label="gravity-field")
    dh.record("F_field_reaches_the_solver",
              np.allclose(f["param_gravity"], field, atol=1e-5),
              {"param_toml": f, "expected": field})

    # H: toggle on with Blender's checkbox off, the solver reads no gravity.
    state.use_scene_gravity = True
    scene.use_gravity = False
    h = build_and_read_params(dh, project=project, connected=True,
                              label="gravity-off")
    dh.record("H_unticked_reaches_the_solver_as_none",
              h["param_gravity"] == [0.0, 0.0, 0.0], {"param_toml": h})

    # G: toggle on, Blender's (2.45, 0, -4.9) is what the solver reads, and
    #    one short solve moves the free cloth along it: the sideways drift is
    #    half the fall, as that vector says.
    scene.use_gravity = True
    want = [float(x) for x in enc.solver_gravity((2.45, 0.0, -4.9))]
    g = build_and_read_params(dh, project=project, connected=True,
                              label="gravity-scene")
    moved = run_built_solve(dh, sheet, frames=FRAME_COUNT, label="gravity-scene")
    dx, _dy, dz = moved["move"]
    dh.record("G_scene_vector_reaches_the_solver",
              np.allclose(g["param_gravity"], want, atol=1e-5)
              and dz < -1e-3 and abs(dx / dz + 0.5) < 0.01,
              {"param_toml": g, "expected": want, "solve": moved,
               "dx_over_dz": dx / dz if dz else None, "want": -0.5})
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
