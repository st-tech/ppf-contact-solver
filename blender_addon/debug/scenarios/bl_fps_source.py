# File: scenarios/bl_fps_source.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The simulation's frame rate has two possible sources, and exactly one of
# them may reach the solver:
#
#   use_scene_fps = False -> the `frame_rate` field
#   use_scene_fps = True  -> the Blender scene's frame rate
#
# `resolve_fps()` (core/encoder/__init__.py) is the single place that
# decides, and every encoder converts frames to seconds through it. The
# trap this test exists for: `frame_rate` KEEPS ITS LAST VALUE while the
# scene override is on. A scene whose field reads 100 while the solver runs
# at 24 is not a bug in itself, but anything that REPORTS 100 in that state
# is lying, and the field disagreeing with the run is how the confusion
# starts. So the scene is set up in exactly that disagreeing state on
# purpose, and each hop is asserted against the effective rate.
#
# Subtests:
#   A. field_mode_resolves:    override off -> resolve_fps is `frame_rate`.
#   B. scene_mode_resolves:    override on  -> resolve_fps is the scene's
#         fps, even though `frame_rate` still reads its old value.
#   C. mcp_reports_effective:  get_scene_parameters() surfaces
#         `effective_fps` = the rate actually in use, not the stale field.
#   D. legacy_key_migrates:    a .blend written before the rename carries
#         `use_frame_rate_in_output` as a raw ID-property;
#         migrate_renamed_state_props() must land it on `use_scene_fps` and
#         drop the stale key, or opening an old file silently changes its
#         time base.
#   E. param_toml_carries_fps (host-side): the solver's literal on-disk
#         input has `fps = <scene fps>`, not the field's value.

from __future__ import annotations

import glob
import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Backend-agnostic: every assertion is a value-propagation invariant, and
# param.toml is written by the frontend before any backend runs.
BACKENDS = ("emulated", "real")

_FRAME_COUNT = 6
# Deliberately different from each other, and from every default, so a hop
# that reads the wrong source is unmistakable.
_FIELD_FPS = 100
_SCENE_FPS = 24


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
FIELD_FPS = <<FIELD_FPS>>
SCENE_FPS = <<SCENE_FPS>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0, x_subdivisions=3,
                                    y_subdivisions=3, location=(0.0, 0.0, 0.0))
    cloth = bpy.context.active_object
    cloth.name = "FpsCloth"

    dh.save_blend(PROBE_DIR, "fps_source.blend")
    root = dh.configure_state(project_name="fps_source",
                              frame_count=FRAME_COUNT,
                              frame_rate=FIELD_FPS,
                              step_size=1.0 / 60.0)
    state = root.state
    bpy.context.scene.render.fps = SCENE_FPS
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    group = dh.api.solver.create_group("Cloth", "SHELL")
    group.add(cloth.name)

    encoder = __import__(pkg + ".core.encoder", fromlist=["resolve_fps"])

    # ---- A: override off -> the field is the source ------------------
    state.use_scene_fps = False
    fps_field_mode = float(encoder.resolve_fps(state))
    dh.record(
        "A_field_mode_resolves",
        fps_field_mode == float(FIELD_FPS),
        {"resolve_fps": fps_field_mode, "expected": FIELD_FPS,
         "scene_fps": SCENE_FPS},
    )

    # ---- B: override on -> the scene is the source -------------------
    # `frame_rate` is deliberately left at FIELD_FPS. This is the state the
    # confusion came from: the field says one thing, the solver runs another.
    state.use_scene_fps = True
    fps_scene_mode = float(encoder.resolve_fps(state))
    dh.record(
        "B_scene_mode_resolves",
        fps_scene_mode == float(SCENE_FPS)
        and int(state.frame_rate) == FIELD_FPS,
        {"resolve_fps": fps_scene_mode, "expected": SCENE_FPS,
         "frame_rate_field_still": int(state.frame_rate)},
    )

    # ---- C: MCP surfaces the EFFECTIVE rate, not the stale field -----
    remote = __import__(pkg + ".mcp.handlers.remote",
                        fromlist=["get_scene_parameters"])
    # @mcp_handler wraps the function to take the tool's args dict and to
    # return {"status": ..., **payload}.
    params = remote.get_scene_parameters({})
    reported = params.get("parameters", {})
    eff = reported.get("effective_fps")
    dh.record(
        "C_mcp_reports_effective",
        eff is not None and float(eff) == float(SCENE_FPS),
        {"effective_fps": eff, "expected": SCENE_FPS,
         "frame_rate_reported": reported.get("frame_rate"),
         "use_scene_fps": reported.get("use_scene_fps")},
    )

    # ---- D: a pre-rename .blend keeps its behavior -------------------
    # Simulate what an older file carries: the retired identifier as a raw
    # ID-property. The value must land on the new field, not be dropped.
    state.use_scene_fps = False
    state["use_frame_rate_in_output"] = True
    mig = __import__(pkg + ".core.migrate_renames",
                     fromlist=["migrate_renamed_state_props"])
    moved = mig.migrate_renamed_state_props(bpy.context.scene)
    dh.record(
        "D_legacy_key_migrates",
        state.use_scene_fps is True
        and "use_frame_rate_in_output" not in state.keys(),
        {"use_scene_fps_after": bool(state.use_scene_fps),
         "stale_key_removed": "use_frame_rate_in_output" not in state.keys(),
         "summary": moved},
    )

    # Leave the scene in the disagreeing state for the build: override on,
    # field still FIELD_FPS. param.toml must show SCENE_FPS.
    state.use_scene_fps = True

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="fps_source:build")

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
        .replace("<<FIELD_FPS>>", str(_FIELD_FPS))
        .replace("<<SCENE_FPS>>", str(_SCENE_FPS))
    )


def _check_param_toml_fps(workspace: str) -> tuple[bool, dict]:
    """Host-side: the solver's literal on-disk input must carry the scene's
    fps, not the `frame_rate` field that is still sitting at 100.

    This is the hop that matters. Everything upstream can agree and still be
    wrong if the value written for the solver came from the other source.
    """
    matches = glob.glob(os.path.join(workspace, "**", "param.toml"),
                        recursive=True)
    if not matches:
        return False, {"error": "no param.toml under workspace",
                       "workspace": workspace}
    found = []
    for path in matches:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("fps"):
                        try:
                            found.append(float(s.split("=", 1)[1].strip()))
                        except (ValueError, IndexError):
                            pass
        except OSError:
            continue
    ok = len(found) > 0 and all(v == float(_SCENE_FPS) for v in found)
    return ok, {"fps_in_param_toml": found, "expected": _SCENE_FPS,
                "field_value_that_must_not_win": _FIELD_FPS,
                "param_toml_paths": matches}


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    ok, details = _check_param_toml_fps(ctx.workspace)
    checks["E_param_toml_carries_fps"] = {"ok": ok, "details": details}
    return r.report_named_checks(checks)
