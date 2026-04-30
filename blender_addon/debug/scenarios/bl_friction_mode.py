# File: scenarios/bl_friction_mode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the friction-mode session parameter. For each of the
# three allowed values (MIN, MAX, MEAN) the scenario sets
# ``state.friction_mode`` in the addon, runs the standard build
# pipeline, then reads ``param.toml`` from the session directory and
# asserts the on-disk value matches.
#
# Why ``param.toml`` is enough proof:
#   - Blender state -> encoder -> frontend -> ``param.toml``: the
#     file's existence with the right key is the witness.
#   - ``param.toml`` -> ``SimArgs`` -> ``ParamSet``: the emulated
#     Rust binary deserializes ``param.toml`` during build. A missing
#     or unknown ``friction_mode`` would fail deserialization, and an
#     unknown value would panic in ``builder.rs::make_param``. So a
#     successful ``build_and_wait`` is itself the witness for the
#     TOML -> Rust -> CUDA-stub leg of the trip.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import glob
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"


def latest_param_toml(project_root):
    matches = glob.glob(os.path.join(project_root, "**", "param.toml"),
                        recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def parse_friction_mode(toml_path):
    with open(toml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("friction_mode"):
                _, _, rhs = line.partition("=")
                return rhs.strip().strip('"')
    return None


def run_pass(dh, plane, mode_id, mode_value):
    root = dh.groups.get_addon_data(bpy.context.scene)
    root.state.friction_mode = mode_id
    dh.log("set friction_mode=%s" % mode_id)

    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes,
                      message="friction-mode-%s" % mode_value)

    toml_path = latest_param_toml(PROJECT_ROOT)
    if not toml_path:
        raise RuntimeError("no param.toml under %s after build" % PROJECT_ROOT)
    seen = parse_friction_mode(toml_path)
    dh.record(
        "friction_mode_%s" % mode_value,
        seen == mode_value,
        {"set": mode_id, "expected": mode_value, "seen": seen,
         "toml": toml_path},
    )
    dh.log("pass %s seen=%r" % (mode_value, seen))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="FrictionModeMesh")
    dh.save_blend(PROBE_DIR, "friction_mode.blend")
    # Use the rig's slot project_name so the BlenderApp writes session
    # files under ctx.project_root.
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=2)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    for mode_id, mode_value in (("MIN", "min"), ("MAX", "max"), ("MEAN", "mean")):
        run_pass(dh, plane, mode_id, mode_value)

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
        .replace("<<PROJECT_ROOT>>", ctx.project_root)
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
