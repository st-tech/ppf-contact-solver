# File: scenarios/bl_pdrd_anchor_release.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PDRD static anchor with a release frame, encoding round-trip.
#
# In the specialized rigid-body panel an Anchor holds a vertex subset fixed,
# with an optional "Release at frame N" (the per-pin use_pin_duration /
# pin_duration controls). The params encoder turns that into a per-vertex
# pin_config entry carrying ``unpin_time = pin_duration / fps`` and NO
# ``pull_strength`` (an anchor is a hard fix, never a soft pull).
#
# This builds an anchored cube corner via the production scripting API +
# the same per-pin fields the panel drives, encodes the params, decodes the
# CBOR envelope, and asserts:
#   A) the pin is anchored hard (no pull_strength in its cfg),
#   B) the release frame encodes as unpin_time = pin_duration / fps,
#   C) a held anchor carries no motion operations.
#
# Encoding-only: the emulated backend has no rigid physics, so this rig
# validates the Blender -> encoder plumbing. The drop-after-release dynamics
# run on a real CUDA host.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

RELEASE_FRAME = 30

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Rigid"
    # Anchor one corner (a single vertex = a pivot point).
    vg = cube.vertex_groups.new(name="corner")
    vg.add([0], 1.0, "REPLACE")

    fps = 100
    root = dh.configure_state(
        project_name="pdrd_anchor_release", frame_count=60, frame_rate=fps
    )
    grp_api = dh.api.solver.create_group("Rigid", "PDRD")
    grp_api.add("Rigid")
    grp_api.create_pin("Rigid", "corner")

    group = root.object_group_0
    pin = group.pin_vertex_groups[0]
    pin.use_pin_duration = True
    pin.pin_duration = RELEASE_FRAME
    # An anchor never pulls: leave use_pull at its default False.

    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    pin_config = decoded.get("pin_config", {})
    if not pin_config:
        raise RuntimeError("pin_config empty; anchor was not encoded")
    cfgs = list(list(pin_config.values())[0].values())
    if not cfgs:
        raise RuntimeError("no per-vertex cfg in pin_config")
    cfg = cfgs[0]
    dh.log(f"cfg_keys={sorted(cfg.keys())}")

    # ----- A: anchored hard (no soft pull) ---------------------------
    dh.record(
        "A_anchor_is_hard_fix",
        "pull_strength" not in cfg,
        {"cfg_keys": sorted(cfg.keys())},
    )

    # ----- B: release frame -> unpin_time = frame / fps --------------
    expected_unpin = float(RELEASE_FRAME) / fps
    got_unpin = cfg.get("unpin_time")
    dh.record(
        "B_release_frame_encodes_unpin_time",
        got_unpin is not None and abs(float(got_unpin) - expected_unpin) < 1e-6,
        {"unpin_time": got_unpin, "expected": expected_unpin},
    )

    # ----- C: a held anchor carries no motion ------------------------
    dh.record(
        "C_anchor_has_no_operations",
        not cfg.get("operations"),
        {"operations": cfg.get("operations")},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
