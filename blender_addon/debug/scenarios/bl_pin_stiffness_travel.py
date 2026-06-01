# File: scenarios/bl_pin_stiffness_travel.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end travel test for the per-pin `pin_stiffness` parameter.
# Sets a distinctive value on a moving (kinematic) pin in Blender and
# verifies that exact value survives every hop of the pipeline without
# being lost or silently defaulted:
#
#   Blender UI (PinVertexGroupItem.pin_stiffness)
#     -> addon encoder (core/encoder/pin.py -> cfg["pin_stiffness"])
#     -> frontend (PinData -> _pin_to_toml_dict -> _ppf_cts_py)
#     -> Rust (scene_loops.rs -> info.toml "[pin-N] stiffness = ...")
#     -> solver/emulator (parses the pin block, builds FixPair, runs)
#
# Because the emulated solver places kinematic pins exactly on target
# (no force assembly), it cannot show the FORCE effect of stiffness;
# this test proves the VALUE arrives intact at the solver's input. The
# CUDA force effect is verified separately on a GPU host.
#
# Subtests:
#   A. ui_value_set:           the Blender property holds the set value.
#   B. encoder_carries_value:  the addon encoder cfg carries it on the wire.
#   C. info_toml_carries_value (host-side): the info.toml the server's
#         frontend+Rust wrote contains `stiffness = <value>` in the pin
#         block. This is the solver's actual on-disk input.
#   D. run_completes:          the emulated solve consumed the pin block
#         (including stiffness) and produced frames.

from __future__ import annotations

import glob
import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

_FRAME_COUNT = 8
# Distinctive, non-default value so a lost field (which would default to
# 1.0) is unmistakable.
_PIN_STIFFNESS = 4.0
_TOL = 1e-4


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
PIN_STIFFNESS = <<PIN_STIFFNESS>>
TOL = <<TOL>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "StiffPlane"
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=4)
    bpy.ops.object.mode_set(mode="OBJECT")

    xmin = min(v.co.x for v in plane.data.vertices)
    edge = [v.index for v in plane.data.vertices if abs(v.co.x - xmin) < 1e-5]
    vg = plane.vertex_groups.new(name="pin")
    vg.add(edge, 1.0, "REPLACE")

    dh.save_blend(PROBE_DIR, "pin_stiffness_travel.blend")
    root = dh.configure_state(
        project_name="pin_stiffness_travel",
        frame_count=FRAME_COUNT,
        frame_rate=24,
        step_size=1.0 / 24,
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    shell = dh.api.solver.create_group("Cloth", "SHELL")
    shell.add(plane.name)
    shell.create_pin(plane.name, "pin")
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    pin_item = group.pin_vertex_groups[0]

    # A MOVE_BY operation makes the pin kinematic (a "moving" pin), the
    # case pin_stiffness applies to.
    op = pin_item.operations.add()
    op.op_type = "MOVE_BY"
    op.delta = (0.0, 0.0, 0.3)
    op.frame_start = 1
    op.frame_end = FRAME_COUNT

    # The value under test.
    pin_item.pin_stiffness = PIN_STIFFNESS

    # ---- A: UI property holds the set value -------------------------
    dh.record(
        "A_ui_value_set",
        abs(pin_item.pin_stiffness - PIN_STIFFNESS) < TOL,
        {"pin_stiffness": float(pin_item.pin_stiffness),
         "expected": PIN_STIFFNESS},
    )

    # ---- B: addon encoder carries it on the wire --------------------
    enc_pin = __import__(pkg + ".core.encoder.pin", fromlist=["_encode_pin_config"])
    state = addon_root.state
    pin_cfg = enc_pin._encode_pin_config(bpy.context, [group], state)
    obj_uuid = pin_item.object_uuid
    obj_cfg = pin_cfg.get(obj_uuid, {})
    cfg_vals = [c.get("pin_stiffness") for c in obj_cfg.values()
                if isinstance(c, dict) and "pin_stiffness" in c]
    encoder_ok = (
        len(cfg_vals) > 0
        and all(abs(float(v) - PIN_STIFFNESS) < TOL for v in cfg_vals)
    )
    dh.record(
        "B_encoder_carries_value",
        encoder_ok,
        {"cfg_pin_stiffness_values": [float(v) for v in cfg_vals],
         "expected": PIN_STIFFNESS, "n_pin_entries": len(obj_cfg)},
    )

    # ---- run the emulated solve (full pipeline) ---------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="pin_stiffness:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(plane)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    n_samples = int(arr.shape[0]) if arr is not None else 0

    # ---- D: the emulated solve consumed the pin block & ran ---------
    dh.record(
        "D_run_completes",
        n_samples >= FRAME_COUNT - 1,
        {"n_samples": n_samples, "expected_min": FRAME_COUNT - 1},
    )

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
        .replace("<<PIN_STIFFNESS>>", repr(_PIN_STIFFNESS))
        .replace("<<TOL>>", repr(_TOL))
    )


def _check_info_toml_stiffness(workspace: str) -> tuple[bool, dict]:
    """Host-side: read the info.toml the server's frontend+Rust wrote
    and confirm the pin block carries the exact stiffness value. This is
    the solver's literal on-disk input, so a match proves the value
    survived UI -> encoder -> frontend -> Rust serialization.

    The server writes under ``<workspace>/project/<project_name>/session``;
    glob for it rather than reconstructing the exact subdir name."""
    matches = glob.glob(os.path.join(workspace, "**", "info.toml"),
                        recursive=True)
    if not matches:
        return False, {"error": "no info.toml under workspace",
                       "workspace": workspace}
    found = []
    for info_path in matches:
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("stiffness"):
                        # `stiffness = 4` / `4.0`
                        try:
                            found.append(float(s.split("=", 1)[1].strip()))
                        except (ValueError, IndexError):
                            pass
        except OSError:
            continue
    ok = len(found) > 0 and all(abs(v - _PIN_STIFFNESS) < _TOL for v in found)
    return ok, {"stiffness_values_in_info_toml": found,
                "expected": _PIN_STIFFNESS, "info_toml_paths": matches}


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    # Host-side hop: the worker dir still exists here (cleanup runs after
    # run()), so read the server-written info.toml directly.
    ok, details = _check_info_toml_stiffness(ctx.workspace)
    checks["C_info_toml_carries_value"] = {"ok": ok, "details": details}
    return r.report_named_checks(checks)
