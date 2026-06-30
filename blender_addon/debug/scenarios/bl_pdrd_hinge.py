# File: scenarios/bl_pdrd_hinge.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PDRD per-object hinge encoding round-trip.
#
# The PDRD hinge is a PER-OBJECT setting (one PDRD group can hold
# several bodies, e.g. a gear train, each pinned on its own axle). Each
# AssignedObject carries ``pdrd_hinge_enable`` + ``pdrd_hinge_axis``, and
# the params encoder emits a per-UUID dict
#   param["group"][i][0]["hinge"][uuid] = <principal-axis index 0/1/2>
# containing ONLY the hinge-enabled bodies. The frontend decoder turns
# each entry into an ``Object.hinge(axis)`` call.
#
# This scenario assigns two PDRD cubes to one group, hinges them on
# DIFFERENT principal axes via the production ``Group.set_hinge`` API
# (the same per-object path the UI / MCP tool drive), encodes the
# params, decodes the CBOR envelope, and asserts:
#   A) set_hinge wrote the per-object enable + axis on each body,
#   B) the encoded "hinge" dict carries each enabled body on its own
#      axis (per-object independence),
#   C) disabling one body removes only its entry (per-object gating).
#
# Encoding-only: no build / run / fetch (mirrors bl_velocity_keyframes).
# The hinge dynamics themselves are exercised on a real CUDA host by
# examples/pdrd_hinge.py and examples/pdrd_gear.py; the emulated backend
# has no physics, so this rig validates the Blender->encoder plumbing.

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

    # Two closed cubes (PDRD needs an enclosed volume) in one group.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-1.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "GearA"
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(1.0, 0.0, 0.0))
    cube_b = bpy.context.active_object
    cube_b.name = "GearB"

    root = dh.configure_state(project_name="pdrd_hinge", frame_count=4)
    gears = dh.api.solver.create_group("Gears", "PDRD")
    gears.add("GearA", "GearB")

    # Per-object hinge via the production scripting API. Distinct axes
    # prove the setting is genuinely per-object, not group-wide.
    gears.set_hinge("GearA", pca_axis=2, enable=True)
    gears.set_hinge("GearB", pca_axis=0, enable=True)

    group = root.object_group_0
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["resolve_assigned"])
    a_assigned = None
    b_assigned = None
    for assigned in group.assigned_objects:
        uuid_registry.resolve_assigned(assigned)
        if assigned.name == "GearA":
            a_assigned = assigned
        elif assigned.name == "GearB":
            b_assigned = assigned
    if a_assigned is None or b_assigned is None:
        raise RuntimeError("could not resolve both assigned PDRD bodies")
    uuid_a = a_assigned.uuid
    uuid_b = b_assigned.uuid

    # ----- A: set_hinge wrote per-object enable + axis ---------------
    authored_ok = (
        a_assigned.pdrd_hinge_enable
        and a_assigned.pdrd_hinge_axis == "2"
        and b_assigned.pdrd_hinge_enable
        and b_assigned.pdrd_hinge_axis == "0"
    )
    dh.record(
        "A_set_hinge_wrote_per_object_state",
        authored_ok,
        {
            "GearA": (a_assigned.pdrd_hinge_enable, a_assigned.pdrd_hinge_axis),
            "GearB": (b_assigned.pdrd_hinge_enable, b_assigned.pdrd_hinge_axis),
        },
    )

    def _hinge_dict():
        param_bytes = dh.encoder_param.encode_param(bpy.context)
        decoded = dh.decode_addon_blob(param_bytes)
        for params, _objs, object_uuids in decoded["group"]:
            if uuid_a in object_uuids or uuid_b in object_uuids:
                return params.get("hinge", {})
        raise RuntimeError("could not locate the PDRD group in decoded params")

    # ----- B: both enabled bodies carry their own axis ---------------
    hinge = _hinge_dict()
    both_ok = (
        int(hinge.get(uuid_a, -1)) == 2
        and int(hinge.get(uuid_b, -1)) == 0
        and len(hinge) == 2
    )
    dh.record(
        "B_per_object_axes_encoded",
        both_ok,
        {
            "hinge": {k: int(v) for k, v in hinge.items()},
            "expected": {uuid_a: 2, uuid_b: 0},
        },
    )

    # ----- C: disabling one body removes only its entry --------------
    gears.set_hinge("GearB", enable=False)
    hinge2 = _hinge_dict()
    gated_ok = (
        int(hinge2.get(uuid_a, -1)) == 2
        and uuid_b not in hinge2
        and len(hinge2) == 1
    )
    dh.record(
        "C_disable_removes_only_that_body",
        gated_ok,
        {"hinge_after_disable": {k: int(v) for k, v in hinge2.items()}},
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
