# File: scenarios/bl_ftetwild_overrides.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-group fTetWild override encoding.
#
# Motivation: commit 71325d15 ("Per-group fTetWild overrides and
# velocity keyframe copy/paste"). SOLID groups gained per-field
# fTetWild overrides (edge_length_fac, epsilon, stop_energy,
# num_opt_iter, optimize, simplify, coarsen) behind per-field enable
# toggles. Only enabled overrides ride through ``param.pickle`` as
# ``param["group"][i][0]["ftetwild"]``; the decoder later fans the
# kwargs out by UUID into ``tetrahedralize()``.
#
# This scenario authors a SOLID group on a primitive cube, sets every
# override toggle on with a non-default value, encodes the params,
# decodes the pickle, and asserts:
#   A. The SOLID group exists with the cube assigned and the
#      ftetwild_override_<field> + ftetwild_<field> properties set to
#      the test values on the PropertyGroup.
#   B. The decoded ``param.pickle`` carries the seven authored values
#      under the SOLID group's ``ftetwild`` dict, and the routing dict
#      lives at ``param["group"][i][0]["ftetwild"]`` rather than at
#      the top level.
#   C. The encoded group entry's ``object_uuids`` carries the cube's
#      UUID, so a downstream decoder can route the kwargs to the right
#      mesh by UUID.
#
# No build / run / fetch is involved; this is encoding-only. Same
# shape as bl_velocity_keyframes and bl_pc2_migration.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import pickle
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


# Authored override values. Every one differs from the default in
# OBJECT_GROUP_DEFAULTS (see blender_addon/models/groups.py) so the
# encoded payload clearly reflects the test values, not defaults:
#   default edge_length_fac = 0.05  -> 0.08
#   default epsilon         = 1e-3  -> 5e-3
#   default stop_energy     = 10.0  -> 12.5
#   default num_opt_iter    = 80    -> 42
#   default optimize        = True  -> False
#   default simplify        = True  -> False
#   default coarsen         = False -> True
AUTHORED_FLOATS = {
    "edge_length_fac": 0.08,
    "epsilon": 5e-3,
    "stop_energy": 12.5,
}
AUTHORED_INTS = {
    "num_opt_iter": 42,
}
AUTHORED_BOOLS = {
    "optimize": False,
    "simplify": False,
    "coarsen": True,
}


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe the scene and create a primitive cube to assign into the
    # SOLID group. SOLID groups need a closed surface to hand to
    # fTetWild, so a cube is the simplest valid input.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "FTetWildCube"

    root = dh.configure_state(project_name="ftetwild_overrides",
                              frame_count=4, frame_rate=100)
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)

    # Reach into the underlying PropertyGroup and set every override
    # toggle + value. The encoder skips fields whose
    # ``ftetwild_override_<field>`` flag is False, so flipping the
    # flags on is what makes the values flow through.
    group = root.object_group_0
    for field, value in AUTHORED_FLOATS.items():
        setattr(group, "ftetwild_override_" + field, True)
        setattr(group, "ftetwild_" + field, value)
    for field, value in AUTHORED_INTS.items():
        setattr(group, "ftetwild_override_" + field, True)
        setattr(group, "ftetwild_" + field, value)
    for field, value in AUTHORED_BOOLS.items():
        setattr(group, "ftetwild_override_" + field, True)
        setattr(group, "ftetwild_" + field, value)

    assigned = group.assigned_objects[0]
    # Resolve UUIDs so the encoder doesn't trip the "no UUID after
    # resolve" guard, and so subtest C can compare against a real id.
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["resolve_assigned"])
    uuid_registry.resolve_assigned(assigned)
    cube_uuid = assigned.uuid

    # ----- A: solid_group_authored ------------------------------
    authored_floats_ok = all(
        getattr(group, "ftetwild_override_" + f, False) is True
        and abs(float(getattr(group, "ftetwild_" + f)) - v) < 1e-6
        for f, v in AUTHORED_FLOATS.items()
    )
    authored_ints_ok = all(
        getattr(group, "ftetwild_override_" + f, False) is True
        and int(getattr(group, "ftetwild_" + f)) == v
        for f, v in AUTHORED_INTS.items()
    )
    authored_bools_ok = all(
        getattr(group, "ftetwild_override_" + f, False) is True
        and bool(getattr(group, "ftetwild_" + f)) is v
        for f, v in AUTHORED_BOOLS.items()
    )
    cube_assigned_ok = (
        group.object_type == "SOLID"
        and len(group.assigned_objects) == 1
        and group.assigned_objects[0].name == cube.name
        and bool(cube_uuid)
    )
    dh.record(
        "A_solid_group_authored",
        authored_floats_ok and authored_ints_ok and authored_bools_ok
        and cube_assigned_ok,
        {
            "object_type": group.object_type,
            "assigned_count": len(group.assigned_objects),
            "assigned_name": (
                group.assigned_objects[0].name
                if group.assigned_objects else None
            ),
            "cube_uuid": cube_uuid,
            "floats": {
                f: float(getattr(group, "ftetwild_" + f))
                for f in AUTHORED_FLOATS
            },
            "ints": {
                f: int(getattr(group, "ftetwild_" + f))
                for f in AUTHORED_INTS
            },
            "bools": {
                f: bool(getattr(group, "ftetwild_" + f))
                for f in AUTHORED_BOOLS
            },
            "override_flags": {
                f: bool(getattr(group, "ftetwild_override_" + f, False))
                for f in list(AUTHORED_FLOATS) + list(AUTHORED_INTS)
                + list(AUTHORED_BOOLS)
            },
        },
    )

    # ----- Encode params and decode the pickle ------------------
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = pickle.loads(param_bytes)
    dh.log("decoded keys=" + repr(sorted(decoded.keys())))

    # group is encoded as [(params_dict, objects, object_uuids), ...].
    # Locate our SOLID group by UUID match, the same way
    # bl_velocity_keyframes does.
    group_params = None
    group_objects = None
    group_uuids = None
    for params, objects, object_uuids in decoded["group"]:
        if cube_uuid in object_uuids:
            group_params = params
            group_objects = objects
            group_uuids = object_uuids
            break
    if group_params is None:
        raise RuntimeError(
            "could not locate SOLID group entry for uuid "
            + repr(cube_uuid) + " in decoded param.pickle"
        )

    ftetwild = group_params.get("ftetwild")

    # ----- B: encoded_param_carries_ftetwild --------------------
    expected = {}
    expected.update(AUTHORED_FLOATS)
    expected.update(AUTHORED_INTS)
    expected.update(AUTHORED_BOOLS)
    routed_under_group = isinstance(ftetwild, dict)
    keys_match = routed_under_group and set(ftetwild.keys()) == set(expected.keys())
    floats_match = routed_under_group and all(
        f in ftetwild and abs(float(ftetwild[f]) - v) < 1e-6
        for f, v in AUTHORED_FLOATS.items()
    )
    ints_match = routed_under_group and all(
        f in ftetwild and int(ftetwild[f]) == v
        for f, v in AUTHORED_INTS.items()
    )
    bools_match = routed_under_group and all(
        f in ftetwild and bool(ftetwild[f]) is v
        for f, v in AUTHORED_BOOLS.items()
    )
    # The encoder lives on the per-group params dict, not at the top
    # level; verify the top-level dict does NOT carry "ftetwild".
    not_at_top_level = "ftetwild" not in decoded
    dh.record(
        "B_encoded_param_carries_ftetwild",
        routed_under_group and keys_match and floats_match
        and ints_match and bools_match and not_at_top_level,
        {
            "ftetwild_keys": (
                sorted(ftetwild.keys()) if routed_under_group else None
            ),
            "ftetwild_payload": (
                {k: ftetwild[k] for k in ftetwild}
                if routed_under_group else None
            ),
            "expected": expected,
            "not_at_top_level": not_at_top_level,
            "top_level_keys": sorted(decoded.keys()),
        },
    )

    # ----- C: overrides_keyed_by_object_uuid --------------------
    # The group entry's third tuple element carries the per-object
    # UUIDs; the decoder uses these to fan the ftetwild kwargs out
    # by UUID into tetrahedralize(). Confirm our cube's UUID lands
    # there and the assigned-object name lines up.
    uuid_in_routing = (
        isinstance(group_uuids, list)
        and cube_uuid in group_uuids
        and len(group_uuids) == 1
    )
    objects_aligned = (
        isinstance(group_objects, list)
        and cube.name in group_objects
        and len(group_objects) == 1
    )
    dh.record(
        "C_overrides_keyed_by_object_uuid",
        uuid_in_routing and objects_aligned,
        {
            "cube_uuid": cube_uuid,
            "group_uuids": list(group_uuids) if group_uuids else None,
            "group_objects": list(group_objects) if group_objects else None,
        },
    )

except Exception as exc:
    result["errors"].append(repr(type(exc).__name__) + ": " + str(exc))
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
