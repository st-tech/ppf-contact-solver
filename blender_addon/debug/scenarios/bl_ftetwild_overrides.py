# File: scenarios/bl_ftetwild_overrides.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-object tetrahedralizer override encoding (fTetWild + TetGen).
#
# Motivation: the tetrahedralizer picker and its per-field overrides
# moved from the group to the AssignedObject (per-object, like Velocity
# Overwrite), and TetGen was re-added as a backend alongside fTetWild.
# Enabled fTetWild overrides ride through ``param.pickle`` as
# ``param["group"][i][0]["ftetwild"][<object_uuid>]``; TetGen objects
# additionally carry ``backend="tetgen"``. The decoder peeks this
# per-UUID map at populate time and passes the kwargs into
# ``tetrahedralize()`` for the matching mesh.
#
# This scenario authors a SOLID group on a primitive cube, sets every
# fTetWild override on the assigned object with a non-default value,
# encodes the params, decodes the pickle, and asserts:
#   A. The SOLID group exists with the cube assigned and the
#      ftetwild_override_<field> + ftetwild_<field> properties set to
#      the test values on the AssignedObject.
#   B. The decoded ``param.pickle`` carries the seven authored values
#      under ``ftetwild[<cube_uuid>]``, and the routing dict lives at
#      ``param["group"][i][0]["ftetwild"]`` rather than at the top level.
#   C. The per-object kwargs are keyed by the cube's UUID, so a
#      downstream decoder routes them to the right mesh.
#   D. Switching the assigned object to the TetGen backend encodes
#      ``backend="tetgen"`` plus the authored TetGen override under the
#      same per-UUID slot.
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

    # The tetrahedralizer backend + overrides live on the AssignedObject
    # now (per-object, like Velocity Overwrite), so author them there.
    # The encoder skips fields whose ``ftetwild_override_<field>`` flag is
    # False, so flipping the flags on is what makes the values flow.
    group = root.object_group_0
    assigned = group.assigned_objects[0]
    # Resolve UUIDs so the encoder doesn't trip the "no UUID after
    # resolve" guard, and so subtest C can compare against a real id.
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["resolve_assigned"])
    uuid_registry.resolve_assigned(assigned)
    cube_uuid = assigned.uuid

    assigned.tet_backend = "FTETWILD"
    for field, value in AUTHORED_FLOATS.items():
        setattr(assigned, "ftetwild_override_" + field, True)
        setattr(assigned, "ftetwild_" + field, value)
    for field, value in AUTHORED_INTS.items():
        setattr(assigned, "ftetwild_override_" + field, True)
        setattr(assigned, "ftetwild_" + field, value)
    for field, value in AUTHORED_BOOLS.items():
        setattr(assigned, "ftetwild_override_" + field, True)
        setattr(assigned, "ftetwild_" + field, value)

    # ----- A: solid_group_authored ------------------------------
    authored_floats_ok = all(
        getattr(assigned, "ftetwild_override_" + f, False) is True
        and abs(float(getattr(assigned, "ftetwild_" + f)) - v) < 1e-6
        for f, v in AUTHORED_FLOATS.items()
    )
    authored_ints_ok = all(
        getattr(assigned, "ftetwild_override_" + f, False) is True
        and int(getattr(assigned, "ftetwild_" + f)) == v
        for f, v in AUTHORED_INTS.items()
    )
    authored_bools_ok = all(
        getattr(assigned, "ftetwild_override_" + f, False) is True
        and bool(getattr(assigned, "ftetwild_" + f)) is v
        for f, v in AUTHORED_BOOLS.items()
    )
    backend_default_ok = assigned.tet_backend == "FTETWILD"
    cube_assigned_ok = (
        group.object_type == "SOLID"
        and len(group.assigned_objects) == 1
        and group.assigned_objects[0].name == cube.name
        and bool(cube_uuid)
    )
    dh.record(
        "A_solid_group_authored",
        authored_floats_ok and authored_ints_ok and authored_bools_ok
        and backend_default_ok and cube_assigned_ok,
        {
            "object_type": group.object_type,
            "tet_backend": assigned.tet_backend,
            "assigned_count": len(group.assigned_objects),
            "assigned_name": (
                group.assigned_objects[0].name
                if group.assigned_objects else None
            ),
            "cube_uuid": cube_uuid,
            "floats": {
                f: float(getattr(assigned, "ftetwild_" + f))
                for f in AUTHORED_FLOATS
            },
            "ints": {
                f: int(getattr(assigned, "ftetwild_" + f))
                for f in AUTHORED_INTS
            },
            "bools": {
                f: bool(getattr(assigned, "ftetwild_" + f))
                for f in AUTHORED_BOOLS
            },
            "override_flags": {
                f: bool(getattr(assigned, "ftetwild_override_" + f, False))
                for f in list(AUTHORED_FLOATS) + list(AUTHORED_INTS)
                + list(AUTHORED_BOOLS)
            },
        },
    )

    # ----- Encode params and decode the CBOR envelope ------------
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
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
    # ``ftetwild`` is now a per-UUID map: {cube_uuid: {field: value}}.
    expected = {}
    expected.update(AUTHORED_FLOATS)
    expected.update(AUTHORED_INTS)
    expected.update(AUTHORED_BOOLS)
    routed_by_uuid = isinstance(ftetwild, dict) and cube_uuid in ftetwild
    obj_kwargs = ftetwild.get(cube_uuid) if routed_by_uuid else None
    obj_is_dict = isinstance(obj_kwargs, dict)
    keys_match = obj_is_dict and set(obj_kwargs.keys()) == set(expected.keys())
    floats_match = obj_is_dict and all(
        f in obj_kwargs and abs(float(obj_kwargs[f]) - v) < 1e-6
        for f, v in AUTHORED_FLOATS.items()
    )
    ints_match = obj_is_dict and all(
        f in obj_kwargs and int(obj_kwargs[f]) == v
        for f, v in AUTHORED_INTS.items()
    )
    bools_match = obj_is_dict and all(
        f in obj_kwargs and bool(obj_kwargs[f]) is v
        for f, v in AUTHORED_BOOLS.items()
    )
    # The encoder lives on the per-group params dict, not at the top
    # level; verify the top-level dict does NOT carry "ftetwild".
    not_at_top_level = "ftetwild" not in decoded
    dh.record(
        "B_encoded_param_carries_ftetwild",
        routed_by_uuid and keys_match and floats_match
        and ints_match and bools_match and not_at_top_level,
        {
            "ftetwild_top_keys": (
                sorted(ftetwild.keys()) if isinstance(ftetwild, dict) else None
            ),
            "obj_kwargs_keys": (
                sorted(obj_kwargs.keys()) if obj_is_dict else None
            ),
            "obj_kwargs_payload": (
                {k: obj_kwargs[k] for k in obj_kwargs} if obj_is_dict else None
            ),
            "expected": expected,
            "not_at_top_level": not_at_top_level,
            "top_level_keys": sorted(decoded.keys()),
        },
    )

    # ----- C: overrides_keyed_by_object_uuid --------------------
    # The kwargs are keyed by object UUID inside the ftetwild map, and
    # the group entry's third tuple element carries the same UUID. The
    # decoder uses these to route the kwargs into tetrahedralize().
    uuid_keys_kwargs = isinstance(ftetwild, dict) and cube_uuid in ftetwild
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
        uuid_keys_kwargs and uuid_in_routing and objects_aligned,
        {
            "cube_uuid": cube_uuid,
            "uuid_keys_kwargs": uuid_keys_kwargs,
            "group_uuids": list(group_uuids) if group_uuids else None,
            "group_objects": list(group_objects) if group_objects else None,
        },
    )

    # ----- D: tetgen_backend_encoded ----------------------------
    # Switch the same object to the TetGen backend with one override and
    # confirm the per-UUID slot carries backend="tetgen" + the knob, and
    # that the (backend-specific) fTetWild fields drop out.
    assigned.tet_backend = "TETGEN"
    assigned.tetgen_override_min_ratio = True
    assigned.tetgen_min_ratio = 1.4
    param_bytes_tg = dh.encoder_param.encode_param(bpy.context)
    decoded_tg = dh.decode_addon_blob(param_bytes_tg)
    tg_obj_kwargs = None
    for params, _objects, object_uuids in decoded_tg["group"]:
        if cube_uuid in object_uuids:
            ftw = params.get("ftetwild")
            if isinstance(ftw, dict):
                tg_obj_kwargs = ftw.get(cube_uuid)
            break
    tg_is_dict = isinstance(tg_obj_kwargs, dict)
    backend_ok = tg_is_dict and tg_obj_kwargs.get("backend") == "tetgen"
    min_ratio_ok = (
        tg_is_dict
        and "min_ratio" in tg_obj_kwargs
        and abs(float(tg_obj_kwargs["min_ratio"]) - 1.4) < 1e-6
    )
    no_ftetwild_fields = tg_is_dict and not (
        set(tg_obj_kwargs.keys())
        & (set(AUTHORED_FLOATS) | set(AUTHORED_INTS) | set(AUTHORED_BOOLS))
    )
    dh.record(
        "D_tetgen_backend_encoded",
        backend_ok and min_ratio_ok and no_ftetwild_fields,
        {
            "tg_obj_kwargs": (
                {k: tg_obj_kwargs[k] for k in tg_obj_kwargs}
                if tg_is_dict else None
            ),
            "backend_ok": backend_ok,
            "min_ratio_ok": min_ratio_ok,
            "no_ftetwild_fields": no_ftetwild_fields,
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
