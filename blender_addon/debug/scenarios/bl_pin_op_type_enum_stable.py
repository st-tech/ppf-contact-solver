# File: scenarios/bl_pin_op_type_enum_stable.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for the ``PinOperation.op_type`` EnumProperty slot
# assignments.
#
# Blender stores EnumProperty values as integers in DNA. When the items
# list uses 3-tuples (no explicit number), Blender auto-numbers them by
# list order, so removing or reordering an item silently shifts every
# saved op's identifier in existing .blend files. Commit f628489d removed
# ``EMBEDDED_MOVE`` from slot 0 and surfaced as SCALE ops reading back as
# TORQUE on previously-saved scenes, with the user-visible symptom
# "torque cannot be mixed with Move/Spin/Scale operations" on a scene
# that had never had torque.
#
# This scenario freezes the slot assignments and roundtrips through
# save/open so a future refactor can't reintroduce the bug:
#
#   A. enum_slots_locked: every identifier maps to its expected integer
#      value (EMBEDDED_MOVE=0, MOVE_BY=1, SPIN=2, SCALE=3, TORQUE=4).
#   B. saved_blend_roundtrips_op_type: write a scene with one op of each
#      type, save_as_mainfile + open_mainfile, assert op_type survives.
#   C. encoder_skips_embedded_move: an EMBEDDED_MOVE op (legacy slot)
#      doesn't appear in the encoder payload, since it has no solver
#      semantics.
#
# Pure UI/encoder scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, os, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


# Expected slot assignments. Touching this dict is the same as touching
# the enum: pre-existing .blend files will decode to the wrong
# identifiers. If you intentionally retire a slot, leave its entry here
# with the same number and filter it at use sites.
EXPECTED_SLOTS = {
    "EMBEDDED_MOVE": 0,
    "MOVE_BY": 1,
    "SPIN": 2,
    "SCALE": 3,
    "TORQUE": 4,
}


try:
    state_types = __import__(pkg + ".ui.state_types",
                             fromlist=["PinOperation"])
    encoder_pin = __import__(pkg + ".core.encoder.pin",
                             fromlist=["_encode_pin_config"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # ----- A: each identifier has its expected stable integer ------
    enum_prop = state_types.PinOperation.bl_rna.properties["op_type"]
    actual_slots = {it.identifier: it.value for it in enum_prop.enum_items}
    record(
        "A_enum_slots_locked",
        actual_slots == EXPECTED_SLOTS,
        {"expected": EXPECTED_SLOTS, "actual": actual_slots},
    )

    # ----- Build a scene with one op of each type -------------------
    # Subdivided plane: 9 vertices, enough to split into two disjoint
    # pin groups while keeping the torque group above the PCA minimum
    # (3 vertices) in encoder/pin.py.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "EnumMesh"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode="OBJECT")
    n_verts = len(plane.data.vertices)
    # TORQUE can't coexist with kinematic ops on the same pin (encoder
    # rejects the mix), and overlapping pin groups would let the second
    # pin's per-vertex cfg overwrite the first's. Use disjoint vertex
    # subsets so both pins land in the encoder's pin_config dict.
    half = n_verts // 2
    vg_kin = plane.vertex_groups.new(name="KinPin")
    vg_kin.add(list(range(half)), 1.0, "REPLACE")
    vg_tq = plane.vertex_groups.new(name="TorquePin")
    vg_tq.add(list(range(half, n_verts)), 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "KinPin")
    pin.move_by(delta=(0.1, 0, 0), frame_start=1, frame_end=10)
    pin.spin(axis=(0, 0, 1), angular_velocity=90.0,
             frame_start=1, frame_end=10)
    pin.scale(factor=0.5, frame_start=1, frame_end=10)
    pin_t = cloth.create_pin(plane.name, "TorquePin")
    pin_t.torque(magnitude=2.0, axis_component="PC3",
                 frame_start=1, frame_end=10)

    state = groups_mod.get_addon_data(bpy.context.scene).state
    state.project_name = "enum_stable"

    # Group UUID is stable across save/open; use it to relocate the
    # pin_item dict after the .blend reload.
    group_uuid = pin._group_uuid


    def find_group_by_uuid(scene, uuid):
        root = groups_mod.get_addon_data(scene)
        for i in range(32):
            g = getattr(root, "object_group_" + str(i), None)
            if g is not None and getattr(g, "uuid", "") == uuid:
                return g
        return None

    group = find_group_by_uuid(bpy.context.scene, group_uuid)
    assert group is not None, "freshly-created group not found by uuid"
    # The kinematic pin and the torque pin sit side by side; capture both
    # so the roundtrip check covers every op_type identifier.
    pre_types = [
        [op.op_type for op in pv.operations]
        for pv in group.pin_vertex_groups
    ]
    log("pre_types=" + repr(pre_types))

    # ----- B: save .blend, reopen, op_type must survive -------------
    blend_path = os.path.join(os.path.dirname(PROBE_DIR),
                              "enum_stable.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    log("saved blend at " + blend_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    log("reopened blend")

    group = find_group_by_uuid(bpy.context.scene, group_uuid)
    if group is None:
        record(
            "B_saved_blend_roundtrips_op_type",
            False,
            {"error": "group disappeared after reopen",
             "group_uuid": group_uuid},
        )
    else:
        post_types = [
            [op.op_type for op in pv.operations]
            for pv in group.pin_vertex_groups
        ]
        record(
            "B_saved_blend_roundtrips_op_type",
            post_types == pre_types,
            {"pre": pre_types, "post": post_types},
        )

    # ----- C: encoder filters EMBEDDED_MOVE -------------------------
    # Inject a legacy EMBEDDED_MOVE op on the kinematic pin (the UI
    # never offers it, but a pre-bug .blend can have one). Encode the
    # group and confirm the legacy slot does not leak into the payload
    # while every real op_type does.
    group = find_group_by_uuid(bpy.context.scene, group_uuid)
    kin_pin = group.pin_vertex_groups[0]
    legacy = kin_pin.operations.add()
    legacy.op_type = "EMBEDDED_MOVE"
    legacy.frame_start = 1
    legacy.frame_end = 10

    state_post = groups_mod.get_addon_data(bpy.context.scene).state
    pin_config = encoder_pin._encode_pin_config(
        bpy.context, [group], state_post,
    )
    # Walk every per-vertex cfg from both pins, union the emitted op
    # types together. Vertex-group overlap is fine: only the types
    # matter for the filter check.
    emitted_ops = set()
    for obj_cfg in pin_config.values():
        for vert_cfg in obj_cfg.values():
            for op in vert_cfg.get("operations", []):
                emitted_ops.add(op.get("type"))
    record(
        "C_encoder_skips_embedded_move",
        "embedded_move" not in emitted_ops
        and emitted_ops == {"move_by", "spin", "scale", "torque"},
        {"emitted_ops": sorted(emitted_ops)},
    )

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
