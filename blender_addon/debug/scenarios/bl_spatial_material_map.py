# File: scenarios/bl_spatial_material_map.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A spatial material map has to vary the parameter ACROSS the surface.
#
# The request was per-vertex weight maps that modulate material properties
# without splitting the mesh or rebuilding between frames. This covers the
# chain on the solver, so it runs on any host:
#
#   A. payload_carries_map: the encoded blob carries the map's parameter, its
#      target, and one weight per vertex.
#   B. session_varies_per_face: the built session's per-face table for that
#      parameter holds MORE THAN ONE distinct value, which is the whole point;
#      a replicated scalar would pass every other check and fail this one.
#   C. blend_endpoints_respected: the smallest per-face value is at or above
#      the group's own slider and the largest at or below the target, because
#      each face averages its own vertices' weights and no face can leave the
#      interval its endpoints define.
#
#   D. target_shares_the_base_conversion: the map TARGET reaches the solver
#      through the same conversion the group's slider does, so both ends of the
#      blend are in one unit.
#   E. a_gated_off_parameter_refuses_a_map: a map cannot reintroduce a
#      parameter the group switched off.
#   F. a_renamed_object_keeps_its_map: weights are resolved by uuid, which is
#      what they are keyed by.
#   G. a_recycled_slot_carries_no_stale_map: deleting a group and making
#      another one in the freed slot starts with no map rows.
#
# The reduction to one value per element is deliberate and checked here rather
# than left implicit: a coefficient varying INSIDE an element would stop the
# force being the gradient of any energy.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

_BASE_BEND = 1.0
_TARGET_BEND = 5000.0


_DRIVER_BODY = r"""
import glob
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
BASE_BEND = <<BASE_BEND>>
TARGET_BEND = <<TARGET_BEND>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MapMesh")
    root = dh.configure_state(project_name="spatial_map", frame_count=4)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    # Keep the object dynamic: fully pinned with no operation becomes a static
    # collider and leaves the solved namespace with its material.
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=3,
                transition="LINEAR")

    # A gradient along the mesh's own y, so the map has something to vary over.
    vg = plane.vertex_groups.new(name="Stiffen")
    ys = [v.co.y for v in plane.data.vertices]
    lo, hi = min(ys), max(ys)
    for i, v in enumerate(plane.data.vertices):
        w = 0.0 if hi == lo else (v.co.y - lo) / (hi - lo)
        vg.add([i], w, "REPLACE")

    group = root.object_group_0
    group.bend = BASE_BEND
    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "VERTEX_GROUP"
    entry.source_name = "Stiffen"
    entry.target_value = TARGET_BEND
    dh.log("map authored")

    data_bytes, param_bytes = dh.encode_payload()
    blob = dh.decode_addon_blob(param_bytes)
    seen = {}
    for e in blob.get("group", []):
        mm = e[0].get("material-maps")
        if mm:
            seen = mm
    bend_map = seen.get("bend") or {}
    weights = list((bend_map.get("weights") or {}).values())
    flat = weights[0] if weights else []
    dh.record(
        "A_payload_carries_map",
        bool(flat) and abs(bend_map.get("target", 0.0) - TARGET_BEND) < 1e-3
        and len(flat) == len(plane.data.vertices),
        {"target": bend_map.get("target"), "n_weights": len(flat),
         "n_verts": len(plane.data.vertices)},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="spatial map")
    dh.log("built")

    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]

    import struct
    path = os.path.join(session, "bin", "param", "tri-bend.bin")
    raw = open(path, "rb").read()
    vals = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    distinct = len(set(round(v, 4) for v in vals))
    dh.record(
        "B_session_varies_per_face",
        distinct > 1,
        {"n_faces": len(vals), "distinct_values": distinct,
         "min": min(vals) if vals else None, "max": max(vals) if vals else None},
    )
    dh.record(
        "C_blend_endpoints_respected",
        bool(vals) and min(vals) >= BASE_BEND - 1e-3
        and max(vals) <= TARGET_BEND + 1e-3,
        {"min": min(vals) if vals else None, "max": max(vals) if vals else None,
         "base": BASE_BEND, "target": TARGET_BEND},
    )

    # A map's TARGET is the far end of a blend whose near end is the group's
    # own slider, so it has to reach the solver through the same conversion.
    # With density normalization off the slider is a true Young's modulus in
    # pascals and the solver wants Pa/rho, so an unconverted target would sit a
    # factor of the density away from the value it is supposed to blend with.
    group.young_mod_density_normalized = False
    group.shell_density = 1200.0
    group.shell_young_modulus = 6000.0
    ym = group.material_maps.add()
    ym.parameter = "young-mod"
    ym.source_type = "VERTEX_GROUP"
    ym.source_name = "Stiffen"
    ym.target_value = 24000.0
    _data2, param2 = dh.encode_payload()
    blob2 = dh.decode_addon_blob(param2)
    ym_target = None
    ym_base = None
    for e in blob2.get("group", []):
        mm = e[0].get("material-maps")
        if mm and "young-mod" in mm:
            ym_target = mm["young-mod"].get("target")
            ym_base = e[0].get("young-mod")
    dh.record(
        "D_target_shares_the_base_conversion",
        ym_target is not None and abs(ym_target - 20.0) < 1e-3
        and ym_base is not None and abs(float(ym_base) - 5.0) < 1e-3,
        {"target_on_wire": ym_target, "expected_target": 24000.0 / 1200.0,
         "base_on_wire": float(ym_base) if ym_base is not None else None,
         "expected_base": 6000.0 / 1200.0},
    )
    group.material_maps.remove(len(group.material_maps) - 1)
    group.young_mod_density_normalized = True

    # A map cannot reintroduce a parameter the group switched off: with the
    # feature disabled the base is zero for the whole solve, so a target would
    # blend from zero toward a value the artist believes is disabled.
    group.enable_plasticity = False
    gated = group.material_maps.add()
    gated.parameter = "plasticity"
    gated.source_type = "VERTEX_GROUP"
    gated.source_name = "Stiffen"
    gated.target_value = 0.5
    refused = ""
    try:
        dh.encode_payload()
    except Exception as gate_exc:
        refused = str(gate_exc)
    dh.record(
        "E_a_gated_off_parameter_refuses_a_map",
        "enable_plasticity" in refused and "plasticity" in refused,
        {"message": refused},
    )
    group.material_maps.remove(len(group.material_maps) - 1)
    group.enable_plasticity = True

    # The weights ship keyed by object uuid, so they are resolved by uuid. A
    # name lookup would drop a renamed object's map while still shipping its
    # uuid as the key, leaving the parameter uniform with nothing reported.
    # Stale the RECORDED name directly and call the map encoder on its own.
    # Going through `encode_payload` would not test this: `encode_obj` runs
    # first and `resolve_assigned` rewrites `assigned.name` from the uuid, so a
    # name lookup would find the object anyway by the time the maps are read.
    encoder_maps = __import__(pkg + ".core.encoder.material_maps",
                              fromlist=["encode_material_maps"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["iterate_active_object_groups"])
    plane.name = "MapMeshRenamed"
    group.assigned_objects[0].name = "AStaleNameNoObjectHas"
    renamed_ok = False
    renamed_note = ""
    try:
        payload, _schedules = encoder_maps.encode_material_maps(
            bpy.context, list(groups_mod.iterate_active_object_groups(bpy.context.scene)),
            60.0, 1,
        )
        for per_key in payload.values():
            if (per_key.get("bend") or {}).get("weights"):
                renamed_ok = True
    except Exception as rename_exc:
        renamed_note = str(rename_exc)
    dh.record(
        "F_a_renamed_object_keeps_its_map",
        renamed_ok,
        {"error": renamed_note},
    )

    # A group SLOT is recycled: `create_group` hands out the lowest free one,
    # and deleting a group is `reset_to_defaults`. A collection that survives
    # that reset reappears on the next group made in the slot, naming vertex
    # groups that belong to whatever object the previous group held.
    slot_before = 0
    locks_mod = __import__(pkg + ".models.material_locks",
                           fromlist=["LOCKABLE_MATERIAL_PROPS", "lock_name"])
    retired_uuid = group.uuid
    locked_prop = locks_mod.LOCKABLE_MATERIAL_PROPS[0]
    setattr(group, locks_mod.lock_name(locked_prop), True)
    bpy.ops.object.delete_group(group_index=0)
    reused = dh.api.solver.create_group("Recycled", "SHELL")
    reused.add(plane.name)
    fresh = root.object_group_0
    # Identity and guard flags are recycled with the slot too: a uuid left
    # behind makes a retired handle resolve to an unrelated live group, and a
    # padlock left behind silently drops that parameter from the next preset.
    dh.record(
        "G_a_recycled_slot_carries_no_stale_map",
        len(fresh.material_maps) == 0 and fresh.material_maps_index == 0
        and fresh.uuid != retired_uuid
        and getattr(fresh, locks_mod.lock_name(locked_prop)) is False,
        {"slot": slot_before, "n_rows": len(fresh.material_maps),
         "rows": [(m.parameter, m.source_name) for m in fresh.material_maps],
         "index": fresh.material_maps_index,
         "uuid_reused": fresh.uuid == retired_uuid,
         "lock_survived": getattr(fresh, locks_mod.lock_name(locked_prop)),
         "locked_prop": locked_prop},
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
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
        .replace("<<BASE_BEND>>", repr(_BASE_BEND))
        .replace("<<TARGET_BEND>>", repr(_TARGET_BEND))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
