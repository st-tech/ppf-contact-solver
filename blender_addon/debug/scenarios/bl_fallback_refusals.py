# File: scenarios/bl_fallback_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Five places where the solver could be handed something other than what the
# artist authored, each without a word: a deformer dropped because a later
# modifier changed the vertex count, a pin that holds nothing transferred as
# no pin, a withdrawn shell model run as another one, a spin center described
# in a frame the code never used, and a partially pinned SOLID whose pin
# field fell back to another vertex set whenever it could not be built. Each
# either reaches the solver as authored or is refused by name.
#
# Subtests:
#   A. armature_before_subdivision_reaches_shipped_pose: an Armature in front
#      of a Subdivision is in the pose the object ships at, at the base vertex
#      count, and a captured pin on that object is no longer refused for the
#      Subdivision.
#   B. deformer_after_subdivision_stays_out: an Armature AFTER a Subdivision
#      is not, because the output cache sits in front of the Subdivision and
#      that Armature deforms the simulated mesh on display. The cut the
#      encoder evaluates at is the cut the cache is placed at.
#   C. empty_vertex_group_pin_refused: a pin on an existing but empty mesh
#      vertex group, and a curve pin whose control-point list is empty, are
#      refused by name by the data encoder and by the param encoder alike.
#   D. shell_stable_neohookean_refused: a SHELL group holding the withdrawn
#      Stable NeoHookean model (still registered at item number 0, with no
#      UI name) is refused by name with the sentence the panel shows, while
#      a SOLID group keeps it.
#   E. absolute_center_is_a_world_position: the Fixed spin and scale center
#      is described as a world position by its RNA and by both MCP pin
#      operation tools, and the encoder reads it as one: a center at the
#      object's origin encodes as the op frame's zero.
#   F. partial_solid_builds_with_diffused_weights: a SOLID pinned on its top
#      face only builds through the strict pin field, and the pull pin the
#      build writes carries the diffused per-vertex weights, graded from held
#      to free. The strictness did not turn a working scene into a refusal.

from __future__ import annotations

from . import REPO_ROOT_POSIX
from . import _driver_lib as dl
from . import _runner as r

NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: subtests A to E encode and never reach the
# solver, and F builds (tetrahedralization and the pin field run in the
# server's frontend) without stepping, so nothing in it depends on which
# backend the build came out of.
BACKENDS = ("real",)


_LIFT = 0.3   # how far the posed bone raises every skinned vertex (meters)
_PULL = 2.0   # pull strength of the partial SOLID pin


_DRIVER_BODY = r"""
import json
import os
import tomllib
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
LIFT = <<LIFT>>
PULL = <<PULL>>


def encode_error(which="both"):
    try:
        if which in ("both", "data"):
            dh.encoder_mesh.encode_obj(bpy.context)
        if which in ("both", "param"):
            dh.encoder_param.encode_param(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def shipped_vert(obj):
    payload = DriverHelpers.decode_addon_blob(
        dh.encoder_mesh.encode_obj(bpy.context))
    for group in payload:
        for info in group.get("object", []):
            if info.get("name") == obj.name:
                return np.asarray(info["vert"], dtype=np.float64)
    raise RuntimeError(f"'{obj.name}' is not in the encoded scene")


def local_co(obj, hidden=()):
    # The evaluated object-local positions with the named modifiers off.
    saved = [m for m in obj.modifiers if m.name in hidden and m.show_viewport]
    for m in saved:
        m.show_viewport = False
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        eo = obj.evaluated_get(dg)
        me = eo.to_mesh()
        try:
            co = np.empty(len(me.vertices) * 3, dtype=np.float64)
            me.vertices.foreach_get("co", co)
            return co.reshape(-1, 3)
        finally:
            eo.to_mesh_clear()
    finally:
        for m in saved:
            m.show_viewport = True


def rest_co(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def grid(name, x, cuts=3):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, 0.0, 0.0))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "Mesh"
    return obj


def rigged(name, x, rig, order):
    # A grid skinned whole to the rig's one bone, carrying an Armature and
    # a Subdivision modifier in the given order.
    obj = grid(name, x)
    obj.vertex_groups.new(name="Bone").add(
        list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    for kind in order:
        if kind == "ARMATURE":
            obj.modifiers.new(name="Armature", type="ARMATURE").object = rig
        else:
            obj.modifiers.new(name="Subdivision", type="SUBSURF")
    return obj


def group_of(api_group):
    return dh.groups.get_active_group_by_uuid(bpy.context.scene,
                                              api_group.uuid)


try:
    dh = DriverHelpers(pkg, result)
    pc2 = __import__(pkg + ".core.pc2",
                     fromlist=["modifiers_after_cache_boundary"])
    pin_mod = __import__(pkg + ".core.encoder.pin",
                         fromlist=["_encode_pin_config"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_object_uuid"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.save_blend(PROBE_DIR, "fallback_refusals.blend")
    root = dh.configure_state(project_name="fallback_refusals",
                              frame_count=6)
    state = root.state
    scene = bpy.context.scene
    created = []

    # One bone standing on +Z, posed to rise LIFT along its own length, so
    # every vertex skinned to it rises LIFT in Z.
    rig_data = bpy.data.armatures.new("RigData")
    rig = bpy.data.objects.new("Rig", rig_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bone = rig_data.edit_bones.new("Bone")
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.pose.bones["Bone"].location = (0.0, LIFT, 0.0)

    # ---- A: an Armature in front of a Subdivision ------------------------
    front = rigged("ArmatureFirst", 0.0, rig, ("ARMATURE", "SUBSURF"))
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    created.append(cloth)
    cloth.add(front.name)
    cloth.create_pin(front.name, "Bone")
    captured = group_of(cloth).pin_vertex_groups[0]
    rest_a = rest_co(front)
    posed_a = local_co(front, hidden=("Subdivision",))
    n_full_a = len(local_co(front))
    shipped_a = shipped_vert(front)
    # A captured pin needs the pose its track starts from, which was refused
    # whenever a modifier changed the vertex count.
    captured.has_captured_anim = True
    err_captured = encode_error("data")
    captured.has_captured_anim = False
    dh.record(
        "A_armature_before_subdivision_reaches_shipped_pose",
        n_full_a != len(rest_a)
        and shipped_a.shape == rest_a.shape
        and np.allclose(shipped_a, posed_a, atol=1e-5)
        and np.allclose(shipped_a[:, 2] - rest_a[:, 2], LIFT, atol=1e-4)
        and err_captured == "",
        {"base_verts": len(rest_a), "evaluated_verts": n_full_a,
         "shipped_shape": list(shipped_a.shape),
         "max_off_pose": float(np.max(np.abs(shipped_a - posed_a)))
         if shipped_a.shape == posed_a.shape else None,
         "mean_rise": float(np.mean(shipped_a[:, 2] - rest_a[:, 2]))
         if shipped_a.shape == rest_a.shape else None,
         "captured_pin_error": err_captured[:300]},
    )

    # ---- B: an Armature after the Subdivision -----------------------------
    back = rigged("SubdivisionFirst", 3.0, rig, ("SUBSURF", "ARMATURE"))
    cloth.add(back.name)
    rest_b = rest_co(back)
    shipped_b = shipped_vert(back)
    cut_front = [m.name for m in pc2.modifiers_after_cache_boundary(front)]
    cut_back = [m.name for m in pc2.modifiers_after_cache_boundary(back)]
    dh.record(
        "B_deformer_after_subdivision_stays_out",
        shipped_b.shape == rest_b.shape
        and np.allclose(shipped_b, rest_b, atol=1e-6)
        and cut_front == ["Subdivision"]
        and cut_back == ["Subdivision", "Armature"],
        {"max_off_rest": float(np.max(np.abs(shipped_b - rest_b)))
         if shipped_b.shape == rest_b.shape else None,
         "cut_front": cut_front, "cut_back": cut_back},
    )
    cloth.delete()
    created.remove(cloth)

    # ---- C: pins that hold nothing ----------------------------------------
    sheet = grid("Sheet", 6.0)
    row = sheet.vertex_groups.new(name="Row")
    row.add([0, 1, 2, 3], 1.0, "REPLACE")
    drape = dh.api.solver.create_group("Drape", "SHELL")
    created.append(drape)
    drape.add(sheet.name)
    drape.create_pin(sheet.name, "Row").pull(strength=1.0)
    ok_c = encode_error()
    row.remove([0, 1, 2, 3])
    mesh_data = encode_error("data")
    mesh_param = encode_error("param")
    row.add([0, 1, 2, 3], 1.0, "REPLACE")
    ok_c_again = encode_error()

    strand_data = bpy.data.curves.new("StrandData", type="CURVE")
    strand_data.dimensions = "3D"
    spline = strand_data.splines.new("POLY")
    spline.points.add(2)
    for i, p in enumerate(spline.points):
        p.co = (0.0, 4.0 + 0.5 * i, 0.0, 1.0)
    strand = bpy.data.objects.new("Strand", strand_data)
    bpy.context.collection.objects.link(strand)
    rods = dh.api.solver.create_group("Strands", "ROD")
    created.append(rods)
    rods.add(strand.name)
    rods.create_pin(strand.name, "Tip", indices=[]).pull(strength=1.0)
    curve_data = encode_error("data")
    curve_param = encode_error("param")
    strand["_pin_Tip"] = json.dumps([0])
    ok_curve = encode_error()
    mesh_words = "Pin 'Row' on 'Sheet' in group 'Drape'"
    curve_words = "Pin 'Tip' on 'Strand' in group 'Strands'"
    dh.record(
        "C_empty_vertex_group_pin_refused",
        ok_c == "" and ok_c_again == "" and ok_curve == ""
        and all(mesh_words in e and "holds no vertices" in e
                for e in (mesh_data, mesh_param))
        and all(curve_words in e and "holds no control points" in e
                for e in (curve_data, curve_param)),
        {"before": ok_c[:300], "mesh_data": mesh_data[:300],
         "mesh_param": mesh_param[:300], "refilled": ok_c_again[:300],
         "curve_data": curve_data[:300], "curve_param": curve_param[:300],
         "curve_filled": ok_curve[:300]},
    )
    rods.delete()
    created.remove(rods)

    # ---- D: the withdrawn shell model -------------------------------------
    pg = group_of(drape)
    prop = pg.bl_rna.properties["shell_model"]
    withdrawn = prop.enum_items[dh.groups.WITHDRAWN_SHELL_MODEL]
    pg.shell_model = "STABLE_NEOHOOKEAN"
    err_d = encode_error("param")
    panel = dh.groups.withdrawn_shell_model_refusal(pg)
    pg.shell_model = "ARAP"
    ok_d = encode_error("param")
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(9.0, 0.0, 0.0))
    block = bpy.context.active_object
    block.name = "Block"
    solid = dh.api.solver.create_group("Solid", "SOLID")
    created.append(solid)
    solid.add(block.name)
    group_of(solid).solid_model = "STABLE_NEOHOOKEAN"
    ok_solid = encode_error("param")
    solid.delete()
    created.remove(solid)
    dh.record(
        "D_shell_stable_neohookean_refused",
        withdrawn.value == 0 and withdrawn.name == ""
        and err_d.startswith("ValueError: ")
        and "Group 'Drape'" in err_d and "Stable NeoHookean" in err_d
        and "ARAP or Baraff-Witkin" in err_d
        and panel is not None and err_d == "ValueError: " + panel
        and ok_d == "" and ok_solid == "",
        {"item_number": withdrawn.value, "item_name": withdrawn.name,
         "error": err_d[:300], "panel": (panel or "")[:300],
         "arap_error": ok_d[:300], "solid_error": ok_solid[:300]},
    )

    # ---- E: the Fixed center is a world position --------------------------
    mover = grid("Mover", 0.0)
    mover.location = (2.0, 1.0, 0.5)
    mover.vertex_groups.new(name="Held").add([0, 1, 2], 1.0, "REPLACE")
    turn = dh.api.solver.create_group("Turn", "SHELL")
    created.append(turn)
    turn.add(mover.name)
    held = turn.create_pin(mover.name, "Held")
    held.spin(axis=(0.0, 0.0, 1.0), angular_velocity=90.0,
              center=(2.0, 1.0, 0.5), frame_start=1, frame_end=5)
    held.scale(factor=0.5, center=(2.0, 1.0, 1.5), frame_start=1,
               frame_end=5)
    pg_e = group_of(turn)
    op = pg_e.pin_vertex_groups[0].operations[0]
    rna = op.bl_rna.properties
    texts = {
        "spin_center": rna["spin_center"].description,
        "scale_center": rna["scale_center"].description,
        "spin_center_mode.ABSOLUTE":
            rna["spin_center_mode"].enum_items["ABSOLUTE"].description,
        "scale_center_mode.ABSOLUTE":
            rna["scale_center_mode"].enum_items["ABSOLUTE"].description,
    }
    __import__(pkg + ".mcp.handlers.object_ops",
               fromlist=["add_pin_operation"])
    registry = __import__(pkg + ".mcp.decorators",
                          fromlist=["get_handler_registry"]
                          ).get_handler_registry()
    for tool in ("add_pin_operation", "set_pin_operation"):
        props = registry[tool]["schema"]["inputSchema"]["properties"]
        for field in ("spin_center", "scale_center"):
            texts[tool + "." + field] = props[field].get("description", "")
    cfg = pin_mod._encode_pin_config(bpy.context, [pg_e], state)
    entries = [e for per_obj in cfg.values() for e in per_obj.values()]
    encoded = {o["type"]: o for o in entries[0]["operations"]} if entries else {}
    spin_c = np.asarray(encoded.get("spin", {}).get("center", [np.nan] * 3))
    scale_c = np.asarray(encoded.get("scale", {}).get("center", [np.nan] * 3))
    dh.record(
        "E_absolute_center_is_a_world_position",
        all("world position" in t for t in texts.values())
        and not any("local frame" in t for t in texts.values())
        and encoded.get("spin", {}).get("center_mode") == "absolute"
        and np.allclose(spin_c, 0.0, atol=1e-6)
        and abs(float(np.linalg.norm(scale_c)) - 1.0) < 1e-6,
        {"texts": texts, "spin_center": spin_c.tolist(),
         "scale_center": scale_c.tolist()},
    )
    for api_group in list(created):
        api_group.delete()
    created.clear()

    # ---- F: a partial SOLID pin builds through the strict field ----------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "PartialBlock"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=1)
    bpy.ops.object.mode_set(mode="OBJECT")
    top = [v.index for v in cube.data.vertices if v.co.z > 0.999]
    cube.vertex_groups.new(name="Top").add(top, 1.0, "REPLACE")
    block_group = dh.api.solver.create_group("Block", "SOLID")
    block_group.add(cube.name)
    block_group.create_pin(cube.name, "Top").pull(strength=PULL)
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild and the pin field both run in the server's frontend during
    # the build; a refusal from either fails it here.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="fallback_refusals:build", timeout=300.0)
    dh.log("built")
    solver_state = dh.facade.engine.state.solver.name
    session_dir = os.path.join(dh.facade.engine.state.remote_root, "session")
    cube_uuid = uuid_mod.get_object_uuid(cube)
    with open(os.path.join(session_dir, "info.toml"), "rb") as f:
        info = tomllib.load(f)
    blocks = []
    weights = []
    for k in range(int(info["count"]["pin_block"])):
        entry = info[f"pin-{k}"]
        if not str(entry.get("pin_group_id", "")).startswith(cube_uuid):
            continue
        path = os.path.join(session_dir, "bin", f"pin-pullw-{k}.bin")
        n = int(entry["pin"])
        w = np.fromfile(path, dtype="<f4") if os.path.isfile(path) else None
        blocks.append({"block": k, "pull": entry.get("pull"), "n": n,
                       "pullw": w is not None and w.size == n})
        if w is not None and w.size == n:
            weights.append(w.astype(np.float64))
    w_all = np.concatenate(weights) if weights else np.zeros(0)
    graded = bool(
        w_all.size
        and np.all(np.isfinite(w_all))
        and w_all.min() > 0.0
        and w_all.max() <= PULL * (1.0 + 1e-5)
        and w_all.min() < 0.5 * w_all.max()
    )
    dh.record(
        "F_partial_solid_builds_with_diffused_weights",
        solver_state in ("READY", "RESUMABLE") and bool(blocks)
        and all(b["pullw"] for b in blocks) and graded,
        {"solver_state": solver_state, "pinned_blender_vertices": len(top),
         "blocks": blocks, "n_weights": int(w_all.size),
         "weight_min": float(w_all.min()) if w_all.size else None,
         "weight_max": float(w_all.max()) if w_all.size else None,
         "session_dir": session_dir},
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
        .replace("<<LIFT>>", repr(_LIFT))
        .replace("<<PULL>>", repr(_PULL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
