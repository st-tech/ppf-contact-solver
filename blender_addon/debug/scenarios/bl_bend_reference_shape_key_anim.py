# File: scenarios/bl_bend_reference_shape_key_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The animated bending rest angle: a shell whose Reference Source is "Shape
# Key Mix (Animated)" takes its hinge rest angles from its own shape-key mix,
# and when the keys' Values are animated, the rest angle follows the mix
# through the simulation.
#
# The scene holds two identical flat grids in one Shell group, each with a
# flat Basis and a "Fold" key that creases the grid 90 degrees along its
# center line. Sheet A keys Fold's Value from 0 at frame 1 to 1 at frame 21
# (linear); sheet B leaves it at 0. Gravity and contact are off, so the only
# thing that can fold a sheet is its rest angle. The checks:
#
#   A. mix_matches_blender          - the add-on's mix reader agrees with
#                                     Blender's own evaluated mesh, with two
#                                     weighted keys, one of them limited by a
#                                     vertex group.
#   B. static_mix_ships_no_schedule - sheet B (mix not animated) ships its
#                                     current mix as the static reference and
#                                     no schedule.
#   C. animated_mix_ships_schedule  - sheet A ships a schedule: offsets from
#                                     0 through the end of the ramp and the
#                                     last frame, the held stretch dropped,
#                                     and the frame half way up the ramp at
#                                     the half-folded mix.
#   D. bend_plasticity_not_shipped  - Bend Plasticity left on in the group
#                                     (the panel hides it under From
#                                     Reference Geometry) reaches the solver
#                                     as zero.
#   E. build_and_run                - the solver builds and runs every frame
#                                     with the schedule in play.
#   F. rest_angle_follows_the_mix   - THE RESULT IS ADMITTED, not just
#                                     dispatched: sheet A ends folded, sheet
#                                     B (its static twin) stays flat, and A
#                                     is folded further at the end than early
#                                     in the ramp.
#   G. refuses_rod_group            - an animated mix in a Rod group refuses
#                                     the upload, naming Shell groups.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import math
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 60
N = 16


def make_sheet(name, x_offset):
    verts = [((i / N - 0.5), (j / N - 0.5), 0.0)
             for i in range(N + 1) for j in range(N + 1)]
    faces = [(i * (N + 1) + j, (i + 1) * (N + 1) + j,
              (i + 1) * (N + 1) + j + 1, i * (N + 1) + j + 1)
             for i in range(N) for j in range(N)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x_offset, 0.0, 0.0)
    obj.shape_key_add(name="Basis", from_mix=False)
    fold = obj.shape_key_add(name="Fold", from_mix=False)
    a = math.radians(45.0)
    for p in fold.data:
        x, y, _ = p.co
        p.co = (x * math.cos(a), y, abs(x) * math.sin(a))
    fold.value = 0.0
    return obj


def crease_deg(p):
    # Fold angle across the center grid line, from the two grid cells that
    # meet at it (the hinges whose rest angle the Fold key changes): 0 flat,
    # 90 a right angle. Cells further out bend only as the sheet follows.
    g = p.reshape(N + 1, N + 1, 3)
    c = g[N // 2]
    a = g[N // 2 - 1] - c
    b = g[N // 2 + 1] - c
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def key_co(obj, name):
    kb = obj.data.shape_keys.key_blocks[name]
    co = np.empty(len(kb.data) * 3)
    kb.data.foreach_get("co", co)
    return co.reshape(-1, 3)


def find_entry(decoded, uuid):
    for grp in decoded:
        for o in grp.get("object", []):
            if o.get("uuid") == uuid:
                return o
    return None


def raw_group(dh, name):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        if g.name == name:
            return g
    return None


try:
    dh = DriverHelpers(pkg, result)
    uuidreg = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_or_create_object_uuid"])
    utils_pkg = __import__(pkg + ".core.utils",
                           fromlist=["shape_key_mix_local_positions"])
    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT
    scene.frame_set(1)

    # --- A: the mix reader against Blender's own evaluation ---
    probe = make_sheet("MixProbe", 4.0)
    k1 = probe.shape_key_add(name="K1", from_mix=False)
    for p in k1.data:
        p.co.z += 0.2 * p.co.x
    k1.value = 0.3
    half = probe.vertex_groups.new(name="Half")
    half.add([v.index for v in probe.data.vertices if v.co.y > 0.0], 0.5, "REPLACE")
    probe.data.shape_keys.key_blocks["Fold"].value = 0.7
    probe.data.shape_keys.key_blocks["Fold"].vertex_group = "Half"
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = probe.evaluated_get(dg).to_mesh()
    ev_co = np.array([v.co[:] for v in ev.vertices])
    probe.evaluated_get(dg).to_mesh_clear()
    mine = utils_pkg.shape_key_mix_local_positions(probe)
    err = float(np.abs(mine - ev_co).max())
    dh.record("A_mix_matches_blender", err < 1e-5, {"max_err": err})
    bpy.data.objects.remove(probe, do_unlink=True)

    sheet_a = make_sheet("FoldAnimated", -1.0)
    sheet_b = make_sheet("FoldStatic", 1.0)
    fold_a = sheet_a.data.shape_keys.key_blocks["Fold"]
    fold_a.value = 0.0
    fold_a.keyframe_insert("value", frame=1)
    fold_a.value = 1.0
    fold_a.keyframe_insert("value", frame=21)
    # Linear, so the Value is exactly 0.5 at frame 11 (offset 10). Blender 5
    # actions keep their curves in layered channelbags, which
    # get_id_fcurves reads.
    utils_any = __import__(pkg + ".core.utils", fromlist=["get_id_fcurves"])
    for fc in utils_any.get_id_fcurves(sheet_a.data.shape_keys):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    scene.frame_set(1)

    a_uuid = uuidreg.get_or_create_object_uuid(sheet_a)
    b_uuid = uuidreg.get_or_create_object_uuid(sheet_b)
    dh.save_blend(PROBE_DIR, "bend_reference_shape_key_anim.blend")
    root = dh.configure_state(
        project_name="bend_reference_shape_key_anim",
        frame_count=FRAME_COUNT,
    )
    cloth = dh.api.solver.create_group("Sheets", "SHELL")
    cloth.add(sheet_a.name)
    cloth.add(sheet_b.name)
    grp = raw_group(dh, "Sheets")
    grp.bend_rest_angle_source = "FROM_REFERENCE"
    for a in grp.assigned_objects:
        a.bend_ref_source = "SHAPE_KEY_MIX"

    data_bytes, _p, _d, _h = encoder_pkg.prepare_upload(bpy.context)
    decoded = DriverHelpers.decode_addon_blob(data_bytes)
    ea = find_entry(decoded, a_uuid)
    eb = find_entry(decoded, b_uuid)
    basis = key_co(sheet_b, "Basis")
    fold = key_co(sheet_a, "Fold")

    # --- B: an unanimated mix is a static reference ---
    brv_b = np.asarray(eb.get("bend_rest_vert"), dtype=np.float64)
    dh.record(
        "B_static_mix_ships_no_schedule",
        eb.get("bend_rest_vert_anim") is None
        and brv_b.shape == basis.shape
        and np.allclose(brv_b, basis, atol=1e-5),
        {"has_anim": eb.get("bend_rest_vert_anim") is not None},
    )

    # --- C: an animated mix ships a schedule ---
    anim = ea.get("bend_rest_vert_anim")
    ok_c = anim is not None
    detail = {}
    if ok_c:
        offs = [float(o) for o in anim["frame_offset"]]
        verts = [np.asarray(v, dtype=np.float64) for v in anim["vert"]]
        detail = {"n": len(offs), "first": offs[0], "last": offs[-1]}
        mid = verts[offs.index(10.0)] if 10.0 in offs else None
        ok_c = (
            offs[0] == 0.0 and offs[-1] == float(FRAME_COUNT - 1)
            and 20.0 in offs and 30.0 not in offs
            and np.allclose(verts[0], basis, atol=1e-5)
            and np.allclose(verts[-1], fold, atol=1e-5)
            and mid is not None
            and np.allclose(mid, basis + 0.5 * (fold - basis), atol=1e-4)
        )
    dh.record("C_animated_mix_ships_schedule", ok_c, detail)

    # --- D: Bend Plasticity a From Reference Geometry group still holds
    # (the panel hides it) reaches the solver as zero ---
    grp.enable_bend_plasticity = True
    grp.bend_plasticity = 0.5
    _d3, param_d, _d4, _h4 = encoder_pkg.prepare_upload(bpy.context)
    grp.enable_bend_plasticity = False

    def collect(node, key, out):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    out.append(v)
                collect(v, key, out)
        elif isinstance(node, (list, tuple)):
            for v in node:
                collect(v, key, out)
        return out

    shipped = collect(DriverHelpers.decode_addon_blob(param_d), "bend-plasticity", [])
    dh.record("D_bend_plasticity_not_shipped",
              len(shipped) > 0 and all(float(v) == 0.0 for v in shipped),
              {"shipped": [float(v) for v in shipped][:8]})

    # --- E/F: run it, and read what the solver did ---
    data_bytes, param_bytes, _d2, _h2 = encoder_pkg.prepare_upload(bpy.context)
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "bend-anim:build", timeout=180.0)
    saw_running = dh.run_and_wait(timeout=300.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    frame = dh.facade.engine.state.frame
    dh.record("E_build_and_run",
              bool(saw_running) and frame >= FRAME_COUNT - 1,
              {"frame": int(frame), "solver": dh.facade.engine.state.solver.name})

    applied, total = dh.fetch_and_drain()
    pa = dh.read_pc2(dh.find_pc2_for(sheet_a))
    pb = dh.read_pc2(dh.find_pc2_for(sheet_b))
    early = crease_deg(pa[5])
    final_a = crease_deg(pa[-1])
    final_b = crease_deg(pb[-1])
    dh.record(
        "F_rest_angle_follows_the_mix",
        final_a > 60.0 and abs(final_b) < 5.0 and final_a > early + 20.0,
        {"early_a": early, "final_a": final_a, "final_b": final_b,
         "frames": int(pa.shape[0]), "applied": applied, "total": total},
    )

    # --- G: an animated mix in a Rod group is refused ---
    rod_mesh = bpy.data.meshes.new("RodLine")
    rod_mesh.from_pydata([(0, 3, k * 0.1) for k in range(6)],
                         [(k, k + 1) for k in range(5)], [])
    rod = bpy.data.objects.new("RodLine", rod_mesh)
    scene.collection.objects.link(rod)
    rod.shape_key_add(name="Basis", from_mix=False)
    bentk = rod.shape_key_add(name="Bent", from_mix=False)
    for p in bentk.data:
        p.co.x += 0.1 * p.co.z * p.co.z
    bentk.value = 0.0
    bentk.keyframe_insert("value", frame=1)
    bentk.value = 1.0
    bentk.keyframe_insert("value", frame=10)
    scene.frame_set(1)
    rods = dh.api.solver.create_group("Rods", "ROD")
    rods.add(rod.name)
    rgrp = raw_group(dh, "Rods")
    rgrp.bend_rest_angle_source = "FROM_REFERENCE"
    for a in rgrp.assigned_objects:
        a.bend_ref_source = "SHAPE_KEY_MIX"
    msg_g = ""
    try:
        encoder_pkg.prepare_upload(bpy.context)
    except ValueError as exc:
        msg_g = str(exc)
    dh.record("G_refuses_rod_group",
              "Shell groups only" in msg_g and rod.name in msg_g,
              {"msg": msg_g[:300]})

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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
