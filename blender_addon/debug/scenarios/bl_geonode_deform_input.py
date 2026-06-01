# File: scenarios/bl_geonode_deform_input.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Geometry-Nodes deform as solver input. A SHELL plane carries a
# Set-Position wave (z = A*sin(k*(t+y))) driven by Scene Time, with one
# edge in a pin vertex group whose per-frame motion is captured. This
# exercises two encoder/client guarantees that only show up when a
# deform-only Geometry Nodes modifier feeds the solver:
#
#   A. transfer_reads_frame_one_geonode_free_verts_no_capture:
#         with NO capture cache and a pin that has no ops, the encoded
#         ``vert`` (the starting mesh sent at Transfer) for FREE verts
#         is the geometry-nodes frame-1 shape (z ~= A*sin(k*(t1+y))),
#         not the flat rest cage. Capture Deformation is not required.
#   B. transfer_reads_frame_one_geonode_pinned_verts_no_capture:
#         same, for PINNED verts: the first-frame starting pose includes
#         pin positions evaluated from geometry nodes at Transfer time,
#         independent of Capture Deformation (which is animation-only).
#   C. first_frame_not_flat:
#         PC2 frame 0 (scene frame 1) shows the deformed interior, not
#         the flat rest mesh the gap-fill used to write.
#   D. pinned_edge_not_double_counted:
#         across the run the emulated kinematic pin lands at the
#         captured wave amplitude, not ~2x it.
#
# The emulated solver moves only kinematic pins (free verts stay at the
# encoded initial), so every number here is deterministic on macOS
# without a CUDA runtime.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 12
_FRAME_RATE = 24
_AMP = 0.25
_FREQ = 3.0
_TOL = 1e-3


_DRIVER_BODY = r"""
import math
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
FRAME_RATE = <<FRAME_RATE>>
AMP = <<AMP>>
FREQ = <<FREQ>>
TOL = <<TOL>>


def _build_wave_modifier(obj, amp, freq):
    # z = amp * sin(freq * (SceneTime.seconds + y)) applied to every
    # vertex via Set Position offset. Mirrors the artist-side setup.
    ng = bpy.data.node_groups.new("WaveGN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links

    def n(t):
        return nodes.new(t)

    gin = n("NodeGroupInput")
    gout = n("NodeGroupOutput")
    pos = n("GeometryNodeInputPosition")
    sep = n("ShaderNodeSeparateXYZ")
    links.new(pos.outputs["Position"], sep.inputs["Vector"])
    tnode = n("GeometryNodeInputSceneTime")
    tpy = n("ShaderNodeMath"); tpy.operation = "ADD"
    links.new(tnode.outputs["Seconds"], tpy.inputs[0])
    links.new(sep.outputs["Y"], tpy.inputs[1])
    phase = n("ShaderNodeMath"); phase.operation = "MULTIPLY"
    links.new(tpy.outputs[0], phase.inputs[0]); phase.inputs[1].default_value = freq
    sine = n("ShaderNodeMath"); sine.operation = "SINE"
    links.new(phase.outputs[0], sine.inputs[0])
    zoff = n("ShaderNodeMath"); zoff.operation = "MULTIPLY"
    links.new(sine.outputs[0], zoff.inputs[0]); zoff.inputs[1].default_value = amp
    comb = n("ShaderNodeCombineXYZ")
    links.new(zoff.outputs[0], comb.inputs["Z"])
    setpos = n("GeometryNodeSetPosition")
    links.new(gin.outputs["Geometry"], setpos.inputs["Geometry"])
    links.new(comb.outputs["Vector"], setpos.inputs["Offset"])
    links.new(setpos.outputs["Geometry"], gout.inputs["Geometry"])
    mod = obj.modifiers.new("WaveGN", "NODES")
    mod.node_group = ng
    return mod


def _sample_pin_world_solver(obj, pin_indices, frame_start, frame_end):
    # Depsgraph-evaluate obj per frame and return (n_frames, n_pin, 3)
    # in solver world space (zup_to_yup @ matrix_world @ co_local).
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    z2y = np.array(transform_mod.zup_to_yup(), dtype=np.float64).reshape(4, 4)
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    pin_arr = np.asarray(pin_indices, dtype=np.int64)
    out = np.empty((n_frames, len(pin_indices), 3), dtype=np.float32)
    try:
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            eo = obj.evaluated_get(dg)
            em = eo.to_mesh()
            try:
                co = np.empty((len(em.vertices), 3), dtype=np.float64)
                em.vertices.foreach_get("co", co.ravel())
                sub = co[pin_arr]
                mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
                m = z2y @ mw
                homog = np.concatenate([sub, np.ones((len(pin_indices), 1))], axis=1)
                out[i] = (homog @ m.T)[:, :3].astype(np.float32, copy=False)
            finally:
                eo.to_mesh_clear()
    finally:
        scene.frame_set(saved)
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # 11x11 plane, local x,y in [-1, 1].
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "WavePlane"
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=9)
    bpy.ops.object.mode_set(mode="OBJECT")

    xs = [v.co.x for v in plane.data.vertices]
    xmin = min(xs)
    edge_idx = sorted(v.index for v in plane.data.vertices if abs(v.co.x - xmin) < 1e-5)
    interior_idx = sorted(v.index for v in plane.data.vertices if abs(v.co.x - xmin) >= 1e-5)
    vg = plane.vertex_groups.new(name="pin")
    vg.add(edge_idx, 1.0, "REPLACE")

    _build_wave_modifier(plane, AMP, FREQ)

    dh.save_blend(PROBE_DIR, "geonode_deform_input.blend")
    root = dh.configure_state(
        project_name="geonode_deform_input",
        frame_count=FRAME_COUNT,
        frame_rate=FRAME_RATE,
        step_size=1.0 / FRAME_RATE,
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    shell = dh.api.solver.create_group("Cloth", "SHELL")
    shell.add(plane.name)
    shell.create_pin(plane.name, "pin")
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    pin_item = group.pin_vertex_groups[0]

    pc2 = __import__(pkg + ".core.pc2", fromlist=["write_pin_anim_pc2", "get_pin_anim_cache"])
    pin_ops = __import__(pkg + ".ui.dynamics.pin_ops", fromlist=["_ensure_embedded_move_op"])

    # ---- expected geometry-nodes wave (local z) per scene frame ----
    def gn_local_z(frame):
        t = frame / FRAME_RATE
        return {v.index: AMP * math.sin(FREQ * (t + v.co.y))
                for v in plane.data.vertices}

    enc = __import__(pkg + ".core.encoder.mesh", fromlist=["_build_obj_data"])

    def encoded_plane_vert():
        bpy.context.scene.frame_set(1)
        data = enc._build_obj_data(bpy.context, persist_topology_hash=False)
        for g in data:
            for o in g["object"]:
                if o.get("name") == plane.name:
                    return np.asarray(o["vert"])
        return None

    # ---- A & B: Transfer reads the frame-1 geometry-nodes mesh -------
    # No capture cache exists yet, and the pin carries no ops. The
    # encoded vert (the first-frame starting mesh sent at Transfer) must
    # already be the geometry-nodes frame-1 shape for FREE *and* PINNED
    # vertices. Capture Deformation is for per-frame animation only and
    # must not be required to get the correct starting pose.
    assert not pc2.has_pin_anim_pc2(plane, "pin"), "no cache expected pre-capture"
    z1 = gn_local_z(1)
    vert = encoded_plane_vert()
    free_err = max(abs(float(vert[i, 2]) - z1[i]) for i in interior_idx)
    free_absmax = max(abs(float(vert[i, 2])) for i in interior_idx)
    pinned_err = max(abs(float(vert[i, 2]) - z1[i]) for i in edge_idx)
    pinned_absmax = max(abs(float(vert[i, 2])) for i in edge_idx)
    dh.record(
        "A_transfer_reads_frame_one_geonode_free_verts_no_capture",
        free_err < TOL and free_absmax > 0.05,
        {"free_z_absmax": round(free_absmax, 5),
         "free_max_err_vs_gn": round(free_err, 6),
         "gn_amp": AMP, "tol": TOL},
    )
    dh.record(
        "B_transfer_reads_frame_one_geonode_pinned_verts_no_capture",
        pinned_err < TOL and pinned_absmax > 0.05,
        {"pinned_z_absmax": round(pinned_absmax, 5),
         "pinned_max_err_vs_gn": round(pinned_err, 6), "tol": TOL},
    )

    # ---- Capture the pin edge's per-frame motion (animation only) ----
    # Headless: call the helpers the modal operator would call per tick.
    # This is the opt-in animation path; the starting pose above did not
    # depend on it. The captured frame-0 row equals the frame-1 pose, so
    # initial == cache[0] and the decoder's delta integration telescopes
    # to the exact wave (verified in subtest D).
    captured = _sample_pin_world_solver(plane, edge_idx, 1, FRAME_COUNT)
    pc2.write_pin_anim_pc2(plane, "pin", captured)
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)

    # ---- run the emulated solve --------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="geonode:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(plane)
    arr = dh.read_pc2(pc2_path) if pc2_path else None  # (n_samples, n_verts, 3) local
    n_samples = int(arr.shape[0]) if arr is not None else 0

    # ---- C: first frame not flat -------------------------------------
    first_interior_absmax = (
        max(abs(float(arr[0, i, 2])) for i in interior_idx)
        if n_samples > 0 else 0.0
    )
    dh.record(
        "C_first_frame_not_flat",
        first_interior_absmax > 0.05,
        {"pc2_frame0_interior_z_absmax": round(first_interior_absmax, 5),
         "n_samples": n_samples},
    )

    # ---- D: pinned edge tracks the captured wave (single) ------------
    # The emulated kinematic pin lands at the decoder's per-frame
    # target. With initial == cache[0] the delta track telescopes to
    # the exact geometry-nodes wave, so the edge amplitude matches the
    # GN wave every frame. A double count would show ~2x; the rejected
    # "pinned at rest" variant showed a ramp (~0.04 at frame 2). PC2 is
    # local space, so compare the edge's local-z amplitude directly.
    worst_amp_err = 0.0
    worst_ratio = 0.0
    per_frame = []
    for s in range(n_samples):
        frame = s + 1
        gz = gn_local_z(frame)
        gn_edge_amp = max(abs(gz[i]) for i in edge_idx)
        pc2_edge_amp = max(abs(float(arr[s, i, 2])) for i in edge_idx)
        per_frame.append({"f": frame,
                          "gn": round(gn_edge_amp, 4),
                          "pc2": round(pc2_edge_amp, 4)})
        worst_amp_err = max(worst_amp_err, abs(pc2_edge_amp - gn_edge_amp))
        if gn_edge_amp > 1e-4:
            worst_ratio = max(worst_ratio, pc2_edge_amp / gn_edge_amp)
    dh.record(
        "D_pinned_edge_tracks_wave_single_count",
        n_samples >= FRAME_COUNT - 1 and worst_amp_err < 0.02,
        {"worst_edge_amp_err_vs_gn": round(worst_amp_err, 5),
         "worst_pc2_to_gn_ratio": round(worst_ratio, 3),
         "n_samples": n_samples, "per_frame": per_frame},
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
        .replace("<<FRAME_RATE>>", str(_FRAME_RATE))
        .replace("<<AMP>>", repr(_AMP))
        .replace("<<FREQ>>", repr(_FREQ))
        .replace("<<TOL>>", repr(_TOL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
