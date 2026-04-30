# File: scenarios/bl_stitch_merge.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cloth-to-cloth stitching across two SHELL pieces, redesigned so the
# stitch is under active separating tension during the run.
#
# Layout (rest, world coordinates):
#
#     A.v3 (-1,1,0)             A.v1 (0,1,0)   B.v1 (0.5,1,0)             B.v3 (1.5,1,0)
#         +----------------------+   gap=0.5    +-------------------------+
#         |  MeshA               |              |  MeshB                  |
#         +----------------------+              +-------------------------+
#     A.v2 (-1,0,0)             A.v0 (0,0,0)   B.v0 (0.5,0,0)             B.v2 (1.5,0,0)
#                                ^----STITCH----^
#
# A spans x in [-1, 0] and B spans x in [0.5, 1.5]. The stitch ties
# A.v0 -> B.v0 and the rest gap is 0.5m, so the stitch has real work
# to do at frame 0. The pinned outer edges (A verts 2,3 at x=-1 and B
# verts 2,3 at x=1.5) are then driven outward in opposite X directions
# via MOVE_BY, applying separating tension across the stitch.
#
# Why this avoids the area-zero panic at src/scene.rs:1341. The
# explicit shell-shell merge in frontend/_scene_.py aliases A.v0 to
# B.v0, so A's two triangles end up referencing B.v0_pos in place of
# A.v0_pos. Computed rest areas after alias:
#   * (B.v0=(0.5,0,0), A.v2=(-1,0,0), A.v3=(-1,1,0))  -> area 0.75
#   * (B.v0=(0.5,0,0), A.v3=(-1,1,0), A.v1=(0,1,0))   -> area 0.50
# Both strictly positive, so the area > 0 assertion passes. The
# original coincident layout (A.v0 == B.v0 at rest) made the assertion
# vacuous: the stitch was a no-op weld at frame 0 and the
# "stays close" check never had to distinguish a working stitch from
# silence.
#
# Subtests:
#   A. ``stitch_encoded``: encode the params via encode_param, decode
#      the pickle, and assert the decoded ``explicit_merge_pairs``
#      section contains a single entry whose source_uuid/target_uuid
#      match the authored UUIDs and whose ``pairs`` row is [0, 0]. This
#      exercises ``_encode_explicit_merge_pairs`` (cross_stitch_json
#      JSON parse + barycentric argmax + UUID resolution) rather than
#      a PropertyGroup string round-trip.
#   B. ``stitched_pair_stays_close``: across every PC2 frame, the
#      world distance between A[0] and B[0] stays under TOL.
#      Tolerance derivation: shell-shell explicit merges create a hard
#      vertex alias in frontend/_scene_.py:3337 (A.v0 and B.v0 share a
#      single solver DOF), so the PC2 readback for A[0] and B[0] are
#      written from the same shared position; the deviation is bounded
#      by float32 PC2 quantization (~1e-6). If the alias were dropped
#      entirely (the failure mode we want to catch), the pin MOVE_BY
#      separates the outer edges by SEP_PER_SIDE * 2 = 0.4m, leaving
#      the inner verts to drift by hundreds of millimeters. TOL=0.01
#      passes comfortably under the working alias and fails by orders
#      of magnitude under a dropped alias.
#   C. ``simulation_completes``: solver leaves FAILED, both PC2 files
#      exist, sample counts match the requested frame range.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import json
import os
import pickle
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def _make_strip(name, inner_x, outer_x):
    # Build a 1m x 1m quad (two triangles) with a stable vertex order:
    #   v0 = (inner_x, 0, 0)   inner edge, stitched to partner
    #   v1 = (inner_x, 1, 0)   inner edge (top), unconstrained
    #   v2 = (outer_x, 0, 0)   outer edge, pinned
    #   v3 = (outer_x, 1, 0)   outer edge (top), pinned
    import bmesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    coords = [
        (inner_x, 0.0, 0.0),
        (inner_x, 1.0, 0.0),
        (outer_x, 0.0, 0.0),
        (outer_x, 1.0, 0.0),
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.verts.ensure_lookup_table()
    # Two triangles forming the quad. Triangles avoid n-gon rejection
    # on the encoder side.
    bm.faces.new((verts[0], verts[2], verts[3]))
    bm.faces.new((verts[0], verts[3], verts[1]))
    bm.to_mesh(mesh)
    bm.free()
    return obj


# Authored separation knob; documented in the header tolerance
# derivation. SEP_PER_SIDE total displacement on each pinned edge,
# applied over the timeline via MOVE_BY in opposite X directions.
SEP_PER_SIDE = 0.2
FRAME_COUNT = 6


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe the scene before adding our two strips.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # MeshA spans x in [-1, 0]; inner edge at x=0.
    # MeshB spans x in [0.5, 1.5]; inner edge at x=0.5.
    # Rest gap between A.v0 and B.v0 is 0.5m, so the stitch is loaded
    # at frame 0 and not just under simulated motion.
    mesh_a = _make_strip("MeshA", inner_x=0.0, outer_x=-1.0)
    mesh_b = _make_strip("MeshB", inner_x=0.5, outer_x=1.5)

    # Pin the outer edge of each piece (verts 2, 3). MOVE_BY on these
    # pins drives the separating load across the stitch.
    vg_a = mesh_a.vertex_groups.new(name="OuterPinA")
    vg_a.add([2, 3], 1.0, "REPLACE")
    vg_b = mesh_b.vertex_groups.new(name="OuterPinB")
    vg_b.add([2, 3], 1.0, "REPLACE")

    # Save the .blend so the encoder/PC2 writer have a stable file
    # location (some MESH_CACHE paths expect it).
    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "stitch_merge.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(project_name="stitch_merge",
                              frame_count=FRAME_COUNT)

    # Single SHELL group containing both meshes.
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(mesh_a.name)
    cloth.add(mesh_b.name)

    # Pin the outer edges and drive them apart in opposite X directions
    # over the timeline. This puts the stitch under active separating
    # tension; without the merge alias the inner verts would drift by
    # the full separation distance.
    pin_a = cloth.create_pin(mesh_a.name, "OuterPinA")
    pin_a.move_by(delta=(-SEP_PER_SIDE, 0.0, 0.0),
                  frame_start=1, frame_end=FRAME_COUNT - 1,
                  transition="LINEAR")
    pin_b = cloth.create_pin(mesh_b.name, "OuterPinB")
    pin_b.move_by(delta=(+SEP_PER_SIDE, 0.0, 0.0),
                  frame_start=1, frame_end=FRAME_COUNT - 1,
                  transition="LINEAR")

    # Resolve UUIDs (the encoder dies on empty source/target uuids).
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    uuid_a = uuid_mod.get_or_create_object_uuid(mesh_a)
    uuid_b = uuid_mod.get_or_create_object_uuid(mesh_b)
    if not uuid_a or not uuid_b:
        raise RuntimeError("could not allocate UUIDs for stitch meshes")

    # Author one stitch pair: A.v0 <-> B.v0. The schema follows the
    # snap operator's SHELL-target branch in mesh_ops/snap_ops.py:
    # ``ind`` rows are [src_vert, t0, t1, t2] over a target triangle,
    # ``w`` rows are [1.0, alpha, beta, gamma]. Using a real B
    # triangle (verts 0, 2, 3 from _make_strip) with weights
    # [1.0, 1.0, 0.0, 0.0] tells _encode_explicit_merge_pairs to pick
    # B.v0 (highest barycentric weight) as the merge partner.
    state = dh.groups.get_addon_data(bpy.context.scene).state
    pair = state.merge_pairs.add()
    pair.object_a = mesh_a.name
    pair.object_b = mesh_b.name
    pair.object_a_uuid = uuid_a
    pair.object_b_uuid = uuid_b
    pair.stitch_stiffness = 1000.0
    cs_payload = {
        "source_uuid": uuid_a,
        "target_uuid": uuid_b,
        "ind": [[0, 0, 2, 3]],
        "w": [[1.0, 1.0, 0.0, 0.0]],
        "target_points": [[0.5, 0.0, 0.0]],
        "a_vert_count": len(mesh_a.data.vertices),
        "b_vert_count": len(mesh_b.data.vertices),
    }
    pair.cross_stitch_json = json.dumps(cs_payload, separators=(",", ":"))
    state.merge_pairs_index = len(state.merge_pairs) - 1
    dh.log(f"authored_stitch pairs={len(state.merge_pairs)}")

    # ----- A: stitch encoded into param.pickle -------------------
    # Encode params and decode the pickle. Assert the decoded
    # explicit_merge_pairs section is present with the expected
    # source/target UUIDs and a [0, 0] index pair. This exercises
    # _encode_explicit_merge_pairs's JSON parse + barycentric argmax,
    # not just a PropertyGroup string round-trip.
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = pickle.loads(param_bytes)
    decoded_emp = decoded.get("explicit_merge_pairs", [])
    encoder_ok = (
        len(decoded_emp) == 1
        and decoded_emp[0].get("source_uuid") == uuid_a
        and decoded_emp[0].get("target_uuid") == uuid_b
        and list(decoded_emp[0].get("pairs", [])) == [[0, 0]]
    )
    dh.record(
        "A_stitch_encoded",
        encoder_ok,
        {
            "n_entries": len(decoded_emp),
            "decoded_entry": decoded_emp[0] if decoded_emp else None,
            "expected_source_uuid": uuid_a,
            "expected_target_uuid": uuid_b,
            "expected_pairs": [[0, 0]],
        },
    )

    # Build + run + fetch.
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    data_bytes, _ = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="stitch_merge:build",
    ))
    deadline = __import__('time').time() + 120.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        __import__('time').sleep(0.3)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=root.state.frame_count - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")

    # ----- C: simulation completes (solver did not fail) ----------
    final_state = dh.facade.engine.state
    pc2_a = dh.find_pc2_for(mesh_a)
    pc2_b = dh.find_pc2_for(mesh_b)
    have_a = bool(pc2_a) and os.path.isfile(pc2_a)
    have_b = bool(pc2_b) and os.path.isfile(pc2_b)
    samples_a = samples_b = 0
    arr_a = arr_b = None
    if have_a:
        arr_a = dh.read_pc2(pc2_a)
        samples_a = arr_a.shape[0]
    if have_b:
        arr_b = dh.read_pc2(pc2_b)
        samples_b = arr_b.shape[0]
    dh.record(
        "C_simulation_completes",
        final_state.solver.name in ("READY", "RESUMABLE")
        and have_a and have_b
        and samples_a >= 2 and samples_b >= 2
        and samples_a == samples_b,
        {
            "solver": final_state.solver.name,
            "pc2_a": pc2_a,
            "pc2_b": pc2_b,
            "samples_a": samples_a,
            "samples_b": samples_b,
        },
    )

    # ----- B: stitched pair stays close every simulated frame -----
    # The PC2 vertex order matches Blender's vertex order (the addon
    # remaps solver -> Blender on fetch). Vertex 0 of each PC2 is the
    # inner-edge vertex we stitched. With the hard alias in place the
    # two readbacks come from the same solver DOF and should match to
    # within float32 PC2 quantization (~1e-6).
    #
    # PC2 sample 0 is the Blender rest geometry (A.v0 at (0,0,0),
    # B.v0 at (0.5,0,0)) written before the solver applies the merge
    # alias, so we skip it and compare from sample 1 onward (the first
    # simulated frame). TOL=0.01 is chosen to fail by orders of
    # magnitude if the alias were dropped: the pin MOVE_BY drives the
    # outer edges by SEP_PER_SIDE in opposite X directions over
    # frames 1..FRAME_COUNT-1, and without the merge constraint the
    # inner verts would drift toward those outer edges, producing a
    # gap on the order of SEP_PER_SIDE * 2 = 0.4m plus the rest 0.5m
    # gap. 0.01 << 0.5 cleanly separates the two regimes.
    TOL = 0.01
    max_dist = 0.0
    per_frame = []
    n_compared = 0
    if arr_a is not None and arr_b is not None:
        n_frames = min(arr_a.shape[0], arr_b.shape[0])
        for f in range(1, n_frames):
            v_a = np.asarray(arr_a[f, 0], dtype=np.float64)
            v_b = np.asarray(arr_b[f, 0], dtype=np.float64)
            d = float(np.linalg.norm(v_a - v_b))
            per_frame.append({"frame": f, "dist": round(d, 6)})
            if d > max_dist:
                max_dist = d
            n_compared += 1
    dh.record(
        "B_stitched_pair_stays_close",
        n_compared >= 2 and max_dist < TOL,
        {
            "tolerance": TOL,
            "max_dist": round(max_dist, 6),
            "n_frames_compared": n_compared,
            "sep_per_side": SEP_PER_SIDE,
            "per_frame_tail": per_frame[-5:],
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
