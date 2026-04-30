# File: scenarios/bl_shared_object_data.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared geometry deduplication: multiple objects sharing the same
# local-space mesh content must share a single canonical entry on the
# wire (``mesh_ref`` references), the remote must accept that
# encoding, and the per-instance simulation results fetched back must
# remain distinct PC2 streams keyed per object UUID.
#
# Three base shapes (icosphere, cube, subdivided plane) are duplicated
# twice each. Each duplicate is a deep copy (``obj.copy()`` plus
# ``data.copy()``) so the mesh data block is distinct but the local
# vertex/face content is byte-identical to the canonical. Per-instance
# scale + rotation is applied via the object transform (NOT baked into
# the mesh) so the local hash still matches.
#
# Subtests:
#   A. encoder_dedup_canonical_count
#         encode_obj output for the SHELL group has exactly 3
#         canonical entries (carry ``vert``) and 6 ``mesh_ref``
#         entries that point at one of those canonical UUIDs.
#   B. uploaded_pickle_preserves_dedup
#         The data.pickle the addon uploaded to the worker's project
#         root carries the same 3+6 split. Confirms the wire format
#         (not just the in-memory encode) honors the dedup.
#   C. each_instance_has_mesh_cache
#         All 9 objects (3 canonical + 6 duplicates) carry a
#         ContactSolverCache MESH_CACHE PC2 modifier after fetch.
#   D. each_instance_has_distinct_pc2
#         All 9 PC2 paths are distinct and exist on disk. A regression
#         that collapsed the per-instance state into one stream would
#         either share a path or drop entries.
#   E. pc2_vertex_counts_match_canonical
#         For each duplicate, the PC2 vertex count equals its
#         canonical's mesh.vertices count. Proves the simulator wrote
#         the right amount of per-instance data even though the wire
#         payload only carried the local mesh once.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import pickle
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def _project_root_for(probe_dir, name):
    workspace = os.path.dirname(probe_dir)
    return os.path.join(workspace, "project", "git-debug", name)


def _make_canonical(name, kind, location):
    if kind == "ICO":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5,
                                              location=location)
    elif kind == "BOX":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    elif kind == "SHEET":
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    else:
        raise ValueError(f"unknown kind {kind}")
    obj = bpy.context.active_object
    obj.name = name
    if kind == "SHEET":
        # Subdivide so the sheet has interior topology, not a single
        # quad. 4 cuts -> 5x5 grid. Done with bmesh because the
        # subdivide operator depends on edit-mode context which is
        # awkward in headless mode.
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4,
                                  use_grid_fill=True)
        bm.to_mesh(obj.data)
        bm.free()
    return obj


def _deep_duplicate(src, name, location, scale, rotation_euler):
    # obj.copy() shares the mesh data block (Alt-D semantics, which the
    # encoder rejects). data.copy() materializes a separate data block
    # with byte-identical local content -- exactly the case the dedup
    # path is supposed to compress. Scale + rotation go on the object
    # transform; we do NOT bake them into the mesh, otherwise local
    # hashes would diverge and the dedup would not fire.
    dup = src.copy()
    dup.data = src.data.copy()
    dup.name = name
    bpy.context.collection.objects.link(dup)
    dup.location = location
    dup.scale = scale
    dup.rotation_euler = rotation_euler
    return dup


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    ico = _make_canonical("ShareIco", "ICO", (0.0, 0.0, 0.0))
    box = _make_canonical("ShareBox", "BOX", (3.0, 0.0, 0.0))
    sheet = _make_canonical("ShareSheet", "SHEET", (-3.0, 0.0, 0.0))

    # AllPin vertex group on the canonical icosphere drives the one
    # active kinematic constraint. We need at least one moving pin so
    # the solver actually advances frames; the rest of the objects
    # sit at rest with gravity=0 and contact disabled.
    n_ico = len(ico.data.vertices)
    vg = ico.vertex_groups.new(name="AllPin")
    vg.add(list(range(n_ico)), 1.0, "REPLACE")

    # Two duplicates per shape, each with a different transform. The
    # local mesh content is byte-identical to the canonical, so the
    # encoder's _local_mesh_hash collapses each pair onto its
    # canonical's UUID via mesh_ref.
    duplicates = [
        _deep_duplicate(ico, "ShareIco_dup1",
                        (0.0, 2.0, 0.0), (1.5, 1.5, 1.5), (0.3, 0.0, 0.0)),
        _deep_duplicate(ico, "ShareIco_dup2",
                        (0.0, -2.0, 0.0), (0.7, 0.7, 0.7), (0.0, 0.5, 0.0)),
        _deep_duplicate(box, "ShareBox_dup1",
                        (3.0, 2.0, 0.0), (1.2, 0.8, 1.0), (0.0, 0.0, 0.4)),
        _deep_duplicate(box, "ShareBox_dup2",
                        (3.0, -2.0, 0.0), (0.6, 1.4, 1.0), (0.5, 0.5, 0.0)),
        _deep_duplicate(sheet, "ShareSheet_dup1",
                        (-3.0, 2.0, 0.0), (2.0, 1.0, 1.0), (0.0, 0.0, 0.6)),
        _deep_duplicate(sheet, "ShareSheet_dup2",
                        (-3.0, -2.0, 0.0), (1.0, 2.0, 1.0), (0.4, 0.0, 0.0)),
    ]
    canonicals = [ico, box, sheet]
    all_objects = canonicals + duplicates
    canonical_uuids_by_name = {}

    blend_basename = "shared_object_data.blend"
    dh.save_blend(PROBE_DIR, blend_basename)

    project_name = "shared_object_data"
    root = dh.configure_state(project_name=project_name, frame_count=4)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    for obj in all_objects:
        cloth.add(obj.name)

    pin = cloth.create_pin(ico.name, "AllPin")
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=3,
                transition="LINEAR")

    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["get_or_create_object_uuid"])

    for obj in canonicals:
        canonical_uuids_by_name[obj.name] = uuid_registry.get_or_create_object_uuid(obj)

    # ----- A: encoder collapses duplicates onto canonicals -------
    encoded_blob = encoder_mesh.encode_obj(bpy.context)
    encoded_data = pickle.loads(encoded_blob)
    canonical_entries = []
    ref_entries = []
    for group in encoded_data:
        for obj_info in group.get("object", []):
            if "mesh_ref" in obj_info:
                ref_entries.append(obj_info)
            elif "vert" in obj_info:
                canonical_entries.append(obj_info)
    canonical_uuids = {e["uuid"] for e in canonical_entries}
    refs_resolve = all(e.get("mesh_ref") in canonical_uuids
                       for e in ref_entries)
    dh.record(
        "A_encoder_dedup_canonical_count",
        (len(canonical_entries) == 3
            and len(ref_entries) == 6
            and refs_resolve),
        {
            "canonical_count": len(canonical_entries),
            "ref_count": len(ref_entries),
            "canonical_names": sorted(e["name"] for e in canonical_entries),
            "ref_names": sorted(e["name"] for e in ref_entries),
            "refs_resolve_to_canonical": refs_resolve,
        },
    )

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="shared_object_data:build",
    ))
    deadline = __import__('time').time() + 90.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        __import__('time').sleep(0.3)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # ----- B: uploaded data.pickle preserves the dedup -----------
    proj_root = _project_root_for(PROBE_DIR, project_name)
    data_pickle_path = os.path.join(proj_root, "data.pickle")
    remote_canonical = []
    remote_refs = []
    if os.path.isfile(data_pickle_path):
        with open(data_pickle_path, "rb") as f:
            remote_data = pickle.load(f)
        for group in remote_data:
            for obj_info in group.get("object", []):
                if "mesh_ref" in obj_info:
                    remote_refs.append(obj_info)
                elif "vert" in obj_info:
                    remote_canonical.append(obj_info)
    remote_canonical_uuids = {e["uuid"] for e in remote_canonical}
    remote_refs_resolve = all(e.get("mesh_ref") in remote_canonical_uuids
                              for e in remote_refs)
    dh.record(
        "B_uploaded_pickle_preserves_dedup",
        (len(remote_canonical) == 3
            and len(remote_refs) == 6
            and remote_refs_resolve),
        {
            "data_pickle": data_pickle_path,
            "exists": os.path.isfile(data_pickle_path),
            "canonical_count": len(remote_canonical),
            "ref_count": len(remote_refs),
            "canonical_names": sorted(e["name"] for e in remote_canonical),
            "ref_names": sorted(e["name"] for e in remote_refs),
            "refs_resolve_to_canonical": remote_refs_resolve,
        },
    )

    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")

    # ----- C: every instance has a MESH_CACHE modifier -----------
    has_cache = {obj.name: dh.has_mesh_cache(obj) for obj in all_objects}
    dh.record(
        "C_each_instance_has_mesh_cache",
        all(has_cache.values()),
        has_cache,
    )

    # ----- D: every instance has its own PC2 file ----------------
    pc2_paths = {}
    pc2_exists = {}
    for obj in all_objects:
        path = dh.find_pc2_for(obj)
        real = os.path.realpath(path) if path else ""
        pc2_paths[obj.name] = real
        pc2_exists[obj.name] = bool(real) and os.path.isfile(real)
    distinct = (
        len({p for p in pc2_paths.values() if p}) == len(all_objects)
    )
    all_exist = all(pc2_exists.values())
    dh.record(
        "D_each_instance_has_distinct_pc2",
        distinct and all_exist,
        {
            "paths": pc2_paths,
            "exist": pc2_exists,
            "distinct": distinct,
        },
    )

    # ----- E: per-instance PC2 vertex counts match canonical -----
    canonical_vert_counts = {
        ico.name: len(ico.data.vertices),
        box.name: len(box.data.vertices),
        sheet.name: len(sheet.data.vertices),
    }
    instance_to_canonical = {
        ico.name: ico.name,
        box.name: box.name,
        sheet.name: sheet.name,
        "ShareIco_dup1": ico.name,
        "ShareIco_dup2": ico.name,
        "ShareBox_dup1": box.name,
        "ShareBox_dup2": box.name,
        "ShareSheet_dup1": sheet.name,
        "ShareSheet_dup2": sheet.name,
    }
    vert_count_mismatch = {}
    sample_frames = {}
    for obj in all_objects:
        path = pc2_paths.get(obj.name) or ""
        if not path or not os.path.isfile(path):
            vert_count_mismatch[obj.name] = "missing_pc2"
            continue
        traj = dh.read_pc2(path)
        expected = canonical_vert_counts[instance_to_canonical[obj.name]]
        actual = traj.shape[1] if traj.ndim == 3 else -1
        n_frames = traj.shape[0] if traj.ndim == 3 else 0
        sample_frames[obj.name] = n_frames
        finite = bool(np.isfinite(traj).all())
        if actual != expected or n_frames < 1 or not finite:
            vert_count_mismatch[obj.name] = {
                "expected_verts": expected,
                "actual_verts": actual,
                "frames": n_frames,
                "all_finite": finite,
            }
    dh.record(
        "E_pc2_vertex_counts_match_canonical",
        not vert_count_mismatch,
        {
            "frames": sample_frames,
            "mismatch": vert_count_mismatch,
            "canonical_vert_counts": canonical_vert_counts,
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
