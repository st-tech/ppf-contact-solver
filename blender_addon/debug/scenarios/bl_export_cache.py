# File: scenarios/bl_export_cache.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Export the simulated mesh sequence to a USD / Alembic cache: coverage for the
# two Export buttons in the Solver panel (solver.export_usd /
# solver.export_alembic). These are a lighter alternative to the shape-key
# bake: instead of freezing every fetched PC2 frame into a shape key, they
# drive Blender's built-in exporters over the simulated frame range and let
# them sample the ContactSolverCache MESH_CACHE-deformed mesh at each frame.
#
# The rig stands up a real emulated run of a pinned, moving plane (so the mesh
# genuinely moves frame to frame), fetches the animation into a local PC2, then
# exercises the operators through bpy.ops with EXEC_DEFAULT (explicit filepath,
# so no file browser opens headless).
#
# Content is verified by READING THE EXPORTED FILES BACK and comparing them to
# the live source simulation:
#   * USD is opened with Blender's bundled OpenUSD Python (``pxr``): the point
#     count, up-axis, time-code range and per-frame point positions are read
#     straight out of the .usdc and matched against the source mesh.
#   * Alembic is re-imported with ``wm.alembic_import`` and the imported mesh's
#     vertex count + per-frame motion are matched against the source (Alembic
#     has no importable Python reader, so a round-trip is the reliable check).
#
# Subtests:
#   A. ops_registered
#         Both solver.export_usd and solver.export_alembic exist.
#   B. poll_true_after_fetch
#         With a fetched PC2 cache present, both operators' poll() is True and
#         the plane is in the exportable-mesh set.
#   C. usdc_content_roundtrips
#         solver.export_usd -> <>.usdc returns FINISHED; the .usdc reopened via
#         pxr is Z-up, spans time codes [1, n], carries one mesh whose point
#         count equals the source, holds 2..n distinct point time samples, is
#         time-varying, and its per-frame points match the source mesh's
#         evaluated vertices; the scene frame range / current frame / selection
#         are restored afterward.
#   D. abc_content_roundtrips
#         solver.export_alembic -> <>.abc returns FINISHED; re-importing the
#         .abc yields a mesh with the source vertex count that animates, and
#         whose start->end motion magnitude matches the source sim.
#   E. refuses_when_unfetched
#         With the fetched-frame record cleared (remote has more frames than
#         local), solver.export_usd('INVOKE_DEFAULT') aborts with the
#         "Fetch all animation frames first" message and writes no file.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

_FRAME_COUNT = 6


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>


def _v3d_override():
    # bpy.ops file exporters / importers (and bpy.ops.X.poll() queries) need a
    # real area context; the --python bootstrap runs area-less, so scope the
    # operator calls to the window's VIEW_3D area.
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                for rgn in a.regions:
                    if rgn.type == "WINDOW":
                        return {"window": w, "area": a, "region": rgn}
    return {}


def _eval_local_verts(obj, frame):
    # Local-space evaluated vertex positions of *obj* at scene *frame* (post
    # MESH_CACHE / MeshSequenceCache deform). This is exactly what the exporter
    # samples, so it is the ground truth for the round-trip comparison.
    scene = bpy.context.scene
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    n = len(ev.data.vertices)
    flat = [0.0] * (n * 3)
    ev.data.vertices.foreach_get("co", flat)
    return np.asarray(flat, dtype=np.float64).reshape(n, 3)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    pc2_mod = __import__(pkg + ".core.pc2",
                        fromlist=["read_pc2_frame_count", "object_pc2_key",
                                  "get_pc2_path", "MODIFIER_NAME"])
    bake = __import__(pkg + ".ui.dynamics.bake_ops",
                     fromlist=["_has_unfetched_frames"])
    export_ops = __import__(pkg + ".ui.dynamics.export_ops",
                           fromlist=["_exportable_meshes",
                                     "SOLVER_OT_ExportUSD",
                                     "SOLVER_OT_ExportAlembic"])

    out_dir = os.path.dirname(PROBE_DIR)
    usdc_path = os.path.join(out_dir, "export_verify.usdc")
    abc_path = os.path.join(out_dir, "export_verify.abc")
    for p in (usdc_path, abc_path):
        if os.path.exists(p):
            os.remove(p)

    # -- pinned, moving plane so the mesh moves frame to frame --
    plane = dh.reset_scene_to_pinned_plane(name="ExportCloth")
    dh.save_blend(PROBE_DIR, "export_cache.blend")
    root = dh.configure_state(project_name="export_cache",
                              frame_count=FRAME_COUNT, frame_rate=24,
                              step_size=1.0 / 24)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.15, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="export:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    bpy.context.view_layer.update()
    dh.log("fetched")

    ctx = bpy.context
    state = dh.groups.get_addon_data(scene).state
    n_verts = len(plane.data.vertices)

    # ---- A: operators registered -------------------------------------
    has_usd = hasattr(bpy.ops.solver, "export_usd")
    has_abc = hasattr(bpy.ops.solver, "export_alembic")
    dh.record("A_ops_registered", has_usd and has_abc,
              {"export_usd": has_usd, "export_alembic": has_abc})

    # ---- B: poll True with a fetched cache ---------------------------
    pc2_path = dh.find_pc2_for(plane)
    pc2_count = (pc2_mod.read_pc2_frame_count(pc2_path)
                 if pc2_path and os.path.exists(pc2_path) else 0)
    exportable = sorted(o.name for o, _ in export_ops._exportable_meshes(ctx))
    poll_usd = bool(export_ops.SOLVER_OT_ExportUSD.poll(ctx))
    poll_abc = bool(export_ops.SOLVER_OT_ExportAlembic.poll(ctx))
    dh.record(
        "B_poll_true_after_fetch",
        poll_usd and poll_abc and plane.name in exportable and pc2_count > 0,
        {"poll_usd": poll_usd, "poll_abc": poll_abc,
         "exportable": exportable, "pc2_count": pc2_count, "n_verts": n_verts},
    )

    # ---- C: USDC content round-trip (read back via pxr) --------------
    scene.frame_set(3)
    plane.select_set(True)
    snap = dict(fs=scene.frame_start, fe=scene.frame_end,
                fc=scene.frame_current,
                sel=sorted(o.name for o in ctx.view_layer.objects
                           if o.select_get()))
    usd_res = set()
    if poll_usd:
        with bpy.context.temp_override(**_v3d_override()):
            usd_res = bpy.ops.solver.export_usd('EXEC_DEFAULT',
                                                filepath=usdc_path)
    restored = (scene.frame_start == snap["fs"]
                and scene.frame_end == snap["fe"]
                and scene.frame_current == snap["fc"]
                and sorted(o.name for o in ctx.view_layer.objects
                           if o.select_get()) == snap["sel"])

    usd_detail = {"result": list(usd_res), "restored": restored,
                  "pc2_count": pc2_count}
    usd_ok = False
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(usdc_path)
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        end_tc = float(stage.GetEndTimeCode())
        start_tc = float(stage.GetStartTimeCode())
        mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        # the rig scene holds only the sim mesh, so exactly one mesh prim
        prim = None
        for p in mesh_prims:
            pa = UsdGeom.Mesh(p).GetPointsAttr()
            first = pa.Get(start_tc)
            if first is not None and len(first) == n_verts:
                prim = p
                break
        if prim is None and mesh_prims:
            prim = mesh_prims[0]
        pts_attr = UsdGeom.Mesh(prim).GetPointsAttr() if prim else None
        n_ts = len(pts_attr.GetTimeSamples()) if pts_attr else 0
        usd_npts = len(pts_attr.Get(start_tc)) if pts_attr else 0

        def _usd_pts(f):
            p = pts_attr.Get(float(f))
            return (np.asarray([[v[0], v[1], v[2]] for v in p],
                               dtype=np.float64) if p else None)

        # per-frame round-trip: exported points == source evaluated verts
        max_pos_err = 0.0
        pos_match = pts_attr is not None
        for f in sorted({2, pc2_count}):
            up_pts = _usd_pts(f)
            src = _eval_local_verts(plane, f)
            if up_pts is None or up_pts.shape != src.shape:
                pos_match = False
                break
            err = float(np.abs(up_pts - src).max())
            max_pos_err = max(max_pos_err, err)
        pos_match = pos_match and max_pos_err < 1e-3
        # animation: first vs last point sample differ
        a = _usd_pts(1)
        b = _usd_pts(pc2_count)
        usd_motion = (float(np.abs(a - b).max())
                      if a is not None and b is not None else 0.0)

        usd_detail.update({
            "up_axis": up_axis, "start_tc": start_tc, "end_tc": end_tc,
            "n_mesh_prims": len(mesh_prims), "usd_npts": usd_npts,
            "n_timesamples": n_ts, "max_pos_err": max_pos_err,
            "usd_motion": usd_motion,
        })
        usd_ok = (
            "FINISHED" in usd_res
            and os.path.exists(usdc_path)
            and up_axis == "Z"
            and start_tc == 1.0
            and end_tc == float(pc2_count)
            and len(mesh_prims) == 1
            and usd_npts == n_verts
            and 2 <= n_ts <= pc2_count
            and pos_match
            and usd_motion > 1e-4
            and restored
        )
    except Exception as exc:
        usd_detail["exception"] = f"{type(exc).__name__}: {exc}"
    dh.record("C_usdc_content_roundtrips", usd_ok, usd_detail)

    # ---- D: ABC content round-trip (re-import via wm.alembic_import) --
    abc_res = set()
    if poll_abc:
        with bpy.context.temp_override(**_v3d_override()):
            abc_res = bpy.ops.solver.export_alembic('EXEC_DEFAULT',
                                                    filepath=abc_path)
    abc_detail = {"result": list(abc_res),
                  "exists": os.path.exists(abc_path),
                  "size": (os.path.getsize(abc_path)
                           if os.path.exists(abc_path) else 0)}
    abc_ok = False
    imported = []
    try:
        before = {o.name for o in bpy.data.objects}
        with bpy.context.temp_override(**_v3d_override()):
            bpy.ops.wm.alembic_import(filepath=abc_path,
                                      as_background_job=False)
        imported = [o for o in bpy.data.objects
                    if o.name not in before and o.type == "MESH"]
        imp = imported[0] if imported else None
        imp_nverts = len(imp.data.vertices) if imp else 0
        # imported mesh must animate, and its start->end motion magnitude must
        # match the source sim's (magnitude is invariant to any import-time
        # axis reorientation, so this is a robust content check).
        if imp is not None:
            imp_a = _eval_local_verts(imp, 1)
            imp_b = _eval_local_verts(imp, pc2_count)
            imp_motion = float(np.abs(imp_a - imp_b).max())
        else:
            imp_motion = 0.0
        src_a = _eval_local_verts(plane, 1)
        src_b = _eval_local_verts(plane, pc2_count)
        src_motion = float(np.abs(src_a - src_b).max())
        abc_detail.update({
            "n_imported": len(imported), "imp_nverts": imp_nverts,
            "imp_motion": imp_motion, "src_motion": src_motion,
        })
        abc_ok = (
            "FINISHED" in abc_res
            and imp is not None
            and imp_nverts == n_verts
            and imp_motion > 1e-4
            and abs(imp_motion - src_motion) < 1e-2
        )
    except Exception as exc:
        abc_detail["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        for o in imported:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
    dh.record("D_abc_content_roundtrips", abc_ok, abc_detail)

    # ---- E: refuses to export when frames are unfetched --------------
    # Clear the fetched-frame record: remote still reports frames but the addon
    # believes none are local (the MESH_CACHE stays). INVOKE_DEFAULT hits the
    # invoke() guard before the file browser opens; bpy.ops raises the
    # {"ERROR"} report as a RuntimeError.
    stale_usd = os.path.join(out_dir, "export_should_not_exist.usdc")
    if os.path.exists(stale_usd):
        os.remove(stale_usd)
    EXPECTED = "Fetch all animation frames first."
    state.clear_fetched_frames()
    guard = bake._has_unfetched_frames(scene)
    err = None
    try:
        with bpy.context.temp_override(**_v3d_override()):
            bpy.ops.solver.export_usd('INVOKE_DEFAULT', filepath=stale_usd)
    except RuntimeError as exc:
        err = str(exc)
    dh.record(
        "E_refuses_when_unfetched",
        guard
        and err is not None
        and EXPECTED in err
        and not os.path.exists(stale_usd),
        {"guard": guard, "error": err,
         "no_file": not os.path.exists(stale_usd)},
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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
