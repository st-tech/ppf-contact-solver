# File: scenarios/bl_deformed_target_snap.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Snap-side coverage for snapping onto a DEFORMED target. Snap and Merge must
# read the EVALUATED (post-modifier) surface the artist sees and the solver
# runs on, not obj.data (the undeformed bind pose). For an armature/shape-key
# driven character the bind pose can sit metres from where the deformed
# surface actually is (a cape whose rest mesh is off to the side), so a
# rest-pose snap chases the wrong geometry: it drags the source across the
# scene and stitches nothing.
#
# This scenario builds that split explicitly with a shape key: the STATIC
# target's obj.data (rest) is a plane 10 units ABOVE the cloth, while a fully
# valued deform shape key translates its evaluated surface down to just above
# the cloth. A snap that read rest would move the cloth ~10 units toward the
# far plane and find no anchors; the evaluated snap moves the cloth a hair to
# the near deformed surface and stitches it.
#
# Blender-only (no solver build / server run), so it is fast and runs on the
# macOS + Linux Blender CI. Complements bl_static_snap_guard.py (which fixes
# the source/target roles and the STATIC no-move guard on undeformed meshes).
#
# Subtests:
#   A. deformed_static_stitched: the cloth snaps to the deformed surface (not
#      the rest plane): anchors are built, their target points sit at the
#      deformed height (~0), and the cloth moved only a small amount.
#   B. overlay_tracks_deformed: the stitch overlay (pins._overlay_object_points)
#      resolves the target endpoints on the DEFORMED surface too, so the yellow
#      stitch lines are short. A rest-pose overlay would draw the target end ~10
#      units up (the bind pose) and the lines would span the whole gap.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import json as _json
import traceback

import bpy

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    snap_mod = __import__(pkg + ".mesh_ops.snap_ops",
                          fromlist=["_snap_pair"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    class _FakeOp:
        # _snap_pair only needs an object with a .report(level, msg) method.
        def __init__(self):
            self.reports = []

        def report(self, level, msg):
            self.reports.append((tuple(level), msg))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.configure_state(
        project_name="deformed_target_snap",
        frame_count=2,
        gravity=(0.0, 0.0, 0.0),
    )

    # Cloth (SHELL) source plane at the origin.
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    cloth = bpy.context.active_object
    cloth.name = "Cloth"

    # STATIC target: rest plane FAR above the cloth (Z=+10), with a fully
    # valued deform shape key that translates every vertex down so the
    # EVALUATED surface sits just above the cloth (Z=~0.04). obj.data stays at
    # Z=+10; only the evaluated mesh is near the cloth.
    _REST_Z = 10.0
    _DEFORMED_Z = 0.04
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, _REST_Z))
    static = bpy.context.active_object
    static.name = "DeformStatic"
    static.shape_key_add(name="Basis")
    sk = static.shape_key_add(name="Deform")
    for v in sk.data:
        v.co = (v.co[0], v.co[1], v.co[2] - (_REST_Z - _DEFORMED_Z))
    sk.value = 1.0

    cloth_group = dh.api.solver.create_group("Cloth", "SHELL")
    cloth_group.add(cloth.name)
    static_group = dh.api.solver.create_group("DeformStatic", "STATIC")
    static_group.add(static.name)
    cloth_uuid = uuid_mod.get_or_create_object_uuid(cloth)
    static_uuid = uuid_mod.get_or_create_object_uuid(static)
    bpy.context.view_layer.update()

    cloth_before = tuple(cloth.matrix_world.translation)
    static_before = tuple(static.matrix_world.translation)

    op = _FakeOp()
    status = snap_mod._snap_pair(op, bpy.context, cloth, static)

    bpy.context.view_layer.update()
    cloth_after = tuple(cloth.matrix_world.translation)
    static_after = tuple(static.matrix_world.translation)
    cloth_moved = max(abs(a - b) for a, b in zip(cloth_before, cloth_after))
    static_moved = max(abs(a - b) for a, b in zip(static_before, static_after))

    # obj.data must be untouched: the rest plane stays at Z=+10 (only the
    # evaluated surface was near the cloth).
    static_rest_z = max(
        (static.matrix_world @ v.co).z for v in static.data.vertices
    )

    cs = None
    for pair in dh.groups.get_addon_data(bpy.context.scene).state.merge_pairs:
        uset = {pair.object_a_uuid, pair.object_b_uuid}
        if uset == {cloth_uuid, static_uuid} and pair.cross_stitch_json:
            cs = _json.loads(pair.cross_stitch_json)
            break
    anchor_count = len(cs.get("ind", [])) if cs else 0
    # Anchor target points must sit at the DEFORMED height (~0), never the rest
    # height (~10): this is the assertion that fails if the snap reads obj.data.
    max_target_z = (
        max(abs(p[2]) for p in cs.get("target_points", [])) if (cs and cs.get("target_points")) else None
    )

    ok = (
        "FINISHED" in status
        and static_moved < 1e-6            # STATIC object never translated
        and static_rest_z > _REST_Z - 1.0  # rest mesh untouched, still far up
        and cs is not None
        and anchor_count >= 1              # stitched to the deformed surface
        and max_target_z is not None
        and max_target_z < 1.0             # anchors on deformed (~0), not rest (~10)
        and cloth_moved < 1.0              # small nudge, not a ~10-unit chase of rest
    )
    dh.record(
        "A_deformed_static_stitched",
        ok,
        {
            "status": sorted(status),
            "anchor_count": anchor_count,
            "max_target_z": max_target_z,
            "cloth_moved": cloth_moved,
            "static_moved": static_moved,
            "static_rest_z": static_rest_z,
            "op_reports": [m for _, m in op.reports],
        },
    )

    # B: reconstruct the yellow stitch lines the viewport overlay draws
    # (pins._overlay_object_points) and assert they are short. A rest-pose
    # overlay would resolve the target end at the bind pose (~10 up), so the
    # lines would span the whole gap.
    overlay_ok = False
    overlay_max_len = None
    if cs:
        pins_mod = __import__(pkg + ".ui.dynamics.overlay_geometry.pins",
                              fromlist=["_overlay_object_points"])
        dgraph = bpy.context.evaluated_depsgraph_get()
        src_pts = pins_mod._overlay_object_points(bpy.context.scene, cs["source_uuid"], dgraph)
        tgt_pts = pins_mod._overlay_object_points(bpy.context.scene, cs["target_uuid"], dgraph)
        lens = []
        if src_pts is not None and tgt_pts is not None:
            for row, wt in zip(cs["ind"], cs["w"]):
                si = int(row[0])
                ti = (int(row[3]), int(row[4]), int(row[5]))
                ww = (wt[3], wt[4], wt[5])
                tp = ww[0] * tgt_pts[ti[0]] + ww[1] * tgt_pts[ti[1]] + ww[2] * tgt_pts[ti[2]]
                lens.append((src_pts[si] - tp).length)
        overlay_max_len = max(lens) if lens else None
        overlay_ok = overlay_max_len is not None and overlay_max_len < 1.0
    dh.record(
        "B_overlay_tracks_deformed",
        overlay_ok,
        {"overlay_max_line_len": overlay_max_len},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
