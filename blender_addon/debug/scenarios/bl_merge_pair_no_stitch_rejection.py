# File: scenarios/bl_merge_pair_no_stitch_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Merge-pair "no stitch points" rejection coverage.
#
# A cross-object stitch is authored by the snap tool, which stamps the
# vertex-to-triangle correspondences into ``MergePairItem.cross_stitch_
# json``. When that capture finds nothing (the pieces only graze each
# other, beyond the contact-gap-scaled snap threshold) the operator still
# reports "Stitched ..." but stores an empty JSON. ``_encode_cross_stitch``
# then silently skips the pair, so the stitch stiffness has no effect and
# the seam never forms: the community-reported "high stiffness but it does
# not stitch" case.
#
# Two surfaces now catch this:
#   * ``mesh_ops.merge_ops.pair_has_stitch`` reports whether a pair carries
#     usable stitch rows (the panel shows an info label when it does not).
#   * ``ui.solver._check_merge_pairs_stitch`` turns an empty pair into a
#     hard error that aborts Transfer instead of shipping a scene that
#     looks stitched but is not.
#
# This scenario is a pure host-side validation check (no server / build),
# mirroring ``bl_hanging_stitch_vertex_rejection``.
#
# Subtests:
#   A. empty_pair_rejected
#         Two SHELL strips in one group joined by a merge pair whose
#         cross_stitch_json is "". ``pair_has_stitch`` is False and
#         ``_check_merge_pairs_stitch`` returns an error naming the pair.
#   B. valid_pair_accepted
#         The same pair with a well-formed one-row cross_stitch_json.
#         ``pair_has_stitch`` is True and the check returns "".
#   C. stale_cleared_pair_rejected
#         A well-formed cross_stitch_json whose stamped a_vert_count no
#         longer matches the mesh (a topology edit after snap).
#         ``cleanup_stale_merge_pairs`` (run inside the check) clears the
#         stale JSON, so the check rejects it exactly like the empty case.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import json
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_strip(name, inner_x, outer_x):
    # A two-triangle strip: verts 0,1 at inner_x and 2,3 at outer_x.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    coords = [
        (inner_x, 0.0, 0.0),  # 0
        (inner_x, 1.0, 0.0),  # 1
        (outer_x, 0.0, 0.0),  # 2
        (outer_x, 1.0, 0.0),  # 3
    ]
    faces = [(0, 1, 3), (0, 3, 2)]
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    merge_ops = __import__(pkg + ".mesh_ops.merge_ops",
                           fromlist=["pair_has_stitch",
                                     "pair_stitch_row_count"])
    solver_mod = __import__(pkg + ".ui.solver",
                            fromlist=["_check_merge_pairs_stitch"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    dh.log("setup_start")
    # reset_scene_to_pinned_plane clears the scene and gives us a saved
    # .blend so ID writes (UUIDs) stick.
    dh.reset_scene_to_pinned_plane(name="Base")
    dh.save_blend(PROBE_DIR, "merge_pair_no_stitch.blend")
    root = dh.configure_state(project_name="merge_pair_no_stitch",
                              frame_count=4)

    mesh_a = make_strip("MeshA", inner_x=0.0, outer_x=-1.0)
    mesh_b = make_strip("MeshB", inner_x=0.5, outer_x=1.5)

    # Both meshes must live in an active group, else cleanup_stale_merge_
    # pairs drops the pair before the stitch check can see it.
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(mesh_a.name)
    cloth.add(mesh_b.name)

    uuid_a = uuid_mod.get_or_create_object_uuid(mesh_a)
    uuid_b = uuid_mod.get_or_create_object_uuid(mesh_b)
    if not uuid_a or not uuid_b:
        raise RuntimeError("could not allocate UUIDs for stitch meshes")

    state = dh.groups.get_addon_data(bpy.context.scene).state
    pair = state.merge_pairs.add()
    pair.object_a = mesh_a.name
    pair.object_b = mesh_b.name
    pair.object_a_uuid = uuid_a
    pair.object_b_uuid = uuid_b
    pair.stitch_stiffness = 1000.0
    state.merge_pairs_index = len(state.merge_pairs) - 1

    def valid_payload(a_vert_count):
        return {
            "source_uuid": uuid_a,
            "target_uuid": uuid_b,
            "ind": [[0, 0, 0, 0, 2, 3]],
            "w": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            "source_points": [[0.0, 0.0, 0.0]],
            "target_points": [[0.5, 0.0, 0.0]],
            "a_vert_count": a_vert_count,
            "b_vert_count": len(mesh_b.data.vertices),
        }

    # ----- A: empty cross_stitch_json is rejected ---------------------
    pair.cross_stitch_json = ""
    err_empty = solver_mod._check_merge_pairs_stitch(bpy.context)
    dh.record(
        "A_empty_pair_rejected",
        merge_ops.pair_has_stitch(pair) is False
        and merge_ops.pair_stitch_row_count(pair) == 0
        and bool(err_empty)
        and "no stitch points" in err_empty
        and "MeshA" in err_empty and "MeshB" in err_empty,
        {"err": err_empty[:240]},
    )

    # ----- B: a well-formed pair passes -------------------------------
    pair.cross_stitch_json = json.dumps(
        valid_payload(len(mesh_a.data.vertices)), separators=(",", ":"),
    )
    err_valid = solver_mod._check_merge_pairs_stitch(bpy.context)
    dh.record(
        "B_valid_pair_accepted",
        merge_ops.pair_has_stitch(pair) is True
        and merge_ops.pair_stitch_row_count(pair) == 1
        and err_valid == "",
        {"has_stitch": merge_ops.pair_has_stitch(pair),
         "rows": merge_ops.pair_stitch_row_count(pair),
         "err": err_valid[:240]},
    )

    # ----- C: a stale (vert-count mismatch) pair is cleared + rejected -
    # Re-fetch the pair: cleanup in check B may have reordered nothing,
    # but the index is stable here (single pair).
    pair = state.merge_pairs[0]
    pair.cross_stitch_json = json.dumps(
        valid_payload(len(mesh_a.data.vertices) + 99), separators=(",", ":"),
    )
    err_stale = solver_mod._check_merge_pairs_stitch(bpy.context)
    # The check runs cleanup_stale_merge_pairs, which clears the stale JSON
    # in place; after the call the stored JSON must be empty and the check
    # must have rejected.
    cleared = (len(state.merge_pairs) > 0
               and not state.merge_pairs[0].cross_stitch_json)
    dh.record(
        "C_stale_cleared_pair_rejected",
        bool(err_stale) and "no stitch points" in err_stale and cleared,
        {"err": err_stale[:240], "cleared": cleared},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
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
