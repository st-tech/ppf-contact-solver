# File: scenarios/bl_merge_pair_no_stitch_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A merge pair whose stitch cannot reach the solver is REFUSED by name, at
# Transfer and at every encode, and nothing rewrites the pair on the way.
#
# A cross-object stitch is authored by the snap tool, which stamps the
# vertex-to-triangle correspondences into ``MergePairItem.cross_stitch_json``.
# When that capture finds nothing, when the mesh changes after the snap, or
# when a SOLID side lacks the points that place it on its tetrahedral surface,
# the seam cannot form. ``mesh_ops.merge_ops.merge_pair_problem`` is the one
# answer to "can this pair ship", read by the Transfer check
# (``ui.solver._check_merge_pairs_stitch``), the encoder
# (``core.encoder.params._encode_cross_stitch``, so Update Params refuses too)
# and the panel. A pair is removed only when an object's group membership
# ends, never by the check or the encode.
#
# This scenario is a pure host-side validation check (no server / build),
# mirroring ``bl_hanging_stitch_vertex_rejection``.
#
# Subtests:
#   A. empty_pair_rejected
#         Two SHELL strips in one group joined by a merge pair whose
#         cross_stitch_json is "". The check names the pair and says it has
#         no stitch points.
#   B. valid_pair_accepted
#         The same pair with a well-formed one-row cross_stitch_json passes
#         the check, and the encoder ships it with its stiffness.
#   C. stale_pair_refused_not_cleared
#         A stamped a_vert_count that no longer matches the mesh is refused
#         as a changed mesh, and the stored JSON is left as it was.
#   D. encode_refuses_the_same_pair
#         encode_param (the Update Params path) raises a ValueError naming
#         the pair and the same reason, and still writes nothing.
#   E. empty_weights_rejected
#         Index rows with no weight rows are refused as having no points.
#   F. legacy_solid_pair_rejected
#         A 4-wide pair (snapped before a SOLID side recorded its points)
#         whose target is SOLID is refused with a Re-snap reason, and so is a
#         6-wide one missing target_points; the same rows with a SHELL
#         target were accepted in B.
#   G. unassigned_endpoint_rejected
#         A pair naming an object in no active group is refused with that
#         object's name.
#   H. removal_from_group_removes_its_pairs
#         Removing an object from its group removes every merge pair naming
#         it, and leaves the other pairs.
#   I. deactivated_group_keeps_its_pairs
#         A group whose active flag is off (as an Undo past Delete Group can
#         leave it) still holds its members, so the deleted-object cleanup
#         that runs after depsgraph updates does not scan the merge pairs
#         away; Transfer refuses such a pair by name until the group is on.
#   J. delete_group_removes_its_members_pairs
#         Delete Group ends its members' membership, so the merge pairs
#         naming them are removed with it, and a pair between two other
#         objects is kept.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


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

    # Both meshes live in an active group, which a pair's endpoints must.
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

    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "_encode_cross_stitch"])

    def problem_of(p):
        return merge_ops.merge_pair_problem(bpy.context.scene, p)

    def set_json(p, payload):
        p.cross_stitch_json = json.dumps(payload, separators=(",", ":"))

    # ----- A: empty cross_stitch_json is rejected ---------------------
    pair.cross_stitch_json = ""
    err_empty = solver_mod._check_merge_pairs_stitch(bpy.context)
    dh.record(
        "A_empty_pair_rejected",
        merge_ops.pair_has_stitch(pair) is False
        and merge_ops.pair_stitch_row_count(pair) == 0
        and "no stitch points" in err_empty
        and "MeshA <-> MeshB" in err_empty,
        {"err": err_empty[:240]},
    )

    # ----- B: a well-formed pair passes and ships ---------------------
    set_json(pair, valid_payload(len(mesh_a.data.vertices)))
    err_valid = solver_mod._check_merge_pairs_stitch(bpy.context)
    shipped = params_mod._encode_cross_stitch(bpy.context)
    dh.record(
        "B_valid_pair_accepted",
        merge_ops.pair_stitch_row_count(pair) == 1
        and err_valid == ""
        and problem_of(pair) is None
        and len(shipped) == 1
        and shipped[0]["stitch_stiffness"] == 1000.0
        and len(shipped[0]["ind"][0]) == 6,
        {"rows": merge_ops.pair_stitch_row_count(pair),
         "err": err_valid[:240], "shipped": len(shipped)},
    )

    # ----- C: a stale pair is refused and left as it was --------------
    set_json(pair, valid_payload(len(mesh_a.data.vertices) + 99))
    stale_json = pair.cross_stitch_json
    err_stale = solver_mod._check_merge_pairs_stitch(bpy.context)
    dh.record(
        "C_stale_pair_refused_not_cleared",
        "changed since the pair was snapped" in err_stale
        and "MeshA" in err_stale
        and state.merge_pairs[0].cross_stitch_json == stale_json
        and len(state.merge_pairs) == 1,
        {"err": err_stale[:240],
         "kept": state.merge_pairs[0].cross_stitch_json == stale_json},
    )

    # ----- D: the encoder refuses the same pair -----------------------
    encode_err = ""
    try:
        params_mod.encode_param(bpy.context)
    except ValueError as exc:
        encode_err = str(exc)
    dh.record(
        "D_encode_refuses_the_same_pair",
        encode_err.startswith("Merge pair MeshA <-> MeshB:")
        and "changed since the pair was snapped" in encode_err
        and state.merge_pairs[0].cross_stitch_json == stale_json,
        {"err": encode_err[:240]},
    )

    # ----- E: index rows without weight rows --------------------------
    payload = valid_payload(len(mesh_a.data.vertices))
    payload["w"] = []
    set_json(pair, payload)
    dh.record(
        "E_empty_weights_rejected",
        "no stitch points" in (problem_of(pair) or ""),
        {"problem": problem_of(pair)},
    )
    set_json(pair, valid_payload(len(mesh_a.data.vertices)))

    # ----- F: a SOLID side without its placement points ---------------
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 3.0, 0.0))
    block = bpy.context.active_object
    block.name = "Block"
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(block.name)
    uuid_block = uuid_mod.get_or_create_object_uuid(block)
    legacy = state.merge_pairs.add()
    legacy.object_a, legacy.object_a_uuid = mesh_a.name, uuid_a
    legacy.object_b, legacy.object_b_uuid = block.name, uuid_block
    set_json(legacy, {
        "source_uuid": uuid_a, "target_uuid": uuid_block,
        "ind": [[0, 0, 1, 2]], "w": [[1.0, 0.2, 0.3, 0.5]],
        "a_vert_count": len(mesh_a.data.vertices),
        "b_vert_count": len(block.data.vertices),
    })
    legacy_4 = problem_of(legacy) or ""
    set_json(legacy, {
        "source_uuid": uuid_a, "target_uuid": uuid_block,
        "ind": [[0, 0, 0, 0, 1, 2]], "w": [[1.0, 0.0, 0.0, 0.2, 0.3, 0.5]],
        "a_vert_count": len(mesh_a.data.vertices),
        "b_vert_count": len(block.data.vertices),
    })
    legacy_6 = problem_of(legacy) or ""
    err_legacy = solver_mod._check_merge_pairs_stitch(bpy.context)
    dh.record(
        "F_legacy_solid_pair_rejected",
        "SOLID side" in legacy_4 and "Re-snap" in legacy_4
        and "SOLID side" in legacy_6
        and "MeshA <-> Block" in err_legacy
        and "MeshA <-> MeshB" not in err_legacy,
        {"four_wide": legacy_4, "six_wide": legacy_6,
         "err": err_legacy[:240]},
    )

    # ----- G: an endpoint in no active group --------------------------
    loose = make_strip("Loose", inner_x=3.0, outer_x=4.0)
    uuid_loose = uuid_mod.get_or_create_object_uuid(loose)
    stray = state.merge_pairs.add()
    stray.object_a, stray.object_a_uuid = mesh_a.name, uuid_a
    stray.object_b, stray.object_b_uuid = loose.name, uuid_loose
    set_json(stray, {
        "source_uuid": uuid_a, "target_uuid": uuid_loose,
        "ind": [[0, 0, 0, 0, 2, 3]], "w": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    })
    stray_problem = problem_of(stray) or ""
    dh.record(
        "G_unassigned_endpoint_rejected",
        "'Loose' is in no active dynamics group" in stray_problem,
        {"problem": stray_problem},
    )

    # ----- H: leaving a group removes the pairs naming the object -----
    before = [(p.object_a, p.object_b) for p in state.merge_pairs]
    solid.remove(block.name)
    after = [(p.object_a, p.object_b) for p in state.merge_pairs]
    h_after = after
    dh.record(
        "H_removal_from_group_removes_its_pairs",
        ("MeshA", "Block") in before
        and ("MeshA", "Block") not in after
        and ("MeshA", "MeshB") in after
        and ("MeshA", "Loose") in after,
        {"before": before, "after": after},
    )

    # ----- I: a deactivated group keeps its pairs -------------------
    group_ops = __import__(pkg + ".ui.dynamics.group_ops",
                           fromlist=["_apply_cleanup"])
    cloth_pg = dh.groups.get_active_group_by_uuid(bpy.context.scene, cloth.uuid)
    cloth_pg.active = False
    group_ops._apply_cleanup()
    kept = [(p.object_a, p.object_b) for p in state.merge_pairs]
    err_i = solver_mod._check_merge_pairs_stitch(bpy.context)
    cloth_pg.active = True
    group_ops._apply_cleanup()
    back = [(p.object_a, p.object_b) for p in state.merge_pairs]
    dh.record(
        "I_deactivated_group_keeps_its_pairs",
        kept == h_after and back == h_after
        and "MeshA <-> MeshB: 'MeshA' is in no active dynamics group" in err_i,
        {"kept": kept, "back": back, "err": err_i[:300]},
    )

    # ----- J: Delete Group removes its members' pairs ----------------
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, -3.0, 0.0))
    other_a = bpy.context.active_object
    other_a.name = "OtherA"
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(2.0, -3.0, 0.0))
    other_b = bpy.context.active_object
    other_b.name = "OtherB"
    others = dh.api.solver.create_group("Others", "SHELL")
    others.add(other_a.name)
    others.add(other_b.name)
    bystander = state.merge_pairs.add()
    bystander.object_a, bystander.object_a_uuid = other_a.name, uuid_mod.get_or_create_object_uuid(other_a)
    bystander.object_b, bystander.object_b_uuid = other_b.name, uuid_mod.get_or_create_object_uuid(other_b)
    before_j = [(p.object_a, p.object_b) for p in state.merge_pairs]
    slot = dh.groups.get_group_slot_index(bpy.context.scene, cloth.uuid)
    bpy.ops.object.delete_group(group_index=slot)
    after_j = [(p.object_a, p.object_b) for p in state.merge_pairs]
    dh.record(
        "J_delete_group_removes_its_members_pairs",
        ("MeshA", "MeshB") in before_j and ("MeshA", "MeshB") not in after_j
        and ("MeshA", "Loose") not in after_j
        and after_j == [("OtherA", "OtherB")],
        {"before": before_j, "after": after_j},
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
