# File: scenarios/bl_shallow_copy.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Linked-duplicate (Alt-D / shallow copy) rejection coverage.
#
# Two layers:
#   - The assignment operator ``OBJECT_OT_AddObjectsToGroup`` refuses
#     to take a Linked Duplicate at all, reporting ERROR and leaving
#     the group's ``assigned_objects`` list untouched.
#   - The encoder ``_build_obj_data`` (run by ``encode_obj`` AND the
#     much-rarer ``compute_data_hash``) walks every active group and
#     raises ``ValueError`` if any assigned object shares a mesh data
#     block with another. This catches out-of-band assignments
#     (older .blends, MCP scripts that touch the PropertyGroup
#     directly) the moment they hit Transfer.
#
# Subtests:
#   A. assignment_rejects_linked_duplicate
#         Two objects share a mesh data block via ``copy()`` (the
#         programmatic equivalent of Alt-D). Selecting and adding
#         the duplicate via ``object.add_objects_to_group`` reports
#         ERROR and the duplicate does NOT appear in the group's
#         assigned_objects.
#
#   B. encoder_rejects_out_of_band_assignment
#         Force-assign the Linked Duplicate by appending directly
#         to the PropertyGroup (bypassing the operator). Calling
#         ``encode_obj`` raises ``ValueError`` mentioning the
#         shallow-copy fact. ``compute_data_hash`` raises the same
#         way (so the debounced cache also surfaces the issue rather
#         than silently caching a stale hash).
#
#   C. clean_scene_passes
#         Sanity: with all duplicates made single-user, both
#         ``encode_obj`` and ``compute_data_hash`` succeed.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    utils_mod = __import__(pkg + ".core.utils",
                           fromlist=["find_linked_duplicate_siblings"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ShallowMesh")
    dh.save_blend(PROBE_DIR, "shallow.blend")
    root = dh.configure_state(project_name="shallow_copy", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    # Make a Linked Duplicate: a NEW object that points at the SAME
    # mesh data block as ``plane``. This is what Alt-D does in the UI.
    dup = bpy.data.objects.new("ShallowDup", plane.data)
    bpy.context.collection.objects.link(dup)
    siblings = utils_mod.find_linked_duplicate_siblings(dup)
    dh.log(f"dup siblings={siblings}")

    # ----- A: assignment operator refuses ------------------------
    # Blender re-raises ``self.report({"ERROR"}, ...)`` from inside
    # ``bpy.ops.*`` as a RuntimeError to the caller, which is the
    # exact user-facing behavior (a popup + an aborted op). The test
    # accepts EITHER a RuntimeError OR a return value of CANCELLED;
    # both prove the assignment was refused. The post-condition --
    # the duplicate is NOT in the assigned_objects collection -- is
    # the load-bearing assertion.
    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True)
    pre_count = len(root.object_group_0.assigned_objects)
    op_raised = False
    op_verdict = None
    try:
        op_verdict = bpy.ops.object.add_objects_to_group(group_index=0)
    except RuntimeError:
        op_raised = True
    post_count = len(root.object_group_0.assigned_objects)
    dh.record(
        "A_assignment_rejects_linked_duplicate",
        post_count == pre_count
        and not any(a.name == dup.name
                    for a in root.object_group_0.assigned_objects)
        and (op_raised or op_verdict == {"CANCELLED"})
        and bool(siblings),
        {
            "pre_count": pre_count, "post_count": post_count,
            "siblings": siblings,
            "op_raised": op_raised,
            "op_verdict": list(op_verdict) if op_verdict else None,
        },
    )

    # ----- B: out-of-band assignment fails at encode -------------
    item = root.object_group_0.assigned_objects.add()
    item.name = dup.name
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    item.uuid = uuid_mod.get_or_create_object_uuid(dup)
    encode_err = ""
    hash_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
    except ValueError as e:
        encode_err = str(e)
    try:
        encoder_mesh.compute_data_hash(bpy.context)
    except ValueError as e:
        hash_err = str(e)
    dh.record(
        "B_encoder_rejects_out_of_band_assignment",
        bool(encode_err)
        and bool(hash_err)
        and "Linked Duplicate" in encode_err
        and "Linked Duplicate" in hash_err,
        {
            "encode_err": encode_err[:120],
            "hash_err": hash_err[:120],
        },
    )

    # ----- C: clean scene encodes / hashes cleanly ---------------
    # Pop the bad assignment back off; make_single_user via
    # ``data.copy()`` so the duplicate gets its own data block.
    root.object_group_0.assigned_objects.remove(
        len(root.object_group_0.assigned_objects) - 1
    )
    dup.data = plane.data.copy()
    encode_clean_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
    except Exception as e:
        encode_clean_err = f"{type(e).__name__}: {e}"
    try:
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        encode_clean_err += f" / hash: {type(e).__name__}: {e}"
    dh.record(
        "C_clean_scene_passes",
        encode_clean_err == "",
        {"err": encode_clean_err},
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
