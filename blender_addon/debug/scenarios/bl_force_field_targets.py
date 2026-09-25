# File: scenarios/bl_force_field_targets.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Force field sources narrowed to object GROUPS (owner's request): each field
# object and the script reaches every group, or with Apply to All Groups off,
# the groups in its list. Driven through the panel's operators, the encoder,
# the Python API and a real solve.
#
# Two cloth groups side by side, gravity off, a Static group far below. A Force
# field under both cloths is narrowed to group A, and the script, which pushes
# along Blender +Y, to group B. Both reach both cloths in space, so what keeps
# each off the other group is the targeting and nothing else.
#
# Subtests:
#   A. operators_narrow_and_edit_the_list: Choose Groups creates the entry with
#      Apply to All Groups off, Add Group appends, Remove Group removes.
#   B. static_and_empty_targets_are_refused: a Static group is refused when
#      set, and a narrowed source with an empty list, or naming a group that
#      no longer exists, fails the encode by name.
#   C. payload_names_groups_by_position: the Force field's grid and the script
#      carry exactly their group's position in the payload's group list.
#   D. python_api_round_trips: get/set_force_field_targets read and write the
#      same state.
#   E. only_the_targeted_groups_move: after a real solve group A rose and did
#      not drift (its y motion is the radial field's own, a small fraction of
#      B's), group B drifted along +Y and did not rise at all.
#   F. lists_follow_rename_and_delete: a renamed group is listed under its new
#      name, and deleting a group removes it from every list.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 20

_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    dh = DriverHelpers(pkg, result)
    targets = __import__(pkg + ".models.force_field_targets", fromlist=["SCRIPT"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    groups_mod = __import__(pkg + ".models.groups", fromlist=["iterate_object_groups"])
    api = dh.api.solver
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def sheet(name, x):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6,
                                        size=1.0, location=(x, 0.0, 0.0))
        o = bpy.context.object
        o.name = name
        return o

    sheet_a = sheet("FFTargetA", -1.2)
    sheet_b = sheet("FFTargetB", 1.2)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 0.0, -4.0))
    collider = bpy.context.object
    collider.name = "FFCollider"
    dh.save_blend(PROBE_DIR, "force_field_targets.blend")
    root = dh.configure_state(project_name="force_field_targets", frame_count=FRAME_COUNT)
    state = root.state

    group_a = api.create_group("GroupA", "SHELL")
    group_a.add(sheet_a.name)
    group_b = api.create_group("GroupB", "SHELL")
    group_b.add(sheet_b.name)
    group_s = api.create_group("GroupS", "STATIC")
    group_s.add(collider.name)

    bpy.ops.object.effector_add(type="FORCE", location=(0.0, 0.0, -0.6))
    push = bpy.context.object
    push.name = "FFPush"
    push.field.strength = 6.0
    text = bpy.data.texts.new("ff_targets.py")
    text.from_string("def eval(x, y, z, t):\n    return (0.0, 2.0, 0.0)\n")
    state.force_field_script = text
    state.force_field_spacing = 0.1
    state.force_field_time_samples = 1

    # ----- A --------------------------------------------------------------
    bpy.ops.scene.force_field_choose_groups(source="FFPush")
    entry = targets.entry_for(state, push)
    narrowed = entry is not None and not entry.apply_all and len(entry.groups) == 0
    bpy.ops.scene.force_field_add_group(source="FFPush", group=group_b.uuid)
    bpy.ops.scene.force_field_add_group(source="FFPush", group=group_a.uuid)
    two = [ref.uuid for ref in entry.groups] == [group_b.uuid, group_a.uuid]
    entry.groups_index = 0
    bpy.ops.scene.force_field_remove_group(source="FFPush")
    one = [ref.uuid for ref in entry.groups] == [group_a.uuid]
    dh.record("A_operators_narrow_and_edit_the_list", narrowed and two and one,
              {"narrowed": narrowed, "two": two, "one": one})

    # ----- B --------------------------------------------------------------
    problems = {}
    try:
        targets.set_targets(scene, state, targets.SCRIPT, [group_s.uuid])
    except ValueError as e:
        problems["static"] = str(e)
    state.force_field_script_all = False
    try:
        params_mod._build_param_dict(bpy.context)
    except ValueError as e:
        problems["empty"] = str(e)
    ghost = state.force_field_script_groups.add()
    ghost.uuid = "no-such-group"
    ghost.name = "Ghost"
    try:
        params_mod._build_param_dict(bpy.context)
    except ValueError as e:
        problems["missing"] = str(e)
    dh.record("B_static_and_empty_targets_are_refused",
              "Static" in problems.get("static", "")
              and "reach nothing" in problems.get("empty", "")
              and "no longer exists" in problems.get("missing", ""),
              problems)
    state.force_field_script_groups.clear()
    targets.set_targets(scene, state, targets.SCRIPT, [group_b.uuid])

    # ----- C --------------------------------------------------------------
    built = params_mod._build_param_dict(bpy.context)
    active = [g for g in groups_mod.iterate_object_groups(scene) if g.active]
    pos = {g.uuid: i for i, g in enumerate(active)}
    ffp = built.get("force_field") or {}
    grid_groups = [g.get("groups") for g in ffp.get("grids", [])]
    script_groups = [s_.get("groups") for s_ in ffp.get("scripts", [])]
    dh.record("C_payload_names_groups_by_position",
              grid_groups == [[pos[group_a.uuid]]] and script_groups == [[pos[group_b.uuid]]],
              {"grids": grid_groups, "scripts": script_groups, "positions": pos})

    # ----- D --------------------------------------------------------------
    got_push = api.get_force_field_targets("FFPush")
    api.set_force_field_targets("SCRIPT", None)
    all_script = api.get_force_field_targets("SCRIPT")
    api.set_force_field_targets("SCRIPT", [group_b])
    back = api.get_force_field_targets("SCRIPT")
    dh.record("D_python_api_round_trips",
              got_push == [group_a.uuid] and all_script is None and back == [group_b.uuid],
              {"push": got_push, "script_all": all_script, "script": back})

    # ----- E --------------------------------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="force_field_targets:build",
                      timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def travel(obj):
        path = dh.find_pc2_for(obj)
        arr = dh.read_pc2(path) if path else None
        if arr is None or arr.shape[0] < 3:
            return None
        return (arr[-1] - arr[1]).mean(axis=0)

    ta, tb = travel(sheet_a), travel(sheet_b)
    dh.record("E_only_the_targeted_groups_move",
              ta is not None and tb is not None
              and ta[2] > 0.01 and abs(ta[1]) < 0.05 * tb[1]
              and tb[1] > 0.005 and abs(tb[2]) < 1e-6,
              {"a": None if ta is None else [float(v) for v in ta],
               "b": None if tb is None else [float(v) for v in tb],
               "error": dh.facade.engine.state.error})

    # ----- F --------------------------------------------------------------
    live = __import__(pkg + ".models.force_field_targets", fromlist=["display_name"])
    raw_a = groups_mod.get_group_by_uuid(scene, group_a.uuid)
    raw_a.name = "Renamed Group"
    shown = live.display_name(scene, targets.entry_for(state, push).groups[0])
    uuid_a = group_a.uuid
    group_a.delete()
    left = [ref.uuid for ref in targets.entry_for(state, push).groups]
    script_left = [ref.uuid for ref in state.force_field_script_groups]
    dh.record("F_lists_follow_rename_and_delete",
              shown == "Renamed Group" and uuid_a not in left
              and script_left == [group_b.uuid],
              {"shown": shown, "push_left": left, "script_left": script_left})

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
