# File: scenarios/bl_pin_reorder_and_gating.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Addon-side checks for the pin UI / encode work (no solver needed):
#   * the pin reorder operator (object.move_pin_vertex_group),
#   * per-vertex last-wins follows pin ORDER (the lower/last pin wins for
#     vertices shared by overlapping pins),
#   * the Pin Stiffness enabled-gate (disabled when Pull is on),
#   * the per-pin Show overlay default (eye open on a freshly added pin).
#
# Subtests:
#   A. reorder_operator_moves_pin
#   B. last_wins_follows_order  (pin-root last => bottom hard; first => pull)
#   C. pin_stiffness_disabled_when_pull
#   D. show_overlay_default_on

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

_SUBDIV = 1


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
SUBDIV = <<SUBDIV>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "SolidBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    n_verts = len(cube.data.vertices)
    zmin = min(v.co.z for v in cube.data.vertices)
    bottom = [v.index for v in cube.data.vertices if abs(v.co.z - zmin) < 1e-4]
    cube.vertex_groups.new(name="pin").add(list(range(n_verts)), 1.0, "REPLACE")
    cube.vertex_groups.new(name="pin-root").add(bottom, 1.0, "REPLACE")

    dh.save_blend(PROBE_DIR, "pin_reorder.blend")
    root = dh.configure_state(project_name="pin_reorder_and_gating",
                             frame_count=2, frame_rate=100, step_size=0.01)
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    solid.create_pin(cube.name, "pin")        # index 0
    solid.create_pin(cube.name, "pin-root")   # index 1 (last)

    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    group.pin_vertex_groups[0].use_pull = True
    group.pin_vertex_groups[0].pull_strength = 1000.0
    group.pin_vertex_groups[1].use_pull = False

    enc = __import__(pkg + ".core.encoder.pin", fromlist=["_encode_pin_config"])
    decode = __import__(pkg + ".core.uuid_registry",
                        fromlist=["get_or_create_object_uuid"])
    state = addon_root.state
    sample_bottom = sorted(bottom)[0]

    def bottom_has_pull():
        pc = enc._encode_pin_config(bpy.context, [group], state)
        for _ouuid, vmap in pc.items():
            c = vmap.get(sample_bottom)
            if c is not None:
                return "pull_strength" in c
        return None

    # ----- D: show_overlay default on for freshly added pins ------
    dh.record(
        "D_show_overlay_default_on",
        bool(group.pin_vertex_groups[0].show_overlay)
        and bool(group.pin_vertex_groups[1].show_overlay),
        {"pin0": bool(group.pin_vertex_groups[0].show_overlay),
         "pin1": bool(group.pin_vertex_groups[1].show_overlay)},
    )

    # ----- C: Pin Stiffness disabled when Pull on -----------------
    # Mirror the panel gate: enabled = not use_pull and (ops or captured).
    def stiffness_enabled(p):
        return (not p.use_pull) and (
            len(p.operations) > 0 or p.has_captured_anim)
    pull_pin = group.pin_vertex_groups[0]   # use_pull True
    hard_pin = group.pin_vertex_groups[1]   # use_pull False
    # give the hard pin an op so its gate would be enabled if not for pull
    op = hard_pin.operations.add()
    op.op_type = "MOVE_BY"
    dh.record(
        "C_pin_stiffness_disabled_when_pull",
        (stiffness_enabled(pull_pin) is False)
        and (stiffness_enabled(hard_pin) is True),
        {"pull_pin_enabled": stiffness_enabled(pull_pin),
         "hard_pin_with_op_enabled": stiffness_enabled(hard_pin)},
    )
    # remove the op so it does not perturb the last-wins check
    hard_pin.operations.remove(0)

    # ----- B (part 1): pin-root LAST => bottom is HARD (no pull) ---
    bottom_pull_before = bottom_has_pull()

    # ----- A: reorder operator moves pin-root up to index 0 -------
    group.pin_vertex_groups_index = 1   # select pin-root
    bpy.ops.object.move_pin_vertex_group(
        "EXEC_DEFAULT", group_index=0, direction=-1)
    order_after = [p.name for p in group.pin_vertex_groups]
    moved = (order_after[0].endswith("[pin-root]")
             and order_after[1].endswith("[pin]")
             and group.pin_vertex_groups_index == 0)
    dh.record("A_reorder_operator_moves_pin", moved,
              {"order_after": order_after,
               "selected_index": group.pin_vertex_groups_index})

    # ----- B (part 2): pin-root FIRST => bottom is PULL (pin wins) -
    bottom_pull_after = bottom_has_pull()
    dh.record(
        "B_last_wins_follows_order",
        bottom_pull_before is False and bottom_pull_after is True,
        {"bottom_has_pull_with_root_last": bottom_pull_before,
         "bottom_has_pull_with_root_first": bottom_pull_after},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<SUBDIV>>", str(_SUBDIV))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
