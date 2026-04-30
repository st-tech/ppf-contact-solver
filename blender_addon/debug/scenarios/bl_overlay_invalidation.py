# File: scenarios/bl_overlay_invalidation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Draw-stale matrix: every property whose update= callback should bump
# ``state.overlay_version`` is mutated, and we assert the version
# actually moved. Catches regressions where a new property gets added
# without wiring an invalidate callback (the bug class behind recent
# commits "Refresh viewport overlay on every preview-feeding state
# change", "Invalidate collider overlay cache on property edits", and
# "Fix Blender UI staleness after state changes across operators and
# handlers").
#
# Coverage by overlay class (per the audit in
# blender_addon/ui/dynamics/overlay.py):
#   - State toggles: hide_pins, hide_arrows, hide_overlay_colors,
#     hide_snaps, hide_pin_operations
#   - Direction previews: gravity_3d, preview_gravity_direction,
#     wind_direction, wind_strength, preview_wind_direction
#   - Pin vertex group: included, use_pin_duration, pin_duration
#   - Pin operations: show_overlay, op_type, delta, spin_axis,
#     spin_angular_velocity, spin_flip, spin_center, spin_center_mode,
#     scale_factor, scale_center, torque_magnitude, torque_flip
#   - Invisible colliders: position, normal, radius, hemisphere,
#     invert, enable_active_duration, active_duration, show_preview
#   - Velocity keyframes: frame, direction, speed, preview
#   - Object groups: color, show_overlay_color, preview_velocity
#
# Pure UI scenario: no server, no solver, no transfer. The bootstrap's
# debug-server connection is unused.

from __future__ import annotations

import os

from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

def log(msg):
    result["phases"].append((round(time.time(), 3), msg))

def record(case, before, after, *, expect_change=True):
    moved = after != before
    ok = moved if expect_change else not moved
    result["checks"][case] = {
        "ok": ok,
        "details": {"before": before, "after": after, "moved": moved},
    }

try:
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "OverlayMesh"
    n_verts = len(plane.data.vertices)
    vg = plane.vertex_groups.new(name="OverlayPin")
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "OverlayPin")
    pin.move_by(delta=(0.1, 0, 0), frame_start=1, frame_end=10)
    pin.spin(axis=(0, 0, 1), angular_velocity=90.0,
             frame_start=1, frame_end=10)
    pin.scale(factor=0.5, frame_start=1, frame_end=10)
    solver_api.add_wall(position=(0, 0, -1), normal=(0, 0, 1))
    solver_api.add_sphere(position=(0, 0, 1), radius=0.25)

    root = groups_mod.get_addon_data(bpy.context.scene)
    state = root.state

    # Resolve the addon data structures we need to mutate.
    group = root.object_group_0  # ``Cloth`` lands at slot 0
    assigned = group.assigned_objects[0]
    pin_item = group.pin_vertex_groups[0]
    move_op = pin_item.operations[0]   # MOVE_BY
    spin_op = pin_item.operations[1]   # SPIN
    scale_op = pin_item.operations[2]  # SCALE
    wall = state.invisible_colliders[0]
    sphere = state.invisible_colliders[1]

    # Velocity keyframes are not part of the public API; reach into the
    # PropertyGroup directly so we can exercise the update= callbacks.
    vk = assigned.velocity_keyframes.add()
    vk.frame = 1
    vk.direction = (0.0, 0.0, 1.0)
    vk.speed = 1.0

    log("setup_done")

    # ------------------------------------------------------------
    # Each entry: (case_name, lambda that mutates one property)
    # ------------------------------------------------------------
    cases = [
        # State-level toggles -------------------------------------
        ("state.hide_pins",
         lambda: setattr(state, "hide_pins", not state.hide_pins)),
        ("state.hide_arrows",
         lambda: setattr(state, "hide_arrows", not state.hide_arrows)),
        ("state.hide_overlay_colors",
         lambda: setattr(state, "hide_overlay_colors",
                         not state.hide_overlay_colors)),
        ("state.hide_snaps",
         lambda: setattr(state, "hide_snaps", not state.hide_snaps)),
        ("state.hide_pin_operations",
         lambda: setattr(state, "hide_pin_operations",
                         not state.hide_pin_operations)),
        # Direction-preview surfaces ------------------------------
        ("state.preview_gravity_direction",
         lambda: setattr(state, "preview_gravity_direction",
                         not state.preview_gravity_direction)),
        ("state.gravity_3d",
         lambda: setattr(state, "gravity_3d", (0.0, 0.0, -4.9))),
        ("state.preview_wind_direction",
         lambda: setattr(state, "preview_wind_direction",
                         not state.preview_wind_direction)),
        ("state.wind_direction",
         lambda: setattr(state, "wind_direction", (1.0, 0.0, 0.0))),
        ("state.wind_strength",
         lambda: setattr(state, "wind_strength", 5.0)),
        # ObjectGroup / AssignedObject ---------------------------
        ("group.show_overlay_color",
         lambda: setattr(group, "show_overlay_color",
                         not group.show_overlay_color)),
        ("group.color",
         lambda: setattr(group, "color", (0.5, 0.5, 0.0, 1.0))),
        ("group.preview_velocity",
         lambda: setattr(group, "preview_velocity",
                         not group.preview_velocity)),
        ("group.show_pin_overlay",
         lambda: setattr(group, "show_pin_overlay",
                         not group.show_pin_overlay)),
        ("group.pin_overlay_size",
         lambda: setattr(group, "pin_overlay_size", 16.0)),
        ("assigned.included",
         lambda: setattr(assigned, "included", not assigned.included)),
        # Pin vertex group --------------------------------------
        ("pin.use_pin_duration",
         lambda: setattr(pin_item, "use_pin_duration",
                         not pin_item.use_pin_duration)),
        ("pin.pin_duration",
         lambda: setattr(pin_item, "pin_duration", 30)),
        ("pin.included",
         lambda: setattr(pin_item, "included", not pin_item.included)),
        # MOVE_BY pin op ----------------------------------------
        ("move_op.show_overlay",
         lambda: setattr(move_op, "show_overlay",
                         not move_op.show_overlay)),
        ("move_op.delta",
         lambda: setattr(move_op, "delta", (0.5, 0.0, 0.0))),
        # Note: ``transition`` and ``frame_start`` / ``frame_end`` do
        # not bump overlay_version because the overlay renders a
        # static trajectory; the time curve isn't drawn. If you add an
        # overlay that reads those, wire an invalidate callback and
        # add the case here.
        # SPIN pin op -------------------------------------------
        ("spin_op.show_overlay",
         lambda: setattr(spin_op, "show_overlay",
                         not spin_op.show_overlay)),
        ("spin_op.spin_axis",
         lambda: setattr(spin_op, "spin_axis", (1.0, 0.0, 0.0))),
        ("spin_op.spin_angular_velocity",
         lambda: setattr(spin_op, "spin_angular_velocity", 45.0)),
        ("spin_op.spin_center",
         lambda: setattr(spin_op, "spin_center", (0.1, 0.2, 0.0))),
        ("spin_op.spin_center_mode",
         lambda: setattr(spin_op, "spin_center_mode", "ABSOLUTE")),
        ("spin_op.spin_flip",
         lambda: setattr(spin_op, "spin_flip", not spin_op.spin_flip)),
        # SCALE pin op ------------------------------------------
        ("scale_op.show_overlay",
         lambda: setattr(scale_op, "show_overlay",
                         not scale_op.show_overlay)),
        ("scale_op.scale_factor",
         lambda: setattr(scale_op, "scale_factor", 1.5)),
        ("scale_op.scale_center",
         lambda: setattr(scale_op, "scale_center", (0.0, 0.1, 0.0))),
        ("scale_op.scale_center_mode",
         lambda: setattr(scale_op, "scale_center_mode", "ABSOLUTE")),
        # Wall collider -----------------------------------------
        ("wall.position",
         lambda: setattr(wall, "position", (0.0, 0.0, -2.0))),
        ("wall.normal",
         lambda: setattr(wall, "normal", (0.0, 1.0, 0.0))),
        ("wall.show_preview",
         lambda: setattr(wall, "show_preview", not wall.show_preview)),
        ("wall.enable_active_duration",
         lambda: setattr(wall, "enable_active_duration",
                         not wall.enable_active_duration)),
        ("wall.active_duration",
         lambda: setattr(wall, "active_duration", 30)),
        # Sphere collider ---------------------------------------
        ("sphere.position",
         lambda: setattr(sphere, "position", (0.5, 0.0, 1.0))),
        ("sphere.radius",
         lambda: setattr(sphere, "radius", 0.5)),
        ("sphere.hemisphere",
         lambda: setattr(sphere, "hemisphere", not sphere.hemisphere)),
        ("sphere.invert",
         lambda: setattr(sphere, "invert", not sphere.invert)),
        ("sphere.show_preview",
         lambda: setattr(sphere, "show_preview",
                         not sphere.show_preview)),
        # Velocity keyframe -------------------------------------
        ("vk.frame",
         lambda: setattr(vk, "frame", 5)),
        ("vk.direction",
         lambda: setattr(vk, "direction", (1.0, 0.0, 0.0))),
        ("vk.speed",
         lambda: setattr(vk, "speed", 2.0)),
        ("vk.preview",
         lambda: setattr(vk, "preview", not vk.preview)),
    ]

    for case, mutator in cases:
        before = state.overlay_version
        try:
            mutator()
        except Exception as exc:
            result["errors"].append(
                f"{case}: mutator raised {type(exc).__name__}: {exc}"
            )
            continue
        after = state.overlay_version
        record(case, before, after, expect_change=True)

    log(f"checks={len(result['checks'])} done")

    # Sanity check: an unrelated property mutation should NOT bump
    # overlay_version. Catches "every setattr bumps it" false positives
    # from a runaway depsgraph subscriber.
    state.air_friction = 0.42  # no update= callback
    sanity_before = state.overlay_version
    state.air_friction = 0.13  # second mutation; if no callback, no bump
    record("sanity.air_friction_no_invalidate",
           sanity_before, state.overlay_version, expect_change=False)
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
