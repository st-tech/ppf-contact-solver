# File: overlay_labels.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import blf  # pyright: ignore
import bpy  # pyright: ignore

from mathutils import Vector  # pyright: ignore


def draw_overlay_labels(overlay_cache):
    """Draw text labels for direction previews, operations, and velocity (POST_PIXEL handler)."""
    context = bpy.context
    dir_labels = overlay_cache.get("direction_labels", [])
    op_labels = overlay_cache.get("op_labels", [])
    vel_labels = overlay_cache.get("velocity_labels", [])
    violation_labels = overlay_cache.get("violation_labels", [])
    if not dir_labels and not op_labels and not vel_labels and not violation_labels:
        return

    try:
        from bpy_extras.view3d_utils import location_3d_to_region_2d  # pyright: ignore

        region = context.region
        rv3d = context.region_data
        if not region or not rv3d:
            return

        font_id = 0
        blf.size(font_id, 24)
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.8)
        blf.shadow_offset(font_id, 1, -1)

        # Direction preview labels
        for data in dir_labels:
            center_2d = location_3d_to_region_2d(region, rv3d, data["label_pos"])
            if center_2d:
                view_mat = rv3d.view_matrix
                cam_right = Vector((view_mat[0][0], view_mat[0][1], view_mat[0][2]))
                radius = data.get("radius", 1.0)
                edge_3d = data["label_pos"] + cam_right * radius
                edge_2d = location_3d_to_region_2d(region, rv3d, edge_3d)
                if edge_2d:
                    screen_radius = abs(edge_2d.x - center_2d.x)
                else:
                    screen_radius = 60
                r, g, b, a = data["label_color"]
                d = data.get("raw_direction", Vector((0, 0, 0)))
                unit = data.get("unit", "")
                unit_str = f" {unit}" if unit else ""
                text = f"{data['label']}: {data['strength']:.1f}{unit_str} ({d.x:.1f}, {d.y:.1f}, {d.z:.1f})"
                tw, _th = blf.dimensions(font_id, text)
                blf.position(
                    font_id,
                    center_2d.x - tw / 2,
                    center_2d.y + screen_radius + 30,
                    0,
                )
                blf.color(font_id, r, g, b, a)
                blf.draw(font_id, text)

        # Operation overlay labels
        blf.size(font_id, 18)
        for data in op_labels:
            center_2d = location_3d_to_region_2d(region, rv3d, data["pos_3d"])
            if center_2d:
                r, g, b, a = data["color"]
                text = data["text"]
                tw, _th = blf.dimensions(font_id, text)
                blf.position(font_id, center_2d.x - tw / 2, center_2d.y + 15, 0)
                blf.color(font_id, r, g, b, a)
                blf.draw(font_id, text)

        # Velocity labels
        blf.size(font_id, 24)
        for data in vel_labels:
            center_2d = location_3d_to_region_2d(region, rv3d, data["pos_3d"])
            if center_2d:
                r, g, b, a = data["color"]
                text = data["text"]
                tw, _th = blf.dimensions(font_id, text)
                blf.position(font_id, center_2d.x - tw / 2, center_2d.y + 15, 0)
                blf.color(font_id, r, g, b, a)
                blf.draw(font_id, text)

        # Violation labels (bold, larger)
        blf.size(font_id, 28)
        for data in violation_labels:
            center_2d = location_3d_to_region_2d(region, rv3d, data["pos_3d"])
            if center_2d:
                r, g, b, a = data["color"]
                text = data["text"]
                tw, _th = blf.dimensions(font_id, text)
                blf.position(font_id, center_2d.x - tw / 2, center_2d.y + 20, 0)
                blf.color(font_id, r, g, b, a)
                blf.draw(font_id, text)

        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass
