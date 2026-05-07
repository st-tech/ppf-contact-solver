# File: previews.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ...state import iterate_active_object_groups

from .primitives import (
    _generate_arrow,
    _generate_sphere_fill,
    _generate_sphere_wireframe,
)


class DirectionPreviewManager:
    """Manages multiple direction preview visualizations in the viewport."""

    def __init__(self):
        self._entries = {}

    def add(self, key, direction, color, label="", strength=0.0, unit="", raw_direction=None):
        """Add or update a direction preview entry."""
        self._entries[key] = {
            "direction": direction,
            "color": color,
            "label": label,
            "strength": strength,
            "unit": unit,
            "raw_direction": raw_direction or direction,
        }

    def remove(self, key):
        """Remove a direction preview entry."""
        self._entries.pop(key, None)

    def clear(self):
        """Remove all entries."""
        self._entries.clear()

    def build_batches(self, radius=1.0):
        """Build GPU batches for all active entries, spaced along X to avoid overlap.

        Returns list of dicts with batches, colors, label info, and center position.
        """
        if not self._entries:
            return []

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        keys = sorted(self._entries.keys())
        n = len(keys)
        spacing = 2.5 * radius
        results = []

        for idx, key in enumerate(keys):
            entry = self._entries[key]
            offset_x = (idx - (n - 1) / 2.0) * spacing
            offset = Vector((offset_x, 0, 0))

            wire_thickness = radius * 0.005
            shaft_thickness = radius * 0.015

            # Filled sphere background translated to offset
            fill_verts = _generate_sphere_fill(radius)
            fill_verts = [v + offset for v in fill_verts]

            # Sphere wireframe (triangle-based for thickness)
            sphere_verts = _generate_sphere_wireframe(radius, thickness=wire_thickness)
            sphere_verts = [v + offset for v in sphere_verts]

            # Arrow -- length proportional to strength
            strength = entry.get("strength", 1.0)
            arrow_scale = min(strength, 20.0) / 5.0 if strength > 0 else 1.0
            shaft_verts, cone_verts = _generate_arrow(
                entry["direction"],
                shaft_length=1.2 * radius * arrow_scale,
                shaft_thickness=shaft_thickness,
                cone_length=0.15 * radius,
                cone_radius=0.06 * radius,
            )
            shaft_verts = [v + offset for v in shaft_verts]
            cone_verts = [v + offset for v in cone_verts]

            r, g, b = entry["color"]
            # Store sphere center and radius for 2D label projection
            label_pos = Vector(offset)
            results.append(
                {
                    "fill_batch": batch_for_shader(
                        shader, "TRIS", {"pos": fill_verts}
                    ),
                    "sphere_batch": batch_for_shader(
                        shader, "TRIS", {"pos": sphere_verts}
                    ),
                    "shaft_batch": batch_for_shader(
                        shader, "TRIS", {"pos": shaft_verts}
                    ),
                    "cone_batch": batch_for_shader(
                        shader, "TRIS", {"pos": cone_verts}
                    ),
                    "fill_color": (r, g, b, 0.08),
                    "sphere_color": (r, g, b, 0.15),
                    "arrow_color": (r, g, b, 0.8),
                    "label": entry["label"],
                    "strength": entry["strength"],
                    "unit": entry.get("unit", ""),
                    "raw_direction": entry.get("raw_direction", Vector((0, 0, 0))),
                    "label_pos": label_pos,
                    "radius": radius,
                    "label_color": (r, g, b, 0.9),
                }
            )
        return results


def _build_velocity_arrow_batches(scene, view_distance):
    """Build arrow + filled mesh overlay for per-object initial velocity visualization.

    Returns (batches, labels) where labels is a list of dicts for 2D text rendering.
    """
    batches = []
    labels = []
    current_frame = bpy.context.scene.frame_current
    scale = view_distance * 0.12
    thickness = scale * 0.008
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    for group in iterate_active_object_groups(scene):
        r, g, b = group.color[:3]
        arrow_color = (r, g, b, 0.8)
        fill_color = (r, g, b, 0.15)

        if not group.preview_velocity:
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            preview_kfs = [kf for kf in assigned.velocity_keyframes if kf.frame == current_frame]
            if not preview_kfs:
                continue
            from ....core.uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            if not obj or obj.type != "MESH":
                continue

            for kf in preview_kfs:
                vel_dir = Vector(kf.direction)
                strength = kf.speed
                if vel_dir.length < 1e-6 or strength < 1e-6:
                    continue
                vel_dir = vel_dir.normalized()

                mesh = obj.data
                mat = obj.matrix_world
                tris = []
                for poly in mesh.polygons:
                    verts = [mat @ mesh.vertices[vi].co for vi in poly.vertices]
                    for i in range(1, len(verts) - 1):
                        tris.extend([verts[0], verts[i], verts[i + 1]])
                if tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": tris})
                    batches.append((batch, fill_color))

                center = mat.translation.copy()
                arrow_len = scale * min(strength, 20.0) / 5.0
                shaft_tris, cone_tris = _generate_arrow(
                    vel_dir,
                    shaft_length=arrow_len,
                    shaft_thickness=thickness * 2,
                    cone_length=arrow_len * 0.18,
                    cone_radius=thickness * 5,
                )
                shaft_tris = [v + center for v in shaft_tris]
                cone_tris = [v + center for v in cone_tris]
                if shaft_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                    batches.append((batch, arrow_color))
                if cone_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                    batches.append((batch, arrow_color))

                tip_3d = center + vel_dir * arrow_len * 1.2
                d = vel_dir
                labels.append({
                    "pos_3d": tip_3d,
                    "text": f"F{kf.frame} {strength:.1f} m/s ({d.x:.1f}, {d.y:.1f}, {d.z:.1f})",
                    "color": (r, g, b, 0.9),
                })
    return batches, labels
