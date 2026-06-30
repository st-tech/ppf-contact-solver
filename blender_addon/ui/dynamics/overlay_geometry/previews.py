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
    _compute_pca_axes,
    _generate_arrow,
    _generate_circle,
    _generate_rotation_arc,
    _generate_sphere_fill,
    _generate_sphere_wireframe,
)

# Shared strength-to-arrow scaling: clamp the physical strength and divide by a
# reference so direction-preview and velocity arrows respond identically.
_ARROW_STRENGTH_CAP = 20.0
_ARROW_STRENGTH_REF = 5.0


def _strength_to_arrow_factor(strength):
    """Return the shared arrow scale factor for a physical strength."""
    return min(strength, _ARROW_STRENGTH_CAP) / _ARROW_STRENGTH_REF


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
            arrow_scale = _strength_to_arrow_factor(strength) if strength > 0 else 1.0
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
                # Spin (angular) preview: a circle + rotation arc about the
                # chosen axis. Principal axes (PC1-3) are previewed from the
                # current mesh so the handedness matches the solver (both
                # canonicalize the eigenvector sign); World X/Y/Z and Custom
                # are fixed world-space directions.
                if kf.enable_angular and kf.angular_speed != 0.0:
                    mat = obj.matrix_world
                    verts_world = [mat @ v.co for v in obj.data.vertices]
                    if verts_world:
                        mode = kf.angular_axis
                        center = sum(verts_world, Vector((0.0, 0.0, 0.0))) / len(verts_world)
                        axis = None
                        label = mode
                        if mode in ("PC1", "PC2", "PC3"):
                            pca = _compute_pca_axes(verts_world)
                            if pca is not None:
                                center, eigvecs = pca
                                comp = {"PC1": 0, "PC2": 1, "PC3": 2}[mode]
                                axis = Vector((
                                    float(eigvecs[0, comp]),
                                    float(eigvecs[1, comp]),
                                    float(eigvecs[2, comp]),
                                ))
                        elif mode == "X":
                            axis = Vector((1.0, 0.0, 0.0))
                        elif mode == "Y":
                            axis = Vector((0.0, 1.0, 0.0))
                        elif mode == "Z":
                            axis = Vector((0.0, 0.0, 1.0))
                        elif mode == "CUSTOM":
                            axis = Vector(kf.angular_axis_custom)
                            label = "Custom"
                        if axis is not None and axis.length >= 1e-6:
                            axis = axis.normalized()
                            total = 0.0
                            for v in verts_world:
                                diff = v - center
                                total += (diff - axis * diff.dot(axis)).length
                            avg_radius = total / max(1, len(verts_world))
                            if avg_radius < 1e-4:
                                avg_radius = 0.1
                            spin_thick = max(0.002, avg_radius * 0.01)
                            spin_tris = _generate_circle(
                                center, axis, avg_radius, thickness=spin_thick,
                            ) + _generate_rotation_arc(
                                center, axis, avg_radius, kf.angular_speed,
                                thickness=spin_thick * 2,
                            )
                            if spin_tris:
                                batch = batch_for_shader(shader, "TRIS", {"pos": spin_tris})
                                batches.append((batch, arrow_color))
                            labels.append({
                                "pos_3d": center + axis * avg_radius * 1.05,
                                "text": f"F{kf.frame} ω={kf.angular_speed:.0f}°/s {label}",
                                "color": (r, g, b, 0.9),
                            })

                vel_dir = Vector(kf.direction)
                strength = kf.speed
                if not kf.enable_translational or vel_dir.length < 1e-6 or strength < 1e-6:
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
                arrow_len = scale * _strength_to_arrow_factor(strength)
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
