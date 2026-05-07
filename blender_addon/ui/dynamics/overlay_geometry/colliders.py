# File: colliders.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import math

import bpy  # pyright: ignore
import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ....models.groups import get_addon_data

from .primitives import (
    _generate_arrow,
    _generate_sphere_wireframe,
    _line_to_tris,
)


def _resolve_scene_dyn_params(state, current_frame):
    """Resolve gravity and wind at the given frame, considering dynamic parameter keyframes.

    Returns (gravity_vec, wind_direction_vec, wind_strength).
    """
    gravity = list(state.gravity_3d)
    wind_dir = list(state.wind_direction)
    wind_strength = state.wind_strength

    for dyn_item in state.dyn_params:
        if len(dyn_item.keyframes) < 2:
            continue

        # Build resolved keyframe list
        resolved = []
        for i, kf in enumerate(dyn_item.keyframes):
            if i == 0:
                if dyn_item.param_type == "GRAVITY":
                    resolved.append((kf.frame, list(state.gravity_3d), 0.0))
                elif dyn_item.param_type == "WIND":
                    resolved.append((kf.frame, list(state.wind_direction), state.wind_strength))
                else:
                    continue
            elif kf.use_hold and resolved:
                prev = resolved[-1]
                resolved.append((kf.frame, list(prev[1]), prev[2]))
            else:
                if dyn_item.param_type == "GRAVITY":
                    resolved.append((kf.frame, list(kf.gravity_value), 0.0))
                elif dyn_item.param_type == "WIND":
                    resolved.append((kf.frame, list(kf.wind_direction_value), kf.wind_strength_value))
                else:
                    continue

        if not resolved:
            continue

        # Interpolate at current_frame
        if current_frame <= resolved[0][0]:
            val, extra = resolved[0][1], resolved[0][2]
        elif current_frame >= resolved[-1][0]:
            val, extra = resolved[-1][1], resolved[-1][2]
        else:
            val, extra = resolved[0][1], resolved[0][2]
            for j in range(len(resolved) - 1):
                f0, v0, e0 = resolved[j]
                f1, v1, e1 = resolved[j + 1]
                if f0 <= current_frame <= f1:
                    dt = f1 - f0
                    w = (current_frame - f0) / dt if dt > 0 else 1.0
                    val = [v0[k] * (1 - w) + v1[k] * w for k in range(3)]
                    extra = e0 * (1 - w) + e1 * w
                    break

        if dyn_item.param_type == "GRAVITY":
            gravity = val
        elif dyn_item.param_type == "WIND":
            wind_dir = val
            wind_strength = extra

    return gravity, wind_dir, wind_strength


def _resolve_collider_state(item, current_frame):
    """Resolve collider position and radius at the given frame by interpolating keyframes.

    Frame 0 (index 0) uses base properties. Subsequent keyframes override.
    Interpolates linearly between keyframes; hold keyframes repeat previous value.

    Returns (position: Vector, radius: float).
    """
    base_pos = Vector(item.position)
    base_radius = item.radius if item.collider_type == "SPHERE" else 0.0

    if len(item.keyframes) <= 1:
        return base_pos, base_radius

    # Build resolved list: (frame, position, radius)
    resolved = []
    for i, kf in enumerate(item.keyframes):
        if i == 0:
            resolved.append((kf.frame, base_pos.copy(), base_radius))
        elif kf.use_hold and resolved:
            prev = resolved[-1]
            resolved.append((kf.frame, prev[1].copy(), prev[2]))
        else:
            resolved.append((kf.frame, Vector(kf.position), kf.radius))

    # Before first keyframe
    if current_frame <= resolved[0][0]:
        return resolved[0][1], resolved[0][2]
    # After last keyframe
    if current_frame >= resolved[-1][0]:
        return resolved[-1][1], resolved[-1][2]

    # Find surrounding keyframes and interpolate
    for i in range(len(resolved) - 1):
        f0, p0, r0 = resolved[i]
        f1, p1, r1 = resolved[i + 1]
        if f0 <= current_frame <= f1:
            dt = f1 - f0
            if dt == 0:
                return p1, r1
            w = (current_frame - f0) / dt
            pos = p0.lerp(p1, w)
            r = r0 * (1 - w) + r1 * w
            return pos, r

    return base_pos, base_radius


def _collider_hue_color(index, collider_type, saturation=0.6, value=0.85, alpha=1.0):
    """Generate a distinct color by rotating hue for each collider index.

    Walls start at blue (hue=0.6), spheres start at green (hue=0.33).
    """
    import colorsys
    base_hue = 0.6 if collider_type == "WALL" else 0.33
    hue = (base_hue + index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, alpha)


def _build_collider_batches(view_distance):
    """Build GPU batches for invisible collider previews (walls and spheres).

    All sizes are proportional to view_distance so they appear constant on screen.
    """
    batches = []
    try:
        state = get_addon_data(bpy.context.scene).state
    except Exception:
        return batches

    scale = view_distance * 0.15
    thickness = scale * 0.006
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    current_frame = bpy.context.scene.frame_current
    wall_idx = 0
    sphere_idx = 0

    for item in state.invisible_colliders:
        if not item.show_preview:
            continue
        # Honor the "Active Until (frame)" cutoff: once the timeline has
        # passed it, the encoder drops this collider from the contact set,
        # so the overlay should hide too.
        if (
            item.enable_active_duration
            and current_frame >= item.active_duration
        ):
            continue

        pos, radius = _resolve_collider_state(item, current_frame)
        if item.collider_type == "WALL":
            base_r, base_g, base_b, _ = _collider_hue_color(wall_idx, "WALL")
            wall_idx += 1
        else:
            base_r, base_g, base_b, _ = _collider_hue_color(sphere_idx, "SPHERE")
            sphere_idx += 1

        if item.collider_type == "WALL":
            normal = Vector(item.normal)
            if normal.length < 1e-9:
                continue
            normal = normal.normalized()

            # Orthonormal basis on the plane
            if abs(normal.z) < 0.9:
                tangent = normal.cross(Vector((0, 0, 1))).normalized()
            else:
                tangent = normal.cross(Vector((1, 0, 0))).normalized()
            bitangent = normal.cross(tangent).normalized()

            # Core grid on the plane (view-scaled)
            grid_size = scale
            grid_steps = 8
            tris = []
            for i in range(grid_steps + 1):
                t = -grid_size + (2 * grid_size * i / grid_steps)
                p1 = pos + tangent * t - bitangent * grid_size
                p2 = pos + tangent * t + bitangent * grid_size
                tris.extend(_line_to_tris(p1, p2, thickness))
                p1 = pos - tangent * grid_size + bitangent * t
                p2 = pos + tangent * grid_size + bitangent * t
                tris.extend(_line_to_tris(p1, p2, thickness))

            if tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": tris})
                batches.append((batch, (base_r, base_g, base_b, 0.3)))

            # Expanding dashed square outlines (infinite plane hint)
            n_rings = 4
            dash_count = 24
            for ring in range(1, n_rings + 1):
                ring_size = grid_size * (1 + ring * 0.3)
                alpha = 0.25 / (ring * 0.8)
                dash_tris = []
                # Four edges of the square, each dashed
                corners = [
                    (pos + tangent * (-ring_size) + bitangent * (-ring_size),
                     pos + tangent * ring_size + bitangent * (-ring_size)),
                    (pos + tangent * ring_size + bitangent * (-ring_size),
                     pos + tangent * ring_size + bitangent * ring_size),
                    (pos + tangent * ring_size + bitangent * ring_size,
                     pos + tangent * (-ring_size) + bitangent * ring_size),
                    (pos + tangent * (-ring_size) + bitangent * ring_size,
                     pos + tangent * (-ring_size) + bitangent * (-ring_size)),
                ]
                for edge_start, edge_end in corners:
                    edge_vec = edge_end - edge_start
                    for d in range(dash_count):
                        t0 = d / dash_count
                        t1 = (d + 0.5) / dash_count
                        p1 = edge_start + edge_vec * t0
                        p2 = edge_start + edge_vec * t1
                        dash_tris.extend(_line_to_tris(p1, p2, thickness * 0.7))
                if dash_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": dash_tris})
                    batches.append((batch, (base_r, base_g, base_b, alpha)))

            # Normal arrow (view-scaled)
            arrow_len = scale * 0.8
            shaft_tris, cone_tris = _generate_arrow(
                normal,
                shaft_length=arrow_len,
                shaft_thickness=thickness * 2,
                cone_length=arrow_len * 0.2,
                cone_radius=thickness * 5,
            )
            shaft_tris = [v + pos for v in shaft_tris]
            cone_tris = [v + pos for v in cone_tris]
            arrow_color = (base_r, base_g, base_b, 0.8)
            if shaft_tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                batches.append((batch, arrow_color))
            if cone_tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                batches.append((batch, arrow_color))

        elif item.collider_type == "SPHERE":
            segments = 24
            rings = 16

            if item.hemisphere:
                # Hemisphere = lower half-sphere + cylinder extending upward.
                # The solver projects the center Y to vertex Y for vertices
                # above the equator, creating a cylindrical extension.

                # Lower hemisphere wireframe (only bottom half: rings below equator)
                half_rings = rings // 2
                grid = []
                for i in range(half_rings + 1):
                    theta = -math.pi / 2 + math.pi / 2 * i / half_rings
                    ring_row = []
                    for j in range(segments):
                        phi = 2 * math.pi * j / segments
                        x = radius * math.cos(theta) * math.cos(phi)
                        y = radius * math.cos(theta) * math.sin(phi)
                        z = radius * math.sin(theta)
                        ring_row.append(Vector((x, y, z)))
                    grid.append(ring_row)
                hemi_tris = []
                for i in range(half_rings + 1):
                    for j in range(segments):
                        hemi_tris.extend(_line_to_tris(
                            grid[i][j], grid[i][(j + 1) % segments], thickness,
                        ))
                for j in range(segments):
                    for i in range(half_rings):
                        hemi_tris.extend(_line_to_tris(
                            grid[i][j], grid[i + 1][j], thickness,
                        ))
                hemi_tris = [v + pos for v in hemi_tris]

                # Cylinder extending upward from equator
                cyl_height = scale * 1.5
                cyl_tris = []
                for j in range(segments):
                    phi = 2 * math.pi * j / segments
                    bx = radius * math.cos(phi)
                    by = radius * math.sin(phi)
                    bottom = pos + Vector((bx, by, 0))
                    top = pos + Vector((bx, by, cyl_height))
                    # Vertical lines
                    cyl_tris.extend(_line_to_tris(bottom, top, thickness))
                # Top and bottom rings
                for z_off in (0.0, cyl_height):
                    ring_pts = []
                    for j in range(segments):
                        phi = 2 * math.pi * j / segments
                        ring_pts.append(pos + Vector((
                            radius * math.cos(phi),
                            radius * math.sin(phi),
                            z_off,
                        )))
                    for j in range(segments):
                        cyl_tris.extend(_line_to_tris(
                            ring_pts[j], ring_pts[(j + 1) % segments], thickness,
                        ))

                all_tris = hemi_tris + cyl_tris
                if all_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                    batches.append((batch, (base_r, base_g, base_b, 0.4)))
            else:
                # Full sphere wireframe
                wire_tris = _generate_sphere_wireframe(
                    radius=radius, thickness=thickness,
                    segments=segments, rings=rings,
                )
                wire_tris = [v + pos for v in wire_tris]
                if wire_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": wire_tris})
                    batches.append((batch, (base_r, base_g, base_b, 0.4)))

            # Normal arrows at 6 cardinal surface points
            # Direction: outward for normal sphere, inward for inverted
            sign = -1.0 if item.invert else 1.0
            arrow_len = scale * 0.4
            arrow_thick = thickness * 1.5
            normal_color = (base_r, base_g, base_b, 0.7)
            cardinal_dirs = [
                Vector((1, 0, 0)), Vector((-1, 0, 0)),
                Vector((0, 1, 0)), Vector((0, -1, 0)),
                Vector((0, 0, -1)),  # bottom always
            ]
            if not item.hemisphere:
                cardinal_dirs.append(Vector((0, 0, 1)))  # top only for full sphere
            for d in cardinal_dirs:
                surface_pt = pos + d * radius
                arrow_dir = d * sign
                shaft_tris, cone_tris = _generate_arrow(
                    arrow_dir,
                    shaft_length=arrow_len,
                    shaft_thickness=arrow_thick,
                    cone_length=arrow_len * 0.25,
                    cone_radius=arrow_thick * 4,
                )
                shaft_tris = [v + surface_pt for v in shaft_tris]
                cone_tris = [v + surface_pt for v in cone_tris]
                if shaft_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                    batches.append((batch, normal_color))
                if cone_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                    batches.append((batch, normal_color))

    return batches
