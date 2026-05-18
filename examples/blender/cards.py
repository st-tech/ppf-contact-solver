# File: cards.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or `blender --python cards.py`)
# to build the same house-of-cards layout as examples/cards.ipynb,
# expressed as SHELL cards struck by a SOLID icosphere via the
# ppf_contact_solver addon.
#
# Coordinate frame.
#   The notebook is Y-up:
#     - Each card sits in the YZ plane (X=0), centered on origin.
#       Y is height (0.25), Z is width (0.1875). Face normal +X.
#     - rotate(-angle, "z") rotates around the lateral-perpendicular
#       axis Z, tilting the card so its top moves in +X — a real
#       leaning card whose bottom edge stays on the floor.
#     - Wall normal +Y is the floor; gravity is along -Y.
#   This script remaps Y -> Z so cards stand vertically in Blender's
#   native Z-up. The swap inverts handedness, so a notebook rotation
#   of theta around Z becomes a Blender rotation of -theta around Y.
#   In Blender:#     - Card mesh lives in the YZ plane (X=0), width along ±Y,
#       height along ±Z, face normal +X.
#     - Left card: rotation_euler.y = +angle  (notebook's -angle / Z).
#     - Right card: rotation_euler.y = -angle (notebook's +angle / Z).
#     - Ceiling card: rotation_euler.y = +90  (notebook's -90 / Z).
#     - Floor wall normal +Z, gravity (0, 0, -9.8).
#     - Sphere starts at (-2, 0, 1) moving along +X.

import colorsys
import math

import bpy
import mathutils
import numpy as np


def make_card_mesh(name: str, res: int, width: float, height: float):
    """Subdivided rectangle in the YZ plane (X=0).

    Face normal +X, width along ±Y (±w/2), height along ±Z (±h/2).
    ``res`` is the per-side subdivision count.  Verts are centered on
    the origin in both Y and Z; placement code shifts each card so its
    rotated bottom-leftmost corner lands at the target floor position.
    """
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])
    mesh = bpy.data.meshes.new(name)
    n = res + 1
    verts = []
    for j in range(n):
        z = -height / 2 + height * j / (n - 1)
        for i in range(n):
            y = -width / 2 + width * i / (n - 1)
            verts.append((0.0, y, z))
    faces = []
    for j in range(res):
        for i in range(res):
            a = j * n + i
            b = a + 1
            c = (j + 1) * n + i + 1
            d = (j + 1) * n + i
            faces.append((a, b, c, d))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def make_sphere_mesh(name: str, radius: float, subdiv: int):
    """Icosphere mesh as an off-scene data block."""
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=radius)
    tmp = bpy.context.active_object
    mesh = tmp.data
    mesh.name = name
    bpy.data.objects.remove(tmp, do_unlink=True)
    return mesh


def _rotated_aabb(obj):
    """World-axis AABB of obj's vertices after rotation; ignores location."""
    rot = obj.rotation_euler.to_matrix()
    pts = [rot @ mathutils.Vector(v.co) for v in obj.data.vertices]
    mn = mathutils.Vector((min(p.x for p in pts),
                           min(p.y for p in pts),
                           min(p.z for p in pts)))
    mx = mathutils.Vector((max(p.x for p in pts),
                           max(p.y for p in pts),
                           max(p.z for p in pts)))
    return mn, mx


def _card_material(name: str, rgb: tuple):
    """Principled-BSDF material with the given RGB Base Color."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.45
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def _hue_for(idx: int) -> tuple:
    """Golden-ratio hue increment so neighboring cards get separated colors."""
    phi = 0.61803398875
    hue = (idx * phi) % 1.0
    sat = 0.75 + 0.25 * ((idx * 7) % 5) / 4.0
    val = 0.70 + 0.30 * ((idx * 11) % 4) / 3.0
    return colorsys.hsv_to_rgb(hue, sat, val)


def _add_card(card_mesh, idx: int, rotate_y_deg: float):
    """New card object with a single-user copy of card_mesh, rotated.

    Each card receives its own colorful material so the rows read at
    a glance in the viewport.
    """
    mesh = card_mesh.copy()
    mesh.name = f"card.{idx:03d}"
    mesh.materials.append(_card_material(f"card-mat-{idx:03d}", _hue_for(idx)))
    obj = bpy.data.objects.new(f"card.{idx:03d}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_euler = (0.0, math.radians(rotate_y_deg), 0.0)
    return obj


def build(mesh_res: int = 8, n_stack: int = 8, card_height: float = 0.25,
          gap: float = 1e-3, angle_deg: float = 25.0,
          sphere_radius: float = 0.15, sphere_subdiv: int = 3,
          sphere_velocity: float = 2.3,
          sphere_start: tuple = (-2.0, 0.0, 1.0)):
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver

    card_width = 0.75 * card_height

    card_mesh = make_card_mesh("card", mesh_res, card_width, card_height)
    sphere_mesh = make_sphere_mesh("sphere", sphere_radius, sphere_subdiv)

    cards: list[bpy.types.Object] = []
    next_idx = [0]

    def add_row(n: int, x0: float, z0: float) -> tuple[float, float]:
        x_cursor = 0.0
        pair_top = 0.0
        running_max_z = 0.0
        ceil_x: list[float] = []
        first_x_out = None
        for i in range(n):
            # Left card: notebook rotates -angle around Z; the Y/Z swap
            # inverts handedness, so Blender rotation = +angle around Y.
            left = _add_card(card_mesh, next_idx[0], +angle_deg)
            next_idx[0] += 1
            l_min, l_max = _rotated_aabb(left)
            lx = x_cursor - l_min.x + (x0 if i == 0 else 0.0)
            lz = z0 - l_min.z
            left.location = (lx, 0.0, lz)
            l_max_x_world = lx + l_max.x
            if first_x_out is None:
                first_x_out = l_max_x_world
            cards.append(left)

            # Right card: notebook +angle / Z -> Blender -angle / Y.
            right = _add_card(card_mesh, next_idx[0], -angle_deg)
            next_idx[0] += 1
            r_min, r_max = _rotated_aabb(right)
            rx = gap + l_max_x_world - r_min.x
            rz = z0 - r_min.z
            right.location = (rx, 0.0, rz)
            r_max_x_world = rx + r_max.x
            r_max_z_world = rz + r_max.z
            if i < n - 1:
                ceil_x.append(r_max_x_world)
            x_cursor = r_max_x_world + gap
            pair_top = max(pair_top, r_max_z_world + gap)
            cards.append(right)

        # Ceiling cards (notebook rotates -90 / Z -> Blender +90 / Y),
        # all anchored at pair_top so they form a single horizontal
        # layer.  Odd-indexed ones get a small gap bump in Z to avoid
        # co-planar contact with the previous ceiling.
        for i, cx in enumerate(ceil_x):
            ceil = _add_card(card_mesh, next_idx[0], +90.0)
            next_idx[0] += 1
            _, c_max = _rotated_aabb(ceil)
            cz = pair_top if i % 2 == 0 else pair_top + gap
            ceil.location = (cx, 0.0, cz)
            cards.append(ceil)
            running_max_z = max(running_max_z, cz + c_max.z)

        return first_x_out or 0.0, max(pair_top, running_max_z) + gap

    x0, z0 = -0.75, gap
    for i in reversed(range(n_stack)):
        x0, z0 = add_row(i + 1, x0, z0)

    # SHELL group bundling every card.
    shell = solver.create_group("Cards", type="SHELL")
    shell.add(*[c.name for c in cards])
    # Match the notebook: absolute contact-gap = 1e-3 m, not the
    # bounding-box-diagonal ratio the addon defaults to.
    shell.param.use_group_bounding_box_diagonal = False
    shell.param.contact_gap = gap
    shell.param.shell_young_modulus = 30000.0
    shell.param.bend = 1e6
    shell.param.friction = 0.5

    # Sphere projectile (SOLID; the server tetrahedralizes via fTetWild).
    sphere_obj = bpy.data.objects.new("sphere", sphere_mesh)
    bpy.context.scene.collection.objects.link(sphere_obj)
    sphere_obj.location = sphere_start
    sphere_group = solver.create_group("Sphere", type="SOLID")
    sphere_group.add("sphere")
    sphere_group.param.use_group_bounding_box_diagonal = False
    sphere_group.param.friction = 0.5
    sphere_group.set_velocity("sphere", direction=(1.0, 0.0, 0.0),
                              speed=sphere_velocity)

    # Floor wall at z=0 normal +Z; Blender-default gravity (0, 0, -9.8).
    solver.add_wall(position=(0.0, 0.0, 0.0),
                    normal=(0.0, 0.0, 1.0)).param.friction = 0.5

    # Session parameters mirroring the notebook.
    solver.param.step_size = 0.01
    solver.param.min_newton_steps = 16
    solver.param.friction_mode = "MAX"
    solver.param.frame_count = 180

    print(f"cards: {len(cards)} SHELL cards + 1 SOLID sphere "
          f"with initial velocity {sphere_velocity} m/s along +X")
    return cards, sphere_obj


if __name__ == "__main__":
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver
    solver.clear()
    build()
