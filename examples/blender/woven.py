# File: woven.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or `blender --python woven.py`)
# to build the same woven cylinder pattern as examples/woven.ipynb,
# expressed as Bezier curves combined into a single object. Each strand
# is a separate spline with its own material slot. Two ROD-group pins
# (added via the ppf_contact_solver addon) fix the open ends.

import colorsys

import bpy
import numpy as np


def make_woven_cylinder(n: int, offset: float, scale: float):
    dx, width = 1.0 / (n - 1), 1.25
    scale = 2.0 * 1.48 * scale
    v_steps = int(25.0 * scale)
    sep, strands = 0.5, []

    for i in range(v_steps):
        theta = 2.0 * np.pi * i / v_steps
        xyz = np.zeros((n, 3))
        xyz[:, 0] = width * (2.0 * dx * np.arange(n) - 1.0)
        xyz[:, 1], xyz[:, 2] = sep * np.sin(theta), sep * np.cos(theta)
        strands.append((xyz, False))

    h_steps = int(30.0 * scale)
    ring_steps = v_steps * 3
    assert ring_steps % 2 == 0, "ring_steps must be even"
    amp, dx_h, half_v = 1.2 * offset, 1.0 / (h_steps - 1), v_steps // 2

    for i in range(1, h_steps - 1):
        sgn = 1.0 if (i % 2 == 0) else -1.0
        xyz = np.zeros((ring_steps, 3))
        xyz[:, 0] = width * (2.0 * dx_h * i - 1.0)
        j_indices = np.arange(ring_steps)
        theta_vals = 2.0 * np.pi * j_indices / ring_steps
        r = sep + sgn * amp * np.cos(half_v * theta_vals)
        xyz[:, 1], xyz[:, 2] = r * np.sin(theta_vals), r * np.cos(theta_vals)
        strands.append((xyz, True))

    return strands


def _strand_material(name: str, rgb: tuple):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.45
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def build(n: int = 256, offset: float = 4e-3, scale: float = 2.0,
          bevel_depth: float = 3e-3, object_name: str = "WovenCylinder",
          group_name: str = "WovenStrands", colorize: bool = True,
          pin: bool = True, motion: bool = True,
          angular_velocity: float = 365.0, move_delta: float = 0.15,
          motion_frame_start: int = 1, motion_frame_end: int = 1200):
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver

    strands = make_woven_cylinder(n, offset, scale)

    # Golden-ratio hue increment so neighboring strands land on
    # well-separated hues instead of a smooth ramp.
    phi = 0.61803398875
    left_indices, right_indices = [], []
    point_cursor = 0

    curve = solver.create_curve(
        object_name, bevel_depth=bevel_depth,
        bevel_resolution=2, resolution_u=4,
    )

    for k, (V, closed) in enumerate(strands):
        spline_index = curve.add_spline(V, closed=closed)
        if colorize:
            hue = (k * phi + (0.0 if closed else 0.37)) % 1.0
            sat = 0.75 + 0.25 * ((k * 7) % 5) / 4.0
            val = 0.70 + 0.30 * ((k * 11) % 4) / 3.0
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            curve.set_material(spline_index, _strand_material(f"mat-{k}", rgb))
        if not closed:
            left_indices.append(point_cursor)
            right_indices.append(point_cursor + len(V) - 1)
        point_cursor += len(V)

    obj = curve.finalize()

    if pin and left_indices:
        # Mirror examples/woven.ipynb's scene parameters. The addon's
        # vertex_air_damp is the analogue of the notebook's
        # "isotropic-air-friction". "MAX" friction-mode and zero gravity
        # match the notebook's twist-spin setup.
        solver.param.frame_count = 450
        solver.param.step_size = 1.0 / 120.0
        solver.param.gravity = (0.0, 0.0, 0.0)
        solver.param.vertex_air_damp = 1e-3
        solver.param.friction_mode = "MAX"

        # Wipe any stale group with the same name so re-runs of the
        # script start from a clean slate.
        for g in solver.get_groups():
            if g.name == group_name:
                g.delete()
                break

        grp = solver.create_group(group_name, type="ROD")
        grp.add(object_name)
        # Mirror the notebook's rod material parameters. The new Bezier
        # 1-per-CP sampling makes the edge length match the CP spacing
        # (~9.8mm verticals, ~7.1mm rings), so contact_offset = 2e-3
        # (tube diameter 4mm) sits well under edge_length / 2 for both
        # strand families.
        grp.param.bend = 1e-3
        grp.param.rod_young_modulus = 1e5
        grp.param.contact_gap = 1.5e-3
        grp.param.contact_offset = 2e-3
        grp.param.friction = 0.01
        grp.param.length_factor = 0.8

        left = grp.create_pin(object_name, "left", indices=left_indices)
        right = grp.create_pin(object_name, "right", indices=right_indices)

        if motion:
            # Match the notebook: each open end spins about the X axis
            # (opposite directions on the two sides) and translates
            # toward the other end, twisting the woven structure.
            left.spin(axis=(1.0, 0.0, 0.0),
                      angular_velocity=angular_velocity,
                      frame_start=motion_frame_start,
                      frame_end=motion_frame_end)
            left.move_by(delta=(move_delta, 0.0, 0.0),
                         frame_start=motion_frame_start,
                         frame_end=motion_frame_end)
            right.spin(axis=(-1.0, 0.0, 0.0),
                       angular_velocity=angular_velocity,
                       frame_start=motion_frame_start,
                       frame_end=motion_frame_end)
            right.move_by(delta=(-move_delta, 0.0, 0.0),
                          frame_start=motion_frame_start,
                          frame_end=motion_frame_end)

    print(f"woven: {len(strands)} splines in '{object_name}'"
          f" (pin groups: {len(left_indices)} left, {len(right_indices)} right)")
    return obj


if __name__ == "__main__":
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver
    solver.clear()
    build()
