# File: noodle.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or `blender --python noodle.py`)
# to build the same noodles-into-bowl scene as examples/noodle.ipynb,
# expressed as a single ROD group (one Bezier spline per strand) and
# an inverted hemispherical sphere collider via the ppf_contact_solver
# addon.
#
# Coordinate frame.
#   The notebook is Y-up: strands span (0, 0.01, 0) -> (0.01, 15, 0)
#   (nearly vertical along Y), grid placement uses notebook
#   (x, 0, y_grid), gravity along -Y, bowl center (0, 1, 0).
#   This script remaps notebook Y -> Blender Z so strands stand
#   vertically in Blender's native Z-up:
#     - strand template (0, 0, 0.01) -> (0.01, 0, 15)
#     - placement at Blender (x, y_grid, 0)
#     - bowl center Blender (0, 0, 1), Blender-default gravity
#     - solver.param.fix_xz = 1.0 keeps each strand's upper portion
#       (z > 1) locked laterally so the noodles dangle from "above"
#       and only the unconstrained tails settle in the bowl.

import colorsys
import math

import bpy
import numpy as np


def _noodle_material(name: str, rgb: tuple):
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
    phi = 0.61803398875
    hue = (idx * phi) % 1.0
    sat = 0.75 + 0.25 * ((idx * 7) % 5) / 4.0
    val = 0.70 + 0.30 * ((idx * 11) % 4) / 3.0
    return colorsys.hsv_to_rgb(hue, sat, val)


def build(n_grid: int = 11, scale: float = 0.05, n_segments: int = 960,
          strand_height: float = 15.0, strand_tilt: float = 0.01,
          jitter_amp: float = 1e-3,
          bowl_radius: float = 0.55, bowl_center: tuple = (0.0, 0.0, 0.4),
          bend: float = 1.5, contact_gap: float = 3e-3,
          contact_offset: float = 3e-3, friction: float = 0.05,
          bowl_friction: float = 0.0,
          frame_count: int = 600, step_size: float = 5e-3,
          min_newton_steps: int = 8, friction_mode: str = "MIN",
          fix_xz_threshold: float = 1.0, air_friction: float = 1e-4,
          object_name: str = "Noodles", group_name: str = "Noodles",
          colorize: bool = True, rng_seed: int = 0):
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver

    rng = np.random.default_rng(rng_seed)
    n_pts = n_segments + 1

    # Strand template in Blender coords: (0, 0, 0.01) -> (tilt, 0, height).
    # Mirrors the notebook's app.mesh.line endpoints after the Y/Z swap.
    t = np.linspace(0.0, 1.0, n_pts)
    template = np.stack([
        strand_tilt * t,
        np.zeros_like(t),
        0.01 + (strand_height - 0.01) * t,
    ], axis=1)

    curve = solver.create_curve(
        object_name,
        bevel_depth=contact_offset,
        bevel_resolution=2,
        resolution_u=1,
    )

    # Center the n_grid x n_grid grid on the origin in the XY plane.
    # The notebook's `i - N/2` leaves a half-step offset for odd N;
    # this uses `(N-1)/2` so the grid is symmetric about (0, 0).
    n_strands = n_grid * n_grid
    grid_center = (n_grid - 1) / 2.0
    for k in range(n_strands):
        i, j = divmod(k, n_grid)
        x = scale * (i - grid_center)
        y = scale * (j - grid_center)
        verts = template + np.array([x, y, 0.0])
        if jitter_amp > 0.0:
            verts = verts + rng.uniform(-jitter_amp, jitter_amp, size=verts.shape)
        spline_index = curve.add_spline(verts, closed=False)
        if colorize:
            curve.set_material(
                spline_index,
                _noodle_material(f"noodle-mat-{k:03d}", _hue_for(k)),
            )
    obj = curve.finalize()

    # ROD group bundling every strand.
    grp = solver.create_group(group_name, type="ROD")
    grp.add(object_name)
    grp.param.use_group_bounding_box_diagonal = False
    grp.param.bend = bend
    grp.param.contact_gap = contact_gap
    grp.param.contact_offset = contact_offset
    grp.param.friction = friction

    # Inverted hemispherical bowl collider. invert() = collide on the
    # inside surface; hemisphere() = only the lower half.
    bowl = solver.add_sphere(position=bowl_center, radius=bowl_radius).invert().hemisphere()
    bowl.param.friction = bowl_friction

    # Session parameters tuned interactively in Blender and dumped via
    # the addon's debug TCP port; differ from examples/noodle.ipynb's
    # defaults (frames=240, MAX friction, vertex_air_damp=1e-5,
    # min_newton_steps left implicit).
    solver.param.frame_count = frame_count
    solver.param.step_size = step_size
    solver.param.min_newton_steps = min_newton_steps
    solver.param.friction_mode = friction_mode
    solver.param.fix_xz = fix_xz_threshold
    solver.param.vertex_air_damp = air_friction

    print(f"noodle: {n_strands} strands x {n_pts} CPs, "
          f"bowl r={bowl_radius} at {bowl_center}, "
          f"fix_xz={fix_xz_threshold}, vertex_air_damp={air_friction}")
    return obj


if __name__ == "__main__":
    from bl_ext.user_default.ppf_contact_solver.ops.api import solver
    solver.clear()
    build()
