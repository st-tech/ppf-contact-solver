# File: yarn.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or `blender --python yarn.py`)
# to build the same interwoven yarn fabric as examples/yarn.ipynb,
# expressed as Bezier curves combined into a single object. Each strand
# is a separate spline with its own material slot. Two ROD-group pins
# (added via the ppf_contact_solver addon) grab the open warp ends and
# stretch them apart along X.

import colorsys

import bpy
import numpy as np


def _resolve_solver():
    # Look up the addon under whichever extension repo Blender installed
    # it into (``user_default`` for Install-from-Disk, or the remote repo
    # id when installed from a hosted repository).
    import addon_utils
    import importlib
    name = next(
        (m.__name__ for m in addon_utils.modules()
         if m.__name__.endswith(".ppf_contact_solver")),
        None,
    )
    if name is None:
        raise ImportError(
            "ZOZO's Contact Solver addon not found; enable it in "
            "Preferences > Add-ons first."
        )
    return importlib.import_module(f"{name}.ops.api").solver


def make_strands(offset: float, width: float, res: float):
    offset = 1.3 * offset
    n_vertical_yarns = int(res * 0.28 / offset)
    n_points_per_seg = int(res * 20)
    n_segs = int(width * n_vertical_yarns)
    n_points = n_segs * n_points_per_seg
    dx = 1.0 / (n_points - 1)
    strands = []

    # Vertical (warp) yarns with a sinusoidal weave displacement.
    for k in range(n_vertical_yarns):
        y_base = (k / (n_vertical_yarns - 1) - 0.5
                  if n_vertical_yarns > 1 else 0.0)
        j_vals = np.arange(n_points)
        x_mod = (j_vals % n_points_per_seg) / n_points_per_seg
        t = 2.0 * np.pi * x_mod
        x_disp = -width * (0.5 / n_segs) * np.sin(2.0 * t)
        y_disp = (0.85 / n_vertical_yarns) * np.sin(t)
        z_disp = 0.75 * offset * np.cos(2.0 * t)
        xyz = np.zeros((n_points, 3))
        xyz[:, 0] = width * (2.0 * dx * j_vals - 1.0) + x_disp
        xyz[:, 1] = y_base + y_disp
        xyz[:, 2] = z_disp
        strands.append((xyz, False))

    # Horizontal selvedge loops along the two long edges.
    for pos_index in range(2):
        for k in range(n_segs - 1):
            dx_local = 2.0 * width / n_segs
            y_base = (0.5 + 0.25 * dx_local
                      if pos_index == 0 else -0.5 - 0.25 * dx_local)
            z_base = 0.15 * dx_local
            x_center = dx_local * (k + 0.77) - width
            if pos_index == 1:
                x_center += 0.5 * dx_local
            j_vals = np.arange(n_points_per_seg)
            t = 2.0 * np.pi * j_vals / n_points_per_seg
            r = 0.78 * width / n_segs
            z_val = r * np.cos(t)
            theta = 0.25 * np.pi
            xyz = np.zeros((n_points_per_seg, 3))
            xyz[:, 0] = x_center + r * np.sin(t)
            if pos_index == 0:
                xyz[:, 1] = y_base + z_val * np.sin(theta)
            else:
                xyz[:, 1] = y_base - z_val * np.sin(theta)
            xyz[:, 2] = z_base + z_val * np.cos(theta)
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


def build(offset: float = 4e-3, width: float = 0.5, res: float = 1.0,
          bevel_depth: float = 2.3e-3, object_name: str = "Yarn",
          group_name: str = "YarnStrands", colorize: bool = True,
          pin: bool = True, motion: bool = True,
          move_delta: float = -5.0,
          motion_frame_start: int = 1, motion_frame_end: int = 1000):
    solver = _resolve_solver()

    strands = make_strands(offset, width, res)

    # Golden-ratio hue increment so neighboring strands land on
    # well-separated hues instead of a smooth ramp.
    phi = 0.61803398875
    left_indices, right_indices = [], []
    point_cursor = 0

    curve = solver.create_curve(
        object_name, bevel_depth=bevel_depth,
        bevel_resolution=2, resolution_u=1,
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
        # Mirror examples/yarn.ipynb's session parameters:
        # frames=120, dt=1e-2, gravity off, MAX friction-mode.
        solver.param.frame_count = 120
        solver.param.step_size = 1e-2
        solver.param.gravity = (0.0, 0.0, 0.0)
        solver.param.friction_mode = "MAX"

        # Wipe any stale group with the same name so re-runs of the
        # script start from a clean slate.
        for g in solver.get_groups():
            if g.name == group_name:
                g.delete()
                break

        grp = solver.create_group(group_name, type="ROD")
        grp.add(object_name)
        # Mirror the notebook's rod material parameters
        # (bend=0, young-mod=1e5, contact-gap=1e-3, contact-offset=2.3e-3,
        # length-factor=0.85).
        grp.param.bend = 0.0
        grp.param.rod_young_modulus = 1e5
        grp.param.contact_gap = 1e-3
        grp.param.contact_offset = 2.3e-3
        grp.param.length_factor = 0.85

        left = grp.create_pin(object_name, "left", indices=left_indices)
        right = grp.create_pin(object_name, "right", indices=right_indices)

        if motion:
            # Match the notebook: leftmost warp ends translate by
            # move_delta along X, rightmost ends by -move_delta, pulling
            # the fabric apart. The notebook's 0..t_end=10 second span
            # at dt=1e-2 maps to frame_start..frame_end = 1..1000.
            left.move_by(delta=(move_delta, 0.0, 0.0),
                         frame_start=motion_frame_start,
                         frame_end=motion_frame_end)
            right.move_by(delta=(-move_delta, 0.0, 0.0),
                          frame_start=motion_frame_start,
                          frame_end=motion_frame_end)

    print(f"yarn: {len(strands)} splines in '{object_name}'"
          f" (pin groups: {len(left_indices)} left, {len(right_indices)} right)")
    return obj


if __name__ == "__main__":
    _resolve_solver().clear()
    build()
