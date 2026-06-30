# File: five-twist.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or `blender --python five-twist.py`)
# to build the same five-cylinder twist scene as examples/five-twist.ipynb,
# expressed as a single SHELL group with one pin per cylinder end, driven
# by the ppf_contact_solver addon.
#
# Coordinate frame.
#   The notebook is Y-up: each cylinder runs along X, the ring of five
#   sits in the YZ plane at radius r=1 around the X axis. Solver-side
#   the addon applies ``zup_to_yup`` ((Bx, By, Bz) -> (Bx, Bz, -By)) on
#   every position, so we choose the Blender placement that lands the
#   solver-frame position on the notebook's (0, cos t, sin t):
#     - notebook (0, cos t, sin t) solver Y-up
#       -> Blender (0, -sin t, cos t) Z-up
#   Cylinder axis stays along X, so per-cylinder spin/move/scale
#   directions are unchanged ([+/-1, 0, 0]).
#
# Pin animation.
#   The addon converts pin frames to solver seconds via
#   ``t = (frame - 1) / frame_rate``, so frame_rate=60 reproduces the
#   notebook's dt=1/60 timeline 1:1.  Simulation duration is
#   ``(frame_count - 1) / frame_rate`` ≈ 8 s, matching the notebook's
#   480 frames at dt=1/60.  Step size is left at the addon's max
#   (0.01 s; the FloatProperty caps it there) so the solver internally
#   substeps the 1/60 s output spacing roughly 1.67× per frame.
#     - phase 1 (frames 1..300, t = 0..5 s): both ends spin about
#       their own X axis at 360 deg/s and scale to 0.75 about the
#       world X-axis tips (-half_length, 0, 0) and (+half_length, 0, 0).
#     - move (frames 1..1500, t = 0..25 s): each tip translates by
#       +/-1 along X.  The simulation only runs 480 frames, so the
#       ramp is partial, matching the notebook (session frames=480
#       against t_end=25 s).
#     - phase 2 (frames 300..900, t = 5..15 s): both ends revolve
#       about the world X-axis at +/-180 deg/s, sweeping the ring as
#       a whole.

import colorsys
import math

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


def _cylinder_mesh_arrays(r: float, min_x: float, max_x: float, n: int):
    """Mirror frontend's app.mesh.cylinder(r, min_x, max_x, n) topology
    but emit quads instead of triangles.

    Tube only (no caps), open along +/-X. Returns (V, F, ny, n_per_ring)
    where n_per_ring = n + 1 (axial samples per ring of constant theta).
    Vertex (idx = (n+1)*j + i): axial index i in [0, n] at theta j.
    Quad winding (v0, v1, v2, v3) traverses CCW when viewed from outside
    the cylinder, so face normals point outward.
    """
    dx = (max_x - min_x) / n
    ny = int(2.0 * math.pi * r / dx)
    dy = 2.0 * math.pi / ny

    V = np.zeros(((n + 1) * ny, 3), dtype=np.float64)
    for j in range(ny):
        theta = j * dy
        s, c = math.sin(theta), math.cos(theta)
        for i in range(n + 1):
            idx = (n + 1) * j + i
            V[idx, 0] = min_x + i * dx
            V[idx, 1] = s * r
            V[idx, 2] = c * r

    F: list[tuple[int, int, int, int]] = []
    for j in range(ny):
        j_next = (j + 1) % ny
        for i in range(n):
            v0 = (n + 1) * j + i
            v1 = (n + 1) * j + i + 1
            v2 = (n + 1) * j_next + i + 1
            v3 = (n + 1) * j_next + i
            F.append((v0, v1, v2, v3))
    return V, F, ny, n + 1


def _cylinder_material(name: str, rgb: tuple):
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
    sat = 0.85
    val = 0.95
    return colorsys.hsv_to_rgb(hue, sat, val)


def _ensure_template_mesh(name: str, V, F):
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in V], [], F)
    mesh.update()
    return mesh


def _make_cylinder_object(template_mesh, idx: int, location,
                          subsurf_viewport: int = 1, subsurf_render: int = 2):
    if f"cyl.{idx:03d}" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[f"cyl.{idx:03d}"], do_unlink=True)
    mesh = template_mesh.copy()
    mesh.name = f"cyl.{idx:03d}"
    mesh.materials.append(_cylinder_material(f"cyl-mat-{idx:03d}", _hue_for(idx)))
    obj = bpy.data.objects.new(f"cyl.{idx:03d}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    # Cosmetic smoothing only: the solver reads obj.data.vertices, so the
    # subsurf modifier output is ignored at transfer/sim time.
    subsurf = obj.modifiers.new(name="Subdivision", type="SUBSURF")
    subsurf.levels = subsurf_viewport
    subsurf.render_levels = subsurf_render
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def _add_vertex_group(obj, name: str, indices):
    vg = obj.vertex_groups.new(name=name)
    vg.add(list(indices), 1.0, "REPLACE")
    return vg


def build(n: int = 180, n_cylinders: int = 5, ring_radius: float = 1.0,
          cylinder_radius: float = 0.55, half_length: float = 2.6,
          angular_velocity: float = 360.0, move_delta: float = 1.0,
          scale_factor: float = 0.75,
          t_wait: float = 5.0, t_move_end: float = 25.0,
          t_spin2_end: float = 15.0,
          bend: float = 500.0, contact_gap: float = 4e-3,
          young_modulus: float = 1e4, poisson_ratio: float = 0.25,
          density: float = 3.5, friction: float = 0.0,
          step_size: float = 0.01, frame_count: int = 480,
          fps: int = 60, jitter_amp: float = 1e-2,
          group_name: str = "Cylinders", rng_seed: int = 0):
    solver = _resolve_solver()

    rng = np.random.default_rng(rng_seed)

    V, F, ny, n_per_ring = _cylinder_mesh_arrays(
        cylinder_radius, -half_length, half_length, n,
    )
    template = _ensure_template_mesh("cyl-template", V, F)

    # i==0 ring is the -X end (left), i==n ring is the +X end (right).
    left_indices = [(n_per_ring) * j + 0 for j in range(ny)]
    right_indices = [(n_per_ring) * j + n for j in range(ny)]

    cylinders: list[bpy.types.Object] = []
    for i in range(n_cylinders):
        t = 2.0 * i / n_cylinders * math.pi
        # Pick Blender (Z-up) coords whose ``zup_to_yup`` image matches
        # the notebook's solver-frame (0, r*cos t, r*sin t).
        y_b = -ring_radius * math.sin(t)
        z_b = ring_radius * math.cos(t)
        jitter = rng.uniform(-jitter_amp, jitter_amp, size=3) if jitter_amp > 0 else (0.0, 0.0, 0.0)
        loc = (0.0 + jitter[0], y_b + jitter[1], z_b + jitter[2])
        obj = _make_cylinder_object(template, i, loc)
        _add_vertex_group(obj, "left", left_indices)
        _add_vertex_group(obj, "right", right_indices)
        cylinders.append(obj)

    grp = solver.create_group(group_name, type="SHELL")
    grp.add(*[obj.name for obj in cylinders])
    grp.param.use_group_bounding_box_diagonal = False
    grp.param.contact_gap = contact_gap
    grp.param.shell_density = density
    grp.param.shell_young_modulus = young_modulus
    grp.param.shell_poisson_ratio = poisson_ratio
    # Shell bend is density-normalized in the solver, so divide by density to
    # reproduce the pre-normalization drape (this example uses density 3.5).
    grp.param.bend = bend / density
    grp.param.friction = friction

    f_wait = max(2, int(round(t_wait * fps)))
    f_move = max(f_wait + 1, int(round(t_move_end * fps)))
    f_spin2 = max(f_wait + 1, int(round(t_spin2_end * fps)))

    # The addon's pin op builders move each new op to position 0, so the
    # stored (= solver-evaluated) order is the REVERSE of the call order.
    # The notebook evaluates ops in call order [spin1, move_by, scale,
    # spin2]; to land on the same stored order in the addon, call them
    # in reverse: spin2, scale, move_by, spin1.
    for obj in cylinders:
        left_pin = grp.create_pin(obj.name, "left")
        # spin2: phase-2 revolution about the world X-axis.
        left_pin.spin(axis=(1.0, 0.0, 0.0),
                      angular_velocity=angular_velocity / 2.0,
                      center=(0.0, 0.0, 0.0),
                      center_mode="ABSOLUTE",
                      frame_start=f_wait, frame_end=f_spin2)
        # scale: collapse the left ring toward the world X-axis tip
        # (-half_length, 0, 0). Notebook's local-frame center
        # (-x-half_length, -y, -z) with obj.position (0, y, z) resolves
        # to the same world point.
        left_pin.scale(factor=scale_factor,
                       center=(-half_length, 0.0, 0.0),
                       center_mode="ABSOLUTE",
                       frame_start=1, frame_end=f_wait)
        # move_by: ramp the tip out along +X to t_move_end (the sim
        # truncates at frame_count).
        left_pin.move_by(delta=(move_delta, 0.0, 0.0),
                         frame_start=1, frame_end=f_move)
        # spin1: twist about the cylinder's own X axis. CENTROID picks
        # the left-ring centroid (= (-half_length, 0, 0) in untranslated
        # solver coords), which is on the cylinder's own axis.
        left_pin.spin(axis=(1.0, 0.0, 0.0),
                      angular_velocity=angular_velocity,
                      frame_start=1, frame_end=f_wait)

        right_pin = grp.create_pin(obj.name, "right")
        right_pin.spin(axis=(-1.0, 0.0, 0.0),
                       angular_velocity=angular_velocity / 2.0,
                       center=(0.0, 0.0, 0.0),
                       center_mode="ABSOLUTE",
                       frame_start=f_wait, frame_end=f_spin2)
        right_pin.scale(factor=scale_factor,
                        center=(half_length, 0.0, 0.0),
                        center_mode="ABSOLUTE",
                        frame_start=1, frame_end=f_wait)
        right_pin.move_by(delta=(-move_delta, 0.0, 0.0),
                          frame_start=1, frame_end=f_move)
        right_pin.spin(axis=(-1.0, 0.0, 0.0),
                       angular_velocity=angular_velocity,
                       frame_start=1, frame_end=f_wait)

    solver.param.step_size = step_size
    solver.param.frame_count = frame_count
    solver.param.frame_rate = fps
    solver.param.gravity = (0.0, 0.0, 0.0)
    solver.param.friction_mode = "MAX"

    print(f"five-twist: {n_cylinders} SHELL cylinders, "
          f"ring r={ring_radius}, frames={frame_count} "
          f"@ dt={step_size:.4f}s, friction-mode=MAX")
    return cylinders


if __name__ == "__main__":
    _resolve_solver().clear()
    build()
