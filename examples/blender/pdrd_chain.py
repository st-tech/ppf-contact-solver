# File: pdrd_chain.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or
# `blender --python examples/blender/pdrd_chain.py`) to build a
# horizontal chain of PDRD bodies dangling from a static top link.
#
# Each chain link is a torus whose major circle has been split at
# x=0 and pulled apart along X so the round ring becomes a stadium-
# shape closed surface with its long axis horizontal. The surface
# stays watertight because the elongation only shifts existing
# vertices and the rings sitting exactly on x=0 (theta = +/- pi/2)
# are left in place. Consecutive links are rotated 90 deg about X
# so their ring planes flip between XZ and XY and the loops interlock
# the way real chain links do.
#
# The first (leftmost) link sits in a STATIC group and serves only
# as a fixed collision anchor. The remaining links sit in a PDRD
# group; at rest the chain extends horizontally along +X, and once
# the solver runs the free end swings down under gravity into the
# usual catenary curve. Each link is exactly rigid regardless of link
# tessellation.

import math

import bpy
import numpy as np


def _resolve_solver():
    # Look up the addon under whichever extension repo Blender installed
    # it into (`user_default` for Install-from-Disk, or the remote repo
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


def make_link_mesh(
    name: str,
    r_major: float = 0.025,
    r_minor: float = 0.012,
    elong: float = 0.025,
    n: int = 16,
):
    """Build a stadium-shape closed surface (torus elongated along X).

    Ring plane is XZ, long axis X (length 2 * (r_major + elong)),
    thickness 2 * r_minor along Y. With ``n`` sections in both major
    and minor directions the link carries n*n vertices and 2*n*n
    triangles.
    """
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])

    # (theta, phi) parameterization. theta runs around the major
    # circle in XZ plane (symmetry axis Y); phi runs around the minor
    # cross-section in the (radial, axial) plane at each theta.
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    verts = []
    for theta in thetas:
        ct, st = math.cos(theta), math.sin(theta)
        center = np.array([r_major * ct, 0.0, r_major * st])
        radial = np.array([ct, 0.0, st])
        axial = np.array([0.0, 1.0, 0.0])
        for phi in phis:
            cp, sp = math.cos(phi), math.sin(phi)
            v = center + r_minor * cp * radial + r_minor * sp * axial
            verts.append(tuple(v))

    # Wind quads so the surface normal points outward, which the
    # frontend's divergence-theorem volume + inertia pass requires.
    # With the (theta, phi) parameterization above the outward normal
    # is (b-a) x (d-a), giving the quad order (a, d, c, b).
    faces = []
    for ti in range(n):
        ti2 = (ti + 1) % n
        for pi in range(n):
            pi2 = (pi + 1) % n
            a = ti * n + pi
            b = ti2 * n + pi
            c = ti2 * n + pi2
            d = ti * n + pi2
            faces.append((a, d, c, b))

    # Elongate along X: pull the +X and -X halves of the major circle
    # apart. The rings at theta = +/- pi/2 sit at x=0 before the
    # shift (cos theta = 0) so they stay put and the surface stays
    # watertight.
    V = np.asarray(verts, dtype=np.float64)
    V[V[:, 0] > 0.0, 0] += elong
    V[V[:, 0] < 0.0, 0] -= elong

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(V.tolist(), [], faces)
    mesh.update()
    return mesh


def _add_link(master_mesh, idx: int, rotate_x_deg: float, location):
    mesh = master_mesh.copy()
    mesh.name = f"chain_link.{idx:02d}"
    obj = bpy.data.objects.new(f"chain_link.{idx:02d}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (math.radians(rotate_x_deg), 0.0, 0.0)
    return obj


def build(
    n_links: int = 8,
    anchor_x: float = 0.0,
    top_z: float = 1.0,
    r_major: float = 0.025,
    r_minor: float = 0.012,
    elong: float = 0.025,
):
    """Build the chain scene and assign groups.

    The total link length along its long axis is
    ``2 * (r_major + elong)``. Adjacent links overlap by ``2 * r_minor``
    (one cross-section diameter), the minimum overlap needed to keep
    them interlocked, with the center-to-center spacing set to
    ``link_length - 2 * r_minor``.
    """
    solver = _resolve_solver()
    solver.clear()

    master = make_link_mesh(
        "chain_link_master", r_major=r_major, r_minor=r_minor, elong=elong
    )

    link_length = 2.0 * (r_major + elong)
    dx = 0.95 * (link_length - 2.0 * r_minor)

    static_objs: list[bpy.types.Object] = []
    pdrd_objs: list[bpy.types.Object] = []
    for i in range(n_links):
        x = anchor_x + i * dx
        # Even-indexed links keep the master orientation (ring plane
        # XZ). Odd-indexed links rotate 90 deg about X so their ring
        # plane becomes XY, perpendicular to the neighbor's. Both
        # have long axis X, so the chain extends along X.
        rot = 90.0 if (i % 2 == 1) else 0.0
        obj = _add_link(master, i, rot, (x, 0.0, top_z))
        if i == 0:
            static_objs.append(obj)
        else:
            pdrd_objs.append(obj)

    # Static anchor: the leftmost link sits in a STATIC group, which
    # the addon realizes as a tri shell with every vertex pinned.
    # Collision dispatch still routes PDRD-vs-STATIC pairs through the
    # standard point-face / edge-edge path.
    anchor = solver.create_group("ChainAnchor", type="STATIC")
    for obj in static_objs:
        anchor.add(obj.name)
    anchor.set_overlay_color(0.4, 0.4, 0.4)

    # PDRD chain group: dangling links. Each link is exactly rigid
    # regardless of link tessellation.
    chain = solver.create_group("ChainLinks", type="PDRD")
    for obj in pdrd_objs:
        chain.add(obj.name)
    chain.param.pdrd_density = 1000.0
    chain.set_overlay_color(0.85, 0.55, 0.05)

    # Session params. The chain has many contacting links; halve the
    # step and give Newton room to converge so the solver keeps up.
    solver.param.gravity = (0.0, 0.0, -9.8)
    solver.param.step_size = 0.005
    solver.param.min_newton_steps = 32
    solver.param.frame_count = 180

    print(
        f"pdrd_chain: 1 STATIC anchor + {n_links - 1} PDRD links "
        f"(N={len(master.vertices)} verts each, "
        f"link_length={link_length:.3f} m, dx={dx:.3f} m)"
    )
    return static_objs, pdrd_objs


if __name__ == "__main__":
    build()
