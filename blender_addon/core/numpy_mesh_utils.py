# File: numpy_mesh_utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np


def extract_mesh_to_numpy(mesh):
    """Extract Blender mesh data to NumPy arrays.

    Args:
        mesh: Blender mesh object

    Returns:
        tuple: (vertices, faces) where vertices is (N, 3) and faces is (M, k)
               where k varies based on face vertex count
    """
    # Extract vertices
    vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)

    # Extract faces (polygons)
    faces = []
    for poly in mesh.polygons:
        faces.append(list(poly.vertices))

    return vertices, faces


def loop_triangle_indices(mesh):
    """Triangle vertex indices from Blender's loop-triangle tessellation.

    Returns an ``(M, 3)`` ``uint32`` array of mesh-vertex indices: exactly
    the triangulation Blender renders in the viewport
    (``mesh.loop_triangles``). This is the single source of truth for how a
    polygon mesh, including quads and N-gons, is split into triangles for
    the solver, so the geometry the artist sees matches the geometry the
    solver runs on.

    Blender's tessellation is used instead of a naive fan-from-first-vertex
    split because a fan diverges from the viewport on concave or non-planar
    N-gons (it can emit overlapping or back-facing triangles). The triangle
    indices reference the original ``mesh.vertices`` order, so pins, vertex
    groups, and per-vertex caches stay aligned with the buffer the encoder
    ships.
    """
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    tri = np.empty(n * 3, dtype=np.uint32)
    mesh.loop_triangles.foreach_get("vertices", tri)
    return tri.reshape(n, 3)


def loop_triangulate_mesh(mesh):
    """Blender loop-triangle tessellation plus aligned per-triangle UVs.

    Returns ``(tri, uv)`` where:
      * ``tri`` is the ``(M, 3)`` ``uint32`` array from
        :func:`loop_triangle_indices`.
      * ``uv`` is an ``(M, 3, 2)`` ``float32`` array of per-corner UVs taken
        from the active UV layer through each loop triangle's loop indices,
        or ``[]`` when the mesh has no active UV layer. Row ``i`` of ``uv``
        corresponds to row ``i`` of ``tri``, so the UV payload lines up with
        the face payload one entry per triangle (the decoder asserts
        ``len(uv) == len(face)``).
    """
    tri = loop_triangle_indices(mesh)
    n = len(tri)

    uv_layer = mesh.uv_layers.active
    if uv_layer is None or len(uv_layer.data) == 0:
        return tri, []

    # ``loops`` are 32-bit RNA indices, same width as the ``vertices`` read
    # above; using uint32 keeps the foreach_get fast path and the dtype
    # consistent. They are non-negative loop indices, safe as array indices.
    loops = np.empty(n * 3, dtype=np.uint32)
    mesh.loop_triangles.foreach_get("loops", loops)

    n_loops = len(uv_layer.data)
    uv_flat = np.empty(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_flat)
    uv = uv_flat.reshape(n_loops, 2)[loops.reshape(n, 3)]

    return tri, np.ascontiguousarray(uv, dtype=np.float32)
