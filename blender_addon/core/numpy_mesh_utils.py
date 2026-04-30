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


def triangulate_numpy_mesh(vertices, faces):
    """Triangulate a mesh represented as NumPy arrays.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: List of face vertex indices (can be triangles, quads, ngons)

    Returns:
        np.ndarray: (M, 3) array of triangulated face indices
    """
    triangulated_faces = []

    for face in faces:
        if len(face) == 3:
            # Already a triangle
            triangulated_faces.append(face)
        elif len(face) == 4:
            # Quad - split into two triangles
            # Split quad [0,1,2,3] into triangles [0,1,2] and [0,2,3]
            triangulated_faces.append([face[0], face[1], face[2]])
            triangulated_faces.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # N-gon - fan triangulation from first vertex
            # Split ngon [0,1,2,...,n] into triangles [0,1,2], [0,2,3], ..., [0,n-1,n]
            for i in range(1, len(face) - 1):
                triangulated_faces.append([face[0], face[i], face[i + 1]])

    return np.array(triangulated_faces, dtype=np.uint32)


def triangulate_uv_data(mesh, triangulated_faces):
    """Extract and triangulate UV data to match triangulated faces.

    Args:
        mesh: Blender mesh object
        triangulated_faces: (M, 3) array of triangulated face indices from triangulate_numpy_mesh

    Returns:
        list: UV coordinates for each triangulated face, or empty list if no UV data
    """
    if not mesh.uv_layers.active:
        return []

    uv_layer = mesh.uv_layers.active.data
    if len(uv_layer) == 0:
        return []
    original_faces = [list(poly.vertices) for poly in mesh.polygons]

    # Build mapping from original face index to UV data
    original_uv_data = {}
    for poly_idx, poly in enumerate(mesh.polygons):
        face_uv = [uv_layer[loop_index].uv[:] for loop_index in poly.loop_indices]
        original_uv_data[poly_idx] = face_uv

    # Map triangulated faces back to original faces to get UV data
    triangulated_uv = []
    tri_face_idx = 0

    for orig_face_idx, orig_face in enumerate(original_faces):
        orig_uv = original_uv_data[orig_face_idx]

        if len(orig_face) == 3:
            # Already triangle - direct mapping
            triangulated_uv.append(np.array(orig_uv, dtype=np.float32))
            tri_face_idx += 1
        elif len(orig_face) == 4:
            # Quad split into 2 triangles: [0,1,2] and [0,2,3]
            triangulated_uv.append(
                np.array([orig_uv[0], orig_uv[1], orig_uv[2]], dtype=np.float32)
            )
            triangulated_uv.append(
                np.array([orig_uv[0], orig_uv[2], orig_uv[3]], dtype=np.float32)
            )
            tri_face_idx += 2
        elif len(orig_face) > 4:
            # N-gon fan triangulation
            for i in range(1, len(orig_face) - 1):
                triangulated_uv.append(
                    np.array([orig_uv[0], orig_uv[i], orig_uv[i + 1]], dtype=np.float32)
                )
                tri_face_idx += 1

    return triangulated_uv
