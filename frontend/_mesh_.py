# File: _mesh_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import platform
import sys
import threading
import time

from typing import Optional

import numpy as np

from numba import njit, prange

from ._bvh_ import compute_frame_mapping as _compute_frame_mapping
from ._bvh_ import interpolate_surface as _interpolate_surface

# =============================================================================
# Numba-optimized mesh generation helpers
# =============================================================================


@njit(cache=True)
def _generate_grid_faces(length_split: int, width_split: int, out_faces: np.ndarray):
    """Generate faces for a grid mesh wrapped along the length axis (used by Mobius strips)."""
    idx = 0
    for i in range(length_split):
        next_i = (i + 1) % length_split
        for j in range(width_split - 1):
            v0 = i * width_split + j
            v1 = i * width_split + j + 1
            v2 = next_i * width_split + j
            v3 = next_i * width_split + j + 1
            out_faces[idx, 0] = v0
            out_faces[idx, 1] = v2
            out_faces[idx, 2] = v1
            out_faces[idx + 1, 0] = v1
            out_faces[idx + 1, 1] = v2
            out_faces[idx + 1, 2] = v3
            idx += 2


@njit(cache=True)
def _generate_rect_faces(
    res_x: int, res_y: int, out_faces: np.ndarray
):
    """Generate faces for a rectangle mesh with alternating diagonal pattern."""
    idx = 0
    for j in range(res_y - 1):
        for i in range(res_x - 1):
            v0 = i * res_y + j
            v1 = v0 + 1
            v2 = v0 + res_y
            v3 = v2 + 1
            if (i % 2) == (j % 2):
                out_faces[idx, 0] = v0
                out_faces[idx, 1] = v1
                out_faces[idx, 2] = v3
                out_faces[idx + 1, 0] = v0
                out_faces[idx + 1, 1] = v3
                out_faces[idx + 1, 2] = v2
            else:
                out_faces[idx, 0] = v0
                out_faces[idx, 1] = v1
                out_faces[idx, 2] = v2
                out_faces[idx + 1, 0] = v1
                out_faces[idx + 1, 1] = v3
                out_faces[idx + 1, 2] = v2
            idx += 2


@njit(cache=True)
def _generate_cylinder_verts(
    n: int, ny: int, min_x: float, dx: float, dy: float, r: float, out_verts: np.ndarray
):
    """Generate vertices for a cylinder mesh."""
    for j in range(ny):
        theta = j * dy
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for i in range(n + 1):
            idx = (n + 1) * j + i
            out_verts[idx, 0] = min_x + i * dx
            out_verts[idx, 1] = sin_t * r
            out_verts[idx, 2] = cos_t * r


@njit(cache=True)
def _generate_cylinder_faces(n: int, ny: int, out_faces: np.ndarray):
    """Generate faces for a cylinder mesh."""
    for j in range(ny):
        for i in range(n):
            idx = j * n + i
            v0 = (n + 1) * j + i
            v1 = (n + 1) * j + i + 1
            v2 = (n + 1) * ((j + 1) % ny) + (i + 1)
            v3 = (n + 1) * ((j + 1) % ny) + i
            if (i % 2) == (j % 2):
                out_faces[2 * idx, 0] = v1
                out_faces[2 * idx, 1] = v2
                out_faces[2 * idx, 2] = v0
                out_faces[2 * idx + 1, 0] = v3
                out_faces[2 * idx + 1, 1] = v0
                out_faces[2 * idx + 1, 2] = v2
            else:
                out_faces[2 * idx, 0] = v0
                out_faces[2 * idx, 1] = v1
                out_faces[2 * idx, 2] = v3
                out_faces[2 * idx + 1, 0] = v2
                out_faces[2 * idx + 1, 1] = v3
                out_faces[2 * idx + 1, 2] = v1


@njit(parallel=True, cache=True)
def _transform_verts_2d(
    verts: np.ndarray, ex: np.ndarray, ey: np.ndarray, out_verts: np.ndarray
):
    """Transform 2D grid vertices using basis vectors."""
    n = len(verts)
    for i in prange(n):
        x, y = verts[i, 0], verts[i, 1]
        out_verts[i, 0] = ex[0] * x + ey[0] * y
        out_verts[i, 1] = ex[1] * x + ey[1] * y
        out_verts[i, 2] = ex[2] * x + ey[2] * y


def create_mobius(
    length_split: int = 70,
    width_split: int = 15,
    twists: int = 1,
    r: float = 1.0,
    flatness: float = 1.0,
    width: float = 1.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a Mobius strip mesh.

    Args:
        length_split: Number of segments along the length of the strip.
        width_split: Number of segments across the width of the strip.
        twists: Number of half-twists in the strip.
        r: Radius of the strip (distance from center to middle of strip).
        flatness: Controls the z-extent of the strip.
        width: Width of the strip.
        scale: Overall scale factor applied to the mesh.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
    """
    # Parameter ranges
    u = np.linspace(0, 2 * np.pi, length_split, endpoint=False)
    v = np.linspace(-width / 2, width / 2, width_split)

    # Create mesh grid
    U, V = np.meshgrid(u, v, indexing="ij")

    # Parametric equations for Mobius strip
    # x = (r + v * cos(twists * u / 2)) * cos(u)
    # y = (r + v * cos(twists * u / 2)) * sin(u)
    # z = flatness * v * sin(twists * u / 2)
    half_twist = twists * U / 2
    cos_half = np.cos(half_twist)
    sin_half = np.sin(half_twist)

    x = (r + V * cos_half) * np.cos(U) * scale
    y = (r + V * cos_half) * np.sin(U) * scale
    z = flatness * V * sin_half * scale

    # Reshape to vertex array
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Generate faces using numba-optimized function
    n_faces = 2 * length_split * (width_split - 1)
    faces = np.zeros((n_faces, 3), dtype=np.int32)
    _generate_grid_faces(length_split, width_split, faces)

    return vertices, faces


def _fix_skinny_triangles(
    vert: np.ndarray, tri: np.ndarray, min_angle_deg: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fix skinny triangles by merging vertices of very short edges.

    Skinny triangles have very small angles which cause numerical issues
    in tetrahedralization. This function detects edges opposite to small
    angles and merges their vertices.

    Args:
        vert: Vertex positions (N, 3)
        tri: Triangle indices (M, 3)
        min_angle_deg: Minimum allowed angle in degrees

    Returns:
        Fixed (vertices, triangles) tuple
    """
    min_angle_rad = np.radians(min_angle_deg)
    cos_max = np.cos(min_angle_rad)  # cos decreases as angle increases

    max_iterations = 10
    for _ in range(max_iterations):
        # Compute edge vectors for each triangle
        v0, v1, v2 = vert[tri[:, 0]], vert[tri[:, 1]], vert[tri[:, 2]]
        e0 = v1 - v0  # edge opposite to vertex 2
        e1 = v2 - v1  # edge opposite to vertex 0
        e2 = v0 - v2  # edge opposite to vertex 1

        # Compute edge lengths
        len0 = np.linalg.norm(e0, axis=1)
        len1 = np.linalg.norm(e1, axis=1)
        len2 = np.linalg.norm(e2, axis=1)

        # Avoid division by zero
        len0 = np.maximum(len0, 1e-12)
        len1 = np.maximum(len1, 1e-12)
        len2 = np.maximum(len2, 1e-12)

        # Compute angles using dot product: cos(angle) = -e_i . e_j / (|e_i| |e_j|)
        # Angle at vertex 0: between edges -e2 and e0
        cos_a0 = np.sum((-e2) * e0, axis=1) / (len2 * len0)
        # Angle at vertex 1: between edges -e0 and e1
        cos_a1 = np.sum((-e0) * e1, axis=1) / (len0 * len1)
        # Angle at vertex 2: between edges -e1 and e2
        cos_a2 = np.sum((-e1) * e2, axis=1) / (len1 * len2)

        # Clamp to valid range
        cos_a0 = np.clip(cos_a0, -1, 1)
        cos_a1 = np.clip(cos_a1, -1, 1)
        cos_a2 = np.clip(cos_a2, -1, 1)

        # Find triangles with small angles (cos > cos_max means angle < min_angle)
        small_at_0 = cos_a0 > cos_max
        small_at_1 = cos_a1 > cos_max
        small_at_2 = cos_a2 > cos_max

        # Find edges to collapse (opposite to small angles)
        # Small angle at vertex 0 -> collapse edge 1 (between v1 and v2)
        # Small angle at vertex 1 -> collapse edge 2 (between v2 and v0)
        # Small angle at vertex 2 -> collapse edge 0 (between v0 and v1)
        edges_to_merge = []
        for i in range(len(tri)):
            if small_at_0[i]:
                edges_to_merge.append((tri[i, 1], tri[i, 2]))
            if small_at_1[i]:
                edges_to_merge.append((tri[i, 2], tri[i, 0]))
            if small_at_2[i]:
                edges_to_merge.append((tri[i, 0], tri[i, 1]))

        if not edges_to_merge:
            break

        # Build vertex merge map using union-find
        parent = np.arange(len(vert))

        def find(x, parent=parent):
            if parent[x] != x:
                parent[x] = find(parent[x], parent)
            return parent[x]

        def union(x, y, parent=parent):
            px, py = find(x, parent), find(y, parent)
            if px != py:
                parent[px] = py

        for v_a, v_b in edges_to_merge:
            union(v_a, v_b)

        # Compress paths
        for i in range(len(parent)):
            parent[i] = find(i)

        # Create new vertex indices
        unique_parents, inverse = np.unique(parent, return_inverse=True)
        new_vert = vert[unique_parents]

        # Remap triangles
        new_tri = inverse[tri]

        # Remove degenerate triangles (where two or more vertices are the same)
        valid = (
            (new_tri[:, 0] != new_tri[:, 1])
            & (new_tri[:, 1] != new_tri[:, 2])
            & (new_tri[:, 2] != new_tri[:, 0])
        )
        new_tri = new_tri[valid]

        vert, tri = new_vert, new_tri

    return vert, tri


class MeshManager:
    """Mesh manager for accessing mesh creation functions.

    Example:
        Reach into the mesh manager via the top-level app object::

            app = App.create("demo")
            V, F = app.mesh.square(res=64)
            V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
            V, E = app.mesh.line([0, 0, 0], [1, 0, 0], n=32)
    """

    def __init__(self, cache_dir: str):
        """Initialize the mesh manager."""
        self._cache_dir = cache_dir
        self._create = CreateManager(cache_dir)

    @property
    def create(self) -> "CreateManager":
        """Get the mesh creation manager.

        Example:
            Access the creation manager to build a parametric mesh::

                V, F = app.mesh.create.square(res=64)
                app.asset.add.tri("sheet", V, F)
        """
        return self._create

    def export(self, V: np.ndarray, F: np.ndarray, path: str):
        """Export a mesh given by vertices ``V`` and faces ``F`` to a file.

        Example:
            Save a generated square as an OBJ on disk::

                V, F = app.mesh.square(res=32)
                app.mesh.export(V, F, "sheet.obj")
        """
        import trimesh

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh.export(path)

    def line(self, _p0: list[float], _p1: list[float], n: int) -> "Rod":
        """Create a line mesh with the given start and end points and resolution.

        Args:
            _p0 (list[float]): the start point of the line
            _p1 (list[float]): the end point of the line
            n (int): the number of segments; the line has ``n + 1`` vertices

        Returns:
            Rod: a line mesh, a pair of vertices and edges

        Example:
            Create a tall vertical rod and register it as an asset::

                V, E = app.mesh.line([0, 0.01, 0], [0.01, 15, 0], 960)
                app.asset.add.rod("strand", V, E)
        """
        p0, p1 = np.array(_p0), np.array(_p1)
        vert = np.vstack([p0 + (p1 - p0) * i / n for i in range(n + 1)])
        edge = np.array([[i, i + 1] for i in range(n)])
        return self.create.rod(vert, edge)

    def box(self, width: float = 1, height: float = 1, depth: float = 1) -> "TriMesh":
        """Create a box mesh.

        Args:
            width (float): the width of the box
            height (float): the height of the box
            depth (float): the depth of the box

        Returns:
            TriMesh: a box mesh, a pair of vertices and triangles

        Example:
            Build a unit box and subdivide it once::

                V, F = app.mesh.box(width=1, height=1, depth=1).subdivide(n=1)
                app.asset.add.tri("box", V, F)
        """
        V = np.array(
            [
                [-width / 2, -height / 2, -depth / 2],
                [width / 2, -height / 2, -depth / 2],
                [-width / 2, height / 2, -depth / 2],
                [width / 2, height / 2, -depth / 2],
                [-width / 2, -height / 2, depth / 2],
                [width / 2, -height / 2, depth / 2],
                [-width / 2, height / 2, depth / 2],
                [width / 2, height / 2, depth / 2],
            ]
        )
        F = np.array(
            [
                [0, 2, 3],
                [0, 3, 1],  # Front face
                [4, 5, 7],
                [4, 7, 6],  # Back face
                [0, 1, 5],
                [0, 5, 4],  # Bottom face
                [2, 6, 7],
                [2, 7, 3],  # Top face
                [0, 4, 6],
                [0, 6, 2],  # Left face
                [1, 3, 7],
                [1, 7, 5],  # Right face
            ]
        )
        return TriMesh.create(V, F, self._cache_dir)

    def tet_box(
        self, width: float = 1, height: float = 1, depth: float = 1
    ) -> "TetMesh":
        """Create a box tetrahedral mesh directly without subdivision.

        Unlike ``box().tetrahedralize()``, this creates a minimal tetrahedral mesh
        with only 8 vertices and 5 tetrahedra, preserving the original box shape
        without any surface subdivision.

        Args:
            width (float): width of the box (x-axis)
            height (float): height of the box (y-axis)
            depth (float): depth of the box (z-axis)

        Returns:
            TetMesh: a tetrahedral box mesh

        Example:
            Register a minimal tet box (8 verts, 5 tets) as a tet asset::

                V, F, T = app.mesh.tet_box(width=1, height=1, depth=1)
                app.asset.add.tet("block", V, F, T)
        """
        V = np.array(
            [
                [-width / 2, -height / 2, -depth / 2],  # 0
                [width / 2, -height / 2, -depth / 2],  # 1
                [-width / 2, height / 2, -depth / 2],  # 2
                [width / 2, height / 2, -depth / 2],  # 3
                [-width / 2, -height / 2, depth / 2],  # 4
                [width / 2, -height / 2, depth / 2],  # 5
                [-width / 2, height / 2, depth / 2],  # 6
                [width / 2, height / 2, depth / 2],  # 7
            ]
        )
        # 12 triangles for the 6 faces
        F = np.array(
            [
                [0, 2, 3],
                [0, 3, 1],  # Front face
                [4, 5, 7],
                [4, 7, 6],  # Back face
                [0, 1, 5],
                [0, 5, 4],  # Bottom face
                [2, 6, 7],
                [2, 7, 3],  # Top face
                [0, 4, 6],
                [0, 6, 2],  # Left face
                [1, 3, 7],
                [1, 7, 5],  # Right face
            ]
        )
        # 5 tetrahedra decomposition of a cube
        # Using diagonal decomposition: one central tet + 4 corner tets
        T = np.array(
            [
                [0, 1, 3, 5],  # corner tet
                [0, 3, 2, 6],  # corner tet
                [0, 5, 4, 6],  # corner tet
                [3, 5, 6, 7],  # corner tet
                [0, 3, 5, 6],  # central tet
            ]
        )
        return TetMesh((V, F, T))

    def rectangle(
        self,
        res_x: int = 32,
        width: float = 2,
        height: float = 1,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
        gen_uv: bool = True,
    ) -> "TriMesh":
        """Create a rectangle mesh with the given resolution, width, and height, spanned by the vectors ``ex`` and ``ey``.

        Args:
            res_x (int): resolution along the ``ex`` axis
            width (float): the width of the rectangle
            height (float): the height of the rectangle
            ex (list[float] | None): a 3D vector to span the rectangle. Defaults to ``[1, 0, 0]``.
            ey (list[float] | None): a 3D vector to span the rectangle. Defaults to ``[0, 1, 0]``.
            gen_uv (bool): if True, include UV coordinates in vertices (Nx5), otherwise Nx3

        Returns:
            TriMesh: a rectangle mesh, a pair of vertices (Nx5: x,y,z,u,v or Nx3: x,y,z) and triangles

        Example:
            Create a tall thin ribbon lying in the xz plane::

                V, F = app.mesh.rectangle(
                    res_x=4, width=0.15, height=12.0,
                    ex=[1, 0, 0], ey=[0, 0, 1],
                )
                app.asset.add.tri("ribbon", V, F)
        """
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        ratio = height / width
        res_y = int(res_x * ratio)
        size_x, size_y = width, width * (res_y / res_x)
        dx = min(size_x / (res_x - 1), size_y / (res_y - 1))
        x = -size_x / 2 + dx * np.arange(res_x)
        y = -size_y / 2 + dx * np.arange(res_y)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X_flat, Y_flat = X.flatten(), Y.flatten()
        Z_flat = np.full_like(X_flat, 0)
        grid_vert = np.vstack((X_flat, Y_flat, Z_flat)).T
        _ex = np.ascontiguousarray(ex, dtype=np.float64)
        _ey = np.ascontiguousarray(ey, dtype=np.float64)
        # Transform vertices using numba
        vert = np.zeros((len(grid_vert), 3), dtype=np.float64)
        _transform_verts_2d(grid_vert, _ex, _ey, vert)
        if gen_uv:
            u_coords = vert @ _ex
            v_coords = vert @ _ey
            vert_with_uv = np.zeros((len(vert), 5))
            vert_with_uv[:, :3] = vert
            vert_with_uv[:, 3] = u_coords
            vert_with_uv[:, 4] = v_coords
        else:
            vert_with_uv = vert
        # Generate faces using numba
        n_faces = 2 * (res_x - 1) * (res_y - 1)
        tri = np.zeros((n_faces, 3), dtype=np.int32)
        _generate_rect_faces(res_x, res_y, tri)
        return TriMesh.create(vert_with_uv, tri, self._cache_dir)

    def square(
        self,
        res: int = 32,
        size: float = 2,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
        gen_uv: bool = True,
    ) -> "TriMesh":
        """Create a square mesh with the given resolution and size, spanned by the vectors ``ex`` and ``ey``.

        Args:
            res (int): resolution of the mesh
            size (float): the side length of the square
            ex (list[float] | None): a 3D vector to span the square. Defaults to ``[1, 0, 0]``.
            ey (list[float] | None): a 3D vector to span the square. Defaults to ``[0, 1, 0]``.
            gen_uv (bool): if True, include UV coordinates in vertices (Nx5), otherwise Nx3

        Returns:
            TriMesh: a square mesh, a pair of vertices and triangles

        Example:
            Build a 128-resolution sheet in the xz plane and register it::

                V, F = app.mesh.square(res=128, ex=[1, 0, 0], ey=[0, 0, 1])
                app.asset.add.tri("sheet", V, F)
        """
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        return self.rectangle(res, size, size, ex, ey, gen_uv)

    def circle(self, n: int = 32, r: float = 1, ntri: int = 1024) -> "TriMesh":
        """Create a circle mesh.

        Args:
            n (int): number of boundary segments
            r (float): radius of the circle
            ntri (int): approximate number of triangles filling the circle

        Returns:
            TriMesh: a circle mesh, a pair of 2D vertices and triangles

        Example:
            Produce a circular patch ready for plotting::

                V, F = app.mesh.circle(n=64, r=1, ntri=2048)
                app.plot.create().tri(V, F)
        """
        pts = []
        for i in range(n):
            t = 2 * np.pi * i / n
            x, y = r * np.cos(t), r * np.sin(t)
            pts.append([x, y])
        return self.create.tri(np.array(pts)).triangulate(ntri)

    def icosphere(self, r: float = 1, subdiv_count: int = 3) -> "TriMesh":
        """Create an icosphere mesh with the given radius and subdivision count.

        Args:
            r (float): radius of the icosphere
            subdiv_count (int): number of subdivision iterations applied to the base icosahedron

        Returns:
            TriMesh: an icosphere mesh, a pair of vertices and triangles

        Example:
            Create a smooth sphere asset and pin it as a static collider::

                V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
                app.asset.add.tri("sphere", V, F)
        """
        # Create icosahedron base
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
        verts_list = [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]
        # Normalize to unit sphere
        norm = np.sqrt(1 + phi * phi)
        verts_list = [[v[0] / norm, v[1] / norm, v[2] / norm] for v in verts_list]

        faces_list = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        # Subdivide
        for _ in range(subdiv_count):
            edge_midpoints = {}
            new_faces = []
            for tri in faces_list:
                a, b, c = tri[0], tri[1], tri[2]
                # Get or create midpoints for each edge
                edges = [(a, b), (b, c), (c, a)]
                mids = []
                for v0, v1 in edges:
                    edge_key = (min(v0, v1), max(v0, v1))
                    if edge_key not in edge_midpoints:
                        # Compute midpoint and project to sphere
                        p0, p1 = verts_list[v0], verts_list[v1]
                        mid = [
                            (p0[0] + p1[0]) / 2,
                            (p0[1] + p1[1]) / 2,
                            (p0[2] + p1[2]) / 2,
                        ]
                        length = np.sqrt(mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2)
                        mid = [mid[0] / length, mid[1] / length, mid[2] / length]
                        edge_midpoints[edge_key] = len(verts_list)
                        verts_list.append(mid)
                    mids.append(edge_midpoints[edge_key])
                m0, m1, m2 = mids  # midpoints of ab, bc, ca
                new_faces.extend(
                    [
                        [a, m0, m2],
                        [m0, b, m1],
                        [m2, m1, c],
                        [m0, m1, m2],
                    ]
                )
            faces_list = new_faces

        verts = np.array(verts_list, dtype=np.float64) * r
        faces = np.array(faces_list, dtype=np.int32)
        return TriMesh.create(verts, faces, self._cache_dir)

    def _from_trimesh(self, trimesh_mesh) -> "TriMesh":
        """Load a mesh from a ``trimesh`` object."""
        return TriMesh.create(
            np.asarray(trimesh_mesh.vertices),
            np.asarray(trimesh_mesh.faces),
            self._cache_dir,
        )

    def cylinder(self, r: float, min_x: float, max_x: float, n: int):
        """Create a cylinder along the x-axis.

        Args:
            r (float): radius of the cylinder
            min_x (float): minimum x coordinate
            max_x (float): maximum x coordinate
            n (int): number of divisions along the x-axis

        Returns:
            tuple: ``(V, F)`` where:
                - ``V``: ndarray of shape (N, 3) containing vertex positions
                - ``F``: ndarray of shape (M, 3) containing triangle indices

        Example:
            Create a cylinder and plot it directly::

                V, F = app.mesh.cylinder(r=0.5, min_x=-1, max_x=1, n=32)
                app.plot.create().tri(V, F)
        """
        dx = (max_x - min_x) / n
        ny = int(2.0 * np.pi * r / dx)
        dy = 2.0 * np.pi / ny
        n_vert = (n + 1) * ny

        # Generate vertices using numba
        V = np.zeros((n_vert, 3), dtype=np.float64)
        _generate_cylinder_verts(n, ny, min_x, dx, dy, r, V)

        # Generate faces using numba
        F = np.zeros((2 * n * ny, 3), dtype=np.int32)
        _generate_cylinder_faces(n, ny, F)

        return V, F

    def cone(
        self,
        Nr: int = 16,
        Ny: int = 16,
        Nb: int = 4,
        radius: float = 0.5,
        height: float = 2,
        sharpen: float = 1.0,
    ) -> "TriMesh":
        """Create a cone mesh with the given radial, vertical, and bottom resolution, radius, and height.

        Args:
            Nr (int): radial resolution
            Ny (int): vertical resolution
            Nb (int): bottom resolution
            radius (float): radius of the cone
            height (float): height of the cone
            sharpen (float): exponent applied to the radial parameter, sharpening the tip

        Returns:
            TriMesh: a cone mesh, a pair of vertices and triangles

        Example:
            Generate a pointed cone and register it::

                V, F = app.mesh.cone(radius=0.5, height=2, sharpen=1.5)
                app.asset.add.tri("cone", V, F)
        """
        V = [[0, 0, height], [0, 0, 0]]
        T = []
        ind_btm_center = 0
        ind_tip = 1
        offset = []
        offset_btm = len(V)

        for k in reversed(range(Ny)):
            if k > 0:
                r = k / (Ny - 1)
                r = r**sharpen
                offset.append(len(V))
                for i in range(Nr):
                    t = 2 * np.pi * i / Nr
                    x, y = radius * r * np.cos(t), radius * r * np.sin(t)
                    V.append([x, y, height * r])

        for j in offset[0:-1]:
            for i in range(Nr):
                ind00, ind10 = i, (i + 1) % Nr
                ind01, ind11 = ind00 + Nr, ind10 + Nr
                if i % 2 == 0:
                    T.append([ind00 + j, ind01 + j, ind10 + j])
                    T.append([ind10 + j, ind01 + j, ind11 + j])
                else:
                    T.append([ind00 + j, ind11 + j, ind10 + j])
                    T.append([ind00 + j, ind01 + j, ind11 + j])

        j = offset[-1]
        for i in range(Nr):
            ind0, ind1 = i, (i + 1) % Nr
            T.append([ind0 + j, ind_tip, ind1 + j])

        offset = []
        for k in reversed(range(Nb)):
            if k > 0:
                r = k / Nb
                offset.append(len(V))
                for i in range(Nr):
                    t = 2 * np.pi * i / Nr
                    x, y = radius * r * np.cos(t), radius * r * np.sin(t)
                    V.append([x, y, height])

        for j in offset[0:-1]:
            for i in range(Nr):
                ind00, ind10 = i, (i + 1) % Nr
                ind01, ind11 = ind00 + Nr, ind10 + Nr
                if i % 2 == 0:
                    T.append([ind00 + j, ind10 + j, ind01 + j])
                    T.append([ind10 + j, ind11 + j, ind01 + j])
                else:
                    T.append([ind00 + j, ind10 + j, ind11 + j])
                    T.append([ind00 + j, ind11 + j, ind01 + j])

        j = offset[-1]
        for i in range(Nr):
            ind0, ind1 = i, (i + 1) % Nr
            T.append([ind0 + j, ind1 + j, ind_btm_center])

        j0, j1 = offset_btm, offset[0]
        for i in range(Nr):
            ind00, ind10 = i + j0, (i + 1) % Nr + j0
            ind01, ind11 = i + j1, (i + 1) % Nr + j1
            if i % 2 == 0:
                T.append([ind00, ind10, ind01])
                T.append([ind10, ind11, ind01])
            else:
                T.append([ind00, ind10, ind11])
                T.append([ind00, ind11, ind01])

        return TriMesh.create(np.array(V), np.array(T), self._cache_dir)

    def torus(self, r: float = 1, R: float = 0.25, n: int = 32) -> "TriMesh":
        """Create a torus mesh with the given major and minor radii and resolution.

        Args:
            r (float): major radius of the torus (passed as ``major_radius``)
            R (float): minor radius of the torus (passed as ``minor_radius``)
            n (int): number of sections used for both the major and minor loops

        Returns:
            TriMesh: a torus mesh, a pair of vertices and triangles

        Example:
            Create a torus to use as a fixed obstacle::

                V, F = app.mesh.torus(r=0.5, R=0.125)
                app.asset.add.tri("torus", V, F)
        """
        import trimesh.creation

        mesh = trimesh.creation.torus(
            major_radius=r, minor_radius=R, major_sections=n, minor_sections=n
        )
        return self._from_trimesh(mesh)

    def mobius(
        self,
        length_split: int = 70,
        width_split: int = 15,
        twists: int = 1,
        r: float = 1,
        flatness: float = 1,
        width: float = 1,
        scale: float = 1,
    ) -> "TriMesh":
        """Create a Mobius strip mesh with the given length split, width split, twists, radius, flatness, width, and scale.

        Args:
            length_split (int): number of segments along the length of the strip
            width_split (int): number of segments across the width of the strip
            twists (int): number of half-twists in the strip
            r (float): radius of the strip (distance from center to middle of strip)
            flatness (float): controls the z-extent of the strip
            width (float): width of the strip
            scale (float): overall scale factor applied to the mesh

        Returns:
            TriMesh: a Mobius strip mesh, a pair of vertices and triangles

        Example:
            Create a Mobius strip and plot it::

                V, F = app.mesh.mobius(length_split=120, width_split=20, twists=1)
                app.plot.create().tri(V, F)
        """
        V, F = create_mobius(
            length_split, width_split, twists, r, flatness, width, scale
        )
        return TriMesh.create(V, F, self._cache_dir)

    def load_tri(self, path: str) -> "TriMesh":
        """Load a triangle mesh from a file.

        Args:
            path (str): path to the mesh file

        Returns:
            TriMesh: a triangle mesh, a pair of vertices and triangles

        Example:
            Load a ribbon mesh from disk and register it as a tri asset::

                V, F = app.mesh.load_tri("fishingknot.ply")
                app.asset.add.tri("ribbon", V, F)
        """
        import trimesh

        mesh = trimesh.load_mesh(path, process=False)
        return self._from_trimesh(mesh)

    def make_cache_dir(self):
        """Create the cache directory if it does not already exist.

        Example:
            Ensure the mesh cache directory is present before loading
            presets or importing files::

                app.mesh.make_cache_dir()
                V, F = app.mesh.preset("armadillo")
        """
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def preset(self, name: str) -> "TriMesh":
        """Load a preset mesh, downloading it from a remote source on first use and caching it locally.

        Args:
            name (str): the name of the preset mesh. Available names are ``armadillo``, ``knot``, and ``bunny``.

        Returns:
            TriMesh: a preset mesh, a pair of vertices and triangles

        Raises:
            ValueError: if ``name`` is not a recognized preset.
            Exception: if the mesh cannot be downloaded after multiple retries.

        Example:
            Load the armadillo, decimate it, then tetrahedralize and normalize::

                V, F, T = (
                    app.mesh.preset("armadillo")
                    .decimate(19000)
                    .tetrahedralize()
                    .normalize()
                )
                app.plot.create().tet(V, T)
        """
        import ssl
        import urllib.request

        import certifi
        import trimesh

        cache_name = os.path.join(self._cache_dir, f"preset__{name}.npz")
        if os.path.exists(cache_name):
            data = np.load(cache_name)
            return TriMesh.create(data["vert"], data["tri"], self._cache_dir)

        # Map preset names to filenames
        mesh_files = {
            "armadillo": "ArmadilloMesh",
            "knot": "KnotMesh",
            "bunny": "BunnyMesh",
        }
        if name not in mesh_files:
            raise ValueError(
                f"Unknown preset: {name}. Available: {list(mesh_files.keys())}"
            )

        # Use downloads subdirectory within cache_dir
        downloads_dir = os.path.join(self._cache_dir, "downloads")
        os.makedirs(downloads_dir, exist_ok=True)

        url = f"https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/{mesh_files[name]}.ply"
        temp_path = os.path.join(downloads_dir, f"{mesh_files[name]}.ply")

        # Create SSL context with certifi certificates (for Windows embedded Python)
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Download with retry logic
        num_try, max_try, success, wait_time = 0, 5, False, 3
        while num_try < max_try:
            try:
                with (
                    urllib.request.urlopen(url, context=ssl_context) as response,
                    open(temp_path, "wb") as out_file,
                ):
                    out_file.write(response.read())
                success = True
                break
            except Exception as e:
                num_try += 1
                print(
                    f"Mesh {name} could not be downloaded: {e}. Retrying... in {wait_time} seconds"
                )
                time.sleep(wait_time)

        if not success:
            raise Exception(f"Mesh {name} could not be downloaded")

        # Load mesh using trimesh
        mesh = trimesh.load_mesh(temp_path, process=False)
        vert = np.asarray(mesh.vertices)
        tri = np.asarray(mesh.faces)

        # Cache the result
        self.make_cache_dir()
        np.savez(cache_name, vert=vert, tri=tri)

        return TriMesh.create(vert, tri, self._cache_dir)


class CreateManager:
    """A manager that provides mesh creation functions.

    This manager provides a set of functions to create various
    types of meshes, such as rods, triangles, and tetrahedra.

    Example:
        Wrap raw numpy arrays into the mesh types expected by the solver::

            V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            E = np.array([[0, 1], [1, 2]])
            rod = app.mesh.create.rod(V, E)

            pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            tri_mesh = app.mesh.create.tri(pts).triangulate(1024)

    """

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

    def rod(self, vert: np.ndarray, edge: np.ndarray) -> "Rod":
        """Create a rod mesh.

        Args:
            vert (np.ndarray): array of vertex positions
            edge (np.ndarray): array of edges

        Returns:
            Rod: a rod mesh, a pair of vertices and edges

        Example:
            Build a custom rod from hand-specified vertices and edges::

                V = np.array([[0, 0, 0], [0.5, 0.2, 0], [1, 0, 0]])
                E = np.array([[0, 1], [1, 2]], dtype=np.uint32)
                rod = app.mesh.create.rod(V, E)
                app.asset.add.rod("strand", *rod)
        """
        return Rod((vert, edge))

    def tri(self, vert: np.ndarray, elm: np.ndarray = None) -> "TriMesh":
        """Create a triangle mesh.

        When ``elm`` is ``None`` or empty, a closed edge loop over all vertices is
        auto-generated instead of triangle faces. The resulting mesh's element
        array then stores edges rather than triangles.

        Args:
            vert (np.ndarray): array of vertex positions
            elm (np.ndarray): array of elements (``None`` or empty to auto-generate an edge loop)

        Returns:
            TriMesh: a triangle (or line-loop) mesh bound to the manager's cache directory

        Example:
            Make a closed 2D curve and triangulate it into a filled patch::

                pts = np.array(
                    [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
                )
                V, F = app.mesh.create.tri(pts).triangulate(target=1024)
                app.plot.create().tri(V, F)
        """
        if elm is None or elm.size == 0:
            cnt = vert.shape[0]
            elm = np.array([[i, (i + 1) % cnt] for i in range(cnt)])
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(self._cache_dir)

    def tet(self, vert: np.ndarray, elm: np.ndarray, tet: np.ndarray) -> "TetMesh":
        """Create a tetrahedral mesh.

        Args:
            vert (np.ndarray): array of vertex positions
            elm (np.ndarray): array of surface triangle elements
            tet (np.ndarray): array of tetrahedral elements

        Returns:
            TetMesh: a tetrahedral mesh containing vertices, surface triangles, and tetrahedra

        Example:
            Wrap precomputed tet arrays into a TetMesh and register it::

                V, F, T = app.mesh.tet_box(1, 1, 1)
                tet = app.mesh.create.tet(V, F, T)
                app.asset.add.tet("block", *tet)
        """
        return TetMesh((vert, elm, tet))


def bbox(vert) -> np.ndarray:
    """Compute the axis-aligned bounding box extents of a mesh.

    Given an array of vertices, this function returns the extent of the mesh
    along each axis.

    Args:
        vert (np.ndarray): array of vertex positions

    Returns:
        np.ndarray: a 3-element array ``[width, height, depth]`` giving the extents along x, y, and z
    """
    width = np.max(vert[:, 0]) - np.min(vert[:, 0])
    height = np.max(vert[:, 1]) - np.min(vert[:, 1])
    depth = np.max(vert[:, 2]) - np.min(vert[:, 2])
    return np.array([width, height, depth])


def normalize(vert: np.ndarray):
    """Normalize a set of vertices.

    Centers the vertices at the origin and rescales them so that the largest
    bounding box extent is 1.

    Args:
        vert (np.ndarray): array of vertex positions

    Returns:
        np.ndarray: a new, normalized copy of the vertex array
    """
    vert = vert.copy()
    vert -= np.mean(vert, axis=0)
    vert /= np.max(bbox(vert))
    return vert


def scale(
    vert: np.ndarray, scale_x: float, scale_y: float, scale_z: float
) -> np.ndarray:
    """Scale a set of vertices around their centroid.

    Scales the vertices around their centroid so that the centroid position
    is preserved.

    Args:
        vert (np.ndarray): array of vertex positions
        scale_x (float): scaling factor for the x-axis
        scale_y (float): scaling factor for the y-axis
        scale_z (float): scaling factor for the z-axis

    Returns:
        np.ndarray: a new, scaled copy of the vertex array
    """
    vert = vert.copy()
    mean = np.mean(vert, axis=0)
    vert -= mean
    vert *= np.array([scale_x, scale_y, scale_z])
    vert += mean
    return vert


class Rod(tuple[np.ndarray, np.ndarray]):
    """A class representing a rod mesh.

    A rod mesh is a pair of vertices and edges. The first element of the tuple
    is the array of vertices and the second element is the array of edges.

    Example:
        Obtain a rod from the line helper and unpack it::

            rod = app.mesh.line([0, 0, 0], [1, 0, 0], n=16)
            V, E = rod
            app.asset.add.rod("strand", V, E)

    """

    def normalize(self) -> "Rod":
        """Normalize the rod mesh in place so that the maximum bounding box extent is 1.

        Example:
            Normalize a freshly built rod and then scale it::

                rod = app.mesh.line([0, 0, 0], [10, 0, 0], n=32).normalize().scale(0.5)
                V, E = rod
        """
        # normalize() returns a copy; write back in-place since tuple[0] is
        # a numpy array bound to this tuple subclass.
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "Rod":
        """Scale the rod mesh in place with the given scaling factors.

        Args:
            scale_x (float): scaling factor for the x-axis
            scale_y (float | None): scaling factor for the y-axis. If ``None``, ``scale_x`` is used.
            scale_z (float | None): scaling factor for the z-axis. If ``None``, ``scale_x`` is used.

        Returns:
            Rod: this rod with scaled vertices

        Example:
            Shrink a rod uniformly to a quarter of its original length::

                rod = app.mesh.line([0, 0, 0], [4, 0, 0], n=32).scale(0.25)
                V, E = rod
        """
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        self[0][:] = scale(self[0], scale_x, scale_y, scale_z)
        return self


class TetMesh(tuple[np.ndarray, np.ndarray, np.ndarray]):
    """A class representing a tetrahedral mesh.

    A tetrahedral mesh is a triple of vertices, surface triangles, and tetrahedra.

    Attributes:
        surface_map: Optional tuple of ``(tri_indices, coefs)`` for reconstructing
            the original surface via frame-embedding interpolation. ``coefs`` are
            the three frame coefficients ``(c1, c2, c3)`` per original vertex,
            where ``c1`` and ``c2`` are affine coordinates in the triangle's edge
            basis, and ``c3`` is the signed normal offset in absolute world units.

    Example:
        Tetrahedralize a surface mesh and unpack the result::

            tet = app.mesh.icosphere(r=0.35, subdiv_count=4).tetrahedralize()
            V, F, T = tet
            app.asset.add.tet("sphere", V, F, T)

    """

    def normalize(self) -> "TetMesh":
        """Return ``self`` after invoking :func:`normalize` on the vertex array.

        Example:
            Normalize the armadillo tet mesh in one chain::

                V, F, T = (
                    app.mesh.preset("armadillo").decimate(19000).tetrahedralize().normalize()
                )
        """
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TetMesh":
        """Scale the tet mesh with the given scaling factors.

        Args:
            scale_x (float): scaling factor for the x-axis
            scale_y (float | None): scaling factor for the y-axis. If ``None``, ``scale_x`` is used.
            scale_z (float | None): scaling factor for the z-axis. If ``None``, ``scale_x`` is used.

        Returns:
            TetMesh: this tet mesh (``self``)

        Example:
            Squash a tet mesh along the y axis before registering it::

                tet = app.mesh.tet_box(1, 1, 1).scale(1.0, 0.25, 1.0)
                V, F, T = tet
        """
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        scale(self[0], scale_x, scale_y, scale_z)
        return self

    def set_surface_mapping(
        self,
        tri_indices: np.ndarray,
        coefs: np.ndarray,
    ) -> "TetMesh":
        """Set the frame-embedding surface mapping used by :meth:`interpolate_surface`.

        Args:
            tri_indices: closest triangle in the tet surface per original vertex, shape ``(N,)``
            coefs: frame coefficients ``(c1, c2, c3)`` per original vertex, shape ``(N, 3)``

        Returns:
            TetMesh: this tet mesh with the surface mapping attached

        Example:
            Attach a precomputed mapping back onto a tet mesh::

                tet = app.mesh.create.tet(V, F, T).set_surface_mapping(tri_idx, coefs)
                orig = tet.interpolate_surface()
        """
        self.surface_map = (tri_indices, coefs)
        return self

    def has_surface_mapping(self) -> bool:
        """Check if this TetMesh has surface mapping data.

        Example:
            Guard a call to :meth:`interpolate_surface` with this check::

                tet = surface_mesh.tetrahedralize()
                if tet.has_surface_mapping():
                    orig = tet.interpolate_surface()
        """
        return hasattr(self, "surface_map") and self.surface_map is not None

    def interpolate_surface(
        self, deformed_vert: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Reconstruct original-resolution positions from the deformed tet surface.

        Uses the stored frame-embedding mapping ``p' = x0' + c1*b1' + c2*b2' + c3*n_hat'``.

        Args:
            deformed_vert: deformed tet mesh vertices. If ``None``, the current vertices are used.

        Returns:
            np.ndarray: reconstructed vertex positions matching the original surface vertex count.

        Raises:
            ValueError: if no surface mapping has been set on this mesh.

        Example:
            Recover the original-resolution surface from a deformed tet mesh::

                tet = surface_mesh.tetrahedralize()
                deformed = tet[0] * 2.0
                orig_surface = tet.interpolate_surface(deformed)
                app.plot.create().tri(orig_surface, surface_mesh[1])
        """
        if not self.has_surface_mapping():
            raise ValueError("No surface mapping available.")

        if deformed_vert is None:
            deformed_vert = self[0]

        tri_indices, coefs = self.surface_map
        surf_tri = self[1]

        return _interpolate_surface(deformed_vert, surf_tri, tri_indices, coefs)


class TriMesh(tuple[np.ndarray, np.ndarray]):
    """A class representing a triangle mesh.

    A triangle mesh is a pair of vertices and triangles.

    Example:
        Create a sphere, decimate it, then tetrahedralize it in one chain::

            tet = (
                app.mesh.icosphere(r=1.0, subdiv_count=4)
                .decimate(2000)
                .tetrahedralize()
            )
            V, F, T = tet

    """

    @staticmethod
    def create(vert: np.ndarray, elm: np.ndarray, cache_dir: str) -> "TriMesh":
        """Create a triangle mesh, recompute its hash, and bind the given cache directory.

        Example:
            Typically invoked automatically by the mesh pipeline. To build
            a TriMesh directly from raw arrays::

                import numpy as np
                from frontend._mesh_ import TriMesh

                mesh = TriMesh.create(V, F, cache_dir="/tmp/ppf-cache")
                V2, F2 = mesh
        """
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(cache_dir)

    def _make_trimesh(self):
        """Build a ``trimesh.Trimesh`` object from this mesh's vertices and faces."""
        import trimesh

        return trimesh.Trimesh(vertices=self[0], faces=self[1], process=False)

    def export(self, path):
        """Export the mesh to a file using ``trimesh``.

        The output format is inferred from the file extension of ``path``
        (for example ``.ply``, ``.obj``, ``.stl``).

        Args:
            path (str): export path

        Example:
            Save an icosphere as a PLY file::

                app.mesh.icosphere(r=1.0, subdiv_count=3).export("sphere.ply")
        """
        mesh = self._make_trimesh()
        mesh.export(path)

    def decimate(self, target_tri: int) -> "TriMesh":
        """Reduce the number of triangles in the mesh to the target count.

        Uses quadric decimation via ``trimesh``, followed by a cleanup pass
        that merges skinny triangles. Results are cached on disk.

        Args:
            target_tri (int): target number of triangles (must be less than the current count)

        Returns:
            TriMesh: a decimated mesh

        Example:
            Decimate the armadillo preset down to 19000 triangles::

                tet = app.mesh.preset("armadillo").decimate(19000).tetrahedralize()
        """
        assert target_tri < self[1].shape[0]
        cache_path = self.compute_cache_path(f"decimate__{target_tri}")
        cached = self.load_cache(cache_path)
        if cached is None:
            if self[1].shape[1] != 3:
                raise Exception("Only triangle meshes are supported")
            mesh = self._make_trimesh()
            mesh = mesh.simplify_quadric_decimation(face_count=target_tri)
            vert = np.asarray(mesh.vertices)
            tri = np.asarray(mesh.faces)

            # Fix skinny triangles by merging close vertices
            vert, tri = _fix_skinny_triangles(vert, tri)

            return TriMesh.create(
                vert,
                tri,
                self.cache_dir,
            ).save_cache(cache_path)
        else:
            return cached

    def subdivide(self, n: int = 1, method: str = "midpoint"):
        """Subdivide the mesh with the given number of iterations and method.

        Results are cached on disk.

        Args:
            n (int): number of subdivision iterations
            method (str): subdivision method. Available methods are ``"midpoint"`` and ``"loop"``.

        Returns:
            TriMesh: a subdivided mesh

        Raises:
            Exception: if the mesh is not a triangle mesh or ``method`` is unknown.

        Example:
            Subdivide a coarse box once using Loop subdivision::

                V, F = app.mesh.box(1, 1, 1).subdivide(n=1, method="loop")
                app.plot.create().tri(V, F)
        """
        import trimesh.remesh

        cache_path = self.compute_cache_path(f"subdiv__{method}__{n}")
        cached = self.load_cache(cache_path)
        if cached is None:
            if self[1].shape[1] != 3:
                raise Exception("Only triangle meshes are supported")
            if method == "midpoint":
                # trimesh.remesh.subdivide performs midpoint subdivision
                vertices, faces = self[0], self[1]
                for _ in range(n):
                    vertices, faces = trimesh.remesh.subdivide(vertices, faces)[:2]
            elif method == "loop":
                mesh = self._make_trimesh()
                mesh = mesh.subdivide_loop(iterations=n)
                vertices, faces = mesh.vertices, mesh.faces
            else:
                raise Exception(f"Unknown subdivision method {method}")
            return TriMesh.create(
                np.asarray(vertices),
                np.asarray(faces),
                self.cache_dir,
            ).save_cache(cache_path)
        else:
            return cached

    def _compute_area(self, pts: np.ndarray) -> float:
        """Compute the area of a closed 2D polygon using the shoelace formula."""
        assert pts.shape[1] == 2
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.abs(np.dot(x, y_next) - np.dot(x_next, y))
        return area

    def triangulate(self, target: int = 1024, min_angle: float = 20) -> "TriMesh":
        """Triangulate a closed 2D line shape.

        The current mesh must be a 2D line mesh (vertices with shape ``(N, 2)``
        and segment elements of shape ``(M, 2)``). Results are cached on disk.

        Args:
            target (int): target number of triangles (used to derive a maximum triangle area)
            min_angle (float): minimum triangle angle in degrees passed to the triangulator

        Returns:
            TriMesh: a triangulated mesh

        Raises:
            Exception: if the element array is not a line mesh.

        Example:
            Turn a closed 2D curve into a filled triangle mesh::

                pts = np.array(
                    [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
                )
                V, F = app.mesh.create.tri(pts).triangulate(target=1024)
        """
        area = 1.6 * self._compute_area(self[0]) / target
        cache_path = self.compute_cache_path(f"triangulate__{area}_{min_angle}")
        cached = self.load_cache(cache_path)
        if cached is None:
            from triangle import triangulate

            if self[1].shape[1] != 2:
                raise Exception("Only line meshes are supported")

            a_str = f"{area:.100f}".rstrip("0").rstrip(".")
            t = triangulate(
                {"vertices": self[0], "segments": self[1]}, f"pa{a_str}q{min_angle}"
            )
            return TriMesh.create(
                t["vertices"], t["triangles"], self.cache_dir
            ).save_cache(cache_path)
        else:
            return cached

    def tetrahedralize(self, *args, **kwargs) -> TetMesh:
        """Tetrahedralize a surface triangle mesh using fTetWild.

        The original surface is preserved via a frame-embedding mapping,
        allowing interpolation of deformed positions back to the original
        surface structure. fTetWild is invoked in a subprocess so that other
        Python threads remain responsive. Results are cached on disk.

        Keyword Args:
            edge_length_fac (float): tetrahedral edge length as a fraction of the bbox diagonal (default: ``0.05``).
            optimize (bool): whether to optimize the mesh (default: ``True``).
            epsilon (float): fTetWild ``epsilon`` tolerance (optional).
            stop_energy (float): fTetWild stop-energy threshold (optional).
            num_opt_iter (int): number of optimization iterations (optional).
            simplify (bool): whether to simplify the input surface (optional).
            coarsen (bool): whether to coarsen the tet mesh (optional).
            status_callback (callable | None): called periodically with a status string while the subprocess runs.
            status_interval (float): polling interval in seconds for ``status_callback`` (default: ``5.0``).

        Returns:
            TetMesh: a tetrahedral mesh with surface mapping attached. Use
            :meth:`TetMesh.interpolate_surface` to recover deformed positions in
            the original surface structure.

        Raises:
            RuntimeError: if the fTetWild subprocess exits with a non-zero status.

        Example:
            Tetrahedralize a sphere and recover the original surface after
            simulation via :meth:`TetMesh.interpolate_surface`::

                tet = app.mesh.icosphere(r=0.35, subdiv_count=4).tetrahedralize(
                    edge_length_fac=0.05
                )
                V, F, T = tet
                app.asset.add.tet("sphere", V, F, T)
        """
        status_callback = kwargs.pop("status_callback", None)
        status_interval = float(kwargs.pop("status_interval", 5.0))
        arg_str = "_".join([str(a) for a in args])
        if len(kwargs) > 0:
            arg_str += "_".join([f"{k}={v}" for k, v in kwargs.items()])
        cache_path = self.compute_cache_path(
            f"{self.hash}_tetrahedralize_{arg_str}.npz"
        )
        # Bumped when the surface-mapping math changes in an incompatible way.
        # v2: switched from in-plane barycentric weights to frame-embedding
        # coefs (fixes tet-surface shrink after simulation).
        _SURFACE_MAP_VERSION = 2

        if os.path.exists(cache_path):
            if status_callback is not None:
                status_callback("loading cached tetra mesh")
            data = np.load(cache_path)
            tri = data["tri"] if "tri" in data else self[1]
            tet_mesh = TetMesh((data["vert"], tri, data["tet"]))
            # Restore surface mapping if the cached one matches the current math.
            cached_ver = int(data["map_version"]) if "map_version" in data else 0
            if cached_ver == _SURFACE_MAP_VERSION and "map_coefs" in data:
                tet_mesh.set_surface_mapping(
                    data["map_tri_indices"], data["map_coefs"]
                )
            elif "map_tri_indices" in data:
                # Legacy cache (v1 bary weights or older): recompute the
                # mapping in-place with the new math rather than dropping it.
                new_tri_indices, new_coefs = _compute_frame_mapping(
                    self[0], data["vert"], tri
                )
                tet_mesh.set_surface_mapping(new_tri_indices, new_coefs)
                np.savez(
                    cache_path,
                    vert=data["vert"],
                    tet=data["tet"],
                    tri=tri,
                    map_tri_indices=new_tri_indices,
                    map_coefs=new_coefs,
                    map_version=_SURFACE_MAP_VERSION,
                )
            return tet_mesh
        else:
            import pytetwild

            # Whitelist pytetwild kwargs we expose via fTetWild UI overrides;
            # anything outside this set is dropped to keep the subprocess
            # script stable against stray keys from callers.
            _FTETWILD_KWARGS = (
                "edge_length_fac",
                "epsilon",
                "stop_energy",
                "num_opt_iter",
                "optimize",
                "simplify",
                "coarsen",
            )
            ftw_kwargs = {k: kwargs[k] for k in _FTETWILD_KWARGS if k in kwargs}
            ftw_kwargs.setdefault("edge_length_fac", 0.05)
            ftw_kwargs.setdefault("optimize", True)
            kwargs_literal = ", ".join(f"{k}={v!r}" for k, v in ftw_kwargs.items())

            # Run fTetWild in a subprocess to avoid holding the GIL.
            # This allows other Python threads (server connection handlers)
            # to respond to status queries during tetrahedralization.
            import subprocess as _sp
            import tempfile as _tf
            import pickle as _pk

            with _tf.NamedTemporaryFile(suffix=".npz", delete=False) as _in_f:
                input_path = _in_f.name
                np.savez(_in_f, vert=self[0], tri=self[1])
            output_path = input_path + ".out.npz"

            tet_script = f"""
import sys, os, numpy as np
data = np.load({input_path!r})
V, F = data["vert"], data["tri"]
import pytetwild
if sys.platform == "win32":
    vert, tet = pytetwild.tetrahedralize(V, F, {kwargs_literal})
else:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        vert, tet = pytetwild.tetrahedralize(V, F, {kwargs_literal})
    finally:
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        os.close(devnull_fd)
        os.close(old1)
        os.close(old2)
if os.path.exists("__tracked_surface.stl"):
    os.remove("__tracked_surface.stl")
np.savez({output_path!r}, vert=vert, tet=tet)
"""
            proc = _sp.Popen(
                [sys.executable, "-c", tet_script],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            start_time = time.time()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=status_interval)
                except _sp.TimeoutExpired:
                    pass
                if proc.poll() is None and status_callback is not None:
                    elapsed = time.time() - start_time
                    status_callback(f"running fTetWild ({elapsed:.0f}s elapsed)")

            if proc.returncode != 0:
                os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                raise RuntimeError(f"fTetWild subprocess failed (exit code {proc.returncode})")

            # NpzFile keeps the .npz open (a ZipFile handle); on Windows
            # that blocks the os.unlink below with WinError 32. Use `with`
            # so the handle is closed before we try to delete the file.
            with np.load(output_path) as out_data:
                vert = out_data["vert"]
                tet = out_data["tet"]
            os.unlink(input_path)
            os.unlink(output_path)

            # Filter out degenerate tetrahedra (volume < threshold)
            # These cause float32 underflow in the Rust solver
            min_volume = 1e-15
            v0 = vert[tet[:, 0]]
            v1 = vert[tet[:, 1]]
            v2 = vert[tet[:, 2]]
            v3 = vert[tet[:, 3]]
            e0 = v1 - v0
            e1 = v2 - v0
            e2 = v3 - v0
            volumes = np.abs(np.einsum("ij,ij->i", np.cross(e0, e1), e2)) / 6.0
            valid_mask = volumes >= min_volume
            tet = tet[valid_mask]

            # Extract surface triangles from tetrahedra (vectorized)
            # A surface face is shared by exactly one tetrahedron
            n_tets = len(tet)
            face_combos = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            opp_indices = np.array([3, 2, 1, 0])

            # Gather face vertices: shape (n_tets, 4, 3) -> (n_tets*4, 3)
            all_faces = tet[:, face_combos].reshape(-1, 3)
            # Gather opposite vertices: shape (n_tets*4,)
            all_opp = tet[:, opp_indices].ravel()

            # Sort each face for uniqueness comparison
            sorted_faces = np.sort(all_faces, axis=1)

            # Find unique faces and their counts
            # Encode each sorted face as a single int for fast comparison
            max_v = int(sorted_faces.max()) + 1
            face_keys = (
                sorted_faces[:, 0].astype(np.int64) * max_v * max_v
                + sorted_faces[:, 1].astype(np.int64) * max_v
                + sorted_faces[:, 2].astype(np.int64)
            )
            _, inverse, counts = np.unique(
                face_keys, return_inverse=True, return_counts=True
            )

            # Surface faces appear exactly once
            surface_mask = counts[inverse] == 1
            surface_faces_orig = all_faces[surface_mask]
            surface_opp = all_opp[surface_mask]

            # Check winding: normal should point AWAY from opposite vertex
            sv0 = vert[surface_faces_orig[:, 0]]
            sv1 = vert[surface_faces_orig[:, 1]]
            sv2 = vert[surface_faces_orig[:, 2]]
            sv_opp = vert[surface_opp]

            edge1 = sv1 - sv0
            edge2 = sv2 - sv0
            normals = np.cross(edge1, edge2)
            face_centers = (sv0 + sv1 + sv2) / 3.0
            to_opposite = sv_opp - face_centers
            dots = np.sum(normals * to_opposite, axis=1)

            # Where normal points toward opposite vertex, flip winding
            needs_flip = dots > 0
            tri = surface_faces_orig.copy()
            tri[needs_flip, 1] = surface_faces_orig[needs_flip, 2]
            tri[needs_flip, 2] = surface_faces_orig[needs_flip, 1]
            tri = tri.astype(np.int32)

            # Reindex vertices to only include used ones
            used_verts = np.unique(np.concatenate([tet.ravel(), tri.ravel()]))
            vert = vert[used_verts]
            vert_map = np.zeros(used_verts.max() + 1, dtype=np.int64)
            vert_map[used_verts] = np.arange(len(used_verts))
            tet = vert_map[tet]
            tri = vert_map[tri]

            # Compute frame-embedding mapping from tet surface back to original
            # surface vertices (preserves absolute normal offset).
            tri_indices, coefs = _compute_frame_mapping(self[0], vert, tri)

            np.savez(
                cache_path,
                vert=vert,
                tet=tet,
                tri=tri,
                map_tri_indices=tri_indices,
                map_coefs=coefs,
                map_version=_SURFACE_MAP_VERSION,
            )
            return TetMesh((vert, tri, tet)).set_surface_mapping(
                tri_indices, coefs
            )

    def recompute_hash(self) -> "TriMesh":
        """Recompute the SHA-256 hash of the mesh from its vertex and element arrays.

        Example:
            Typically invoked automatically after mutating operations. To
            refresh the hash manually (for example after editing ``mesh[0]``
            in place)::

                mesh.recompute_hash()
                cache_path = mesh.compute_cache_path("custom")
        """
        import hashlib

        self.hash = hashlib.sha256(
            np.concatenate(
                [
                    np.array(self[0].shape),
                    self[0].ravel(),
                    np.array(self[1].shape),
                    self[1].ravel(),
                ]
            ).tobytes()
        ).hexdigest()
        return self

    def set_cache_dir(self, cache_dir: str) -> "TriMesh":
        """Set the cache directory used by this mesh.

        Example:
            Typically invoked automatically by the mesh pipeline. To
            retarget an existing mesh to a different cache location::

                mesh.set_cache_dir("/tmp/ppf-cache-alt")
        """
        self.cache_dir = cache_dir
        return self

    def compute_cache_path(self, name: str) -> str:
        """Compute a cache file path derived from the mesh hash and the given tag.

        Example:
            Typically invoked automatically by the mesh pipeline. To look
            up where a tagged cache entry would live on disk::

                path = mesh.compute_cache_path("decimate__19000")
                print(path)
        """
        return os.path.join(self.cache_dir, f"{self.hash}__{name}.npz")

    def save_cache(self, path: str) -> "TriMesh":
        """Save the mesh's vertex and triangle arrays to the given ``.npz`` path.

        Example:
            Typically invoked automatically by operations like
            :meth:`decimate`. To snapshot a mesh to disk manually::

                path = mesh.compute_cache_path("snapshot")
                mesh.save_cache(path)
        """
        np.savez(
            path,
            vert=self[0],
            tri=self[1],
        )
        return self

    def load_cache(self, path: str) -> Optional["TriMesh"]:
        """Load a cached mesh from the given path, or return ``None`` if it does not exist.

        Example:
            Typically invoked automatically by the mesh pipeline. To
            probe for a cached result before recomputing::

                cached = mesh.load_cache(mesh.compute_cache_path("decimate__19000"))
                if cached is None:
                    cached = mesh.decimate(19000)
        """
        if os.path.exists(path):
            data = np.load(path)
            return TriMesh.create(data["vert"], data["tri"], self.cache_dir)
        else:
            return None

    def normalize(self) -> "TriMesh":
        """Return ``self`` after invoking :func:`normalize` on the vertex array.

        Example:
            Rescale an imported mesh so its largest extent is 1::

                V, F = app.mesh.load_tri("big.ply").normalize()
        """
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TriMesh":
        """Scale the triangle mesh with the given scaling factors.

        Args:
            scale_x (float): scaling factor for the x-axis
            scale_y (float | None): scaling factor for the y-axis. If ``None``, ``scale_x`` is used.
            scale_z (float | None): scaling factor for the z-axis. If ``None``, ``scale_x`` is used.

        Returns:
            TriMesh: this triangle mesh (``self``)

        Example:
            Flatten a sphere into an oblate shape before registering it::

                V, F = app.mesh.icosphere(r=1.0, subdiv_count=3).scale(1.0, 0.3, 1.0)
                app.asset.add.tri("pill", V, F)
        """
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        scale(self[0], scale_x, scale_y, scale_z)
        return self
