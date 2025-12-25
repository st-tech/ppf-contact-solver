# File: _mesh_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import platform
import time

from typing import Optional

import numpy as np

from ._bvh_ import compute_barycentric_mapping as _compute_barycentric_mapping
from ._bvh_ import interpolate_surface as _interpolate_surface


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

    # Generate faces (triangles)
    faces = []
    for i in range(length_split):
        for j in range(width_split - 1):
            # Current vertex index
            v0 = i * width_split + j
            v1 = i * width_split + j + 1
            # Next column (wrap around for closed loop)
            next_i = (i + 1) % length_split
            v2 = next_i * width_split + j
            v3 = next_i * width_split + j + 1

            # Two triangles per quad
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.int32)

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
    """Mesh Manager for accessing mesh creation functions"""

    def __init__(self, cache_dir: str):
        """Initialize the mesh manager"""
        self._cache_dir = cache_dir
        self._create = CreateManager(cache_dir)

    @property
    def create(self) -> "CreateManager":
        """Get the mesh creation manager"""
        return self._create

    def export(self, V: np.ndarray, F: np.ndarray, path: str):
        """Export the mesh to a file"""
        import trimesh

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh.export(path)

    def line(self, _p0: list[float], _p1: list[float], n: int) -> "Rod":
        """Create a line mesh with a given start and end points and resolution.

        Args:
            _p0 (list[float]): a start point of the line
            _p1 (list[float]): an end point of the line
            n (int): a resolution of the line

        Returns:
            Rod: a line mesh, a pair of vertices and edges
        """
        p0, p1 = np.array(_p0), np.array(_p1)
        vert = np.vstack([p0 + (p1 - p0) * i / n for i in range(n + 1)])
        edge = np.array([[i, i + 1] for i in range(n)])
        return self.create.rod(vert, edge)

    def box(self, width: float = 1, height: float = 1, depth: float = 1) -> "TriMesh":
        """Create a box mesh

        Args:
            width (float): a width of the box
            hight (float): a height of the box
            depth (float): a depth of the box

        Returns:
            TriMesh: a box mesh, a pair of vertices and triangles
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
        """Create a box tetrahedral mesh directly without subdivision

        Unlike box().tetrahedralize(), this creates a minimal tetrahedral mesh
        with only 8 vertices and 5 tetrahedra, preserving the original box shape
        without any surface subdivision.

        Args:
            width (float): width of the box (x-axis)
            height (float): height of the box (y-axis)
            depth (float): depth of the box (z-axis)

        Returns:
            TetMesh: a tetrahedral box mesh
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
        """Create a rectangle mesh with a given resolution, width, height, and spanned by the given vectors `ex` and `ey`.

        Args:
            res_x (int): resolution of the mesh
            width (float): a width of the rectangle
            height (float): a height of the rectangle
            ex (list[float]): a 3D vector to span the rectangle
            ey (list[float]): a 3D vector to span the rectangle
            gen_uv (bool): if True, include UV coordinates in vertices (Nx5), otherwise Nx3

        Returns:
            TriMesh: a rectangle mesh, a pair of vertices (Nx5: x,y,z,u,v or Nx3: x,y,z) and triangles
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
        vert = np.vstack((X_flat, Y_flat, Z_flat)).T
        _ex, _ey = np.array(ex), np.array(ey)
        for i, v in enumerate(vert):
            x, y, _ = v
            vert[i] = _ex * x + _ey * y
        if gen_uv:
            u_coords = vert @ _ex
            v_coords = vert @ _ey
            vert_with_uv = np.zeros((len(vert), 5))
            vert_with_uv[:, :3] = vert
            vert_with_uv[:, 3] = u_coords
            vert_with_uv[:, 4] = v_coords
        else:
            vert_with_uv = vert
        n_faces = 2 * (res_x - 1) * (res_y - 1)
        tri = np.zeros((n_faces, 3), dtype=np.int32)
        tri_idx = 0
        for j in range(res_y - 1):
            for i in range(res_x - 1):
                v0 = i * res_y + j
                v1 = v0 + 1
                v2 = v0 + res_y
                v3 = v2 + 1
                if (i % 2) == (j % 2):
                    tri[tri_idx] = [v0, v1, v3]
                    tri[tri_idx + 1] = [v0, v3, v2]
                else:
                    tri[tri_idx] = [v0, v1, v2]
                    tri[tri_idx + 1] = [v1, v3, v2]
                tri_idx += 2
        return TriMesh.create(vert_with_uv, tri, self._cache_dir)

    def square(
        self,
        res: int = 32,
        size: float = 2,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
        gen_uv: bool = True,
    ) -> "TriMesh":
        """Create a square mesh with a given resolution and size, spanned by the given vectors `ex` and `ey`.

        Args:
            res (int): resolution of the mesh
            size (float): a diameter of the square
            ex (list[float]): a 3D vector to span the square
            ey (list[float]): a 3D vector to span the square
            gen_uv (bool): if True, include UV coordinates in vertices (Nx5), otherwise Nx3

        Returns:
            TriMesh: a square mesh, a pair of vertices and triangles
        """
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        return self.rectangle(res, size, size, ex, ey, gen_uv)

    def circle(self, n: int = 32, r: float = 1, ntri: int = 1024) -> "TriMesh":
        """Create a circle mesh

        Args:
            n (int): resolution of the circle
            r (float): radius of the circle
            ntri (int): approximate number of triangles filling the circle

        Returns:
            TriMesh: a circle mesh, a pair of 2D vertices and triangles
        """
        pts = []
        for i in range(n):
            t = 2 * np.pi * i / n
            x, y = r * np.cos(t), r * np.sin(t)
            pts.append([x, y])
        return self.create.tri(np.array(pts)).triangulate(ntri)

    def icosphere(self, r: float = 1, subdiv_count: int = 3) -> "TriMesh":
        """Create an icosphere mesh with a given radius and subdivision count.

        Args:
            r (float): radius of the icosphere
            subdiv_count (int): subdivision count of the icosphere

        Returns:
            TriMesh: an icosphere mesh, a pair of vertices and triangles
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
        """Load a mesh from a trimesh object"""
        return TriMesh.create(
            np.asarray(trimesh_mesh.vertices),
            np.asarray(trimesh_mesh.faces),
            self._cache_dir,
        )

    def cylinder(self, r: float, min_x: float, max_x: float, n: int):
        """Create a cylinder along x-axis

        Args:
            r (float): Radius of the cylinder
            min_x (float): Minimum x coordinate
            max_x (float): Maximum x coordinate
            n (int): Number of divisions along x-axis

        Returns:
            tuple: (V, F) where:
                - V: ndarray of shape (#x3) containing vertex positions
                - F: ndarray of shape (#x3) containing triangle indices
        """
        dx = (max_x - min_x) / n
        ny = int(2.0 * np.pi * r / dx)
        dy = 2.0 * np.pi / ny
        n_vert = (n + 1) * ny

        V = np.zeros((n_vert, 3))
        for j in range(ny):
            for i in range(n + 1):
                theta = j * dy
                idx = (n + 1) * j + i
                x = min_x + i * dx
                y = np.sin(theta) * r
                z = np.cos(theta) * r
                V[idx] = [x, y, z]

        F = np.zeros((2 * n * ny, 3), dtype=np.int32)
        for j in range(ny):
            for i in range(n):
                idx = j * n + i
                v0 = (n + 1) * j + i
                v1 = (n + 1) * j + i + 1
                v2 = (n + 1) * ((j + 1) % ny) + (i + 1)
                v3 = (n + 1) * ((j + 1) % ny) + i
                if (i % 2) == (j % 2):
                    F[2 * idx] = [v1, v2, v0]
                    F[2 * idx + 1] = [v3, v0, v2]
                else:
                    F[2 * idx] = [v0, v1, v3]
                    F[2 * idx + 1] = [v2, v3, v1]

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
        """Create a cone mesh with a given number of radial, vertical, and bottom resolution, radius, and height.

        Args:
            Nr (int): number of radial resolution
            Ny (int): number of vertical resolution
            Nb (int): number of bottom resolution
            radius (float): radius of the cone
            height (float): height of the cone
            sharpen (float): sharpening subdivision factor at the top

        Returns:
            TriMesh: a cone mesh, a pair of vertices and triangles
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
        """Create a torus mesh with a given radius, major radius, and resolution.

        Args:
            r (float): hole radius of the torus
            R (float): major radius of the torus
            n (int): resolution of the torus

        Returns:
            TriMesh: a torus mesh, a pair of vertices and triangles
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
        """Creatre a mobius mesh with a given length split, width split, twists, radius, flatness, width, and scale.

        Args:
            length_split (int): number of length split
            width_split (int): number of width split
            twists (int): number of twists
            r (float): radius of the mobius
            flatness (float): flatness of the mobius
            width (float): width of the mobius
            scale (float): scale of the mobius

        Returns:
            TriMesh: a mobius mesh, a pair of vertices and triangles
        """
        V, F = create_mobius(
            length_split, width_split, twists, r, flatness, width, scale
        )
        return TriMesh.create(V, F, self._cache_dir)

    def load_tri(self, path: str) -> "TriMesh":
        """Load a triangle mesh from a file

        Args:
            path (str): a path to the file

        Returns:
            TriMesh: a triangle mesh, a pair of vertices and triangles
        """
        import trimesh

        mesh = trimesh.load_mesh(path, process=False)
        return self._from_trimesh(mesh)

    def make_cache_dir(self):
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def preset(self, name: str) -> "TriMesh":
        """Load a preset mesh

        Args:
            name (str): a name of the preset mesh. Available names are `armadillo`, `knot`, and `bunny`.

        Returns:
            TriMesh: a preset mesh, a pair of vertices and triangles
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
    """A Manger tghat provides mesh creation functions

    This manager provides a set of functions to create various
    types of meshes, such as rods, triangles, and tetrahedra.

    """

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

    def rod(self, vert: np.ndarray, edge: np.ndarray) -> "Rod":
        """Create a rod mesh

        Args:
            vert (np.ndarray): a list of vertices
            edge (np.ndarray): a list of edges

        Returns:
            Rod: a rod mesh, a pair of vertices and edges
        """
        return Rod((vert, edge))

    def tri(self, vert: np.ndarray, elm: np.ndarray = np.zeros(0)) -> "TriMesh":
        """Create a triangle mesh

        Args:
            vert (np.ndarray): a list of vertices
            elm (np.ndarray): a list of elements

        Returns:
            TriMesh: a triangle mesh, a pair of vertices and triangles
        """
        if elm.size == 0:
            cnt = vert.shape[0]
            elm = np.array([[i, (i + 1) % cnt] for i in range(cnt)])
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(self._cache_dir)

    def tet(self, vert: np.ndarray, elm: np.ndarray, tet: np.ndarray) -> "TetMesh":
        """Create a tetrahedral mesh

        Args:
            vert (np.ndarray): a list of vertices
            elm (np.ndarray): a list of surface triangle elements
            tet (np.ndarray): a list of tetrahedra elements

        Returns:
            TetMesh: a tetrahedral mesh, a pair of vertices and tetrahedra
        """
        return TetMesh((vert, elm, tet))


def bbox(vert) -> np.ndarray:
    """Compute a bounding box of a mesh

    Given a list of vertices, this function computes a bounding box of the mesh.

    Args:
        vert (np.ndarray): a list of vertices

    Returns:
        3D array: a bounding box of the mesh, represented as [width, height, depth]
    """
    width = np.max(vert[:, 0]) - np.min(vert[:, 0])
    height = np.max(vert[:, 1]) - np.min(vert[:, 1])
    depth = np.max(vert[:, 2]) - np.min(vert[:, 2])
    return np.array([width, height, depth])


def normalize(vert: np.ndarray):
    """Normalize a set of vertices

    Normalize a set of vertices so that the maximum bounding box size becomes 1.

    Args:
        vert (np.ndarray): a list of vertices

    Return:
        np.ndarray: a normalized set of vertices
    """
    vert -= np.mean(vert, axis=0)
    vert /= np.max(bbox(vert))


def scale(
    vert: np.ndarray, scale_x: float, scale_y: float, scale_z: float
) -> np.ndarray:
    """Scale a set of vertices

    Scale a set of vertices with given scaling factors.

    Args:
        vert (np.ndarray): a list of vertices
        scale_x (float): a scaling factor for the x-axis
        scale_y (float): a scaling factor for the y-axis
        scale_z (float): a scaling factor for the z-axis

    Return:
        np.ndarray: a scaled set of vertices
    """
    mean = np.mean(vert, axis=0)
    vert -= mean
    vert *= np.array([scale_x, scale_y, scale_z])
    vert += mean
    return vert


class Rod(tuple[np.ndarray, np.ndarray]):
    """A class representing a rod mesh

    This class represents a rod mesh, which is a pair of vertices and edges.
    The first element of the tuple is a list of vertices, and the second element is a list of edges.

    """

    def normalize(self) -> "Rod":
        """Normalize the rod mesh

        It normalizes the rod mesh so that the maximum bounding box size becomes 1.

        """
        normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "Rod":
        """Scale the rod mesh

        Scale the rod mesh with given scaling factors.

        Args:
            scale_x (float): a scaling factor for the x-axis
            scale_y (float): a scaling factor for the y-axis. If None, it is set to the same value as scale_x.
            scale_z (float): a scaling factor for the z-axis. If None, it is set to the same value as scale_x.

        Returns:
            Rod: a scaled rod mesh
        """
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        scale(self[0], scale_x, scale_y, scale_z)
        return self


class TetMesh(tuple[np.ndarray, np.ndarray, np.ndarray]):
    """A class representing a tetrahedral mesh

    This class represents a tetrahedral mesh, which is a pair of vertices, surface triangles, and tetrahedra.

    Attributes:
        surface_map: Optional tuple of (tri_indices, bary_coords) for interpolating back to original surface

    """

    def normalize(self) -> "TetMesh":
        """Normalize the tetrahedral mesh

        It normalizes the tetrahedral mesh so that the maximum bounding box size becomes 1.

        """
        normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TetMesh":
        """Scale the tet mesh

        Scle the tet mesh with given scaling factors.

        Args:
            scale_x (float): a scaling factor for the x-axis
            scale_y (float): a scaling factor for the y-axis. If None, it is set to the same value as scale_x.
            scale_z (float): a scaling factor for the z-axis. If None, it is set to the same value as scale_x.

        Returns:
            TetMesh: a scaled tet mesh
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
        bary_coords: np.ndarray,
    ) -> "TetMesh":
        """Set the surface mapping for interpolating back to original surface.

        Args:
            tri_indices: Triangle indices in tet surface for each original vertex (N,)
            bary_coords: Barycentric coordinates for each original vertex (N, 3)

        Returns:
            TetMesh: self with surface mapping set
        """
        self.surface_map = (tri_indices, bary_coords)
        return self

    def has_surface_mapping(self) -> bool:
        """Check if this TetMesh has surface mapping data."""
        return hasattr(self, "surface_map") and self.surface_map is not None

    def interpolate_surface(
        self, deformed_vert: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Interpolate deformed tet mesh vertices back to original surface vertices.

        Uses the stored barycentric mapping to compute positions of original surface
        vertices based on the deformed tet mesh surface.

        Args:
            deformed_vert: Deformed tet mesh vertices. If None, uses current vertices.

        Returns:
            Interpolated vertex positions matching original surface vertex count.
        """
        if not self.has_surface_mapping():
            raise ValueError("No surface mapping available.")

        if deformed_vert is None:
            deformed_vert = self[0]

        tri_indices, bary_coords = self.surface_map
        surf_tri = self[1]  # Surface triangles of tet mesh

        return _interpolate_surface(deformed_vert, surf_tri, tri_indices, bary_coords)


class TriMesh(tuple[np.ndarray, np.ndarray]):
    """A class representing a triangle mesh

    This class represents a triangle mesh, which is a pair of vertices and triangles.

    """

    @staticmethod
    def create(vert: np.ndarray, elm: np.ndarray, cache_dir: str) -> "TriMesh":
        """Create a triangle mesh and recompute the hash"""
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(cache_dir)

    def _make_trimesh(self):
        """Create a trimesh object"""
        import trimesh

        return trimesh.Trimesh(vertices=self[0], faces=self[1], process=False)

    def export(self, path):
        """Export mesh as PLY or OBJ

        Args:
            path (str): export path
        """
        mesh = self._make_trimesh()
        mesh.export(path)

    def decimate(self, target_tri: int) -> "TriMesh":
        """Mesh decimation

        Reduce the number of triangles in the mesh to the target number.

        Args:
            target_tri (int): a target number of triangles

        Returns:
            TriMesh: a decimated mesh
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
        """Mesh subdivision

        Subdivide the mesh with a given number of subdivisions and method.

        Args:
            n (int): a number of subdivisions
            method (str): a method of subdivision. Available methods are "midpoint" and "loop".
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
        """Compute the area of a 2D shape"""
        assert pts.shape[1] == 2
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.abs(np.dot(x, y_next) - np.dot(x_next, y))
        return area

    def triangulate(self, target: int = 1024, min_angle: float = 20) -> "TriMesh":
        """Triangulate a closed line shape with 2D coordinates

        This function triangulates a closed 2D line shape with a given
        target number of triangles and minimum angle.

        Args:
            target (int): a target number of triangles
            min_angle (float): a minimum angle of the triangles

        Returns:
            TriMesh: a triangulated mesh
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
        """Tetrahedralize a surface triangle mesh

        This function tetrahedralizes a surface triangle mesh using fTetWild.
        The original surface is preserved via barycentric mapping, allowing
        interpolation of deformed positions back to the original surface structure.

        Args:
            edge_length_fac (float): Tetrahedral edge length as fraction of bbox diagonal (default: 0.05)
            optimize (bool): Whether to optimize the mesh (default: True)

        Returns:
            TetMesh: A tetrahedral mesh with surface mapping. Use `interpolate_surface()`
                     to get deformed positions in the original surface structure.
        """
        arg_str = "_".join([str(a) for a in args])
        if len(kwargs) > 0:
            arg_str += "_".join([f"{k}={v}" for k, v in kwargs.items()])
        cache_path = self.compute_cache_path(
            f"{self.hash}_tetrahedralize_{arg_str}.npz"
        )
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            tri = data["tri"] if "tri" in data else self[1]
            tet_mesh = TetMesh((data["vert"], tri, data["tet"]))
            # Restore surface mapping if available
            if "map_tri_indices" in data:
                tet_mesh.set_surface_mapping(
                    data["map_tri_indices"], data["map_bary_coords"]
                )
            return tet_mesh
        else:
            import pytetwild

            # edge_length_fac: tetrahedral edge length as fraction of bbox diagonal
            edge_length_fac = kwargs.get("edge_length_fac", 0.05)
            optimize = kwargs.get("optimize", True)

            # Suppress verbose C++ output from fTetWild
            if platform.system() == "Windows":
                # On Windows embedded Python, output redirection can cause hangs
                # Just run pytetwild without suppression - verbose but works
                print("Running pytetwild...")
                vert, tet = pytetwild.tetrahedralize(  # type: ignore[attr-defined]
                    self[0], self[1], edge_length_fac=edge_length_fac, optimize=optimize
                )
                # Clean up fTetWild temp file
                if os.path.exists("__tracked_surface.stl"):
                    os.remove("__tracked_surface.stl")
                print("Tetrahedralization complete.")
                print(f"Number of vertices: {len(vert)}")
                print(f"Number of tetrahedra: {len(tet)}")
            else:
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                old_stdout_fd = os.dup(1)
                old_stderr_fd = os.dup(2)
                try:
                    os.dup2(devnull_fd, 1)
                    os.dup2(devnull_fd, 2)
                    vert, tet = pytetwild.tetrahedralize(  # type: ignore[attr-defined]
                        self[0],
                        self[1],
                        edge_length_fac=edge_length_fac,
                        optimize=optimize,
                    )
                finally:
                    os.dup2(old_stdout_fd, 1)
                    os.dup2(old_stderr_fd, 2)
                    os.close(devnull_fd)
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                    # Clean up temporary file created by fTetWild
                    if os.path.exists("__tracked_surface.stl"):
                        os.remove("__tracked_surface.stl")

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

            # Extract surface triangles from tetrahedra
            # A surface face is shared by exactly one tetrahedron
            from collections import Counter

            # For each tet face, store (face_indices, opposite_vertex_index)
            face_indices = [
                ((0, 1, 2), 3),
                ((0, 1, 3), 2),
                ((0, 2, 3), 1),
                ((1, 2, 3), 0),
            ]
            all_faces = []
            face_to_data = {}  # map sorted face to (original_face, opposite_vertex)
            for t in tet:
                for fi, opp_idx in face_indices:
                    original_face = (t[fi[0]], t[fi[1]], t[fi[2]])
                    opposite_vert = t[opp_idx]
                    sorted_face = tuple(sorted(original_face))
                    all_faces.append(sorted_face)
                    face_to_data[sorted_face] = (original_face, opposite_vert)
            face_counts = Counter(all_faces)
            # Surface faces appear exactly once
            surface_faces = []
            for f, count in face_counts.items():
                if count == 1:
                    orig, opp_vi = face_to_data[f]
                    v0, v1, v2 = vert[orig[0]], vert[orig[1]], vert[orig[2]]
                    v_opp = vert[opp_vi]
                    # Normal should point AWAY from the opposite vertex
                    edge1, edge2 = v1 - v0, v2 - v0
                    normal = np.cross(edge1, edge2)
                    # Vector from face to opposite vertex
                    face_center = (v0 + v1 + v2) / 3.0
                    to_opposite = v_opp - face_center
                    # If normal points toward opposite vertex, flip winding
                    if np.dot(normal, to_opposite) > 0:
                        surface_faces.append((orig[0], orig[2], orig[1]))
                    else:
                        surface_faces.append(orig)
            tri = np.array(surface_faces, dtype=np.int32)

            # Reindex vertices to only include used ones
            used_verts = np.unique(np.concatenate([tet.ravel(), tri.ravel()]))
            vert = vert[used_verts]
            vert_map = np.zeros(used_verts.max() + 1, dtype=np.int64)
            vert_map[used_verts] = np.arange(len(used_verts))
            tet = vert_map[tet]
            tri = vert_map[tri]

            # Compute barycentric mapping from new surface to original surface vertices
            tri_indices, bary_coords = _compute_barycentric_mapping(self[0], vert, tri)

            np.savez(
                cache_path,
                vert=vert,
                tet=tet,
                tri=tri,
                map_tri_indices=tri_indices,
                map_bary_coords=bary_coords,
            )
            return TetMesh((vert, tri, tet)).set_surface_mapping(
                tri_indices, bary_coords
            )

    def recompute_hash(self) -> "TriMesh":
        """Recompute the hash of the mesh"""
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
        """Set the cache directory of the mesh"""
        self.cache_dir = cache_dir
        return self

    def compute_cache_path(self, name: str) -> str:
        """Compute the cache path of the mesh"""
        return os.path.join(self.cache_dir, f"{self.hash}__{name}.npz")

    def save_cache(self, path: str) -> "TriMesh":
        """Save the mesh to a cache"""
        np.savez(
            path,
            vert=self[0],
            tri=self[1],
        )
        return self

    def load_cache(self, path: str) -> Optional["TriMesh"]:
        """Load a cached mesh"""
        if os.path.exists(path):
            data = np.load(path)
            return TriMesh.create(data["vert"], data["tri"], self.cache_dir)
        else:
            return None

    def normalize(self) -> "TriMesh":
        """Normalize the triangle mesh

        This function normalizes the triangle mesh so that the maximum bounding box size becomes 1.

        """
        normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TriMesh":
        """Scale the triangle mesh

        Scale the triangle mesh with given scaling factors.

        Args:
            scale_x (float): a scaling factor for the x-axis
            scale_y (float): a scaling factor for the y-axis. If None, it is set to the same value as scale_x.
            scale_z (float): a scaling factor for the z-axis. If None, it is set to the same value as scale_x.

        Returns:
            TriMesh: a scaled triangle mesh
        """
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        scale(self[0], scale_x, scale_y, scale_z)
        return self
