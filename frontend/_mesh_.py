# File: _mesh_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import time

from typing import Optional

import numpy as np

os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "true"


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
        import open3d as o3d

        o3d.io.write_triangle_mesh(
            path,
            o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(V),
                triangles=o3d.utility.Vector3iVector(F),
            ),
        )

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

    def rectangle(
        self,
        res_x: int = 32,
        width: float = 2,
        height: float = 1,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
    ) -> "TriMesh":
        """Create a rectangle mesh with a given resolution, width, height, and spanned by the given vectors `ex` and `ey`.

        Args:
            res_x (int): resolution of the mesh
            width (float): a width of the rectangle
            height (float): a height of the rectangle
            ex (list[float]): a 3D vector to span the rectangle
            ey (list[float]): a 3D vector to span the rectangle

        Returns:
            TriMesh: a rectangle mesh, a pair of vertices and triangles
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
        return TriMesh.create(vert, tri, self._cache_dir)

    def square(
        self,
        res: int = 32,
        size: float = 2,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
    ) -> "TriMesh":
        """Create a square mesh with a given resolution and size, spanned by the given vectors `ex` and `ey`.

        Args:
            res (int): resolution of the mesh
            size (float): a diameter of the square
            ex (list[float]): a 3D vector to span the square
            ey (list[float]): a 3D vector to span the square

        Returns:
            TriMesh: a square mesh, a pair of vertices and triangles
        """
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        return self.rectangle(res, size, size, ex, ey)

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
            sunbdiv_count (int): subdivision count of the icosphere

        Returns:
            TriMesh: an icosphere mesh, a pair of vertices and triangles
        """
        import gpytoolbox as gpy

        mV, F = gpy.icosphere(subdiv_count)
        mV *= r
        return TriMesh.create(mV, F, self._cache_dir)

    def _from_o3d(self, o3d_mesh) -> "TriMesh":
        """Load a mesh from an Open3D mesh"""
        return TriMesh.create(
            np.asarray(o3d_mesh.vertices),
            np.asarray(o3d_mesh.triangles),
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
        import open3d as o3d

        return self._from_o3d(o3d.geometry.TriangleMesh.create_torus(r, R, n))

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
        import open3d as o3d

        return self._from_o3d(
            o3d.geometry.TriangleMesh.create_mobius(
                length_split, width_split, twists, r, flatness, width, scale
            )
        )

    def load_tri(self, path: str) -> "TriMesh":
        """Load a triangle mesh from a file

        Args:
            path (str): a path to the file

        Returns:
            TriMesh: a triangle mesh, a pair of vertices and triangles
        """
        import open3d as o3d

        return self._from_o3d(o3d.io.read_triangle_mesh(path))

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
        cache_name = os.path.join(self._cache_dir, f"preset__{name}.npz")
        if os.path.exists(cache_name):
            data = np.load(cache_name)
            return TriMesh.create(data["vert"], data["tri"], self._cache_dir)
        else:
            import open3d as o3d

            mesh = None
            num_try, max_try, success, wait_time = 0, 5, False, 3
            while num_try < max_try:
                try:
                    if name == "armadillo":
                        mesh = o3d.data.ArmadilloMesh()
                    elif name == "knot":
                        mesh = o3d.data.KnotMesh()
                    elif name == "bunny":
                        mesh = o3d.data.BunnyMesh()
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

            if mesh is not None:
                mesh = o3d.io.read_triangle_mesh(mesh.path)
                vert = np.asarray(mesh.vertices)
                tri = np.asarray(mesh.triangles)
                self.make_cache_dir()
                np.savez(
                    cache_name,
                    vert=vert,
                    tri=tri,
                )
                return TriMesh.create(vert, tri, self._cache_dir)
            else:
                raise Exception(f"Mesh {name} not found")


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
        scale_y: float | None = None,
        scale_z: float | None = None,
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
        scale_y: float | None = None,
        scale_z: float | None = None,
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


class TriMesh(tuple[np.ndarray, np.ndarray]):
    """A class representing a triangle mesh

    This class represents a triangle mesh, which is a pair of vertices and triangles.

    """

    @staticmethod
    def create(vert: np.ndarray, elm: np.ndarray, cache_dir: str) -> "TriMesh":
        """Create a triangle mesh and recompute the hash"""
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(cache_dir)

    def _make_o3d(self):
        """Create an Open3D triangle mesh"""
        import open3d as o3d

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self[0]),
            o3d.utility.Vector3iVector(self[1]),
        )

    def export(self, path):
        """Export mesh as PLY or OBJ

        Args:
            path (str): export path
        """
        import open3d as o3d

        o3d_mesh = self._make_o3d()
        o3d.io.write_triangle_mesh(path, o3d_mesh)

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
            mesh = self._make_o3d().simplify_quadric_decimation(target_tri)
            return TriMesh.create(
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
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
        cache_path = self.compute_cache_path(f"subdiv__{method}__{n}")
        cached = self.load_cache(cache_path)
        if cached is None:
            if self[1].shape[1] != 3:
                raise Exception("Only triangle meshes are supported")
            if method == "midpoint":
                mesh = self._make_o3d().subdivide_midpoint(n)
            elif method == "loop":
                mesh = self._make_o3d().subdivide_loop(n)
            else:
                raise Exception(f"Unknown subdivision method {method}")
            return TriMesh.create(
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
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

        This function tetrahedralizes a surface triangle mesh with a given TetGen arguments.

        Args:
            args: a list of arguments
            kwargs: a list of keyword arguments

        Returns:
            TetMesh: a tetrahedral mesh
        """
        arg_str = "_".join([str(a) for a in args])
        if len(kwargs) > 0:
            arg_str += "_".join([f"{k}={v}" for k, v in kwargs.items()])
        cache_path = self.compute_cache_path(
            f"{self.hash}_tetrahedralize_{arg_str}.npz"
        )
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return TetMesh((data["vert"], self[1], data["tet"]))
        else:
            import tetgen

            result = tetgen.TetGen(self[0], self[1]).tetrahedralize(*args, **kwargs)
            vert, tet = result[0], result[1]
            np.savez(
                cache_path,
                vert=vert,
                tet=tet,
            )
            return TetMesh((vert, self[1], tet))

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
        scale_y: float | None = None,
        scale_z: float | None = None,
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
