# File: _mesh_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
from typing import Optional
import os


class MeshManager:
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        self.create = CreateManager(cache_dir)

    def line(self, _p0: list[float], _p1: list[float], n: int) -> "Rod":
        p0, p1 = np.array(_p0), np.array(_p1)
        vert = np.vstack([p0 + (p1 - p0) * i / n for i in range(n + 1)])
        edge = np.array([[i, i + 1] for i in range(n)])
        return self.create.rod(vert, edge)

    def box(self, width: float = 1, height: float = 1, depth: float = 1) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(
            o3d.geometry.TriangleMesh.create_box(width, height, depth)
        )

    def rectangle(
        self, res_x: int = 32, width: float = 2, height: float = 1
    ) -> "TriMesh":
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

    def square(self, res: int = 32, size: float = 2) -> "TriMesh":
        return self.rectangle(res, size, size)

    def circle(self, n: int = 32, r: float = 1, ntri: int = 1024) -> "TriMesh":
        pts = []
        for i in range(n):
            t = 2 * np.pi * i / n
            x, y = r * np.cos(t), r * np.sin(t)
            pts.append([x, y])
        return self.create.tri(np.array(pts)).triangulate(ntri)

    def icosphere(self, r: float = 1, subdiv_count: int = 3) -> "TriMesh":
        import gpytoolbox as gpy

        V, F = gpy.icosphere(subdiv_count)
        V *= r
        return TriMesh.create(V, F, self._cache_dir)

    def _from_o3d(self, o3d_mesh) -> "TriMesh":
        if o3d_mesh.is_self_intersecting():
            print("Warning: Mesh is self-intersecting")
        return TriMesh.create(
            np.asarray(o3d_mesh.vertices),
            np.asarray(o3d_mesh.triangles),
            self._cache_dir,
        )

    def cylinder(self, r: float = 1, height: float = 2, n: int = 32) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(o3d.geometry.TriangleMesh.create_cylinder(r, height, n))

    def cone(self, r: float = 1, height: float = 2, n: int = 32) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(o3d.geometry.TriangleMesh.create_cone(r, height, n))

    def torus(self, r: float = 1, R: float = 0.25, n: int = 32) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(o3d.geometry.TriangleMesh.create_torus(r, R, n))

    def mobius(
        self,
        length_split=70,
        width_split=15,
        twists=1,
        r=1,
        flatness=1,
        width=1,
        scale=1,
    ) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(
            o3d.geometry.TriangleMesh.create_mobius(
                length_split, width_split, twists, r, flatness, width, scale
            )
        )

    def load_tri(self, path: str) -> "TriMesh":
        import open3d as o3d

        return self._from_o3d(o3d.io.read_triangle_mesh(path))

    def make_cache_dir(self):
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def preset(self, name: str) -> "TriMesh":
        cache_name = os.path.join(self._cache_dir, f"preset__{name}.npz")
        if os.path.exists(cache_name):
            data = np.load(cache_name)
            return TriMesh.create(data["vert"], data["tri"], self._cache_dir)
        else:
            import open3d as o3d

            mesh = None
            if name == "armadillo":
                mesh = o3d.data.ArmadilloMesh()
            elif name == "knot":
                mesh = o3d.data.KnotMesh()
            elif name == "bunny":
                mesh = o3d.data.BunnyMesh()
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
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

    def rod(self, vert: np.ndarray, edge: np.ndarray) -> "Rod":
        return Rod((vert, edge))

    def tri(self, vert: np.ndarray, elm: np.ndarray = np.zeros(0)) -> "TriMesh":
        if elm.size == 0:
            cnt = vert.shape[0]
            elm = np.array([[i, (i + 1) % cnt] for i in range(cnt)])
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(self._cache_dir)

    def tet(self, vert: np.ndarray, elm: np.ndarray, tet: np.ndarray) -> "TetMesh":
        return TetMesh((vert, elm, tet))


def bbox(vert) -> np.ndarray:
    width = np.max(vert[:, 0]) - np.min(vert[:, 0])
    height = np.max(vert[:, 1]) - np.min(vert[:, 1])
    depth = np.max(vert[:, 2]) - np.min(vert[:, 2])
    return np.array([width, height, depth])


def normalize(vert):
    vert -= np.mean(vert, axis=0)
    vert /= np.max(bbox(vert))


class Rod(tuple[np.ndarray, np.ndarray]):
    def normalize(self) -> "Rod":
        normalize(self[0])
        return self


class TetMesh(tuple[np.ndarray, np.ndarray, np.ndarray]):
    def normalize(self) -> "TetMesh":
        normalize(self[0])
        return self


class TriMesh(tuple[np.ndarray, np.ndarray]):
    @staticmethod
    def create(vert: np.ndarray, elm: np.ndarray, cache_dir: str) -> "TriMesh":
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(cache_dir)

    def _make_o3d(self):
        import open3d as o3d

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self[0]),
            o3d.utility.Vector3iVector(self[1]),
        )

    def decimate(self, target_tri: int) -> "TriMesh":
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
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.abs(np.dot(x, y_next) - np.dot(x_next, y))
        return area

    def triangulate(self, target: int = 1024, min_angle: float = 20) -> "TriMesh":
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

            vert, tet = tetgen.TetGen(self[0], self[1]).tetrahedralize(*args, **kwargs)
            np.savez(
                cache_path,
                vert=vert,
                tet=tet,
            )
            return TetMesh((vert, self[1], tet))

    def recompute_hash(self) -> "TriMesh":
        import hashlib

        self.hash = hashlib.sha256(
            np.concatenate(
                [
                    np.array(self[0].shape),
                    self[0].ravel(),
                    np.array(self[1].shape),
                    self[1].ravel(),
                ]
            )
        ).hexdigest()
        return self

    def set_cache_dir(self, cache_dir: str) -> "TriMesh":
        self.cache_dir = cache_dir
        return self

    def compute_cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{self.hash}__{name}.npz")

    def save_cache(self, path: str) -> "TriMesh":
        np.savez(
            path,
            vert=self[0],
            tri=self[1],
        )
        return self

    def load_cache(self, path: str) -> Optional["TriMesh"]:
        if os.path.exists(path):
            data = np.load(path)
            return TriMesh.create(data["vert"], data["tri"], self.cache_dir)
        else:
            return None

    def normalize(self) -> "TriMesh":
        normalize(self[0])
        return self
