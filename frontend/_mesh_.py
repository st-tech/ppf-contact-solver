# File: _mesh_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Thin Python shim around the Rust mesh kernels in
# `crates/ppf-cts-py/src/mesh_py.rs` (validation, primitive-shape
# generation, parameter normalization, cache-path arithmetic, string
# formatting). The methods that stay in Python wrap third-party
# libraries:
#
#   * `trimesh` calls (mesh I/O, decimation, subdivision, torus).
#   * `pytetwild.tetrahedralize` (subprocess, Python script gen).
#   * `triangle.triangulate` (CGAL Python binding).
#   * `urllib.request` for preset downloads.
#   * `hashlib.sha256` for cache-key digesting.
#   * `np.load` / `np.savez` for `.npz` cache I/O (numpy lib).

import hashlib
import os
import sys
import time

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]

from ._bvh_ import frame_mapping as _frame_mapping
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
    """Create a Mobius strip mesh."""
    return _rust.mesh_mobius(
        length_split, width_split, int(twists), float(r), float(flatness), float(width), float(scale)
    )


def _fix_skinny_triangles(
    vert: np.ndarray, tri: np.ndarray, min_angle_deg: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fix skinny triangles by merging vertices of very short edges."""
    vert = np.ascontiguousarray(vert, dtype=np.float64)
    tri = np.ascontiguousarray(tri, dtype=np.int32)
    return _rust.mesh_fix_skinny_triangles(vert, tri, float(min_angle_deg))


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
        self._cache_dir = cache_dir
        self._create = CreateManager(cache_dir)

    @property
    def create(self) -> "CreateManager":
        """Get the mesh creation manager."""
        return self._create

    def export(self, V: np.ndarray, F: np.ndarray, path: str):
        """Export a mesh given by vertices ``V`` and faces ``F`` to a file."""
        import trimesh

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh.export(path)

    def line(self, _p0: list[float], _p1: list[float], n: int) -> "Rod":
        """Create a line mesh with the given start and end points and resolution."""
        vert, edge = _rust.scene_line_mesh(
            [float(_p0[0]), float(_p0[1]), float(_p0[2])],
            [float(_p1[0]), float(_p1[1]), float(_p1[2])],
            int(n),
        )
        return self.create.rod(vert, edge)

    def box(self, width: float = 1, height: float = 1, depth: float = 1) -> "TriMesh":
        """Create a box mesh."""
        V, F = _rust.scene_box_mesh(float(width), float(height), float(depth))
        return TriMesh.create(V, F, self._cache_dir)

    def tet_box(
        self, width: float = 1, height: float = 1, depth: float = 1
    ) -> "TetMesh":
        """Create a box tetrahedral mesh directly without subdivision."""
        V, F, T = _rust.scene_tet_box_mesh(float(width), float(height), float(depth))
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
        """Create a rectangle mesh in the plane spanned by ``ex`` and ``ey``."""
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        _ex = np.ascontiguousarray(ex, dtype=np.float64)
        _ey = np.ascontiguousarray(ey, dtype=np.float64)
        verts, tri = _rust.mesh_rectangle_with_uv(
            int(res_x), float(width), float(height), _ex, _ey, bool(gen_uv)
        )
        return TriMesh.create(verts, tri, self._cache_dir)

    def square(
        self,
        res: int = 32,
        size: float = 2,
        ex: Optional[list[float]] = None,
        ey: Optional[list[float]] = None,
        gen_uv: bool = True,
    ) -> "TriMesh":
        """Create a square mesh."""
        if ey is None:
            ey = [0, 1, 0]
        if ex is None:
            ex = [1, 0, 0]
        return self.rectangle(res, size, size, ex, ey, gen_uv)

    def circle(self, n: int = 32, r: float = 1, ntri: int = 1024) -> "TriMesh":
        """Create a circle mesh."""
        pts = _rust.mesh_circle_points_2d(int(n), float(r))
        return self.create.tri(pts).triangulate(ntri)

    def icosphere(self, r: float = 1, subdiv_count: int = 3) -> "TriMesh":
        """Create an icosphere mesh."""
        verts, faces = _rust.mesh_icosphere(float(r), int(subdiv_count))
        return TriMesh.create(verts, faces, self._cache_dir)

    def _from_trimesh(self, trimesh_mesh) -> "TriMesh":
        """Load a mesh from a ``trimesh`` object."""
        return TriMesh.create(
            np.asarray(trimesh_mesh.vertices),
            np.asarray(trimesh_mesh.faces),
            self._cache_dir,
        )

    def cylinder(self, r: float, min_x: float, max_x: float, n: int):
        """Create a cylinder along the x-axis."""
        dx, ny, dy = _rust.mesh_cylinder_dx_ny_dy(
            float(min_x), float(max_x), int(n), float(r)
        )
        V = _rust.mesh_generate_cylinder_verts(int(n), int(ny), float(min_x), float(dx), float(dy), float(r))
        F = _rust.mesh_generate_cylinder_faces(int(n), int(ny))
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
        """Create a cone mesh."""
        V, T = _rust.scene_cone_mesh(
            int(Nr), int(Ny), int(Nb), float(radius), float(height), float(sharpen)
        )
        return TriMesh.create(V, T, self._cache_dir)

    def torus(self, r: float = 1, R: float = 0.25, n: int = 32) -> "TriMesh":
        """Create a torus mesh."""
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
        """Create a Mobius strip mesh."""
        V, F = create_mobius(
            length_split, width_split, twists, r, flatness, width, scale
        )
        return TriMesh.create(V, F, self._cache_dir)

    def load_tri(self, path: str) -> "TriMesh":
        """Load a triangle mesh from a file."""
        import trimesh

        mesh = trimesh.load_mesh(path, process=False)
        return self._from_trimesh(mesh)

    def make_cache_dir(self):
        """Create the cache directory if it does not already exist."""
        _rust.make_dir(self._cache_dir)

    def preset(self, name: str) -> "TriMesh":
        """Load a preset mesh, downloading it from a remote source on first use and caching it locally."""
        import ssl
        import urllib.request

        import certifi
        import trimesh

        # Legacy preset cache path: `<cache_dir>/preset__{name}.npz`.
        # Mirrors `mesh_cache_path(cache_dir, "preset", name)`
        # (`<cache_dir>/preset__<name>.npz`).
        cache_name = _rust.mesh_cache_path(self._cache_dir, "preset", name)
        if os.path.exists(cache_name):
            data = np.load(cache_name)
            return TriMesh.create(data["vert"], data["tri"], self._cache_dir)

        # Resolve preset filename + URL via Rust (raises ValueError on
        # unknown names with a list of known names).
        stem, url = _rust.mesh_preset_lookup(name)

        # Use downloads subdirectory within cache_dir.
        downloads_dir = os.path.join(self._cache_dir, "downloads")
        _rust.make_dir(downloads_dir)
        temp_path = os.path.join(downloads_dir, f"{stem}.ply")

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Download with retry logic.
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

        mesh = trimesh.load_mesh(temp_path, process=False)
        vert = np.asarray(mesh.vertices)
        tri = np.asarray(mesh.faces)

        self.make_cache_dir()
        np.savez(cache_name, vert=vert, tri=tri)

        return TriMesh.create(vert, tri, self._cache_dir)


class CreateManager:
    """A manager that provides mesh creation functions.

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
        """Create a rod mesh."""
        return Rod((vert, edge))

    def tri(self, vert: np.ndarray, elm: np.ndarray = None) -> "TriMesh":
        """Create a triangle mesh.

        When ``elm`` is ``None`` or empty, a closed edge loop over all vertices is
        auto-generated instead of triangle faces.
        """
        if elm is None or elm.size == 0:
            elm = _rust.mesh_tri_edge_loop(int(vert.shape[0]))
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(self._cache_dir)

    def tet(self, vert: np.ndarray, elm: np.ndarray, tet: np.ndarray) -> "TetMesh":
        """Create a tetrahedral mesh."""
        return TetMesh((vert, elm, tet))


def bbox(vert) -> np.ndarray:
    """Compute the axis-aligned bounding box extents of a mesh."""
    return _rust.mesh_bbox(np.ascontiguousarray(vert, dtype=np.float64))


def normalize(vert: np.ndarray):
    """Normalize a set of vertices."""
    return _rust.mesh_normalize_verts(np.ascontiguousarray(vert, dtype=np.float64))


def scale(
    vert: np.ndarray, scale_x: float, scale_y: float, scale_z: float
) -> np.ndarray:
    """Scale a set of vertices around their centroid."""
    return _rust.mesh_scale_per_axis(
        np.ascontiguousarray(vert, dtype=np.float64),
        float(scale_x),
        float(scale_y),
        float(scale_z),
    )


class Rod(tuple[np.ndarray, np.ndarray]):
    """A class representing a rod mesh."""

    def normalize(self) -> "Rod":
        """Normalize the rod mesh in place so that the maximum bounding box extent is 1."""
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "Rod":
        """Scale the rod mesh in place with the given scaling factors."""
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        self[0][:] = scale(self[0], scale_x, scale_y, scale_z)
        return self


class TetMesh(tuple[np.ndarray, np.ndarray, np.ndarray]):
    """A class representing a tetrahedral mesh."""

    def normalize(self) -> "TetMesh":
        """Return ``self`` after invoking :func:`normalize` on the vertex array."""
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TetMesh":
        """Scale the tet mesh with the given scaling factors."""
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        self[0][:] = scale(self[0], scale_x, scale_y, scale_z)
        return self

    def set_surface_mapping(
        self,
        tri_indices: np.ndarray,
        coefs: np.ndarray,
    ) -> "TetMesh":
        """Set the frame-embedding surface mapping used by :meth:`interpolate_surface`."""
        self.surface_map = (tri_indices, coefs)
        return self

    def has_surface_mapping(self) -> bool:
        """Check if this TetMesh has surface mapping data."""
        return hasattr(self, "surface_map") and self.surface_map is not None

    def interpolate_surface(
        self, deformed_vert: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Reconstruct original-resolution positions from the deformed tet surface."""
        if not self.has_surface_mapping():
            raise ValueError("No surface mapping available.")

        if deformed_vert is None:
            deformed_vert = self[0]

        tri_indices, coefs = self.surface_map
        surf_tri = self[1]

        return _interpolate_surface(deformed_vert, surf_tri, tri_indices, coefs)


class TriMesh(tuple[np.ndarray, np.ndarray]):
    """A class representing a triangle mesh."""

    @staticmethod
    def create(vert: np.ndarray, elm: np.ndarray, cache_dir: str) -> "TriMesh":
        """Create a triangle mesh, recompute its hash, and bind the given cache directory."""
        return TriMesh((vert, elm)).recompute_hash().set_cache_dir(cache_dir)

    def _make_trimesh(self):
        """Build a ``trimesh.Trimesh`` object from this mesh's vertices and faces."""
        import trimesh

        return trimesh.Trimesh(vertices=self[0], faces=self[1], process=False)

    def export(self, path):
        """Export the mesh to a file using ``trimesh``."""
        mesh = self._make_trimesh()
        mesh.export(path)

    def decimate(self, target_tri: int) -> "TriMesh":
        """Reduce the number of triangles in the mesh to the target count."""
        assert target_tri < self[1].shape[0]
        cache_path = self.cache_path(f"decimate__{target_tri}")
        cached = self.load_cache(cache_path)
        if cached is None:
            if self[1].shape[1] != 3:
                raise Exception("Only triangle meshes are supported")
            mesh = self._make_trimesh()
            mesh = mesh.simplify_quadric_decimation(face_count=target_tri)
            vert = np.asarray(mesh.vertices)
            tri = np.asarray(mesh.faces)
            vert, tri = _fix_skinny_triangles(vert, tri)
            return TriMesh.create(
                vert,
                tri,
                self.cache_dir,
            ).save_cache(cache_path)
        else:
            return cached

    def subdivide(self, n: int = 1, method: str = "midpoint"):
        """Subdivide the mesh with the given number of iterations and method."""
        import trimesh.remesh

        cache_path = self.cache_path(f"subdiv__{method}__{n}")
        cached = self.load_cache(cache_path)
        if cached is None:
            if self[1].shape[1] != 3:
                raise Exception("Only triangle meshes are supported")
            if method == "midpoint":
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
        return _rust.mesh_polygon_area_2d(np.ascontiguousarray(pts, dtype=np.float64))

    def triangulate(self, target: int = 1024, min_angle: float = 20) -> "TriMesh":
        """Triangulate a closed 2D line shape."""
        area = 1.6 * self._compute_area(self[0]) / target
        cache_path = self.cache_path(f"triangulate__{area}_{min_angle}")
        cached = self.load_cache(cache_path)
        if cached is None:
            from triangle import triangulate

            if self[1].shape[1] != 2:
                raise Exception("Only line meshes are supported")

            args_str = _rust.mesh_format_triangulate_args(float(area), float(min_angle))
            t = triangulate(
                {"vertices": self[0], "segments": self[1]}, args_str
            )
            return TriMesh.create(
                t["vertices"], t["triangles"], self.cache_dir
            ).save_cache(cache_path)
        else:
            return cached

    def tetrahedralize(self, *args, **kwargs) -> TetMesh:
        """Tetrahedralize a surface triangle mesh using fTetWild."""
        status_callback = kwargs.pop("status_callback", None)
        status_interval = float(kwargs.pop("status_interval", 5.0))
        # Build the cache-key arg string in Rust to keep formatting
        # logic out of Python.
        arg_strs = [str(a) for a in args]
        kwarg_pairs = [(str(k), str(v)) for k, v in kwargs.items()]
        arg_str = _rust.mesh_tetrahedralize_arg_str(arg_strs, kwarg_pairs)
        cache_path = self.cache_path(
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
            cached_ver = int(data["map_version"]) if "map_version" in data else 0
            if cached_ver == _SURFACE_MAP_VERSION and "map_coefs" in data:
                tet_mesh.set_surface_mapping(
                    data["map_tri_indices"], data["map_coefs"]
                )
            elif "map_tri_indices" in data:
                new_tri_indices, new_coefs = _frame_mapping(
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
            import pytetwild  # noqa: F401

            # Whitelist + default fTetWild kwargs in Rust to keep the
            # subprocess script stable against stray keys from callers.
            _ftw_pairs_in = [(str(k), repr(v)) for k, v in kwargs.items()]
            ftw_pairs = _rust.mesh_ftetwild_kwargs(_ftw_pairs_in)
            kwargs_literal = ", ".join(f"{k}={v}" for k, v in ftw_pairs)

            # Run fTetWild in a subprocess to avoid holding the GIL.
            import subprocess as _sp
            import tempfile as _tf

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

            vert, tri, tet = _rust.mesh_tet_extract_surface(
                np.ascontiguousarray(vert, dtype=np.float64),
                np.ascontiguousarray(tet, dtype=np.int32),
                1e-15,
            )

            tri_indices, coefs = _frame_mapping(self[0], vert, tri)

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
        """Recompute the SHA-256 hash of the mesh from its vertex and element arrays."""
        # hashlib.sha256 stays Python (allowlist; Rust replacement not
        # worth a sha2/hex dep on the core crate).
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
        """Set the cache directory used by this mesh."""
        self.cache_dir = cache_dir
        return self

    def cache_path(self, name: str) -> str:
        """Compute a cache file path derived from the mesh hash and the given tag."""
        return _rust.mesh_cache_path(self.cache_dir, self.hash, name)

    def save_cache(self, path: str) -> "TriMesh":
        """Save the mesh's vertex and triangle arrays to the given ``.npz`` path."""
        np.savez(
            path,
            vert=self[0],
            tri=self[1],
        )
        return self

    def load_cache(self, path: str) -> Optional["TriMesh"]:
        """Load a cached mesh from the given path, or return ``None`` if it does not exist."""
        if os.path.exists(path):
            data = np.load(path)
            return TriMesh.create(data["vert"], data["tri"], self.cache_dir)
        else:
            return None

    def normalize(self) -> "TriMesh":
        """Return ``self`` after invoking :func:`normalize` on the vertex array."""
        self[0][:] = normalize(self[0])
        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Optional[float] = None,
        scale_z: Optional[float] = None,
    ) -> "TriMesh":
        """Scale the triangle mesh with the given scaling factors."""
        if scale_y is None:
            scale_y = scale_x
        if scale_z is None:
            scale_z = scale_x
        self[0][:] = scale(self[0], scale_x, scale_y, scale_z)
        return self
