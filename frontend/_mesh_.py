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
#   * `pytetwild.tetrahedralize` (fTetWild backend; subprocess, Python
#     script gen) and `tetgen.TetGen` (TetGen backend; in-process,
#     surface-preserving).
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

# Shared remediation hint appended to every "no enclosed volume" /
# tetrahedralization failure raise so the user sees consistent guidance
# regardless of which check tripped (zero tets, TetGen exception, or the
# 1-1 surface-preservation checks).
_SHELL_NOT_SOLID_HINT = (
    "Assign thin, open, coplanar, or non-manifold surfaces to a SHELL "
    "group instead of SOLID, or use the fTetWild backend."
)

# Empirical target-density factor for triangulate: scales the requested
# triangle count into a per-triangle area budget. Keep the value stable to
# preserve triangulate cache keys (it feeds the cache-path string).
_TRIANGULATE_AREA_SCALE = 1.6

# Minimum |V| below which a tet is treated as numerically degenerate during
# surface extraction. Matches the Rust default in mesh_py.rs and is shared by
# both the fTetWild and TetGen backends so the two cannot drift apart.
_MIN_TET_VOLUME = 1e-15


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

    def set_cache_dir(self, path: str):
        """Redirect the cache directory for both this manager and its
        :class:`CreateManager`. The latter is required because ``create.tri``
        derives the tetra cache path from ``CreateManager._cache_dir``, so
        updating ``self._cache_dir`` alone would leave the tetra cache pointed
        at the original location.
        """
        self._cache_dir = path
        self._create._cache_dir = path

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

    def cylinder(self, r: float, min_x: float, max_x: float, n: int) -> "TriMesh":
        """Create a cylinder along the x-axis."""
        dx, ny, dy = _rust.mesh_cylinder_dx_ny_dy(
            float(min_x), float(max_x), int(n), float(r)
        )
        V = _rust.mesh_generate_cylinder_verts(int(n), int(ny), float(min_x), float(dx), float(dy), float(r))
        F = _rust.mesh_generate_cylinder_faces(int(n), int(ny))
        return TriMesh.create(V, F, self._cache_dir)

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
        current_tri = self[1].shape[0]
        if target_tri >= current_tri:
            # Nothing to reduce: the requested count meets or exceeds the
            # current triangle count, so return the mesh unchanged rather
            # than asking trimesh to "decimate" upward.
            return self
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
        area = _TRIANGULATE_AREA_SCALE * self._compute_area(self[0]) / target
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
        """Tetrahedralize a surface triangle mesh.

        The ``backend`` kwarg selects the mesher:

        * ``"ftetwild"`` (default): fTetWild via :mod:`pytetwild`. Tolerant
          of open, cracked, or non-manifold input, but it resamples the
          surface, so the original vertices survive only through a
          frame-embedding surface map.
        * ``"tetgen"``: TetGen with boundary-preserving switches
          (``nobisect`` / ``-Y``). The input surface is carried through
          unchanged, giving an exact 1-1 map from input vertices to the
          tet surface (asserted in :meth:`_assert_tetgen_surface_unchanged`).
          Requires a clean, closed, manifold input.

        The ``status_interval`` kwarg (default 5.0s) controls how often
        ``status_callback`` emits a progress line; it applies only to the
        fTetWild backend, which runs in a polled subprocess. The TetGen
        backend runs synchronously in-process and emits a single status, so
        ``status_interval`` has no effect there.
        """
        status_callback = kwargs.pop("status_callback", None)
        status_interval = float(kwargs.pop("status_interval", 5.0))
        # fTetWild wall-clock guard (opt-in), popped here BEFORE the cache-key
        # arg string so they never change the cache key. Default None ->
        # resolved from PPF_FTETWILD_TIMEOUT / PPF_FTETWILD_RETRIES in the
        # subprocess driver. fTetWild backend only (TetGen runs in-process).
        _ftw_timeout = kwargs.pop("timeout", None)
        _ftw_retries = kwargs.pop("retries", None)
        # Build the cache-key arg string in Rust to keep formatting
        # logic out of Python. ``backend`` (when present) rides in the
        # kwargs here, so fTetWild and TetGen results cache under distinct
        # keys; an fTetWild-default mesh carries no ``backend`` kwarg and
        # so keeps its pre-existing cache key.
        arg_strs = [str(a) for a in args]
        kwarg_pairs = [(str(k), str(v)) for k, v in kwargs.items()]
        arg_str = _rust.mesh_tetrahedralize_arg_str(arg_strs, kwarg_pairs)
        cache_path = self.cache_path(
            f"{self.hash}_tetrahedralize_{arg_str}.npz"
        )
        backend = str(kwargs.pop("backend", "ftetwild")).lower()
        # Source of truth for the surface-map format version. Imported lazily
        # so this module doesn't pull cbor2 at import time. The npz key name
        # ``map_version`` stays unchanged for on-disk back-compat.
        from ._cbor_bridge_ import SURFACE_MAP_VERSION as _SURFACE_MAP_VERSION

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
            if not tet_mesh.has_surface_mapping():
                # No current-version map was attached above. This covers an
                # older-version cache (map_tri_indices present but a stale
                # map_version) as well as a legacy cache that wrote only
                # vert/tet/tri with no map keys at all. Rebuild the map from
                # scratch and re-save with the current version so later
                # interpolate_surface / decoder surface reconstruction works.
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
            if backend == "tetgen":
                vert, tri, tet = self._tetrahedralize_tetgen(
                    kwargs, status_callback
                )
            elif backend == "ftetwild":
                vert, tri, tet = self._tetrahedralize_ftetwild(
                    kwargs, status_callback, status_interval,
                    timeout=_ftw_timeout, retries=_ftw_retries,
                )
            else:
                raise ValueError(
                    f"Unknown tetrahedralize backend {backend!r}; "
                    "expected 'ftetwild' or 'tetgen'."
                )

            # Both backends can return zero tets (or only-degenerate tets
            # that were filtered during surface extraction) when the input
            # mesh has no enclosed volume, e.g. a single flat plane assigned
            # to a SOLID group. Without this guard the next call would feed
            # an empty surface into ``frame_mapping`` and panic with
            # "index out of bounds: the len is 0 but the index is 0".
            if tet.shape[0] == 0:
                raise ValueError(
                    "Tetrahedralization produced no tetrahedra: the input mesh "
                    "has no enclosed volume (likely flat, coplanar, or "
                    "non-manifold). " + _SHELL_NOT_SOLID_HINT
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

    def _tetrahedralize_ftetwild(
        self, kwargs, status_callback, status_interval,
        timeout=None, retries=None,
    ):
        """Tetrahedralize with fTetWild (pytetwild) in a subprocess.

        Returns ``(vert, tri, tet)`` after degenerate-tet filtering and
        surface extraction. fTetWild resamples the surface, so the input
        vertices are recovered later through the frame-embedding map.

        The temp .npz files and the child process are cleaned up in a
        ``finally`` block so a failure inside the savez, the poll loop (e.g.
        a raising ``status_callback``), or the result load cannot leak a
        temp file or orphan the fTetWild child.
        """
        # Run fTetWild in a subprocess to avoid holding the GIL.
        import tempfile as _tf

        with _tf.NamedTemporaryFile(suffix=".npz", delete=False) as _in_f:
            input_path = _in_f.name
        output_path = input_path + ".out.npz"

        # Cleared up front so the finally never reaps a child from a prior
        # call; _run_ftetwild_subprocess assigns it once the child spawns.
        self._ftw_proc = None
        try:
            np.savez(input_path, vert=self[0], tri=self[1])
            return self._run_ftetwild_subprocess(
                input_path,
                output_path,
                kwargs,
                status_callback,
                status_interval,
                timeout=timeout,
                retries=retries,
            )
        finally:
            # Reap a still-running child (e.g. a status_callback raised
            # mid-poll) so it does not orphan, then remove both temp files
            # exactly once whatever path we leave through.
            proc = self._ftw_proc
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            self._ftw_proc = None
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.unlink(p)

    def _run_ftetwild_subprocess(
        self, input_path, output_path, kwargs, status_callback,
        status_interval, timeout=None, retries=None,
    ):
        """Spawn the fTetWild subprocess and return the extracted surface.

        Temp-file and child-process cleanup is handled by the caller's
        ``finally`` block; this method only drives the subprocess and loads
        its result.

        ``timeout`` bounds each attempt's wall-clock. fTetWild's C++ call
        cannot be interrupted from Python, so a stalled run (a nondeterministic
        pathological case, seen hanging ~400s on CI) would otherwise burn the
        caller's whole per-scenario budget. On timeout the child is killed and
        the run is retried up to ``retries`` times (a fresh run clears the
        stall in practice); if every attempt times out a clear ``RuntimeError``
        is raised. ``timeout=None`` disables the cap (unbounded historical
        behavior), so nothing changes unless a caller or the env opts in.
        """
        import pytetwild  # noqa: F401
        import subprocess as _sp

        # Resolve the cap + retry budget: explicit arg wins, else the env
        # knobs (so CI can bound a hang without a code change), else no cap /
        # one retry. Retries only apply when a timeout is set (a non-timeout
        # failure raises immediately).
        if timeout is None:
            _env_t = os.environ.get("PPF_FTETWILD_TIMEOUT")
            timeout = float(_env_t) if _env_t else None
        if retries is None:
            _env_r = os.environ.get("PPF_FTETWILD_RETRIES")
            retries = int(_env_r) if _env_r else 1

        # Whitelist + default fTetWild kwargs in Rust to keep the
        # subprocess script stable against stray keys from callers.
        _ftw_pairs_in = [(str(k), repr(v)) for k, v in kwargs.items()]
        ftw_pairs = _rust.mesh_ftetwild_kwargs(_ftw_pairs_in)
        kwargs_literal = ", ".join(f"{k}={v}" for k, v in ftw_pairs)

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
        # One attempt normally; with a timeout, up to retries+1 tries,
        # re-spawning a fresh child after each stall.
        attempts = (retries + 1) if timeout is not None else 1
        last_timeout_err = None
        for attempt in range(attempts):
            proc = _sp.Popen(
                [sys.executable, "-c", tet_script],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            # Expose the child so the caller's finally can reap it if the poll
            # loop below leaves through an exception (e.g. a raising callback).
            self._ftw_proc = proc
            start_time = time.time()
            timed_out = False
            while proc.poll() is None:
                try:
                    proc.wait(timeout=status_interval)
                except _sp.TimeoutExpired:
                    pass
                elapsed = time.time() - start_time
                if (
                    timeout is not None
                    and elapsed > timeout
                    and proc.poll() is None
                ):
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    break
                if proc.poll() is None and status_callback is not None:
                    status_callback(f"running fTetWild ({elapsed:.0f}s elapsed)")

            if timed_out:
                last_timeout_err = RuntimeError(
                    f"fTetWild subprocess timed out after {timeout:.0f}s "
                    f"(attempt {attempt + 1}/{attempts}); killed"
                )
                if status_callback is not None:
                    _more = attempt + 1 < attempts
                    status_callback(
                        f"fTetWild timed out after {timeout:.0f}s; "
                        + ("retrying" if _more else "giving up")
                    )
                continue

            if proc.returncode != 0:
                raise RuntimeError(
                    f"fTetWild subprocess failed (exit code {proc.returncode})"
                )

            # NpzFile keeps the .npz open (a ZipFile handle); on Windows that
            # blocks unlink with WinError 32. Use `with` so the handle is
            # closed before the caller's finally tries to delete the file.
            with np.load(output_path) as out_data:
                vert = out_data["vert"]
                tet = out_data["tet"]

            return _rust.mesh_tet_extract_surface(
                np.ascontiguousarray(vert, dtype=np.float64),
                np.ascontiguousarray(tet, dtype=np.int32),
                _MIN_TET_VOLUME,
            )

        # Every attempt hit the wall-clock cap.
        raise last_timeout_err

    def _tetrahedralize_tetgen(self, kwargs, status_callback):
        """Tetrahedralize with TetGen, preserving the input surface 1-1.

        ``nobisect`` (``-Y``) forbids Steiner points on the input
        boundary, so every input vertex survives and the surface stays
        identical; quality refinement only inserts interior points. The
        result is verified to be an exact 1-1 input-to-surface map.

        Returns ``(vert, tri, tet)`` after degenerate-tet filtering and
        surface extraction.
        """
        import tetgen

        if status_callback is not None:
            status_callback("running TetGen")

        V = np.ascontiguousarray(self[0], dtype=np.float64)
        F = np.ascontiguousarray(self[1], dtype=np.int32)
        n_input = len(V)

        tg_kwargs = {"order": 1, "nobisect": True, "quality": True}
        if "min_ratio" in kwargs:
            tg_kwargs["minratio"] = float(kwargs["min_ratio"])
        if "max_volume" in kwargs:
            tg_kwargs["maxvolume"] = float(kwargs["max_volume"])

        tg = tetgen.TetGen(V, F)
        # tetgen returns (nodes, elems, ...); index rather than unpack so
        # it survives the extra trailing arrays newer versions return.
        try:
            if sys.platform == "win32":
                result = tg.tetrahedralize(**tg_kwargs)
            else:
                # Suppress verbose C++ output from TetGen, matching fTetWild.
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                old1, old2 = os.dup(1), os.dup(2)
                try:
                    os.dup2(devnull_fd, 1)
                    os.dup2(devnull_fd, 2)
                    result = tg.tetrahedralize(**tg_kwargs)
                finally:
                    os.dup2(old1, 1)
                    os.dup2(old2, 2)
                    os.close(devnull_fd)
                    os.close(old1)
                    os.close(old2)
        except Exception as e:
            # TetGen rejects coplanar / open / non-manifold input outright
            # (e.g. "All vertices are coplanar"). Re-raise as ValueError so
            # the decoder prepends the object name and the message matches
            # the fTetWild zero-volume guidance.
            raise ValueError(
                f"TetGen failed to tetrahedralize the input surface "
                f"({type(e).__name__}: {e}). The TetGen backend needs a "
                "clean, closed, manifold mesh with enclosed volume. "
                + _SHELL_NOT_SOLID_HINT
            ) from e
        vert, tet = result[0], result[1]

        vert, tri, tet = _rust.mesh_tet_extract_surface(
            np.ascontiguousarray(vert, dtype=np.float64),
            np.ascontiguousarray(tet, dtype=np.int32),
            _MIN_TET_VOLUME,
        )
        self._assert_tetgen_surface_unchanged(vert, tri, V, n_input)
        return vert, tri, tet

    def _assert_tetgen_surface_unchanged(self, vert, tri, input_vert, n_input):
        """Verify TetGen carried the input surface through as an exact 1-1 map.

        The contract the user relies on: every input vertex coincides with
        exactly one tet-surface vertex and the surface has no extra
        vertices, so ``frame_mapping`` reconstructs each input position
        exactly (it lands on a surface-triangle corner). This is
        order-independent: it compares the surface vertex set against the
        input vertex set rather than assuming TetGen kept the input
        ordering.
        """
        surf_used = np.unique(tri)
        if len(surf_used) != n_input:
            raise ValueError(
                "TetGen did not preserve the input surface 1-1: the tet "
                f"surface references {len(surf_used)} vertices but the input "
                f"surface has {n_input}. The TetGen backend needs a clean, "
                "closed, manifold mesh. " + _SHELL_NOT_SOLID_HINT
            )
        surf_vert = vert[surf_used]
        si = np.lexsort((surf_vert[:, 2], surf_vert[:, 1], surf_vert[:, 0]))
        ii = np.lexsort((input_vert[:, 2], input_vert[:, 1], input_vert[:, 0]))
        if not np.allclose(surf_vert[si], input_vert[ii], rtol=0.0, atol=1e-9):
            raise ValueError(
                "TetGen moved input surface vertices, so the map to the input "
                "mesh is not an exact 1-1 correspondence. " + _SHELL_NOT_SOLID_HINT
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
