# File: _scene_fixed_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""FixedScene: the immutable, validated result of :meth:`Scene.build`.

Split out of ``_scene_.py``. The class wraps the assembled mesh,
collider, pin, and stitch data, runs final validation through Rust,
and exposes preview/export helpers.
"""

import os
import shutil
from enum import Enum
from typing import Any, Optional

import numpy as np
from tqdm.auto import tqdm

from . import _rust  # type: ignore[attr-defined]

from ._plot_ import Plot, PlotManager
from ._render_ import MitsubaRenderer, Rasterizer
from ._scene_collider_ import Sphere, Wall
from ._scene_pin_ import (
    MoveByOperation,
    MoveToOperation,
    PinData,
    SpinData,
    TransformKeyframeOperation,
    _pin_to_toml_dict,
)
from ._scene_transform_ import TransformAnimation
from ._utils_ import Utils


class ValidationError(ValueError):
    """ValueError with structured violation data for visualization."""

    def __init__(self, message, violations=None):
        super().__init__(message)
        self.violations = violations or []


class EnumColor(Enum):
    """Dynamic face color enumeration."""

    NONE = 0
    AREA = 1


def _compute_triangle_areas_vectorized(vert: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Compute triangle areas (Rust-backed)."""
    return _rust.scene_triangle_areas(
        np.ascontiguousarray(vert, dtype=np.float64),
        np.ascontiguousarray(tri, dtype=np.int64),
    )


def _compute_area(vert: np.ndarray, tri: np.ndarray, area: np.ndarray):
    """Compute areas for all triangles and store in the provided array."""
    area[:] = _compute_triangle_areas_vectorized(vert, tri)


class FixedScene:
    """A fixed scene class.

    ``FixedScene`` is the immutable, validated result of :meth:`Scene.build`.
    Hand it to :meth:`SessionManager.create` to drive a solver run.

    Example:
        Build a scene, inspect it, then pass it into a session::

            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            fixed = scene.build().report()
            fixed.preview()
            session = app.session.create(fixed).build()
    """

    def __init__(
        self,
        plot: Optional[PlotManager],
        name: str,
        map_by_name: dict[str, list[int]],
        displacement: np.ndarray,
        vert: tuple[np.ndarray, np.ndarray],
        color: np.ndarray,
        dyn_face_color: list[EnumColor],
        dyn_face_intensity: list[float],
        vel: np.ndarray,
        uv: list[np.ndarray],
        rod: np.ndarray,
        tri: np.ndarray,
        tet: np.ndarray,
        rod_param: dict[str, list[Any]],
        tri_param: dict[str, list[Any]],
        tet_param: dict[str, list[Any]],
        wall: list[Wall],
        sphere: list[Sphere],
        rod_vert_range: tuple[int, int],
        shell_vert_range: tuple[int, int],
        rod_count: int,
        shell_count: int,
        tri_is_collider: np.ndarray,
        rod_is_collider: np.ndarray,
        pinned_vertices: Optional[set[int]] = None,
        static_vert_for_check: Optional[np.ndarray] = None,
        static_tri_for_check: Optional[np.ndarray] = None,
        surface_map_by_name: Optional[dict[str, tuple]] = None,
        concat_rest_vert: Optional[np.ndarray] = None,
        rest_vert_mask: Optional[np.ndarray] = None,
    ):
        """Initialize the fixed scene.

        Args:
            plot (Optional[PlotManager]): The plot manager.
            name (str): The name of the scene.
            map_by_name (dict[str, list[int]]): Mapping from object name to per-object vertex index arrays.
            displacement (np.ndarray): The per-object displacement vectors.
            vert (tuple[np.ndarray, np.ndarray]): The vertices of the scene. The first array is the displacement map reference, the second is local positions.
            color (np.ndarray): The colors of the vertices.
            dyn_face_color (list[EnumColor]): The dynamic face colors.
            dyn_face_intensity (list[float]): The dynamic face color intensities.
            vel (np.ndarray): The velocities of the vertices.
            uv (list[np.ndarray]): The per-face UV coordinates for shell faces.
            rod (np.ndarray): The rod elements.
            tri (np.ndarray): The triangle elements.
            tet (np.ndarray): The tetrahedral elements.
            rod_param (dict[str, list[Any]]): The parameters for the rod elements.
            tri_param (dict[str, list[Any]]): The parameters for the triangle elements.
            tet_param (dict[str, list[Any]]): The parameters for the tetrahedral elements.
            wall (list[Wall]): The invisible walls.
            sphere (list[Sphere]): The invisible spheres.
            rod_vert_range (tuple[int, int]): The index range of the rod vertices.
            shell_vert_range (tuple[int, int]): The index range of the shell vertices.
            rod_count (int): The number of rod elements.
            shell_count (int): The number of shell elements.
            tri_is_collider (np.ndarray): Boolean array indicating collider triangles.
            rod_is_collider (np.ndarray): Boolean array indicating collider rod edges.
            pinned_vertices (Optional[set[int]]): Set of pinned vertex indices. Used to skip checking pinned vertices against invisible colliders.
            static_vert_for_check (Optional[np.ndarray]): Static mesh vertices for intersection check.
            static_tri_for_check (Optional[np.ndarray]): Static mesh triangles for intersection check.
            surface_map_by_name (Optional[dict[str, tuple]]): Frame-embedding surface maps for tetrahedralized objects.
            concat_rest_vert (Optional[np.ndarray]): Concatenated rest-shape vertices for objects whose pins release at ``unpin_time``.
            rest_vert_mask (Optional[np.ndarray]): Per-vertex uint8 mask marking entries in ``concat_rest_vert`` that are valid.
        """

        self._map_by_name = map_by_name
        self._plot = plot
        self._name = name
        self._displacement = displacement
        self._vert = vert
        self._color = color
        self._dyn_face_color = dyn_face_color
        self._dyn_face_intensity = dyn_face_intensity
        self._vel = vel
        self._velocity_schedules = {}
        self._collision_windows_data = {}
        self._uv = uv
        self._rod = rod
        self._tri = tri
        self._tet = tet
        self._rod_param = rod_param
        self._tri_param = tri_param
        self._tet_param = tet_param
        self._concat_rest_vert = concat_rest_vert
        self._rest_vert_mask = rest_vert_mask
        self._pin: list[PinData] = []
        self._spin: list[SpinData] = []
        self._static_vert = (np.zeros(0, dtype=np.uint32), np.zeros(0))
        self._static_color = np.zeros((0, 0))
        self._static_tri = np.zeros((0, 0))
        self._stitch_ind = np.zeros((0, 0))
        self._stitch_w = np.zeros((0, 0))
        self._static_param = {}
        self._static_transform_animations: list[tuple[int, TransformAnimation]] = []
        self._excluded_from_output: set[str] = set()
        self._wall = wall
        self._sphere = sphere
        self._rod_vert_range = rod_vert_range
        self._shell_vert_range = shell_vert_range
        self._rod_count = rod_count
        self._shell_count = shell_count
        self._has_dyn_color = any(entry != EnumColor.NONE for entry in dyn_face_color)

        self._surface_map_by_name = surface_map_by_name

        # Violation flags - set during validation checks
        self._has_self_intersection = False
        self._has_contact_offset_violation = False
        self._has_wall_violation = False
        self._has_sphere_violation = False

        assert len(self._vert[0]) == len(self._color)
        assert len(self._vert[1]) == len(self._color)
        assert len(self._tri) == len(self._dyn_face_color)
        assert len(self._uv) == shell_count

        for key, value in self._rod_param.items():
            if value:
                assert len(value) == len(self._rod), (
                    f"{key} has {len(value)} entries, but rod has {len(self._rod)} rods"
                )
        for key, value in self._tri_param.items():
            if value:
                assert len(value) == len(self._tri), (
                    f"{key} has {len(value)} entries, but tri has {len(self._tri)} faces"
                )
        for key, value in self._tet_param.items():
            if value:
                assert len(value) == len(self._tet), (
                    f"{key} has {len(value)} entries, but tet has {len(self._tet)} tets"
                )

        # ------------------------------------------------------------------
        # Validation + derived data: hand the entire constructor body to
        # the Rust kernel `scene_fixed_scene_assemble`:
        #   * self-intersection scan over (dynamic + static) tris,
        #   * rod-tri contact-offset pre-check (raises early on fatal),
        #   * contact-offset scan over the unified element namespace,
        #   * wall + sphere scans (skip pinned),
        #   * raise ValidationError with a structured violations payload,
        #   * compute self._area + self._face_to_vert_weights.
        n_tris = len(self._tri)
        n_rods = len(self._rod)
        tri_offset = self._tri_param.get("contact-offset", [])
        rod_offset = self._rod_param.get("contact-offset", [])

        # Filter walls/spheres to the "static" subset (single-keyframe
        # only). Each filter is a single list comprehension; the kernel
        # consumes the list directly. Kinematic colliders are checked
        # elsewhere.
        def _wall_static_entry(w):
            entry = w.get_entry()
            if not entry or len(entry) > 1:
                return None
            return (list(entry[0][0]), list(getattr(w, "normal", [0.0, 1.0, 0.0])))

        wall_data: list[tuple[list[float], list[float]]] = [
            d for d in (_wall_static_entry(w) for w in wall) if d is not None
        ]

        def _sphere_static_entry(s):
            entry = s.get_entry()
            if not entry or len(entry) > 1:
                return None
            pos, radius, _ = entry[0]
            return (
                list(pos),
                float(radius),
                bool(s.is_inverted),
                bool(s.is_hemisphere),
            )

        sphere_data: list[tuple[list[float], float, bool, bool]] = [
            d for d in (_sphere_static_entry(s) for s in sphere) if d is not None
        ]

        pinned_list: list[int] = (
            [int(i) for i in pinned_vertices] if pinned_vertices else []
        )

        # Visible progress for the BVH-based scene checks. The Rust
        # kernel runs every phase in one call, so we can't surface
        # per-phase progress without a Python callback bridge. Print a
        # tqdm bar that names the phases as they advance.
        checks_pbar = tqdm(
            total=3,
            desc="scene checks",
            bar_format="{desc}: {n_fmt}/{total_fmt} {postfix}",
        )
        checks_pbar.set_postfix_str("self-intersection BVH")
        result = _rust.scene_fixed_scene_assemble(
            np.ascontiguousarray(self._vert[0], dtype=np.uint32),
            np.ascontiguousarray(self._vert[1], dtype=np.float64),
            np.ascontiguousarray(self._displacement, dtype=np.float64),
            np.ascontiguousarray(self._tri, dtype=np.int32)
            if n_tris > 0
            else np.zeros((0, 3), dtype=np.int32),
            np.ascontiguousarray(self._rod, dtype=np.int32)
            if n_rods > 0
            else np.zeros((0, 2), dtype=np.int32),
            np.ascontiguousarray(tri_is_collider, dtype=bool),
            np.ascontiguousarray(rod_is_collider, dtype=bool),
            list(tri_offset) if tri_offset else [],
            list(rod_offset) if rod_offset else [],
            np.ascontiguousarray(static_vert_for_check, dtype=np.float64)
            if static_vert_for_check is not None
            else None,
            np.ascontiguousarray(static_tri_for_check, dtype=np.int32)
            if static_tri_for_check is not None
            else None,
            pinned_list,
            wall_data,
            sphere_data,
            bool(self._has_dyn_color),
        )
        checks_pbar.update(1)
        checks_pbar.set_postfix_str("rod-tri offset")
        checks_pbar.update(1)
        checks_pbar.set_postfix_str("contact-offset proximity")
        checks_pbar.update(1)
        checks_pbar.close()

        self._has_self_intersection = bool(result["has_self_intersection"])
        self._has_contact_offset_violation = bool(result["has_contact_offset_violation"])
        self._has_wall_violation = bool(result["has_wall_violation"])
        self._has_sphere_violation = bool(result["has_sphere_violation"])

        n_self = int(result.get("n_self_intersections_total", 0))
        n_tri_tri = int(result.get("n_tri_tri", 0))
        n_rod_tri = int(result.get("n_rod_tri", 0))
        n_off = int(result.get("n_contact_offset_total", 0))
        if n_self == 0 and n_off == 0:
            print("scene checks: clean (0 self-intersections, 0 contact-offset)")
        else:
            parts = []
            if n_self > 0:
                parts.append(
                    f"{n_self} self-intersections ({n_tri_tri} tri-tri, "
                    f"{n_rod_tri} rod-tri)"
                )
            if n_off > 0:
                parts.append(f"{n_off} contact-offset violations")
            print(f"scene checks found: {'; '.join(parts)}")

        all_violations = result["violations"]
        if all_violations:
            raise ValidationError(result["combined_message"], violations=all_violations)

        self._area = np.asarray(result["area"], dtype=np.float64)
        ftvw = result["face_to_vert_weights"]
        self._face_to_vert_weights = (
            np.asarray(ftvw, dtype=np.float64) if ftvw is not None else None
        )

    @property
    def tri_param(self) -> dict[str, list[Any]]:
        """Get the triangle parameters.

        Example:
            Inspect per-triangle parameter arrays captured at build time::

                fixed = scene.build()
                for key, values in fixed.tri_param.items():
                    print(key, len(values))
        """
        return self._tri_param

    @property
    def has_violations(self) -> bool:
        """Check if the scene has any violations that prevent simulation.

        Example:
            Guard the solver launch behind a validation check::

                fixed = scene.build()
                if fixed.has_violations:
                    print(fixed.get_violation_messages())
        """
        return _rust.scene_fixed_scene_has_violations(
            bool(self._has_self_intersection),
            bool(self._has_contact_offset_violation),
            bool(self._has_wall_violation),
            bool(self._has_sphere_violation),
        )

    def get_violation_messages(self) -> list[str]:
        """Get a list of violation messages for the scene.

        Example:
            Print any validation violations before running the solver::

                for msg in fixed.get_violation_messages():
                    print(msg)
        """
        return list(_rust.scene_violation_messages(
            bool(self._has_self_intersection),
            bool(self._has_contact_offset_violation),
            bool(self._has_wall_violation),
            bool(self._has_sphere_violation),
        ))

    def report(self) -> "FixedScene":
        """Print a summary of the scene.

        Returns:
            FixedScene: The fixed scene (for chaining).

        Example:
            Chain a summary printout into the build step::

                fixed = scene.build().report()
                fixed.preview()
        """
        n_pin = sum([len(pin.index) for pin in self._pin]) if len(self._pin) else 0
        n_static_vert = len(self._static_vert[1]) if len(self._static_vert) else 0
        n_static_tri = len(self._static_tri) if len(self._static_tri) else 0
        entries = _rust.scene_fixed_scene_report_entries(
            int(len(self._vert[1])),
            int(len(self._rod)),
            int(len(self._tri)),
            int(len(self._tet)),
            int(n_pin),
            int(n_static_vert),
            int(n_static_tri),
            int(len(self._stitch_ind)),
            int(len(self._stitch_w)),
        )
        data = {label: [value] for label, value in entries}

        from IPython.display import HTML, display

        from ._utils_ import dict_to_html_table

        if self._plot is not None and self._plot.is_jupyter_notebook():
            html = dict_to_html_table(data, classes="table")
            display(HTML(html))
        else:
            print(data)
        return self

    def color(self, vert: np.ndarray, hint: Optional[dict] = None) -> np.ndarray:
        """Compute the per-vertex color for the scene given a vertex array.

        Args:
            vert (np.ndarray): The current vertex positions.
            hint (dict, optional): Optional hints for the color computation (e.g. ``"max-area"``). Defaults to None.

        Returns:
            np.ndarray: The per-vertex colors of the scene.

        Example:
            Compute colors for the current vertex positions before previewing::

                fixed = scene.build()
                V = fixed.vertex()
                colors = fixed.color(V, hint={"max-area": 2.0})
                fixed.preview(V)
        """
        if hint is None:
            hint = {}
        if self._has_dyn_color:
            assert self._face_to_vert_weights is not None
            assert self._area is not None

            max_area = 2.0

            if "max-area" in hint:
                max_area = hint["max-area"]

            v = np.ascontiguousarray(vert, dtype=np.float64)
            tris = np.ascontiguousarray(self._tri, dtype=np.int64)
            init = np.ascontiguousarray(self._area, dtype=np.float64)
            weights = np.ascontiguousarray(
                self._face_to_vert_weights, dtype=np.float64
            )
            dfc = np.ascontiguousarray(
                [0 if c == EnumColor.NONE else 1 for c in self._dyn_face_color],
                dtype=np.uint8,
            )
            dfi = np.ascontiguousarray(self._dyn_face_intensity, dtype=np.float64)
            base = np.ascontiguousarray(self._color, dtype=np.float64)
            return _rust.scene_dynamic_color(
                v, tris, init, weights, dfc, dfi, base, float(max_area),
            )
        else:
            return self._color

    def vertex(self, transform: bool = True) -> np.ndarray:
        """Get the vertices of the scene.

        Args:
            transform (bool, optional): Whether to transform the vertices. Defaults to True.

        Returns:
            np.ndarray: The vertices of the scene.

        Example:
            Fetch the initial world-space vertex positions::

                vert = fixed.vertex()
                print(vert.shape)
        """
        if transform:
            return self._vert[1] + self._displacement[self._vert[0]]
        else:
            return self._vert[1]

    def export(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        path: str,
        include_static: bool = True,
        args: Optional[dict] = None,
        delete_exist: bool = False,
    ) -> "FixedScene":
        """Export the scene to a mesh file.

        The vertices and vertex colors must be supplied explicitly so callers
        can pass time-evaluated positions.

        Args:
            vert (np.ndarray): The vertices of the scene.
            color (np.ndarray): The colors of the vertices.
            path (str): The path to the mesh file. Supported formats include ``.ply`` and ``.obj``.
            include_static (bool, optional): Whether to include the static mesh. Defaults to True.
            args (dict, optional): Additional arguments passed to the renderer.
            delete_exist (bool, optional): Whether to delete any existing file at the path. Defaults to False.

        Returns:
            FixedScene: The fixed scene.

        Example:
            Write out the scene as a .ply at the initial time::

                vert = fixed.vertex()
                color = fixed.color(vert)
                fixed.export(vert, color, "/tmp/scene.ply", delete_exist=True)
        """

        if args is None:
            args = {}
        image_path = path + ".png"
        if delete_exist:
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(image_path):
                os.remove(image_path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        seg, tri = self._rod, None
        if not os.path.exists(path) or not os.path.exists(image_path):
            if include_static and len(self._static_vert) and len(self._static_tri):
                static_vert = (
                    self._static_vert[1] + self._displacement[self._static_vert[0]]
                )
                tri = np.concatenate([self._tri, self._static_tri + len(vert)])
                vert = np.concatenate([vert, static_vert], axis=0)
                color = np.concatenate([color, self._static_color], axis=0)
            else:
                tri = self._tri

        if tri is not None and len(tri) == 0:
            tri = np.array([[0, 0, 0]])

        # Check if rendering should be skipped (e.g., on Windows headless)
        skip_render = args.get("skip_render", False)

        # Export mesh file (also in CI mode when skip_render is set)
        if not os.path.exists(path) and (Utils.ci_name() is None or skip_render):
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=vert, faces=tri, vertex_colors=color, process=False
            )
            mesh.export(path)

        # Skip rendering if skip_render is set
        if not skip_render and not os.path.exists(image_path):
            renderer_type = args.get("renderer", "software")
            if Utils.ci_name() is not None:
                args["width"] = 320
                args["height"] = 240
            if renderer_type == "mitsuba":
                assert shutil.which("mitsuba") is not None
                renderer = MitsubaRenderer(args)
            elif renderer_type == "software":
                renderer = Rasterizer(args)
            else:
                raise Exception("unsupported renderer")

            assert tri is not None
            assert color is not None
            renderer.render(vert, color, seg, tri, image_path)

        return self

    def export_fixed(self, path: str, delete_exist: bool) -> "FixedScene":
        """Export the fixed scene into a set of data files that are read by the simulator.

        Args:
            path (str): The path to the output directory.
            delete_exist (bool): Whether to delete the existing directory.

        Returns:
            FixedScene: The fixed scene.

        Example:
            Typically invoked internally by :meth:`Session.build`, but can be
            called directly to materialize the solver's data directory::

                fixed = scene.build()
                fixed.export_fixed("/tmp/my_scene_data", delete_exist=True)
        """

        steps = 14
        pbar = tqdm(total=steps, desc="build session")

        if os.path.exists(path):
            if delete_exist:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            else:
                raise Exception(f"file {path} already exists")
        else:
            os.makedirs(path)
        pbar.update(1)

        from . import _cbor_bridge_ as _cbor

        map_path = os.path.join(path, "map.pickle")
        with open(map_path, "wb") as f:
            exported_map = {k: v for k, v in self._map_by_name.items()
                           if k not in self._excluded_from_output}
            f.write(_cbor.dumps_envelope(_cbor.KIND_VERTEX_MAP, exported_map))
        if self._surface_map_by_name:
            # Wire format v2: frame-embedding coefs replace barycentric weights.
            # Wrapped in a versioned envelope so legacy clients (which expected
            # a bare dict of bary maps) fail loudly instead of silently
            # applying the wrong reconstruction math.
            surface_map_path = os.path.join(path, "surface_map.pickle")
            with open(surface_map_path, "wb") as f:
                f.write(_cbor.dumps_envelope(
                    _cbor.KIND_SURFACE_MAP,
                    {"version": 2, "maps": self._surface_map_by_name},
                ))
        pbar.update(1)

        info_path = os.path.join(path, "info.toml")
        # Pin to UTF-8: the Rust solver reads info.toml with
        # fs::read_to_string() which requires UTF-8. Python's default
        # text mode picks up the locale encoding (cp1252 on Windows),
        # so a Blender object name like "BézierCurve" gets written
        # as Latin-1 (é = 0xe9) and then panics the solver with
        # "stream did not contain valid UTF-8".
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("[count]\n")
            f.write(f"vert = {len(self._vert[1])}\n")
            f.write(f"rod = {len(self._rod)}\n")
            f.write(f"tri = {len(self._tri)}\n")
            f.write(f"tet = {len(self._tet)}\n")
            f.write(f"static_vert = {len(self._static_vert[1])}\n")
            f.write(f"static_tri = {len(self._static_tri)}\n")
            f.write(f"pin_block = {len(self._pin)}\n")
            f.write(f"wall = {len(self._wall)}\n")
            f.write(f"sphere = {len(self._sphere)}\n")
            f.write(f"stitch = {len(self._stitch_ind)}\n")
            f.write(f"rod_vert_start = {self._rod_vert_range[0]}\n")
            f.write(f"rod_vert_end = {self._rod_vert_range[1]}\n")
            f.write(f"shell_vert_start = {self._shell_vert_range[0]}\n")
            f.write(f"shell_vert_end = {self._shell_vert_range[1]}\n")
            f.write(f"rod_count = {self._rod_count}\n")
            f.write(f"shell_count = {self._shell_count}\n")
            f.write(f"has_rest_vert = {str(self._concat_rest_vert is not None).lower()}\n")
            f.write("\n")

            if self._static_transform_animations:
                total_kf = sum(len(a.times) for _, a in self._static_transform_animations)
                f.write(_rust.scene_format_static_transform_header(
                    len(self._static_transform_animations),
                    int(total_kf),
                    [len(a.times) for _, a in self._static_transform_animations],
                    [len(a.local_vert) for _, a in self._static_transform_animations],
                    [int(off) for off, _ in self._static_transform_animations],
                ))

            # Pin TOML: build pin/op descriptor dicts, then format the
            # whole block in one Rust call (no per-line write loop).
            f.write(_rust.scene_format_pin_toml(
                [_pin_to_toml_dict(pin) for pin in self._pin]
            ))

            # Wall TOML: pre-render each param value with Python's str()
            # to preserve the heterogeneous int/float/string formatting,
            # then let Rust stitch the section. Same idea for spheres.
            f.write(_rust.scene_format_wall_toml([
                {
                    "keyframe": len(wall.entry),
                    "normal": [float(wall.normal[0]), float(wall.normal[1]), float(wall.normal[2])],
                    "transition": wall.transition,
                    "params": [(str(k), str(v)) for k, v in wall.param.list().items()],
                }
                for wall in self._wall
            ]))

            f.write(_rust.scene_format_sphere_toml([
                {
                    "keyframe": len(sphere.entry),
                    "is_hemisphere": bool(sphere.is_hemisphere),
                    "is_inverted": bool(sphere.is_inverted),
                    "transition": sphere.transition,
                    "params": [(str(k), str(v)) for k, v in sphere.param.list().items()],
                }
                for sphere in self._sphere
            ]))
        pbar.update(1)

        bin_path = os.path.join(path, "bin")
        os.makedirs(bin_path)
        param_path = os.path.join(bin_path, "param")
        os.makedirs(param_path)
        pbar.update(1)

        def export_param(param: dict[str, list[Any]], basepath: str, name: str):
            """Export parameters to a binary file."""
            for key, value in param.items():
                if value:
                    filepath = os.path.join(basepath, f"{name}-{key}.bin")
                    if key == "model":
                        model_map = {
                            "arap": 0,
                            "stvk": 1,
                            "baraff-witkin": 2,
                            "snhk": 3,
                        }
                        assert all(name in model_map for name in value)
                        np.array(
                            [model_map[name] for name in value], dtype=np.uint8
                        ).tofile(filepath)
                    else:
                        np.array(value, dtype=np.float32).tofile(filepath)

        self._displacement.astype(np.float64).tofile(
            os.path.join(bin_path, "displacement.bin")
        )
        self._vert[0].astype(np.uint32).tofile(os.path.join(bin_path, "vert_dmap.bin"))
        self._vert[1].astype(np.float64).tofile(os.path.join(bin_path, "vert.bin"))
        if self._concat_rest_vert is not None:
            self._concat_rest_vert.astype(np.float64).tofile(
                os.path.join(bin_path, "rest_vert.bin")
            )
            self._rest_vert_mask.astype(np.uint8).tofile(
                os.path.join(bin_path, "rest_vert_mask.bin")
            )
        self._color.astype(np.float32).tofile(os.path.join(bin_path, "color.bin"))
        self._vel.astype(np.float32).tofile(os.path.join(bin_path, "vel.bin"))

        # Write velocity schedule and collision windows as dyn_param entries
        dyn_entries = {}
        dyn_entries.update(self._velocity_schedules)
        # Collision windows: each entry is a (t_start, t_end) pair written as 2 floats
        for key, windows in self._collision_windows_data.items():
            dyn_entries[key] = windows

        if dyn_entries:
            dyn_path = os.path.join(path, "dyn_param.txt")
            mode = "a" if os.path.exists(dyn_path) else "w"
            # Rust formats the entire file body in one shot. Each entry
            # is normalized into one of two tuple shapes (velocity
            # `(t, [vx, vy, vz])` or collision-window `(t_start, t_end)`)
            # via a list comprehension (single C-level allocation, no
            # dynamic list growth); the kernel discriminates on shape.
            def _normalize_entry(entry):
                if isinstance(entry[1], (list, tuple)):
                    t, vel = entry
                    return (float(t), [float(vel[0]), float(vel[1]), float(vel[2])])
                return (float(entry[0]), float(entry[1]))

            blocks = [
                (
                    str(key),
                    [_normalize_entry(e) for e in entries
                     if isinstance(e, (list, tuple)) and len(e) == 2],
                )
                for key, entries in dyn_entries.items()
            ]
            with open(dyn_path, mode) as f:
                f.write(_rust.scene_format_dyn_param_toml(blocks))

        pbar.update(1)

        if self._uv:
            with open(os.path.join(bin_path, "uv.bin"), "wb") as f:
                for uv in self._uv:
                    uv.astype(np.float32).tofile(f)
        pbar.update(1)

        if len(self._rod):
            self._rod.astype(np.uint64).tofile(os.path.join(bin_path, "rod.bin"))
            export_param(self._rod_param, param_path, "rod")
        pbar.update(1)

        if len(self._tri):
            self._tri.astype(np.uint64).tofile(os.path.join(bin_path, "tri.bin"))
            export_param(self._tri_param, param_path, "tri")
        pbar.update(1)

        if len(self._tet):
            self._tet.astype(np.uint64).tofile(os.path.join(bin_path, "tet.bin"))
            export_param(self._tet_param, param_path, "tet")
        pbar.update(1)

        if len(self._static_vert[0]):
            self._static_vert[0].astype(np.uint32).tofile(
                os.path.join(bin_path, "static_vert_dmap.bin")
            )
            self._static_vert[1].astype(np.float64).tofile(
                os.path.join(bin_path, "static_vert.bin")
            )
            self._static_tri.astype(np.uint64).tofile(
                os.path.join(bin_path, "static_tri.bin")
            )
            self._static_color.astype(np.float32).tofile(
                os.path.join(bin_path, "static_color.bin")
            )
            export_param(self._static_param, param_path, "static")
        if self._static_transform_animations:
            all_local = np.vstack([a.local_vert for _, a in self._static_transform_animations])
            all_local.astype(np.float64).tofile(
                os.path.join(bin_path, "static_local_vert.bin")
            )
            # Concat per-anim keyframes via numpy.concatenate (single C
            # memcpy per stream, no Python list growth). Each anim's
            # `.times` is a list[float] and the trans/quat/scale entries
            # are 1D numpy arrays; we flatten then stack in one shot.
            kf_counts = [len(a.times) for _, a in self._static_transform_animations]
            all_times = np.concatenate([
                np.asarray(a.times, dtype=np.float64)
                for _, a in self._static_transform_animations
            ])
            all_trans = np.concatenate([
                np.asarray(a.translations, dtype=np.float64).reshape(-1, 3)
                for _, a in self._static_transform_animations
            ])
            all_quats = np.concatenate([
                np.asarray(a.quaternions, dtype=np.float64).reshape(-1, 4)
                for _, a in self._static_transform_animations
            ])
            all_scales = np.concatenate([
                np.asarray(a.scales, dtype=np.float64).reshape(-1, 3)
                for _, a in self._static_transform_animations
            ])
            # Rust validates total_kf alignment between the four streams.
            _rust.scene_concat_static_transform_anims(
                kf_counts,
                np.ascontiguousarray(all_times, dtype=np.float64),
                np.ascontiguousarray(all_trans, dtype=np.float64),
                np.ascontiguousarray(all_quats, dtype=np.float64),
                np.ascontiguousarray(all_scales, dtype=np.float64),
            )
            all_times.astype(np.float64).tofile(
                os.path.join(bin_path, "static_transform_time.bin")
            )
            all_trans.astype(np.float64).tofile(
                os.path.join(bin_path, "static_transform_translation.bin")
            )
            all_quats.astype(np.float64).tofile(
                os.path.join(bin_path, "static_transform_quaternion.bin")
            )
            all_scales.astype(np.float64).tofile(
                os.path.join(bin_path, "static_transform_scale.bin")
            )
        pbar.update(1)

        if len(self._stitch_ind) and len(self._stitch_w):
            self._stitch_ind.astype(np.uint64).tofile(
                os.path.join(bin_path, "stitch_ind.bin")
            )
            self._stitch_w.astype(np.float32).tofile(
                os.path.join(bin_path, "stitch_w.bin")
            )
        pbar.update(1)

        for i, pin in enumerate(self._pin):
            # Write pin indices
            with open(os.path.join(bin_path, f"pin-ind-{i}.bin"), "wb") as f:
                np.array(pin.index, dtype=np.uint64).tofile(f)

            # Write operation data
            for j, op in enumerate(pin.operations):
                if isinstance(op, MoveByOperation):
                    # MoveBy operations need to write position delta to binary file
                    op_path = os.path.join(bin_path, f"pin-{i}-op-{j}.bin")
                    with open(op_path, "wb") as f:
                        np.array(op.delta, dtype=np.float64).tofile(f)
                elif isinstance(op, MoveToOperation):
                    # MoveTo operations need to write target positions to binary file
                    op_path = os.path.join(bin_path, f"pin-{i}-op-{j}.bin")
                    with open(op_path, "wb") as f:
                        np.array(op.target, dtype=np.float64).tofile(f)
                elif isinstance(op, TransformKeyframeOperation):
                    # TRS keyframes: separate binaries for verts, times, TRS
                    # arrays, and per-segment interpolation metadata.
                    base = os.path.join(bin_path, f"pin-{i}-op-{j}")
                    np.asarray(op.local_vert, dtype=np.float64).tofile(base + ".bin")
                    np.asarray(op.times, dtype=np.float64).tofile(
                        base + "-time.bin"
                    )
                    np.asarray(op.translations, dtype=np.float64).tofile(
                        base + "-translation.bin"
                    )
                    np.asarray(op.quaternions, dtype=np.float64).tofile(
                        base + "-quaternion.bin"
                    )
                    np.asarray(op.scales, dtype=np.float64).tofile(
                        base + "-scale.bin"
                    )
                    # Per-segment: 1 byte interp code (0=linear, 1=bezier,
                    # 2=constant) and 4 f64 handles. Written as two files to
                    # avoid struct padding headaches. Defaults stay
                    # Python-side because dict.get(...) reads
                    # heterogeneous keys; the kernel just packs the
                    # resolved values into the two contiguous buffers.
                    segs = op.segments
                    interp_strs = [s.get("interpolation", "LINEAR") for s in segs]
                    handles_right = np.asarray(
                        [s.get("handle_right", [1/3, 0.0]) for s in segs],
                        dtype=np.float64,
                    ).reshape(-1, 2)
                    handles_left = np.asarray(
                        [s.get("handle_left", [2/3, 1.0]) for s in segs],
                        dtype=np.float64,
                    ).reshape(-1, 2)
                    interp_codes, handles = _rust.scene_pack_transform_keyframe_segments(
                        interp_strs,
                        np.ascontiguousarray(handles_right, dtype=np.float64),
                        np.ascontiguousarray(handles_left, dtype=np.float64),
                    )
                    interp_codes.tofile(base + "-interp.bin")
                    handles.tofile(base + "-handles.bin")
                # Spin and Scale operations have all data in info.toml
        pbar.update(1)

        for i, wall in enumerate(self._wall):
            with open(os.path.join(bin_path, f"wall-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _ in wall.entry for p in pos], dtype=np.float64
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"wall-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, t in wall.entry], dtype=np.float64)
                timing.tofile(f)
        pbar.update(1)

        for i, sphere in enumerate(self._sphere):
            with open(os.path.join(bin_path, f"sphere-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _, _ in sphere.entry for p in pos], dtype=np.float64
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"sphere-radius-{i}.bin"), "wb") as f:
                radius = np.array([r for _, r, _ in sphere.entry], dtype=np.float32)
                radius.tofile(f)
            with open(os.path.join(bin_path, f"sphere-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, _, t in sphere.entry], dtype=np.float64)
                timing.tofile(f)
        pbar.update(1)
        pbar.close()
        return self

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the bounding box of the scene.

        Returns:
            tuple[np.ndarray, np.ndarray]: The maximum and minimum coordinates of the bounding box.

        Example:
            Print the extents of the built scene::

                hi, lo = fixed.bbox()
                print("size:", hi - lo)
        """
        lv = np.ascontiguousarray(self._vert[1], dtype=np.float64)
        idx = np.ascontiguousarray(self._vert[0], dtype=np.int64)
        disp = np.ascontiguousarray(self._displacement, dtype=np.float64)
        hi, lo = _rust.scene_bbox_displaced(lv, idx, disp)
        return (hi, lo)

    def center(self) -> np.ndarray:
        """Compute the area-weighted center of the scene.

        Returns:
            np.ndarray: The area-weighted center of the scene.

        Example:
            Aim a camera at the scene's area-weighted center::

                target = fixed.center()
                fixed.preview(options={"lookat": target.tolist()})
        """
        vert = self._vert[1] + self._displacement[self._vert[0]]
        v = np.ascontiguousarray(vert, dtype=np.float64)
        t = np.ascontiguousarray(self._tri, dtype=np.int64)
        cx, cy, cz = _rust.scene_area_weighted_center(v, t)
        return np.array([cx, cy, cz])

    def _average_tri_area(self) -> float:
        """Compute the average triangle area of the scene.

        Returns:
            float: The average triangle area of the scene.
        """
        return _rust.scene_average_tri_area(
            np.ascontiguousarray(self._area, dtype=np.float64)
        )

    def set_pin(self, pin: list[PinData]):
        """Set the pinning data of all the objects.

        Args:
            pin (list[PinData]): A list of pinning data.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject pinning data into a fixed scene::

                fixed = scene.build()
                fixed.set_pin(list_of_pin_data)
        """
        self._pin = pin

    def set_spin(self, spin: list[SpinData]):
        """Set the spinning data of all the objects.

        Args:
            spin (list[SpinData]): A list of spinning data.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject spin data::

                fixed = scene.build()
                fixed.set_spin(list_of_spin_data)
        """
        self._spin = spin

    def set_static(
        self,
        vert: tuple[np.ndarray, np.ndarray],
        tri: np.ndarray,
        color: np.ndarray,
        param: dict[str, list[Any]],
        transform_animations: Optional[list[tuple[int, TransformAnimation]]] = None,
    ):
        """Set the static mesh data.

        Args:
            vert (tuple[np.ndarray, np.ndarray]): The vertices of the static mesh. The first array is the displacement map reference; the second is local positions.
            tri (np.ndarray): The triangle elements of the static mesh.
            color (np.ndarray): The colors of the static mesh.
            param (dict[str, list[Any]]): Parameters for the static mesh elements.
            transform_animations: Optional list of ``(vert_offset, TransformAnimation)`` for animated static objects.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to attach a static collider mesh::

                fixed = scene.build()
                fixed.set_static((ref, V), F, colors, {"area": areas})
        """
        self._static_vert = vert
        self._static_tri = tri
        self._static_color = color
        self._static_param = param
        if transform_animations:
            self._static_transform_animations = transform_animations

    def set_stitch(self, ind: np.ndarray, w: np.ndarray):
        """Set the stitch data.

        Args:
            ind (np.ndarray): The stitch indices.
            w (np.ndarray): The stitch weights.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject stitch constraints::

                fixed = scene.build()
                fixed.set_stitch(stitch_indices, stitch_weights)
        """
        self._stitch_ind = ind
        self._stitch_w = w

    def time(self, time: float) -> np.ndarray:
        """Compute the vertex positions at a specific time.

        Args:
            time (float): The time to compute the vertex positions.

        Returns:
            np.ndarray: The vertex positions at the specified time.

        Example:
            Evaluate the scene midway through a pin animation::

                vert_mid = fixed.time(0.5)
                print(vert_mid.shape)
        """
        vert = self._vert[1].copy()
        initial = self._vert[1]

        for pin in self._pin:
            # Reset to initial before applying ops (matches Rust solver
            # which computes position = initial + ops independently per pin)
            vert[pin.index] = initial[pin.index]
            for op in pin.operations:
                vert[pin.index] = op.apply(vert[pin.index], time)

        vert += time * self._vel
        vert += self._displacement[self._vert[0]]
        return vert

    def static_time(self, time: float) -> np.ndarray:
        """Compute animated static vertex positions at a specific time.

        Args:
            time (float): The time to evaluate.

        Returns:
            np.ndarray: The static vertex positions at the specified time.

        Example:
            Sample the animated collider mesh halfway through the animation::

                fixed = scene.build()
                V_static = fixed.static_time(0.5)
                print(V_static.shape)
        """
        if not self._static_transform_animations:
            if len(self._static_vert[1]) == 0:
                return np.zeros((0, 3))
            return self._static_vert[1] + self._displacement[self._static_vert[0]]
        base = self._static_vert[1] + self._displacement[self._static_vert[0]]
        result = base.copy()
        for vert_offset, anim in self._static_transform_animations:
            n = len(anim.local_vert)
            result[vert_offset:vert_offset + n] = anim.evaluate(time)
        return result

    def preview(
        self,
        vert: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        show_slider: bool = True,
        engine: str = "threejs",
    ) -> Optional["Plot"]:
        """Preview the scene.

        Args:
            vert (Optional[np.ndarray], optional): The vertices to preview. Defaults to None, in which case ``self.vertex()`` is used.
            options (dict, optional): The options for the plot. Defaults to None.
            show_slider (bool, optional): Whether to show the time slider. Defaults to True.
            engine (str, optional): The rendering engine. Defaults to ``"threejs"``.

        Returns:
            Optional[Plot]: The plot object if in a Jupyter notebook, otherwise None.

        Example:
            Preview the initial scene with a custom camera and no pin markers::

                opts = {"eye": [0, 1.4, 2.5], "pin": False, "wireframe": True}
                fixed.preview(options=opts)
        """
        if options is None:
            options = {}
        default_opts = {
            "flat_shading": False,
            "wireframe": True,
            "stitch": True,
            "pin": True,
        }
        options = dict(options)
        for key, value in default_opts.items():
            if key not in options:
                options[key] = value

        if self._plot is not None and self._plot.is_jupyter_notebook():
            if vert is None:
                vert = self.vertex()
            assert vert is not None
            color = self.color(vert, options)
            assert len(color) == len(vert)
            tri = self._tri.copy()
            edge = self._rod.copy()
            pts = np.zeros(0)
            plotter = self._plot.create(engine)

            n_dynamic_vert = len(vert)
            has_static = len(self._static_vert[1]) > 0 or self._static_transform_animations
            if has_static:
                static_vert = self.static_time(0.0)
                static_color = np.zeros_like(static_vert)
                static_color[:, :] = self._static_color
                if len(tri):
                    tri = np.vstack([tri, self._static_tri + len(vert)])
                else:
                    tri = self._static_tri + len(vert)
                vert = np.vstack([vert, static_vert])
                color = np.vstack([color, static_color])
            assert vert is not None and color is not None
            assert len(color) == len(vert)

            if options["stitch"] and len(self._stitch_ind) and len(self._stitch_w):
                # 4D format: ind=[src, t0, t1, t2], w=[ws, w1, w2, w3].
                # Rust kernel returns absolute edge indices (n_vert + 2*i,
                # n_vert + 2*i + 1) so the appended edge buffer is ready.
                stitch_vert, stitch_edge = _rust.scene_stitch_preview_lines(
                    np.ascontiguousarray(vert, dtype=np.float64),
                    np.ascontiguousarray(self._stitch_ind, dtype=np.int64).reshape(-1, 4),
                    np.ascontiguousarray(self._stitch_w, dtype=np.float64).reshape(-1, 4),
                )
                stitch_color = np.tile(np.array([1.0, 1.0, 1.0]), (len(stitch_vert), 1))
                vert = np.vstack([vert, stitch_vert])
                edge = np.vstack([edge, stitch_edge]) if len(edge) else stitch_edge
                color = np.vstack([color, stitch_color])

            if options["pin"] and self._pin:
                # Triangle area is the natural marker scale for surface
                # meshes; for rod-only scenes, fall back to a fraction of
                # the average rod edge length so pin dots stay visible.
                if len(self._area):
                    options["pts_scale"] = float(np.sqrt(self._area.mean()))
                elif len(self._rod):
                    rest = self._vert[1]
                    diff = rest[self._rod[:, 0]] - rest[self._rod[:, 1]]
                    options["pts_scale"] = 0.5 * float(
                        np.linalg.norm(diff, axis=1).mean()
                    )
                # Static-moving pin-shells pin every vertex as an
                # implementation detail; hide those dots in preview.
                # Rust kernel filters by hide_in_preview and concatenates
                # in one pass.
                pts = _rust.scene_collect_pin_marker_indices(
                    [bool(pin.hide_in_preview) for pin in self._pin],
                    [list(pin.index) for pin in self._pin],
                )

            plotter.plot(vert, color, tri, edge, pts, options)

            has_vel = np.linalg.norm(self._vel) > 0
            has_static_anim = bool(self._static_transform_animations)
            if show_slider and (self._pin or has_vel or has_static_anim):
                max_time = 0
                if self._pin:
                    for pin in self._pin:
                        for op in pin.operations:
                            _, t_end = op.get_time_range()
                            if t_end == float("inf"):
                                max_time = max(max_time, 1.0)
                            else:
                                max_time = max(max_time, t_end)
                if has_vel:
                    max_time = max(max_time, 1.0)
                for _, anim in self._static_transform_animations:
                    max_time = max(max_time, anim.max_time())
                if max_time > 0:

                    def update(time=0):
                        dyn_vert = self.time(time)
                        if has_static:
                            static_v = self.static_time(time)
                            combined = np.vstack([dyn_vert, static_v])
                        else:
                            combined = dyn_vert
                        plotter.update(combined)

                    from ipywidgets import interact

                    interact(update, time=(0, max_time, 0.01))
            return plotter
        else:
            return None
