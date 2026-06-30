# File: _scene_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Scene + Object orchestration.

This module is the public entry point for the scene API. It defines
:class:`Scene`, :class:`FixedScene` (the immutable, validated result of
:meth:`Scene.build`), :class:`EnumColor`, and :class:`ValidationError`.
Lighter helpers live in sibling modules:

* ``_scene_collider_``: :class:`Wall`, :class:`Sphere` and their params.
* ``_scene_transform_``: quaternion / TRS helpers and :class:`TransformAnimation`.
* ``_scene_pin_``: pin operations, :class:`PinHolder`, :class:`PinData`.

Everything is re-exported here for backward compatibility (older code and
tests import names directly from ``frontend._scene_``).
"""

import colorsys
import os
import shutil
from dataclasses import replace as _dc_replace
from enum import Enum
from typing import Any, Optional

import numpy as np
from tqdm.auto import tqdm

from . import _rust  # type: ignore[attr-defined]

from ._asset_ import AssetManager
from ._param_ import ParamHolder
from ._plot_ import Plot, PlotManager
from ._render_ import MitsubaRenderer, Rasterizer
from ._scene_collider_ import Sphere, Wall
from ._scene_pin_ import (
    MoveByOperation,
    MoveToOperation,
    PinData,
    PinHolder,
    TorqueOperation,
    TransformKeyframeOperation,
    _pin_to_toml_dict,
)
from ._scene_transform_ import (
    TransformAnimation,
    _apply_transform_to_verts,
    _axis_angle_to_quat,
    _mat3_to_quat,
    _quat_multiply,
    _quat_slerp,
    _quat_to_mat3,
)
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
        rest_vert_anim: Optional[np.ndarray] = None,
        rest_vert_times: Optional[np.ndarray] = None,
        bend_rest_vert: Optional[np.ndarray] = None,
        bend_rest_vert_mask: Optional[np.ndarray] = None,
        pdrd_body_rows: Optional[list[float]] = None,
        pdrd_vert_index: Optional[np.ndarray] = None,
        pdrd_vert_list: Optional[list[int]] = None,
        pdrd_rest_centered: Optional[np.ndarray] = None,
        sand_param: Optional[dict[str, list[Any]]] = None,
        quiet: bool = False,
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
            rest_vert_mask (Optional[np.ndarray]): Per-vertex uint8 mask marking entries in ``concat_rest_vert`` (and ``rest_vert_anim``) that are valid.
            rest_vert_anim (Optional[np.ndarray]): Frame-major time-varying rest shape, shape ``(n_frames * n_vert, 3)``, from a captured pull-pin deformation. The solver recomputes ``inv_rest`` per frame and interpolates.
            rest_vert_times (Optional[np.ndarray]): Keyframe times in seconds, length ``n_frames``, aligned to ``rest_vert_anim``.
            bend_rest_vert (Optional[np.ndarray]): Concatenated reference vertices for the bending rest angle, one row per global vertex (unmasked rows equal the initial vert). The solver computes hinge rest angles from these positions for masked objects.
            bend_rest_vert_mask (Optional[np.ndarray]): Per-vertex uint8 mask marking rows of ``bend_rest_vert`` that belong to an object with an enabled reference rest angle.
            quiet (bool): When True, suppress the scene-check summary prints. Defaults to False, preserving the interactive diagnostic output.
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
        # Scalar SAND (faceless point cloud) params. None for a scene with no
        # points object; exported on the "sand-" channel (one float per key).
        self._sand_param = sand_param
        self._concat_rest_vert = concat_rest_vert
        self._rest_vert_mask = rest_vert_mask
        self._rest_vert_anim = rest_vert_anim
        self._rest_vert_times = rest_vert_times
        self._bend_rest_vert = bend_rest_vert
        self._bend_rest_vert_mask = bend_rest_vert_mask
        # PDRD per-body data, written to bin files at export time.
        # `pdrd_body_rows` is a flat list of 23 f32 per body (see the
        # PDRD body table builder); empty when the scene has no PDRD bodies.
        self._pdrd_body_rows = pdrd_body_rows or []
        self._pdrd_vert_index = pdrd_vert_index
        self._pdrd_vert_list = pdrd_vert_list or []
        self._pdrd_rest_centered = pdrd_rest_centered
        self._pin: list[PinData] = []
        self._static_vert = (np.zeros(0, dtype=np.uint32), np.zeros(0))
        self._static_color = np.zeros((0, 0))
        self._static_tri = np.zeros((0, 0))
        self._stitch_ind = np.zeros((0, 0))
        self._stitch_w = np.zeros((0, 0))
        self._stitch_stiffness = np.zeros(0)
        self._static_param = {}
        self._static_transform_animations: list[tuple[int, TransformAnimation]] = []
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
            if not w.is_static_collider():
                return None
            entry = w.get_entry()
            return (list(entry[0][0]), list(getattr(w, "normal", [0.0, 1.0, 0.0])))

        wall_data: list[tuple[list[float], list[float]]] = [
            d for d in (_wall_static_entry(w) for w in wall) if d is not None
        ]

        def _sphere_static_entry(s):
            if not s.is_static_collider():
                return None
            pos, radius, _ = s.get_entry()[0]
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
        # Per-dynamic-tri 1-based PDRD body id (0 = not rigid), keyed on each
        # triangle's first vertex (every vertex of a rigid body shares its
        # id). Two triangles of the same body are an intra-rigid-body self-
        # intersection the solver tolerates (a rigid body never deforms), so
        # the self-intersection scan skips them while still reporting rigid-
        # vs-external overlaps. Absent PDRD bodies -> all zeros (old behavior).
        if self._pdrd_vert_index is not None and n_tris > 0:
            tri_body_id = np.ascontiguousarray(
                np.asarray(self._pdrd_vert_index)[
                    np.ascontiguousarray(self._tri, dtype=np.int64)[:, 0]
                ],
                dtype=np.int32,
            )
        else:
            tri_body_id = np.zeros(n_tris, dtype=np.int32)
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
            tri_body_id,
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
        if not quiet:
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
        return any([
            bool(self._has_self_intersection),
            bool(self._has_contact_offset_violation),
            bool(self._has_wall_violation),
            bool(self._has_sphere_violation),
        ])

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
        n_static_vert = len(self._static_vert[1])
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

        args = dict(args) if args is not None else {}
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
            if include_static and len(self._static_vert[1]) and len(self._static_tri):
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

    def export_fixed(
        self, path: str, delete_exist: bool, preserve_output: bool = False
    ) -> "FixedScene":
        """Export the fixed scene into a set of data files that are read by the simulator.

        Args:
            path (str): The path to the output directory.
            delete_exist (bool): Whether to delete the existing directory.
            preserve_output (bool): When True and ``delete_exist`` is set,
                skip the solver ``output/`` child while clearing the
                directory, so a resume can re-decode edited scene input
                without losing the saved checkpoints.

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
                    if preserve_output and item == "output":
                        continue
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
            f.write(_cbor.dumps_envelope(_cbor.KIND_VERTEX_MAP, self._map_by_name))
        if self._surface_map_by_name:
            # Wire format v2: frame-embedding coefs replace barycentric weights.
            # Wrapped in a versioned envelope so legacy clients (which expected
            # a bare dict of bary maps) fail loudly instead of silently
            # applying the wrong reconstruction math.
            surface_map_path = os.path.join(path, "surface_map.pickle")
            with open(surface_map_path, "wb") as f:
                f.write(_cbor.dumps_envelope(
                    _cbor.KIND_SURFACE_MAP,
                    {
                        "version": _cbor.SURFACE_MAP_VERSION,
                        "maps": self._surface_map_by_name,
                    },
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
            f.write(
                f"has_rest_vert_anim = {str(self._rest_vert_anim is not None).lower()}\n"
            )
            n_rest_frames = (
                len(self._rest_vert_times) if self._rest_vert_times is not None else 0
            )
            f.write(f"rest_vert_anim_frames = {n_rest_frames}\n")
            f.write(
                f"has_bend_rest_vert = {str(self._bend_rest_vert is not None).lower()}\n"
            )
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
                        # Authoritative name->id table lives in the Rust
                        # core (ppf_cts_core::datamodel::elastic_model) and is
                        # shared with the solver's reader, so the numeric
                        # encoding can't drift. Unknown names raise ValueError.
                        np.array(
                            [_rust.scene_model_name_to_id(name) for name in value],
                            dtype=np.uint8,
                        ).tofile(filepath)
                    else:
                        np.array(value, dtype=np.float32).tofile(filepath)

        self._displacement.astype(np.float64).tofile(
            os.path.join(bin_path, "displacement.bin")
        )
        self._vert[0].astype(np.uint32).tofile(os.path.join(bin_path, "vert_dmap.bin"))
        self._vert[1].astype(np.float64).tofile(os.path.join(bin_path, "vert.bin"))
        # rest_vert_mask is shared between the static rest_vert and the
        # time-varying rest_vert_anim, so write it whenever either is present.
        if self._rest_vert_mask is not None:
            self._rest_vert_mask.astype(np.uint8).tofile(
                os.path.join(bin_path, "rest_vert_mask.bin")
            )
        if self._concat_rest_vert is not None:
            self._concat_rest_vert.astype(np.float64).tofile(
                os.path.join(bin_path, "rest_vert.bin")
            )
        if self._rest_vert_anim is not None:
            self._rest_vert_anim.astype(np.float64).tofile(
                os.path.join(bin_path, "rest_vert_anim.bin")
            )
            self._rest_vert_times.astype(np.float64).tofile(
                os.path.join(bin_path, "rest_vert_times.bin")
            )
        # Bending reference rest angle: full per-vertex reference set plus the
        # per-vertex mask of which objects opted in. Written together so the
        # solver can build MeshSet.bend_rest_vertex and gate the from-geometry
        # rest-angle path on the mask.
        if self._bend_rest_vert is not None:
            self._bend_rest_vert.astype(np.float64).tofile(
                os.path.join(bin_path, "bend_rest_vert.bin")
            )
            if self._bend_rest_vert_mask is not None:
                self._bend_rest_vert_mask.astype(np.uint8).tofile(
                    os.path.join(bin_path, "bend_rest_vert_mask.bin")
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

        # ``_uv`` is the per-triangle UV layout: a single (n_tri, 3, 2) float32
        # array (or, when loaded from an older pickle, a list of (3, 2)
        # arrays). ``len()`` guards the empty case for both. Writing each
        # triangle individually used to issue ~1.8M ``tofile`` calls and
        # dominated the export (~15s); ``np.concatenate`` flattens both shapes
        # to identical, back-to-back float32 bytes for a single write.
        if len(self._uv):
            uv_all = np.ascontiguousarray(np.concatenate(self._uv), dtype=np.float32)
            uv_all.tofile(os.path.join(bin_path, "uv.bin"))
        pbar.update(1)

        if len(self._rod):
            self._rod.astype(np.uint64).tofile(os.path.join(bin_path, "rod.bin"))
            export_param(self._rod_param, param_path, "rod")
        pbar.update(1)

        if len(self._tri):
            self._tri.astype(np.uint64).tofile(os.path.join(bin_path, "tri.bin"))
            export_param(self._tri_param, param_path, "tri")
        pbar.update(1)

        # PDRD per-body and per-vertex tables. Absent files mean the
        # scene has no PDRD bodies; the solver tolerates that path.
        if self._pdrd_body_rows:
            np.asarray(self._pdrd_body_rows, dtype=np.float32).tofile(
                os.path.join(bin_path, "pdrd_body.bin")
            )
            assert self._pdrd_vert_index is not None
            assert self._pdrd_rest_centered is not None
            self._pdrd_vert_index.astype(np.uint32).tofile(
                os.path.join(bin_path, "pdrd_vert_index.bin")
            )
            np.asarray(self._pdrd_vert_list, dtype=np.uint32).tofile(
                os.path.join(bin_path, "pdrd_vert_list.bin")
            )
            np.ascontiguousarray(self._pdrd_rest_centered, dtype=np.float32).tofile(
                os.path.join(bin_path, "pdrd_rest_centered.bin")
            )

        if len(self._tet):
            self._tet.astype(np.uint64).tofile(os.path.join(bin_path, "tet.bin"))
            export_param(self._tet_param, param_path, "tet")
        pbar.update(1)

        # SAND scalar params: one float per key, written on the "sand-"
        # channel (sand-particle-mass.bin, sand-grain-radius.bin,
        # sand-contact-gap.bin, sand-friction.bin). Read by the solver outside
        # the per-element length assert. There are no sand element indices to
        # write; the grains are plain loose vertices already in vert.bin.
        if self._sand_param:
            export_param(self._sand_param, param_path, "sand")

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
            # Width-6 ABI contract: the solver reads stitch_ind.bin /
            # stitch_w.bin as 6-wide barycentric-barycentric rows. A wrong
            # width here silently mis-strides every stitch in the solver,
            # so guard the producer before the bin write (H2 tripwire).
            assert self._stitch_ind.shape[1] == 6, (
                "stitch_ind must be (M, 6); got {}".format(self._stitch_ind.shape)
            )
            assert self._stitch_w.shape[1] == 6, (
                "stitch_w must be (M, 6); got {}".format(self._stitch_w.shape)
            )
            self._stitch_ind.astype(np.uint64).tofile(
                os.path.join(bin_path, "stitch_ind.bin")
            )
            self._stitch_w.astype(np.float32).tofile(
                os.path.join(bin_path, "stitch_w.bin")
            )
            # Per-stitch-row stiffness (M,), parallel to stitch_ind/stitch_w.
            # Fall back to ones for legacy scenes built before per-object
            # stitch stiffness so the count still matches.
            n_stitch = self._stitch_ind.shape[0]
            stiffness = np.asarray(self._stitch_stiffness, dtype=np.float32)
            if stiffness.shape[0] != n_stitch:
                stiffness = np.ones(n_stitch, dtype=np.float32)
            stiffness.tofile(os.path.join(bin_path, "stitch_stiffness.bin"))
        pbar.update(1)

        for i, pin in enumerate(self._pin):
            # Write pin indices
            with open(os.path.join(bin_path, f"pin-ind-{i}.bin"), "wb") as f:
                np.array(pin.index, dtype=np.uint64).tofile(f)

            # Optional per-vertex pull weights (aligned to pin.index). The
            # solver probes for this file; absent means scalar pull / hard
            # pin as before.
            if getattr(pin, "pull_weights", None) is not None:
                with open(os.path.join(bin_path, f"pin-pullw-{i}.bin"), "wb") as f:
                    np.asarray(pin.pull_weights, dtype=np.float32).tofile(f)

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
        vert = self.vertex()
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

    def set_stitch(
        self, ind: np.ndarray, w: np.ndarray, stiffness: Optional[np.ndarray] = None
    ):
        """Set the stitch data.

        Args:
            ind (np.ndarray): The stitch indices.
            w (np.ndarray): The stitch weights.
            stiffness (Optional[np.ndarray]): Per-stitch-row stiffness (M,),
                resolved from each stitch's owning object. Defaults to all
                ones when omitted.

        Example:
            Typically invoked internally by :meth:`Scene.build`, but can be
            called directly to inject stitch constraints::

                fixed = scene.build()
                fixed.set_stitch(stitch_indices, stitch_weights)
        """
        self._stitch_ind = ind
        self._stitch_w = w
        if stiffness is None:
            stiffness = np.ones(len(ind), dtype=np.float32)
        self._stitch_stiffness = np.ascontiguousarray(stiffness, dtype=np.float32)

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
                # 6D format: ind=[s0, s1, s2, t0, t1, t2],
                # w=[ws0, ws1, ws2, wt0, wt1, wt2]. Rust kernel returns
                # absolute edge indices (n_vert + 2*i, n_vert + 2*i + 1) so
                # the appended edge buffer is ready.
                stitch_vert, stitch_edge = _rust.scene_stitch_preview_lines(
                    np.ascontiguousarray(vert, dtype=np.float64),
                    np.ascontiguousarray(self._stitch_ind, dtype=np.int64).reshape(-1, 6),
                    np.ascontiguousarray(self._stitch_w, dtype=np.float64).reshape(-1, 6),
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


# Imported after the classes above so ``_scene_object_`` (which imports
# EnumColor from this module) resolves cleanly during the import cycle.
from ._scene_object_ import Object


class SceneManager:
    """SceneManager class. Use this to manage scenes.

    Example:
        Create a scene through the app's scene manager and build it::

            app = App.create("demo")
            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            fixed = scene.build()
    """

    def __init__(self, plot: Optional[PlotManager], asset: AssetManager):
        """Initialize the scene manager."""
        self._plot = plot
        self._asset = asset
        self._scene: dict[str, Scene] = {}

    def create(self, name: str = "") -> "Scene":
        """Create a new scene.

        If a scene with the given name already exists, it is replaced.

        Args:
            name (str): The name of the scene to create. If empty, defaults to ``"scene"``.

        Returns:
            Scene: The created scene.

        Example:
            Create two named scenes side by side::

                cloth_scene = app.scene.create("cloth")
                rods_scene = app.scene.create("rods")
        """
        if name == "":
            name = "scene"

        if name in self._scene:
            del self._scene[name]

        scene = Scene(name, self._plot, self._asset)
        self._scene[name] = scene
        return scene

    def select(self, name: str, create: bool = True) -> "Scene":
        """Select a scene.

        If the scene exists, it is returned. If it does not exist and ``create`` is True, a new scene is created and returned.

        Args:
            name (str): The name of the scene to select.
            create (bool, optional): Whether to create a new scene if it does not exist. Defaults to True.

        Returns:
            Scene: The selected (or newly created) scene.

        Example:
            Fetch an existing scene by name, creating it lazily::

                scene = app.scene.select("cloth")
                scene.add("sheet")
        """
        if create and name not in self._scene:
            return self.create(name)
        else:
            return self._scene[name]

    def remove(self, name: str):
        """Remove a scene from the manager.

        Args:
            name (str): The name of the scene to remove.

        Example:
            Drop a scene that is no longer needed::

                app.scene.remove("cloth")
        """
        if name in self._scene:
            del self._scene[name]

    def clear(self):
        """Clear all the scenes in the manager.

        Example:
            Remove every scene before rebuilding from scratch::

                app.scene.clear()
                scene = app.scene.create()
        """
        self._scene = {}

    def list(self) -> list[str]:
        """List all the scenes in the manager.

        Returns:
            list[str]: A list of scene names.

        Example:
            Inspect which scenes are currently registered::

                for name in app.scene.list():
                    print(name)
        """
        return list(self._scene.keys())


class SceneInfo:
    """Lightweight metadata handle carrying the scene name.

    Example:
        Look up the name of the scene you are currently editing::

            print(scene.info.name)
    """

    def __init__(self, name: str, scene: "Scene"):
        self._scene = scene
        self.name = name


class InvisibleAdder:
    """Helper for attaching invisible colliders (walls and spheres) to a scene.

    Obtained via ``scene.add.invisible``.

    Example:
        Add a ground wall and a ball collider together::

            scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
            scene.add.invisible.sphere([0, 0.5, 0], 0.25)
    """

    def __init__(self, scene: "Scene"):
        self._scene = scene

    def sphere(self, position: list[float], radius: float) -> Sphere:
        """Add an invisible sphere to the scene.

        Args:
            position (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.
        Returns:
            Sphere: The invisible sphere.

        Example:
            Place an inverted hemispherical bowl under the cloth::

                scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
        """
        sphere = Sphere().add(position, radius)
        self._scene.sphere_list.append(sphere)
        return sphere

    def wall(self, position: list[float], normal: list[float]) -> Wall:
        """Add an invisible wall to the scene.

        Args:
            position (list[float]): The position of the wall.
            normal (list[float]): The outer normal of the wall.
        Returns:
            Wall: The invisible wall.

        Example:
            Seal a simulation box with four side walls::

                scene.add.invisible.wall([1, 0, 0], [-1, 0, 0])
                scene.add.invisible.wall([-1, 0, 0], [1, 0, 0])
                scene.add.invisible.wall([0, 0, 1], [0, 0, -1])
                scene.add.invisible.wall([0, 0, -1], [0, 0, 1])
        """
        wall = Wall().add(position, normal)
        self._scene.wall_list.append(wall)
        return wall


class ObjectAdder:
    """Factory for introducing meshes into a :class:`Scene`.

    Reached as ``scene.add``. Calling it returns an :class:`Object` that can
    be chained with transforms, pins, and colors. Invisible colliders live
    under ``scene.add.invisible``.

    Example:
        Drop a cloth sheet and an invisible ground wall into a scene::

            scene = app.scene.create()
            scene.add("sheet").at(0, 0.6, 0)
            scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
            fixed = scene.build()
    """

    def __init__(self, scene: "Scene"):
        self._scene = scene
        self.invisible = InvisibleAdder(
            scene
        )  #: InvisibleAdder: The invisible object adder.

    def __call__(self, mesh_name: str, ref_name: str = "") -> "Object":
        """Add a mesh to the scene.

        Args:
            mesh_name (str): The name of the mesh to add.
            ref_name (str, optional): The reference name of the object.

        Returns:
            Object: The added object.

        Example:
            Drop two instances of a registered asset into the scene,
            the second as a static collider::

                scene = app.scene.create()
                scene.add("sheet").at(0, 0.6, 0)
                scene.add("sphere").at(0, 0, 0).pin()  # static collider
                scene = scene.build()
        """
        if ref_name == "":
            ref_name = mesh_name
            count = 0
            while ref_name in self._scene.object_dict:
                count += 1
                ref_name = f"{mesh_name}_{count}"
        mesh_list = self._scene.asset_manager.list()
        if mesh_name not in mesh_list:
            raise Exception(f"mesh_name '{mesh_name}' does not exist")
        elif ref_name in self._scene.object_dict:
            raise Exception(f"ref_name '{ref_name}' already exists")
        else:
            obj = Object(self._scene.asset_manager, mesh_name)
            self._scene.object_dict[ref_name] = obj
            # Auto-set UV from asset if available
            asset_data = self._scene.asset_manager.fetch.get(mesh_name)
            if "UV" in asset_data and "F" in asset_data:
                vertex_uv = asset_data["UV"]
                faces = asset_data["F"]
                # Convert vertex UV to per-face UV via Rust kernel; emits
                # an (n_face, 3, 2) array. set_uv expects a list of (3,2)
                # arrays so we wrap with `list(...)` (an O(n) shallow
                # iteration over a numpy view, no per-element append).
                uv_arr = _rust.scene_face_uv_expand(
                    np.ascontiguousarray(vertex_uv, dtype=np.float64),
                    np.ascontiguousarray(faces, dtype=np.int64),
                )
                obj.set_uv(list(uv_arr))
            return obj


class Scene:
    """A scene class.

    A ``Scene`` collects objects, pins, invisible colliders, and stitch data,
    then compiles them into a :class:`FixedScene` via :meth:`build`.

    Example:
        Build a tiny drape scene from registered assets::

            scene = app.scene.create()
            sheet = scene.add("sheet").at(0, 0.6, 0)
            sheet.pin(sheet.grab([-1, 0, -1]) + sheet.grab([1, 0, -1]))
            scene.add("sphere").at(0, 0, 0).pin()
            fixed = scene.build().report()
    """

    def __init__(self, name: str, plot: Optional[PlotManager], asset: AssetManager):
        self._name = name
        self._plot = plot
        self._asset = asset
        self._object: dict[str, Object] = {}
        self._sphere: list[Sphere] = []
        self._wall: list[Wall] = []
        self._surface_map_by_name: dict[str, tuple] = {}
        self._cross_stitch: list[dict] = []
        self.add = ObjectAdder(self)  #: ObjectAdder: The object adder.
        self.info = SceneInfo(name, self)  #: SceneInfo: The scene information.

    def clear(self) -> "Scene":
        """Clear all objects from the scene.

        Returns:
            Scene: The cleared scene.

        Example:
            Start from a blank slate before re-adding objects::

                scene.clear()
                scene.add("sheet").at(0, 0.6, 0)
        """
        self._object.clear()
        return self

    def set_surface_map(
        self,
        name: str,
        tri_indices: np.ndarray,
        coefs: np.ndarray,
        surf_tri: np.ndarray,
    ):
        """Store frame-embedding surface mapping for a tetrahedralized object.

        Args:
            name: Object UUID key.
            tri_indices: Closest triangle index in tet surface per original vertex (N,).
            coefs: Frame coefficients (c1, c2, c3) per original vertex (N, 3).
            surf_tri: Surface triangles of the tet mesh (Q, 3).

        Example:
            Typically invoked internally by the Blender add-on decoder when a
            TetMesh is registered, but can be called directly to attach a
            surface map::

                scene.set_surface_map("obj-uuid", tri_idx, coefs, surf_tri)
        """
        _rust.scene_validate_surface_map_key(name)
        self._surface_map_by_name[name] = (tri_indices, coefs, surf_tri)

    def select(self, name: str) -> "Object":
        """Select an object from the scene by its name.

        Args:
            name (str): The reference name of the object to select.

        Returns:
            Object: The selected object.

        Example:
            Adjust an already-added object by ref name::

                scene.add("sheet")
                sheet = scene.select("sheet")
                sheet.at(0, 0.6, 0)
        """
        _rust.scene_validate_scene_select(name in self._object, name)
        return self._object[name]

    def _axis_bound(self, axis: str, is_min: bool) -> float:
        """Compute the min or max vertex coordinate along an axis.

        ``is_min`` selects the reduction (True for the lower bound, False
        for the upper bound). Both :meth:`min` and :meth:`max` delegate
        here so the per-object gather/cumsum/concat path is defined once.
        """
        ax = _rust.scene_axis_letter_to_index(axis)
        # Stack every object's vertices into a single (sum_n, 3) buffer
        # plus per-object cumulative offsets, then let Rust compute the
        # bound in one pass. Cumulative offsets come from `np.cumsum` on
        # the per-object vertex counts (single C-level allocation).
        per_obj_vert = [
            v for v in (obj.vertex(True) for obj in self._object.values())
            if v is not None
        ]
        if not per_obj_vert:
            return _rust.scene_reduce_axis_bound([], "min" if is_min else "max")
        counts = np.asarray([len(v) for v in per_obj_vert], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
        flat = np.concatenate([np.asarray(v, dtype=np.float64) for v in per_obj_vert])
        return _rust.scene_per_object_axis_bound(
            np.ascontiguousarray(flat, dtype=np.float64),
            offsets,
            ax,
            is_min,
        )

    def min(self, axis: str) -> float:
        """Get the minimum value of the scene along a specific axis.

        Args:
            axis (str): The axis to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The minimum vertex coordinate along the specified axis.

        Example:
            Place a ground wall just below the lowest vertex::

                y_min = scene.min("y")
                scene.add.invisible.wall([0, y_min - 0.01, 0], [0, 1, 0])
        """
        return self._axis_bound(axis, True)

    def max(self, axis: str) -> float:
        """Get the maximum value of the scene along a specific axis.

        Args:
            axis (str): The axis to get the maximum value along, either "x", "y", or "z".

        Returns:
            float: The maximum vertex coordinate along the specified axis.

        Example:
            Place a ceiling wall just above the highest vertex::

                y_max = scene.max("y")
                scene.add.invisible.wall([0, y_max + 0.01, 0], [0, -1, 0])
        """
        return self._axis_bound(axis, False)

    @property
    def sphere_list(self) -> list[Sphere]:
        """Get the list of spheres.

        Example:
            Enumerate invisible sphere colliders currently on the scene::

                for sphere in scene.sphere_list:
                    print(sphere.entry)
        """
        return self._sphere

    @property
    def wall_list(self) -> list[Wall]:
        """Get the list of walls.

        Example:
            Enumerate invisible wall colliders currently on the scene::

                for wall in scene.wall_list:
                    print(wall.normal, wall.entry)
        """
        return self._wall

    @property
    def object_dict(self) -> dict[str, "Object"]:
        """Get the object dictionary.

        Example:
            Iterate every added object by its reference name::

                for name, obj in scene.object_dict.items():
                    print(name, obj.obj_type)
        """
        return self._object

    @property
    def asset_manager(self) -> AssetManager:
        """Get the asset manager.

        Example:
            Reach into the asset manager attached to this scene::

                mgr = scene.asset_manager
                print(mgr.list())
        """
        return self._asset

    def build(self, progress_callback=None, quiet: bool = False) -> FixedScene:
        """Build the fixed scene from the current scene.

        Args:
            progress_callback: Optional callable ``f(fraction, step_info)`` invoked as the build progresses.
            quiet: When True, suppress the merge-pair and scene-check diagnostic prints. Batch/headless drivers that already pass a ``progress_callback`` can set this to silence the bare stdout logs. Defaults to False, preserving the interactive output.

        Returns:
            FixedScene: The built fixed scene.

        Example:
            Compile the scene and chain a summary print::

                fixed = scene.build().report()
                session = app.session.create(fixed).build()
        """
        total_steps = 13
        completed_steps = 0

        def report(step_info: str):
            if progress_callback is not None:
                progress_callback(completed_steps / total_steps, step_info)

        def advance(step_info: str):
            nonlocal completed_steps
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(completed_steps / total_steps, step_info)

        pbar = tqdm(total=total_steps, desc="build scene")
        report("Building scene: preparing objects...")
        for _, obj in self._object.items():
            obj.update_static()
        pbar.update(1)
        advance("Building scene: preparing objects...")

        dyn_objects = [
            (name, obj) for name, obj in self._object.items() if not obj.static
        ]
        n = len(dyn_objects)
        for i, (_, obj) in enumerate(dyn_objects):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.75, 1.0)
            if obj.object_color is None:
                obj.default_color(r, g, b)

        # Rod/shell/finalize index-map construction via the Rust kernel.
        # The per-object dict packaging is a single list comprehension
        # (no .append in a loop).
        def _pack_rust_object(name, obj):
            vert = obj.get("V")
            if vert is None:
                return None
            edge = obj.get("E")
            tri = obj.get("F")
            tet = obj.get("T")
            return {
                "name": name,
                "n_verts": int(len(vert)),
                "edges": (
                    np.ascontiguousarray(edge, dtype=np.int64)
                    if edge is not None else None
                ),
                "faces": (
                    np.ascontiguousarray(tri, dtype=np.int64)
                    if tri is not None else None
                ),
                "tets": (
                    np.ascontiguousarray(tet, dtype=np.int64)
                    if tet is not None else None
                ),
            }

        rust_objects = [
            d for d in (_pack_rust_object(name, obj) for name, obj in dyn_objects)
            if d is not None
        ]

        result = _rust.scene_build_index_map(rust_objects)
        map_by_name = dict(result["map_by_name"])
        concat_count = int(result["concat_count"])
        rod_vert_start, rod_vert_end = result["rod_vert_range"]
        shell_vert_start, shell_vert_end = result["shell_vert_range"]
        pbar.update(3)
        advance("Building scene: indexing rod topology...")
        advance("Building scene: indexing shell topology...")
        advance("Building scene: finalizing vertex map...")

        # ----- Build kernel input: dyn_objects -----
        # For each dynamic object, prepare numpy arrays the kernel reads.
        # The per-object pin-index concat runs through Rust; the dyn_input
        # dict is built via a list comprehension (single allocation, no
        # incremental append). Velocity schedules + collision-window
        # lookups are dict assignments, not list growth.
        velocity_schedules: dict[str, Any] = {}
        angular_velocity_schedules: dict[str, Any] = {}
        angular_world_velocity_schedules: dict[str, Any] = {}
        collision_windows: dict[str, Any] = {}
        pinned_indices_by_name: dict[str, list[int]] = {}
        # Single source of truth: the cap is defined once as a Rust
        # constant (ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS)
        # and exported through the PyO3 module so this guard, the solver's
        # collision-window table builder, and the GPU-side #define cannot
        # drift apart.
        MAX_COLLISION_WINDOWS = _rust.MAX_COLLISION_WINDOWS
        for name, obj in dyn_objects:
            pinned_indices_by_name[name] = _rust.scene_concat_i64_lists(
                [list(p.index) for p in obj.pin_list]
            ).tolist()
            if obj._velocity_schedule:
                velocity_schedules[name] = [
                    (t, list(v)) for t, v in obj._velocity_schedule
                ]
            # Principal-axis angular overwrite: carry (pca_index, speed) into
            # the dyn_param Vec3 slot as [pca_index, speed, 0.0]; the solver
            # resolves the world axis from the live geometry each firing, so
            # NO eigendecomposition happens here.
            if obj._angular_velocity_schedule_pca:
                angular_velocity_schedules[name] = [
                    (t, [float(pca_index), float(speed), 0.0])
                    for t, pca_index, speed in obj._angular_velocity_schedule_pca
                ]
            # Fixed world-axis angular: carry the full ω vector (solver space)
            # into the dyn_param Vec3 slot; the solver applies it directly.
            if obj._angular_velocity_schedule_world:
                angular_world_velocity_schedules[name] = [
                    (t, [float(v[0]), float(v[1]), float(v[2])])
                    for t, v in obj._angular_velocity_schedule_world
                ]
            if obj._collision_windows:
                if len(obj._collision_windows) > MAX_COLLISION_WINDOWS:
                    raise ValueError(
                        f"Object '{name}' has {len(obj._collision_windows)} collision windows "
                        f"(max {MAX_COLLISION_WINDOWS})"
                    )
                collision_windows[name] = obj._collision_windows

        def _pack_dyn_input(name, obj):
            edge = obj.get("E")
            tri = obj.get("F")
            tet = obj.get("T")
            uv_list = obj.uv_coords
            uv_arr = None
            if uv_list is not None and len(uv_list):
                uv_arr = np.ascontiguousarray(
                    np.stack([np.asarray(u, dtype=np.float64).reshape(-1) for u in uv_list]),
                    dtype=np.float64,
                )
            stitch_ind = obj.get("Ind")
            stitch_w = obj.get("W")
            color_arr = np.asarray(obj.get("color"), dtype=np.float64)
            n_verts_obj = int(obj.vertex(False).shape[0])
            if color_arr.ndim == 1 and color_arr.shape == (3,):
                color_arr = np.broadcast_to(color_arr, (n_verts_obj, 3)).copy()
            return {
                "name": name,
                "obj_type": obj.obj_type,
                "vertex": np.ascontiguousarray(obj.vertex(False), dtype=np.float64),
                "color": np.ascontiguousarray(color_arr, dtype=np.float64),
                "velocity": list(obj.object_velocity),
                "position": list(obj.position),
                "edges": (
                    np.ascontiguousarray(edge, dtype=np.int64)
                    if edge is not None else None
                ),
                "faces": (
                    np.ascontiguousarray(tri, dtype=np.int64)
                    if tri is not None else None
                ),
                "tets": (
                    np.ascontiguousarray(tet, dtype=np.int64)
                    if tet is not None else None
                ),
                "uv": uv_arr,
                "dynamic_color": int(obj.dynamic_color.value),
                "dynamic_intensity": float(obj.dynamic_intensity),
                "pinned_indices": pinned_indices_by_name[name],
                "stitch_ind": (
                    np.ascontiguousarray(stitch_ind, dtype=np.int64)
                    if stitch_ind is not None else None
                ),
                "stitch_w": (
                    np.ascontiguousarray(stitch_w, dtype=np.float64)
                    if stitch_w is not None else None
                ),
                "stitch_stiffness": float(obj.param.get("stitch-stiffness")),
            }

        dyn_inputs = [_pack_dyn_input(name, obj) for name, obj in dyn_objects]

        # ----- Build kernel input: static_objects -----
        # List comprehension over the static-only filter; no incremental
        # .append on the outer list.
        def _pack_static_input(name, obj):
            vert = obj.get("V")
            tri = obj.get("F")
            if vert is None or tri is None:
                return None
            color = obj.get("color")
            return {
                "name": name,
                "vertex_world": np.ascontiguousarray(
                    obj.apply_transform(vert, False), dtype=np.float64
                ),
                "color": [float(color[0]), float(color[1]), float(color[2])],
                "faces": np.ascontiguousarray(tri, dtype=np.int64),
                "position": list(obj.position),
            }

        static_inputs = [
            d for d in (
                _pack_static_input(name, obj)
                for name, obj in self._object.items()
                if obj.static
            )
            if d is not None
        ]

        # ----- dmap order: every object (dyn + static) in insertion order -----
        dmap_order = [
            (name, [float(x) for x in obj.position])
            for name, obj in self._object.items()
        ]
        dmap = {name: i for i, (name, _) in enumerate(dmap_order)}

        # ----- Cross-stitch validation + repackaging -----
        # Rust validates every (source, target) pair against the known
        # name set in one pass; the per-entry numpy reshape stays in
        # Python because the inputs are heterogeneous arrays.
        _rust.scene_validate_cross_stitch_names(
            [(cs["source_name"], cs["target_name"]) for cs in self._cross_stitch],
            list(map_by_name.keys()),
        )
        cross_stitches = [
            {
                "source_name": cs["source_name"],
                "target_name": cs["target_name"],
                "ind": np.ascontiguousarray(np.asarray(cs["ind"]), dtype=np.int64).reshape(-1, 6),
                "w": np.ascontiguousarray(np.asarray(cs["w"]), dtype=np.float64).reshape(-1, 6),
                "stitch_stiffness": float(cs.get("stitch_stiffness", 1.0)),
            }
            for cs in self._cross_stitch
        ]

        # ----- Run the Rust assembly kernel -----
        pbar.update(1)
        advance("Building scene: gathering dynamic vertices...")
        assembled = _rust.scene_build_fixed(
            dyn_inputs,
            static_inputs,
            dmap_order,
            map_by_name,
            int(concat_count),
            cross_stitches,
        )

        concat_displacement = assembled["concat_displacement"]
        concat_vert = assembled["concat_vert"]
        concat_color = assembled["concat_color"]
        concat_vel = assembled["concat_vel"]
        concat_vert_dmap = assembled["concat_vert_dmap"]
        concat_rod = assembled["concat_rod"]
        concat_tri = assembled["concat_tri"]
        concat_tet = assembled["concat_tet"]
        concat_uv_arr = assembled["concat_uv"]
        concat_dyn_tri_color_arr = assembled["concat_dyn_tri_color"]
        concat_dyn_tri_intensity_arr = assembled["concat_dyn_tri_intensity"]
        concat_rod_is_collider_arr = assembled["concat_rod_is_collider"]
        concat_tri_is_collider_arr = assembled["concat_tri_is_collider"]
        concat_stitch_ind_arr = assembled["concat_stitch_ind"]
        concat_stitch_w_arr = assembled["concat_stitch_w"]
        concat_stitch_stiffness_arr = assembled["concat_stitch_stiffness"]
        rod_count = int(assembled["rod_count"])
        shell_count = int(assembled["shell_count"])
        stats_by_name = assembled["stats_by_name"]

        # ----- Reconstruct UV / dyn-color list-shaped layout for FixedScene -----
        # FixedScene._uv is consumed only as a length and as the source bytes
        # for uv.bin export, both of which work on a single (n_tri, 3, 2)
        # array. Keep concat_uv_arr as one array instead of exploding it into
        # ~1.8M tiny (3, 2) arrays: that avoids a multi-million-iteration build
        # loop and, more importantly, lets _uv pickle as a single array in the
        # build / app-state snapshots rather than a multi-million-element list
        # (which dominated the serialize phases of the build).
        concat_uv: np.ndarray = np.ascontiguousarray(
            concat_uv_arr, dtype=np.float32
        ).reshape(-1, 3, 2)
        concat_dyn_tri_color = [
            EnumColor(int(v)) for v in concat_dyn_tri_color_arr
        ]
        concat_dyn_tri_intensity = [
            float(v) for v in concat_dyn_tri_intensity_arr
        ]

        pbar.update(5)
        advance("Building scene: assembling rods...")
        advance("Building scene: assembling shell triangles...")
        advance("Building scene: assembling solid surfaces...")
        advance("Building scene: assembling tetrahedra...")
        # Name this step for what it actually does at this point: the
        # assembler has produced stitch data here, while pins are applied
        # later (decoder ParamDecoder.apply_pin_config), so the old
        # "collecting pins and stitches" read as busywork on scenes with
        # neither. Only mention stitches when present; otherwise a neutral
        # finalize label.
        advance(
            "Building scene: collecting stitches..."
            if len(concat_stitch_ind_arr)
            else "Building scene: finalizing scene topology..."
        )

        # Bake initial angular velocities into per-vertex velocities.
        # The Rust assembler replicates the object-level linear
        # velocity uniformly across vertices; angular velocity adds
        # the rotational component ω × (x̄ − c̄), where the centroid
        # is the object's own vertex centroid (matching the PDRD
        # per-body rest centroid).
        for name, obj in dyn_objects:
            omega = getattr(obj, "_angular_velocity", None)
            if omega is None or all(o == 0.0 for o in omega):
                continue
            indices = list(map_by_name[name])
            if not indices:
                continue
            idx_arr = np.asarray(indices, dtype=np.int64)
            positions = np.asarray(concat_vert[idx_arr], dtype=np.float64)
            c_local = positions.mean(axis=0)
            r = positions - c_local
            w = np.asarray(omega, dtype=np.float64)
            # v = ω × r, broadcast across vertices.
            spin_vel = np.cross(np.broadcast_to(w, r.shape), r)
            concat_vel[idx_arr] += spin_vel.astype(concat_vel.dtype)

        # ----- Param replication using counts the Rust kernel returned -----
        # Mirrors `extend_param` from the original loop. Each object's param
        # values are appended N times where N is its surviving element count
        # for the relevant element kind.
        concat_rod_param: dict[str, list] = {}
        concat_tri_param: dict[str, list] = {}
        concat_tet_param: dict[str, list] = {}
        concat_static_param: dict[str, list] = {}

        # Keys carried in obj.param for key-set parity across kinds but
        # NOT meaningful at per-face granularity. They live on a
        # per-body table assembled separately; replicating them
        # per-face would waste space and risks pretending they vary
        # within a body. `stitch-stiffness` is resolved per object into a
        # per-stitch-row array (concat_stitch_stiffness) by the assembly
        # kernel, so it never enters the per-element param arrays.
        per_body_only_keys = {"stitch-stiffness"}

        def _extend_param(
            param: ParamHolder,
            concat_param: dict[str, list],
            count: int,
        ):
            param_keys = [k for k in param.key_list() if k not in per_body_only_keys]
            if len(concat_param.keys()):
                assert param_keys == list(concat_param.keys()), (
                    f"param keys mismatch: {param_keys} vs {list(concat_param.keys())}"
                )
            for key, value in param.items():
                if key in per_body_only_keys:
                    continue
                if key not in concat_param:
                    concat_param[key] = []
                concat_param[key].extend([value] * count)

        # Param replication must mirror the Rust kernel's element
        # layout in `concat_tri`: rods first, then pure shells, then
        # SOLID surface tris (matching `assemble_dyn_scene`'s order).
        # `concat_tri_param` is read by the solver as a prefix-aligned
        # slot for each tri, so any deviation from this order
        # mis-pairs material parameters with their tris.
        for name, obj in dyn_objects:
            stats = stats_by_name[name]
            rod_added = int(stats["rod_added"])
            tet_added = int(stats["tet_added"])
            if obj.obj_type == "rod" and rod_added:
                _extend_param(obj.param, concat_rod_param, rod_added)
            if tet_added:
                _extend_param(obj.param, concat_tet_param, tet_added)
        # Pass A: pure shells.
        for name, obj in dyn_objects:
            stats = stats_by_name[name]
            tri_added = int(stats["tri_added"])
            tet_added = int(stats["tet_added"])
            if tri_added and tet_added == 0:
                _extend_param(obj.param, concat_tri_param, tri_added)
        # Pass B: SOLID surface triangles (objects carrying both tri and tet).
        for name, obj in dyn_objects:
            stats = stats_by_name[name]
            tri_added = int(stats["tri_added"])
            tet_added = int(stats["tet_added"])
            if tri_added and tet_added:
                _extend_param(obj.param, concat_tri_param, tri_added)

        # ----- SAND (faceless point cloud) scalar params -----
        # A "points" object has no elements, so the per-element replication
        # above produces nothing for it. The solver reads grain mass / contact
        # radius / gap / friction as scalars (one float each) on the "sand-"
        # channel instead. The grain radius is authored on the standard
        # contact-offset key, and the contact gap on contact-gap; mass and
        # friction come from the sand-* keys. Every grain shares one set, so a
        # single value per file. First SAND object wins (a SAND-only scene is
        # the supported case; multiple clouds would share these scalars).
        concat_sand_param: dict[str, list] = {}
        for name, obj in dyn_objects:
            if obj.obj_type != "points":
                continue
            p = obj.param
            concat_sand_param = {
                "particle-mass": [float(p.get("sand-particle-mass"))],
                "grain-radius": [float(p.get("contact-offset"))],
                "contact-gap": [float(p.get("contact-gap"))],
                "friction": [float(p.get("sand-friction"))],
            }
            break

        # ----- Pin assembly (Python-side: PinData is a Python dataclass) -----
        # Flatten every (object, pin) pair into a single list comprehension
        # so there's no incremental .append on the outer concat list nor
        # on the inner mapped_ops list. Each comprehension is a single
        # CPython allocation.
        def _remap_op(op, map_arr):
            if isinstance(op, TorqueOperation) and op.hint_vertex >= 0:
                return _dc_replace(op, hint_vertex=int(map_arr[op.hint_vertex]))
            return op

        concat_pin: list[PinData] = [
            PinData(
                index=np.asarray(map_by_name[name], dtype=np.int64)[
                    np.asarray(p.index, dtype=np.int64)
                ].tolist(),
                operations=[_remap_op(op, np.asarray(map_by_name[name])) for op in p.operations],
                unpin_time=p.unpin_time,
                pull_strength=p.pull_strength,
                # Per-vertex pull weights are aligned to p.index; the index
                # remap above is element-wise (same order), so the array
                # stays aligned. Carry it so pin-pullw is exported.
                pull_weights=getattr(p, "pull_weights", None),
                pin_stiffness=p.pin_stiffness,
                transition=p.transition,
                pin_group_id=p.pin_group_id,
                hide_in_preview=bool(getattr(obj, "_is_static_moving", False)),
                rest_shape_track=bool(getattr(p, "rest_shape_track", False)),
            )
            for name, obj in dyn_objects
            for p in obj.pin_list
        ]

        # ----- PDRD per-body table -----
        # For each PDRD-flagged object, collect its vertex indices and
        # precompute the rest-shape moments used for best-fit rigid
        # reconstruction plus mass/inertia data (centroid, Gram inverse,
        # enclosed volume, per-vertex volumetric mass). Bodies' vertex
        # sets need not be
        # contiguous in the global vertex array; the kernel reads
        # them through a flat index list (`pdrd_vert_list`).
        pdrd_body_rows: list[float] = []  # PDRD_BODY_ROW_LEN floats per body
        # Floats per PDRD body row; must match `PDRD_BODY_ROW_LEN` in
        # crates/ppf-cts-solver/src/data.rs. Layout: vertex_start,
        # vertex_count, volume, centroid[3], rest_gram_inv[9 row-major],
        # mass_per_vertex, joint_mode, joint_axis[3], joint_pin[3]
        # (16 base + 7 joint = 23 floats).
        PDRD_BODY_ROW_LEN = 23
        pdrd_body_densities: list[float] = []  # one float per body, used to fill mass_per_vertex after volume is known
        pdrd_body_centroids: list[np.ndarray] = []  # rest centroid per body (local frame), used in the inertia pass
        pdrd_body_gram_trace: list[float] = []  # trace(S̄) per body, used to compute I_uniform
        pdrd_vert_index = np.zeros(concat_count, dtype=np.uint32)
        pdrd_vert_list: list[int] = []
        pdrd_rest_centered_chunks: list[np.ndarray] = []
        next_body_id = 0
        for name, obj in dyn_objects:
            if not getattr(obj, "_is_pdrd", False):
                continue
            indices = list(map_by_name[name])
            if not indices:
                continue
            idx_arr = np.asarray(indices, dtype=np.int64)
            vertex_start = len(pdrd_vert_list)
            vertex_count = int(len(idx_arr))
            positions = np.asarray(concat_vert[idx_arr], dtype=np.float64)
            c_bar = positions.mean(axis=0)
            y_bar = positions - c_bar
            gram = y_bar.T @ y_bar
            try:
                gram_inv = np.linalg.inv(gram)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f"PDRD body {name!r}: rest-shape Gram matrix is singular "
                    "(degenerate body geometry, at least 4 non-coplanar vertices required)"
                ) from e
            density = float(obj.param.get("density"))
            assert density > 0.0, (
                f"PDRD body {name!r}: density must be > 0 (got {density})"
            )

            body_id = next_body_id + 1  # 1-based; 0 means "not PDRD"
            next_body_id += 1
            pdrd_vert_index[idx_arr] = body_id
            pdrd_vert_list.extend(int(i) for i in idx_arr)
            pdrd_rest_centered_chunks.append(y_bar.astype(np.float32))
            pdrd_body_densities.append(density)
            pdrd_body_centroids.append(c_bar.copy())
            pdrd_body_gram_trace.append(float(np.trace(gram)))
            # Row: vertex_start (into pdrd_vert_list, NOT global!),
            # vertex_count, volume (filled by the divergence-theorem
            # pass below), centroid (3), gram_inv (9 row-major),
            # mass_per_vertex (filled below), then the joint block:
            # joint_mode, joint_axis (3, world axle), joint_pin (3, world
            # pivot). joint_mode = 0 leaves the body free (no DOF
            # filtering); a hinge sets the axle to a principal axis of the
            # rest Gram and pins the centroid.
            joint_mode = 0.0
            joint_axis = [0.0, 0.0, 0.0]
            joint_pin = [0.0, 0.0, 0.0]
            joint = getattr(obj, "_joint", None)
            if joint is not None:
                kind, pca_index = joint
                if kind == "hinge":
                    # Principal axes = eigenvectors of the rest Gram
                    # (computed in placed/world coords, so the axle is a
                    # fixed world direction at t=0). Order by DESCENDING
                    # eigenvalue so index 0 = largest extent, 2 = thinnest;
                    # this convention must match the Blender overlay.
                    evals, evecs = np.linalg.eigh(gram)
                    order = np.argsort(evals)[::-1]
                    axis_vec = np.asarray(
                        evecs[:, order[pca_index]], dtype=np.float64
                    )
                    nrm = float(np.linalg.norm(axis_vec))
                    if nrm <= 1e-12:
                        raise ValueError(
                            f"PDRD body {name!r}: degenerate principal axis "
                            "for hinge (rest shape too symmetric to pick an axle)"
                        )
                    axis_vec = axis_vec / nrm
                    joint_mode = 1.0
                    joint_axis = [float(a) for a in axis_vec]
                    joint_pin = [float(c_bar[0]), float(c_bar[1]), float(c_bar[2])]
            row = [
                float(vertex_start),
                float(vertex_count),
                0.0,  # volume, filled below.
                float(c_bar[0]), float(c_bar[1]), float(c_bar[2]),
            ]
            row.extend(float(v) for v in gram_inv.flatten(order="C"))
            row.append(0.0)  # mass_per_vertex, filled below.
            row.append(joint_mode)
            row.extend(joint_axis)
            row.extend(joint_pin)
            assert len(row) == PDRD_BODY_ROW_LEN
            pdrd_body_rows.extend(row)
        pdrd_rest_centered = (
            np.vstack(pdrd_rest_centered_chunks)
            if pdrd_rest_centered_chunks
            else np.zeros((0, 3), dtype=np.float32)
        )

        # Per-body enclosed volume via the divergence theorem AND
        # volumetric inertia trace via centroid-tet decomposition.
        # For each surface triangle of a body, form the tetrahedron
        # (c̄, v0, v1, v2). The signed volumes sum to the body's
        # enclosed volume; ∫_tet ||r − c̄||² dV summed gives
        # ∫_body ||r − c̄||² dV, and trace(I_solid) = 2ρ · that.
        #
        # Per-vertex mass is then scaled so the body's effective
        # rotational inertia matches the volumetric one:
        #
        #   m_per_vertex = (ρV / N) · trace(I_solid) / trace(I_uniform)
        #
        # where I_uniform = (ρV/N) · (tr(S̄) I − S̄) is the surface-
        # shell inertia with uniform per-vertex mass. For symmetric
        # bodies (cube, sphere) both tensors are isotropic and the
        # trace ratio is exact per-axis. For asymmetric bodies it's
        # an approximation that preserves the average rotational
        # behavior. Gravity-driven translation is unaffected because
        # IPC applies gravity as per-vertex acceleration; the trade-
        # off is that effective mass for collision momentum is
        # reduced by the scale factor (for a cube ≈ 1/3).
        if next_body_id > 0 and len(concat_tri) > 0:
            tri_arr = np.asarray(concat_tri, dtype=np.int64)
            b0 = pdrd_vert_index[tri_arr[:, 0]]
            b1 = pdrd_vert_index[tri_arr[:, 1]]
            b2 = pdrd_vert_index[tri_arr[:, 2]]
            same_nonzero = (b0 != 0) & (b0 == b1) & (b1 == b2)
            for bid in range(1, next_body_id + 1):
                body_tri_mask = same_nonzero & (b0 == bid)
                body_tris = tri_arr[body_tri_mask]
                if len(body_tris) == 0:
                    raise ValueError(f"PDRD body {bid}: no triangles found")
                c_bar_b = pdrd_body_centroids[bid - 1]
                va = concat_vert[body_tris[:, 0]].astype(np.float64) - c_bar_b
                vb = concat_vert[body_tris[:, 1]].astype(np.float64) - c_bar_b
                vc = concat_vert[body_tris[:, 2]].astype(np.float64) - c_bar_b
                # Signed tet volumes (each tet = centroid + triangle).
                vt = np.einsum("ij,ij->i", va, np.cross(vb, vc)) / 6.0
                vol = float(np.abs(np.sum(vt)))
                if vol < 1e-10:
                    raise ValueError(
                        f"PDRD body {bid}: enclosed volume {vol:.3e} is below the "
                        "watertight-mesh floor. Mesh may have inverted or unclosed "
                        "boundary; PDRD requires a closed surface mesh."
                    )
                # Second moment ∫_tet ||r||² dV (origin = body
                # centroid; closed-form for tet with one vertex at
                # origin):
                #   V_t/20 · (||v_a+v_b+v_c||² + ||v_a||² + ||v_b||²
                #             + ||v_c||²)
                sum_v = va + vb + vc
                sum_sq = (
                    np.einsum("ij,ij->i", sum_v, sum_v)
                    + np.einsum("ij,ij->i", va, va)
                    + np.einsum("ij,ij->i", vb, vb)
                    + np.einsum("ij,ij->i", vc, vc)
                )
                second_moment = float(np.sum(vt / 20.0 * sum_sq))
                density = pdrd_body_densities[bid - 1]
                trace_I_solid = 2.0 * density * second_moment
                row_off = (bid - 1) * PDRD_BODY_ROW_LEN
                vertex_count = int(pdrd_body_rows[row_off + 1])
                total_mass = density * vol
                trace_I_uniform = (
                    2.0 * total_mass / float(vertex_count)
                    * pdrd_body_gram_trace[bid - 1]
                )
                if trace_I_uniform <= 0.0:
                    raise ValueError(
                        f"PDRD body {bid}: degenerate rest configuration "
                        "(trace of Σ ȳ ȳᵀ is non-positive)"
                    )
                scale = trace_I_solid / trace_I_uniform
                if not (1e-6 < scale < 100.0):
                    raise ValueError(
                        f"PDRD body {bid}: inertia trace ratio {scale:.3e} out "
                        "of bounds. Mesh may be non-closed or extremely thin."
                    )
                mass_per_vertex = (total_mass / float(vertex_count)) * scale
                pdrd_body_rows[row_off + 2] = vol
                pdrd_body_rows[row_off + 15] = mass_per_vertex

        # ----- Global pinned-vertex set + rest_vert computation -----
        global_pinned_vertices: set[int] = set()
        for pin in concat_pin:
            global_pinned_vertices.update(pin.index)

        concat_rest_vert = concat_vert.copy()
        rest_vert_mask = np.zeros(concat_count, dtype=np.uint8)
        for name, obj in dyn_objects:
            indices = map_by_name[name]
            obj_indices_set = set(np.asarray(indices).tolist())
            obj_pins = [
                p for p in concat_pin
                if set(p.index) & obj_indices_set
            ]
            if not obj_pins:
                continue
            covered: set[int] = set()
            all_have_duration = True
            for p in obj_pins:
                if p.unpin_time is None:
                    all_have_duration = False
                    break
                covered.update(p.index)
            if not all_have_duration or not obj_indices_set.issubset(covered):
                continue
            for p in obj_pins:
                positions = concat_rest_vert[p.index].copy()
                for op in p.operations:
                    positions = op.apply(positions, p.unpin_time)
                concat_rest_vert[p.index] = positions
                rest_vert_mask[p.index] = 1
        has_rest_vert = bool(rest_vert_mask.any())

        # ----- Bending reference rest angle -----
        # Each shell object that opted in carries a world-space reference
        # vertex buffer (set by the decoder), aligned 1:1 with its own
        # vertices. Scatter it into a full per-vertex array (initialized from
        # the initial vert, so non-reference objects are unchanged) via the
        # local->global index map, and mark the participating vertices so the
        # solver knows which hinges take their rest angle from the reference
        # instead of the initial pose.
        concat_bend_rest_vert = concat_vert.copy()
        bend_rest_vert_mask = np.zeros(concat_count, dtype=np.uint8)
        for name, obj in dyn_objects:
            brv = getattr(obj, "_bend_rest_vert", None)
            if brv is None:
                continue
            indices = np.asarray(map_by_name[name], dtype=np.int64)
            if len(brv) != len(indices):
                raise ValueError(
                    f"Bending reference for '{name}' has {len(brv)} vertices "
                    f"but the object has {len(indices)}."
                )
            concat_bend_rest_vert[indices] = np.asarray(
                brv, dtype=concat_bend_rest_vert.dtype
            )
            bend_rest_vert_mask[indices] = 1
        has_bend_rest_vert = bool(bend_rest_vert_mask.any())

        # ----- Time-varying rest shape (captured full-pin deformation) -----
        # A pin flagged rest_shape_track is a FULL pin (every vertex captured;
        # the encoder gates the toggle on that). Its move ops carry the captured
        # target trajectory for the whole object, so sampling them at the
        # capture frame boundaries gives a time-varying rest shape the solver
        # follows (Scene::rest_shape_schedule): the dynamic body's stress-free
        # shape becomes the captured deformation while the pins guide it and
        # contact resolves. The base is the post-static rest pose, so a
        # non-captured object's static (unpin_time) rest stays put every frame.
        rest0 = concat_rest_vert.copy()
        track_pins = [p for p in concat_pin if getattr(p, "rest_shape_track", False)]
        rest_vert_anim = None
        rest_vert_times = None
        if track_pins:
            boundary_times: set[float] = set()
            for p in track_pins:
                for op in p.operations:
                    boundary_times.add(float(op.t_start))
                    boundary_times.add(float(op.t_end))
            times_sorted = sorted(boundary_times)
            if times_sorted:
                anim_mask = rest_vert_mask.copy()
                # The encoder flags rest_shape_track only on a FULL pin (every
                # vertex captured). For such a pin ``p.index`` is the WHOLE
                # object (surface + harmonic interior) and its operations drive
                # every one to the captured position, so the rest pose IS the
                # captured deformation: apply the operations to all pin vertices.
                # No reconstruction, no free region, no boundary tear.
                frames = []
                for t_k in times_sorted:
                    frame = rest0.copy()
                    for p in track_pins:
                        idx = np.asarray(p.index, dtype=np.int64)
                        if idx.size == 0:
                            continue
                        pos = rest0[idx].copy()
                        for op in p.operations:
                            pos = op.apply(pos, t_k)
                        frame[idx] = pos
                        anim_mask[idx] = 1
                    frames.append(frame)
                # Frame-major: (n_frames * n_vert, 3), reshaped solver-side.
                rest_vert_anim = np.concatenate(frames, axis=0).astype(np.float64)
                rest_vert_times = np.asarray(times_sorted, dtype=np.float64)
                # The solver shares one mask between the static rest_vert and
                # the schedule, so widen it to the union of driven verts.
                rest_vert_mask = anim_mask
        has_rest_vert_anim = rest_vert_anim is not None

        # ----- Static side: param replication + transform animations -----
        pbar.update(1)
        advance("Building scene: assembling static geometry...")
        static_vert = assembled["static_vert"]
        static_tri = assembled["static_tri"]
        static_color = assembled["static_color"]
        static_vert_dmap = assembled["static_vert_dmap"]
        static_per_object = assembled["static_per_object"]
        # Replicate static-side params + collect transform anims. The
        # param replication still runs as a side effect so we keep the
        # for-loop, but the (offset, anim) collect drops to a single
        # comprehension over the resolved entries.
        for name, obj in self._object.items():
            if not obj.static:
                continue
            entry = static_per_object.get(name)
            if entry is None:
                continue
            _extend_param(obj.param, concat_static_param, int(entry["n_face"]))
        concat_static_transform_anims: list[tuple[int, TransformAnimation]] = [
            (int(static_per_object[name]["offset"]), obj._transform_animation)
            for name, obj in self._object.items()
            if obj.static
            and static_per_object.get(name) is not None
            and obj._transform_animation is not None
        ]

        # Re-key velocity schedules / collision windows by displacement idx
        # now that the dmap ordering is fixed.
        velocity_schedules_by_dmap = {
            f"velocity:{dmap[name]}": entries
            for name, entries in velocity_schedules.items()
        }
        angular_velocity_schedules_by_dmap = {
            f"angular_velocity:{dmap[name]}": entries
            for name, entries in angular_velocity_schedules.items()
        }
        angular_world_velocity_schedules_by_dmap = {
            f"angular_velocity_world:{dmap[name]}": entries
            for name, entries in angular_world_velocity_schedules.items()
        }
        collision_windows_by_dmap = {
            f"collision_window:{dmap[name]}": entries
            for name, entries in collision_windows.items()
        }

        pbar.update(1)
        advance("Building scene: finalizing fixed scene...")

        # Param key clearing to enforce per-element-kind scoping. Mirrors
        # the closing block of the original Python builder.
        for key in ["model"]:
            concat_rod_param[key] = []
            concat_static_param[key] = []
        for key in ["poiss-rat"]:
            concat_rod_param[key] = []
        for key in ["strain-limit"]:
            concat_tet_param[key] = []
            concat_static_param[key] = []
        for key in ["shrink-x", "shrink-y"]:
            concat_rod_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []
        for key in ["shrink"]:
            concat_rod_param[key] = []
            concat_tri_param[key] = []
            concat_static_param[key] = []
        for key in ["friction", "contact-gap", "contact-offset", "bend",
                    "bending-damping"]:
            concat_tet_param[key] = []
        for key in ["bend-plasticity", "bend-plasticity-threshold",
                    "bend-rest-from-geometry"]:
            concat_tet_param[key] = []
            concat_static_param[key] = []
        for key in ["young-mod", "poiss-rat", "bend", "density"]:
            concat_static_param[key] = []
        for key in ["length-factor"]:
            concat_tri_param[key] = []
            concat_tet_param[key] = []
            concat_static_param[key] = []

        # Shrink/extend invalidates strain limiting on shells.
        sx = concat_tri_param.get("shrink-x", [])
        sy = concat_tri_param.get("shrink-y", [])
        sl = concat_tri_param.get("strain-limit", [])
        if sx and sy and sl:
            conflict = _rust.scene_shell_shrink_strain_limit_conflict(
                [float(v) for v in sx],
                [float(v) for v in sy],
                [float(v) for v in sl],
            )
            if conflict is not None:
                i, x, y, s = conflict
                raise ValueError(
                    f"Shell face {i}: shrink (x={x}, y={y}) "
                    f"conflicts with strain-limit ({s}). "
                    "Set strain-limit to 0 or keep shrink at 1.0."
                )

        # Compute static vertices with displacement for intersection check
        static_vert_for_check = None
        static_tri_for_check = None
        if static_vert.shape[0] > 0 and static_tri.shape[0] > 0:
            static_vert_for_check = (
                static_vert + concat_displacement[static_vert_dmap]
            )
            static_tri_for_check = static_tri

        fixed = FixedScene(
            self._plot,
            self.info.name,
            map_by_name,
            concat_displacement,
            (concat_vert_dmap, concat_vert),
            concat_color,
            concat_dyn_tri_color,
            concat_dyn_tri_intensity,
            concat_vel,
            concat_uv,
            concat_rod,
            concat_tri,
            concat_tet,
            concat_rod_param,
            concat_tri_param,
            concat_tet_param,
            self._wall,
            self._sphere,
            (rod_vert_start, rod_vert_end),
            (shell_vert_start, shell_vert_end),
            rod_count,
            shell_count,
            concat_tri_is_collider_arr.astype(bool),
            concat_rod_is_collider_arr.astype(bool),
            global_pinned_vertices if global_pinned_vertices else None,
            static_vert_for_check,
            static_tri_for_check,
            self._surface_map_by_name if self._surface_map_by_name else None,
            concat_rest_vert=concat_rest_vert if has_rest_vert else None,
            # The mask is shared between the static rest_vert and the
            # time-varying schedule, so pass it whenever either is present.
            rest_vert_mask=(
                rest_vert_mask if (has_rest_vert or has_rest_vert_anim) else None
            ),
            rest_vert_anim=rest_vert_anim if has_rest_vert_anim else None,
            rest_vert_times=rest_vert_times if has_rest_vert_anim else None,
            bend_rest_vert=concat_bend_rest_vert if has_bend_rest_vert else None,
            bend_rest_vert_mask=bend_rest_vert_mask if has_bend_rest_vert else None,
            pdrd_body_rows=pdrd_body_rows if pdrd_body_rows else None,
            pdrd_vert_index=pdrd_vert_index if next_body_id > 0 else None,
            pdrd_vert_list=pdrd_vert_list if next_body_id > 0 else None,
            pdrd_rest_centered=pdrd_rest_centered if next_body_id > 0 else None,
            sand_param=concat_sand_param if concat_sand_param else None,
            quiet=quiet,
        )

        # Linear (velocity:) and angular (angular_velocity:) overwrite
        # schedules share the same dyn_param table (distinct key prefixes); the
        # solver dispatches on the prefix.
        merged_velocity_schedules = {
            **velocity_schedules_by_dmap,
            **angular_velocity_schedules_by_dmap,
            **angular_world_velocity_schedules_by_dmap,
        }
        if merged_velocity_schedules:
            fixed._velocity_schedules = merged_velocity_schedules
        if collision_windows_by_dmap:
            fixed._collision_windows_data = collision_windows_by_dmap

        if len(concat_pin):
            fixed.set_pin(concat_pin)

        if static_vert.shape[0]:
            fixed.set_static(
                (static_vert_dmap, static_vert),
                static_tri,
                static_color,
                concat_static_param,
                concat_static_transform_anims if concat_static_transform_anims else None,
            )

        if concat_stitch_ind_arr.shape[0] and concat_stitch_w_arr.shape[0]:
            fixed.set_stitch(
                concat_stitch_ind_arr,
                concat_stitch_w_arr,
                concat_stitch_stiffness_arr,
            )

        pbar.update(1)
        advance("Building scene: validating constraints...")
        pbar.close()

        return fixed
