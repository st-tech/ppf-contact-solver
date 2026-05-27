# File: _scene_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Scene + Object orchestration.

This module is the thin public entry point for the scene API. The
heavy implementation lives in sibling modules:

* ``_scene_collider_``: :class:`Wall`, :class:`Sphere` and their params.
* ``_scene_transform_``: quaternion / TRS helpers and :class:`TransformAnimation`.
* ``_scene_pin_``: pin operations, :class:`PinHolder`, :class:`PinData`.
* ``_scene_fixed_``: :class:`FixedScene` (the validated build result),
  :class:`EnumColor`, :class:`ValidationError`.

Everything is re-exported here for backward compatibility (older code and
tests import names directly from ``frontend._scene_``).
"""

import colorsys
from dataclasses import replace as _dc_replace
from typing import Any, Optional

import numpy as np
from tqdm.auto import tqdm

from . import _rust  # type: ignore[attr-defined]

from ._asset_ import AssetManager
from ._param_ import ParamHolder
from ._plot_ import Plot, PlotManager
from ._scene_collider_ import Sphere, Wall
from ._scene_fixed_ import EnumColor, FixedScene, ValidationError
from ._scene_object_ import Object
from ._scene_pin_ import PinData, PinHolder, TorqueOperation
from ._scene_transform_ import TransformAnimation
from ._utils_ import Utils

EPS = 1e-3


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
        self._explicit_merge_pairs: list[dict] = []
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

    def set_explicit_merge_pairs(self, pairs: list[dict]):
        """Set explicit per-vertex merge pairs captured at snap time.

        Each entry must contain non-empty ``source_uuid`` and ``target_uuid``;
        missing keys raise :class:`ValueError`.

        Example:
            Typically invoked internally by the Blender add-on decoder, but
            can be called directly to wire up per-vertex merge constraints::

                scene.set_explicit_merge_pairs([
                    {"source_uuid": "a", "target_uuid": "b",
                     "source_vert": 0, "target_vert": 12},
                ])
        """
        uuids = [
            (pair.get("source_uuid", ""), pair.get("target_uuid", ""))
            for pair in pairs
        ]
        _rust.scene_validate_merge_pair_uuids(uuids)
        self._explicit_merge_pairs = pairs

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
            return _rust.scene_reduce_axis_bound([], "min")
        counts = np.asarray([len(v) for v in per_obj_vert], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
        flat = np.concatenate([np.asarray(v, dtype=np.float64) for v in per_obj_vert])
        return _rust.scene_per_object_axis_bound(
            np.ascontiguousarray(flat, dtype=np.float64),
            offsets,
            ax,
            True,
        )

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
        ax = _rust.scene_axis_letter_to_index(axis)
        per_obj_vert = [
            v for v in (obj.vertex(True) for obj in self._object.values())
            if v is not None
        ]
        if not per_obj_vert:
            return _rust.scene_reduce_axis_bound([], "max")
        counts = np.asarray([len(v) for v in per_obj_vert], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
        flat = np.concatenate([np.asarray(v, dtype=np.float64) for v in per_obj_vert])
        return _rust.scene_per_object_axis_bound(
            np.ascontiguousarray(flat, dtype=np.float64),
            offsets,
            ax,
            False,
        )

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

    def build(self, progress_callback=None) -> FixedScene:
        """Build the fixed scene from the current scene.

        Args:
            progress_callback: Optional callable ``f(fraction, step_info)`` invoked as the build progresses.

        Returns:
            FixedScene: The built fixed scene.

        Example:
            Compile the scene and chain a summary print::

                fixed = scene.build().report()
                session = app.session.create(fixed).build()
        """
        total_steps = 12
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

        # Build vertex alias map from merge pairs
        # Objects are registered by UUID; merge pairs carry source_uuid/target_uuid
        vertex_alias: dict[tuple[str, int], tuple[str, int]] = {}
        if self._explicit_merge_pairs:
            for pair in self._explicit_merge_pairs:
                source_name = pair.get("source_uuid", "")
                target_name = pair.get("target_uuid", "")
                index_pairs = pair.get("pairs", [])
                if not source_name or not target_name:
                    raise ValueError("Merge pair missing source_uuid/target_uuid")
                source_obj = self._object.get(source_name)
                target_obj = self._object.get(target_name)
                if source_obj is None or target_obj is None:
                    raise ValueError(
                        f"Merge pair references unknown object: "
                        f"source={source_name!r} target={target_name!r}"
                    )
                if source_obj.static or target_obj.static:
                    continue
                merged_count = 0
                for src_i, tgt_i in index_pairs:
                    vertex_alias[(target_name, int(tgt_i))] = (source_name, int(src_i))
                    merged_count += 1
                if merged_count > 0:
                    print(
                        f"Explicit merge pair {source_name} <-> {target_name}: "
                        f"{merged_count} vertices aliased"
                    )
        # Alias resolution + rod/shell/finalize index-map construction
        # via the Rust kernel. The per-object dict packaging is a single
        # list comprehension (no .append in a loop).
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
        # Repack alias map into Rust's grouped-by-(source, target) form.
        flat_alias = [
            ((tgt_name, int(tgt_vi)), (src_name, int(src_vi)))
            for (tgt_name, tgt_vi), (src_name, src_vi) in vertex_alias.items()
        ]
        rust_merge = _rust.scene_group_vertex_alias(flat_alias)

        result = _rust.scene_build_index_map(rust_objects, rust_merge)
        map_by_name = dict(result["map_by_name"])
        concat_count = int(result["concat_count"])
        rod_vert_start, rod_vert_end = result["rod_vert_range"]
        shell_vert_start, shell_vert_end = result["shell_vert_range"]
        pbar.update(3)
        advance("Building scene: indexing rod topology...")
        advance("Building scene: indexing shell topology...")
        advance("Building scene: finalizing vertex map...")

        has_merge = bool(vertex_alias)

        # ----- Build kernel input: dyn_objects -----
        # For each dynamic object, prepare numpy arrays the kernel reads.
        # The per-object pin-index concat runs through Rust; the dyn_input
        # dict is built via a list comprehension (single allocation, no
        # incremental append). Velocity schedules + collision-window
        # lookups are dict assignments, not list growth.
        velocity_schedules: dict[str, Any] = {}
        collision_windows: dict[str, Any] = {}
        pinned_indices_by_name: dict[str, list[int]] = {}
        MAX_COLLISION_WINDOWS = 8
        for name, obj in dyn_objects:
            pinned_indices_by_name[name] = _rust.scene_concat_i64_lists(
                [list(p.index) for p in obj.pin_list]
            ).tolist()
            if obj._velocity_schedule:
                velocity_schedules[name] = [
                    (t, list(v)) for t, v in obj._velocity_schedule
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
                "ind": np.ascontiguousarray(np.asarray(cs["ind"]), dtype=np.int64).reshape(-1, 4),
                "w": np.ascontiguousarray(np.asarray(cs["w"]), dtype=np.float64).reshape(-1, 4),
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
            bool(has_merge),
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
        rod_count = int(assembled["rod_count"])
        shell_count = int(assembled["shell_count"])
        stats_by_name = assembled["stats_by_name"]

        # ----- Reconstruct UV / dyn-color list-shaped layout for FixedScene -----
        concat_uv: list[np.ndarray] = [
            concat_uv_arr[i].astype(np.float32).reshape(3, 2)
            for i in range(concat_uv_arr.shape[0])
        ]
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
        advance("Building scene: collecting pins and stitches...")

        # ----- Param replication using counts the Rust kernel returned -----
        # Mirrors `extend_param` from the original loop. Each object's param
        # values are appended N times where N is its surviving element count
        # for the relevant element kind.
        concat_rod_param: dict[str, list] = {}
        concat_tri_param: dict[str, list] = {}
        concat_tet_param: dict[str, list] = {}
        concat_static_param: dict[str, list] = {}

        def _extend_param(
            param: ParamHolder,
            concat_param: dict[str, list],
            count: int,
        ):
            if len(concat_param.keys()):
                assert param.key_list() == list(concat_param.keys()), (
                    f"param keys mismatch: {param.key_list()} vs {list(concat_param.keys())}"
                )
            for key, value in param.items():
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
                transition=p.transition,
                pin_group_id=p.pin_group_id,
                hide_in_preview=bool(getattr(obj, "_is_static_moving", False)),
            )
            for name, obj in dyn_objects
            for p in obj.pin_list
        ]

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
        for key in ["friction", "contact-gap", "contact-offset", "bend"]:
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
            rest_vert_mask=rest_vert_mask if has_rest_vert else None,
        )

        if velocity_schedules_by_dmap:
            fixed._velocity_schedules = velocity_schedules_by_dmap
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
            )

        pbar.update(1)
        advance("Building scene: validating constraints...")
        pbar.close()

        return fixed
