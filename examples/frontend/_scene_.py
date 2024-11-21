# File: _scene_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os
import colorsys
from typing import Optional
from ._plot_ import PlotManager, Plot
from ._asset_ import AssetManager
from IPython.display import display, HTML
from ipywidgets import interact

EPS = 1e-3


class SceneManager:
    def __init__(self, plot: PlotManager, asset: AssetManager, save_func):
        self._plot = plot
        self._asset = asset
        self._scene: dict[str, Scene] = {}
        self._save_func = save_func

    def create(self, name: str) -> "Scene":
        if name in self._scene.keys():
            raise Exception(f"scene {name} already exists")
        else:
            scene = Scene(self._plot, self._asset, self._save_func)
            self._scene[name] = scene
            return scene

    def select(self, name: str, create: bool = True) -> "Scene":
        if create and name not in self._scene.keys():
            return self.create(name)
        else:
            return self._scene[name]

    def remove(self, name: str):
        if name in self._scene.keys():
            del self._scene[name]

    def clear(self):
        self._scene = {}

    def list(self):
        return list(self._scene.keys())


class Wall:
    def __init__(self):
        self._normal = [0, 1, 0]
        self._entry = []

    def get_entry(self) -> list[tuple[list[float], float]]:
        return self._entry

    def add(self, pos: list[float], normal: list[float]) -> "Wall":
        if len(self._entry):
            raise Exception("wall already exists")
        else:
            self._normal = normal
            self._entry.append((pos, 0.0))
            return self

    def _check_time(self, time: float):
        if time <= self._entry[-1][1]:
            raise Exception("time must be greater than the last time")

    def move_to(self, pos: list[float], time: float) -> "Wall":
        self._check_time(time)
        self._entry.append((pos, time))
        return self

    def move_by(self, delta: list[float], time: float) -> "Wall":
        self._check_time(time)
        pos = self._entry[-1][0] + delta
        self._entry.append((pos, time))
        return self


class Sphere:
    def __init__(self):
        self._entry = []
        self._hemisphere = False
        self._invert = False

    def hemisphere(self) -> "Sphere":
        self._hemisphere = True
        return self

    def invert(self) -> "Sphere":
        self._invert = True
        return self

    def get_entry(self) -> list[tuple[list[float], float, float]]:
        return self._entry

    def add(self, pos: list[float], radius: float) -> "Sphere":
        if len(self._entry):
            raise Exception("sphere already exists")
        else:
            self._entry.append((pos, radius, 0.0))
            return self

    def _check_time(self, time: float):
        if time <= self._entry[-1][2]:
            raise Exception(
                "time must be greater than the last time. last time is %f"
                % self._entry[-1][2]
            )

    def transform_to(self, pos: list[float], radius: float, time: float) -> "Sphere":
        self._check_time(time)
        self._entry.append((pos, radius, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Sphere":
        self._check_time(time)
        radius = self._entry[-1][1]
        self._entry.append((pos, radius, time))
        return self

    def move_by(self, delta: list[float], time: float) -> "Sphere":
        self._check_time(time)
        pos = self._entry[-1][0] + delta
        radius = self._entry[-1][1]
        self._entry.append((pos, radius, time))
        return self

    def radius(self, radius: float, time: float) -> "Sphere":
        self._check_time(time)
        pos = self._entry[-1][0]
        self._entry.append((pos, radius, time))
        return self


class FixedScene:
    def __init__(
        self,
        plot: PlotManager,
        vert: np.ndarray,
        color: np.ndarray,
        vel: np.ndarray,
        uv: np.ndarray,
        rod: np.ndarray,
        tri: np.ndarray,
        tet: np.ndarray,
        wall: list[Wall],
        sphere: list[Sphere],
        rod_vert_range: tuple[int, int],
        shell_vert_range: tuple[int, int],
        rod_count: int,
        shell_count: int,
    ):
        self._plot = plot
        self._vert = vert
        self._color = color
        self._vel = vel
        self._uv = uv
        self._rod = rod
        self._tri = tri
        self._tet = tet
        self._pin = []
        self._static_vert = np.zeros(0)
        self._static_color = np.zeros(0)
        self._static_tri = np.zeros(0)
        self._stitch_ind = np.zeros(0)
        self._stitch_w = np.zeros(0)
        self._wall = wall
        self._sphere = sphere
        self._rod_vert_range = rod_vert_range
        self._shell_vert_range = shell_vert_range
        self._rod_count = rod_count
        self._shell_count = shell_count

    def report(self) -> "FixedScene":
        data = {}
        data["#vert"] = len(self._vert)
        if len(self._rod):
            data["#rod"] = len(self._rod)
        if len(self._tri):
            data["#tri"] = len(self._tri)
        if len(self._tet):
            data["#tet"] = len(self._tet)
        if len(self._pin):
            val = sum([len(pin) for pin, _, _, _, _ in self._pin])
            data["#pin"] = val
        if len(self._static_vert) and len(self._static_tri):
            data["#static_vert"] = len(self._static_vert)
            data["#static_tri"] = len(self._static_tri)
        if len(self._stitch_ind) and len(self._stitch_w):
            data["#stitch_ind"] = len(self._stitch_ind)
        for key, value in data.items():
            if isinstance(value, int):
                data[key] = [f"{value:,}"]
            elif isinstance(value, float):
                data[key] = [f"{value:.2e}"]
            else:
                data[key] = [str(value)]

        df = pd.DataFrame(data)
        html = df.to_html(classes="table", index=False)
        display(HTML(html))
        return self

    def export(
        self, vert: np.ndarray, path: str, include_static: bool = True
    ) -> "FixedScene":
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vert),
            o3d.utility.Vector3iVector(self._tri),
        )
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self._color)

        if include_static and len(self._static_vert) and len(self._static_tri):
            o3d_static = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self._static_vert),
                o3d.utility.Vector3iVector(self._static_tri),
            )
            o3d_static.vertex_colors = o3d.utility.Vector3dVector(self._static_color)
            o3d_mesh += o3d_static

        o3d.io.write_triangle_mesh(path, o3d_mesh)
        return self

    def export_fixed(self, path: str, delete_exist: bool) -> "FixedScene":
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
        info_path = os.path.join(path, "info.toml")
        with open(info_path, "w") as f:
            f.write("[count]\n")
            f.write(f"vert = {len(self._vert)}\n")
            f.write(f"rod = {len(self._rod)}\n")
            f.write(f"tri = {len(self._tri)}\n")
            f.write(f"tet = {len(self._tet)}\n")
            f.write(f"static_vert = {len(self._static_vert)}\n")
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
            f.write("\n")

            for i, (pin, target, _, unpin, pull) in enumerate(self._pin):
                f.write(f"[pin-{i}]\n")
                f.write(f"keyframe = {len(target)}\n")
                f.write(f"pin = {len(pin)}\n")
                f.write(f"pull = {float(pull)}\n")
                f.write(f"unpin = {'true' if unpin else 'false'}\n")
                f.write("\n")

            for i, wall in enumerate(self._wall):
                normal = wall._normal
                f.write(f"[wall-{i}]\n")
                f.write(f"keyframe = {len(wall._entry)}\n")
                f.write(f"nx = {float(normal[0])}\n")
                f.write(f"ny = {float(normal[1])}\n")
                f.write(f"nz = {float(normal[2])}\n")
                f.write("\n")

            for i, sphere in enumerate(self._sphere):
                f.write(f"[sphere-{i}]\n")
                f.write(f"keyframe = {len(sphere._entry)}\n")
                f.write(f"hemisphere = {'true' if sphere._hemisphere else 'false'}\n")
                f.write(f"invert = {'true' if sphere._invert else 'false'}\n")
                f.write("\n")

        bin_path = os.path.join(path, "bin")
        os.makedirs(bin_path)

        self._vert.astype(np.float32).tofile(os.path.join(bin_path, "vert.bin"))
        self._color.astype(np.float32).tofile(os.path.join(bin_path, "color.bin"))
        self._vel.astype(np.float32).tofile(os.path.join(bin_path, "vel.bin"))
        if np.linalg.norm(self._uv) > 0:
            self._uv.astype(np.float32).tofile(os.path.join(bin_path, "uv.bin"))

        if len(self._rod):
            self._rod.astype(np.uint64).tofile(os.path.join(bin_path, "rod.bin"))
        if len(self._tri):
            self._tri.astype(np.uint64).tofile(os.path.join(bin_path, "tri.bin"))
        if len(self._tet):
            self._tet.astype(np.uint64).tofile(os.path.join(bin_path, "tet.bin"))
        if len(self._static_vert):
            self._static_vert.astype(np.float32).tofile(
                os.path.join(bin_path, "static_vert.bin")
            )
            self._static_tri.astype(np.uint64).tofile(
                os.path.join(bin_path, "static_tri.bin")
            )
            self._static_color.astype(np.float32).tofile(
                os.path.join(bin_path, "static_color.bin")
            )
        if len(self._stitch_ind) and len(self._stitch_w):
            self._stitch_ind.astype(np.uint64).tofile(
                os.path.join(bin_path, "stitch_ind.bin")
            )
            self._stitch_w.astype(np.float32).tofile(
                os.path.join(bin_path, "stitch_w.bin")
            )
        for i, (pin, target, timing, _, _) in enumerate(self._pin):
            with open(os.path.join(bin_path, f"pin-ind-{i}.bin"), "wb") as f:
                np.array(pin, dtype=np.uint64).tofile(f)
            if len(target):
                target_dir = os.path.join(bin_path, f"pin-{i}")
                os.makedirs(target_dir)
                for j, pos in enumerate(target):
                    with open(os.path.join(target_dir, f"{j}.bin"), "wb") as f:
                        np.array(pos, dtype=np.float32).tofile(f)
            if len(timing):
                with open(os.path.join(bin_path, f"pin-timing-{i}.bin"), "wb") as f:
                    np.array(timing, dtype=np.float32).tofile(f)

        for i, wall in enumerate(self._wall):
            with open(os.path.join(bin_path, f"wall-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _ in wall._entry for p in pos], dtype=np.float32
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"wall-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, t in wall._entry], dtype=np.float32)
                timing.tofile(f)

        for i, sphere in enumerate(self._sphere):
            with open(os.path.join(bin_path, f"sphere-pos-{i}.bin"), "wb") as f:
                pos = np.array(
                    [p for pos, _, _ in sphere._entry for p in pos], dtype=np.float32
                )
                pos.tofile(f)
            with open(os.path.join(bin_path, f"sphere-radius-{i}.bin"), "wb") as f:
                radius = np.array([r for _, r, _ in sphere._entry], dtype=np.float32)
                radius.tofile(f)
            with open(os.path.join(bin_path, f"sphere-timing-{i}.bin"), "wb") as f:
                timing = np.array([t for _, _, t in sphere._entry], dtype=np.float32)
                timing.tofile(f)

        return self

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        vert = self._vert
        return (np.max(vert, axis=0), np.min(vert, axis=0))

    def center(self) -> np.ndarray:
        vert = self._vert
        tri = self._tri
        center = np.zeros(3)
        area_sum = 0
        for f in tri:
            a, b, c = vert[f[0]], vert[f[1]], vert[f[2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            center += area * (a + b + c) / 3
            area_sum += area
        if area_sum == 0:
            raise Exception("no area")
        else:
            return center / area_sum

    def _average_tri_area(self):
        vert = self._vert
        tri = self._tri
        area = 0
        for f in tri:
            a, b, c = vert[f[0]], vert[f[1]], vert[f[2]]
            area += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        return area / len(tri)

    def set_pin(
        self, pin: list[tuple[list[int], list[np.ndarray], list[float], bool, float]]
    ):
        self._pin = pin

    def set_static(self, vert: np.ndarray, tri: np.ndarray, color: np.ndarray):
        self._static_vert = vert
        self._static_tri = tri
        self._static_color = color

    def set_stitch(self, ind: np.ndarray, w: np.ndarray):
        self._stitch_ind = ind
        self._stitch_w = w

    def time(self, time: float) -> np.ndarray:
        vert = self._vert.copy()
        if len(self._pin):
            for pin, _target, _timing, _, _ in self._pin:
                for i, (_, _) in enumerate(zip(_target, _timing)):
                    if time >= _timing[-1]:
                        vert[pin] = _target[-1]
                        break
                    elif time >= _timing[i] and time < _timing[i + 1]:
                        q1, q2 = _target[i], _target[i + 1]
                        r = (time - _timing[i]) / (_timing[i + 1] - _timing[i])
                        vert[pin] = q1 * (1 - r) + q2 * r
                        break
        vert += time * self._vel
        return vert

    def check_intersection(self) -> "FixedScene":
        import open3d as o3d

        if len(self._vert) and len(self._tri):
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self._vert),
                o3d.utility.Vector3iVector(self._tri),
            )
            if o3d_mesh.is_self_intersecting():
                print("WARNING: mesh is self-intersecting")
            if len(self._static_vert) and len(self._static_tri):
                o3d_static = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(self._static_vert),
                    o3d.utility.Vector3iVector(self._static_tri),
                )
                if o3d_static.is_self_intersecting():
                    print("WARNING: static mesh is self-intersecting")
                if o3d_static.is_intersecting(o3d_mesh):
                    print("WARNING: mesh is intersecting with static mesh")
        return self

    def preview(
        self,
        vert: Optional[np.ndarray] = None,
        shading: dict = {},
        show_stitch: bool = True,
        show_pin: bool = True,
    ) -> "Plot":
        if vert is None:
            vert = self._vert
        color = self._color
        plot = None

        if len(self._tri):
            if plot is None:
                plot = self._plot.create().tri(
                    vert, self._tri, color=color, shading=shading
                )
            else:
                plot.add.tri(vert, self._tri, color)

        if len(self._static_vert):
            if plot is None:
                plot = self._plot.create().tri(
                    self._static_vert,
                    self._static_tri,
                    color=self._static_color,
                )
            else:
                plot.add.tri(
                    self._static_vert,
                    self._static_tri,
                    self._static_color,
                )

        if len(self._rod):
            if plot is None:
                plot = self._plot.create().curve(vert, self._rod, shading=shading)
            else:
                plot.add.edge(vert, self._rod)

        if plot is not None:
            if show_stitch and len(self._stitch_ind) and len(self._stitch_w):
                stitch_vert, stitch_edge = [], []
                for ind, w in zip(self._stitch_ind, self._stitch_w):
                    x0, y0, y1 = vert[ind[0]], vert[ind[1]], vert[ind[2]]
                    w0, w1 = w[0], w[1]
                    idx0, idx1 = len(stitch_vert), len(stitch_vert) + 1
                    stitch_vert.append(x0)
                    stitch_vert.append(w0 * y0 + w1 * y1)
                    stitch_edge.append([idx0, idx1])
                stitch_vert = np.array(stitch_vert)
                stitch_edge = np.array(stitch_edge)
                if plot._darkmode:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                plot.add.edge(stitch_vert, stitch_edge)

            has_vel = np.linalg.norm(self._vel) > 0
            if show_pin and (self._pin or has_vel):
                max_time = 0
                if self._pin:
                    avg_area = self._average_tri_area()
                    avg_length = np.sqrt(avg_area)
                    shading["point_size"] = 5 * avg_length
                    pin_verts = np.vstack([vert[pin] for pin, _, _, _, _ in self._pin])
                    plot.add.point(pin_verts)
                    max_time = max(
                        [
                            timing[-1] if len(timing) else 0
                            for _, _, timing, _, _ in self._pin
                        ]
                    )
                if has_vel:
                    max_time = max(max_time, 1.0)
                if max_time > 0:

                    def update(time=0):
                        nonlocal plot
                        vert = self.time(time)
                        if plot is not None:
                            plot.update(vert)

                    interact(update, time=(0, max_time, 0.01))
            return plot
        else:
            raise Exception("no plot")


class Scene:
    def __init__(self, plot: PlotManager, asset: AssetManager, save_func):
        self._plot = plot
        self._asset = asset
        self._save_func = save_func
        self._object = {}
        self._sphere = []
        self._wall = []

    def clear(self) -> "Scene":
        self._object.clear()
        return self

    def add(self, mesh_name: str, ref_name: str = "") -> "Object":
        if ref_name == "":
            ref_name = mesh_name
            count = 0
            while ref_name in self._object.keys():
                count += 1
                ref_name = f"{mesh_name}_{count}"
        mesh_list = self._asset.list()
        if mesh_name not in mesh_list:
            raise Exception(f"mesh_name '{mesh_name}' does not exist")
        elif ref_name in self._object.keys():
            raise Exception(f"ref_name '{ref_name}' already exists")
        else:
            obj = Object(self._asset, mesh_name)
            self._object[ref_name] = obj
            return obj

    def pick(self, name: str) -> "Object":
        if name not in self._object.keys():
            raise Exception(f"object {name} does not exist")
        else:
            return self._object[name]

    def add_invisible_sphere(self, position: list[float], radius: float) -> Sphere:
        sphere = Sphere().add(position, radius)
        self._sphere.append(sphere)
        return sphere

    def add_invisible_wall(self, position: list[float], normal: list[float]) -> Wall:
        wall = Wall().add(position, normal)
        self._wall.append(wall)
        return wall

    def build(self) -> FixedScene:
        pbar = tqdm(total=10, desc="build", ncols=70)
        concat_count = 0
        dyn_objects = [
            obj
            for obj in self._object.values()
            if not obj.is_static() and not obj._color
        ]
        n = len(dyn_objects)
        for i, obj in enumerate(dyn_objects):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.75, 1.0)
            obj.default_color(r, g, b)

        def add_entry(
            map,
            entry,
        ):
            nonlocal concat_count
            for e in entry:
                for vi in e:
                    if map[vi] == -1:
                        map[vi] = concat_count
                        concat_count += 1

        tag = {}
        for name, obj in self._object.items():
            assert name not in tag
            if not obj.is_static():
                vert = obj.get("V")
                tag[name] = [-1] * len(vert)

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static() and obj.get("T") is None:
                map = tag[name]
                vert, edge = obj.get("V"), obj.get("E")
                if vert is not None and edge is not None:
                    add_entry(
                        map,
                        edge,
                    )
        rod_vert_start, rod_vert_end = 0, concat_count

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static() and obj.get("T") is None:
                map = tag[name]
                vert, tri = obj.get("V"), obj.get("F")
                if vert is not None and tri is not None:
                    add_entry(
                        map,
                        tri,
                    )
        shell_vert_start, shell_vert_end = rod_vert_end, concat_count

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                vert = obj.get("V")
                map = tag[name]
                for i in range(len(vert)):
                    if map[i] == -1:
                        map[i] = concat_count
                        concat_count += 1

        concat_vert = np.zeros((concat_count, 3))
        concat_color = np.zeros((concat_count, 3))
        concat_vel = np.zeros((concat_count, 3))
        concat_uv = np.zeros((concat_count, 2))
        concat_pin = []
        concat_rod = []
        concat_tri = []
        concat_tet = []
        concat_static_vert = []
        concat_static_tri = []
        concat_static_color = []
        concat_stitch_ind = []
        concat_stitch_w = []

        def vec_map(map, elm):
            result = elm.copy()
            for i in range(len(elm)):
                result[i] = [map[vi] for vi in elm[i]]
            return result

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                map = tag[name]
                vert, uv = obj.vertex(), obj._uv
                if vert is not None:
                    concat_vert[map] = vert
                    concat_vel[map] = obj._velocity
                    concat_color[map] = obj.get("color")
                if uv is not None:
                    concat_uv[map] = uv

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                map = tag[name]
                edge = obj.get("E")
                if edge is not None and obj.get("T") is None:
                    concat_rod.extend(vec_map(map, edge))
        rod_count = len(concat_rod)

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                map = tag[name]
                tri = obj.get("F")
                if tri is not None and obj.get("T") is None:
                    concat_tri.extend(vec_map(map, tri))
        shell_count = len(concat_tri)

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                map = tag[name]
                tet = obj.get("T")
                tri = obj.get("F")
                if tet is not None and tri is not None:
                    concat_tri.extend(vec_map(map, tri))
                    concat_tet.extend(vec_map(map, tet))

        pbar.update(1)
        for name, obj in self._object.items():
            if not obj.is_static():
                map = tag[name]
                stitch_ind = obj.get("Ind")
                stitch_w = obj.get("W")
                if len(obj._pin):
                    concat_pin.append(
                        (
                            [map[vi] for vi in obj._pin],
                            obj._move,
                            obj._time,
                            obj._unpin,
                            obj._pull,
                        )
                    )
                if stitch_ind is not None and stitch_w is not None:
                    concat_stitch_ind.extend(vec_map(map, stitch_ind))
                    concat_stitch_w.extend(stitch_w)

        pbar.update(1)
        for name, obj in self._object.items():
            if obj.is_static():
                color = obj.get("color")
                offset = len(concat_static_vert)
                tri = obj.get("F")
                if tri is not None:
                    vert = obj.get("V")
                    concat_static_tri.extend(tri + offset)
                    concat_static_vert.extend(obj.apply_transform(vert))
                    concat_static_color.extend([color] * len(vert))

        self._save_func()
        pbar.update(1)

        fixed = FixedScene(
            self._plot,
            concat_vert,
            concat_color,
            concat_vel,
            concat_uv,
            np.array(concat_rod),
            np.array(concat_tri),
            np.array(concat_tet),
            self._wall,
            self._sphere,
            (rod_vert_start, rod_vert_end),
            (shell_vert_start, shell_vert_end),
            rod_count,
            shell_count,
        )

        if len(concat_pin):
            fixed.set_pin(concat_pin)

        if len(concat_static_vert):
            fixed.set_static(
                np.array(concat_static_vert),
                np.array(concat_static_tri),
                np.array(concat_static_color),
            )

        if len(concat_stitch_ind) and len(concat_stitch_w):
            fixed.set_stitch(
                np.array(concat_stitch_ind),
                np.array(concat_stitch_w),
            )

        return fixed


class Object:
    def __init__(self, asset: AssetManager, name: str):
        self._asset = asset
        self._name = name
        self.clear()

    def clear(self):
        self._param = {}
        self._at = [0, 0, 0]
        self._scale = 1.0
        self._rotation = np.eye(3)
        self._color = []
        self._static_color = [0.75, 0.75, 0.75]
        self._default_color = [1.0, 0.85, 0.0]
        self._velocity = [0, 0, 0]
        self._pin = []
        self._unpin = False
        self._pull = 0
        self._move = []
        self._time = []
        self._normalize = False
        self._stitch = None
        self._uv = None

    def report(self):
        print("at:", self._at)
        print("scale:", self._scale)
        print("rotation:")
        print(self._rotation)
        print("color:", self._color)
        print("velocity:", self._velocity)
        print("normalize:", self._normalize)
        print("pin:", len(self._pin))

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        vert = self.get("V")
        if vert is None:
            raise Exception("vertex does not exist")
        else:
            transformed = self.apply_transform(vert)
            max_x, max_y, max_z = np.max(transformed, axis=0)
            min_x, min_y, min_z = np.min(transformed, axis=0)
            return (
                np.array(
                    [
                        max_x - min_x,
                        max_y - min_y,
                        max_z - min_z,
                    ]
                ),
                np.array(
                    [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2]
                ),
            )

    def normalize(self) -> "Object":
        if self._normalize:
            raise Exception("already normalized")
        else:
            self._bbox, self._center = self.bbox()
            self._normalize = True
            return self

    def get(self, key: str) -> Optional[np.ndarray]:
        if key == "color":
            if self._color:
                return np.array(self._color)
            else:
                if self.is_static():
                    return np.array(self._static_color)
                else:
                    return np.array(self._default_color)
        elif key == "Ind":
            if self._stitch is not None:
                return self._stitch[0]
            else:
                return None
        elif key == "W":
            if self._stitch is not None:
                return self._stitch[1]
            else:
                return None
        else:
            result = self._asset.fetch.get(self._name)
            if key in result.keys():
                return result[key]
            else:
                return None

    def vertex(self) -> np.ndarray:
        vert = self.get("V")
        if vert is None:
            raise Exception("vertex does not exist")
        else:
            transformed = self.apply_transform(vert)
            return transformed

    def grab(self, uv: list[float], eps: float = 1e-3) -> list[int]:
        vert = self.vertex()
        val = np.max(np.dot(vert, np.array(uv)))
        return np.where(np.dot(vert, uv) > val - eps)[0].tolist()

    def set(self, key: str, value) -> "Object":
        self._param[key] = value
        return self

    def at(self, x: float, y: float, z: float) -> "Object":
        self._at = [x, y, z]
        return self

    def atop(self, object: "Object", margin: float = 0.0) -> "Object":
        a_bbox, a_center = self.bbox()
        b_bbox, b_center = object.bbox()
        center = b_center - a_center
        center[1] += (a_bbox[1] + b_bbox[1]) / 2.0 + margin
        return self.at(*center)

    def scale(self, _scale: float) -> "Object":
        self._scale = _scale
        return self

    def rotate(self, angle: float, axis: str) -> "Object":
        theta = angle / 180.0 * np.pi
        if axis.lower() == "x":
            self._rotation = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
                @ self._rotation
            )
        elif axis.lower() == "y":
            self._rotation = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
                @ self._rotation
            )
        elif axis.lower() == "z":
            self._rotation = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
                @ self._rotation
            )
        else:
            raise Exception("invalid axis")
        return self

    def max(self, dim: int):
        vert = self.vertex()
        return np.max([x[dim] for x in vert])

    def min(self, dim: int):
        vert = self.vertex()
        return np.min([x[dim] for x in vert])

    def apply_transform(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            raise Exception("vertex should be 2D array")
        else:
            x = x.transpose()
        if self._normalize:
            x = (x - self._center) / np.max(self._bbox)
        x = self._rotation @ x
        x = x * self._scale + np.array(self._at).reshape((3, 1))
        return x.transpose()

    def static_color(self, red: float, green: float, blue: float) -> "Object":
        self._static_color = [red, green, blue]
        return self

    def default_color(self, red: float, green: float, blue: float) -> "Object":
        self._default_color = [red, green, blue]
        return self

    def color(self, red: float, green: float, blue: float) -> "Object":
        self._color = [red, green, blue]
        return self

    def velocity(self, u: float, v: float, w: float) -> "Object":
        if self.is_static():
            raise Exception("object is static")
        else:
            self._velocity = np.array([u, v, w])
            return self

    def is_static(self) -> bool:
        if len(self._pin) == 0:
            return False
        else:
            vert = self.get("V")
            if vert is None:
                return False
            else:
                n_vert = len(vert)
                return len(self._move) == 0 and self._pin == list(range(n_vert))

    def pin(self, ind: Optional[list[int]] = None) -> "Object":
        if ind is None:
            vert: np.ndarray = self.vertex()
            ind = list(range(len(vert)))
        self._pin = ind
        return self

    def pull_pin(self, value: float = 1.0, ind: Optional[list[int]] = None) -> "Object":
        self.pin(ind)
        self._pull = value
        return self

    def unpin(self) -> "Object":
        if len(self._pin) == 0:
            raise Exception("pin must be set before unpinning")
        self._unpin = True
        return self

    def move_by(self, delta_pos, time: float) -> "Object":
        delta_pos = np.array(delta_pos).reshape((-1, 3))
        if len(self._move) == 0:
            target = self.vertex()[self._pin] + delta_pos
        else:
            target = self._move[-1] + delta_pos
        return self.move_to(target, time)

    def hold(self, time: float) -> "Object":
        return self.move_by([0, 0, 0], time)

    def move_to(
        self,
        target: np.ndarray,
        time: float,
    ) -> "Object":
        if len(self._pin) == 0:
            raise Exception("pin must be set before moving")
        elif len(target) != len(self._pin):
            raise Exception("target must have the same length as pin")
        elif time == 0:
            raise Exception("time must be greater than zero")
        else:
            if len(self._time) and time <= self._time[-1]:
                raise Exception("time must be greater than the last time")
            elif len(self._time) == 0:
                vert = self.vertex()[self._pin]
                self._move = [vert, target]
                self._time = [0, time]
            else:
                self._move.append(target)
                self._time.append(time)
            return self

    def stitch(self, name: str) -> "Object":
        if self.is_static():
            raise Exception("object is static")
        else:
            stitch = self._asset.fetch.get(name)
            if "Ind" not in stitch.keys():
                raise Exception("Ind not found in stitch")
            elif "W" not in stitch.keys():
                raise Exception("W not found in stitch")
            else:
                self._stitch = (stitch["Ind"], stitch["W"])
                return self

    def direction(self, _ex: list[float], _ey: list[float]) -> "Object":
        vert, tri = self.vertex(), self.get("F")
        ex = np.array(_ex)
        ex = ex / np.linalg.norm(ex)
        ey = np.array(_ey)
        ey = ey / np.linalg.norm(ey)
        if abs(np.dot(ex, ey)) > EPS:
            raise Exception(f"ex and ey must be orthogonal. ex: {ex}, ey: {ey}")
        elif vert is None:
            raise Exception("vertex does not exist")
        elif tri is None:
            raise Exception("face does not exist")
        else:
            for t in tri:
                a, b, c = vert[t]
                n = np.cross(b - a, c - a)
                n = n / np.linalg.norm(n)
                if abs(np.dot(n, _ex)) > EPS:
                    raise Exception(
                        f"ex must be orthogonal to the face normal. normal: {n}"
                    )
                elif abs(np.dot(n, _ey)) > EPS:
                    raise Exception(
                        f"ey must be orthogonal to the face normal. normal: {n}"
                    )
            self._uv = np.zeros((len(vert), 2))
            for i, x in enumerate(vert):
                u, v = x.dot(ex), x.dot(ey)
                self._uv[i] = [u, v]
        return self
