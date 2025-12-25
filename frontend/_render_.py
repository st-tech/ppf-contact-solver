# File: _render_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import shutil

from typing import Optional

import numpy as np

from ._rasterizer_ import SoftwareRenderer
from ._utils_ import get_cache_dir

# SoftwareRenderer is always available (uses numpy + numba, works on all platforms)

default_args = {
    "variant": "cuda_ad_rgb",
    "max_depth": 12,
    "width": 640,
    "height": 480,
    "fov": 20,
    "camera": None,
    "up": [0, 1, 0],
    "sample_count": 64,
    "tmp_path": os.path.join(get_cache_dir(), "tmp_mesh.ply"),
}


def update_default_args(args: dict):
    for key, value in default_args.items():
        if key not in args:
            args[key] = value


# Rasterizer is the main rendering class (pure numpy + numba, works on all platforms)
Rasterizer = SoftwareRenderer


class MitsubaRenderer:
    def __init__(self, args: Optional[dict] = None):
        if args is None:
            args = {}
        assert shutil.which("mitsuba") is not None
        update_default_args(args)
        self._args = args

    def __del__(self):
        if os.path.exists(self._args["tmp_path"]):
            os.remove(self._args["tmp_path"])

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        path: str,
    ):
        import mitsuba as mi  # type: ignore[import-not-found]

        mi.set_variant(self._args["variant"])
        self._export_ply(self._args["tmp_path"], vert, color, face)
        mesh = mi.load_dict(
            {
                "type": "ply",
                "filename": self._args["tmp_path"],
                "bsdf": {
                    "type": "twosided",
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "mesh_attribute",
                            "name": "vertex_color",
                        },
                    },
                },
            }
        )
        if len(seg):
            print("Mitsuba does not support line primitives with varying colors.")

        bounds = np.max(vert, axis=0) - np.min(vert, axis=0)
        width, height = self._args["width"], self._args["height"]
        if type(self._args["camera"]) is dict:
            origin = self._args["camera"]["origin"]
            target = self._args["camera"]["target"]
        else:
            rat = width / height
            target = (bounds / 2) + np.min(vert, axis=0)
            origin = target + np.max(bounds) * 0.5 * rat * np.array([0, 1, 5])
            target, origin = target.tolist(), origin.tolist()
        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": self._args["max_depth"]},
                "sensor": {
                    "type": "perspective",
                    "fov": self._args["fov"],
                    "fov_axis": "x",
                    "to_world": mi.ScalarTransform4f().look_at(
                        origin=mi.ScalarPoint3f(origin),
                        target=mi.ScalarPoint3f(target),
                        up=mi.ScalarPoint3f(self._args["up"]),
                    ),
                    "sampler": {
                        "type": "independent",
                        "sample_count": self._args["sample_count"],
                    },
                    "film": {
                        "type": "hdrfilm",
                        "width": width,
                        "height": height,
                    },
                },
                "emitter-1": {
                    "type": "directional",
                    "direction": [1, -1, -1],
                    "irradiance": {"type": "rgb", "value": 2.0},
                },
                "emitter-2": {
                    "type": "constant",
                    "radiance": {"type": "rgb", "value": 0.25},
                },
                "wall": {
                    "type": "rectangle",
                    "emitter": {
                        "type": "area",
                        "radiance": {"type": "rgb", "value": 0.75},
                    },
                    "to_world": mi.ScalarTransform4f()
                    .scale(mi.ScalarPoint3f([1000, 1000, 1]))
                    .translate(mi.ScalarPoint3f([0, 0, -5 * bounds[2]])),
                },
                "my-mesh": mesh,
            }
        )
        image = mi.render(scene)  # pyright: ignore
        mi.util.write_bitmap(path, image)

    def _export_ply(
        self, path: str, vertex: np.ndarray, color: np.ndarray, face: np.ndarray
    ):
        with open(path, "wb") as ply_file:
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"element vertex {vertex.shape[0]}\n".encode())
            ply_file.write(b"property float x\n")
            ply_file.write(b"property float y\n")
            ply_file.write(b"property float z\n")
            ply_file.write(b"property float red\n")
            ply_file.write(b"property float green\n")
            ply_file.write(b"property float blue\n")
            ply_file.write(f"element face {face.shape[0]}\n".encode())
            ply_file.write(b"property list uchar int vertex_indices\n")
            ply_file.write(b"end_header\n")
            for i in range(vertex.shape[0]):
                assert vertex[i].shape == (3,)
                assert color[i].shape == (3,)
                ply_file.write(vertex[i].astype(np.float32).tobytes())
                ply_file.write(color[i].astype(np.float32).tobytes())
            for i in range(face.shape[0]):
                assert face[i].shape == (3,)
                ply_file.write(np.array([3], dtype=np.uint8).tobytes())
                ply_file.write(face[i].astype(np.int32).tobytes())


if __name__ == "__main__":
    import argparse

    import trimesh

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", "--engine", type=str, required=True)
    arg_parser.add_argument("-i", "--input", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str, required=True)
    arg_parser.add_argument("--origin", type=float, nargs=3)
    arg_parser.add_argument("--target", type=float, nargs=3)
    arg_parser.add_argument("--width", type=int, default=640)
    arg_parser.add_argument("--height", type=int, default=480)
    arg_parser.add_argument("--sample-count", type=int, default=64)
    args = arg_parser.parse_args()
    if args.origin is not None and args.target is not None:
        options = {
            "camera": {
                "origin": args.origin,
                "target": args.target,
            },
        }
    else:
        options = {}
    options["width"] = args.width
    options["height"] = args.height
    options["sample_count"] = args.sample_count

    if args.engine == "software":
        r = Rasterizer(options)
    elif args.engine == "mitsuba":
        r = MitsubaRenderer(options)
    else:
        raise ValueError(f"unsupported engine {args.engine}")

    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        entries = []
        for filename in os.listdir(args.input):
            if filename.endswith(".ply"):
                filepath = os.path.join(args.input, filename)
                segpath = filepath.replace(".ply", ".seg")
                if os.path.exists(segpath):
                    seg = np.loadtxt(segpath, dtype=np.uint32).reshape((-1, 2))
                else:
                    seg = np.zeros((0, 2), dtype=np.uint32)
                print(f"loading mesh {filepath}")
                mesh = trimesh.load_mesh(filepath)
                vert = mesh.vertices
                face = mesh.faces
                color = mesh.visual.vertex_colors  # type: ignore
                color = color[:, :3] / 255.0
                output_file = os.path.join(
                    args.output, f"{os.path.splitext(filename)[0]}.png"
                )
                entries.append((vert, color, seg, face, output_file))
        for vert, color, seg, face, output_file in entries:
            print(f"rendering mesh to {output_file}")
            r.render(vert, color, seg, face, output_file)
    else:
        assert args.input.endswith(".ply")
        print(f"loading mesh {args.input}")
        mesh = trimesh.load_mesh(args.input)
        vert = mesh.vertices
        segpath = args.input.replace(".ply", ".seg")
        if os.path.exists(segpath):
            seg = np.loadtxt(segpath, dtype=np.uint32).reshape((-1, 2))
        else:
            seg = np.zeros((0, 2), dtype=np.uint32)
        face = mesh.faces
        color = mesh.visual.vertex_colors  # type: ignore
        color = color[:, :3] / 255.0
        print(f"rendering mesh to {args.output}")
        r.render(vert, color, seg, face, args.output)
