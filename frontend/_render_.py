# File: _render_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import shutil

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]

from ._rasterizer_ import DEFAULT_HEIGHT, DEFAULT_WIDTH, SoftwareRenderer
from ._utils_ import get_cache_dir

# SoftwareRenderer dispatches to the Rust rasterizer kernel.

def update_default_args(args: dict):
    """Fill missing entries in ``args`` with the renderer defaults.

    The default-args literal lives in
    :func:`_ppf_cts_py.render_default_args`. ``args`` is mutated in
    place: every missing key is filled in.

    Args:
        args (dict): The arguments dictionary to populate in place.
    """
    _rust.render_default_args(args, os.path.join(get_cache_dir(), "tmp_mesh.ply"))


# Rasterizer is the main rendering class; dispatches to the Rust kernel.
Rasterizer = SoftwareRenderer


class MitsubaRenderer:
    """Path-traced mesh renderer backed by the Mitsuba 3 executable.

    Requires the ``mitsuba`` command-line tool to be available on ``PATH`` and
    the ``mitsuba`` Python package to be importable at render time.
    """

    def __init__(self, args: Optional[dict] = None):
        """Initialize the renderer and validate that Mitsuba is available.

        Args:
            args (Optional[dict]): Optional configuration; missing keys are
                filled by :func:`update_default_args` (which delegates to
                :func:`_ppf_cts_py.render_default_args`). Recognized keys:
                ``variant``, ``max_depth``, ``width``, ``height``, ``fov``,
                ``camera``, ``up``, ``sample_count``, ``tmp_path``.
        """
        self._args = None
        if args is None:
            args = {}
        assert shutil.which("mitsuba") is not None
        update_default_args(args)
        self._args = args

    def __del__(self):
        """Remove the temporary PLY file written during rendering, if any."""
        args = getattr(self, "_args", None)
        if args is not None and os.path.exists(args["tmp_path"]):
            os.remove(args["tmp_path"])

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        path: str,
    ):
        """Render the mesh to an image file using Mitsuba.

        The mesh is exported to a temporary PLY file and then loaded into a
        Mitsuba scene with a directional light, a constant emitter, and a
        back wall. Line segments are not rendered; a notice is printed if
        ``seg`` is non-empty.

        Args:
            vert (np.ndarray): The vertices (Nx3) of the mesh.
            color (np.ndarray): Per-vertex colors (Nx3) in [0, 1].
            seg (np.ndarray): Line segment indices (Sx2); not drawn.
            face (np.ndarray): Triangle face indices (Fx3).
            path (str): Output image file path.
        """
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

        width, height = self._args["width"], self._args["height"]
        if type(self._args["camera"]) is dict:
            origin = self._args["camera"]["origin"]
            target = self._args["camera"]["target"]
            wall_z = float(np.max(vert[:, 2]) - np.min(vert[:, 2]))
        else:
            origin, target, wall_z = _rust.mitsuba_auto_camera(
                np.ascontiguousarray(vert, dtype=np.float32), int(width), int(height)
            )
            origin, target = list(origin), list(target)
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
                    .translate(mi.ScalarPoint3f([0, 0, -5 * wall_z])),
                },
                "my-mesh": mesh,
            }
        )
        image = mi.render(scene)  # pyright: ignore
        mi.util.write_bitmap(path, image)

    def _export_ply(
        self, path: str, vertex: np.ndarray, color: np.ndarray, face: np.ndarray
    ):
        """Write the mesh to ``path`` as a binary little-endian PLY file.

        Per-vertex RGB colors are stored as float properties named ``red``,
        ``green``, and ``blue`` so that Mitsuba can sample them via the
        ``mesh_attribute`` plugin as ``vertex_color``.
        """
        _rust.write_ply_binary(
            path,
            np.ascontiguousarray(vertex, dtype=np.float32),
            np.ascontiguousarray(color[:, :3], dtype=np.float32),
            np.ascontiguousarray(face, dtype=np.int32),
        )


if __name__ == "__main__":
    import argparse

    import trimesh

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", "--engine", type=str, required=True)
    arg_parser.add_argument("-i", "--input", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str, required=True)
    arg_parser.add_argument("--origin", type=float, nargs=3)
    arg_parser.add_argument("--target", type=float, nargs=3)
    arg_parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    arg_parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
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

    def load_mesh_entry(ply_path):
        """Load a ``.ply`` mesh and its optional ``.seg`` sidecar.

        Returns the vertices, per-vertex colors normalized to [0, 1], the line
        segment indices (Sx2, empty if no sidecar), and the triangle faces.
        """
        print(f"loading mesh {ply_path}")
        mesh = trimesh.load_mesh(ply_path)
        segpath = os.path.splitext(ply_path)[0] + ".seg"
        if os.path.exists(segpath):
            seg = np.loadtxt(segpath, dtype=np.uint32).reshape((-1, 2))
        else:
            seg = np.zeros((0, 2), dtype=np.uint32)
        color = mesh.visual.vertex_colors  # type: ignore
        color = color[:, :3] / 255.0
        return mesh.vertices, color, seg, mesh.faces

    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        entries = []
        for filename in os.listdir(args.input):
            if filename.endswith(".ply"):
                filepath = os.path.join(args.input, filename)
                vert, color, seg, face = load_mesh_entry(filepath)
                output_file = os.path.join(
                    args.output, f"{os.path.splitext(filename)[0]}.png"
                )
                entries.append((vert, color, seg, face, output_file))
        for vert, color, seg, face, output_file in entries:
            print(f"rendering mesh to {output_file}")
            r.render(vert, color, seg, face, output_file)
    else:
        assert args.input.endswith(".ply")
        vert, color, seg, face = load_mesh_entry(args.input)
        print(f"rendering mesh to {args.output}")
        r.render(vert, color, seg, face, args.output)
