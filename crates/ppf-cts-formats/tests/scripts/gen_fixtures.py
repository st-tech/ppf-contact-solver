#!/usr/bin/env python3
# File: crates/ppf-cts-formats/tests/scripts/gen_fixtures.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cross-language fixture generator.
#
# Builds representative dicts shaped like what the Blender addon's
# encoders emit, runs cbor2.dumps, and writes the bytes to
# tests/cbor_fixtures/. The matching Rust
# integration test (tests/cross_lang.rs) deserializes the same files
# through serde, proving the producer ↔ schema round-trip works at the
# language boundary, not just within Rust.
#
# Run:
#   .venv/bin/python crates/ppf-cts-formats/tests/scripts/gen_fixtures.py
#
# Re-run after any schema or addon-encoder change. Fixture files are
# checked in so the Rust test is reproducible without invoking Python.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ADDON_ENCODER = REPO / "blender_addon" / "core" / "encoder" / "cbor_encode.py"
OUT = Path(__file__).resolve().parents[1] / "cbor_fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def _load_addon_encoder():
    """Import the addon's CBOR encoder without dragging in bpy.

    Side-steps blender_addon's relative-import init chain so we can
    drive the same dumps_envelope() the addon will call at upload time.
    The encoder's lazy ``from ..module import get_cbor2`` is satisfied by
    registering minimal stub parent packages, so we never import
    blender_addon's bpy-dependent __init__ chain.
    """
    def _ensure_pkg(name):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as a package so submodules resolve
            sys.modules[name] = pkg
        return sys.modules[name]

    _ensure_pkg("blender_addon")
    _ensure_pkg("blender_addon.core")
    _ensure_pkg("blender_addon.core.encoder")
    core_module = _ensure_pkg("blender_addon.core.module")
    core_module.get_cbor2 = lambda: __import__("cbor2")

    fqname = "blender_addon.core.encoder.cbor_encode"
    spec = importlib.util.spec_from_file_location(fqname, ADDON_ENCODER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = mod
    spec.loader.exec_module(mod)
    return mod


_addon = _load_addon_encoder()


def dumps(kind: str, payload: object) -> bytes:
    """Single chokepoint: every fixture goes through the addon encoder."""
    return _addon.dumps_envelope(kind, payload)


def make_scene_payload() -> list:
    """Mirror blender_addon/core/encoder/mesh.py:_encode_obj_inner output.

    Uses ndarrays + np scalars to match what `_build_obj_data` actually
    produces; the encoder's `_numpy_default` callback then turns them
    into wire-form lists / floats.
    """
    eye4 = np.eye(4, dtype=np.float64)

    canonical_shell = {
        "name": "cloth_a",
        "uuid": "uuid-shell-1",
        "vert": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "face": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        # detect_stitch_edges returns (edges_int_array, weights_float_array)
        "stitch": (
            np.array([[0, 1]], dtype=np.uint32),
            np.array([1.0], dtype=np.float32),
        ),
        "uv": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ],
        "pin": [0, 3],
        "transform": eye4.copy(),
    }

    dup_xform = eye4.copy()
    dup_xform[0, 3] = 2.5  # translated +2.5 on X
    duplicate_shell = {
        "name": "cloth_b",
        "uuid": "uuid-shell-2",
        "mesh_ref": "uuid-shell-1",
        "transform": dup_xform,
    }

    rod = {
        "name": "thread",
        "uuid": "uuid-rod-1",
        "vert": np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32
        ),
        "edge": np.array([[0, 1], [1, 2]], dtype=np.uint32),
        "pin": [0],
        "transform": eye4.copy(),
    }

    static_with_anim = {
        "name": "ground",
        "uuid": "uuid-static-1",
        "vert": np.array(
            [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 0.0, 10.0]],
            dtype=np.float32,
        ),
        "face": np.array([[0, 1, 2]], dtype=np.uint32),
        "transform_animation": {
            "time": [0.0, 1.0, 2.0],
            "translation": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            "quaternion": [[0.0, 0.0, 0.0, 1.0]] * 3,
            "scale": [[1.0, 1.0, 1.0]] * 3,
        },
        "transform": eye4.copy(),
    }

    static_with_ops = {
        "name": "spinner",
        "uuid": "uuid-static-2",
        "vert": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        "face": np.array([[0, 1, 2]], dtype=np.uint32),
        "static_ops": [
            {"op_type": "MOVE_BY", "t_start": 0.0, "t_end": 1.0,
             "transition": "linear", "delta": [1.0, 0.0, 0.0]},
            {"op_type": "SPIN", "t_start": 1.0, "t_end": 3.0,
             "transition": "ease_in_out", "axis": [0.0, 1.0, 0.0],
             "angular_velocity": 90.0},
            {"op_type": "SCALE", "t_start": 3.0, "t_end": 4.0,
             "transition": "linear", "factor": 0.5},
        ],
        "transform": eye4.copy(),
    }

    return [
        {"type": "SHELL", "object": [canonical_shell, duplicate_shell]},
        {"type": "ROD", "object": [rod]},
        {"type": "STATIC", "object": [static_with_anim, static_with_ops]},
    ]


def make_param_payload() -> dict:
    """Mirror blender_addon/core/encoder/params.py:_build_param_dict output.

    Uses np.float32 / np.array shapes the addon's _encode_scene_params
    actually emits, so the encoder's numpy fallback gets exercised.
    """
    scene = {
        "dt": np.float32(1e-3),
        "min-newton-steps": 0,
        "air-density": np.float32(1e-3),
        "air-friction": np.float32(0.2),
        "friction-mode": "min",
        "gravity": np.array([0.0, -9.8, 0.0], dtype=np.float64),
        "wind": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "frames": 59,
        "fps": 60,
        "csrmat-max-nnz": 10_000_000,
        "isotropic-air-friction": np.float32(0.0),
        "auto-save": 0,
        "line-search-max-t": np.float32(1.25),
        "constraint-ghat": np.float32(1e-3),
        "cg-max-iter": 10000,
        "cg-tol": np.float32(1e-3),
        "include-face-mass": False,
        "disable-contact": False,
        # SHELL group present, so inactive-momentum populated:
        "inactive-momentum": 0.5,
    }

    shell_group_params = {
        "model": "baraff-witkin",
        "density": 1.0,
        "young-mod": 1000.0,
        "poiss-rat": 0.35,
        "friction": 0.2,
        "contact-gap": 1e-3,
        "contact-offset": 0.0,
        "strain-limit": 0.0,
        "bend": 10.0,
        "shrink-x": 1.0,
        "shrink-y": 1.0,
        "pressure": 0.0,
        "plasticity": 0.0,
        "plasticity-threshold": 0.0,
        "bend-plasticity": 0.0,
        "bend-plasticity-threshold": 0.0,
        "bend-rest-from-geometry": 0.0,
        "velocity": {"uuid-shell-1": [0.0, 0.0, 0.0]},
        "velocity-schedule": {"uuid-shell-1": []},
        "collision-windows": {},
    }

    solid_group_params = {
        "model": "snhk",
        "density": 1000.0,
        "young-mod": 500.0,
        "poiss-rat": 0.4,
        "friction": 0.5,
        "contact-gap": 1e-3,
        "contact-offset": 0.0,
        "shrink": 1.0,
        "plasticity": 0.0,
        "plasticity-threshold": 0.0,
        "velocity": {"uuid-solid-1": [0.0, 0.0, 0.0]},
        "velocity-schedule": {"uuid-solid-1": []},
        "collision-windows": {"uuid-solid-1": [(0.5, 1.5), (2.0, 3.0)]},
        "ftetwild": {"epsilon": 0.001, "max_its": 100},
    }

    pin_config = {
        "uuid-shell-1": {
            0: {
                "unpin_time": 1.0,
                "pull_strength": 50.0,
                "pin_group_id": "uuid-shell-1:vg-a",
                "operations": [
                    {"type": "spin", "t_start": 0.0, "t_end": 2.0,
                     "transition": "linear", "center": [0.0, 0.0, 0.0],
                     "center_mode": "absolute", "axis": [0.0, 1.0, 0.0],
                     "angular_velocity": 360.0},
                    {"type": "move_by", "t_start": 2.0, "t_end": 3.0,
                     "transition": "linear", "delta": [0.5, 0.0, 0.0]},
                ],
                "pin_anim": {0: {"time": [0.0, 1.0],
                                 "position": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}},
            },
            3: {"operations": []},
        },
    }

    cross_stitch = [
        {"source_uuid": "uuid-shell-2", "target_uuid": "uuid-solid-1",
         # 6-wide barycentric-barycentric: degenerate shell source
         # [0, 0, 0] / [1, 0, 0], target bary over solid tri (1, 2, 3).
         "ind": [[0, 0, 0, 1, 2, 3]],
         "w": [[1.0, 0.0, 0.0, 0.1, 0.4, 0.3]],
         "source_points": [[0.0, 0.0, 0.0]],
         "target_points": [[1.0, 2.0, 3.0]],
         "stitch_stiffness": 0.75},
    ]

    invisible_colliders = {
        "walls": [
            {"position": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0],
             "contact_gap": 1e-3, "friction": 0.3, "active_duration": -1.0,
             "thickness": 1.0,
             "keyframes": [{"position": [0.0, 0.0, 0.0], "time": 0.0},
                           {"position": [0.0, 0.5, 0.0], "time": 1.0}]},
        ],
        "spheres": [
            {"position": [2.0, 1.0, 0.0], "radius": 0.5, "hemisphere": False,
             "invert": False, "contact_gap": 1e-3, "friction": 0.0,
             "active_duration": -1.0, "thickness": 1.0,
             "keyframes": [{"position": [2.0, 1.0, 0.0], "radius": 0.5, "time": 0.0}]},
        ],
    }

    return {
        "scene": scene,
        "group": [
            (shell_group_params, ["cloth_a"], ["uuid-shell-1"]),
            (solid_group_params, ["block"], ["uuid-solid-1"]),
        ],
        "pin_config": pin_config,
        "cross_stitch": cross_stitch,
        "invisible_colliders": invisible_colliders,
    }


def write(name: str, payload: bytes) -> None:
    path = OUT / name
    path.write_bytes(payload)
    print(f"  wrote {path.relative_to(Path.cwd())} ({len(payload)} bytes)")


def main() -> int:
    print("Generating CBOR fixtures via the addon encoder...")
    write("scene.cbor", dumps("Scene", make_scene_payload()))
    write("param.cbor", dumps("Param", make_param_payload()))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
