# File: frontend/tests/_pin_spin_axis_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A spin turns its vertices about an axis the rotation normalizes, so a
# zero-length axis leaves it nothing to turn about: the solver's formula then
# scales every vertex's offset from the center by the cosine of the swept
# angle, contracting the pin through its center instead of turning it. A spin
# that moves at all is therefore refused without an axis, on a notebook pin
# and on the pin shell the decoder builds for a STATIC collider's Spin
# operation alike.
#
# Covers:
#   * `PinHolder.spin` refuses a zero-length axis with a nonzero angular
#     velocity, and records nothing;
#   * it accepts a zero-length axis at zero angular velocity, which moves
#     nothing, and any axis with length;
#   * a decoded STATIC Spin operation with a zero-length axis is refused while
#     its pin shell is built, and one with an axis is kept.

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401
    from frontend._asset_ import AssetManager
    from frontend._decoder_ import SceneDecoder
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
    from frontend._scene_pin_ import SpinOperation
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_pin_spin_axis"


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
        V, F = mesh.square(res=4)
        asset.add.tri("sheet", V, F)
        yield Scene("spin_axis", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_spin_without_an_axis_is_refused(scene):
    holder = scene.add("sheet").pin([0, 1, 2])
    with pytest.raises(ValueError, match="has no length"):
        holder.spin(axis=[0.0, 0.0, 0.0], angular_velocity=90.0)
    assert holder.operations == []


def test_a_spin_that_does_not_move_needs_no_axis(scene):
    holder = scene.add("sheet").pin([0, 1, 2])
    holder.spin(axis=[0.0, 0.0, 0.0], angular_velocity=0.0)
    holder.spin(axis=[0.0, 0.0, 2.0], angular_velocity=90.0)
    assert [type(op) for op in holder.operations] == [SpinOperation] * 2


def _static_spin_payload(axis):
    return {
        "static_ops": [{
            "op_type": "SPIN",
            "frame_offset_start": 0.0,
            "frame_offset_end": 10.0,
            "transition": "linear",
            "axis": axis,
            "angular_velocity_anim": 90.0,
        }],
    }


def _populate_static(scene, obj):
    decoder = SceneDecoder.__new__(SceneDecoder)
    decoder._asset = scene.asset_manager
    decoder._object_info = {}
    mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
    V, F = mesh.box(0.5, 0.5, 0.5)
    V = np.ascontiguousarray(V, dtype=np.float32)
    return decoder._populate_static(
        scene, obj, "spinner", "spinner-uuid", V,
        np.ascontiguousarray(F, dtype=np.uint32), np.eye(4), V,
        solver_fps=100.0, time_scale=1.0,
    )


def test_a_decoded_static_spin_without_an_axis_is_refused(scene):
    with pytest.raises(ValueError, match="has no length"):
        _populate_static(scene, _static_spin_payload([0.0, 0.0, 0.0]))


def test_a_decoded_static_spin_with_an_axis_is_kept(scene):
    spinner = _populate_static(scene, _static_spin_payload([0.0, 1.0, 0.0]))
    [pin] = spinner.pin_list
    [op] = pin.operations
    assert isinstance(op, SpinOperation)
    assert op.angular_velocity == pytest.approx(90.0)
