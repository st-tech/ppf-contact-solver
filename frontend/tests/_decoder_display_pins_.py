# File: frontend/tests/_decoder_display_pins_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for ``ParamDecoder._build_display_pins``.
#
# A SOLID's pinned Blender vertices are not simulation vertices, so the
# decoder records, for every exact pin group, a holder over the group's own
# Blender vertices carrying its script. The solver evaluates that script at
# every output frame and the addon places those vertices exactly.
#
# Load-bearing invariants exercised here:
#   * exact groups become display pins; pull and torque groups do not;
#   * rest positions are the Blender vertices in the object's untranslated
#     frame (the translation comes back through the displacement group);
#   * each holder carries its group's operations, captured per-vertex tracks
#     included, over its own Blender vertices;
#   * an object with no surface-mapped SOLID pin gets no display pin.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401  (PinHolder builds a Rust mirror)
    from frontend._decoder_ import ParamDecoder
    from frontend._scene_pin_ import MoveByOperation, PinHolder, SpinOperation
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


TRANSLATION = np.array([10.0, 0.0, 0.0])
# World-space Blender vertices (object translation included).
BLENDER_WORLD = np.array([
    [11.0, 0.0, 0.0],
    [11.0, 1.0, 0.0],
    [9.0, 0.0, 0.0],
    [9.0, 1.0, 0.0],
    [10.0, 0.0, 1.0],
])

SPIN = {
    "type": "spin",
    "center_mode": "absolute",
    "center": [1.0, 0.0, 0.0],
    "axis": [1.0, 0.0, 0.0],
    "angular_velocity": 90.0,
    "t_start": 0.0,
    "t_end": 1.0,
}


class _FakeDyn:
    """Stand-in for a frontend ``Object``: ``pin_list``, ``pin()``, ``name``
    and the translation ``position``."""

    def __init__(self):
        self._pin = []
        self.name = "t"
        self.position = TRANSLATION.tolist()

    @property
    def pin_list(self):
        return self._pin

    def pin(self, ind):
        holder = PinHolder(self, list(ind))
        self._pin.append(holder)
        return holder


def _solid_dyn():
    dyn = _FakeDyn()
    holder = dyn.pin([100, 101, 102])
    holder._data._blender_pin_indices = [0, 1, 2, 3, 4]
    holder._data._blender_vert = BLENDER_WORLD.copy()
    return dyn


def _build(dyn, obj_cfg):
    ParamDecoder()._build_display_pins(dyn, "t", obj_cfg, False)
    return dyn._display_pins


def test_exact_groups_become_display_pins_and_others_do_not():
    exact = {"pin_group_id": "L", "operations": [SPIN]}
    pull = {"pin_group_id": "P", "pull_strength": 5.0, "operations": [SPIN]}
    torque = {"pin_group_id": "T", "operations": [
        {"type": "torque", "magnitude": 1.0, "axis_component": 2,
         "hint_vertex": 4, "t_start": 0.0, "t_end": 1.0},
    ]}
    obj_cfg = {1: exact, 0: exact, 2: pull, 3: pull, 4: torque}
    display = _build(_solid_dyn(), obj_cfg)

    assert len(display) == 1
    block = display[0]
    np.testing.assert_array_equal(block["blender_index"], [0, 1])
    np.testing.assert_allclose(block["rest"], BLENDER_WORLD[[0, 1]] - TRANSLATION)
    data = block["holder"]._data
    assert data.pull_strength == 0.0
    assert data.index == [0, 1]
    assert len(data.operations) == 1
    assert isinstance(data.operations[0], SpinOperation)
    np.testing.assert_allclose(data.operations[0].axis, [1.0, 0.0, 0.0])


def test_captured_tracks_become_per_vertex_moves():
    obj_cfg = {}
    for b, dx in ((0, 1.0), (1, 2.0)):
        track = {"time": [0.0, 1.0],
                 "position": np.array([[0.0, 0.0, 0.0], [dx, 0.0, 0.0]])}
        obj_cfg[b] = {"pin_group_id": "C", "embedded_move_index": 0,
                      "pin_anim": {b: track}}
    display = _build(_solid_dyn(), obj_cfg)

    assert len(display) == 1
    ops = display[0]["holder"]._data.operations
    assert len(ops) == 1 and isinstance(ops[0], MoveByOperation)
    delta = np.asarray(ops[0].delta)
    np.testing.assert_allclose(delta, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_an_object_without_a_surface_mapped_pin_gets_none():
    dyn = _FakeDyn()
    dyn.pin([0, 1])
    display = _build(dyn, {0: {"pin_group_id": "L", "operations": [SPIN]}})
    assert display == []


def test_a_pinned_vertex_outside_the_mesh_is_refused():
    dyn = _solid_dyn()
    with pytest.raises(ValueError, match="outside"):
        _build(dyn, {7: {"pin_group_id": "L", "operations": [SPIN]}})
