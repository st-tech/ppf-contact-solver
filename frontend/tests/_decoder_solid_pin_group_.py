# File: frontend/tests/_decoder_solid_pin_group_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for how a SOLID's pin holders are resolved to pin groups.
#
# A SOLID's pins reach ``ParamDecoder.apply_pin_config`` as ONE surface-mapped
# holder spanning every pinned Blender vertex of the object, whatever pin
# group each vertex belongs to. Every per-pin setting (operations, pull
# strength, unpin time, group id, fix weight threshold) must still reach
# exactly the vertices of its own group (discussion #148: two spin pins on one
# SOLID turned the same way). A driven vertex belongs to the group that
# contributes most of its pin weight.
#
# Load-bearing invariants exercised here:
#   * partial pin: owners come from the diffused weight field applied per
#     group, and every per-vertex array (keep mask, pull weights) is sliced
#     with the holder index;
#   * full pin: owners come from the corner weights and their harmonic
#     extension, and captured motion is still built for exactly the holder's
#     own vertices;
#   * surface-only pin: corner lists are sliced with the holder index;
#   * a holder resolving to one group is left as the intent split built it;
#   * the intent split runs unless EVERY group pulls, and reads the fix weight
#     threshold of the group owning each vertex;
#   * each group's operations reach only its own vertices.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import scipy.sparse as sp

    from frontend import _rust  # noqa: F401  (PinHolder builds a Rust mirror)
    from frontend._decoder_ import ParamDecoder, _SparseLinearMap
    from frontend._scene_pin_ import PinHolder
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py / scipy not importable in this environment: {exc}",
        allow_module_level=True,
    )


class _FakeDyn:
    """Minimal stand-in for a frontend ``Object``: ``pin_list``, ``pin()``,
    and ``name``."""

    def __init__(self):
        self._pin = []
        self.name = "t"
        self.position = [0.0, 0.0, 0.0]

    @property
    def pin_list(self):
        return self._pin

    def pin(self, ind):
        holder = PinHolder(self, list(ind))
        self._pin.append(holder)
        return holder


def _linear_map(rows):
    """A ``_SparseLinearMap`` that applies ``rows`` exactly (identity system),
    so a test states what each output takes from each input."""
    rhs = sp.csr_matrix(np.asarray(rows, dtype=np.float64))
    return _SparseLinearMap(sp.eye(rhs.shape[0], format="csc"), rhs)


SPIN_POS_X = {
    "type": "spin",
    "center_mode": "absolute",
    "center": [0.0, 0.0, 0.0],
    "axis": [1.0, 0.0, 0.0],
    "angular_velocity": 90.0,
    "t_start": 0.0,
    "t_end": 1.0,
}
SPIN_NEG_X = {**SPIN_POS_X, "axis": [-1.0, 0.0, 0.0]}


def _cfg(group, **extra):
    return {"pin_group_id": group, "fix_weight_threshold": 0.5, **extra}


def _two_spin_groups():
    # Blender 0, 1 in "L" spinning about +X; 2, 3 in "R" spinning about -X.
    left = _cfg("L", operations=[SPIN_POS_X])
    right = _cfg("R", operations=[SPIN_NEG_X])
    return {0: left, 1: left, 2: right, 3: right}


# Partial-pin holder: surface tet vertices 10..13 each take the weight of
# Blender vertex 0..3; interior 14 averages 10, 11 and interior 15 averages
# 12, 13.
DRIVEN_FULL = [10, 11, 12, 13, 14, 15]
N_SURF = 4
# World-space Blender vertices every holder here maps from (indices 0..4).
BLENDER_REST = np.zeros((5, 3))


def _make_partial(dyn, full_w=(0.9, 0.9, 0.9, 0.9, 0.6, 0.6)):
    """Mirror what ``SceneDecoder._apply_pin_mapping`` attaches to a
    partial-pin SOLID holder."""
    full_w = np.asarray(full_w, dtype=np.float64)
    keep = full_w > 1e-4
    holder = dyn.pin([DRIVEN_FULL[k] for k in range(len(DRIVEN_FULL)) if keep[k]])
    d = holder._data
    d._blender_pin_indices = [0, 1, 2, 3]
    d._tet_V = None
    d._blender_vert = BLENDER_REST
    d._solid_pin_weights = full_w[keep].astype(np.float32)
    d._solid_pin = {
        "surface_map": None,
        "interior_map": _linear_map([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]),
        "motion_cache": {},
        "keep": keep,
        "n_input": 4,
    }
    d._solid_weight_map = _linear_map(np.eye(N_SURF))
    d._solid_full_w = full_w
    d._solid_driven_full = list(DRIVEN_FULL)
    d._solid_surf_mask = np.arange(len(DRIVEN_FULL)) < N_SURF
    d._sim_blender_weights = [[(b, 1.0)] for b in range(N_SURF)]
    return holder


def _resolve(dyn, obj_cfg):
    decoder = ParamDecoder()
    groups = decoder._solid_pin_groups(dyn, obj_cfg)
    decoder._split_solid_holder_by_threshold(dyn, obj_cfg, False, groups)
    decoder._split_solid_holders_by_pin_group(dyn, groups, False)
    return dyn


def _holders_by_index(dyn):
    return {tuple(sorted(h.index)): h for h in dyn.pin_list}


def _groups_of(holder, obj_cfg):
    return {obj_cfg[b]["pin_group_id"] for b in holder._data._blender_pin_indices}


def test_partial_pin_groups_are_partitioned_by_owner():
    dyn = _FakeDyn()
    _make_partial(dyn)
    obj_cfg = _two_spin_groups()
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(10, 11), (12, 13), (14,), (15,)]
    for index, group in (((10, 11), "L"), ((12, 13), "R"),
                         ((14,), "L"), ((15,), "R")):
        assert _groups_of(holders[index], obj_cfg) == {group}
    # The surface stays an exact pin and the interior a soft pull, per group.
    assert holders[(10, 11)]._data.pull_strength == 0.0
    assert holders[(12, 13)]._data.pull_strength == 0.0
    for index, axis in (((14,), 4), ((15,), 5)):
        h = holders[index]
        assert h._data.pull_strength == 1.0
        np.testing.assert_allclose(h._data.pull_weights, [0.6])
        np.testing.assert_allclose(h._data._solid_pin_weights, [0.6])
        np.testing.assert_array_equal(
            np.flatnonzero(h._data._solid_pin["keep"]), [axis])


def test_apply_pin_config_gives_each_group_its_own_spin():
    dyn = _FakeDyn()
    _make_partial(dyn)
    decoder = ParamDecoder()
    decoder._pin_config = {"t": _two_spin_groups()}
    decoder.apply_pin_config(SimpleNamespace(object_dict={"t": dyn}))

    axis_x = {}
    group = {}
    for h in dyn.pin_list:
        assert len(h.operations) == 1
        for v in h.index:
            axis_x[v] = h.operations[0].axis[0]
            group[v] = h._data.pin_group_id
    assert axis_x == {10: 1.0, 11: 1.0, 14: 1.0, 12: -1.0, 13: -1.0, 15: -1.0}
    assert group == {10: "L", 11: "L", 14: "L", 12: "R", 13: "R", 15: "R"}
    # Each exact group also becomes a display pin over its own Blender
    # vertices, carrying its own spin.
    display = {
        tuple(d["blender_index"].tolist()): d["holder"]._data.operations[0].axis[0]
        for d in dyn._display_pins
    }
    assert display == {(0, 1): 1.0, (2, 3): -1.0}


def test_single_group_holder_is_left_as_the_intent_split_built_it():
    dyn = _FakeDyn()
    holder = _make_partial(dyn)
    # One group needs no owner weights at all.
    del holder._data._solid_weight_map
    left = _cfg("L", operations=[SPIN_POS_X])
    _resolve(dyn, {b: left for b in range(4)})

    assert sorted(_holders_by_index(dyn)) == [(10, 11, 12, 13), (14, 15)]


def test_two_groups_without_a_weight_map_raise():
    dyn = _FakeDyn()
    holder = _make_partial(dyn)
    del holder._data._solid_weight_map
    with pytest.raises(RuntimeError, match="weight map"):
        ParamDecoder()._solid_pin_groups(dyn, _two_spin_groups())


def test_intent_split_runs_when_the_first_vertex_pulls():
    # Blender 0, 1 pull ("P", listed first); 2, 3 are exact with a spin ("H").
    dyn = _FakeDyn()
    _make_partial(dyn)
    pull = _cfg("P", pull_strength=10.0)
    hard = _cfg("H", operations=[SPIN_NEG_X])
    obj_cfg = {0: pull, 1: pull, 2: hard, 3: hard}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(10, 11, 14, 15), (12, 13)]
    exact = holders[(12, 13)]
    assert exact._data.pull_strength == 0.0
    assert _groups_of(exact, obj_cfg) == {"H"}
    soft = holders[(10, 11, 14, 15)]
    assert soft._data.pull_strength == 1.0
    assert _groups_of(soft, obj_cfg) == {"P"}


def test_all_pull_groups_skip_the_intent_split_and_are_partitioned():
    dyn = _FakeDyn()
    _make_partial(dyn)
    left = _cfg("L", pull_strength=10.0, operations=[SPIN_POS_X])
    right = _cfg("R", pull_strength=20.0, operations=[SPIN_NEG_X])
    obj_cfg = {0: left, 1: left, 2: right, 3: right}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(10, 11, 14), (12, 13, 15)]
    np.testing.assert_allclose(
        holders[(10, 11, 14)]._data._solid_pin_weights, [0.9, 0.9, 0.6])
    np.testing.assert_array_equal(
        np.flatnonzero(holders[(12, 13, 15)]._data._solid_pin["keep"]),
        [2, 3, 5])


def test_fix_weight_threshold_is_read_from_the_owning_group():
    # Both groups follow a captured track with no pull. "L" hardens every
    # surface vertex (threshold 0); "R" asks for more weight than its
    # surface carries (threshold 0.95), so its surface stays soft.
    dyn = _FakeDyn()
    holder = _make_partial(dyn)
    holder._data._solid_frame_map = {
        "triangles": np.array([[0, 1, 0], [0, 1, 0], [2, 3, 2], [2, 3, 2]],
                              dtype=np.int64),
        "coefs": np.zeros((N_SURF, 3), dtype=np.float64),
    }
    track = {"time": [0.0, 1.0], "position": np.zeros((2, 3))}
    obj_cfg = {}
    for b, (group, thr) in enumerate((("L", 0.0), ("L", 0.0),
                                      ("R", 0.95), ("R", 0.95))):
        obj_cfg[b] = {**_cfg(group), "fix_weight_threshold": thr,
                      "embedded_move_index": 0, "pin_anim": {b: track}}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert holders[(10, 11)]._data.pull_strength == 0.0
    assert _groups_of(holders[(10, 11)], obj_cfg) == {"L"}
    assert not any(set(h.index) & {12, 13} and h._data.pull_strength == 0.0
                   for h in dyn.pin_list)


def _make_full(dyn):
    """Mirror what ``SceneDecoder._apply_pin_mapping`` attaches to a full-pin
    SOLID holder: surface tet vertices 20..23 each take Blender vertex 0..3,
    interior 24 averages 20, 21 and interior 25 averages 22, 23."""
    surf, interior = [20, 21, 22, 23], [24, 25]
    holder = dyn.pin(surf + interior)
    d = holder._data
    d._blender_pin_indices = list(range(len(surf)))
    d._tet_V = None
    d._blender_vert = BLENDER_REST
    d._sim_blender_weights = [[(b, 1.0)] for b in range(len(surf))]
    d._harmonic = (
        len(surf),
        _linear_map([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]),
    )
    return holder


def test_full_pin_groups_partition_and_keep_their_captured_motion():
    dyn = _FakeDyn()
    _make_full(dyn)
    obj_cfg = {}
    for b, (group, dx) in enumerate((("L", 1.0), ("L", 1.0),
                                     ("R", -1.0), ("R", -1.0))):
        track = {"time": [0.0, 1.0],
                 "position": np.array([[0.0, 0.0, 0.0], [dx, 0.0, 0.0]])}
        obj_cfg[b] = {**_cfg(group), "embedded_move_index": 0,
                      "pin_anim": {b: track}}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(20, 21, 24), (22, 23, 25)]
    for index, dx in (((20, 21, 24), 1.0), ((22, 23, 25), -1.0)):
        ops = ParamDecoder._build_embedded_move_ops(holders[index], obj_cfg)
        assert len(ops) == 1
        delta = np.asarray(ops[0].delta)
        assert delta.shape == (3, 3)
        np.testing.assert_allclose(delta[:, 0], dx)
        np.testing.assert_allclose(delta[:, 1:], 0.0)


def test_full_pin_mixed_intent_extraction_is_not_partitioned_further():
    # A pull pin over most of the surface with a hard pin added last over the
    # rest resolves to one group per holder in the intent split already.
    dyn = _FakeDyn()
    _make_full(dyn)
    hard = {"pin_group_id": "root"}
    pull = {"pull_strength": 1000.0, "pin_group_id": "pin"}
    obj_cfg = {0: hard, 1: hard, 2: pull, 3: pull}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(20, 21), (20, 21, 22, 23, 24, 25)]
    assert _groups_of(holders[(20, 21)], obj_cfg) == {"root"}
    assert _groups_of(holders[(20, 21, 22, 23, 24, 25)], obj_cfg) == {"pin"}


def test_surface_only_pin_slices_its_corner_lists():
    dyn = _FakeDyn()
    holder = dyn.pin([30, 31, 32])
    d = holder._data
    d._blender_pin_indices = [0, 2]
    d._tet_V = None
    d._blender_vert = BLENDER_REST
    d._sim_blender_weights = [[(0, 1.0)], [(0, 0.4), (2, 0.6)], [(2, 1.0)]]
    obj_cfg = {0: _cfg("L", operations=[SPIN_POS_X]),
               2: _cfg("R", operations=[SPIN_NEG_X])}
    _resolve(dyn, obj_cfg)

    holders = _holders_by_index(dyn)
    assert sorted(holders) == [(30,), (31, 32)]
    assert holders[(30,)]._data._sim_blender_weights == [[(0, 1.0)]]
    assert holders[(31, 32)]._data._sim_blender_weights == [
        [(0, 0.4), (2, 0.6)], [(2, 1.0)]]
