# File: frontend/tests/_decoder_solid_threshold_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for ``ParamDecoder._split_solid_holder_by_threshold``.
#
# A hard-intent partial-pin SOLID is split at decode time into a hard
# (FixPair) surface sub-holder and a soft (PullPair) sub-holder. The
# split runs UNCONDITIONALLY (interior fix pins crash the solver); the
# per-pin ``fix_weight_threshold`` (carried in cfg, default 0) only
# controls how much of the SURFACE is hard vs soft skirt.
#
# Load-bearing invariants exercised here:
#   * hard / soft partition over the FULL ``driven_full`` axis;
#   * hard promotion is SURFACE-ONLY (an interior fix pin would be a
#     zero-diagonal CG nan: the fix barrier is dispatched over surface
#     verts only in contact.cu and inertia is gated off for fix pins);
#   * the helper owns the pull calls (a hard-intent cfg has no
#     ``pull_strength`` so ``_apply_pin_cfg_entry`` would never pull);
#   * the gates (pull intent, torque op) and idempotency.

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
    from frontend._scene_pin_ import PinHolder
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


# Tet vertex indices on the full surf+interior axis, and the Blender pin
# indices that key the per-vertex cfg.
DF = [10, 11, 12, 13]
BL = [0, 1]


class _FakeDyn:
    """Minimal stand-in for a frontend ``Object`` exposing only what the
    split helper touches: ``pin_list``, ``pin()``, and ``name``."""

    def __init__(self):
        self._pin = []
        self.name = "t"

    @property
    def pin_list(self):
        return self._pin

    def pin(self, ind):
        holder = PinHolder(self, list(ind))
        self._pin.append(holder)
        return holder


def _make_original(dyn, full_w, driven_full, bl_indices, n_surf=None):
    """Mirror what ``SceneDecoder._apply_pin_mapping`` attaches to a
    partial-pin SOLID holder. ``n_surf`` = number of leading driven_full
    entries that are surface verts (``driven_full == surf_ids +
    interior_ids``); defaults to all-surface."""
    full_w = np.asarray(full_w, dtype=float)
    if n_surf is None:
        n_surf = len(driven_full)
    keep = full_w > 1e-4
    driven = [int(driven_full[k]) for k in range(len(driven_full)) if keep[k]]
    h = dyn.pin(driven)
    h._data._blender_pin_indices = list(bl_indices)
    h._data._tet_V = None
    h._data._blender_vert = None
    h._data._solid_pin_weights = full_w[keep].astype(np.float32)
    h._data._solid_pin = {"S_t": np.zeros((2, 3)), "M": None,
                          "keep": keep, "n_input": 3}
    h._data._solid_full_w = full_w
    h._data._solid_driven_full = list(driven_full)
    h._data._solid_surf_mask = np.arange(len(driven_full)) < n_surf
    return h


def _split(full_w, cfg, n_surf=None):
    dyn = _FakeDyn()
    _make_original(dyn, full_w, DF, BL, n_surf=n_surf)
    obj_cfg = {b: cfg for b in BL}
    ParamDecoder()._split_solid_holder_by_threshold(dyn, obj_cfg, verbose=False)
    return dyn


def _classify(dyn):
    hard = [h for h in dyn.pin_list if h._data.pull_strength == 0.0]
    soft = [h for h in dyn.pin_list if h._data.pull_strength == 1.0]
    return hard, soft


def _thr(value):
    return {"fix_weight_threshold": value}


def test_general_split():
    dyn = _split([0.9, 0.6, 0.3, 0.05], _thr(0.5))
    assert len(dyn.pin_list) == 2
    hard, soft = _classify(dyn)
    assert len(hard) == 1 and len(soft) == 1
    assert sorted(hard[0].index) == [10, 11]
    assert sorted(soft[0].index) == [12, 13]
    assert hard[0]._data.pull_weights is None
    np.testing.assert_allclose(
        sorted(soft[0]._data.pull_weights), sorted([0.3, 0.05]), rtol=1e-5)
    assert int(np.asarray(hard[0]._data._solid_pin["keep"]).sum()) == len(hard[0].index)
    assert int(np.asarray(soft[0]._data._solid_pin["keep"]).sum()) == len(soft[0].index)


def test_idempotent_resplit():
    dyn = _split([0.9, 0.6, 0.3, 0.05], _thr(0.5))
    assert len(dyn.pin_list) == 2
    # A second pass must not re-split the already-split holders.
    ParamDecoder()._split_solid_holder_by_threshold(
        dyn, {b: _thr(0.5) for b in BL}, verbose=False)
    assert len(dyn.pin_list) == 2


def test_threshold_zero_all_surface_is_all_hard():
    dyn = _split([0.9, 0.6, 0.3, 0.05], _thr(0.0))
    hard, soft = _classify(dyn)
    assert len(dyn.pin_list) == 1 and len(hard) == 1 and len(soft) == 0
    assert sorted(hard[0].index) == [10, 11, 12, 13]
    assert hard[0]._data.pull_weights is None


def test_threshold_one_is_all_soft():
    # Diffused weights never reach exactly 1.0, so thr=1.0 hardens nothing.
    dyn = _split([0.9, 0.6, 0.3, 0.05], _thr(1.0))
    hard, soft = _classify(dyn)
    assert len(dyn.pin_list) == 1 and len(soft) == 1 and len(hard) == 0
    assert sorted(soft[0].index) == [10, 11, 12, 13]
    assert soft[0]._data.pull_weights is not None
    assert len(soft[0]._data.pull_weights) == 4


def test_boundary_weight_equal_threshold_is_hard():
    dyn = _split([0.5, 0.4, 0.5, 0.2], _thr(0.5))
    hard, soft = _classify(dyn)
    assert sorted(hard[0].index) == [10, 12]   # the two 0.5s
    assert sorted(soft[0].index) == [11, 13]


def test_default_threshold_is_half():
    # No fix_weight_threshold key in cfg => default 0.5, i.e. the same split as
    # an explicit 0.5: high-weight surface hard, low-weight surface soft.
    dyn = _split([0.9, 0.6, 0.3, 0.05], {})
    hard, soft = _classify(dyn)
    assert len(hard) == 1 and len(soft) == 1
    assert sorted(hard[0].index) == [10, 11]
    assert sorted(soft[0].index) == [12, 13]


def test_legacy_hard_pin_no_threshold_splits_interior_soft():
    # The crash case: a plain hard partial-pin SOLID (no cfg keys, no pull)
    # must still keep interior verts soft so the solver never sees an interior
    # fix pin. n_surf=2 -> 10,11 surface (hard), 12,13 interior (soft).
    dyn = _split([0.9, 0.6, 0.3, 0.05], {}, n_surf=2)
    hard, soft = _classify(dyn)
    assert len(hard) == 1 and len(soft) == 1
    assert sorted(hard[0].index) == [10, 11]   # surface hard
    assert sorted(soft[0].index) == [12, 13]   # interior soft, never FixPair


def test_pull_intent_is_noop():
    # A pure-pull pin (pull_strength present) must never harden.
    cfg = {"fix_weight_threshold": 0.5, "pull_strength": 0.7}
    dyn = _split([0.9, 0.6, 0.3, 0.05], cfg)
    assert len(dyn.pin_list) == 1


def test_torque_op_is_noop():
    cfg = {"fix_weight_threshold": 0.5, "operations": [{"type": "torque"}]}
    dyn = _split([0.9, 0.6, 0.3, 0.05], cfg)
    assert len(dyn.pin_list) == 1


def test_interior_high_weight_stays_soft():
    # DF=[10,11,12,13], n_surf=2 -> 10,11 surface; 12,13 interior. All
    # weights high, but interior verts must NOT become FixPairs (the solver
    # assembles the fix barrier over surface verts only).
    dyn = _split([0.9, 0.6, 0.9, 0.9], _thr(0.5), n_surf=2)
    hard, soft = _classify(dyn)
    assert len(hard) == 1 and len(soft) == 1
    assert sorted(hard[0].index) == [10, 11]    # surface only
    assert sorted(soft[0].index) == [12, 13]    # interior -> soft
    np.testing.assert_allclose(
        sorted(soft[0]._data.pull_weights), sorted([0.9, 0.9]), rtol=1e-5)


def test_threshold_zero_with_interior_keeps_interior_soft():
    # thr=0 must still keep interior verts soft (no interior FixPair), i.e. it
    # is NOT an all-hard collapse when interior verts exist.
    dyn = _split([0.9, 0.6, 0.3, 0.05], _thr(0.0), n_surf=2)
    hard, soft = _classify(dyn)
    assert len(hard) == 1 and len(soft) == 1
    assert sorted(hard[0].index) == [10, 11]    # surface hard
    assert sorted(soft[0].index) == [12, 13]    # interior soft


# ---- Harmonic mixed-intent (full-pin) last-wins extraction ----------
# A fully-pinned SOLID with a soft PULL pin over every vert and a hard pin
# added LAST over a subset (e.g. "pin-root" on the bottom) decodes to ONE
# harmonic holder carrying ``_harmonic`` + ``_sim_blender_weights``.
# Per-vertex last-wins makes the subset hard (no pull_strength) while the
# rest stay pull. The split must (1) extract the hard SURFACE verts into a
# FixPair held at rest and (2) re-point the surviving (pull) holder's cfg
# lookup at a pull vert, so the per-holder loop still resolves pull + the
# captured move even when the holder's first stored blender vert is hard.

HARM_HARD = {"pin_group_id": "root"}                  # no pull_strength
HARM_PULL = {"pull_strength": 1000.0, "pin_group_id": "pin"}


def _make_harmonic_original(dyn, surf_sim, interior_sim, sim_weights,
                            bl_indices):
    """Mirror what ``_apply_pin_mapping`` attaches to a full-pin harmonic
    SOLID holder: sim indices = surf verts then interior, plus the
    ``_harmonic`` (n_surf, M) tuple and per-surf-vert ``_sim_blender_weights``
    corners. The split reads only ``harmonic[0]`` so ``M`` is left None."""
    h = dyn.pin(list(surf_sim) + list(interior_sim))
    h._data._blender_pin_indices = list(bl_indices)
    h._data._tet_V = None
    h._data._blender_vert = None
    h._data._harmonic = (len(surf_sim), None)
    h._data._sim_blender_weights = [list(c) for c in sim_weights]
    return h


def _harmonic_split(surf_sim, interior_sim, sim_weights, bl_indices, obj_cfg):
    dyn = _FakeDyn()
    _make_harmonic_original(dyn, surf_sim, interior_sim, sim_weights,
                            bl_indices)
    ParamDecoder()._split_solid_holder_by_threshold(dyn, obj_cfg, verbose=False)
    return dyn


def test_harmonic_mixed_extracts_hard_surface_fixpair():
    # 4 surf verts (sim 20..23) + 2 interior (24,25). Blender verts 0,1 are
    # HARD (root, listed FIRST), 2,3 are PULL (body). Surf 20,21 map to the
    # hard blender verts; 22,23 to the pull ones.
    surf, interior = [20, 21, 22, 23], [24, 25]
    sim_weights = [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(3, 1.0)]]
    obj_cfg = {0: HARM_HARD, 1: HARM_HARD, 2: HARM_PULL, 3: HARM_PULL}
    dyn = _harmonic_split(surf, interior, sim_weights, [0, 1, 2, 3], obj_cfg)

    assert len(dyn.pin_list) == 2
    full = [h for h in dyn.pin_list if len(h.index) == len(surf) + len(interior)]
    fix = [h for h in dyn.pin_list if len(h.index) == 2]
    assert len(full) == 1 and len(fix) == 1
    full, fix = full[0], fix[0]

    # Hard SURFACE verts extracted into a FixPair (default pull_strength 0,
    # so the per-holder loop's hard cfg leaves it a stationary FixPair).
    assert sorted(fix.index) == [20, 21]
    assert fix._data.pull_strength == 0.0
    assert sorted(fix._data._blender_pin_indices) == [0, 1]
    assert getattr(fix._data, "_solid_split_done", False)

    # The surviving holder keeps every vert and is re-pointed at the PULL
    # blender verts so the per-holder loop resolves pull + the captured move,
    # even though its FIRST original blender vert (0) was a hard/root vert.
    assert sorted(full.index) == [20, 21, 22, 23, 24, 25]
    assert sorted(full._data._blender_pin_indices) == [2, 3]
    assert getattr(full._data, "_solid_split_done", False)

    # cfg-resolution mirror (apply_pin_config's first-cfg-vert break):
    # survivor -> pull cfg; FixPair -> hard cfg.
    survivor_cfg = next(obj_cfg[v] for v in full._data._blender_pin_indices
                        if v in obj_cfg)
    fix_cfg = next(obj_cfg[v] for v in fix._data._blender_pin_indices
                   if v in obj_cfg)
    assert "pull_strength" in survivor_cfg
    assert "pull_strength" not in fix_cfg


def test_harmonic_all_pull_is_noop():
    # No hard verts among the surface: nothing to extract, the single
    # harmonic holder stays (and is marked done so it is not re-split).
    surf, interior = [20, 21, 22, 23], [24, 25]
    sim_weights = [[(2, 1.0)], [(2, 1.0)], [(3, 1.0)], [(3, 1.0)]]
    obj_cfg = {2: HARM_PULL, 3: HARM_PULL}
    dyn = _harmonic_split(surf, interior, sim_weights, [2, 3], obj_cfg)
    assert len(dyn.pin_list) == 1
    assert getattr(dyn.pin_list[0]._data, "_solid_split_done", False)


def test_harmonic_split_idempotent():
    surf, interior = [20, 21, 22, 23], [24, 25]
    sim_weights = [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(3, 1.0)]]
    obj_cfg = {0: HARM_HARD, 1: HARM_HARD, 2: HARM_PULL, 3: HARM_PULL}
    dyn = _harmonic_split(surf, interior, sim_weights, [0, 1, 2, 3], obj_cfg)
    assert len(dyn.pin_list) == 2
    ParamDecoder()._split_solid_holder_by_threshold(dyn, obj_cfg, verbose=False)
    assert len(dyn.pin_list) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
