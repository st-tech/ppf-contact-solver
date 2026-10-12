# File: frontend/tests/_decoder_weighted_pull_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for the weighted pull of ``ParamDecoder.apply_pin_config``.
#
# A weighted pin ships each held vertex's own ``pull_weight`` beside the
# group's ``pull_strength``, and the decoder pulls every vertex with the
# strength times its weight. A vertex with no cfg at all is one a plain pin
# holds exactly: the encoder lets the later pin in the list win a vertex two
# pins share, so a plain pin listed after a pulling one takes its vertices
# out of that pin's cfg.
#
# Load-bearing invariants exercised here:
#   * the per-vertex pull is strength times weight, aligned to the holder's
#     own vertices, and the scalar pull stays as the soft marker;
#   * a pulling pin with no weights is left on its scalar;
#   * vertices a plain pin took back form their own exact holder, and only
#     the rest are pulled;
#   * a strength of zero is an exact hold and takes no per-vertex pull;
#   * a weight carried by some vertices of a group and not others, a weight
#     outside (0, 1] and a product that underflows are each refused by name.

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

NAME = "hair-uuid"
GROUP = f"{NAME}:Goal"


class _FakeDyn:
    """Stand-in for a frontend ``Object``: ``pin_list``, ``pin()``, ``name``
    and ``position``."""

    def __init__(self):
        self._pin = []
        self.name = "hair"
        self.position = [0.0, 0.0, 0.0]

    @property
    def pin_list(self):
        return self._pin

    def pin(self, ind):
        holder = PinHolder(self, list(ind))
        self._pin.append(holder)
        return holder


class _FakeScene:
    def __init__(self, dyn):
        self.object_dict = {NAME: dyn}


def _decode(pinned, obj_cfg):
    """Apply `obj_cfg` to an object whose single shipped holder spans
    `pinned`, as ``_apply_pin_mapping`` leaves it, and return its holders."""
    dyn = _FakeDyn()
    dyn.pin(pinned)
    decoder = ParamDecoder()
    decoder._pin_config = {NAME: obj_cfg}
    decoder.apply_pin_config(_FakeScene(dyn))
    return dyn.pin_list


def _goal(strength, weights):
    return {
        vi: {"pin_group_id": GROUP, "pull_strength": strength, "pull_weight": w}
        for vi, w in weights.items()
    }


def test_each_vertex_is_pulled_with_the_strength_times_its_weight():
    weights = {4: 1.0, 5: 0.75, 6: 0.5, 7: 0.125}
    (holder,) = _decode([4, 5, 6, 7], _goal(20.0, weights))
    assert holder.index == [4, 5, 6, 7]
    assert holder.pull_strength == 20.0
    np.testing.assert_array_equal(
        holder.pull_weights, np.float32([20.0, 15.0, 10.0, 2.5])
    )
    assert holder.pull_weights.dtype == np.float32


def test_the_weights_follow_the_holders_own_vertex_order():
    weights = {7: 0.125, 4: 1.0, 6: 0.5, 5: 0.75}
    (holder,) = _decode([6, 4, 7, 5], _goal(8.0, weights))
    assert holder.index == [6, 4, 7, 5]
    np.testing.assert_array_equal(
        holder.pull_weights, np.float32([4.0, 8.0, 1.0, 6.0])
    )


def test_a_pulling_pin_with_no_weights_stays_on_its_scalar():
    cfg = {"pin_group_id": GROUP, "pull_strength": 3.0}
    (holder,) = _decode([0, 1, 2], {0: cfg, 1: cfg, 2: cfg})
    assert holder.pull_strength == 3.0
    assert holder.pull_weights is None


def test_vertices_a_plain_pin_took_back_are_held_exactly():
    # The roots, 0 and 1, carry no cfg: a plain pin listed after the weighted
    # one took them. The shipped list names them twice, once per pin.
    weights = {2: 0.75, 3: 0.5, 4: 0.25}
    holders = _decode([0, 1, 2, 3, 4, 0, 1], _goal(10.0, weights))
    by_group = {h.pin_group_id: h for h in holders}
    assert len(holders) == 2
    exact = next(h for h in holders if h.pin_group_id != GROUP)
    assert sorted(set(exact.index)) == [0, 1]
    assert exact.pull_strength == 0.0 and exact.pull_weights is None
    pulled = by_group[GROUP]
    assert pulled.index == [2, 3, 4]
    np.testing.assert_array_equal(pulled.pull_weights, np.float32([7.5, 5.0, 2.5]))


def test_a_strength_of_zero_is_an_exact_hold_and_takes_no_weights():
    (holder,) = _decode([0, 1], _goal(0.0, {0: 1.0, 1: 0.5}))
    assert holder.pull_strength == 0.0
    assert holder.pull_weights is None


def test_a_weight_on_only_some_vertices_of_a_group_is_refused():
    obj_cfg = _goal(5.0, {0: 1.0, 1: 0.5})
    obj_cfg[2] = {"pin_group_id": GROUP, "pull_strength": 5.0}
    with pytest.raises(ValueError, match="none for vertex 2"):
        _decode([0, 1, 2], obj_cfg)


@pytest.mark.parametrize("bad", [0.0, -0.25, 1.5, float("nan"), float("inf")])
def test_a_weight_outside_the_interval_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="pin 'Goal' on 'hair-uuid'.*vertex 1"):
        _decode([0, 1], _goal(5.0, {0: 1.0, 1: bad}))


def test_a_product_that_underflows_to_an_exact_hold_is_refused():
    # 1e-3 * 1e-44 is below the smallest single-precision number, and a pull
    # of exactly zero is how the solver spells a fixed vertex.
    with pytest.raises(ValueError, match="too small to export"):
        _decode([0, 1], _goal(1e-3, {0: 1.0, 1: 1e-44}))
