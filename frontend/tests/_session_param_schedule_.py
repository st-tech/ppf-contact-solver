# File: frontend/tests/_session_param_schedule_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The dynamic-parameter schedule a session writes for the solver, which reads
# every window that covers a time and lets the last one win, so the order of a
# key's entries IS its meaning.
#
# Covers:
#   * a second `ParamManager.dyn` on a key continues that key's schedule, so a
#     time before what is already scheduled is refused rather than appended
#     out of order;
#   * the decoder turns the add-on's Inactive Momentum duration into exactly
#     "on from 0, off at T" and nothing else.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401
    from frontend._decoder_ import ParamDecoder
    from frontend._session_param_ import ParamManager
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )


def test_a_second_dyn_continues_the_schedule():
    param = ParamManager()
    g = param.get("gravity")
    param.dyn("gravity").time(1.0).change([0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        param.dyn("gravity").time(0.5)
    param.dyn("gravity").time(2.0).change(g)
    times = [t for t, _v in param._dyn_param["gravity"]]
    assert times == sorted(times) == [0.0, 1.0, 2.0]


def test_another_key_starts_at_zero():
    param = ParamManager()
    param.dyn("gravity").time(3.0).change([0.0, 0.0, 0.0])
    param.dyn("wind").time(0.5).change([1.0, 0.0, 0.0])
    assert [t for t, _v in param._dyn_param["wind"]] == [0.0, 0.5]


def test_inactive_momentum_is_on_then_off_at_its_duration():
    decoder = ParamDecoder()
    decoder._data = {"scene": {"inactive-momentum": 0.5}, "group": []}
    session = SimpleNamespace(param=ParamManager())
    decoder.apply_to_session(session)
    assert session.param._dyn_param["inactive-momentum"] == [
        (0.0, True),
        (0.5, True),
        (0.5, False),
    ]
