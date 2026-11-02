# File: addon_host_tests/_step_size_limit_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Host-side gates for ``blender_addon/core/limits.py``.
#
# The add-on caps its Step Size field and refuses a larger value at encode
# with a COPY of the solver's limit, because it never imports the frontend.
# The original is ``MAX_STEP_SIZE`` in ``crates/ppf-cts-core/src/datamodel/
# params.rs``; this reads it from the source, so no build is needed, and holds
# the copy equal to it.

from __future__ import annotations

import math
import re

from conftest import REPO_ROOT, load_addon_module


def _core_limit() -> float:
    source = (REPO_ROOT / "crates" / "ppf-cts-core" / "src" / "datamodel" / "params.rs").read_text()
    found = re.findall(r"pub const MAX_STEP_SIZE: f64 = ([0-9.eE+-]+);", source)
    assert len(found) == 1, f"params.rs declares MAX_STEP_SIZE {len(found)} times"
    return float(found[0])


def test_the_addon_limit_is_the_solver_limit():
    limits = load_addon_module("core.limits")
    assert limits.MAX_STEP_SIZE == _core_limit()


def test_a_step_past_the_limit_is_refused_and_one_at_it_is_not():
    limits = load_addon_module("core.limits")
    assert limits.step_size_refusal(limits.MAX_STEP_SIZE) is None
    assert limits.step_size_refusal(0.001) is None
    for bad in (limits.MAX_STEP_SIZE * 1.5, 0.1, 0.0, -0.01, math.nan):
        refusal = limits.step_size_refusal(bad)
        assert refusal is not None, f"step size {bad} was accepted"
        assert f"{limits.MAX_STEP_SIZE:g}" in refusal
