# File: frontend/tests/_lock_translation_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for ``Object.lock_translation()``.
#
# Lock Translation constrains an object's mass-weighted center of mass
# to the fixed world-space line through its initial position, along a
# caller-given axis; rotation and deformation stay free. This is a
# frontend/data-pipeline-only feature today: the solver-side constraint
# is future work, so these tests cover only the state/validation the
# frontend owns:
#   * the axis is normalized (only direction matters, not magnitude);
#   * a non-finite or zero-norm axis is rejected loudly rather than
#     being replaced with a plausible default;
#   * a static object cannot be locked (it has no free motion to
#     constrain in the first place);
#   * the object starts unlocked (``_translation_lock is None``), and
#     there is no "unlock" call: the disabled state is simply "never
#     called ``lock_translation()``".

from __future__ import annotations

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
    from frontend._scene_object_ import Object
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build` or "
        "`cargo build-emul` first",
        allow_module_level=True,
    )


def _make_tri_object(name: str = "sheet") -> Object:
    """Build a minimal standalone triangle Object for unit testing.

    Bypasses ``Scene``/``ObjectAdder`` (which need a full app/session)
    since ``lock_translation()`` only touches asset-independent state.
    """
    asset = AssetManager()
    V = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    F = np.array([[0, 1, 2]], dtype=np.int64)
    asset.add.tri(name, V, F)
    return Object(asset, name)


def test_starts_unlocked():
    obj = _make_tri_object()
    assert obj._translation_lock is None


def test_normalizes_axis():
    obj = _make_tri_object()
    obj.lock_translation(2.0, 0.0, 0.0)
    assert obj._translation_lock is not None
    np.testing.assert_allclose(obj._translation_lock, [1.0, 0.0, 0.0])


def test_normalizes_non_axis_aligned_direction():
    obj = _make_tri_object()
    obj.lock_translation(1.0, 1.0, 0.0)
    axis = obj._translation_lock
    assert axis is not None
    assert np.isclose(np.linalg.norm(axis), 1.0)
    np.testing.assert_allclose(axis, [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])


def test_returns_self_for_chaining():
    obj = _make_tri_object()
    ret = obj.lock_translation(0.0, 1.0, 0.0)
    assert ret is obj


def test_zero_axis_raises():
    obj = _make_tri_object()
    with pytest.raises(ValueError, match="non-zero"):
        obj.lock_translation(0.0, 0.0, 0.0)
    assert obj._translation_lock is None


def test_non_finite_axis_raises():
    obj = _make_tri_object()
    with pytest.raises(ValueError, match="finite"):
        obj.lock_translation(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        obj.lock_translation(float("inf"), 0.0, 0.0)
    assert obj._translation_lock is None


def test_static_object_raises():
    obj = _make_tri_object()
    obj.pin()
    obj.update_static()
    assert obj.static
    with pytest.raises(Exception):
        obj.lock_translation(1.0, 0.0, 0.0)
    assert obj._translation_lock is None


def test_clear_resets_translation_lock():
    obj = _make_tri_object()
    obj.lock_translation(1.0, 0.0, 0.0)
    assert obj._translation_lock is not None
    obj.clear()
    assert obj._translation_lock is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
