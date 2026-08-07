# File: frontend/tests/_lock_rotation_.py
# License: Apache v2.0
#
# Unit tests for ``Object.lock_rotation()``.
#
# Lock Rotation restricts an object's mass-weighted best-fit rigid
# rotation to rotation about a caller-given fixed world-space axis
# only; translation and deformation stay free, and it coexists
# independently with Lock Translation (either, both, or neither may be
# set on the same object). This is a frontend/data-pipeline-only
# feature today: the solver-side constraint is future work, so these
# tests cover only the state/validation the frontend owns:
#   * the axis is normalized (only direction matters, not magnitude);
#   * a non-finite or zero-norm axis is rejected loudly rather than
#     being replaced with a plausible default;
#   * a static object cannot be locked (it has no free rotation to
#     constrain in the first place);
#   * the object starts unlocked (``_rotation_lock is None``), and
#     there is no "unlock" call: the disabled state is simply "never
#     called ``lock_rotation()``";
#   * enabling Lock Rotation does not disturb Lock Translation state
#     (and vice versa);
#   * ``lock_rotation_prohibit_axis()`` flips the axis from a whitelist
#     (default; only rotation about it is allowed) to a blacklist
#     (rotation about it is forbidden, the perpendicular plane stays
#     free), and raises loudly if called before ``lock_rotation()``
#     since there is no axis yet for the mode to modify.

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
    since ``lock_rotation()`` only touches asset-independent state.
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
    assert obj._rotation_lock is None


def test_normalizes_axis():
    obj = _make_tri_object()
    obj.lock_rotation(2.0, 0.0, 0.0)
    assert obj._rotation_lock is not None
    np.testing.assert_allclose(obj._rotation_lock, [1.0, 0.0, 0.0])


def test_normalizes_non_axis_aligned_direction():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 1.0, 0.0)
    axis = obj._rotation_lock
    assert axis is not None
    assert np.isclose(np.linalg.norm(axis), 1.0)
    np.testing.assert_allclose(axis, [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])


def test_returns_self_for_chaining():
    obj = _make_tri_object()
    ret = obj.lock_rotation(0.0, 1.0, 0.0)
    assert ret is obj


def test_zero_axis_raises():
    obj = _make_tri_object()
    with pytest.raises(ValueError, match="non-zero"):
        obj.lock_rotation(0.0, 0.0, 0.0)
    assert obj._rotation_lock is None


def test_non_finite_axis_raises():
    obj = _make_tri_object()
    with pytest.raises(ValueError, match="finite"):
        obj.lock_rotation(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        obj.lock_rotation(float("inf"), 0.0, 0.0)
    assert obj._rotation_lock is None


def test_static_object_raises():
    obj = _make_tri_object()
    obj.pin()
    obj.update_static()
    assert obj.static
    with pytest.raises(Exception):
        obj.lock_rotation(1.0, 0.0, 0.0)
    assert obj._rotation_lock is None


def test_clear_resets_rotation_lock():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    assert obj._rotation_lock is not None
    obj.clear()
    assert obj._rotation_lock is None


def test_coexists_independently_with_translation_lock():
    obj = _make_tri_object()
    obj.lock_translation(1.0, 0.0, 0.0)
    obj.lock_rotation(0.0, 1.0, 0.0)
    np.testing.assert_allclose(obj._translation_lock, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obj._rotation_lock, [0.0, 1.0, 0.0])


def test_prohibit_axis_defaults_to_false():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    assert obj._rotation_lock_prohibit_axis is False


def test_prohibit_axis_flips_mode():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    obj.lock_rotation_prohibit_axis(True)
    assert obj._rotation_lock_prohibit_axis is True
    obj.lock_rotation_prohibit_axis(False)
    assert obj._rotation_lock_prohibit_axis is False


def test_prohibit_axis_returns_self_for_chaining():
    obj = _make_tri_object()
    ret = obj.lock_rotation(0.0, 1.0, 0.0).lock_rotation_prohibit_axis(True)
    assert ret is obj


def test_prohibit_axis_without_lock_rotation_raises():
    obj = _make_tri_object()
    with pytest.raises(ValueError, match="lock_rotation"):
        obj.lock_rotation_prohibit_axis(True)
    assert obj._rotation_lock_prohibit_axis is False


def test_clear_resets_prohibit_axis():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    obj.lock_rotation_prohibit_axis(True)
    obj.clear()
    assert obj._rotation_lock is None
    assert obj._rotation_lock_prohibit_axis is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
