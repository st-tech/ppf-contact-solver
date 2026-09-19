# File: frontend/tests/_lock_rotation_.py
# License: Apache v2.0
#
# Unit tests for ``Object.lock_rotation()``.
#
# Lock Rotation restricts an object's mass-weighted best-fit rigid
# rotation to rotation about a caller-given fixed world-space axis
# only, or forbids every axis at once; translation and deformation stay
# free, and it coexists independently with Lock Translation (either,
# both, or neither may be set on the same object). These tests cover the
# state and validation the frontend owns, which is what the solver-side
# projector is handed:
#   * the axis is normalized (only direction matters, not magnitude);
#   * a non-finite or zero-norm axis is rejected loudly rather than
#     being replaced with a plausible default;
#   * a static object cannot be locked (it has no free rotation to
#     constrain in the first place);
#   * the object starts unlocked (``_rotation_lock is None`` and
#     ``_rotation_lock_all is False``), and there is no "unlock" call:
#     the disabled state is simply "never called ``lock_rotation()`` or
#     ``lock_all_rotations()``";
#   * enabling Lock Rotation does not disturb Lock Translation state
#     (and vice versa);
#   * ``lock_rotation_prohibit_axis()`` flips the axis from a whitelist
#     (default; only rotation about it is allowed) to a blacklist
#     (rotation about it is forbidden, the perpendicular plane stays
#     free), and raises loudly if there is no axis for the mode to
#     modify, whether because ``lock_rotation()`` was never called or
#     because ``lock_all_rotations()`` took the axis away;
#   * ``lock_all_rotations()`` / ``lock_all_translations()`` set an
#     all-axes lock that carries NO axis, so the flag alone says whether
#     the lock is on. The two spellings of each lock are mutually
#     exclusive and the last call wins, each clearing the other, so an
#     object never holds an axis and an all-axes flag at once.

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
        "frontend._rust extension not built; run `cargo build --release` first",
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


def test_starts_with_no_all_axes_lock():
    obj = _make_tri_object()
    assert obj._rotation_lock_all is False
    assert obj._translation_lock_all is False


def test_lock_all_rotations_sets_flag_and_carries_no_axis():
    obj = _make_tri_object()
    obj.lock_all_rotations()
    assert obj._rotation_lock_all is True
    # The flag carries the enable bit: there is no axis to read in this
    # mode, and none is invented to stand in for one.
    assert obj._rotation_lock is None


def test_lock_all_rotations_returns_self_for_chaining():
    obj = _make_tri_object()
    ret = obj.lock_all_rotations()
    assert ret is obj


def test_lock_all_rotations_clears_axis_and_prohibit_axis():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    obj.lock_rotation_prohibit_axis(True)
    obj.lock_all_rotations()
    assert obj._rotation_lock_all is True
    assert obj._rotation_lock is None
    assert obj._rotation_lock_prohibit_axis is False


def test_lock_rotation_clears_all_axes_flag():
    obj = _make_tri_object()
    obj.lock_all_rotations()
    obj.lock_rotation(0.0, 0.0, 1.0)
    assert obj._rotation_lock_all is False
    np.testing.assert_allclose(obj._rotation_lock, [0.0, 0.0, 1.0])


def test_prohibit_axis_under_lock_all_rotations_raises():
    obj = _make_tri_object()
    obj.lock_all_rotations()
    with pytest.raises(ValueError, match="lock_rotation"):
        obj.lock_rotation_prohibit_axis(True)
    assert obj._rotation_lock_prohibit_axis is False
    assert obj._rotation_lock_all is True


def test_lock_all_rotations_on_static_object_raises():
    obj = _make_tri_object()
    obj.pin()
    obj.update_static()
    assert obj.static
    with pytest.raises(Exception):
        obj.lock_all_rotations()
    assert obj._rotation_lock_all is False


def test_clear_resets_lock_all_rotations():
    obj = _make_tri_object()
    obj.lock_all_rotations()
    obj.clear()
    assert obj._rotation_lock_all is False
    assert obj._rotation_lock is None


def test_lock_all_translations_sets_flag_and_carries_no_axis():
    obj = _make_tri_object()
    obj.lock_all_translations()
    assert obj._translation_lock_all is True
    assert obj._translation_lock is None


def test_lock_all_translations_returns_self_for_chaining():
    obj = _make_tri_object()
    ret = obj.lock_all_translations()
    assert ret is obj


def test_lock_all_translations_clears_axis():
    obj = _make_tri_object()
    obj.lock_translation(1.0, 0.0, 0.0)
    obj.lock_all_translations()
    assert obj._translation_lock_all is True
    assert obj._translation_lock is None


def test_lock_translation_clears_all_axes_flag():
    obj = _make_tri_object()
    obj.lock_all_translations()
    obj.lock_translation(0.0, 1.0, 0.0)
    assert obj._translation_lock_all is False
    np.testing.assert_allclose(obj._translation_lock, [0.0, 1.0, 0.0])


def test_lock_all_translations_on_static_object_raises():
    obj = _make_tri_object()
    obj.pin()
    obj.update_static()
    assert obj.static
    with pytest.raises(Exception):
        obj.lock_all_translations()
    assert obj._translation_lock_all is False


def test_clear_resets_lock_all_translations():
    obj = _make_tri_object()
    obj.lock_all_translations()
    obj.clear()
    assert obj._translation_lock_all is False
    assert obj._translation_lock is None


def test_all_axes_locks_coexist_independently():
    obj = _make_tri_object()
    obj.lock_all_translations()
    obj.lock_all_rotations()
    assert obj._translation_lock_all is True
    assert obj._rotation_lock_all is True
    # The families mix modes freely: an all-axes translation lock beside
    # a single-axis rotation lock is a legal pairing.
    obj.lock_rotation(1.0, 0.0, 0.0)
    assert obj._translation_lock_all is True
    assert obj._rotation_lock_all is False
    np.testing.assert_allclose(obj._rotation_lock, [1.0, 0.0, 0.0])


def test_clear_resets_prohibit_axis():
    obj = _make_tri_object()
    obj.lock_rotation(1.0, 0.0, 0.0)
    obj.lock_rotation_prohibit_axis(True)
    obj.clear()
    assert obj._rotation_lock is None
    assert obj._rotation_lock_prohibit_axis is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
