# File: _asset_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for AssetManager pickle round-trip.

Covers ``AssetManager.__getstate__`` / ``__setstate__`` (lines 101-117 of
``frontend/_asset_.py``); confirms a registry can be snapshotted and
rehydrated without losing entries or array contents.
"""

import numpy as np

from .._asset_ import AssetManager


def _make_simple_tri():
    V = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2]], dtype=np.uint32)
    return V, F


def _make_simple_rod():
    V = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5)
    E = np.array([[i, i + 1] for i in range(len(V) - 1)], dtype=np.uint32)
    return V, E


def test_pickle_roundtrip_preserves_entries():
    """A pickled-then-restored manager keeps the same names, types, and
    array contents."""
    print("  Testing AssetManager pickle round-trip...")

    src = AssetManager()
    V_tri, F_tri = _make_simple_tri()
    V_rod, E_rod = _make_simple_rod()
    src.add.tri("sheet", V_tri, F_tri)
    src.add.rod("strand", V_rod, E_rod)

    state = src.__getstate__()
    assert isinstance(state, dict), f"state must be a dict, got {type(state)}"
    assert "snapshot" in state, "state missing 'snapshot' key"

    dst = AssetManager()
    dst.__setstate__(state)

    src_names = sorted(src.list())
    dst_names = sorted(dst.list())
    assert src_names == dst_names, (
        f"Names differ after round-trip: {src_names} vs {dst_names}"
    )

    assert dst.fetch.get_type("sheet") == "tri"
    assert dst.fetch.get_type("strand") == "rod"

    V_tri_dst, F_tri_dst = dst.fetch.tri("sheet")
    assert np.array_equal(V_tri_dst, V_tri), "tri vertices changed in round-trip"
    assert np.array_equal(F_tri_dst, F_tri), "tri faces changed in round-trip"

    V_rod_dst, E_rod_dst = dst.fetch.rod("strand")
    assert np.array_equal(V_rod_dst, V_rod), "rod vertices changed in round-trip"
    assert np.array_equal(E_rod_dst, E_rod), "rod edges changed in round-trip"

    print("    Names/types preserved: PASS")
    print("    Array contents preserved: PASS")


def test_pickle_roundtrip_empty_registry():
    """An empty registry round-trips to an empty registry without error."""
    print("  Testing empty AssetManager pickle round-trip...")

    src = AssetManager()
    state = src.__getstate__()
    dst = AssetManager()
    dst.__setstate__(state)
    assert dst.list() == [], f"Expected empty list, got {dst.list()}"

    print("    Empty round-trip: PASS")


def run_tests() -> bool:
    """Run all asset tests. Returns True if all tests pass."""
    print("=" * 50)
    print("AssetManager Pickle Tests")
    print("=" * 50)

    try:
        test_pickle_roundtrip_preserves_entries()
        test_pickle_roundtrip_empty_registry()
        print("\nAll Asset tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
