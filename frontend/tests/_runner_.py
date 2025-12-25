# File: _runner_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test runner for all frontend tests."""

from . import _bvh_ as bvh_tests
from . import _invisible_collider_ as invisible_collider_tests
from . import _proximity_ as proximity_tests
from . import _self_intersection_ as self_intersection_tests


def run_all_tests() -> bool:
    """Run all frontend tests.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print()
    print("=" * 60)
    print("Running Frontend Tests")
    print("=" * 60)
    print()

    all_passed = True

    # Run BVH tests
    if not bvh_tests.run_tests():
        all_passed = False
    print()

    # Run self-intersection tests
    if not self_intersection_tests.run_tests():
        all_passed = False
    print()

    # Run proximity tests
    if not proximity_tests.run_tests():
        all_passed = False
    print()

    # Run invisible collider tests
    if not invisible_collider_tests.run_tests():
        all_passed = False
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("ALL FRONTEND TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    print()

    return all_passed


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
