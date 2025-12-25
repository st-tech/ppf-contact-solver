# File: __init__.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test suite for the frontend module."""

from .._bvh_ import compute_barycentric_mapping, interpolate_surface
from .._intersection_ import check_self_intersection

__all__ = [
    "check_self_intersection",
    "compute_barycentric_mapping",
    "interpolate_surface",
]
