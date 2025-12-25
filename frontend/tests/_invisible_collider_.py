# File: _invisible_collider_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for invisible collider (wall and sphere) violation detection.

These tests verify that vertices are on the correct side of invisible colliders:
- Wall: vertices must be on the positive normal side (signed_dist >= 0)
- Sphere (normal): vertices must be outside (dist >= radius)
- Sphere (inverted): vertices must be inside (dist <= radius)
- Hemisphere: above center.y, only horizontal distance matters (cylinder-like)

Note: Gap/offset separation is NOT checked here - that's handled by the simulator.
"""

import numpy as np

from .._invisible_collider_ import (
    check_invisible_collider_violations,
    check_sphere_violations,
    check_wall_violations,
)

# =============================================================================
# Mock classes for Wall and Sphere
# =============================================================================


class MockWall:
    """Mock Wall class for testing."""

    def __init__(
        self,
        position: list[float],
        normal: list[float],
    ):
        self._position = position
        self._normal = normal

    def get_entry(self):
        return [(self._position, 0.0)]

    @property
    def normal(self):
        return self._normal


class MockSphere:
    """Mock Sphere class for testing."""

    def __init__(
        self,
        center: list[float],
        radius: float,
        inverted: bool = False,
        hemisphere: bool = False,
    ):
        self._center = center
        self._radius = radius
        self._inverted = inverted
        self._hemisphere = hemisphere

    def get_entry(self):
        return [(self._center, self._radius, 0.0)]

    @property
    def is_inverted(self):
        return self._inverted

    @property
    def is_hemisphere(self):
        return self._hemisphere


# =============================================================================
# Wall Tests
# =============================================================================


def _create_vertices_above_wall():
    """Create vertices that are all above a ground plane (y=0, normal up)."""
    return np.array(
        [
            [0.0, 0.5, 0.0],   # Above
            [1.0, 0.3, 0.0],   # Above
            [-1.0, 0.1, 0.5],  # Above (just barely)
        ],
        dtype=np.float64,
    )


def _create_vertices_below_wall():
    """Create vertices with some below a ground plane (y=0, normal up)."""
    return np.array(
        [
            [0.0, 0.5, 0.0],   # Above
            [1.0, -0.1, 0.0],  # Below
            [-1.0, -0.5, 0.5], # Below
        ],
        dtype=np.float64,
    )


def test_wall_no_violations():
    """Test vertices that don't violate wall constraints."""
    print("  Testing wall - no violations...")

    vertices = _create_vertices_above_wall()
    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    violations = check_wall_violations(vertices, [wall])
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}"
    print("    Vertices above wall: PASS")


def test_wall_violations():
    """Test vertices that violate wall constraints (on wrong side)."""
    print("  Testing wall - violations...")

    vertices = _create_vertices_below_wall()
    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    violations = check_wall_violations(vertices, [wall])
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    # Check that the right vertices were flagged
    violating_verts = {v[0] for v in violations}
    assert violating_verts == {1, 2}, f"Expected vertices {{1, 2}}, got {violating_verts}"
    print("    Vertices below wall: PASS - 2 violations detected")


def test_wall_at_boundary():
    """Test vertices exactly on the wall plane (signed_dist = 0)."""
    print("  Testing wall - at boundary...")

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # Exactly on plane (OK, signed_dist = 0)
            [1.0, 0.0, 0.0],   # Exactly on plane (OK)
            [-1.0, -0.001, 0.5],  # Just below (violates)
        ],
        dtype=np.float64,
    )
    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    violations = check_wall_violations(vertices, [wall])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 2, "Expected vertex 2 to violate"
    print("    Vertices at wall boundary: PASS")


def test_wall_different_normals():
    """Test wall with different normal directions."""
    print("  Testing wall - different normals...")

    # Wall pointing in +X direction at x=0
    vertices = np.array(
        [
            [0.5, 0.0, 0.0],   # In front (OK)
            [-0.5, 0.0, 0.0],  # Behind (violates)
        ],
        dtype=np.float64,
    )
    wall = MockWall([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    violations = check_wall_violations(vertices, [wall])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 1, "Expected vertex 1 to violate"
    print("    Wall with +X normal: PASS")

    # Wall pointing in -Z direction at z=0
    vertices = np.array(
        [
            [0.0, 0.0, -0.5],  # In front of -Z wall (OK)
            [0.0, 0.0, 0.5],   # Behind -Z wall (violates)
        ],
        dtype=np.float64,
    )
    wall = MockWall([0.0, 0.0, 0.0], [0.0, 0.0, -1.0])

    violations = check_wall_violations(vertices, [wall])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 1, "Expected vertex 1 to violate"
    print("    Wall with -Z normal: PASS")


def test_wall_pinned_vertices_excluded():
    """Test that pinned vertices are excluded from checks."""
    print("  Testing wall - pinned vertices excluded...")

    vertices = _create_vertices_below_wall()
    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    # Pin the violating vertices
    pinned = {1, 2}
    violations = check_wall_violations(vertices, [wall], pinned_vertices=pinned)
    assert len(violations) == 0, f"Expected 0 violations (pinned), got {len(violations)}"
    print("    Pinned vertices below wall: PASS - excluded from check")


# =============================================================================
# Sphere Tests (Normal)
# =============================================================================


def _create_vertices_outside_sphere():
    """Create vertices that are all outside a sphere at origin with radius 1."""
    return np.array(
        [
            [2.0, 0.0, 0.0],   # Outside
            [0.0, 1.5, 0.0],   # Outside
            [0.0, 0.0, -2.0],  # Outside
        ],
        dtype=np.float64,
    )


def _create_vertices_inside_sphere():
    """Create vertices with some inside a sphere at origin with radius 1."""
    return np.array(
        [
            [2.0, 0.0, 0.0],   # Outside
            [0.5, 0.0, 0.0],   # Inside
            [0.0, 0.3, 0.4],   # Inside (distance = 0.5)
        ],
        dtype=np.float64,
    )


def test_sphere_no_violations():
    """Test vertices outside normal sphere."""
    print("  Testing sphere - no violations...")

    vertices = _create_vertices_outside_sphere()
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}"
    print("    Vertices outside sphere: PASS")


def test_sphere_violations():
    """Test vertices inside normal sphere."""
    print("  Testing sphere - violations...")

    vertices = _create_vertices_inside_sphere()
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    violating_verts = {v[0] for v in violations}
    assert violating_verts == {1, 2}, f"Expected vertices {{1, 2}}, got {violating_verts}"
    print("    Vertices inside sphere: PASS - 2 violations detected")


def test_sphere_at_boundary():
    """Test vertices exactly on sphere surface (dist = radius)."""
    print("  Testing sphere - at boundary...")

    # Vertex at exactly radius (OK, dist = radius)
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],   # At radius = 1.0 (OK)
            [0.0, 1.0, 0.0],   # At radius = 1.0 (OK)
            [0.99, 0.0, 0.0],  # Just inside (violates)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 2, "Expected vertex 2 to violate"
    print("    Vertices at sphere boundary: PASS")


# =============================================================================
# Inverted Sphere Tests
# =============================================================================


def test_inverted_sphere_no_violations():
    """Test vertices inside inverted sphere (bowl/container)."""
    print("  Testing inverted sphere - no violations...")

    # For inverted sphere, vertices must be INSIDE (dist <= radius)
    vertices = np.array(
        [
            [0.5, 0.0, 0.0],   # Inside (OK)
            [0.0, 0.3, 0.4],   # Inside (OK)
            [0.0, 0.0, 0.8],   # Inside (OK)
            [1.0, 0.0, 0.0],   # At boundary (OK, dist = radius)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0, inverted=True)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}"
    print("    Vertices inside inverted sphere: PASS")


def test_inverted_sphere_violations():
    """Test vertices outside inverted sphere."""
    print("  Testing inverted sphere - violations...")

    # For inverted sphere, vertices OUTSIDE violate (dist > radius)
    vertices = np.array(
        [
            [0.5, 0.0, 0.0],   # Inside (OK)
            [1.5, 0.0, 0.0],   # Outside (violates)
            [0.0, 0.0, 2.0],   # Outside (violates)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0, inverted=True)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    print("    Vertices outside inverted sphere: PASS - 2 violations detected")


# =============================================================================
# Hemisphere (Bowl) Tests
# =============================================================================


def test_hemisphere_below_center():
    """Test hemisphere behavior below center.y (acts like normal sphere)."""
    print("  Testing hemisphere - below center...")

    # Hemisphere at origin, radius 1
    # Below center.y, acts like normal sphere (must be outside)
    vertices = np.array(
        [
            [0.0, -0.5, 0.5],  # Below center, inside sphere (violates)
            [0.0, -0.5, 1.5],  # Below center, outside sphere (OK)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0, hemisphere=True)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 0, "Expected vertex 0 to violate"
    print("    Vertex inside hemisphere (below center): PASS - violation detected")


def test_hemisphere_above_center():
    """Test hemisphere behavior above center.y (acts like cylinder)."""
    print("  Testing hemisphere - above center...")

    # Hemisphere at origin, radius 1
    # Above center.y, only horizontal distance matters (cylinder)
    vertices = np.array(
        [
            [0.5, 2.0, 0.0],   # Above center, horizontal dist 0.5 < 1 (violates)
            [1.5, 2.0, 0.0],   # Above center, horizontal dist 1.5 > 1 (OK)
            [0.0, 5.0, 0.5],   # Above center, horizontal dist 0.5 < 1 (violates)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0, hemisphere=True)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    violating_verts = {v[0] for v in violations}
    assert violating_verts == {0, 2}, f"Expected vertices {{0, 2}}, got {violating_verts}"
    print("    Vertices inside hemisphere cylinder (above center): PASS - 2 violations")


def test_hemisphere_transition():
    """Test hemisphere at the transition point (y = center.y)."""
    print("  Testing hemisphere - at transition...")

    # At exactly y = center.y, should use spherical distance
    vertices = np.array(
        [
            [0.5, 0.0, 0.0],  # At center.y, inside sphere (violates)
            [1.5, 0.0, 0.0],  # At center.y, outside sphere (OK)
        ],
        dtype=np.float64,
    )
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0, hemisphere=True)

    violations = check_sphere_violations(vertices, [sphere])
    assert len(violations) == 1, f"Expected 1 violation, got {len(violations)}"
    assert violations[0][0] == 0, "Expected vertex 0 to violate"
    print("    Vertex at hemisphere transition: PASS")


# =============================================================================
# Combined Tests
# =============================================================================


def test_combined_wall_and_sphere():
    """Test combined wall and sphere violations."""
    print("  Testing combined wall and sphere...")

    # Sphere at (3, 0.5, 0) with radius 0.5
    # Vertex 0 at (0, 0.5, 0) - distance to sphere center = 3.0 > 0.5 (OK)
    # Vertex 1 at (0, -0.5, 0) - below wall (violates wall), outside sphere (OK)
    # Vertex 2 at (3.2, 0.5, 0) - distance to sphere center = 0.2 < 0.5 (violates sphere)
    vertices = np.array(
        [
            [0.0, 0.5, 0.0],   # Above wall, outside sphere (OK)
            [0.0, -0.5, 0.0],  # Below wall (violates wall)
            [3.2, 0.5, 0.0],   # Above wall, inside sphere (violates sphere)
        ],
        dtype=np.float64,
    )

    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    sphere = MockSphere([3.0, 0.5, 0.0], 0.5)

    wall_v, sphere_v = check_invisible_collider_violations(
        vertices, [wall], [sphere]
    )

    assert len(wall_v) == 1, f"Expected 1 wall violation, got {len(wall_v)}"
    assert wall_v[0][0] == 1, "Expected vertex 1 for wall violation"

    assert len(sphere_v) == 1, f"Expected 1 sphere violation, got {len(sphere_v)}"
    assert sphere_v[0][0] == 2, "Expected vertex 2 for sphere violation"
    print("    Combined violations: PASS - 1 wall, 1 sphere")


def test_multiple_walls():
    """Test multiple walls."""
    print("  Testing multiple walls...")

    vertices = np.array(
        [
            [0.5, 0.5, 0.5],   # Inside box (OK)
            [-0.5, 0.5, 0.5],  # Outside -X wall (violates)
            [0.5, -0.5, 0.5],  # Outside -Y wall (violates)
        ],
        dtype=np.float64,
    )

    # Box walls: +X, +Y (vertices must be in positive quadrant)
    walls = [
        MockWall([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # +X
        MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # +Y
    ]

    violations = check_wall_violations(vertices, walls)
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    print("    Multiple walls: PASS - 2 violations detected")


def test_multiple_spheres():
    """Test multiple spheres."""
    print("  Testing multiple spheres...")

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # Inside sphere 1 (violates)
            [3.0, 0.0, 0.0],   # Inside sphere 2 (violates)
            [1.5, 0.0, 0.0],   # Outside both (OK)
        ],
        dtype=np.float64,
    )

    spheres = [
        MockSphere([0.0, 0.0, 0.0], 0.5),
        MockSphere([3.0, 0.0, 0.0], 0.5),
    ]

    violations = check_sphere_violations(vertices, spheres)
    assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}"
    print("    Multiple spheres: PASS - 2 violations detected")


def test_sphere_pinned_excluded():
    """Test that pinned vertices are excluded from sphere checks."""
    print("  Testing sphere - pinned vertices excluded...")

    vertices = _create_vertices_inside_sphere()
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0)

    # Pin all violating vertices
    pinned = {1, 2}
    violations = check_sphere_violations(vertices, [sphere], pinned_vertices=pinned)
    assert len(violations) == 0, f"Expected 0 violations (pinned), got {len(violations)}"
    print("    Pinned vertices inside sphere: PASS - excluded from check")


# =============================================================================
# Empty/Edge Cases
# =============================================================================


def test_empty_colliders():
    """Test with no colliders."""
    print("  Testing empty colliders...")

    vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    wall_v, sphere_v = check_invisible_collider_violations(vertices, [], [])
    assert len(wall_v) == 0, f"Expected 0 wall violations, got {len(wall_v)}"
    assert len(sphere_v) == 0, f"Expected 0 sphere violations, got {len(sphere_v)}"
    print("    Empty colliders: PASS")


def test_all_vertices_pinned():
    """Test when all vertices are pinned."""
    print("  Testing all vertices pinned...")

    vertices = np.array(
        [
            [0.0, -1.0, 0.0],  # Below wall (would violate)
            [0.5, 0.0, 0.0],   # Inside sphere (would violate)
        ],
        dtype=np.float64,
    )

    wall = MockWall([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    sphere = MockSphere([0.0, 0.0, 0.0], 1.0)

    pinned = {0, 1}
    wall_v, sphere_v = check_invisible_collider_violations(
        vertices, [wall], [sphere], pinned_vertices=pinned
    )
    assert len(wall_v) == 0, f"Expected 0 wall violations, got {len(wall_v)}"
    assert len(sphere_v) == 0, f"Expected 0 sphere violations, got {len(sphere_v)}"
    print("    All vertices pinned: PASS - no violations")


# =============================================================================
# Test Runner
# =============================================================================


def run_tests() -> bool:
    """Run all invisible collider tests. Returns True if all tests pass."""
    print("=" * 50)
    print("Invisible Collider Violation Tests")
    print("=" * 50)

    try:
        # Wall tests
        test_wall_no_violations()
        test_wall_violations()
        test_wall_at_boundary()
        test_wall_different_normals()
        test_wall_pinned_vertices_excluded()

        # Sphere tests
        test_sphere_no_violations()
        test_sphere_violations()
        test_sphere_at_boundary()

        # Inverted sphere tests
        test_inverted_sphere_no_violations()
        test_inverted_sphere_violations()

        # Hemisphere tests
        test_hemisphere_below_center()
        test_hemisphere_above_center()
        test_hemisphere_transition()

        # Combined tests
        test_combined_wall_and_sphere()
        test_multiple_walls()
        test_multiple_spheres()
        test_sphere_pinned_excluded()

        # Edge cases
        test_empty_colliders()
        test_all_vertices_pinned()

        print("\nAll invisible collider tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
