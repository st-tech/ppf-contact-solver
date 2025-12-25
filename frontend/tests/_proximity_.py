# File: _proximity_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for contact-offset proximity detection."""

import numpy as np

from .._proximity_ import check_contact_offset_violation


def _create_two_triangles_close():
    """Two parallel triangles that are close but not intersecting."""
    verts = np.array(
        [
            # Triangle 1 at z=0
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            # Triangle 2 at z=0.05 (close)
            [0, 0, 0.05],
            [1, 0, 0.05],
            [0.5, 1, 0.05],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_two_triangles_far():
    """Two parallel triangles that are far apart."""
    verts = np.array(
        [
            # Triangle 1 at z=0
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            # Triangle 2 at z=1.0 (far)
            [0, 0, 1.0],
            [1, 0, 1.0],
            [0.5, 1, 1.0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_two_edges_close():
    """Two parallel edges that are close but not intersecting."""
    verts = np.array(
        [
            # Edge 1 along x-axis at z=0
            [0, 0, 0],
            [1, 0, 0],
            # Edge 2 along x-axis at z=0.05 (close)
            [0, 0, 0.05],
            [1, 0, 0.05],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    return verts, edges


def _create_two_edges_far():
    """Two parallel edges that are far apart."""
    verts = np.array(
        [
            # Edge 1 along x-axis at z=0
            [0, 0, 0],
            [1, 0, 0],
            # Edge 2 along x-axis at z=1.0 (far)
            [0, 0, 1.0],
            [1, 0, 1.0],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    return verts, edges


def _create_triangle_and_edge_close():
    """A triangle and an edge that are close but not intersecting."""
    verts = np.array(
        [
            # Triangle at z=0
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            # Edge at z=0.05 (close)
            [0.25, 0.25, 0.05],
            [0.75, 0.25, 0.05],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    edges = np.array([[3, 4]], dtype=np.int32)
    return verts, tris, edges


def _create_triangle_and_edge_far():
    """A triangle and an edge that are far apart."""
    verts = np.array(
        [
            # Triangle at z=0
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            # Edge at z=1.0 (far)
            [0.25, 0.25, 1.0],
            [0.75, 0.25, 1.0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    edges = np.array([[3, 4]], dtype=np.int32)
    return verts, tris, edges


def _create_crossing_edges_close():
    """Two crossing edges (in XY plane) at different Z heights."""
    verts = np.array(
        [
            # Edge 1 along x-axis at z=0
            [-1, 0, 0],
            [1, 0, 0],
            # Edge 2 along y-axis at z=0.05 (close, crossing over)
            [0, -1, 0.05],
            [0, 1, 0.05],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    return verts, edges


def test_triangle_triangle_proximity():
    """Test triangle-triangle proximity detection."""
    print("  Testing triangle-triangle proximity...")

    # Close triangles with contact offset 0.1 - should detect violation
    v, t = _create_two_triangles_close()
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(v, F=t, contact_offset=offset)
    assert len(violations) == 1, f"Close triangles: expected 1 violation, got {len(violations)}"
    print("    Close triangles (offset 0.1, dist 0.05): PASS - violation detected")

    # Close triangles with small contact offset - should not detect violation
    offset = np.array([0.02, 0.02], dtype=np.float64)  # required: 0.04, actual: 0.05
    violations = check_contact_offset_violation(v, F=t, contact_offset=offset)
    assert len(violations) == 0, f"Close triangles small offset: expected 0 violations, got {len(violations)}"
    print("    Close triangles (offset 0.04, dist 0.05): PASS - no violation")

    # Far triangles - should not detect violation
    v, t = _create_two_triangles_far()
    offset = np.array([0.1, 0.1], dtype=np.float64)
    violations = check_contact_offset_violation(v, F=t, contact_offset=offset)
    assert len(violations) == 0, f"Far triangles: expected 0 violations, got {len(violations)}"
    print("    Far triangles (offset 0.2, dist 1.0): PASS - no violation")


def test_edge_edge_proximity():
    """Test edge-edge proximity detection."""
    print("  Testing edge-edge proximity...")

    # Close edges with contact offset 0.1 - should detect violation
    v, e = _create_two_edges_close()
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(v, E=e, contact_offset=offset)
    assert len(violations) == 1, f"Close edges: expected 1 violation, got {len(violations)}"
    print("    Close edges (offset 0.1, dist 0.05): PASS - violation detected")

    # Close edges with small contact offset - should not detect violation
    offset = np.array([0.02, 0.02], dtype=np.float64)  # required: 0.04, actual: 0.05
    violations = check_contact_offset_violation(v, E=e, contact_offset=offset)
    assert len(violations) == 0, f"Close edges small offset: expected 0 violations, got {len(violations)}"
    print("    Close edges (offset 0.04, dist 0.05): PASS - no violation")

    # Far edges - should not detect violation
    v, e = _create_two_edges_far()
    offset = np.array([0.1, 0.1], dtype=np.float64)
    violations = check_contact_offset_violation(v, E=e, contact_offset=offset)
    assert len(violations) == 0, f"Far edges: expected 0 violations, got {len(violations)}"
    print("    Far edges (offset 0.2, dist 1.0): PASS - no violation")

    # Crossing edges (close in Z) - should detect violation
    v, e = _create_crossing_edges_close()
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(v, E=e, contact_offset=offset)
    assert len(violations) == 1, f"Crossing edges: expected 1 violation, got {len(violations)}"
    print("    Crossing edges (offset 0.1, dist 0.05): PASS - violation detected")


def test_triangle_edge_proximity():
    """Test triangle-edge proximity detection."""
    print("  Testing triangle-edge proximity...")

    # Close triangle and edge - should detect violation
    v, t, e = _create_triangle_and_edge_close()
    # Offset array: [triangle_offset, edge_offset]
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(v, F=t, E=e, contact_offset=offset)
    assert len(violations) == 1, f"Close tri-edge: expected 1 violation, got {len(violations)}"
    print("    Close triangle-edge (offset 0.1, dist 0.05): PASS - violation detected")

    # Close triangle and edge with small offset - should not detect violation
    offset = np.array([0.02, 0.02], dtype=np.float64)  # required: 0.04, actual: 0.05
    violations = check_contact_offset_violation(v, F=t, E=e, contact_offset=offset)
    assert len(violations) == 0, f"Close tri-edge small offset: expected 0 violations, got {len(violations)}"
    print("    Close triangle-edge (offset 0.04, dist 0.05): PASS - no violation")

    # Far triangle and edge - should not detect violation
    v, t, e = _create_triangle_and_edge_far()
    offset = np.array([0.1, 0.1], dtype=np.float64)
    violations = check_contact_offset_violation(v, F=t, E=e, contact_offset=offset)
    assert len(violations) == 0, f"Far tri-edge: expected 0 violations, got {len(violations)}"
    print("    Far triangle-edge (offset 0.2, dist 1.0): PASS - no violation")


def test_collider_exclusion():
    """Test that collider pairs are correctly excluded."""
    print("  Testing collider exclusion...")

    # Close triangles - both marked as colliders, should be skipped
    v, t = _create_two_triangles_close()
    offset = np.array([0.05, 0.05], dtype=np.float64)
    is_collider = np.array([True, True], dtype=bool)
    violations = check_contact_offset_violation(v, F=t, is_collider=is_collider, contact_offset=offset)
    assert len(violations) == 0, f"Both colliders: expected 0 violations, got {len(violations)}"
    print("    Both triangles colliders: PASS - pair skipped")

    # Close triangles - only one marked as collider, should NOT be skipped
    is_collider = np.array([True, False], dtype=bool)
    violations = check_contact_offset_violation(v, F=t, is_collider=is_collider, contact_offset=offset)
    assert len(violations) == 1, f"One collider: expected 1 violation, got {len(violations)}"
    print("    One triangle collider: PASS - pair checked")

    # Close edges - both marked as colliders, should be skipped
    v, e = _create_two_edges_close()
    is_collider = np.array([True, True], dtype=bool)
    violations = check_contact_offset_violation(v, E=e, is_collider=is_collider, contact_offset=offset)
    assert len(violations) == 0, f"Both edge colliders: expected 0 violations, got {len(violations)}"
    print("    Both edges colliders: PASS - pair skipped")

    # Mixed triangle and edge - both colliders, should be skipped
    v, t, e = _create_triangle_and_edge_close()
    is_collider = np.array([True, True], dtype=bool)
    violations = check_contact_offset_violation(v, F=t, E=e, is_collider=is_collider, contact_offset=offset)
    assert len(violations) == 0, f"Tri-edge both colliders: expected 0 violations, got {len(violations)}"
    print("    Triangle and edge both colliders: PASS - pair skipped")

    # Mixed triangle and edge - only triangle is collider, should check
    is_collider = np.array([True, False], dtype=bool)
    violations = check_contact_offset_violation(v, F=t, E=e, is_collider=is_collider, contact_offset=offset)
    assert len(violations) == 1, f"Only tri collider: expected 1 violation, got {len(violations)}"
    print("    Only triangle collider: PASS - pair checked")


def test_point_point_proximity():
    """Test point-point proximity (implicit through edge endpoints and triangle vertices)."""
    print("  Testing point-point proximity...")

    # Two short edges where closest points are at endpoints (point-point case)
    # Edge 1: single point at origin, Edge 2: single point at (0, 0, 0.05)
    verts = np.array(
        [
            [0, 0, 0],
            [0.001, 0, 0],  # Very short edge (essentially a point)
            [0, 0, 0.05],
            [0.001, 0, 0.05],  # Very short edge (essentially a point)
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(verts, E=edges, contact_offset=offset)
    assert len(violations) == 1, f"Point-point via edges: expected 1 violation, got {len(violations)}"
    print("    Close points via edges (offset 0.1, dist 0.05): PASS - violation detected")

    # Small offset - should not detect violation
    offset = np.array([0.02, 0.02], dtype=np.float64)  # required: 0.04, actual: 0.05
    violations = check_contact_offset_violation(verts, E=edges, contact_offset=offset)
    assert len(violations) == 0, f"Point-point small offset: expected 0 violations, got {len(violations)}"
    print("    Close points via edges (offset 0.04, dist 0.05): PASS - no violation")

    # Two triangles where closest points are vertices (corner-to-corner)
    verts = np.array(
        [
            # Triangle 1 - tip at origin
            [0, 0, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            # Triangle 2 - tip close to origin
            [0.05, 0, 0],
            [1.05, -1, 0],
            [1.05, 1, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    offset = np.array([0.05, 0.05], dtype=np.float64)  # required: 0.1, actual: 0.05
    violations = check_contact_offset_violation(verts, F=tris, contact_offset=offset)
    assert len(violations) == 1, f"Point-point via triangles: expected 1 violation, got {len(violations)}"
    print("    Close vertices via triangles (offset 0.1, dist 0.05): PASS - violation detected")

    # Small offset - should not detect violation
    offset = np.array([0.02, 0.02], dtype=np.float64)  # required: 0.04, actual: 0.05
    violations = check_contact_offset_violation(verts, F=tris, contact_offset=offset)
    assert len(violations) == 0, f"Point-point triangles small offset: expected 0 violations, got {len(violations)}"
    print("    Close vertices via triangles (offset 0.04, dist 0.05): PASS - no violation")


def test_adjacent_elements():
    """Test that adjacent elements (sharing vertices) are skipped."""
    print("  Testing adjacent element exclusion...")

    # Two triangles sharing an edge
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, -1, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    offset = np.array([0.5, 0.5], dtype=np.float64)  # Large offset
    violations = check_contact_offset_violation(verts, F=tris, contact_offset=offset)
    assert len(violations) == 0, f"Adjacent triangles: expected 0 violations, got {len(violations)}"
    print("    Adjacent triangles (shared edge): PASS - pair skipped")

    # Two edges sharing a vertex (strand)
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    offset = np.array([0.5, 0.5], dtype=np.float64)
    violations = check_contact_offset_violation(verts, E=edges, contact_offset=offset)
    assert len(violations) == 0, f"Adjacent edges: expected 0 violations, got {len(violations)}"
    print("    Adjacent edges (shared vertex): PASS - pair skipped")


def run_tests() -> bool:
    """Run all proximity tests. Returns True if all tests pass."""
    print("=" * 50)
    print("Contact-Offset Proximity Tests")
    print("=" * 50)

    try:
        test_triangle_triangle_proximity()
        test_edge_edge_proximity()
        test_triangle_edge_proximity()
        test_point_point_proximity()
        test_collider_exclusion()
        test_adjacent_elements()
        print("\nAll proximity tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
