# File: _self_intersection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for self-intersection detection."""

import time

import numpy as np

from .._intersection_ import check_self_intersection


def _create_grid_mesh(n: int = 10):
    """Create a flat grid mesh (no height variation)."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])

    return verts.astype(np.float64), np.array(tris, dtype=np.int32)


def _create_cube_mesh():
    """Create a simple cube mesh (12 triangles, no self-intersection)."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return verts, tris


def _create_sphere_mesh(subdivisions: int = 2):
    """Create a simple icosphere mesh (no self-intersection)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    verts = np.array(verts, dtype=np.float64)
    verts = verts / np.linalg.norm(verts[0])

    tris = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    for _ in range(subdivisions):
        new_tris = []
        edge_midpoints = {}
        verts_list = list(verts)

        for t in tris:
            v0, v1, v2 = t

            def get_or_create_midpoint(i1, i2, emp=edge_midpoints, vl=verts_list):
                key = (min(i1, i2), max(i1, i2))
                if key in emp:
                    return emp[key]
                mid = (vl[i1] + vl[i2]) / 2
                mid = mid / np.linalg.norm(mid)
                idx = len(vl)
                vl.append(mid)
                emp[key] = idx
                return idx

            m01 = get_or_create_midpoint(v0, v1)
            m12 = get_or_create_midpoint(v1, v2)
            m20 = get_or_create_midpoint(v2, v0)
            new_tris.append([v0, m01, m20])
            new_tris.append([v1, m12, m01])
            new_tris.append([v2, m20, m12])
            new_tris.append([m01, m12, m20])

        verts = np.array(verts_list, dtype=np.float64)
        tris = new_tris

    return verts, np.array(tris, dtype=np.int32)


def _create_two_intersecting_triangles():
    """Two triangles that definitely intersect (cross each other)."""
    verts = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 2, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_two_coplanar_overlapping():
    """Two coplanar triangles that overlap."""
    verts = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [1, 2, 0],
            [1, 0, 0],
            [3, 0, 0],
            [2, 2, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_two_coplanar_non_overlapping():
    """Two coplanar triangles that don't overlap."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.5, 0],
            [2, 0, 0],
            [3, 0, 0],
            [2.5, 0.5, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_adjacent_triangles():
    """Two triangles that share an edge (same vertex indices)."""
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
    return verts, tris


def _create_near_touching_triangles():
    """Two triangles that are very close but don't intersect.

    This tests numerical robustness - triangles on a curved surface
    may have overlapping bounding boxes and nearly coplanar geometry
    but should not be reported as intersecting.
    """
    # Two triangles facing roughly the same direction, very close but not touching
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0, 0, 0.01],  # Offset by small z
            [1, 0, 0.01],
            [0.5, 1, 0.01],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def _create_nearly_touching_coplanar_triangles():
    """Two coplanar triangles with a tiny gap between them.

    These should NOT be reported as intersecting.
    """
    gap = 1e-6
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0, -gap, 0],  # Tiny gap from edge
            [1, -gap, 0],
            [0.5, -1, 0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, tris


def test_non_intersecting():
    """Test non-intersecting meshes."""
    print("  Testing non-intersecting meshes...")

    v, t = _create_cube_mesh()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Cube: expected 0 intersections, got {len(pairs)}"
    print("    Cube: PASS")

    v, t = _create_grid_mesh(10)
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Grid: expected 0 intersections, got {len(pairs)}"
    print("    Flat grid 10x10: PASS")

    v, t = _create_sphere_mesh(2)
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Sphere: expected 0 intersections, got {len(pairs)}"
    print("    Icosphere (subdiv 2): PASS")

    # High subdivision icosphere - tests numerical robustness
    v, t = _create_sphere_mesh(3)
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Sphere subdiv 3: expected 0 intersections, got {len(pairs)}"
    print("    Icosphere (subdiv 3): PASS")


def test_intersecting():
    """Test intersecting meshes."""
    print("  Testing intersecting meshes...")

    v, t = _create_two_intersecting_triangles()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 1, f"Crossing: expected 1 intersection, got {len(pairs)}"
    print("    Crossing triangles: PASS")

    v, t = _create_two_coplanar_overlapping()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 1, f"Coplanar overlapping: expected 1, got {len(pairs)}"
    print("    Coplanar overlapping: PASS")


def test_edge_cases():
    """Test edge cases."""
    print("  Testing edge cases...")

    v, t = _create_two_coplanar_non_overlapping()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Coplanar non-overlapping: expected 0, got {len(pairs)}"
    print("    Coplanar non-overlapping: PASS")

    v, t = _create_adjacent_triangles()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Adjacent: expected 0 (filtered), got {len(pairs)}"
    print("    Adjacent triangles (shared vertices): PASS")

    v, t = _create_near_touching_triangles()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Near-touching: expected 0, got {len(pairs)}"
    print("    Near-touching triangles: PASS")

    v, t = _create_nearly_touching_coplanar_triangles()
    pairs = check_self_intersection(v, t)
    assert len(pairs) == 0, f"Nearly touching coplanar: expected 0, got {len(pairs)}"
    print("    Nearly touching coplanar: PASS")


def test_collider_intersection():
    """Test intersection detection with collider flags (static vs dynamic)."""
    print("  Testing collider intersection...")

    # Create a flat sheet (dynamic) and a sphere (static collider) that intersect
    # Similar to intersect.ipynb scenario

    # Sheet: flat grid on xz plane at y=0
    n = 5
    x = np.linspace(-0.5, 0.5, n)
    z = np.linspace(-0.5, 0.5, n)
    xx, zz = np.meshgrid(x, z)
    yy = np.zeros_like(xx)
    sheet_verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    sheet_tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            sheet_tris.append([v0, v1, v2])
            sheet_tris.append([v1, v3, v2])
    sheet_tris = np.array(sheet_tris, dtype=np.int32)
    n_sheet_tris = len(sheet_tris)

    # Sphere: icosphere at y=-0.25 with radius 0.5 (intersects sheet at y=0)
    sphere_verts, sphere_tris = _create_sphere_mesh(1)
    sphere_verts = sphere_verts * 0.5  # radius 0.5
    sphere_verts[:, 1] -= 0.25  # offset to y=-0.25

    # Combine meshes
    n_sheet_verts = len(sheet_verts)
    combined_verts = np.vstack([sheet_verts, sphere_verts])
    combined_tris = np.vstack([sheet_tris, sphere_tris + n_sheet_verts])

    # Test 1: Without collider flags, should detect intersections
    pairs = check_self_intersection(combined_verts, combined_tris)
    assert len(pairs) > 0, "Expected intersections between sheet and sphere"
    print(f"    Without collider flags: {len(pairs)} intersections detected: PASS")

    # Test 2: Mark sphere as collider, sheet as dynamic - should still detect
    is_collider = np.zeros(len(combined_tris), dtype=bool)
    is_collider[n_sheet_tris:] = True  # sphere triangles are colliders
    pairs = check_self_intersection(combined_verts, combined_tris, is_collider)
    assert len(pairs) > 0, "Expected intersections between dynamic sheet and static sphere"
    print(f"    Dynamic vs static collider: {len(pairs)} intersections detected: PASS")

    # Test 3: Mark BOTH as colliders - should skip (allow self-intersection)
    is_collider_both = np.ones(len(combined_tris), dtype=bool)
    pairs = check_self_intersection(combined_verts, combined_tris, is_collider_both)
    assert len(pairs) == 0, f"Collider vs collider should be skipped, got {len(pairs)}"
    print("    Collider vs collider (both static): skipped as expected: PASS")


def test_performance():
    """Test performance."""
    print("  Testing performance...")

    for subdiv in [3, 4]:
        v, t = _create_sphere_mesh(subdiv)
        t0 = time.perf_counter()
        _ = check_self_intersection(v, t)
        t1 = time.perf_counter()
        print(f"    Icosphere (subdiv {subdiv}): {len(t)} tris, {(t1-t0)*1000:.2f}ms")

    for n in [30, 50]:
        v, t = _create_grid_mesh(n)
        t0 = time.perf_counter()
        _ = check_self_intersection(v, t)
        t1 = time.perf_counter()
        print(f"    Flat grid {n}x{n}: {len(t)} tris, {(t1-t0)*1000:.2f}ms")


def run_tests() -> bool:
    """Run all self-intersection tests. Returns True if all tests pass."""
    print("=" * 50)
    print("Self-Intersection Detection Tests")
    print("=" * 50)

    try:
        test_non_intersecting()
        test_intersecting()
        test_edge_cases()
        test_collider_intersection()
        test_performance()
        print("\nAll self-intersection tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
