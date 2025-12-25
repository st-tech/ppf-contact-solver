# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for BVH barycentric mapping and interpolation."""

import time

import numpy as np

from .._bvh_ import compute_barycentric_mapping, interpolate_surface


def _create_test_mesh(n_verts_per_side: int = 10):
    """Create a simple grid mesh for testing."""
    x = np.linspace(0, 1, n_verts_per_side)
    y = np.linspace(0, 1, n_verts_per_side)
    xx, yy = np.meshgrid(x, y)
    zz = np.sin(xx * np.pi) * np.sin(yy * np.pi) * 0.2

    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    tris = []
    for i in range(n_verts_per_side - 1):
        for j in range(n_verts_per_side - 1):
            v0 = i * n_verts_per_side + j
            v1 = v0 + 1
            v2 = v0 + n_verts_per_side
            v3 = v2 + 1
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])

    return verts.astype(np.float64), np.array(tris, dtype=np.int32)


def test_basic_functionality():
    """Test basic BVH functionality."""
    print("  Testing basic functionality...")

    verts, tris = _create_test_mesh(5)
    query_points = verts.copy()

    tri_indices, bary_coords = compute_barycentric_mapping(query_points, verts, tris)

    # Verify barycentric coordinates sum to 1
    bary_sum = bary_coords.sum(axis=1)
    assert np.allclose(bary_sum, 1.0), f"Barycentric coords don't sum to 1: {bary_sum}"

    # Test interpolation
    interpolated = interpolate_surface(verts, tris, tri_indices, bary_coords)
    max_error = np.abs(interpolated - query_points).max()
    assert max_error < 1e-10, f"Interpolation error too large: {max_error}"

    print("    Barycentric coordinates: PASS")
    print("    Interpolation accuracy: PASS")


def test_performance():
    """Test performance with larger mesh."""
    print("  Testing performance...")

    verts, tris = _create_test_mesh(50)
    np.random.seed(42)
    n_queries = 10000
    query_points = np.random.rand(n_queries, 3).astype(np.float64)
    query_points[:, 0] *= 1.0
    query_points[:, 1] *= 1.0
    query_points[:, 2] = query_points[:, 2] * 0.4 - 0.1

    # Warm-up
    _ = compute_barycentric_mapping(query_points[:10], verts, tris)

    # Timed run
    t0 = time.perf_counter()
    tri_indices, bary_coords = compute_barycentric_mapping(query_points, verts, tris)
    t1 = time.perf_counter()

    print(f"    Mapping {n_queries} points: {(t1-t0)*1000:.2f}ms")

    # Test interpolation performance
    t0 = time.perf_counter()
    for _ in range(100):
        _ = interpolate_surface(verts, tris, tri_indices, bary_coords)
    t1 = time.perf_counter()
    print(f"    Interpolation (100 iters): {(t1-t0)*1000:.2f}ms")


def test_edge_cases():
    """Test edge cases."""
    print("  Testing edge cases...")

    # Single triangle
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)

    # Point inside triangle
    query = np.array([[0.25, 0.25, 0]], dtype=np.float64)
    tri_idx, bary = compute_barycentric_mapping(query, verts, tris)
    assert tri_idx[0] == 0, "Wrong triangle"
    assert np.allclose(bary[0].sum(), 1.0), "Barycentric coords don't sum to 1"

    # Point at vertex
    query = np.array([[0, 0, 0]], dtype=np.float64)
    tri_idx, bary = compute_barycentric_mapping(query, verts, tris)
    interpolated = interpolate_surface(verts, tris, tri_idx, bary)
    assert np.allclose(interpolated[0], [0, 0, 0]), "Wrong interpolation at vertex"

    # Point outside triangle
    query = np.array([[2, 2, 0]], dtype=np.float64)
    tri_idx, bary = compute_barycentric_mapping(query, verts, tris)
    assert np.allclose(bary[0].sum(), 1.0), "Barycentric coords don't sum to 1"
    assert np.all(bary[0] >= -1e-10), "Negative barycentric coord"

    print("    Point inside triangle: PASS")
    print("    Point at vertex: PASS")
    print("    Point outside triangle: PASS")


def test_deformation():
    """Test with mesh deformation."""
    print("  Testing deformation interpolation...")

    orig_verts, tris = _create_test_mesh(10)
    new_verts = orig_verts + np.random.randn(*orig_verts.shape) * 0.01

    tri_indices, bary_coords = compute_barycentric_mapping(orig_verts, new_verts, tris)

    # Apply deformation
    deformed_new = new_verts.copy()
    deformed_new[:, 2] += 0.5

    # Interpolate back
    deformed_orig = interpolate_surface(deformed_new, tris, tri_indices, bary_coords)

    z_diff = deformed_orig[:, 2].mean() - orig_verts[:, 2].mean()
    assert abs(z_diff - 0.5) < 0.1, f"Unexpected Z displacement: {z_diff}"

    print("    Deformation interpolation: PASS")


def run_tests() -> bool:
    """Run all BVH tests. Returns True if all tests pass."""
    print("=" * 50)
    print("BVH Barycentric Mapping Tests")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_performance()
        test_edge_cases()
        test_deformation()
        print("\nAll BVH tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
