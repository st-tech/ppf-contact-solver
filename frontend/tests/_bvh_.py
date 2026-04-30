# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for BVH frame mapping and surface reconstruction."""

import time

import numpy as np

from .._bvh_ import compute_frame_mapping, interpolate_surface


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
    """Points ON the surface must round-trip exactly (c3 ~ 0, in-plane coords
    recover the vertex)."""
    print("  Testing basic functionality...")

    verts, tris = _create_test_mesh(5)
    query_points = verts.copy()

    tri_indices, coefs = compute_frame_mapping(query_points, verts, tris)

    # A vertex that lies on the tet surface has zero normal offset.
    max_c3 = np.abs(coefs[:, 2]).max()
    assert max_c3 < 1e-9, f"On-surface c3 should be ~0, got {max_c3}"

    interpolated = interpolate_surface(verts, tris, tri_indices, coefs)
    max_error = np.abs(interpolated - query_points).max()
    assert max_error < 1e-9, f"Reconstruction error too large: {max_error}"

    print("    On-surface c3 ~ 0: PASS")
    print("    Reconstruction accuracy: PASS")


def test_round_trip_offset():
    """Points OFF the surface — the canonical shrink scenario. The original
    surface vertex sits at some distance from the tet triangle; the round-trip
    must reproduce that distance exactly."""
    print("  Testing off-surface round-trip...")

    verts, tris = _create_test_mesh(8)
    rng = np.random.default_rng(0)
    # Perturb along mesh-z by up to 0.05 — simulates original surface sitting
    # slightly above the tetrahedralized surface.
    query_points = verts.copy()
    query_points[:, 2] += rng.uniform(-0.05, 0.05, size=len(verts))

    tri_indices, coefs = compute_frame_mapping(query_points, verts, tris)
    interpolated = interpolate_surface(verts, tris, tri_indices, coefs)
    max_error = np.abs(interpolated - query_points).max()
    assert max_error < 1e-9, f"Off-surface round-trip error: {max_error}"

    print("    Off-surface round-trip: PASS")


def test_rigid_motion():
    """Under rigid translation+rotation of the tet surface, reconstructed
    points move by the same rigid transform."""
    print("  Testing rigid motion...")

    verts, tris = _create_test_mesh(8)
    rng = np.random.default_rng(1)
    query_points = verts.copy()
    query_points[:, 2] += rng.uniform(-0.05, 0.05, size=len(verts))

    tri_indices, coefs = compute_frame_mapping(query_points, verts, tris)

    # Rigid: 30deg rot about z, then translate.
    th = np.pi / 6
    R = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    t = np.array([0.7, -0.3, 1.2])
    deformed = verts @ R.T + t
    expected = query_points @ R.T + t

    interpolated = interpolate_surface(deformed, tris, tri_indices, coefs)
    max_error = np.abs(interpolated - expected).max()
    assert max_error < 1e-8, f"Rigid motion error: {max_error}"

    print("    Rigid motion: PASS")


def _create_flat_mesh(n: int = 8):
    """Flat xy grid (z=0) so every triangle has normal = +z exactly."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1).astype(
        np.float64
    )
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    return verts, np.array(tris, dtype=np.int32)


def test_shrink_regression():
    """Regression test for the shrink bug: under in-plane scaling of the tet
    surface, the absolute normal offset must be preserved. The pure-bary
    method shrinks it; the frame method does not."""
    print("  Testing shrink regression...")

    verts, tris = _create_flat_mesh(8)
    offset = 0.01
    query_points = verts.copy()
    query_points[:, 2] += offset  # pure normal offset on a flat mesh

    tri_indices, coefs = compute_frame_mapping(query_points, verts, tris)

    # c3 should equal the offset exactly (normal is +z, unit-length).
    assert np.allclose(coefs[:, 2], offset, atol=1e-12), (
        f"c3 mismatch on flat mesh: {coefs[:, 2].min()}..{coefs[:, 2].max()}"
    )

    # Shrink in-plane by 50%; normal stays +z unit.
    deformed = verts * np.array([0.5, 0.5, 1.0])
    interpolated = interpolate_surface(deformed, tris, tri_indices, coefs)

    # z offset relative to the deformed surface (still flat at z=0) must stay 0.01.
    assert np.allclose(interpolated[:, 2], offset, atol=1e-10), (
        f"Normal offset collapsed under scaling: "
        f"z range {interpolated[:, 2].min()}..{interpolated[:, 2].max()}"
    )
    # In-plane: follows the 0.5x scale.
    assert np.allclose(interpolated[:, :2], query_points[:, :2] * 0.5, atol=1e-10)

    print("    Normal offset preserved under scaling: PASS")
    print("    In-plane follows scaling: PASS")


def test_degenerate_triangle():
    """A zero-area triangle must not produce NaNs; the fallback zeros c and
    reconstructs x0'."""
    print("  Testing degenerate triangle...")

    # Collinear verts → zero-area triangle.
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    query = np.array([[0.5, 0.3, 0.1]], dtype=np.float64)

    tri_idx, coefs = compute_frame_mapping(query, verts, tris)
    assert np.all(np.isfinite(coefs)), "NaN/inf in degenerate coefs"
    assert np.allclose(coefs, 0.0), f"Degenerate coefs should be zero, got {coefs}"

    interpolated = interpolate_surface(verts, tris, tri_idx, coefs)
    assert np.all(np.isfinite(interpolated)), "NaN/inf in degenerate reconstruction"

    print("    Degenerate triangle fallback: PASS")


def test_performance():
    """Test performance with larger mesh."""
    print("  Testing performance...")

    verts, tris = _create_test_mesh(50)
    rng = np.random.default_rng(42)
    n_queries = 10000
    query_points = rng.random((n_queries, 3)).astype(np.float64)
    query_points[:, 0] *= 1.0
    query_points[:, 1] *= 1.0
    query_points[:, 2] = query_points[:, 2] * 0.4 - 0.1

    # Warm-up
    _ = compute_frame_mapping(query_points[:10], verts, tris)

    t0 = time.perf_counter()
    tri_indices, coefs = compute_frame_mapping(query_points, verts, tris)
    t1 = time.perf_counter()

    print(f"    Mapping {n_queries} points: {(t1-t0)*1000:.2f}ms")

    t0 = time.perf_counter()
    for _ in range(100):
        _ = interpolate_surface(verts, tris, tri_indices, coefs)
    t1 = time.perf_counter()
    print(f"    Interpolation (100 iters): {(t1-t0)*1000:.2f}ms")


def test_edge_cases():
    """Test edge cases."""
    print("  Testing edge cases...")

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)

    # Point at vertex: round-trip exact.
    query = np.array([[0, 0, 0]], dtype=np.float64)
    tri_idx, coefs = compute_frame_mapping(query, verts, tris)
    interpolated = interpolate_surface(verts, tris, tri_idx, coefs)
    assert np.allclose(interpolated[0], [0, 0, 0]), "Wrong interpolation at vertex"

    # Point above triangle plane: c3 should equal the normal-distance.
    query = np.array([[0.25, 0.25, 0.1]], dtype=np.float64)
    tri_idx, coefs = compute_frame_mapping(query, verts, tris)
    assert abs(coefs[0, 2] - 0.1) < 1e-12, (
        f"Expected c3=0.1 for point 0.1 above xy-plane, got {coefs[0, 2]}"
    )

    # Point outside triangle (in-plane): the in-plane coords extrapolate,
    # and the round-trip is exact on the rest mesh.
    query = np.array([[2.0, 2.0, 0.0]], dtype=np.float64)
    tri_idx, coefs = compute_frame_mapping(query, verts, tris)
    interpolated = interpolate_surface(verts, tris, tri_idx, coefs)
    assert np.allclose(interpolated[0], query[0]), (
        f"Extrapolation round-trip failed: {interpolated[0]}"
    )

    print("    Point at vertex: PASS")
    print("    Normal offset recovered: PASS")
    print("    Extrapolation round-trip: PASS")


def test_deformation():
    """Test with mesh deformation (translation along z should transfer
    to reconstructed vertices unchanged)."""
    print("  Testing deformation interpolation...")

    orig_verts, tris = _create_test_mesh(10)
    rng = np.random.default_rng(3)
    new_verts = orig_verts + rng.standard_normal(orig_verts.shape) * 0.01

    tri_indices, coefs = compute_frame_mapping(orig_verts, new_verts, tris)

    deformed_new = new_verts.copy()
    deformed_new[:, 2] += 0.5

    deformed_orig = interpolate_surface(deformed_new, tris, tri_indices, coefs)

    z_diff = deformed_orig[:, 2].mean() - orig_verts[:, 2].mean()
    assert abs(z_diff - 0.5) < 0.1, f"Unexpected Z displacement: {z_diff}"

    print("    Deformation interpolation: PASS")


def run_tests() -> bool:
    """Run all BVH tests. Returns True if all tests pass."""
    print("=" * 50)
    print("BVH Frame Mapping Tests")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_round_trip_offset()
        test_rigid_motion()
        test_shrink_regression()
        test_degenerate_triangle()
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
