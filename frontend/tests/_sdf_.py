# File: _sdf_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for SDF marching-cubes round-trip."""

import numpy as np

from .._sdf_ import SphereSDF, marching_cubes


def test_sphere_marching_cubes_closed_mesh():
    """Marching cubes on a unit sphere SDF must produce a non-empty,
    non-degenerate triangle mesh."""
    print("  Testing sphere marching cubes round-trip...")

    sphere = SphereSDF(radius=1.0, center=(0.0, 0.0, 0.0))
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    step = 0.2

    verts, faces = marching_cubes(sphere, bounds, step)

    assert verts.shape[0] > 0, f"Expected vertices, got shape {verts.shape}"
    assert faces.shape[0] > 0, f"Expected faces, got shape {faces.shape}"
    assert verts.shape[1] == 3, f"Expected (N,3) vertices, got {verts.shape}"
    assert faces.shape[1] == 3, f"Expected (M,3) faces, got {faces.shape}"

    # No degenerate triangles: each tri's three indices must be distinct.
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    assert np.all(a != b), "Found triangle with repeated index a==b"
    assert np.all(b != c), "Found triangle with repeated index b==c"
    assert np.all(a != c), "Found triangle with repeated index a==c"

    # All indices in range.
    assert faces.min() >= 0, f"Negative face index: {faces.min()}"
    assert faces.max() < verts.shape[0], (
        f"Face index {faces.max()} exceeds vertex count {verts.shape[0]}"
    )

    # Vertices must lie near the sphere surface (radius ~ 1.0).
    radii = np.linalg.norm(verts, axis=1)
    assert radii.min() > 0.7, f"Vertex too far inside sphere: {radii.min()}"
    assert radii.max() < 1.3, f"Vertex too far outside sphere: {radii.max()}"

    print(
        f"    Sphere mesh: {verts.shape[0]} verts, {faces.shape[0]} tris, "
        f"radius range {radii.min():.3f}..{radii.max():.3f}: PASS"
    )


def run_tests() -> bool:
    """Run all SDF tests. Returns True if all tests pass."""
    print("=" * 50)
    print("SDF Marching Cubes Tests")
    print("=" * 50)

    try:
        test_sphere_marching_cubes_closed_mesh()
        print("\nAll SDF tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
