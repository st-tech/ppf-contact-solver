# File: _rasterizer_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for the software rasterizer."""

import numpy as np

from .._rasterizer_ import SoftwareRenderer


def test_triangle_produces_coverage():
    """Rasterizing a single colored triangle into a small framebuffer
    must produce a non-zero coverage region (i.e. some pixels are no
    longer the white background)."""
    print("  Testing triangle rasterization coverage...")

    width, height = 64, 48
    renderer = SoftwareRenderer({"width": width, "height": height})

    # Triangle in the canonical [-0.5, 0.5] cube spanned by the renderer's
    # auto-fit pipeline, with a vivid red color so it stands out from the
    # white background.
    vert = np.array(
        [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]],
        dtype=np.float32,
    )
    color = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    seg = np.zeros((0, 2), dtype=np.int32)
    face = np.array([[0, 1, 2]], dtype=np.int32)

    img = renderer.render(vert, color, seg, face, output=None)

    assert img is not None, "render(output=None) must return an array"
    assert img.shape == (height, width, 4), (
        f"Expected ({height},{width},4) RGBA, got {img.shape}"
    )
    assert img.dtype == np.uint8, f"Expected uint8 image, got {img.dtype}"

    # Pixels that aren't pure white background indicate the triangle was
    # drawn somewhere. White background is (255,255,255,255).
    rgb = img[..., :3]
    non_bg = np.any(rgb != 255, axis=-1)
    coverage = int(non_bg.sum())
    assert coverage > 0, "Triangle produced zero non-background pixels"

    # A red triangle should give us at least some pixels where the red
    # channel exceeds the green channel.
    red_dominant = (rgb[..., 0] > rgb[..., 1]) & non_bg
    assert int(red_dominant.sum()) > 0, "No red-dominant pixels found"

    print(
        f"    Triangle coverage: {coverage} pixels, "
        f"red-dominant: {int(red_dominant.sum())}: PASS"
    )


def run_tests() -> bool:
    """Run all rasterizer tests. Returns True if all tests pass."""
    print("=" * 50)
    print("Software Rasterizer Tests")
    print("=" * 50)

    try:
        test_triangle_produces_coverage()
        print("\nAll rasterizer tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
