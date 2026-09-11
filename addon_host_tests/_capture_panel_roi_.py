# File: addon_host_tests/_capture_panel_roi_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gate on the coordinate space of the capture ROI
# (``blender_addon/ui/capture.py``).
#
# ``WidgetLocator.panel_bbox`` returns the rectangle every widget diff is
# restricted to. Region coordinates are device pixels measured from the
# window's BOTTOM; a screenshot is measured from its TOP; so the rect is
# built by subtracting the region's extent from the image height. Getting
# that height from ``window.height * pixel_size`` is what this guards
# against: the product is right only where the window reports LOGICAL
# points against a scaled framebuffer (a Retina Mac) and wrong wherever
# the window already reports device pixels while Blender still infers a
# pixel_size above 1 — an Xvfb sized 1920x1800 reports pixel_size 2.0 and
# window.height 1800, so the product is 3600 and every ROI lands entirely
# below the image.
#
# That failure is silent in the worst way. An off-image ROI makes each
# widget diff come back empty, and the caller reports the widget as "not
# rendered in the current panel state" — a message about the panel, for
# what is really a unit mismatch. Docs screenshots then regenerate with
# every widget skipped and no error anywhere.
#
# The fix is to measure rather than predict: read the height out of the
# baseline PNG, which is by construction the same image the diff runs on.

from __future__ import annotations

import struct
import types
import zlib

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def cap():
    return load_addon_module("ui.capture")


def _write_png(path, width: int, height: int) -> None:
    """Write a minimal, valid greyscale PNG of the given size."""
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _install_window(cap, *, window_height: int, pixel_size: float,
                    region) -> None:
    """Point the module's stub ``bpy`` at one window holding *region*."""
    area = types.SimpleNamespace(type="VIEW_3D", regions=[region])
    window = types.SimpleNamespace(
        height=window_height, width=1920,
        screen=types.SimpleNamespace(areas=[area]),
    )
    cap.bpy.context = types.SimpleNamespace(
        window_manager=types.SimpleNamespace(windows=[window]),
        preferences=types.SimpleNamespace(
            system=types.SimpleNamespace(pixel_size=pixel_size)
        ),
    )
    cap.bpy.types.MOCK_PT_Panel = types.SimpleNamespace(
        bl_space_type="VIEW_3D", bl_region_type="UI",
    )


def test_png_size_reads_header(cap, tmp_path):
    png = tmp_path / "frame.png"
    _write_png(png, 1920, 1800)
    assert cap._png_size(str(png)) == (1920, 1800)


def test_png_size_rejects_non_png(cap, tmp_path):
    junk = tmp_path / "frame.png"
    junk.write_bytes(b"not a png at all, but long enough to read 24 bytes")
    with pytest.raises(ValueError):
        cap._png_size(str(junk))


def test_roi_stays_inside_the_image_when_pixel_size_is_two(cap, tmp_path):
    """The HiDPI-reported-but-device-sized case that broke Linux capture.

    window.height already equals the framebuffer height, so multiplying it
    by pixel_size would put the whole rect below the bottom of the frame.
    """
    baseline = tmp_path / "baseline.png"
    _write_png(baseline, 1920, 1800)
    region = types.SimpleNamespace(type="UI", x=990, y=176, width=584,
                                   height=1461)
    _install_window(cap, window_height=1800, pixel_size=2.0, region=region)

    loc = cap.WidgetLocator(str(baseline), str(tmp_path / "scratch"))
    left, top, right, bottom = loc.panel_bbox("MOCK_PT_Panel")

    assert (left, right) == (990, 1574)
    # Flipped about the image height, not about twice it.
    assert (top, bottom) == (1800 - (176 + 1461), 1800 - 176)
    assert 0 <= top < bottom <= 1800, "ROI must land inside the frame"


def test_roi_ignores_pixel_size_entirely(cap, tmp_path):
    """Same window and region, different pixel_size, same rectangle.

    The image is the only authority on the image's size, so a pixel_size
    the platform reports differently must not move the ROI.
    """
    baseline = tmp_path / "baseline.png"
    _write_png(baseline, 1920, 1800)
    region = types.SimpleNamespace(type="UI", x=400, y=100, width=600,
                                   height=1200)

    seen = set()
    for pixel_size in (1.0, 2.0, 3.0):
        _install_window(cap, window_height=1800, pixel_size=pixel_size,
                        region=region)
        loc = cap.WidgetLocator(str(baseline), str(tmp_path / "scratch"))
        seen.add(tuple(loc.panel_bbox("MOCK_PT_Panel")))

    assert len(seen) == 1, f"pixel_size moved the ROI: {seen}"
