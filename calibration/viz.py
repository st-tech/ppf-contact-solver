# File: viz.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared, dependency-light rendering for the fabric calibration report:
# a numpy orthographic z-buffer rasterizer with flat Lambert shading, plus
# PIL helpers for overlays/labels and base64 embedding. Pure software (numpy +
# Pillow), so it runs headless on the CUDA host alongside the solver with no
# GPU/display. Used by calibration/report.py to render the top-down drape views
# and the side-on cantilever-bend views.

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    """A truetype font at *size* if one is findable, else PIL's default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10
    except Exception:
        return ImageFont.load_default()


# View definitions: how a world (x, y, z) point maps to screen (s, t) and a
# depth d (smaller d = nearer the camera), plus the world-space light direction
# used for shading. Gravity is -Y, so "top" looks straight down -Y.
def _project(V: np.ndarray, view: str):
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    if view == "top":        # look down -Y; screen = (X, Z)
        s, t, d = x, z, -y
        light = np.array([0.35, 0.85, 0.40])
    elif view == "side":     # look along -Z; screen = (X, Y)
        s, t, d = x, y, -z
        light = np.array([0.45, 0.55, 0.70])
    else:
        raise ValueError(f"unknown view: {view}")
    return s, t, d, light / np.linalg.norm(light)


def render_ortho(V: np.ndarray, F: np.ndarray, view: str, *,
                 w: int = 720, h: int = 560, margin: float = 0.12,
                 bounds=None, ambient: float = 0.35,
                 base_color=(168, 190, 232)):
    """Render mesh (V, F) orthographically with flat Lambert shading.

    Returns ``(image_uint8 HxWx3, to_px)`` where ``to_px(s, t)`` maps a point in
    the view's screen-plane world coordinates to pixel coordinates, so overlays
    (reference circles, angle lines) can be drawn in the same frame. ``bounds``
    optionally fixes the view extent ``(smin, smax, tmin, tmax)`` so several
    images share one scale; otherwise it auto-fits the mesh.
    """
    s, t, d, light = _project(V, view)

    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    # abs(): the draped sheet is two-sided, light either face.
    shade = ambient + (1.0 - ambient) * np.abs(n @ light)

    if bounds is None:
        smin, smax, tmin, tmax = s.min(), s.max(), t.min(), t.max()
    else:
        smin, smax, tmin, tmax = bounds
    sr = max(smax - smin, 1e-9)
    tr = max(tmax - tmin, 1e-9)
    scale = min((1 - 2 * margin) * w / sr, (1 - 2 * margin) * h / tr)
    cs, ct = (smin + smax) / 2, (tmin + tmax) / 2

    def to_px(sv, tv):
        return (w / 2 + (sv - cs) * scale, h / 2 - (tv - ct) * scale)

    PX = w / 2 + (s - cs) * scale
    PY = h / 2 - (t - ct) * scale

    img = np.full((h, w, 3), 255, np.uint8)
    zbuf = np.full((h, w), np.inf)
    bc = np.asarray(base_color, float)

    for fi in range(len(F)):
        i0, i1, i2 = F[fi]
        x0, y0 = PX[i0], PY[i0]
        x1, y1 = PX[i1], PY[i1]
        x2, y2 = PX[i2], PY[i2]
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        if maxx < minx or maxy < miny:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if den == 0:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx + 1) + 0.5,
                             np.arange(miny, maxy + 1) + 0.5)
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        dd = a * d[i0] + b * d[i1] + c * d[i2]
        sub = zbuf[miny:maxy + 1, minx:maxx + 1]
        closer = inside & (dd < sub)
        if not closer.any():
            continue
        col = np.clip(bc * shade[fi], 0, 255).astype(np.uint8)
        region = img[miny:maxy + 1, minx:maxx + 1]
        region[closer] = col
        sub[closer] = dd[closer]

    return img, to_px


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img, mode="RGB")


def data_uri(pil_img: Image.Image) -> str:
    """PNG -> base64 data URI for self-contained HTML embedding."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def draw_label(pil_img: Image.Image, xy, lines, *, size=20,
               color=(20, 20, 20), bg=(255, 255, 255, 210)):
    """Paste a small multi-line text box with a translucent background onto
    *pil_img* (an RGB image) at top-left ``xy``."""
    font = _font(size)
    x, y = xy
    pad = 6
    measure = ImageDraw.Draw(pil_img)
    widths, heights = [], []
    for ln in lines:
        bb = measure.textbbox((0, 0), ln, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1] + 6)
    box_w = max(widths) + 2 * pad
    box_h = sum(heights) + 2 * pad
    overlay = Image.new("RGBA", (box_w, box_h), bg)
    od = ImageDraw.Draw(overlay)
    cy = pad
    for ln, hh in zip(lines, heights):
        od.text((pad, cy), ln, fill=color, font=font)
        cy += hh
    pil_img.paste(overlay, (int(x), int(y)), overlay)
