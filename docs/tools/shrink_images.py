#!/usr/bin/env python3
"""Shrink oversized PNG screenshots to <=60KB while keeping them readable.

Strategy per file:
  1. If the file is already <=60KB: skip.
  2. Try palette quantization at 256/192/128/96/64/48/32 colors (Floyd-Steinberg
     dither). Keep the result with the highest color count that fits under
     60KB.
  3. If palette alone is not enough, resize to 90%, 80%, 70%, 60%, 50% of
     original width (Lanczos) then re-run the palette cascade.
  4. Write the best candidate back to the same filename.
"""
from __future__ import annotations

import io
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "blender_addon" / "images"
MAX_BYTES = 60 * 1024
COLOR_STEPS = [256, 192, 128, 96, 64, 48, 32, 24, 16]
SCALE_STEPS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]


def encode_png_palette(img: Image.Image, colors: int) -> bytes:
    working = img
    if working.mode == "RGBA":
        bg = Image.new("RGB", working.size, (255, 255, 255))
        bg.paste(working, mask=working.split()[-1])
        working = bg
    elif working.mode not in ("RGB", "L"):
        working = working.convert("RGB")
    q = working.quantize(
        colors=colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.FLOYDSTEINBERG,
    )
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def find_best(img: Image.Image) -> tuple[bytes, str]:
    w0, h0 = img.size
    for scale in SCALE_STEPS:
        if scale == 1.0:
            work = img
        else:
            work = img.resize(
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                Image.LANCZOS,
            )
        for colors in COLOR_STEPS:
            data = encode_png_palette(work, colors)
            size = len(data)
            if size <= MAX_BYTES:
                return data, f"scale={scale:.2f} colors={colors} size={size/1024:.1f}KB"
    scale = SCALE_STEPS[-1]
    work = img.resize(
        (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
        Image.LANCZOS,
    )
    data = encode_png_palette(work, COLOR_STEPS[-1])
    return data, f"fallback scale={scale:.2f} colors={COLOR_STEPS[-1]} size={len(data)/1024:.1f}KB"


def main() -> int:
    changed = 0
    skipped = 0
    for p in sorted(ROOT.rglob("*.png")):
        before = p.stat().st_size
        if before <= MAX_BYTES:
            skipped += 1
            continue
        with Image.open(p) as im:
            im.load()
            data, label = find_best(im)
        p.write_bytes(data)
        after = p.stat().st_size
        print(f"{p.relative_to(ROOT)}: {before/1024:.1f}KB -> {after/1024:.1f}KB  [{label}]")
        changed += 1
    print(f"\n{changed} file(s) shrunk; {skipped} already under {MAX_BYTES/1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
