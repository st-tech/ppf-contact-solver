#!/usr/bin/env python3
# File: report.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Generate a GitHub-renderable Markdown report (and a browser-viewable HTML twin)
# for the fabric presets: for each SHELL preset it runs the Cusick drape test and
# the cantilever bending test, renders labeled images, and writes report.md +
# report.html plus a report_images/ subdirectory of PNGs. Both report.md and
# report.html link the SAME image files (no base64): report.md renders natively
# on GitHub; report.html opens in any web browser.
#
# This is the visual, reviewable counterpart to the numeric calibration: the
# drape images show the draped shadow + the buckled folds (with the drape
# coefficient and observed fold count labeled), and the bending images show the
# cantilever droop with its measured angle. Both the drape DC and the bending
# angle measure BENDING relative to weight (a fabric barely stretches under its
# own weight), not in-plane stiffness.
#
# Images are palette-quantized to stay under 32 KiB each (these flat-shaded
# renders have very few colors, so quantization is near-lossless).
#
# CUDA only (the underlying sims need a GPU). Usage (from worktree root):
#   PYTHONPATH=. python calibration/report.py            # all fabrics
#   PYTHONPATH=. python calibration/report.py --fabric Cotton --fabric Silk

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cusick_drape"))
sys.path.insert(0, os.path.join(_HERE, "cantilever_bend"))
import viz  # noqa: E402
import cusick_drape as cd  # noqa: E402
import cantilever_bend as cb  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

_REPORT_PATH = os.path.join(_HERE, "report.md")
_REPORT_HTML_PATH = os.path.join(_HERE, "report.html")
_IMAGES_DIR = os.path.join(_HERE, "report_images")
# Intermediate per-fabric simulation cache (settled meshes + measured metrics).
# Persisted so the report can be re-styled/re-laid-out via --render-only without
# re-running the GPU sims. Local-only (gitignored); fetched from the CUDA host.
_DATA_DIR = os.path.join(_HERE, "report_data")
_MAX_IMAGE_BYTES = 32 * 1024


def _rot(ax_deg, ay_deg):
    ax, ay = math.radians(ax_deg), math.radians(ay_deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx


def _polyline(draw, to_px, pts, **kw):
    px = [to_px(s, t) for s, t in pts]
    draw.line(px, **kw)


def _world_circle(draw, to_px, radius, *, n=180, **kw):
    pts = [(radius * math.cos(2 * math.pi * i / n),
            radius * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
    _polyline(draw, to_px, pts, **kw)


# ---------------------------------------------------------------------------
# Image compression: palette-quantize each PNG under the size cap
# ---------------------------------------------------------------------------

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def save_image(pil_img: Image.Image, path: str,
               max_bytes: int = _MAX_IMAGE_BYTES) -> int:
    """Write *pil_img* as the smallest acceptable PNG.

    These renders are flat-shaded with few distinct colors, so adaptive-palette
    quantization is near-lossless and always smaller than 24-bit RGB; we always
    apply it and keep the smaller result. Use the most colors (highest quality)
    that fits *max_bytes*, stepping the palette down only as needed, and
    downscale only as a last resort. Returns the final file size in bytes.
    """
    rgb = pil_img.convert("RGB")
    best = _png_bytes(rgb)  # 24-bit baseline
    for colors in (256, 192, 128, 96, 64, 48, 32):
        cand = _png_bytes(rgb.quantize(colors=colors, method=Image.FASTOCTREE))
        if len(cand) < len(best):
            best = cand          # quantization always wins for these images
        if len(best) <= max_bytes:
            break
    if len(best) > max_bytes:    # last resort: downscale + quantize
        w, h = rgb.size
        scale = 0.85
        while len(best) > max_bytes and scale > 0.3:
            small = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            cand = _png_bytes(small.quantize(colors=64, method=Image.FASTOCTREE))
            if len(cand) < len(best):
                best = cand
            scale *= 0.85
    with open(path, "wb") as f:
        f.write(best)
    return len(best)


# ---------------------------------------------------------------------------
# Drape: top-down (with reference circles) + oblique (to show folds)
# ---------------------------------------------------------------------------

def render_drape(name: str, sim: dict, target) -> dict:
    V, F = sim["vert"], sim["face"]
    R = cd.R_SPECIMEN
    bounds = (-R * 1.08, R * 1.08, -R * 1.08, R * 1.08)

    top, to_px = viz.render_ortho(V, F, "top", w=560, h=560, bounds=bounds)
    timg = viz.to_pil(top)
    d = ImageDraw.Draw(timg)
    _world_circle(d, to_px, cd.R_SPECIMEN, fill=(120, 120, 120), width=1)
    _world_circle(d, to_px, cd.R_SUPPORT, fill=(200, 90, 90), width=2)
    band = f"{target[0]:.0f}-{target[1]:.0f}%" if target else "n/a"
    viz.draw_label(timg, (10, 10), [
        f"{name}  (top view)",
        f"Drape coefficient: {sim['dc']:.1f}%   target {band}",
        f"Folds (nodes): {sim['fold_k']}",
    ], size=19)
    viz.draw_label(timg, (10, 560 - 58), [
        "red = 9 cm support disc",
        "gray = 15 cm specimen rim",
    ], size=15, color=(70, 70, 70))

    Vob = V @ _rot(-58, 26).T
    ob, _ = viz.render_ortho(Vob, F, "top", w=560, h=560)
    oimg = viz.to_pil(ob)
    viz.draw_label(oimg, (10, 10), [f"{name}  (oblique view)"], size=19)

    return {
        "top_img": timg, "oblique_img": oimg,
        "dc": sim["dc"], "fold_k": sim["fold_k"], "target": target,
    }


# ---------------------------------------------------------------------------
# Bend: side-on cantilever with the droop angle drawn
# ---------------------------------------------------------------------------

def render_bend(name: str, r: dict) -> dict:
    V, F = r["vert"], r["face"]
    # Frame to the strip + its droop, not a fixed tall window, so low-droop
    # strips are not rendered tiny.
    oh = max(r["overhang"], 1e-3)
    bounds = (r["xmin"], r["xmax"], -(oh * 1.08), oh * 0.20)
    # Faint shaded strip behind; the droop is read from the bold centerline so
    # the fabric's width-curl (real anticlastic curvature) does not clutter the
    # edge-on profile.
    side, to_px = viz.render_ortho(V, F, "side", w=720, h=440, bounds=bounds,
                                   base_color=(225, 230, 238), ambient=0.78)
    img = viz.to_pil(side)
    d = ImageDraw.Draw(img)
    cx = r["clamp_x"]
    cl = r["centerline"]
    cpts = [to_px(V[idx, 0], V[idx, 1]) for idx in cl]
    if len(cpts) >= 2:
        d.line(cpts, fill=(40, 70, 150), width=4)
    _polyline(d, to_px, [(cx, 0.0), (r["xmax"], 0.0)],
              fill=(150, 150, 150), width=2)
    _polyline(d, to_px, [(cx, 0.0), (r["tip"][0], r["tip"][1])],
              fill=(200, 90, 90), width=2)
    p0 = to_px(cx, 0.0)
    d.ellipse([p0[0] - 5, p0[1] - 5, p0[0] + 5, p0[1] + 5], fill=(30, 30, 30))
    viz.draw_label(img, (10, 10), [
        f"{name}  (cantilever, side view)",
        f"Tip droop angle: {r['angle_deg']:.1f} deg",
        f"overhang {r['overhang'] * 100:.1f} cm, drop {r['drop'] * 100:.1f} cm",
    ], size=19)
    viz.draw_label(img, (10, 440 - 40), [
        "blue = centerline,  gray = horizontal,  red = clamp-to-tip chord",
    ], size=15, color=(70, 70, 70))
    return {"side_img": img, "angle": r["angle_deg"], "overhang": r["overhang"]}


# ---------------------------------------------------------------------------
# Markdown assembly (relative image links render on GitHub)
# ---------------------------------------------------------------------------

def build_markdown(cards, meta, img_rel) -> str:
    L = []
    L.append("# Fabric preset calibration report")
    L.append("")
    L.append("How each bundled fabric preset behaves: how much it drapes, how "
             "much it droops under its own weight, and how easily it stretches. "
             "Each fabric is shown with simulated images and its parameters.")
    L.append("")
    L.append("- **Drape**: a round fabric sheet hangs over a smaller disc. The "
             "drape coefficient is how much of its flat area the shadow still "
             "covers: a soft, drapey fabric collapses into deep folds and covers "
             "little (low %), while a stiff fabric stays spread out (high %). The "
             "values match published measurements for each fabric. The fold count "
             "is the number of waves around the rim (approximate).")
    L.append(f"- **Bending**: a {meta['strip_cm']:.0f} cm strip is held at one end "
             "and droops under gravity. The tip droop angle shows bending "
             "stiffness: a stiffer fabric droops less.")
    L.append("- **Stretch**: the young-mod and Poisson values describe how the "
             "fabric stretches in its plane. Silk and wool stretch more easily; "
             "denim and leather barely stretch.")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Fabric | Drape DC % | Target % | Folds | Bend droop (deg) | "
             "bend | young-mod | Poisson |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in cards:
        t = c["drape"]["target"]
        tband = f"{t[0]:.0f}-{t[1]:.0f}" if t else "-"
        L.append(f"| {c['name']} | {c['drape']['dc']:.1f} | {tband} | "
                 f"{c['drape']['fold_k']} | {c['bend']['angle']:.1f} | "
                 f"{c['preset']['bend']} | {c['preset']['shell_young_modulus']:.0f} | "
                 f"{c['preset']['shell_poisson_ratio']} |")
    L.append("")
    for c in cards:
        nm, dr, bd = c["name"], c["drape"], c["bend"]
        rel = img_rel[nm]
        L.append(f"## {nm}")
        L.append("")
        L.append(f"Drape coefficient **{dr['dc']:.1f}%** &nbsp;|&nbsp; observed "
                 f"folds **{dr['fold_k']}** &nbsp;|&nbsp; cantilever tip droop "
                 f"**{bd['angle']:.1f} deg** &nbsp;|&nbsp; bend "
                 f"`{c['preset']['bend']}`, young-mod "
                 f"`{c['preset']['shell_young_modulus']:.0f}`, Poisson "
                 f"`{c['preset']['shell_poisson_ratio']}`")
        L.append("")
        L.append(f"![{nm} drape, top view]({rel['top']})")
        L.append("")
        L.append("Looking straight down at the draped fabric. The shaded area is "
                 "its shadow (the drape coefficient); the wavy edge is its folds.")
        L.append("")
        L.append(f"![{nm} drape, oblique view]({rel['oblique']})")
        L.append("")
        L.append("The same drape seen at an angle, showing the folds.")
        L.append("")
        L.append(f"![{nm} cantilever bend]({rel['side']})")
        L.append("")
        L.append("A strip held at the left, drooping under gravity. The blue line "
                 "is its centerline; the angle it makes with horizontal (gray) is "
                 "the droop.")
        L.append("")
    L.append("---")
    L.append("")
    L.append("Drape coefficients follow the Cusick method (BS 5058 / ISO 9073-9); "
             "cantilever bending follows ASTM D1388. Per-fabric reference values "
             "and citations are in the `calibration/` folder.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML assembly (browser view; links the SAME report_images/ files, not base64)
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 margin:0;background:#f4f5f7;color:#1c1e21}
header{background:#23272e;color:#fff;padding:22px 32px}
header h1{margin:0 0 8px;font-size:22px}
header p,header li{color:#c8cdd4;font-size:13px;line-height:1.5}
header ul{margin:6px 0;max-width:1000px}
.wrap{padding:24px 32px}
table.summary{border-collapse:collapse;margin:8px 0 24px;font-size:13px;background:#fff}
table.summary th,table.summary td{border:1px solid #d9dce1;padding:6px 12px;text-align:right}
table.summary th{background:#eef0f3}
table.summary td:first-child,table.summary th:first-child{text-align:left}
.card{background:#fff;border:1px solid #e0e3e8;border-radius:8px;margin:0 0 22px;
 padding:16px 18px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.card h2{margin:0 0 8px;font-size:18px}
.row{display:flex;flex-wrap:wrap;gap:18px}
.fig{flex:1 1 320px}
.fig img{width:100%;height:auto;border:1px solid #e6e8ec;border-radius:6px;background:#fff}
.fig .cap{font-size:12px;color:#555;margin-top:6px}
.metrics{font-size:13px;color:#333;margin:6px 0 12px}
.metrics b{color:#111}
footer{padding:18px 32px;color:#666;font-size:12px;line-height:1.6}
code{background:#eceef1;padding:1px 4px;border-radius:3px}
"""


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(cards, meta, img_rel) -> str:
    rows = ["<tr><th>Fabric</th><th>Drape DC %</th><th>Target %</th><th>Folds</th>"
            "<th>Bend droop</th><th>bend</th><th>young-mod</th><th>Poisson</th></tr>"]
    for c in cards:
        t = c["drape"]["target"]
        tband = f"{t[0]:.0f}-{t[1]:.0f}" if t else "-"
        rows.append(
            f"<tr><td>{_esc(c['name'])}</td><td>{c['drape']['dc']:.1f}</td>"
            f"<td>{tband}</td><td>{c['drape']['fold_k']}</td>"
            f"<td>{c['bend']['angle']:.1f}&deg;</td>"
            f"<td>{c['preset']['bend']}</td>"
            f"<td>{c['preset']['shell_young_modulus']:.0f}</td>"
            f"<td>{c['preset']['shell_poisson_ratio']}</td></tr>")
    summary = "<table class='summary'>" + "".join(rows) + "</table>"

    cards_html = []
    for c in cards:
        nm, dr, bd = c["name"], c["drape"], c["bend"]
        rel = img_rel[nm]
        cards_html.append(f"""
<div class="card">
  <h2>{_esc(nm)}</h2>
  <div class="metrics">
    Drape coefficient <b>{dr['dc']:.1f}%</b>
    &nbsp;|&nbsp; observed folds <b>{dr['fold_k']}</b>
    &nbsp;|&nbsp; cantilever tip droop <b>{bd['angle']:.1f}&deg;</b>
    &nbsp;|&nbsp; bend <code>{c['preset']['bend']}</code>,
    young-mod <code>{c['preset']['shell_young_modulus']:.0f}</code>,
    Poisson <code>{c['preset']['shell_poisson_ratio']}</code>
  </div>
  <div class="row">
    <div class="fig"><img src="{rel['top']}" alt="{_esc(nm)} drape top"/>
      <div class="cap">Looking straight down at the draped fabric. The shaded
      area is its shadow (the drape coefficient); the wavy edge is its folds.</div></div>
    <div class="fig"><img src="{rel['oblique']}" alt="{_esc(nm)} drape oblique"/>
      <div class="cap">The same drape seen at an angle, showing the folds.</div></div>
    <div class="fig"><img src="{rel['side']}" alt="{_esc(nm)} cantilever"/>
      <div class="cap">A strip held at the left, drooping under gravity. The blue
      line is its centerline; its angle from horizontal is the droop.</div></div>
  </div>
</div>""")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fabric preset calibration report</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>Fabric preset calibration report</h1>
  <p>How each bundled fabric preset behaves: how much it drapes, how much it
  droops under its own weight, and how easily it stretches. Each fabric is shown
  with simulated images and its parameters.</p>
  <ul>
    <li><b>Drape</b>: a round fabric sheet hangs over a smaller disc. The drape
    coefficient is how much of its flat area the shadow still covers: a soft,
    drapey fabric collapses into deep folds and covers little (low&nbsp;%), while
    a stiff fabric stays spread out (high&nbsp;%). The values match published
    measurements for each fabric. The fold count is the number of waves around
    the rim (approximate).</li>
    <li><b>Bending</b>: a {meta['strip_cm']:.0f}&nbsp;cm strip is held at one end
    and droops under gravity. The tip droop angle shows bending stiffness: a
    stiffer fabric droops less.</li>
    <li><b>Stretch</b>: the young-mod and Poisson values describe how the fabric
    stretches in its plane. Silk and wool stretch more easily; denim and leather
    barely stretch.</li>
  </ul>
</header>
<div class="wrap">
{summary}
{"".join(cards_html)}
</div>
<footer>
  Drape coefficients follow the Cusick method (BS 5058 / ISO 9073-9); cantilever
  bending follows ASTM D1388. Per-fabric reference values and citations are in
  the calibration folder.
</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Simulation cache (intermediate format): run the GPU sims once, save the
# settled meshes + measured metrics, so the report can be re-styled offline.
# ---------------------------------------------------------------------------

def simulate_fabric(name: str, preset: dict, *, drape_frames: int,
                    bend_frames: int, dt: float) -> dict:
    """Run the drape + cantilever sims (CUDA) and return a flat, savez-able dict
    of the settled meshes and measured metrics (the intermediate cache record)."""
    sim = cd.simulate_drape(name, preset, frames=drape_frames, dt=dt,
                            rings=cd.DEFAULT_RINGS, seg=cd.DEFAULT_SEG,
                            grid=cd.DEFAULT_GRID)
    bend = cb.simulate_bend(name, preset, frames=bend_frames, dt=dt)
    return {
        "drape_vert": sim["vert"], "drape_face": sim["face"],
        "dc": np.float64(sim["dc"]), "fold_k": np.int64(sim["fold_k"]),
        "bend_vert": bend["vert"], "bend_face": bend["face"],
        "bend_rest": bend["rest"],
        "bend_centerline": np.asarray(bend["centerline"], dtype=np.int64),
        "clamp_x": np.float64(bend["clamp_x"]), "xmin": np.float64(bend["xmin"]),
        "xmax": np.float64(bend["xmax"]),
        "tip": np.asarray(bend["tip"], dtype=np.float64),
        "drop": np.float64(bend["drop"]),
        "overhang": np.float64(bend["overhang"]),
        "angle_deg": np.float64(bend["angle_deg"]),
    }


def render_fabric(name: str, data: dict, target) -> tuple:
    """Render the drape + bend images from a cached data record (no GPU)."""
    sim = {"vert": data["drape_vert"], "face": data["drape_face"],
           "dc": float(data["dc"]), "fold_k": int(data["fold_k"])}
    bend = {"vert": data["bend_vert"], "face": data["bend_face"],
            "rest": data["bend_rest"],
            "centerline": np.asarray(data["bend_centerline"]).tolist(),
            "clamp_x": float(data["clamp_x"]), "xmin": float(data["xmin"]),
            "xmax": float(data["xmax"]), "tip": np.asarray(data["tip"]),
            "drop": float(data["drop"]), "overhang": float(data["overhang"]),
            "angle_deg": float(data["angle_deg"])}
    return render_drape(name, sim, target), render_bend(name, bend)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the fabric calibration "
                                 "report (drape + bending) to Markdown + images. "
                                 "Caches the sim meshes so the report can be "
                                 "re-styled with --render-only (no GPU).")
    ap.add_argument("--fabric", action="append",
                    help="Restrict to these fabrics (repeatable). Default: all.")
    ap.add_argument("--render-only", action="store_true",
                    help="Re-render report.md + images from the cached sim data "
                    "in --data-dir; do NOT run the GPU sims (works without a GPU).")
    ap.add_argument("--drape-frames", type=int, default=cd.DEFAULT_FRAMES)
    ap.add_argument("--bend-frames", type=int, default=cb.DEFAULT_FRAMES)
    ap.add_argument("--dt", type=float, default=cd.DEFAULT_DT)
    ap.add_argument("--out", default=_REPORT_PATH, help="Markdown output path.")
    ap.add_argument("--html-out", default=_REPORT_HTML_PATH,
                    help="HTML output path (browser view; same images).")
    ap.add_argument("--images-dir", default=_IMAGES_DIR)
    ap.add_argument("--data-dir", default=_DATA_DIR)
    args = ap.parse_args(argv)

    presets = cd.load_presets()
    targets = cd.load_targets()
    names = args.fabric or cd.shell_fabrics(presets)
    os.makedirs(args.images_dir, exist_ok=True)
    img_dirname = os.path.basename(args.images_dir.rstrip("/"))
    meta_path = os.path.join(args.data_dir, "meta.json")

    if args.render_only and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
        meta = {"drape_frames": args.drape_frames,
                "bend_frames": args.bend_frames, "dt": args.dt,
                "strip_cm": cb.STRIP_LENGTH * 100}
    if not args.render_only:
        os.makedirs(args.data_dir, exist_ok=True)

    cards, img_rel = [], {}
    for name in names:
        if name not in presets:
            print(f"skip unknown preset: {name}", file=sys.stderr)
            continue
        dpath = os.path.join(args.data_dir, f"{name}.npz")
        if args.render_only:
            if not os.path.exists(dpath):
                print(f"skip {name}: no cached data at {dpath}", file=sys.stderr)
                continue
            data = dict(np.load(dpath))
            print(f"[{name}] render from cache", flush=True)
        else:
            print(f"[{name}] simulate...", flush=True)
            data = simulate_fabric(name, presets[name],
                                   drape_frames=args.drape_frames,
                                   bend_frames=args.bend_frames, dt=args.dt)
            np.savez_compressed(dpath, **data)
            print(f"[{name}] DC={float(data['dc']):.1f}% "
                  f"folds={int(data['fold_k'])} "
                  f"droop={float(data['angle_deg']):.1f}deg "
                  f"-> cached {os.path.basename(dpath)}", flush=True)

        dr, bd = render_fabric(name, data, targets.get(name))
        rel = {}
        for key, img in (("top", dr["top_img"]), ("oblique", dr["oblique_img"]),
                         ("side", bd["side_img"])):
            fname = f"{name}_{key}.png"
            nbytes = save_image(img, os.path.join(args.images_dir, fname))
            rel[key] = f"{img_dirname}/{fname}"
            print(f"  {fname}: {nbytes / 1024:.1f} KiB", flush=True)
        img_rel[name] = rel
        cards.append({"name": name, "preset": presets[name],
                      "drape": dr, "bend": bd})

    if not args.render_only:
        json.dump(meta, open(meta_path, "w"), indent=2)
    with open(args.out, "w") as f:
        f.write(build_markdown(cards, meta, img_rel))
    with open(args.html_out, "w") as f:
        f.write(build_html(cards, meta, img_rel))
    print(f"\nwrote {args.out} + {args.html_out} + {len(cards) * 3} images in "
          f"{args.images_dir}"
          + ("" if args.render_only else f"; sim cache in {args.data_dir}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
