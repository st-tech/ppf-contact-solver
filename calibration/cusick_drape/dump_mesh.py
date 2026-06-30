#!/usr/bin/env python3
# File: dump_mesh.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Run one fabric's Cusick drape and save the settled mesh to an .npz, so the
# draped geometry can be rendered/inspected offline (e.g. on a non-GPU host).
# This is the inspection counterpart to cusick_drape.py's area-only DC output:
# it lets us SEE whether the specimen actually buckles into folds (it must, from
# hoop compression of the overhang) rather than assuming the drape is smooth.
#
# Usage (CUDA host, from worktree root):
#   PYTHONPATH=. python calibration/cusick_drape/dump_mesh.py --fabric Cotton \
#       --out /tmp/cotton.npz

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cusick_drape as cd  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fabric", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=cd.DEFAULT_FRAMES)
    ap.add_argument("--dt", type=float, default=cd.DEFAULT_DT)
    ap.add_argument("--rings", type=int, default=cd.DEFAULT_RINGS)
    ap.add_argument("--seg", type=int, default=cd.DEFAULT_SEG)
    args = ap.parse_args(argv)

    presets = cd.load_presets()
    if args.fabric not in presets:
        print(f"unknown preset: {args.fabric}", file=sys.stderr)
        return 2
    sim = cd.simulate_drape(args.fabric, presets[args.fabric],
                            frames=args.frames, dt=args.dt,
                            rings=args.rings, seg=args.seg, grid=512)
    np.savez(args.out, vert=sim["vert"], face=sim["face"],
             dc=sim["dc"], rings=args.rings, seg=args.seg,
             r_specimen=cd.R_SPECIMEN, r_support=cd.R_SUPPORT)
    print(f"dumped {args.out}  DC={sim['dc']:.1f}%  "
          f"verts={len(sim['vert'])} faces={len(sim['face'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
