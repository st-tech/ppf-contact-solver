#!/usr/bin/env python3
# File: .github/workflows/scripts/cpu-acceptance-report.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reads the example-suite reports cpu-acceptance.yml writes, for the CPU backend
# on x86_64 and on aarch64.
#
#     cpu-acceptance-report.py fast-check REPORT.json [--expect-all]
#     cpu-acceptance-report.py compare --x86 FRAMES.json --arm FRAMES.json
#
# FAST-CHECK IS A VERDICT. Every scene the report holds has to have run and
# moved: a failed run, a harness fault, an aborted run and a refusal all fail
# here, each by name. A refusal is not a failure for run_suite.py, which reports
# a capability the backend declines; it IS one for this item, whose whole claim
# is that every simulating example produces moving output. With --expect-all the
# report must also cover every simulating notebook run_suite.py itself counts,
# so a scene that silently dropped out of the run cannot pass by absence.
#
# COMPARE IS A MEASUREMENT, NOT A VERDICT, and says so in its output. The solver
# is not bit-reproducible run to run, and across two architectures the shared
# bodies contract into different fused instructions, so agreement can only be
# read against each build's OWN run-to-run spread. This
# prints, per scene, disp_at_5 on each architecture with the spread of its
# repeats, and whether the aarch64 values fall inside the x86_64 range widened by
# the larger of the two spreads. It fails only on what is unambiguous: a scene
# missing from one side, a run that did not reach frame 5, or a vertex count that
# differs within one architecture's repeats. The acceptance rule on the
# displacement itself is decided from the first measured table, not written here
# before any exists.

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read {path}: {exc}")


def simulating_notebooks():
    """The notebook stems run_suite.py runs and counts as simulations."""
    spec = importlib.util.spec_from_file_location("run_suite", REPO / "tools" / "run_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    return sorted(nb.stem for nb in suite.notebooks() if suite.simulates(nb))


def fast_check(args):
    records = load(args.report)
    if not records:
        print("the report holds no runs, so nothing was shown to move")
        return 1
    bad = []
    moved = []
    for r in records:
        name, stage = r.get("name"), r.get("stage")
        if stage == "run-nosim":
            if not r.get("ok"):
                bad.append(f"{name}: the non-simulating notebook failed (rc={r.get('rc')})")
            continue
        if stage != "run":
            reason = r.get("reason") or r.get("error") or ""
            bad.append(f"{name}: stage {stage}, not a run {reason[:160]}".rstrip())
            continue
        if not r.get("ok"):
            bad.append(f"{name}: did not pass (frames={r.get('frames')}, "
                       f"moving={r.get('moving_frames')}/{r.get('steps')}, rc={r.get('rc')})")
        elif r.get("aborted"):
            bad.append(f"{name}: aborted: {r.get('abort_reason')}")
        elif not r.get("moving_frames"):
            bad.append(f"{name}: no frame moved")
        else:
            moved.append(name)
    print(f"{len(moved)} simulating scene(s) produced moving output: {' '.join(sorted(moved))}")
    if args.expect_all:
        want = simulating_notebooks()
        missing = sorted(set(want) - set(moved) - {b.split(':', 1)[0] for b in bad})
        if missing:
            bad.append(f"not in the report at all: {' '.join(missing)}")
        print(f"run_suite.py counts {len(want)} simulating notebooks")
    for line in bad:
        print(f"FAIL {line}")
    return 1 if bad else 0


def by_scene(records, label):
    scenes = {}
    for r in records:
        if r.get("stage") != "run":
            continue
        scenes.setdefault(r["name"], []).append(r)
    if not scenes:
        sys.exit(f"the {label} report holds no runs")
    return scenes


def spread(values):
    return max(values) - min(values)


def compare(args):
    x86 = by_scene(load(args.x86), "x86_64")
    arm = by_scene(load(args.arm), "aarch64")
    problems = []
    rows = []
    for name in sorted(set(x86) | set(arm)):
        if name not in x86 or name not in arm:
            problems.append(f"{name}: present on {'x86_64' if name in x86 else 'aarch64'} only")
            continue
        sides = {}
        for label, runs in (("x86_64", x86[name]), ("aarch64", arm[name])):
            values = [r.get("disp_at_5") for r in runs]
            if any(v is None for v in values):
                problems.append(f"{name}: a {label} run did not reach frame 5")
                break
            vertices = {r.get("vertices") for r in runs}
            if len(vertices) != 1:
                problems.append(f"{name}: the {label} repeats disagree on the vertex count {sorted(vertices)}")
                break
            sides[label] = (values, vertices.pop())
        if len(sides) != 2:
            continue
        xv, xn = sides["x86_64"]
        av, an = sides["aarch64"]
        width = max(spread(xv), spread(av))
        inside = all(min(xv) - width <= v <= max(xv) + width for v in av)
        rows.append((name, xn, an, min(xv), max(xv), spread(xv), min(av), max(av), spread(av), inside))

    print("disp_at_5 per scene, over each architecture's repeats. A MEASUREMENT: see the header.")
    header = (f"{'scene':<14} {'verts x86':>9} {'verts arm':>9} {'x86 min':>11} {'x86 max':>11} "
              f"{'x86 spread':>11} {'arm min':>11} {'arm max':>11} {'arm spread':>11}  inside")
    print(header)
    for name, xn, an, xmin, xmax, xs, amin, amax, asp, inside in rows:
        print(f"{name:<14} {xn:>9} {an:>9} {xmin:>11.4e} {xmax:>11.4e} {xs:>11.3e} "
              f"{amin:>11.4e} {amax:>11.4e} {asp:>11.3e}  {'yes' if inside else 'no'}")
    for line in problems:
        print(f"FAIL {line}")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    fc = sub.add_parser("fast-check")
    fc.add_argument("report")
    fc.add_argument("--expect-all", action="store_true",
                    help="require every simulating notebook run_suite.py counts")
    cp = sub.add_parser("compare")
    cp.add_argument("--x86", required=True)
    cp.add_argument("--arm", required=True)
    args = ap.parse_args()
    return fast_check(args) if args.mode == "fast-check" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
