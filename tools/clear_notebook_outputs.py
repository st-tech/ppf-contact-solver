#!/usr/bin/env python3
# File: tools/clear_notebook_outputs.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Strip cell outputs and execution counts from every .ipynb under a target
# directory (default: examples/, recursive). Notebooks are rewritten in
# place with no dependency on jupyter/nbformat.
#
#     python tools/clear_notebook_outputs.py                       # real run on examples/
#     python tools/clear_notebook_outputs.py --dry-run
#     python tools/clear_notebook_outputs.py path/to/dir
#     python tools/clear_notebook_outputs.py a.ipynb b.ipynb        # individual files
#
# Used by hooks/pre-commit to scrub staged notebooks before they land.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _clear(nb: dict) -> bool:
    """Mutate notebook in place, returning True if anything changed."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        elif "outputs" not in cell:
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    return changed


def _process(path: Path, dry_run: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    if not _clear(nb):
        return False
    if dry_run:
        return True
    # Trailing newline matches Jupyter's on-disk format.
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        help="Notebook files or directories to scan (default: examples).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which notebooks would be rewritten, change nothing.",
    )
    args = ap.parse_args()

    skip_dirs = {".ipynb_checkpoints", ".virtual_documents"}
    raw_paths = [Path(p) for p in (args.paths or ["examples"])]
    notebooks: list[Path] = []
    for p in raw_paths:
        if p.is_dir():
            notebooks.extend(
                q for q in p.rglob("*.ipynb")
                if not skip_dirs.intersection(q.parts)
            )
        elif p.is_file() and p.suffix == ".ipynb":
            notebooks.append(p)
        else:
            print(f"error: {p} is not a notebook or directory", file=sys.stderr)
            return 2
    notebooks = sorted(set(notebooks))
    if not notebooks:
        print("no notebooks to process")
        return 0

    cleared = 0
    for nb_path in notebooks:
        try:
            if _process(nb_path, args.dry_run):
                cleared += 1
                print(f"{'would clear' if args.dry_run else 'cleared'}: {nb_path}")
        except json.JSONDecodeError as e:
            print(f"skip (invalid json): {nb_path}: {e}", file=sys.stderr)

    verb = "would clear" if args.dry_run else "cleared"
    print(f"{verb} {cleared}/{len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
