#!/usr/bin/env python3
# File: example-list.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from pathlib import Path


def get_example_notebooks():
    """Get all .ipynb files from the examples directory."""
    repo_root = Path(__file__).parent.parent.parent.parent
    examples_dir = repo_root / "examples"

    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return []

    notebooks = sorted([f.stem for f in examples_dir.glob("*.ipynb")])

    return notebooks


def write_examples_list():
    """Write the list of example notebooks to examples.txt."""
    script_dir = Path(__file__).parent
    output_file = script_dir / "examples.txt"

    notebooks = get_example_notebooks()

    if not notebooks:
        print("No notebook files found in examples directory")
        return

    with open(output_file, "w") as f:
        for notebook in notebooks:
            f.write(f"{notebook}\n")

    print(f"Written {len(notebooks)} notebooks to {output_file}")
    for notebook in notebooks:
        print(f"  - {notebook}")


if __name__ == "__main__":
    write_examples_list()

