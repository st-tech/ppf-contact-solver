#!/usr/bin/env python3
# File: example-gen.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""
Generate one GitHub workflow file per example listed in examples.txt.

The template at .github/workflows/template/aws-template.yml carries the
shared body with `<<example>>` placeholders. For each name in
examples.txt this script substitutes the placeholder and writes
`.github/workflows/<name>.yml`.

Idempotent: rerunning overwrites the per-example yamls in place.
"""

import sys
from pathlib import Path


def read_examples(examples_file: Path) -> list[str]:
    with open(examples_file) as f:
        return [line.strip() for line in f if line.strip()]


def generate_workflow(template_path: Path, name: str, output_path: Path) -> None:
    with open(template_path) as f:
        content = f.read()
    workflow = content.replace("<<example>>", name)
    with open(output_path, "w") as f:
        f.write(workflow)
    print(f"Generated: {output_path}")


def main() -> int:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent

    template_path = (
        repo_root / ".github" / "workflows" / "template" / "aws-template.yml"
    )
    examples_file = script_dir / "examples.txt"
    workflows_dir = repo_root / ".github" / "workflows"

    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        return 1
    if not examples_file.exists():
        print(f"Error: examples.txt not found at {examples_file}")
        return 1

    names = read_examples(examples_file)
    if not names:
        print(f"Error: examples.txt is empty: {examples_file}")
        return 1

    for name in names:
        output_path = workflows_dir / f"{name}.yml"
        generate_workflow(template_path, name, output_path)

    print(f"Done. Generated {len(names)} workflow file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
