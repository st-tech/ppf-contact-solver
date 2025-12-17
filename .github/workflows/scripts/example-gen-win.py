#!/usr/bin/env python3
# File: example-gen-win.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""
Generate GitHub workflow files for each Jupyter notebook example (Windows Native).

This script reads the Windows AWS template and creates a workflow file for each
notebook found in the examples directory.
"""

import sys

from pathlib import Path


def read_examples(examples_file):
    """Read examples from the examples.txt file."""
    with open(examples_file) as f:
        examples = [line.strip() for line in f if line.strip()]
    return examples


def generate_workflow(template_path, notebook_name, output_path):
    """Generate a workflow file from template for a specific notebook."""
    with open(template_path) as f:
        template_content = f.read()

    # Replace <<example>> placeholder with the notebook name (without .ipynb)
    workflow_content = template_content.replace("<<example>>", notebook_name)

    # Write the workflow file
    with open(output_path, "w") as f:
        f.write(workflow_content)

    print(f"Generated: {output_path}")


def main():
    # Get the repository root (assuming script is in .github/workflows/scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent

    # Define paths
    template_path = (
        repo_root / ".github" / "workflows" / "template" / "aws-template-win.yml"
    )
    examples_file = script_dir / "examples.txt"
    examples_dir = repo_root / "examples"
    workflows_dir = repo_root / ".github" / "workflows"

    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)

    # Check if examples.txt exists
    if not examples_file.exists():
        print(f"Error: examples.txt not found at {examples_file}")
        sys.exit(1)

    # Check if examples directory exists
    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    # Read examples from examples.txt
    examples = read_examples(examples_file)

    if not examples:
        print(f"No examples found in {examples_file}")
        return

    print(f"Found {len(examples)} example(s) in {examples_file}")
    print(f"Using template: {template_path}")
    print(f"Output directory: {workflows_dir}")
    print("-" * 50)

    # Generate workflow for each example
    generated_count = 0
    skipped_count = 0

    for example_name in examples:
        notebook_path = examples_dir / f"{example_name}.ipynb"

        # Check if the notebook file exists
        if not notebook_path.exists():
            print(f"Warning: Skipping '{example_name}' - notebook not found at {notebook_path}")
            skipped_count += 1
            continue

        workflow_filename = f"{example_name}-win.yml"
        output_path = workflows_dir / workflow_filename

        generate_workflow(template_path, example_name, output_path)
        generated_count += 1

    print("-" * 50)
    print(f"Successfully generated {generated_count} workflow file(s)")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} example(s) (notebook files not found)")


if __name__ == "__main__":
    main()
