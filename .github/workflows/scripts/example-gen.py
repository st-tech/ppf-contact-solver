#!/usr/bin/env python3
"""
Generate GitHub workflow files for each Jupyter notebook example.

This script reads the AWS template and creates a workflow file for each
notebook found in the examples directory.
"""

import sys

from pathlib import Path


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
        repo_root / ".github" / "workflows" / "template" / "aws-template.yml"
    )
    examples_dir = repo_root / "examples"
    workflows_dir = repo_root / ".github" / "workflows"

    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)

    # Check if examples directory exists
    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    # Find all .ipynb files in examples directory
    notebooks = list(examples_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"No .ipynb files found in {examples_dir}")
        return

    print(f"Found {len(notebooks)} notebook(s) in {examples_dir}")
    print(f"Using template: {template_path}")
    print(f"Output directory: {workflows_dir}")
    print("-" * 50)

    # Generate workflow for each notebook
    for notebook_path in notebooks:
        notebook_name = notebook_path.stem  # Get filename without extension
        workflow_filename = f"{notebook_name}.yml"
        output_path = workflows_dir / workflow_filename

        generate_workflow(template_path, notebook_name, output_path)

    print("-" * 50)
    print(f"Successfully generated {len(notebooks)} workflow file(s)")


if __name__ == "__main__":
    main()

