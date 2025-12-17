#!/usr/bin/env python3
# File: run-all.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import subprocess
import sys
from pathlib import Path
import time


def read_examples_list():
    """Read the list of examples from examples.txt."""
    script_dir = Path(__file__).parent
    examples_file = script_dir / "examples.txt"

    if not examples_file.exists():
        print(f"Error: {examples_file} not found.")
        sys.exit(1)

    with open(examples_file, "r") as f:
        examples = [line.strip() for line in f if line.strip()]

    return examples


def trigger_workflow(
    example_name, instance_type="g6e.2xlarge", region="us-east-1", branch=None
):
    """Trigger a GitHub Actions workflow for a specific example."""
    workflow_file = f"{example_name}.yml"

    cmd = [
        "gh",
        "workflow",
        "run",
        workflow_file,
        "-f",
        f"instance_type={instance_type}",
        "-f",
        f"region={region}",
    ]

    if branch:
        cmd.extend(["--ref", branch])

    print(f"Triggering workflow: {workflow_file}")
    print(f"  Instance type: {instance_type}")
    print(f"  Region: {region}")
    if branch:
        print(f"  Branch: {branch}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Successfully triggered {workflow_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to trigger {workflow_file}")
        print(f"    Error: {e.stderr}")
        return False


def main():
    instance_type = sys.argv[1] if len(sys.argv) > 1 else "g6e.2xlarge"
    region = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"

    # Get current branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        current_branch = "unknown"

    print(f"Running all workflows with:")
    print(f"  Branch: {current_branch}")
    print(f"  Instance type: {instance_type}")
    print(f"  Region: {region}")
    print()

    examples = read_examples_list()

    if not examples:
        print("No examples found in examples.txt")
        sys.exit(1)

    print(f"Found {len(examples)} examples to run:")
    for example in examples:
        print(f"  - {example}")
    print()

    successful = 0
    failed = 0

    for i, example in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] Processing {example}")

        if trigger_workflow(example, instance_type, region, current_branch):
            successful += 1
        else:
            failed += 1

        if i < len(examples):
            time.sleep(2)

        print()

    print("Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(examples)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

