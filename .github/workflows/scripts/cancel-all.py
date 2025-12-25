#!/usr/bin/env python3
# File: cancel-all.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json
import subprocess
import sys


def main():
    print("Fetching workflow runs...", flush=True)
    result = subprocess.run(
        ["gh", "run", "list", "--limit", "100", "--json", "status,workflowName,databaseId"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to list runs: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    runs = json.loads(result.stdout)
    to_cancel = [r for r in runs if r["status"] in ("in_progress", "queued")]

    if not to_cancel:
        print("No runs to cancel", flush=True)
        return

    print(f"Found {len(to_cancel)} runs to cancel", flush=True)
    for run in to_cancel:
        run_id = run["databaseId"]
        workflow_name = run["workflowName"]
        print(f"Canceling '{workflow_name}' (#{run_id})", flush=True)
        subprocess.run(["gh", "run", "cancel", str(run_id)])


if __name__ == "__main__":
    main()
