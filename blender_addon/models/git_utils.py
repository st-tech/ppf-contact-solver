# File: git_utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Git branch detection utility.
# Extracted from ui/state.py to keep PropertyGroup definitions clean.

import os
import subprocess


def get_git_branch():
    """Detect the current git branch name."""
    git_branch = "unknown"

    # First try to read from .git/branch_name.txt
    try:
        branch_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..",
            ".git",
            "branch_name.txt"
        )
        if os.path.exists(branch_file):
            with open(branch_file, "r") as f:
                git_branch = f.read().strip()
                if not git_branch:
                    git_branch = "unknown"
    except Exception as e:
        print(f"[git_utils] Failed to read branch_name.txt: {e}")

    # Fallback to git command if branch_name.txt not found or empty
    if git_branch == "unknown":
        try:
            git_branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if not git_branch:
                git_branch = "unknown"
        except Exception as e:
            print(f"[git_utils] Failed to detect git branch: {e}")
            git_branch = "unknown"

    return git_branch
