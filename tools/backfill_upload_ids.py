#!/usr/bin/env python3
# File: tools/backfill_upload_ids.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Walk ~/.local/share/ppf-cts/git-*/*/ and stamp a fresh upload_id.txt on
# every project that has data.pickle + param.pickle but no upload_id.
# Invoked once after upgrading to protocol 0.03 to migrate legacy project
# directories so both the server's select_project and frontend's
# BlenderApp.open() stop raising on them.
#
#     python tools/backfill_upload_ids.py           # real run
#     python tools/backfill_upload_ids.py --dry-run
#

from __future__ import annotations

import argparse
import glob
import os
import sys
import uuid


def _mint_id() -> str:
    """Generate a 12-hex-char upload id, matching the server's format."""
    return uuid.uuid4().hex[:12]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stamp upload_id.txt on legacy ppf-cts project dirs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be stamped without writing anything.",
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/.local/share/ppf-cts"),
        help="ppf-cts base directory (default: ~/.local/share/ppf-cts).",
    )
    args = parser.parse_args()

    stamped = 0
    already = 0
    skipped = 0
    for project in sorted(glob.glob(os.path.join(args.root, "git-*", "*"))):
        if not os.path.isdir(project):
            continue
        data = os.path.join(project, "data.pickle")
        param = os.path.join(project, "param.pickle")
        uid_path = os.path.join(project, "upload_id.txt")
        if not (os.path.exists(data) and os.path.exists(param)):
            skipped += 1
            continue
        if os.path.exists(uid_path):
            already += 1
            continue
        uid = _mint_id()
        if args.dry_run:
            print(f"[dry-run] would stamp {uid} -> {uid_path}")
        else:
            tmp = f"{uid_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(uid)
            os.replace(tmp, uid_path)
            print(f"stamped {uid} -> {uid_path}")
        stamped += 1

    print(
        f"\nSummary: stamped={stamped}, already_had_id={already}, "
        f"skipped_no_pickles={skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
