# File: run_jobs.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""Run a file of independent command lines N at a time, and stop at the first
failure with that command's own output.

    python run_jobs.py JOBS_FILE [--jobs N]

Each line of JOBS_FILE is `<label>|<command line>`; blank lines are skipped.
Every command is handed to cmd.exe exactly as written, so a batch file can
build the line the way it would have run it and hand it here instead.

WHY THIS EXISTS. `build-cuda.bat` compiles the ABI TU, the mechanism TUs,
every generated entry point and the host C++ TUs, each an independent
`nvcc -dc`, and it ran them one after another in a `for` loop: measured on the
g6e.2xlarge release builder, the CUDA library took about ten minutes that way
with seven of eight cores idle. The ROCm build beside it (`build_rocm.py`)
already compiles through a pool. This gives the batch file the same pool
without rewriting it in Python: the loops append their command lines to a
file and one call here runs them all.

FAILURE IS THE FIRST FAILING COMMAND, WITH ITS OUTPUT, AND NOTHING AFTER IT.
Output is captured per command and printed in job order, so the log reads as
the serial loop's did rather than as N interleaved compilers. When a command
fails, its label, exit code and full output are printed, the remaining jobs
are cancelled, and the exit status is that command's, which the batch file
tests exactly as it tested the loop.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time


def run(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, shell=True, capture_output=True, text=True,
                          errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs_file")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="commands to run at once (default: the CPU count)")
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    with open(args.jobs_file, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            label, sep, command = line.partition("|")
            if not sep or not command.strip():
                print(f"ERROR: malformed job line (expected label|command): {line!r}")
                return 2
            jobs.append((label.strip(), command))
    if not jobs:
        print(f"ERROR: {args.jobs_file} holds no jobs")
        return 2

    width = max(1, min(args.jobs, len(jobs)))
    print(f"=== {len(jobs)} commands, {width} at a time ===", flush=True)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(run, command) for _label, command in jobs]
        for (label, command), future in zip(jobs, futures):
            result = future.result()
            if result.returncode != 0:
                pool.shutdown(wait=False, cancel_futures=True)
                print(f"=== FAILED: {label} (exit {result.returncode}) ===")
                print(command)
                sys.stdout.write(result.stdout)
                sys.stdout.write(result.stderr)
                sys.stdout.flush()
                return result.returncode
            text = (result.stdout + result.stderr).strip()
            print(f"  {label}: ok" + (f"\n{text}" if text else ""), flush=True)
    print(f"=== all {len(jobs)} commands ok in {time.time() - started:.0f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
