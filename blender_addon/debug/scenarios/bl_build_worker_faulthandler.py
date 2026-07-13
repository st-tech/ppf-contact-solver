# File: scenarios/bl_build_worker_faulthandler.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for the "build worker exited with code 1" mystery on
# Windows.
#
# The build worker (frontend/build_worker.py) wraps its work in a broad
# ``except BaseException`` that emits a structured ``ERROR`` line, so any
# ordinary Python failure reaches the add-on with a real message. But a
# NATIVE crash inside a C extension (the classic case: an ABI-mismatched
# scipy/numpy blowing up in the SuperLU solve of the partial-pin SOLID
# harmonic extension, shipped by an unpinned warmup) kills the interpreter
# WITHOUT a Python exception. The ``except`` never runs, no ``ERROR`` line
# is emitted, and the server falls back to the useless generic
# "build worker exited with code 1" at ~10%, leaving no way to tell what
# crashed. build_worker now calls ``faulthandler.enable()`` at import so a
# native crash dumps a Python-level traceback to stderr, which the server
# forwards to ``server.log`` as ``[BUILD stderr]``.
#
# Checks (both run a fresh interpreter as a subprocess; no server, no
# solver, so this runs on every platform incl. macOS):
#   A. build_worker_enables_faulthandler: importing build_worker turns
#      faulthandler on. Importing it does NOT pull in the frontend cdylib
#      (the ``from frontend import ...`` lives inside main()), so this is a
#      cheap, dependency-free import.
#   B. faulthandler_surfaces_native_crash: with faulthandler enabled the
#      way build_worker enables it, a deliberate native crash
#      (``ctypes.string_at(0)``, a read of address 0) prints a fatal-error
#      traceback to stderr instead of dying silently.

from __future__ import annotations

import os

from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import os
import subprocess
import sys
import traceback

result.setdefault("checks", {})
result.setdefault("errors", [])

REPO = r"<<REPO_ROOT>>"
FRONTEND = os.path.join(REPO, "frontend")


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    # ----- A: importing build_worker enables faulthandler -----
    code_a = (
        "import sys, faulthandler\n"
        "sys.path.insert(0, r'" + FRONTEND + "')\n"
        "import build_worker\n"
        "print('FAULTHANDLER_ENABLED=' + str(faulthandler.is_enabled()))\n"
    )
    pa = subprocess.run(
        [sys.executable, "-c", code_a],
        capture_output=True, text=True, timeout=60,
    )
    out_a = (pa.stdout or "") + (pa.stderr or "")
    record(
        "A_build_worker_enables_faulthandler",
        pa.returncode == 0 and "FAULTHANDLER_ENABLED=True" in out_a,
        {"rc": pa.returncode, "output": out_a.strip()[:300]},
    )

    # ----- B: faulthandler surfaces a native crash -----
    # Mirror build_worker's setup (faulthandler.enable()) then fault-inject a
    # read of address 0. On a native crash faulthandler prints a fatal-error
    # banner + Python stack to stderr; assert on that (not the exit code,
    # since ctypes may also raise an OSError on some platforms -- the banner
    # is emitted either way, which is the diagnostic we care about).
    code_b = (
        "import faulthandler, ctypes\n"
        "faulthandler.enable()\n"
        "ctypes.string_at(0)\n"
    )
    pb = subprocess.run(
        [sys.executable, "-c", code_b],
        capture_output=True, text=True, timeout=60,
    )
    err_b = pb.stderr or ""
    markers = (
        "Fatal Python error",
        "Windows fatal exception",
        "Segmentation fault",
        "access violation",
        "Current thread",
    )
    surfaced = any(m in err_b for m in markers)
    record(
        "B_faulthandler_surfaces_native_crash",
        surfaced,
        {"rc": pb.returncode, "stderr": err_b.strip()[:400]},
    )

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE.replace("<<REPO_ROOT>>", REPO_ROOT_POSIX)


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
