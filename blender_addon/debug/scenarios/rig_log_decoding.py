# File: scenarios/rig_log_decoding.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Guard for how the rig DECODES the logs it collects. Every one of them was
# written by another process: the server's `stdout.log` and `stderr.log` are
# opened "wb" and hold whatever the Rust server emitted, Blender's two logs
# the same, and the solver writes its own session log. All of them are UTF-8
# writers, and the rig must read them as UTF-8 rather than as whatever
# `locale.getpreferredencoding()` answers on the host it happens to run on.
#
# This is not hypothetical. On the Windows rigs that answer is cp1252, which
# leaves five bytes undefined. The build worker draws tqdm bars, and a bar
# whose filled fraction is not a whole eighth of its width emits a PARTIAL
# block: U+258D encodes to `e2 96 8d`, and 0x8d is one of the five. Decoding
# it raised UnicodeDecodeError, which is a ValueError rather than an OSError,
# so it went straight past the orchestrator's read guard, out of `run_one`
# and out of `main`. Blender CI run 35448531061 lost 26 of shard 3's 55
# scenarios to one character of a progress bar and reported a verdict for
# none of them; the run before it had passed on the same content, because
# whether a redraw lands on a partial eighth is a matter of timing.
#
# THE CHECK RUNS IN A CHILD UNDER THE C LOCALE, which is what lets it fail on
# every platform rather than only on Windows. `open()` with no encoding is
# UTF-8 on a Linux rig, so a check in THIS process agrees with the fix by
# accident and could never have caught the defect. Under `LC_ALL=C` with
# UTF-8 mode off the default becomes ASCII, and ASCII refuses the same bytes
# cp1252 refuses, so a tree without the fix fails here on Linux, on macOS and
# on Windows alike.
#
# Server-only: this reads staged files with the orchestrator's own helpers,
# so it needs neither a solver nor Blender.

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from . import _runner as r

# No solver and no Blender, so this holds on the real-GPU jobs too.
BACKENDS = ("real",)

# The three the Windows rig died on, plus one that is not valid UTF-8 at all:
# a solver killed partway through a write leaves a truncated sequence, and a
# log is evidence either way, so the reader must not raise on it.
_FULL_BLOCK = "\u2588"          # e2 96 88, every byte defined in cp1252
_PARTIAL_BLOCK = "\u258d"       # e2 96 8d, byte 0x8d undefined in cp1252
_THIN_BLOCK = "\u258f"          # e2 96 8f, byte 0x8f undefined in cp1252
_TRUNCATED = b"\xe2\x96"        # a multi-byte sequence cut off mid-character

_BUILD_LINE = (
    r"solver build: target_dir=C:\ppf-contact-solver\target\cuda backend=cuda"
)


def _server_log_bytes() -> bytes:
    """A server `stdout.log` shaped like the one that killed shard 3."""
    lines = [
        "2026-09-19T15:07:28.001 INFO ppf::serve - "
        "ppf-cts-server listening on 127.0.0.1:52001",
        "2026-09-19T15:07:28.002 INFO ppf::serve - " + _BUILD_LINE,
        # The build worker's bar, as ppf::build relays it.
        "2026-09-19T15:07:31.999 WARN ppf::build - [BUILD stderr] "
        "solid_solid_stitch: build scene:  37%|"
        + _FULL_BLOCK * 3 + _PARTIAL_BLOCK + _THIN_BLOCK
        + "     | 3/8 [00:00<00:00, 9.1it/s]",
        "2026-09-19T15:07:32.000 INFO ppf::executor - Build complete.",
    ]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8") + _TRUNCATED


# Run in a child so the locale is the child's problem: changing it in this
# process would not reach an already-imported `_io`, and would leak into
# every scenario that shares the worker.
_CHILD = textwrap.dedent(
    """
    import json, locale, os, sys
    sys.path.insert(0, sys.argv[1])
    import orchestrator

    server_dir = sys.argv[2]
    out = {"preferred": locale.getpreferredencoding(False)}
    try:
        out["build_line"] = orchestrator._server_build_line(server_dir)
    except Exception as exc:
        out["build_error"] = f"{type(exc).__name__}: {exc}"
    try:
        text = orchestrator._read_text(
            os.path.join(server_dir, "stdout.log"))
        out["read_len"] = len(text)
        out["saw_bar"] = "build scene" in text
    except Exception as exc:
        out["read_error"] = f"{type(exc).__name__}: {exc}"
    print("RESULT " + json.dumps(out))
    """
).strip()


def _child_env() -> dict:
    env = os.environ.copy()
    # An ASCII default is what makes this fail on a POSIX rig too. UTF-8 mode
    # and the C.UTF-8 coercion both override the locale, so both go off.
    for key in ("LANG", "LC_CTYPE", "LC_ALL"):
        env.pop(key, None)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PYTHONUTF8"] = "0"
    env["PYTHONCOERCECLOCALE"] = "0"
    return env


def run(ctx: r.ScenarioContext) -> dict:
    violations: list[str] = []
    notes: list[str] = []

    server_dir = os.path.join(ctx.workspace, "decoding", "server")
    os.makedirs(server_dir, exist_ok=True)
    blob = _server_log_bytes()
    # "wb", exactly as `_spawn_server` opens the real one.
    with open(os.path.join(server_dir, "stdout.log"), "wb") as handle:
        handle.write(blob)
    with open(os.path.join(server_dir, "stderr.log"), "wb") as handle:
        handle.write(b"")

    if b"\x8d" not in blob:
        violations.append(
            "the staged log carries no 0x8d byte, so this scenario is not "
            "exercising the sequence that killed the Windows rig"
        )

    debug_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, debug_dir, server_dir],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, env=_child_env(),
        )
    except subprocess.TimeoutExpired:
        return r.failed(["the decoding child did not answer within 120s"])

    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
    if payload is None:
        return r.failed([
            "the decoding child printed no RESULT line "
            f"(exit {proc.returncode}); stdout={proc.stdout[-800:]!r} "
            f"stderr={proc.stderr[-800:]!r}"
        ])

    notes.append(f"child default encoding: {payload.get('preferred')}")

    if "build_error" in payload:
        violations.append(
            "_server_build_line raised on a server log carrying a tqdm "
            "partial block: " + payload["build_error"] + ". This is the "
            "defect that ended Blender CI run 35448531061's Windows shard 3 "
            "at scenario 30 of 55, and it takes every scenario after it."
        )
    elif payload.get("build_line") != _BUILD_LINE[len("solver build:"):].strip():
        violations.append(
            "_server_build_line returned "
            f"{payload.get('build_line')!r}, want "
            f"{_BUILD_LINE[len('solver build:'):].strip()!r}; the line the "
            "rig reports the build from must survive the bar above it"
        )

    if "read_error" in payload:
        violations.append(
            "_read_text raised on a log written by another process: "
            + payload["read_error"] + ". Every log the rig attaches to a "
            "result goes through it, so this loses the run's evidence as "
            "well as the run"
        )
    elif not payload.get("saw_bar"):
        violations.append(
            "_read_text returned text without the progress-bar line "
            f"(length {payload.get('read_len')}); an undecodable byte must "
            "cost that byte, not the rest of the file"
        )

    return r.failed(violations, notes) if violations else r.passed(notes)
