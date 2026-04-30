# File: scenarios/_runner.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared helpers for Phase 1 protocol-level scenarios.
#
# A ScenarioContext carries the per-worker configuration (server host/port,
# project name, paths). Each scenario gets one and uses ``ProtoClient``
# to drive the server through its public wire protocol -- the same one
# the addon's communicator speaks.

from __future__ import annotations

import json
import os
import pickle
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional


HEADER_TEXT_CMD = b"TCMD"
HEADER_JSON_DATA = b"JSON"


@dataclass
class ScenarioContext:
    """Per-scenario configuration handed in by the orchestrator."""

    host: str
    server_port: int
    project_name: str
    workspace: str
    project_root: str  # PPF_CTS_DATA_ROOT/git-debug/<project_name>
    timeout: float = 30.0
    log_path: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    # Knobs propagated to the spawned server. The scenario can read these
    # to gate behavior (e.g. solver_crash needs PPF_FAKE_SOLVER_CRASH_FRAME)
    # without trying to introspect a foreign process's environment.
    knobs: dict[str, str] = field(default_factory=dict)

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}\n"
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line)


# ---------------------------------------------------------------------------
# Protocol client (mirrors the subset of core/protocol.py the addon uses)
# ---------------------------------------------------------------------------

class ProtoClient:
    """Thin client for the debug server. One TCP connection per call,
    matching the addon's short-lived-channel pattern."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self._host = host
        self._port = port
        self._timeout = timeout

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect((self._host, self._port))
        return s

    def text_cmd(self, args: dict[str, Any]) -> dict:
        """Send a text command (HEADER_TEXT_CMD) and parse the JSON reply.

        ``args`` must include ``name``. Other keys map 1:1 to the
        ``--key value`` flags the production server's text-command parser
        understands. Returns the parsed response dict; raises on socket
        error."""
        if "name" not in args:
            raise ValueError("text_cmd requires a 'name' key")
        flat = " ".join(f"--{k} {v}" for k, v in args.items())
        with self._connect() as s:
            s.sendall(HEADER_TEXT_CMD)
            s.sendall(flat.encode())
            s.shutdown(socket.SHUT_WR)
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            raise RuntimeError("empty response")
        return json.loads(buf.decode())

    def upload_atomic(self, name: str, data_bytes: bytes, param_bytes: bytes,
                      project_root: str) -> dict:
        """Atomically upload ``data.pickle`` and ``param.pickle`` for *name*.

        Either payload may be empty, but at least one must be non-empty
        (mirrors the production rule). Returns ``{"ok": True}`` on success
        or ``{"error": str}`` on protocol error."""
        if not data_bytes and not param_bytes:
            raise ValueError("upload_atomic needs at least one payload")
        meta = {
            "request": "upload_atomic",
            "name": name,
            "path": project_root,
            "data_size": len(data_bytes),
            "param_size": len(param_bytes),
        }
        with self._connect() as s:
            s.sendall(HEADER_JSON_DATA)
            s.sendall((json.dumps(meta) + "\n").encode())
            if data_bytes:
                s.sendall(data_bytes)
            if param_bytes:
                s.sendall(param_bytes)
            s.shutdown(socket.SHUT_WR)
            reply = s.recv(4096).decode().strip()
        if reply.startswith("OK"):
            return {"ok": True}
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return {"error": f"unparseable reply: {reply!r}"}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def wait_for(predicate, *, timeout: float, interval: float = 0.1,
             desc: str = "") -> bool:
    """Poll ``predicate()`` until it returns truthy or ``timeout`` elapses.

    Returns the predicate's final value. ``desc`` is recorded in the
    AssertionError message on timeout."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise TimeoutError(f"wait_for timed out after {timeout}s ({desc}); last={last!r}")


def make_minimal_param(frames: int = 6, fps: int = 30) -> bytes:
    """Build a minimal param.pickle that the fake builder accepts and
    that drives the fake solver to ``frames`` total frames."""
    return pickle.dumps({"frames": frames, "fps": fps})


def make_minimal_data() -> bytes:
    """Stand-in data.pickle. Phase 1 doesn't validate its contents; the
    fake build only cares that the file exists and unpickles."""
    return pickle.dumps({
        "vertices": [],
        "tris": [],
        "tets": [],
        "rods": [],
        "groups": [],
        "map_by_name": {},
    })


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def passed(notes: list[str] | None = None) -> dict:
    return {"status": "pass", "violations": [], "notes": notes or []}


def failed(violations: list[str], notes: list[str] | None = None) -> dict:
    return {"status": "fail", "violations": violations, "notes": notes or []}


# ---------------------------------------------------------------------------
# Blender-driver result handling
# ---------------------------------------------------------------------------
#
# Every Blender-driven scenario's ``run()`` ends the same way:
#
#   1. Pull blender_spec / blender_proc out of ctx.artifacts.
#   2. ``bh.wait_for_result`` until the bootstrap writes scenario_result.json.
#   3. If the driver hit an exception, surface it as the verdict.
#   4. Walk the driver's named-check dict (``checks`` / ``subtests``)
#      and turn each False into a violation.
#
# The two helpers below collapse that boilerplate. ``wait_blender_result``
# returns ``(result, error_verdict)`` where ``error_verdict`` is non-None
# only on attach failure / timeout / driver exception; the caller short-
# circuits with that verdict and does its own check walking otherwise.
# ``report_named_checks`` does the standard walk for scenarios that just
# need pass/fail per named check.


def wait_blender_result(ctx: "ScenarioContext", *,
                        timeout: float | None = None,
                        max_errors_in_violation: int = 3,
                        ) -> tuple[dict, dict | None]:
    """Resolve a Blender-driven scenario's result file.

    Returns ``(result_dict, error_verdict)``. When ``error_verdict`` is
    not None the caller should return it directly; the result_dict is
    still populated for diagnostics.
    """
    import blender_harness as bh  # noqa: E402

    bspec = ctx.artifacts.get("blender_spec")
    proc = ctx.artifacts.get("blender_proc")
    if bspec is None or proc is None:
        return {}, failed(["no Blender process attached"])

    effective_timeout = timeout if timeout is not None else max(ctx.timeout, 90.0)
    try:
        result = bh.wait_for_result(bspec, proc, timeout=effective_timeout)
    except TimeoutError as e:
        return {}, failed([str(e)])

    if result.get("errors"):
        phases_tail = result.get("phases", [])[-5:]
        return result, failed(
            result["errors"][:max_errors_in_violation],
            notes=[f"Blender phases: {phases_tail}"],
        )
    return result, None


def report_named_checks(checks: dict, *, label: str = "checks",
                        max_violations: int = 6) -> dict:
    """Convert a name -> {"ok": bool, "details": ...} mapping into a
    pass/fail verdict. Used by every scenario that records its assertions
    as a flat dict on the result. ``checks`` is the dict the driver
    wrote; an empty dict is treated as failure (the driver produced no
    assertions to evaluate, which always indicates a broken driver).
    """
    if not checks:
        return failed([f"driver wrote no {label}"])

    notes = [f"{label}: {len(checks)}"]
    for name, info in sorted(checks.items()):
        notes.append(f"  {name}: {'pass' if info.get('ok') else 'fail'}")

    failures = [(name, info) for name, info in sorted(checks.items())
                if not info.get("ok")]
    if not failures:
        return passed(notes=notes)

    violations = [
        f"{name}: details={json.dumps(info.get('details'))}"
        for name, info in failures[:max_violations]
    ]
    return failed(violations, notes=notes)
