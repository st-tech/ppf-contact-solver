# File: scenarios/rig_server_idle_stop.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# `ppf-cts-server --idle-timeout N` stops itself N seconds after its last
# request ended, when no solve or build is running, and a request keeps it up.
# The rig's own server runs with `--idle-timeout 0`, so this scenario starts
# servers of its own, from the rig's build, on free ports, and times them with
# the wall clock. (`serve::tests` in the server crate covers a running solve
# and a build holding the server up, in paused time.)
#
# Subtests:
#   A. an_untouched_server_stops_at_the_timeout: started with --idle-timeout 3
#      and asked nothing after its readiness query, it exits with status 0
#      between 3 and 5 s after that query answered, and its log names the
#      idle stop. The status response reports idle_timeout_s 3.
#   B. requests_keep_it_up: queried once a second for 8 s, it is up after
#      every query, then exits 3 to 5 s after the last one.
#   C. zero_never_stops: started with --idle-timeout 0, it still answers 6 s
#      after its last query and reports idle_timeout_s 0.

from __future__ import annotations

import os
import socket
import subprocess
import time

from . import _runner as r


BACKENDS = ("real",)

TIMEOUT_S = 3
# The server sleeps until its deadline, so it stops close to TIMEOUT_S after
# the last request; the slack covers a loaded host. The lower bound sits a
# little under, because the server starts counting when it has written the
# answer, a moment before the client has read it.
EARLIEST_S = TIMEOUT_S - 0.2
LATEST_S = TIMEOUT_S + 2.0
SERVER_EXE = "ppf-cts-server.exe" if os.name == "nt" else "ppf-cts-server"


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _Server:
    """One `ppf-cts-server` of this scenario's own, started with an idle time."""

    def __init__(self, ctx: r.ScenarioContext, tag: str, idle_timeout: int):
        import orchestrator  # on the path the orchestrator sets up

        build_dir = orchestrator.server_build_dir()
        self.dir = os.path.join(ctx.workspace, f"idle_stop_{tag}")
        os.makedirs(self.dir, exist_ok=True)
        self.port = _free_port()
        self.log_path = os.path.join(self.dir, "server.log")
        env = os.environ.copy()
        env["PPF_CTS_DATA_ROOT"] = os.path.join(self.dir, "data")
        env["CARGO_TARGET_DIR"] = os.path.dirname(build_dir.rstrip(os.sep))
        env["PPF_SOLVER_SCAN_DESCENDANTS"] = "1"
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [os.path.join(build_dir, SERVER_EXE), "--port", str(self.port),
             "--idle-timeout", str(idle_timeout)],
            cwd=self.dir, env=env, stdout=self._log, stderr=subprocess.STDOUT,
        )
        self.client = r.ProtoClient("127.0.0.1", self.port, timeout=5.0)

    def query(self) -> tuple[dict, float]:
        """A status query, and the monotonic time its answer arrived."""
        response = self.client.text_cmd({"name": "idle_stop_probe"})
        return response, time.monotonic()

    def ready(self, within: float = 60.0) -> tuple[dict, float]:
        """Wait for the first answer: the server's first request."""
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode} before answering; "
                                   f"log {self.log_path}")
            try:
                return self.query()
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"server did not answer within {within} s; log {self.log_path}")

    def wait_exit(self, within: float) -> float | None:
        """Monotonic time the process was seen gone, or None within `within`."""
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return time.monotonic()
            time.sleep(0.02)
        return None

    def log_text(self) -> str:
        self._log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


def _stopped_in_window(server: _Server, last_answer: float) -> dict:
    gone = server.wait_exit(LATEST_S + 5.0)
    after = None if gone is None else round(gone - last_answer, 3)
    log = server.log_text()
    return {
        "stopped_after_s": after,
        "exit_code": server.proc.returncode,
        "logged_idle_stop": "idle stop: no request for" in log,
        "ok": (after is not None and EARLIEST_S <= after <= LATEST_S
               and server.proc.returncode == 0 and "idle stop: no request for" in log),
    }


def run(ctx: r.ScenarioContext) -> dict:
    checks: dict = {}
    servers: list[_Server] = []
    try:
        # A: untouched after its readiness query.
        a = _Server(ctx, "a", TIMEOUT_S)
        servers.append(a)
        response, answered = a.ready()
        detail = _stopped_in_window(a, answered)
        detail["idle_timeout_s"] = response.get("idle_timeout_s")
        checks["A_an_untouched_server_stops_at_the_timeout"] = {
            "ok": detail.pop("ok") and response.get("idle_timeout_s") == TIMEOUT_S,
            "details": detail,
        }

        # B: a query every second holds it up.
        b = _Server(ctx, "b", TIMEOUT_S)
        servers.append(b)
        _, answered = b.ready()
        down_after = []
        for i in range(8):
            time.sleep(1.0)
            if b.proc.poll() is not None:
                down_after.append(i)
                break
            _, answered = b.query()
        detail = {"down_during_queries": down_after}
        held = not down_after
        if held:
            detail.update(_stopped_in_window(b, answered))
        checks["B_requests_keep_it_up"] = {"ok": held and detail.pop("ok", False),
                                           "details": detail}

        # C: 0 never stops.
        c = _Server(ctx, "c", 0)
        servers.append(c)
        response, _ = c.ready()
        time.sleep(6.0)
        alive = c.proc.poll() is None
        answered_again = None
        if alive:
            answered_again, _ = c.query()
        checks["C_zero_never_stops"] = {
            "ok": alive and answered_again is not None and response.get("idle_timeout_s") == 0,
            "details": {"alive_after_6_s": alive, "idle_timeout_s": response.get("idle_timeout_s"),
                        "log": c.log_path},
        }
    finally:
        for server in servers:
            server.close()
    return r.report_named_checks(checks)
