# File: frontend/tests/_solver_lifetime_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A BLOCKING START OWNS ITS SOLVER. The solver runs in a session of its own, so
# nothing that ends the driver reaches it on its own: unguarded, a Ctrl-C, a
# `timeout` wrapper's SIGTERM or a dropped terminal's SIGHUP kills the Python
# driver and leaves the solver running on the GPU with nobody reading it.
# `_solver_bound_to_caller` is the guard the blocking waits run inside.
#
# A stand-in solver (a shell running `sleep`, launched exactly as the real one
# is: through a shell, in a new session) is enough, because what is under test
# is the process handling, not the solve.
#
# Covers:
#   * an exception leaving the wait (KeyboardInterrupt) stops the solver, the
#     shell AND the process under it, before the exception propagates;
#   * a SIGTERM delivered to a driver blocked in the wait stops the solver,
#     and the driver still dies by SIGTERM, so its parent sees the same status;
#   * a normal exit from the wait leaves a finished solver alone and restores
#     the signal handlers it replaced.

from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend._session_ import _solver_bound_to_caller
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows", reason="POSIX process groups"
)


def alive(pid: int) -> bool:
    """Whether `pid` is a live process, counting a zombie as dead."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def spawn_stand_in(marker: Path) -> subprocess.Popen:
    """A shell in a session of its own whose child writes its pid and sleeps,
    as `start` launches `command.sh`."""
    return subprocess.Popen(
        f"sh -c 'echo $$ > {marker}; exec sleep 300'",
        shell=True,
        start_new_session=True,
    )


def wait_for(path: Path) -> int:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_an_interrupt_stops_the_solver(tmp_path):
    marker = tmp_path / "pid"
    process = spawn_stand_in(marker)
    inner = wait_for(marker)
    with pytest.raises(KeyboardInterrupt):
        with _solver_bound_to_caller(process):
            raise KeyboardInterrupt
    assert process.poll() is not None
    assert not alive(inner), "the process under the shell outlived the driver"


def test_a_sigterm_to_the_driver_stops_the_solver(tmp_path):
    marker = tmp_path / "pid"
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from frontend._session_ import _solver_bound_to_caller
        process = subprocess.Popen(
            "sh -c 'echo $$ > {marker}; exec sleep 300'",
            shell=True, start_new_session=True)
        with _solver_bound_to_caller(process):
            while process.poll() is None:
                time.sleep(0.1)
    """))
    proc = subprocess.Popen([sys.executable, str(driver)])
    inner = wait_for(marker)
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == -signal.SIGTERM
    deadline = time.time() + 5.0
    while alive(inner) and time.time() < deadline:
        time.sleep(0.05)
    assert not alive(inner), "the solver outlived a driver ended by SIGTERM"


def test_a_clean_exit_restores_the_handlers(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    process = subprocess.Popen("true", shell=True, start_new_session=True)
    with _solver_bound_to_caller(process):
        process.wait()
    assert signal.getsignal(signal.SIGTERM) == before
    assert process.returncode == 0
