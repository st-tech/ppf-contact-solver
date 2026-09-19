# File: server_kill.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Ending a ppf-cts-server process, with or without a live backend.
#
# Two kills live here, one per place a server can run:
#
# - ``kill_local_server`` ends whatever listens on a loopback port of THIS
#   machine. It is what the local, Windows Native and macOS Native backends
#   run from ``stop_server``, and what the Force Terminate Process button runs while the
#   add-on is NOT connected: a native Connect is refused while a server from
#   an earlier session still holds the port, and in that state there is no
#   backend to ask, only the port.
# - ``kill_remote_server`` ends the server on a host reached through a
#   backend's ``exec_command`` (SSH, Docker, Docker over SSH). It needs a live
#   transport, so it runs only while connected.
#
# Both return a ``KillReport`` rather than raising: the caller is a button
# whose whole job is recovery, so a failure to kill has to be read as a
# reason in the report, never as a traceback in the UI.

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

# The server binary's name, which is unique to this project. Both kills below
# match it with ``-f``, against the whole command line, and the remote one
# then compares each candidate against ``ps -o comm=``. That comparison is
# what the length matters for: Linux truncates ``comm`` to 15 characters, and
# this name is 14, so it survives intact and the comparison is exact rather
# than a prefix. macOS reports a full path there instead, which is why the
# comparison is written as a substring match at both ends.
SERVER_PROCESS_NAME = "ppf-cts-server"

# How long a SIGTERM is given to close the listen socket before SIGKILL. The
# server passes a cancel to an in-flight build worker on its own shutdown
# path, which is why the first signal is the polite one.
_TERM_GRACE_S = 2.0

# Subprocess bound for every tool call here (lsof, netstat, taskkill, pkill).
# All of them answer in well under a second; the bound exists so a hung tool
# cannot hang the button.
_TOOL_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class KillReport:
    """What a kill found and did.

    ``where`` names the machine or container the kill ran on, ``killed`` the
    pids it signaled, ``survivors`` the pids still present after the forced
    pass, and ``error`` the reason nothing could be attempted (a tool missing,
    a transport failure). ``checked`` is False exactly when ``error`` says the
    port could not even be inspected, which separates "nothing was listening"
    from "could not tell".
    """

    where: str
    port: int
    killed: tuple[int, ...] = ()
    survivors: tuple[int, ...] = ()
    error: str = ""
    checked: bool = True

    @property
    def nothing_found(self) -> bool:
        return self.checked and not self.killed and not self.survivors

    def describe(self) -> str:
        """One sentence for the artist, naming pid and port."""
        if not self.checked:
            return (
                f"Could not inspect port {self.port} on {self.where}: "
                f"{self.error}"
            )
        if self.nothing_found:
            return f"No ppf-cts-server was listening on port {self.port} on {self.where}."
        pids = ", ".join(str(p) for p in self.killed)
        text = f"Killed pid {pids} (port {self.port} on {self.where})."
        if self.survivors:
            left = ", ".join(str(p) for p in self.survivors)
            text += f" Still running: pid {left}."
        if self.error:
            text += f" {self.error}"
        return text


# ---------------------------------------------------------------------------
# This machine
# ---------------------------------------------------------------------------

def find_local_listener_pids(port: int) -> list[int]:
    """The pids of every process LISTENING on loopback *port*, or ``[]``.

    ``netstat -ano`` on Windows, ``lsof -ti tcp:N`` elsewhere; both ship
    with the OS. A missing tool or a parse failure raises ``OSError`` so the
    caller can report that the port could not be inspected, which is a
    different answer from "nobody is listening".
    """
    if os.name == "nt":
        r = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=_TOOL_TIMEOUT_S,
        )
        if r.returncode != 0:
            raise OSError(f"netstat exited {r.returncode}")
        suffix = f":{port}"
        pids: list[int] = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3].upper() == "LISTENING":
                if parts[1].endswith(suffix):
                    try:
                        pid = int(parts[4])
                    except ValueError:
                        continue
                    if pid not in pids:
                        pids.append(pid)
        return pids
    r = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        capture_output=True, text=True, timeout=_TOOL_TIMEOUT_S,
    )
    # lsof exits 1 when nothing matches, which is an answer, not a failure.
    if r.returncode not in (0, 1):
        raise OSError(f"lsof exited {r.returncode}: {r.stderr.strip()}")
    pids = []
    for tok in r.stdout.split():
        try:
            pids.append(int(tok))
        except ValueError:
            continue
    return pids


def _wait_port_free(port: int, deadline: float) -> list[int]:
    """Poll the port until no listener is left or *deadline* passes; return
    the listeners still present."""
    while True:
        try:
            left = find_local_listener_pids(port)
        except (OSError, subprocess.SubprocessError):
            left = []
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(0.1)


def kill_local_server(port: int) -> KillReport:
    """End whatever listens on loopback *port* of this machine, and any
    ``ppf-cts-server`` launched for that port.

    POSIX: SIGTERM every listener, wait ``_TERM_GRACE_S`` for the socket to
    close, SIGKILL what remains, then sweep a detached wrapper by command
    line. THE SWEEP CARRIES THE PORT: the rig runs one server per worker
    slot, each on its own port, and a name-wide kill would end another
    worker's solve mid-flight (``bl_server_stop_is_real`` states that
    invariant).

    Windows: ``taskkill /F /T`` on every listener, so the solver child a
    server spawned goes with it. THERE IS NO IMAGE-NAME SWEEP ON WINDOWS
    EITHER: the rig runs its Windows workers in parallel, one server each on
    its own port, and ``taskkill /F /IM ppf-cts-server.exe`` would end every
    one of them. The listener walk names the pid, which is what the sweep
    existed to do without. A socket inherited by a solver child whose parent
    is already gone (see ``WinNativeBackend.disconnect``) shows in
    ``netstat`` under the dead parent's pid; that kill fails and the pid is
    reported as a survivor rather than hidden.

    Never raises; a tool that is missing or fails is the report's ``error``.
    """
    where = "this machine"
    try:
        pids = find_local_listener_pids(port)
    except (OSError, subprocess.SubprocessError) as exc:
        return KillReport(where, port, error=str(exc), checked=False)

    errors: list[str] = []
    if os.name == "nt":
        for pid in pids:
            r = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=_TOOL_TIMEOUT_S,
                check=False,
            )
            if r.returncode != 0:
                errors.append(f"taskkill pid {pid}: {(r.stderr or r.stdout).strip()}")
    else:
        import signal

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except OSError as exc:
                errors.append(f"kill pid {pid}: {exc}")
        left = _wait_port_free(port, time.monotonic() + _TERM_GRACE_S) if pids else []
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except OSError as exc:
                errors.append(f"kill -9 pid {pid}: {exc}")
        # A detached wrapper (``nohup bash -c "...; ppf-cts-server --port N"``)
        # holds no socket, so the listener walk cannot see it.
        subprocess.run(
            ["pkill", "-f", f"{SERVER_PROCESS_NAME} .*--port {port}"],
            capture_output=True, timeout=_TOOL_TIMEOUT_S, check=False,
        )

    survivors = _wait_port_free(port, time.monotonic() + 1.0) if pids else []
    return KillReport(
        where, port,
        killed=tuple(pids),
        survivors=tuple(survivors),
        error="; ".join(errors),
    )


# ---------------------------------------------------------------------------
# A host reached through a backend
# ---------------------------------------------------------------------------

# Runs as ``/bin/sh -c`` on the solver host or inside the container. It
# prints two lines the caller parses: the pids it signaled and the pids
# still present after the forced pass.
#
# THE KILL CARRIES THE PORT, as the local sweep does. A name-wide
# ``pkill -x ppf-cts-server`` ends every server on the host, and a solver
# host is not always one person's: on a shared machine it takes down other
# people's runs, and a name-wide pass run as a test has done exactly that,
# ending a server a different session was using.
# The add-on launches the remote server as ``ppf-cts-server ... --port N``
# (``effect_runner._do_launch_server``), so the command line names the port.
#
# TWO FILTERS, AND THE SECOND IS WHAT MAKES ``-f`` SAFE. ``pgrep -f`` matches
# whole COMMAND LINES, so its candidates include any process that merely
# MENTIONS the server and the port: an editor holding the command, another
# ssh carrying it, the shell running this very script. ``pkill -f`` matches
# the same way, so it carries the same trap. Every candidate is therefore
# checked against ``ps -o comm=``, its process NAME, which none of them
# passes: a shell is named ``sh``.
#
# The issuing shell additionally happens not to match the pattern at all,
# because the port reaches the pattern as ``$PORT`` and is expanded inside
# the shell rather than written into its argv. Do not let that remove the
# name check: it is an accident of this spelling, and it says nothing about
# the other mentions above.
#
# A server started by hand WITHOUT ``--port`` on its command line is not
# matched. That is a deliberate miss: reporting nothing found and leaving it
# running is recoverable, and killing the wrong process is not.
def _remote_kill_script(port: int) -> str:
    name = SERVER_PROCESS_NAME
    return (
        f"PORT={int(port)}; PIDS=''; "
        # The trailing group bounds the number, so port 909 does not match a
        # server on 9090.
        f'for p in $(pgrep -f "{name}.*--port $PORT([^0-9]|$)" 2>/dev/null); do '
        f'case "$(ps -o comm= -p $p 2>/dev/null)" in *{name}*) PIDS="$PIDS $p";; esac; '
        "done; "
        'echo "KILLED $PIDS"; '
        '[ -n "$PIDS" ] && kill $PIDS 2>/dev/null; '
        "for i in 1 2 3 4 5; do LEFT=''; "
        'for p in $PIDS; do kill -0 $p 2>/dev/null && LEFT="$LEFT $p"; done; '
        '[ -z "$LEFT" ] && break; sleep 1; done; '
        '[ -n "$LEFT" ] && kill -9 $LEFT 2>/dev/null; '
        "sleep 1; LEFT=''; "
        'for p in $PIDS; do kill -0 $p 2>/dev/null && LEFT="$LEFT $p"; done; '
        'echo "LEFT $LEFT"; true'
    )


def _parse_pid_line(lines: list[str], tag: str) -> tuple[int, ...]:
    for line in lines:
        if line.startswith(tag + " ") or line == tag:
            out = []
            for tok in line[len(tag):].split():
                try:
                    out.append(int(tok))
                except ValueError:
                    continue
            return tuple(out)
    return ()


def kill_remote_server(exec_command, *, where: str, port: int) -> KillReport:
    """End the ``ppf-cts-server`` serving *port* on the host *exec_command*
    reaches.

    *exec_command* is a backend's ``exec_command`` (SSH runs it on the host
    or inside the container, Docker inside the container). The kill is
    scoped to *port*, not to the process name, for the reason
    ``_remote_kill_script`` records: a solver host can be shared, and a
    name-wide pass ends other people's runs. *where* labels the report.

    Never raises. A transport failure, or a host without ``pgrep`` or
    ``ps``, is the report's ``error`` with ``checked=False``.
    """
    try:
        result = exec_command(_remote_kill_script(port), shell=True)
    except Exception as exc:  # noqa: BLE001 - the report is the channel
        return KillReport(where, port, error=str(exc), checked=False)
    stdout = list(result.get("stdout") or [])
    stderr = list(result.get("stderr") or [])
    if result.get("exit_code") != 0 or not any(
        line.startswith("KILLED") for line in stdout
    ):
        detail = " ".join(stderr).strip() or " ".join(stdout).strip()
        return KillReport(
            where, port,
            error=f"kill command failed (exit {result.get('exit_code')}): "
                  f"{detail or 'no output'}",
            checked=False,
        )
    return KillReport(
        where, port,
        killed=_parse_pid_line(stdout, "KILLED"),
        survivors=_parse_pid_line(stdout, "LEFT"),
        error=" ".join(stderr).strip(),
    )
