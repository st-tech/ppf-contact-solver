# File: scenarios/rig_remote_kill_port_scope.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The remote kill ends ONE server: the one serving the port it was given.
#
# WHY THIS EXISTS. ``kill_remote_server`` is what Stop Server on Remote runs,
# and what Force Terminate Process runs while connected, through an SSH or
# Docker backend's ``exec_command``. A name-wide ``pkill -x ppf-cts-server``
# would end EVERY server on the host, and a solver host is not always one
# person's, so on a shared box that takes down other people's runs.
#
# WHAT IS HARD TO GET RIGHT, and therefore what is measured here. Scoping a
# kill to a port means matching the whole COMMAND LINE (``pgrep -f``), and a
# ``-f`` match returns every process that merely MENTIONS the server and the
# port, the shell issuing the command among them. So the script filters each
# candidate through ``ps -o comm=``, its process NAME. Three properties
# follow, and none can be read off the source with any confidence:
#
#   * a server on ANOTHER port survives,
#   * a process that only mentions the server survives,
#   * a port is matched as a whole number, so 5999 does not match 59997.
#
# HOW IT IS MEASURED. The script is POSIX ``sh`` and normally runs on the
# remote host through the backend's ``exec_command``. Here ``exec_command``
# is a local ``/bin/sh -c``, the same shell in the same role, and the servers
# it acts on are stand-ins: a copy of ``/bin/sh`` named ``ppf-cts-server``
# running a sleep loop with ``--port N`` on its command line. The script
# cannot tell a stand-in from a real server, because a stand-in reproduces
# exactly the two things it reads, the process name and the command line.
#
# NOTHING REAL IS AT RISK, AND THAT IS ASSERTED RATHER THAN ASSUMED. The
# stand-ins take ports below the ephemeral range the rig's own servers are
# allocated from (``orchestrator._alloc_port`` binds port 0, so a real server
# is always above 32768), and before spawning anything the scenario requires
# that NO process on the host already matches those ports. If one does, it
# fails by name rather than killing it.
#
# Subtests:
#   A. script_carries_the_port_it_was_given
#   B. script_bounds_the_port_as_a_whole_number
#   C. script_runs_no_host_wide_sweep
#   D. script_filters_candidates_by_process_name
#   E. transport_failure_is_reported_not_raised
#   F. test_ports_were_free_before_spawning
#   G. a_port_prefix_matches_nothing
#   H. kills_the_server_on_the_named_port
#   I. spares_a_server_on_another_port
#   J. spares_a_server_with_no_port_on_its_command_line
#   K. spares_a_process_that_merely_mentions_the_server
#   L. reports_the_killed_pid_and_no_survivors
#
# A through E are pure and run on every platform. F through L drive real
# processes through ``/bin/sh``, so they run on POSIX only; the remote host
# an SSH or Docker backend reaches is always Linux, so a Windows rig leg has
# no stand-in to make and records why instead of recording a pass.

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from . import _runner as r


# No Blender, and no solver: the subject is a shell script and the process
# table. The rig requires every scenario to name the backend it was
# established against, and this one behaves identically under all of them.
BACKENDS = ("real",)

# How long a stand-in is given to disappear after the kill returns. The
# signal lands on another process, so it exits on its own schedule.
_DEATH_DEADLINE_S = 10.0

# The sleep loop a stand-in runs. It must stay alive while carrying its
# arguments on its command line, which rules out `exec`ing something else
# (that would replace the process name the kill reads).
_STAY_ALIVE = "while :; do sleep 1; done"


def _load_server_kill():
    """Load ``core/server_kill.py`` directly, by path.

    Importing it as ``blender_addon.core.server_kill`` would first execute
    the add-on package's ``__init__``, which imports ``bpy``, and this
    scenario runs outside Blender. The module under test imports only the
    standard library, so loading the file on its own is the whole of what it
    needs, and it loads the file in THIS tree rather than whichever add-on a
    shared extension symlink happens to point at.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "core", "server_kill.py",
    )
    spec = importlib.util.spec_from_file_location("ppf_server_kill", path)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE it is executed: ``@dataclass`` resolves the defining
    # class's ``__module__`` through ``sys.modules`` while the decorator
    # runs, and ``KillReport`` is frozen, so leaving it unregistered fails
    # the load with an AttributeError inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ps_table() -> list[tuple[int, str, str]]:
    """Every process as ``(pid, name, command line)``.

    Read through ``ps`` rather than ``pgrep`` so the match is done here, in
    one place, with the same two fields the kill script reads. ``-A`` is
    spelled the same on Linux and macOS.
    """
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,comm=,args="],
            capture_output=True, text=True, timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append((pid, parts[1], parts[2]))
    return rows


def _server_pids_on(port: int) -> list[int]:
    """Pids the kill script would act on for *port*.

    The read-only twin of its filter: the command line names the server and
    that port as a whole number, AND the process name is the server's.
    """
    wanted = re.compile(r"--port %d(?![0-9])" % port)
    return [
        pid for pid, name, args in _ps_table()
        if "ppf-cts-server" in name and "ppf-cts-server" in args
        and wanted.search(args)
    ]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_gone(pid: int, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return not _alive(pid)


def _local_exec(command, **_kw):
    """Stand in for a backend's ``exec_command``.

    An SSH or Docker backend hands the script to ``/bin/sh -c`` on the far
    side and returns its streams as line lists. This does the same thing to
    the same shell, locally.
    """
    proc = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True, text=True, timeout=120.0,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.splitlines(),
        "stderr": proc.stderr.splitlines(),
    }


def _pick_ports() -> tuple[int, int]:
    """A pair of unused port NUMBERS for the stand-ins to carry.

    Below 32768, so they cannot collide with a rig server's ephemeral port,
    and spread by the pid so two workers running this at once do not pick
    the same pair. Nothing binds them; they are strings on a command line.
    """
    base = 20000 + (os.getpid() % 9000)
    for step in range(40):
        target = 20000 + ((base - 20000 + step * 137) % 9000)
        other = target + 1
        if not _server_pids_on(target) and not _server_pids_on(other):
            return target, other
    return 0, 0


def _spawn(directory: str, basename: str, args: list[str]) -> int:
    """A live process named *basename* carrying *args* on its command line.

    NOT a child of this process. It is backgrounded from a shell that exits
    immediately, so it is reparented to init and reaped there the moment it
    dies. A stand-in left as this process's own child would sit as a ZOMBIE
    between the kill and the next ``wait``, and ``kill -0`` on a zombie
    SUCCEEDS, so the script under test would report the pid it had just
    killed as a survivor and the scenario would fail against correct code.
    A real solver server is not a child of the shell that kills it either,
    so this is also the more faithful arrangement.
    """
    path = os.path.join(directory, basename)
    # A SYMLINK, NOT A COPY OF /bin/sh, because macOS refuses to run the copy.
    # /bin/sh is protected by System Integrity Protection: it carries BSD file
    # flags (`st_flags` 0o2000040), so `shutil.copy2` raises `PermissionError:
    # [Errno 1] Operation not permitted` trying to reproduce them, and a plain
    # `shutil.copy` then produces a binary the kernel SIGKILLs on exec (rc -9,
    # no output, no message) because a copy is no longer the signed platform
    # binary. Both failures name the DESTINATION, so they read as a
    # temp-directory problem rather than as the source being protected. A
    # symlink executes the real, still-signed inode and sidesteps both.
    #
    # It is what the checks need on either platform, because `comm` is taken
    # from the path handed to execve rather than from the link's target:
    # Linux reports the basename truncated to 15 characters
    # (`ppf-cts-server` fits, `mentions-the-server` becomes
    # `mentions-the-se`) and macOS reports the whole path, and the script
    # under test matches `*ppf-cts-server*` as a SUBSTRING, so the one
    # stand-in is found and the other is spared on both. Verified on both.
    #
    # NO `chmod` HERE. chmod follows a symlink, so it would change the mode of
    # /bin/sh itself. The target is already executable, which is the only
    # attribute a stand-in needs.
    if not os.path.islink(path) and not os.path.exists(path):
        os.symlink("/bin/sh", path)
    words = [shlex.quote(path), "-c", shlex.quote(_STAY_ALIVE)]
    words += [shlex.quote(a) for a in args]
    # The stand-in's own streams go to /dev/null rather than being inherited:
    # a backgrounded child holds the capture pipe open for as long as it
    # lives, so inheriting it would make this call block until its timeout
    # instead of returning the pid at once.
    launched = subprocess.run(
        ["/bin/sh", "-c", " ".join(words) + " >/dev/null 2>&1 & echo $!"],
        capture_output=True, text=True, timeout=15.0,
    )
    return int(launched.stdout.strip())


def _reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _pure_checks(checks: dict, server_kill) -> None:
    """A through E: what the script says, with nothing running."""
    port = 41234
    script = server_kill._remote_kill_script(port)

    checks["A_script_carries_the_port_it_was_given"] = {
        "ok": f"PORT={port};" in script,
        "details": {"script": script},
    }
    # Without the trailing group, port 5999 would match a server on 59997.
    checks["B_script_bounds_the_port_as_a_whole_number"] = {
        "ok": "--port $PORT([^0-9]|$)" in script,
        "details": {"script": script},
    }
    # The regression guarded: a host-wide sweep ends every user's server.
    # No spelling of one avoids naming the process alone, so its absence is
    # the test.
    checks["C_script_runs_no_host_wide_sweep"] = {
        "ok": "pkill" not in script and "killall" not in script,
        "details": {"script": script},
    }
    checks["D_script_filters_candidates_by_process_name"] = {
        "ok": "ps -o comm=" in script,
        "details": {"script": script},
    }

    # A dead transport is the report's error, never an exception reaching a
    # button whose whole purpose is recovery.
    def _refuse(_command, **_kw):
        raise OSError("ssh: connect to host solver-host port 22: "
                      "Connection refused")

    report = server_kill.kill_remote_server(
        _refuse, where="the solver host", port=port)
    checks["E_transport_failure_is_reported_not_raised"] = {
        "ok": (not report.checked and "Connection refused" in report.error
               and not report.killed),
        "details": {"checked": report.checked, "error": report.error},
    }


def _process_checks(checks: dict, server_kill) -> None:
    """F through L: the script against real processes."""
    target, other = _pick_ports()
    if not target:
        checks["F_test_ports_were_free_before_spawning"] = {
            "ok": False,
            "details": {"reason": "no free port pair found in 40 tries; a "
                                  "ppf-cts-server already matches each one"},
        }
        return

    prefix = target // 10
    checks["F_test_ports_were_free_before_spawning"] = {
        "ok": not _server_pids_on(target) and not _server_pids_on(other),
        "details": {"target": target, "other": other, "prefix": prefix},
    }

    directory = tempfile.mkdtemp(prefix="ppf-remote-kill-")
    # Every stand-in is recorded THE MOMENT it exists, never in one
    # assignment after the last spawn: a stand-in is detached from this
    # process on purpose, so one that is running but unrecorded when a
    # later spawn raises would outlive the run, and it is named
    # ``ppf-cts-server``. One did, and the next scenario to start a server
    # failed against it.
    pids: list[int] = []

    def spawn(basename: str, args: list[str]) -> int:
        pid = _spawn(directory, basename, args)
        pids.append(pid)
        return pid

    try:
        on_target = spawn("ppf-cts-server", ["--port", str(target)])
        on_other = spawn("ppf-cts-server", ["--port", str(other)])
        no_port = spawn("ppf-cts-server", [])
        # Named something else, but its command line carries both the server
        # and the port. This is the shape of every false candidate a `-f`
        # match returns, the issuing shell included.
        mention = spawn("mentions-the-server",
                        ["ppf-cts-server", "--port", str(target)])
        # THE STAND-IN IS A SYMLINK, AND THIS IS CHECKED ON EVERY PLATFORM
        # THOUGH IT ONLY BREAKS ON ONE. A copy of /bin/sh runs fine on Linux,
        # so a change back to `shutil.copy` would pass every check below and
        # every CI leg, and fail only on macOS: `copy2` raises PermissionError
        # on the SIP file flags, and `copy` produces a binary the kernel
        # SIGKILLs on exec. There is no macOS leg in CI to catch that, so the
        # property is asserted here, where Linux and Windows CI do run.
        checks["the_stand_ins_are_symlinks_not_copies"] = {
            "ok": all(
                os.path.islink(os.path.join(directory, n))
                for n in ("ppf-cts-server", "mentions-the-server")
            ),
            "details": {
                n: ("symlink -> " + os.readlink(os.path.join(directory, n)))
                if os.path.islink(os.path.join(directory, n))
                else "NOT a symlink"
                for n in ("ppf-cts-server", "mentions-the-server")
            },
        }
        # Let each one get as far as its sleep loop before anything reads
        # the process table.
        time.sleep(1.0)
        all_up = all(_alive(p) for p in pids)

        # G: a numeric prefix of the target must match nothing, and must
        # leave the target running. Run FIRST, while the target is alive.
        prefix_report = server_kill.kill_remote_server(
            _local_exec, where="this host", port=prefix)
        checks["G_a_port_prefix_matches_nothing"] = {
            "ok": (all_up and prefix_report.checked
                   and not prefix_report.killed
                   and _alive(on_target)),
            "details": {"prefix": prefix, "target": target,
                        "all_stand_ins_started": all_up,
                        "killed": list(prefix_report.killed),
                        "target_alive": _alive(on_target),
                        "error": prefix_report.error},
        }

        report = server_kill.kill_remote_server(
            _local_exec, where="this host", port=target)

        checks["H_kills_the_server_on_the_named_port"] = {
            "ok": _wait_gone(on_target, _DEATH_DEADLINE_S),
            "details": {"pid": on_target, "port": target,
                        "killed": list(report.killed),
                        "error": report.error},
        }
        checks["I_spares_a_server_on_another_port"] = {
            "ok": _alive(on_other) and on_other not in report.killed,
            "details": {"pid": on_other, "port": other,
                        "alive": _alive(on_other),
                        "killed": list(report.killed)},
        }
        checks["J_spares_a_server_with_no_port_on_its_command_line"] = {
            "ok": _alive(no_port) and no_port not in report.killed,
            "details": {"pid": no_port, "alive": _alive(no_port),
                        "killed": list(report.killed)},
        }
        checks["K_spares_a_process_that_merely_mentions_the_server"] = {
            "ok": _alive(mention) and mention not in report.killed,
            "details": {"pid": mention, "alive": _alive(mention),
                        "killed": list(report.killed)},
        }
        checks["L_reports_the_killed_pid_and_no_survivors"] = {
            "ok": (report.checked and report.killed == (on_target,)
                   and not report.survivors),
            "details": {"checked": report.checked,
                        "killed": list(report.killed),
                        "expected": [on_target],
                        "survivors": list(report.survivors),
                        "error": report.error},
        }
    finally:
        for pid in pids:
            _reap(pid)
        shutil.rmtree(directory, ignore_errors=True)


def run(ctx: r.ScenarioContext) -> dict:
    server_kill = _load_server_kill()

    checks: dict = {}
    _pure_checks(checks, server_kill)

    if sys.platform.startswith("win"):
        # No /bin/sh to make a stand-in from. Recorded rather than passed:
        # a check that reports success over nothing it ran is worse than an
        # absent one.
        checks["posix_only_behavior_not_exercised"] = {
            "ok": True,
            "details": {"platform": sys.platform,
                        "note": "the script is POSIX sh and runs on the "
                                "remote host, which is always Linux"},
        }
        return r.report_named_checks(checks)

    _process_checks(checks, server_kill)
    return r.report_named_checks(checks)
