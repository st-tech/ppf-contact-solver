# File: addon_host_tests/_bounded_command_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""A bounded remote command runs, and is killed on time, on a host with no
GNU `timeout`.

macOS ships no `timeout`, so a bounded command prefixed with a bare
`timeout --signal=KILL` fails on a Mac solver host with `command not found:
timeout` before it starts, and Connect stops at the build listing.
`core.backends.bounded_command` falls back to Perl there.
These tests RUN the wrapper under `/bin/sh` twice, with the host's PATH and with
a PATH holding no `timeout`, so the fallback is exercised on a Linux developer
machine as well as a Mac.

The kill is checked on a command that runs as a CHILD of `/bin/sh -c`: killing
the shell alone left such a child running and holding the channel open (a
`sleep 30` bounded at 1 s ran 30 s), which is why both paths kill the group.
"""

import os
import shutil
import subprocess
import time

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def backends():
    return load_addon_module("core.backends")


def _path_without_timeout(tmp_path):
    """A PATH directory holding the tools the commands below use, and no
    `timeout`."""
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("perl", "sleep", "echo", "ls"):
        found = shutil.which(name)
        if found is None:
            pytest.skip(f"{name} is not on this host")
        os.symlink(found, tools / name)
    return str(tools)


def _paths(tmp_path):
    out = [("without timeout", _path_without_timeout(tmp_path))]
    if shutil.which("timeout") is not None:
        out.append(("with timeout", os.environ["PATH"]))
    return out


def _run(command, path):
    start = time.monotonic()
    done = subprocess.run(
        ["/bin/sh", "-c", command],
        env=dict(os.environ, PATH=path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done, time.monotonic() - start


def test_output_and_exit_status_pass_through(backends, tmp_path):
    command = backends.bounded_command(
        "echo 'one quoted'; echo \"two words\"; exit 3", 5
    )
    for label, path in _paths(tmp_path):
        done, _ = _run(command, path)
        assert done.stdout == "one quoted\ntwo words\n", (label, done.stderr)
        assert done.returncode == 3, (label, done.stderr)


def test_a_hung_child_of_the_shell_is_killed_on_time(backends, tmp_path):
    # `sleep` runs as a child of the `/bin/sh -c` the wrapper starts, and the
    # trailing `echo` keeps the shell from replacing itself with it.
    command = backends.bounded_command("sleep 30; echo late", 1)
    for label, path in _paths(tmp_path):
        done, seconds = _run(command, path)
        assert seconds < 10.0, (label, seconds)
        assert done.returncode == 137, (label, done.returncode, done.stderr)
        assert "late" not in done.stdout, label


def test_the_ssh_backend_bounds_a_command_a_mac_can_run(backends, tmp_path):
    """What `SSHBackend.exec_command(timeout=...)` sends runs on a host with
    no `timeout`, from the directory it names."""

    class _Chan:
        def recv_exit_status(self):
            return 0

    class _Stream:
        channel = _Chan()

        def read(self):
            return b""

    class _Instance:
        def __init__(self):
            self.commands = []

        def exec_command(self, command, timeout=None):
            self.commands.append(command)
            return None, _Stream(), _Stream()

    be = object.__new__(backends.SSHBackend)
    be._instance = _Instance()
    be._directory = str(tmp_path)
    be._container = ""
    be.exec_command("ls -d .", shell=True, timeout=5)
    sent = be._instance.commands[-1]
    assert not sent.split("&&")[-1].strip().startswith("timeout "), sent
    done, _ = _run(sent, _path_without_timeout(tmp_path))
    assert done.returncode == 0, done.stderr
    assert done.stdout == ".\n"
