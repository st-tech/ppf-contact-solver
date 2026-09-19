# File: addon_host_tests/_remote_exec_contract_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""What a remote backend GUARANTEES about the command it sends.

Two properties that everything above them assumes, that nothing asserted, and
that both fail silently on the machines where they matter.

THE POSIX WRAPPER. `core/remote_builds.probe_command` emits POSIX `sh`: `if [ -f
... ]; then ...; fi`. What runs it is the solver host's LOGIN shell, which is
not always POSIX. On a host whose user runs fish, the bare script dies with
`fish: Missing end to balance this if statement`, the probe records a failure,
and the Compute Device rows have nothing to offer on a host that holds both
builds. It works today only because `exec_command(shell=True)` wraps the command
in `/bin/sh -c` first. That wrapper is one line with no test, and the failure it
prevents cannot appear on a developer's own machine unless their login shell is
one of the unusual ones. Measured on a real fish host: wrapped, rc 0 and both
markers; bare, a parse error.

The port gate that Docker-over-SSH needs (`docker port` answering before a
launch is attempted) is the same class of property, and it is checked where the
launch is actually driven: `bl_remote_device_select` check J.
"""

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def backends():
    return load_addon_module("core.backends")


class _Chan:
    def __init__(self, code):
        self._code = code

    def recv_exit_status(self):
        return self._code


class _Stream:
    def __init__(self, text="", code=0):
        self._text = text
        self.channel = _Chan(code)

    def read(self):
        return self._text.encode()


class _Instance:
    """Records what the backend asked the far side to run."""

    def __init__(self, exit_code=0, stdout=""):
        self.commands = []
        self._exit_code = exit_code
        self._stdout = stdout

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        return None, _Stream(self._stdout, self._exit_code), _Stream("", 0)


def _ssh_backend(backends, *, container=""):
    """An SSHBackend with only the attributes `exec_command` touches.

    Built without the constructor on purpose: connecting is what the
    constructor does, and none of it bears on the command composed here.
    """
    be = object.__new__(backends.SSHBackend)
    be._instance = _Instance()
    be._directory = "/home/u/solver"
    be._container = container
    return be


def test_a_shell_command_is_wrapped_in_bin_sh_so_a_fish_login_shell_still_runs_it(backends):
    be = _ssh_backend(backends)
    be.exec_command("if [ -f x ]; then echo y; fi", shell=True)
    sent = be._instance.commands[-1]
    assert "/bin/sh -c " in sent, sent
    # And the script reaches that shell as ONE argument, not as loose words the
    # login shell would parse itself.
    assert "if [ -f x ]" in sent
    assert sent.index("/bin/sh -c") < sent.index("if [ -f x ]")


def test_the_probe_command_specifically_goes_through_that_wrapper(backends):
    """The caller that made this matter, named so a change to it is noticed."""
    remote_builds = load_addon_module("core.remote_builds")
    be = _ssh_backend(backends)
    be.exec_command(remote_builds.probe_command("/home/u/solver"), shell=True)
    sent = be._instance.commands[-1]
    assert "/bin/sh -c " in sent, sent


def test_an_unwrapped_command_is_left_alone(backends):
    """`shell=False` must NOT wrap, or a bare argv gains a shell it never asked
    for and its quoting changes meaning."""
    be = _ssh_backend(backends)
    be.exec_command("nvidia-smi --query-gpu=name --format=csv")
    assert "/bin/sh -c" not in be._instance.commands[-1]


def test_a_container_command_is_still_wrapped_before_docker_exec(backends):
    be = _ssh_backend(backends, container="ppf-test")
    be.exec_command("if [ -f x ]; then echo y; fi", shell=True)
    sent = be._instance.commands[-1]
    assert sent.startswith("docker exec "), sent
    assert "/bin/sh -c " in sent, sent


class _DockerContainer:
    """A stand-in for the docker-py container object.

    `DockerBackend` reaches the far side through `exec_run`, not through
    paramiko's `exec_command`, so it needs its own recorder: a stand-in shaped
    like the SSH one records nothing and the assertion below would fail on an
    empty list rather than on the property it is about.
    """

    def __init__(self):
        self.commands = []

    def exec_run(self, command, workdir=None, demux=False):
        self.commands.append(command)
        return 0, (b"", b"")


def test_the_plain_docker_backend_wraps_too(backends):
    """`DockerBackend` carries its OWN copy of the wrapper.

    Two backends spell this independently, so a change made in one and not the
    other leaves half the transports exposed. That is the same shape as the
    defect this file's neighbours cover: one property, two call sites, and
    nothing comparing them.
    """
    be = object.__new__(backends.DockerBackend)
    be._instance = _DockerContainer()
    be._directory = "/root/ppf-contact-solver"
    be._container = "ppf-test"
    be.exec_command("if [ -f x ]; then echo y; fi", shell=True)
    sent = be._instance.commands[-1]
    assert "/bin/sh -c " in sent, sent
    assert "if [ -f x ]" in sent
