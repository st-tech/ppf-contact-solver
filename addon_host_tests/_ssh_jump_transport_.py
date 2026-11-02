# File: addon_host_tests/_ssh_jump_transport_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Gate for the code that OPENS a ProxyJump chain, against real SSH servers.
#
# ``blender_addon/core/backends.py`` reaches a solver behind a bastion by
# opening one paramiko client per hop and, on each, a ``direct-tcpip``
# channel aimed at the next hop; the last channel becomes the ``sock=``
# the destination client connects over. Its neighbors are covered
# elsewhere and neither covers this: ``_ssh_proxy_jump_.py`` stops at the
# resolved chain and opens no socket, and the ``bl_ssh_proxy_jump`` rig
# scenario substitutes a paramiko whose ``open_channel`` returns a stub.
# So the hop loop, the channel destinations it computes, and the teardown
# are exercised here or nowhere.
#
# The servers are two ``sshd`` processes on loopback with keys generated
# per run, so the gate needs no bastion, no fixture host, and nothing from
# the developer's ``~/.ssh``. The add-on reads its config through an
# explicit path argument, which is what lets the whole chain be described
# by a file this module writes.
#
# WHAT MAKES A PASS MEAN SOMETHING: the target server is configured
# ``AllowTcpForwarding no`` while the jump server is configured ``yes``.
# A chain routed through the target is therefore refused at the channel
# open, so the passing case cannot be a direct dial that ignored the hop.
# ``test_a_hop_that_forbids_forwarding_is_refused`` is that control, and
# it fails if the hop loop is ever reduced to connecting straight to the
# destination.

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time

import pytest


paramiko = pytest.importorskip(
    "paramiko", reason="the add-on installs paramiko at run time; absent here"
)

def _sshd_path() -> str:
    """Absolute path to ``sshd``, which is not on a normal user's PATH."""
    for candidate in ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "/sbin/sshd"):
        if os.path.exists(candidate):
            return candidate
    return shutil.which("sshd") or ""


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not shutil.which("ssh-keygen") or not _sshd_path(),
    reason="needs a POSIX sshd and ssh-keygen",
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


SSHD_TEMPLATE = """\
Port {port}
ListenAddress 127.0.0.1
HostKey {hostkey}
AuthorizedKeysFile {authorized_keys}
PidFile {pidfile}
# The lab lives under a temp directory, whose permissions StrictModes rejects.
StrictModes no
UsePAM no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AllowTcpForwarding {forwarding}
PrintMotd no
LogLevel VERBOSE
"""

CONFIG_TEMPLATE = """\
Host jump-a
    HostName 127.0.0.1
    Port {jump_port}
    User {user}
    IdentityFile {key}

Host target-b
    HostName 127.0.0.1
    Port {target_port}
    User {user}
    IdentityFile {key}
    ProxyJump jump-a

Host target-via-dead
    HostName 127.0.0.1
    Port {target_port}
    User {user}
    IdentityFile {key}
    ProxyJump dead-hop

Host dead-hop
    HostName 127.0.0.1
    Port {dead_port}
    User {user}
    IdentityFile {key}
"""


class _Lab:
    """Paths and ports of a running two-server lab."""

    def __init__(self, root, user, jump_port, target_port, dead_port):
        self.root = root
        self.user = user
        self.jump_port = jump_port
        self.target_port = target_port
        self.dead_port = dead_port
        self.key = str(root / "id_test")
        self.config = str(root / "ssh_config")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """Two sshd processes on loopback, plus the config file naming them.

    Skips rather than fails when a server does not come up: this module
    gates the add-on's hop loop, not the portability of running sshd as an
    unprivileged user, and a platform that refuses that should not be read
    as the add-on being broken. The server log goes into the skip reason so
    the difference is visible rather than guessed at.
    """
    sshd = _sshd_path()
    if not sshd:
        pytest.skip("no sshd binary")

    root = tmp_path_factory.mktemp("jumplab")
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "root"

    def keygen(name):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)],
            check=True,
        )

    keygen("id_test")
    keygen("hostkey_jump")
    keygen("hostkey_target")

    authorized = root / "authorized_keys"
    authorized.write_text((root / "id_test.pub").read_text())
    authorized.chmod(0o600)

    jump_port, target_port = _free_port(), _free_port()
    # Bound and released, so nothing answers on it for the whole run.
    dead_port = _free_port()

    processes = []
    for name, port, forwarding in (
        ("jump", jump_port, "yes"),
        ("target", target_port, "no"),
    ):
        conf = root / f"sshd_{name}.conf"
        conf.write_text(
            SSHD_TEMPLATE.format(
                port=port,
                hostkey=root / f"hostkey_{name}",
                authorized_keys=authorized,
                pidfile=root / f"sshd_{name}.pid",
                forwarding=forwarding,
            )
        )
        log = open(root / f"sshd_{name}.log", "wb")
        processes.append(
            (
                name,
                subprocess.Popen(
                    [sshd, "-f", str(conf), "-D", "-e"], stdout=log, stderr=log
                ),
                log,
            )
        )

    def reachable(port):
        with socket.socket() as probe:
            probe.settimeout(0.25)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if reachable(jump_port) and reachable(target_port):
            break
        time.sleep(0.1)
    else:
        for _, proc, log in processes:
            proc.terminate()
            log.close()
        logs = "".join(
            (root / f"sshd_{name}.log").read_text(errors="ignore")
            for name, _, _ in processes
        )
        pytest.skip(f"sshd did not start in this environment: {logs[-400:]}")

    (root / "ssh_config").write_text(
        CONFIG_TEMPLATE.format(
            jump_port=jump_port,
            target_port=target_port,
            dead_port=dead_port,
            user=user,
            key=root / "id_test",
        )
    )

    yield _Lab(root, user, jump_port, target_port, dead_port)

    for _, proc, log in processes:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        log.close()


def _hops(ssh_config, lab, alias):
    """The resolved chain for *alias*, in the dict shape backends expects."""
    entry = ssh_config.resolve_ssh_config(alias, 22, lab.config)
    chain = ssh_config.resolve_jump_chain(entry.proxy_jump, 22, lab.config)
    return entry, [
        {
            "host": hop.hostname,
            "port": hop.port,
            "username": hop.user,
            "key_path": hop.identity_file,
        }
        for hop in chain
    ]


# ---------------------------------------------------------------------------
# The chain the add-on resolves from a config file it is handed
# ---------------------------------------------------------------------------

def test_the_hop_is_read_from_the_given_config_file(ssh_config, lab):
    entry, hops = _hops(ssh_config, lab, "target-b")
    assert entry.port == lab.target_port
    assert [(h["host"], h["port"]) for h in hops] == [("127.0.0.1", lab.jump_port)]
    assert hops[0]["key_path"] == lab.key


def test_a_missing_config_file_does_not_fall_back_to_the_users_own(ssh_config, lab):
    """An unreadable path must answer with no ProxyJump, not with ~/.ssh/config.

    The add-on always reads the user's config in production, so the path
    argument exists for callers that must not: a fallback would make this
    module's result depend on the developer's own file.
    """
    entry = ssh_config.resolve_ssh_config(
        "target-b", 22, str(lab.root / "does-not-exist")
    )
    assert not entry.proxy_jump


# ---------------------------------------------------------------------------
# The transport itself
# ---------------------------------------------------------------------------

def test_a_command_runs_through_the_hop(ssh_config, backends, lab):
    """The end-to-end property: a session on the destination, over the hop."""
    entry, hops = _hops(ssh_config, lab, "target-b")
    clients, sock = backends._open_jump_chain(
        paramiko, hops, entry.hostname, entry.port, 30
    )
    try:
        assert len(clients) == len(hops)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=entry.hostname,
            port=entry.port,
            username=entry.user,
            key_filename=entry.identity_file,
            sock=sock,
            compress=True,
        )
        try:
            _, stdout, _ = client.exec_command("echo tunneled; echo $SSH_CONNECTION")
            lines = stdout.read().decode().split()
            assert lines[0] == "tunneled"
            # The destination reports the connection arriving on ITS port,
            # which is the hop's forwarded socket rather than a direct dial.
            assert lines[-1] == str(lab.target_port)
        finally:
            client.close()
    finally:
        backends._close_jump_clients(clients)


def test_teardown_closes_every_hop(ssh_config, backends, lab):
    entry, hops = _hops(ssh_config, lab, "target-b")
    clients, _ = backends._open_jump_chain(
        paramiko, hops, entry.hostname, entry.port, 30
    )
    assert all(c.get_transport().is_active() for c in clients)
    backends._close_jump_clients(clients)
    assert not any(
        c.get_transport() and c.get_transport().is_active() for c in clients
    )


# ---------------------------------------------------------------------------
# Controls. Each of these must refuse, or the passing case above is empty.
# ---------------------------------------------------------------------------

def test_a_hop_that_forbids_forwarding_is_refused(backends, lab):
    """THE control for this module. See the header.

    The destination server runs ``AllowTcpForwarding no``, so asking it to
    carry a chain must fail at the channel open even though connecting to
    it directly succeeds, as the two tests above do. If the hop loop ever
    stops opening a channel and dials the destination instead, this is the
    test that notices.
    """
    hops = [
        {
            "host": "127.0.0.1",
            "port": lab.target_port,
            "username": lab.user,
            "key_path": lab.key,
        }
    ]
    with pytest.raises(Exception) as excinfo:
        backends._open_jump_chain(
            paramiko, hops, "127.0.0.1", lab.target_port, 30
        )
    assert "prohibited" in str(excinfo.value).lower()


def test_a_hop_with_an_unusable_key_is_refused(backends, lab):
    hops = [
        {
            "host": "127.0.0.1",
            "port": lab.jump_port,
            "username": lab.user,
            "key_path": str(lab.root / "hostkey_target"),
        }
    ]
    with pytest.raises(Exception) as excinfo:
        backends._open_jump_chain(
            paramiko, hops, "127.0.0.1", lab.target_port, 30
        )
    assert "authentication" in str(excinfo.value).lower()


def test_a_refusing_hop_is_named_with_its_port(ssh_config, backends, lab):
    """The message has to identify WHICH hop failed.

    A chain reports one error for several hosts, so a message that names
    none of them leaves the user guessing which link is down.
    """
    entry, hops = _hops(ssh_config, lab, "target-via-dead")
    with pytest.raises(Exception) as excinfo:
        backends._open_jump_chain(
            paramiko, hops, entry.hostname, entry.port, 30
        )
    message = str(excinfo.value)
    assert message.startswith("Jump host")
    assert f"127.0.0.1:{lab.dead_port}" in message
