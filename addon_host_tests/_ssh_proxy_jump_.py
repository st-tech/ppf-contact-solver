# File: addon_host_tests/_ssh_proxy_jump_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Host-side gates for the two modules that decide WHERE an SSH connection
# goes: ``blender_addon/core/ssh_config.py`` (the ~/.ssh/config reader and
# the ProxyJump chain resolver) and ``blender_addon/core/ssh_command.py``
# (the ``ssh ...`` command the SSH Command backends are configured with).
#
# Both are pure text-to-connection-parameters, so they are exercised here
# rather than in Blender. What they hand to paramiko (the hop chain, the
# panel field, the profile) is covered by the ``bl_ssh_proxy_jump`` rig
# scenario, which needs the add-on loaded.
#
# The property that makes these worth pinning is that a wrong answer is
# silent: a spec read one way opens a working connection to the wrong
# host, or resolves a jump host as the destination, and the user sees a
# connection error that names neither.

from __future__ import annotations

import os

import pytest


CONFIG_TEXT = """
Host gpu-host
    HostName 192.0.2.5
    User ubuntu
    IdentityFile ~/.ssh/id_gpu
    ProxyJump inner

Host inner
    HostName inner.example.com
    Port 2202
    User inner-user
    IdentityFile ~/.ssh/id_inner
    ProxyJump outer

Host outer
    HostName outer.example.com
    Port 2201
    User outer-user

Host direct-host
    HostName 192.0.2.9
    ProxyJump none

Host loop-a
    ProxyJump loop-b

Host loop-b
    ProxyJump loop-a

Host *
    User fallback-user
"""


@pytest.fixture(scope="module")
def config_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("ssh") / "config"
    path.write_text(CONFIG_TEXT)
    return str(path)


def chain(ssh_config, config_path, spec):
    return [
        (hop.hostname, hop.port, hop.user, hop.identity_file)
        for hop in ssh_config.resolve_jump_chain(spec, 22, config_path)
    ]


# ---------------------------------------------------------------------------
# ssh_config: the ProxyJump keyword and the chain it resolves to
# ---------------------------------------------------------------------------

def test_proxyjump_is_read_from_the_config(ssh_config, config_path):
    """The keyword reaches the resolved entry, and its absence reads as None
    rather than as an empty jump that would be dialed."""
    entry = ssh_config.resolve_ssh_config("gpu-host", 22, config_path)
    assert entry.hostname == "192.0.2.5"
    assert entry.proxy_jump == "inner"
    assert ssh_config.resolve_ssh_config("outer", 22, config_path).proxy_jump is None


def test_chain_is_ordered_outward_from_this_machine(ssh_config, config_path):
    """A jump host that carries a ProxyJump of its own contributes its hops
    AHEAD of itself: the first entry is the one reached directly.

    Ordering is the whole contract, since each hop is dialed over the
    previous one. Reversed, the add-on would ask the unreachable inner host
    to forward to the reachable bastion.
    """
    assert chain(ssh_config, config_path, "inner") == [
        ("outer.example.com", 2201, "outer-user", None),
        ("inner.example.com", 2202, "inner-user", os.path.expanduser("~/.ssh/id_inner")),
    ]


def test_hop_written_by_hand_outranks_the_config(ssh_config, config_path):
    """A user or port in the spec wins over the alias's own config, which is
    the precedence ssh gives the command line. Fields the spec leaves out
    still come from the config."""
    assert chain(ssh_config, config_path, "admin@inner:2299")[-1] == (
        "inner.example.com", 2299, "admin", os.path.expanduser("~/.ssh/id_inner"),
    )


def test_comma_separated_hops_stay_in_order(ssh_config, config_path):
    assert chain(ssh_config, config_path, "outer,admin@192.0.2.7:2022") == [
        ("outer.example.com", 2201, "outer-user", None),
        ("192.0.2.7", 2022, "admin", None),
    ]


def test_none_is_a_direct_connection(ssh_config, config_path):
    """``ProxyJump none`` is how a host opts out of a jump host a wildcard
    block set for everything else, so it must resolve to no hops rather than
    to a host literally named "none"."""
    assert ssh_config.resolve_jump_chain("none", 22, config_path) == []
    assert ssh_config.resolve_ssh_config("direct-host", 22, config_path).proxy_jump == "none"


def test_jump_loop_is_refused(ssh_config, config_path):
    """Two hosts naming each other would recurse until the stack ran out."""
    with pytest.raises(ValueError, match="loop"):
        ssh_config.resolve_jump_chain("loop-a", 22, config_path)


@pytest.mark.parametrize(
    "spec",
    ["outer,,inner", "user@", ":22", "host:notaport", "host:99999", "outer,none", "  "],
)
def test_malformed_spec_is_refused(ssh_config, config_path, spec):
    """Refused before anything is dialed. Silently dropping a hop the add-on
    cannot read would connect somewhere the user did not ask for."""
    with pytest.raises(ValueError):
        ssh_config.resolve_jump_chain(spec, 22, config_path)


def test_ipv6_literal_needs_brackets_for_a_port(ssh_config):
    """Every colon in a bare literal belongs to the address, so only the
    bracketed form can carry a port."""
    assert ssh_config.split_host_spec("[2001:db8::1]:2222") == (None, "2001:db8::1", 2222)
    assert ssh_config.split_host_spec("2001:db8::1") == (None, "2001:db8::1", None)
    assert ssh_config.split_host_spec("me@host") == ("me", "host", None)
    with pytest.raises(ValueError):
        ssh_config.split_host_spec("[2001:db8::1")


# ---------------------------------------------------------------------------
# ssh_command: the pasted command line
# ---------------------------------------------------------------------------

def test_reads_destination_and_the_options_that_matter(ssh_command):
    parsed = ssh_command.parse_ssh_command(
        "ssh -J me@bastion -p 2222 -i ~/.ssh/id_gpu ubuntu@gpu-host"
    )
    assert parsed.host == "gpu-host"
    assert parsed.port == 2222
    assert parsed.username == "ubuntu"
    assert parsed.key_path == os.path.expanduser("~/.ssh/id_gpu")
    assert parsed.proxy_jump == "me@bastion"


@pytest.mark.parametrize(
    "command,host",
    [
        # -J's argument holds an "@" and names no destination.
        ("ssh -J me@bastion gpu-host", "gpu-host"),
        # A scan that treats every bare word as a candidate host reads the
        # port as one, and then keeps it because the real host comes later.
        ("ssh -p 2222 gpu-alias", "gpu-alias"),
        ("ssh -i /keys/id gpu-alias", "gpu-alias"),
        # Anything past the second bare word is the remote command.
        ("ssh -J bastion gpu-host uname -a", "gpu-host"),
    ],
)
def test_an_option_consumes_its_own_argument(ssh_command, command, host):
    assert ssh_command.parse_ssh_command(command).host == host


def test_attached_arguments_and_clustered_flags(ssh_command):
    parsed = ssh_command.parse_ssh_command("ssh -Cv -Jme@bastion -p2222 root@gpu-host")
    assert (parsed.host, parsed.port, parsed.username, parsed.proxy_jump) == (
        "gpu-host", 2222, "root", "me@bastion",
    )


def test_dash_o_settings(ssh_command):
    """The four settings the add-on acts on are read in either spelling, an
    unread one is ignored rather than refused, and the first value of a
    setting given twice wins."""
    parsed = ssh_command.parse_ssh_command(
        'ssh -o ProxyJump=bastion -o "Port 2222" -o StrictHostKeyChecking=no host'
    )
    assert (parsed.host, parsed.port, parsed.proxy_jump) == ("host", 2222, "bastion")
    explicit = ssh_command.parse_ssh_command("ssh -p 22 -o Port=2222 host")
    assert explicit.port == 22


def test_options_after_the_destination_still_count(ssh_command):
    """ssh re-enters its option loop once it has the destination, so
    ``ssh host -p 2222`` is a working command and has to parse as one. The
    remote command ends parsing."""
    parsed = ssh_command.parse_ssh_command("ssh gpu-host -p 2222 -J bastion")
    assert (parsed.port, parsed.proxy_jump) == (2222, "bastion")
    assert ssh_command.parse_ssh_command("ssh gpu-host uname -a -p 99").port is None


def test_uri_destination(ssh_command):
    parsed = ssh_command.parse_ssh_command("ssh ssh://ubuntu@gpu-host:2222/")
    assert (parsed.host, parsed.port, parsed.username) == ("gpu-host", 2222, "ubuntu")


def test_no_destination_reports_no_host(ssh_command):
    """The panel words this one itself, so it is a None host rather than a
    raise."""
    assert ssh_command.parse_ssh_command("ssh -J bastion").host is None
    assert ssh_command.parse_ssh_command("").host is None


@pytest.mark.parametrize(
    "command",
    [
        "ssh -Z gpu-host",          # not an ssh option: the next token is unclear
        "ssh -J",                   # missing argument
        "ssh -p two gpu-host",      # not a port
        "ssh -p 70000 gpu-host",    # out of range
        'ssh -J "bastion gpu-host',  # unbalanced quote
    ],
)
def test_unusable_command_is_refused(ssh_command, command):
    """ssh would not have run any of these either. Guessing past one opens a
    connection the command did not describe."""
    with pytest.raises(ValueError):
        ssh_command.parse_ssh_command(command)
