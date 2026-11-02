# File: ssh_command.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Parser for the ``ssh ...`` shell command the SSH Command backends are
# configured with. It reads the destination and the few options the add-on
# acts on (port, identity file, login name, jump hosts) and ignores the rest.
#
# Options are matched against the real ssh(1) option set rather than by
# "starts with a dash", because an option that takes an argument has to
# consume it: in ``ssh -p 2222 gpu-host`` the destination is the third token,
# and in ``ssh -J me@bastion gpu-host`` the second token holds an ``@`` while
# naming no destination at all.

from dataclasses import dataclass
from typing import List, Optional
import os
import shlex

from .ssh_config import split_host_spec

# ssh(1) options that take a separate argument.
_ARG_OPTIONS = set("BbcDEeFIiJLlmOoQpRSWw")

# ssh(1) options that stand alone.
_FLAG_OPTIONS = set("46AaCfGgKkMNnqsTtVvXxYy")

# -o settings the add-on acts on. Every other setting is accepted and
# ignored, the same way an unread ssh_config keyword is: the add-on only
# needs enough of the command to open the connection.
_READ_OPTIONS = ("proxyjump", "port", "user", "identityfile")


@dataclass
class SSHCommand:
    """What an ``ssh ...`` command says about the connection to open.

    Every field except ``host`` is None when the command does not set it, so
    a caller can fall back to the ssh config or to its own default without
    having to tell "unset" from "set to the default".
    """

    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    key_path: Optional[str] = None
    proxy_jump: Optional[str] = None


def parse_ssh_command(command: str) -> SSHCommand:
    """Parse an ``ssh ...`` command line.

    Returns an SSHCommand whose ``host`` is None when the command names no
    destination, which is the one malformed case the panel reports in its own
    words. Everything else raises ValueError: an unknown option or a missing
    option argument means the command would not have run under ssh either, and
    guessing past it would open a connection the user did not ask for.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Cannot split the command into arguments: {exc}") from exc

    result = SSHCommand()
    if not tokens:
        return result

    index = 0
    if os.path.splitext(os.path.basename(tokens[0]))[0] == "ssh":
        index = 1

    while index < len(tokens):
        token = tokens[index]
        index += 1

        if token == "--":
            # Everything after it is positional: the destination, then the
            # remote command.
            if index < len(tokens) and result.host is None:
                _set_destination(result, tokens[index])
            return result

        if len(token) > 1 and token.startswith("-"):
            index = _read_option(result, token, tokens, index)
            continue

        if result.host is not None:
            # The second positional token starts the remote command, which
            # the add-on does not run, and nothing past it is an option.
            return result

        # ssh re-enters its option loop once it has the destination, so
        # `ssh gpu-host -p 2222` sets the port the way `ssh -p 2222 gpu-host`
        # does. Keep reading rather than stopping here.
        _set_destination(result, token)

    return result


def _read_option(result: SSHCommand, token: str, tokens: List[str], index: int) -> int:
    """Consume one option token, and its argument when it takes one.

    Single-letter options cluster (``-Cv``), and the argument of the option
    that takes one may be attached (``-p2222``) or separate (``-p 2222``).
    Returns the index of the next unread token.
    """
    position = 1
    while position < len(token):
        letter = token[position]
        position += 1
        if letter in _FLAG_OPTIONS:
            continue
        if letter not in _ARG_OPTIONS:
            raise ValueError(f"Unrecognized ssh option '-{letter}'")

        value = token[position:]
        if not value:
            if index >= len(tokens):
                raise ValueError(f"Option '-{letter}' is missing its argument")
            value = tokens[index]
            index += 1
        _apply_option(result, letter, value)
        break

    return index


def _apply_option(result: SSHCommand, letter: str, value: str) -> None:
    """Record the value of an option the add-on acts on.

    The first occurrence wins, which is how ssh treats a setting given twice,
    and it is also what lets ``-p`` outrank a later ``-o Port=``.
    """
    if letter == "p":
        _set_once(result, "port", _parse_port(value))
    elif letter == "i":
        _set_once(result, "key_path", os.path.expanduser(value))
    elif letter == "l":
        _set_once(result, "username", value)
    elif letter == "J":
        _set_once(result, "proxy_jump", value)
    elif letter == "o":
        _apply_setting(result, value)


def _apply_setting(result: SSHCommand, value: str) -> None:
    """Record an ``-o Keyword=Value`` (or ``-o "Keyword Value"``) setting."""
    if "=" in value:
        keyword, _, setting = value.partition("=")
    else:
        keyword, _, setting = value.partition(" ")
    keyword = keyword.strip().lower()
    setting = setting.strip()
    if not setting or keyword not in _READ_OPTIONS:
        return
    if keyword == "proxyjump":
        _set_once(result, "proxy_jump", setting)
    elif keyword == "port":
        _set_once(result, "port", _parse_port(setting))
    elif keyword == "user":
        _set_once(result, "username", setting)
    elif keyword == "identityfile":
        _set_once(result, "key_path", os.path.expanduser(setting))


def _set_once(result: SSHCommand, field: str, value) -> None:
    if getattr(result, field) is None:
        setattr(result, field, value)


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a port number") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"Port {port} is outside 1-65535")
    return port


def _set_destination(result: SSHCommand, token: str) -> None:
    """Read ``[user@]host`` or ``ssh://[user@]host[:port]``.

    Only the URI form carries a port, since ssh reads a colon in the plain
    form as part of an IPv6 address rather than as a port separator.
    """
    text = token
    is_uri = text.lower().startswith("ssh://")
    if is_uri:
        text = text[len("ssh://"):]
        # A URI may carry a path, which names no part of the connection.
        text = text.split("/", 1)[0]
    user, host, port = split_host_spec(text, allow_port=is_uri)
    result.host = host
    if user:
        _set_once(result, "username", user)
    if port:
        _set_once(result, "port", port)
