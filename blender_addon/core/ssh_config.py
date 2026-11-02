# License: GPL-3.0-or-later
"""SSH config file parsing utility for resolving host aliases.

Besides the connection fields for a single host, this resolves ``ProxyJump``
into the ordered list of hops a connection has to be tunneled through, which
is what lets the add-on reach a solver host that is only routable from a
bastion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import fnmatch


# Per-resolve trace logging is off by default because resolve_ssh_config runs
# on every SSH connect and would otherwise print the user's identity_file
# (private-key path) to the console each time. Set PPF_SSH_CONFIG_DEBUG=1 to
# enable the verbose trace.
_DEBUG = os.environ.get("PPF_SSH_CONFIG_DEBUG", "") not in ("", "0")


# ``ProxyJump none`` is how ssh_config spells "reach this host directly", so
# a Host block can opt out of a jump host a wildcard block set for everything
# else. It is the only value that is not a jump destination.
JUMP_NONE = "none"


@dataclass
class SSHConfigEntry:
    """Resolved SSH configuration for a host."""

    hostname: str
    port: int
    user: Optional[str]
    identity_file: Optional[str]
    proxy_jump: Optional[str] = None


@dataclass
class SSHHostConfig:
    """Configuration for a single Host block."""

    patterns: List[str] = field(default_factory=list)
    hostname: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    identity_file: Optional[str] = None
    proxy_jump: Optional[str] = None


def _parse_ssh_config_file(
    config_path: str, ssh_dir: str, visited: set
) -> List[SSHHostConfig]:
    """
    Parse a single SSH config file and return list of host configurations.

    Args:
        config_path: Path to the config file
        ssh_dir: The ~/.ssh directory for resolving relative Include paths
        visited: Set of already visited files to prevent infinite loops

    Returns:
        List of SSHHostConfig entries
    """
    # Normalize path and check for cycles
    config_path = os.path.normpath(os.path.expanduser(config_path))
    if config_path in visited:
        return []
    visited.add(config_path)

    if not os.path.exists(config_path):
        return []

    hosts: List[SSHHostConfig] = []
    current_host: Optional[SSHHostConfig] = None

    try:
        with open(config_path, "r") as f:
            for line in f:
                # Strip whitespace and skip empty lines/comments
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split into keyword and value
                # Handle both "Key Value" and "Key=Value" formats
                if "=" in line and " " not in line.split("=")[0]:
                    parts = line.split("=", 1)
                else:
                    parts = line.split(None, 1)

                if len(parts) < 2:
                    continue

                keyword = parts[0].lower()
                value = parts[1].strip()

                # Handle Include directive
                if keyword == "include":
                    include_path = value
                    # Resolve relative paths from ~/.ssh/
                    if not os.path.isabs(include_path):
                        include_path = os.path.join(ssh_dir, include_path)
                    # Expand ~ and globs
                    include_path = os.path.expanduser(include_path)

                    # Handle glob patterns in Include
                    if "*" in include_path or "?" in include_path:
                        import glob

                        for matched_path in sorted(glob.glob(include_path)):
                            hosts.extend(
                                _parse_ssh_config_file(matched_path, ssh_dir, visited)
                            )
                    else:
                        hosts.extend(
                            _parse_ssh_config_file(include_path, ssh_dir, visited)
                        )
                    continue

                # Handle Host directive - starts a new block
                if keyword == "host":
                    # Save previous host block
                    if current_host is not None:
                        hosts.append(current_host)
                    # Start new host block - can have multiple patterns
                    patterns = value.split()
                    current_host = SSHHostConfig(patterns=patterns)
                    continue

                # Handle directives within a Host block
                if current_host is not None:
                    if keyword == "hostname":
                        current_host.hostname = value
                    elif keyword == "port":
                        try:
                            current_host.port = int(value)
                        except ValueError:
                            pass
                    elif keyword == "user":
                        current_host.user = value
                    elif keyword == "identityfile":
                        current_host.identity_file = os.path.expanduser(value)
                    elif keyword == "proxyjump":
                        current_host.proxy_jump = value

        # Don't forget the last host block
        if current_host is not None:
            hosts.append(current_host)

    except Exception as e:
        _log(f"Error parsing {config_path}: {e}")

    return hosts


def _match_host(pattern: str, host: str) -> bool:
    """Check if a host matches a pattern (supports * and ? wildcards)."""
    return fnmatch.fnmatch(host, pattern)


def _log(message: str):
    """Log a message to the Blender console."""
    try:
        from ..models.console import console
        console.write(f"[SSH Config] {message}")
    except Exception:
        print(f"[SSH Config] {message}")


def resolve_ssh_config(
    host: str, default_port: int = 22, config_path: Optional[str] = None
) -> SSHConfigEntry:
    """
    Resolve SSH config for a host alias.

    Parses ~/.ssh/config and returns connection parameters for the given host.
    If no config file exists or the host isn't found, returns the host as-is
    with default values.

    Args:
        host: The host alias or hostname to look up
        default_port: Default port if not specified in config (default: 22)
        config_path: Path to SSH config file (default: ~/.ssh/config)

    Returns:
        SSHConfigEntry with resolved connection parameters
    """
    if _DEBUG:
        _log(f"Resolving SSH config for host: {host}")

    ssh_dir = os.path.expanduser("~/.ssh")
    if config_path is None:
        config_path = os.path.join(ssh_dir, "config")

    if _DEBUG:
        _log(f"Using config file: {config_path}")

    # If config file doesn't exist, return defaults
    if not os.path.exists(config_path):
        if _DEBUG:
            _log("Config file does not exist, returning defaults")
        return SSHConfigEntry(
            hostname=host, port=default_port, user=None, identity_file=None
        )

    # Parse all config files (including Include'd files)
    visited: set = set()
    all_hosts = _parse_ssh_config_file(config_path, ssh_dir, visited)

    if _DEBUG:
        _log(f"Parsed {len(all_hosts)} host entries from config")

    # SSH config uses first-match semantics, but later entries can fill in
    # values not set by earlier matches. Wildcard (*) entries apply to all.
    resolved_hostname: Optional[str] = None
    resolved_port: Optional[int] = None
    resolved_user: Optional[str] = None
    resolved_identity_file: Optional[str] = None
    resolved_proxy_jump: Optional[str] = None

    for host_config in all_hosts:
        # Check if any pattern matches the host
        matches = any(_match_host(p, host) for p in host_config.patterns)
        if not matches:
            continue

        if _DEBUG:
            _log(
                f"Matched pattern {host_config.patterns} -> "
                f"hostname={host_config.hostname}, user={host_config.user}"
            )

        # Fill in values that haven't been set yet (first match wins)
        if resolved_hostname is None and host_config.hostname is not None:
            resolved_hostname = host_config.hostname
        if resolved_port is None and host_config.port is not None:
            resolved_port = host_config.port
        if resolved_user is None and host_config.user is not None:
            resolved_user = host_config.user
        if resolved_identity_file is None and host_config.identity_file is not None:
            resolved_identity_file = host_config.identity_file
        if resolved_proxy_jump is None and host_config.proxy_jump is not None:
            resolved_proxy_jump = host_config.proxy_jump

    result = SSHConfigEntry(
        hostname=resolved_hostname if resolved_hostname else host,
        port=resolved_port if resolved_port else default_port,
        user=resolved_user,
        identity_file=resolved_identity_file,
        proxy_jump=resolved_proxy_jump,
    )

    if _DEBUG:
        _log(
            f"Resolved: hostname={result.hostname}, port={result.port}, "
            f"user={result.user}, identity_file={result.identity_file}, "
            f"proxy_jump={result.proxy_jump}"
        )

    return result


def split_host_spec(
    spec: str, allow_port: bool = True
) -> Tuple[Optional[str], str, Optional[int]]:
    """Split ``[user@]host[:port]`` into ``(user, host, port)``.

    An IPv6 literal carries colons of its own, so a port may only be attached
    to the bracketed form (``[2001:db8::1]:2222``); a bare literal is read as
    an address with no port. ``allow_port`` is False for an ssh destination,
    where ``host:port`` is not the syntax ssh accepts and a colon therefore
    belongs to the address.

    Raises ValueError when the spec names no host or carries an unusable port.
    """
    text = spec.strip()
    if not text:
        raise ValueError("empty host specification")

    user: Optional[str] = None
    if "@" in text:
        user, _, text = text.rpartition("@")
        if not user:
            raise ValueError(f"'{spec}' has an empty user name")
        if not text:
            raise ValueError(f"'{spec}' names no host")

    port: Optional[int] = None
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            raise ValueError(f"'{spec}' has an unterminated '[' around the address")
        host = text[1:end]
        rest = text[end + 1:]
        if rest.startswith(":"):
            port = _parse_port(rest[1:], spec)
        elif rest:
            raise ValueError(f"'{spec}' has trailing text after the address")
    elif allow_port and text.count(":") == 1:
        host, _, port_text = text.partition(":")
        port = _parse_port(port_text, spec)
    else:
        host = text

    if not host:
        raise ValueError(f"'{spec}' names no host")
    return user, host, port


def _parse_port(text: str, spec: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise ValueError(f"'{spec}' has a non-numeric port '{text}'") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"'{spec}' has a port outside 1-65535")
    return port


def resolve_jump_chain(
    spec: str, default_port: int = 22, config_path: Optional[str] = None
) -> List[SSHConfigEntry]:
    """Resolve a ``ProxyJump`` spec into the hops to tunnel through, in order.

    The list is ordered outward from the client: hop 0 is reached directly,
    hop 1 through hop 0, and the final target through the last entry. Each hop
    is resolved through the ssh config exactly as a target host is, so an alias
    brings its own ``HostName`` / ``Port`` / ``User`` / ``IdentityFile``, and a
    hop that carries a ``ProxyJump`` of its own contributes its hops ahead of
    itself. A user name or port written into the spec overrides what the config
    says for that alias, which is the precedence ssh gives the command line.

    ``none`` resolves to an empty chain. Raises ValueError on a malformed spec
    or on a jump loop, since either one would otherwise surface as a connection
    attempt against the wrong host.
    """
    return _resolve_jump_chain(spec, default_port, config_path, ())


def _resolve_jump_chain(
    spec: str,
    default_port: int,
    config_path: Optional[str],
    pending: Tuple[str, ...],
) -> List[SSHConfigEntry]:
    if spec.strip().lower() == JUMP_NONE:
        return []

    hops: List[SSHConfigEntry] = []
    for token in spec.split(","):
        if not token.strip():
            raise ValueError(f"'{spec}' has an empty jump host")
        if token.strip().lower() == JUMP_NONE:
            raise ValueError(f"'{spec}' combines 'none' with a jump host")

        user, host, port = split_host_spec(token)
        if host in pending:
            trail = " -> ".join(pending + (host,))
            raise ValueError(f"jump hosts loop back on themselves: {trail}")

        config = resolve_ssh_config(host, default_port, config_path)
        if config.proxy_jump:
            hops.extend(
                _resolve_jump_chain(
                    config.proxy_jump, default_port, config_path, pending + (host,)
                )
            )
        hops.append(
            SSHConfigEntry(
                hostname=config.hostname,
                port=port if port else config.port,
                user=user if user else config.user,
                identity_file=config.identity_file,
            )
        )

    return hops
