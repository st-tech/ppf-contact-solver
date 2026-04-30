"""SSH config file parsing utility for resolving host aliases."""

from dataclasses import dataclass, field
from typing import Optional, List
import os
import fnmatch


@dataclass
class SSHConfigEntry:
    """Resolved SSH configuration for a host."""

    hostname: str
    port: int
    user: Optional[str]
    identity_file: Optional[str]


@dataclass
class SSHHostConfig:
    """Configuration for a single Host block."""

    patterns: List[str] = field(default_factory=list)
    hostname: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    identity_file: Optional[str] = None


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

        # Don't forget the last host block
        if current_host is not None:
            hosts.append(current_host)

    except Exception as e:
        print(f"[SSH Config] Error parsing {config_path}: {e}")

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
    _log(f"Resolving SSH config for host: {host}")

    ssh_dir = os.path.expanduser("~/.ssh")
    if config_path is None:
        config_path = os.path.join(ssh_dir, "config")

    _log(f"Using config file: {config_path}")

    # If config file doesn't exist, return defaults
    if not os.path.exists(config_path):
        _log(f"Config file does not exist, returning defaults")
        return SSHConfigEntry(
            hostname=host, port=default_port, user=None, identity_file=None
        )

    # Parse all config files (including Include'd files)
    visited: set = set()
    all_hosts = _parse_ssh_config_file(config_path, ssh_dir, visited)

    _log(f"Parsed {len(all_hosts)} host entries from config")

    # SSH config uses first-match semantics, but later entries can fill in
    # values not set by earlier matches. Wildcard (*) entries apply to all.
    resolved_hostname: Optional[str] = None
    resolved_port: Optional[int] = None
    resolved_user: Optional[str] = None
    resolved_identity_file: Optional[str] = None

    for host_config in all_hosts:
        # Check if any pattern matches the host
        matches = any(_match_host(p, host) for p in host_config.patterns)
        if not matches:
            continue

        _log(f"Matched pattern {host_config.patterns} -> hostname={host_config.hostname}, user={host_config.user}")

        # Fill in values that haven't been set yet (first match wins)
        if resolved_hostname is None and host_config.hostname is not None:
            resolved_hostname = host_config.hostname
        if resolved_port is None and host_config.port is not None:
            resolved_port = host_config.port
        if resolved_user is None and host_config.user is not None:
            resolved_user = host_config.user
        if resolved_identity_file is None and host_config.identity_file is not None:
            resolved_identity_file = host_config.identity_file

    result = SSHConfigEntry(
        hostname=resolved_hostname if resolved_hostname else host,
        port=resolved_port if resolved_port else default_port,
        user=resolved_user,
        identity_file=resolved_identity_file,
    )

    _log(f"Resolved: hostname={result.hostname}, port={result.port}, user={result.user}, identity_file={result.identity_file}")

    return result
