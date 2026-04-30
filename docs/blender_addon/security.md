# 🔒 Security

Most connection types ([SSH](connections/ssh.md),
[SSH Command](connections/ssh.md#setup---command-mode),
[Docker over SSH](connections/docker_over_ssh.md#setup---custom-mode), and
[Docker over SSH Command](connections/docker_over_ssh.md#setup---command-mode))
reach the solver over a network, so the add-on's security story is
mostly the SSH endpoint's security story. This page collects the things
you need to know before pointing it at a host you do not fully control,
or a host that is reachable from the wider internet.

If you use [Local](connections/local.md) or
[Windows Native](connections/windows.md) the network-attacker sections
below do not apply: the solver is a child process on your own machine
and no TCP socket leaves it. Skip to
[Code execution risk: MCP](#code-execution-risk-mcp) and
[Credential files on disk](#credential-files-on-disk).

## Where the Trust Boundaries Are

The diagram on the [Connections](connections/index.md) page shows the
transport topology. The trust boundaries that matter for security are:

| Boundary | What crosses it | Protection |
| -------- | --------------- | ---------- |
| Blender workstation -> remote solver host | SSH session (paramiko) | SSH key auth + SSH encryption |
| SSH session -> `server.py` on remote | `direct-tcpip` channel to the remote's `localhost:<server_port>` | Stays inside the SSH tunnel; not a separate network hop |
| Blender workstation <-> local solver | UNIX loopback TCP socket | No encryption; no encryption needed (never leaves the host) |
| Blender workstation <-> local MCP server | UNIX loopback TCP socket on port `9633` | Bound to `localhost` only |

The important consequence: when you run Docker-over-SSH or plain SSH,
the solver's TCP socket never crosses an untrusted network directly.
paramiko opens a `direct-tcpip` channel to `localhost:<server_port>`
*on the remote*. By default `server.py` binds to `127.0.0.1` (loopback
only) on the remote in plain SSH, Local, and Windows Native modes, so
the SSH tunnel is the only way in. In Docker-family modes the add-on
launches `server.py` with `--host 0.0.0.0` *inside the container*
because docker `-p HOST:CONTAINER` forwards traffic to the container's
external interface (eth0), not loopback; external reachability of the
container is then controlled by the host-side `-p` mapping (see below).

## SSH Authentication

The add-on only supports **public-key** authentication. There is no
password field in the UI, and paramiko is never asked to try password
auth. Concretely:

- Provide a private key in **SSH Key** (Custom mode) or via `-i`
  (Command mode). Ed25519 and RSA keys are supported; PuTTY `.ppk`
  keys are not.
- The matching public key must be in the remote account's
  `~/.ssh/authorized_keys`.
- Encrypted (passphrase-protected) keys are **not supported** by the
  add-on's paramiko integration: paramiko is invoked without a
  passphrase argument or prompt callback, so an encrypted key file
  raises an exception at connect time. Use `ssh-agent` (or `keychain`,
  `pageant`, `gnome-keyring`) to unlock the key once per login session
  and let paramiko pick it up through the agent, or decrypt the key
  file before pointing the add-on at it.
- Use a dedicated key for the solver host where practical, so revoking
  access to the solver does not require revoking access everywhere
  else that key is trusted.

On the server side, the usual SSH hardening rules apply and are not the
add-on's responsibility: disable password auth in `sshd_config`
(`PasswordAuthentication no`), keep `PermitRootLogin` tight, and limit
`authorized_keys` with `from=`, `command=`, or `restrict` options if
the account is dedicated to the solver.

## Host-Key Verification (Trust-on-First-Use)

:::{warning}
The add-on accepts unknown host keys silently. It does **not**
consult `~/.ssh/known_hosts`, and it does **not** honor
`StrictHostKeyChecking` from `~/.ssh/config`. Internally, paramiko is
configured with `AutoAddPolicy`.
:::

This means the very first connection to a host has no out-of-band
verification: a man-in-the-middle on that first handshake can present
their own key and the add-on will trust it. Mitigations:

- **Do the first connect from a trusted network.** A home or office
  LAN with known routing is safer than a hotel or airport Wi-Fi for
  the initial handshake.
- **Prefer a jump host you already trust.** Bring up `ssh -L` from a
  terminal (which *does* honor `known_hosts`), then point the add-on
  at `localhost:<forwarded-port>`. The add-on's weak policy now only
  applies to a loopback handshake, which a network attacker cannot
  influence.
- **Avoid public networks for first-run on production hosts.** If you
  have to, verify the SSH fingerprint out of band after connecting
  (for example, by logging in from a known-good machine and checking
  the remote's `/etc/ssh/ssh_host_*_key.pub`).

The add-on cannot be configured to reject unknown keys from the panel;
the tunnel-through-your-own-`ssh` pattern is the supported escape hatch.

## Network Exposure of the Solver Port

`server.py` listens on TCP. Where that TCP socket is reachable from
depends on how the backend is deployed, not on the add-on:

- **Plain SSH / SSH Command.** `server.py` runs in the remote user's
  shell and the add-on launches it with `--host 127.0.0.1`, so the
  socket is loopback-only on the remote. The add-on reaches it through
  the SSH session's `direct-tcpip` channel; nothing else on the network
  can connect, regardless of host firewall. If you need the port
  reachable from the remote's wider network for some other tool, you
  must rebind it yourself (e.g. by relaunching `server.py --host 0.0.0.0`
  outside the add-on).
- **Docker / Docker over SSH.** Inside the container the add-on
  launches `server.py --host 0.0.0.0` because docker `-p HOST:CONTAINER`
  forwards traffic to the container's external interface (eth0), not
  loopback. External reachability of the *host* is then controlled by
  the `-p` mapping you choose: the container must publish the server
  port (`-p 9090:9090` or the equivalent in your compose file),
  otherwise the add-on refuses to start. The default `-p 9090:9090`
  binds to **all interfaces** on the Docker host, which can expose the
  solver to any network that reaches that host. Prefer
  `-p 127.0.0.1:9090:9090` so the publish is loopback-only; the add-on
  will still reach it over the SSH tunnel.
- **Windows Native / Local.** The server binds to the local machine
  only (loopback). No container and no host firewall change to worry about.

A quick check on the remote (or inside the container):

```bash
# On the solver host (plain SSH) or inside the container (Docker)
ss -tlnp | grep 9090
# LISTEN 0 ... 127.0.0.1:9090 ...  → loopback-only (default for plain SSH / Local / Windows)
# LISTEN 0 ... 0.0.0.0:9090   ...  → all interfaces (default inside Docker; host -p mapping controls external exposure)
```

Plain SSH now binds loopback-only by default. For Docker-over-SSH the
in-container bind is `0.0.0.0` (required for port publishing); restrict
the host-side mapping with `-p 127.0.0.1:9090:9090` if you do not want
the port exposed beyond the Docker host's loopback.

## What the Add-on Does *Not* Support from `~/.ssh/config`

The bundled SSH parser is intentionally minimal; several options that
you might rely on for security elsewhere are silently ignored. See
{ref}`Supported ssh_config options <supported-ssh-config-options>` for
the authoritative list. The ones worth flagging here:

| Option | Consequence |
| ------ | ----------- |
| `StrictHostKeyChecking` | Ignored. See [Host-key verification](#host-key-verification-trust-on-first-use). |
| `UserKnownHostsFile` | Ignored. `known_hosts` is never consulted. |
| `ProxyJump` / `ProxyCommand` | Ignored. Bring up the jump yourself with `ssh -L`. |
| `IdentitiesOnly` | Ignored. paramiko will try the configured key, then every key the agent offers, then any `id_rsa` / `id_dsa` / `id_ecdsa` / `id_ed25519` key it finds in `~/.ssh/`. |
| `PreferredAuthentications` | Ignored. paramiko picks the auth method. |
| `CertificateFile` | Ignored as an `ssh_config` directive. However, if a matching `<key>-cert.pub` file exists alongside a private key (either the configured `key_path` or a discoverable `~/.ssh/id_*` key), paramiko loads and uses it automatically. Explicit `CertificateFile` entries in `~/.ssh/config` are not read. |

If your environment depends on any of these for its security posture
(bastion-only access, certificate-gated auth, pinned host keys), bring
up `ssh -L` yourself from a terminal so that the system `ssh` binary
handles the sensitive part of the connection and the add-on only talks
to `localhost`.

## Shared Solver Hosts

Two users on the same box share more than a port; they share a GPU,
a filesystem, and (unless you configure otherwise) a user account.
For a safer shared setup:

- Give each user their own remote Linux account. Key auth keeps them
  apart at the OS level.
- Give each user (or project) a distinct **Remote Path** and a distinct
  **Server Port**. See
  [Multiple users on one solver host](connections/ssh.md#multiple-users-on-one-solver-host).
  This prevents one client from overwriting another's checkpoints or
  accidentally fetching another run's frames.
- Do not share private keys. One key per user means revocation is one
  line in `authorized_keys`.
- Remember the solver has file-system access to whatever the remote
  account can read. Do not run it under an account that also owns
  sensitive data.

## Code Execution Risk: MCP

The bundled [MCP server](integrations/mcp.md) exposes
`run_python_script` and `execute_shell_command` by design. There is
no sandbox, no allowlist, and no authentication. The protections are
purely network-scoping:

- The MCP server binds to `localhost` only (see
  {ref}`Connections ports <connections-under-the-hood>`). Do not
  forward the MCP port over `ssh -R`, `ngrok`, `gh codespaces`, or any
  reverse proxy unless the machine is disposable.
- Any process on the Blender machine can reach the MCP port. Treat
  local-user isolation as your security boundary: a malicious local
  user already owns the Blender session.
- LLM outputs wired into `run_python_script` are effectively shell on
  your machine. Always review the agent's tool calls before approving
  them; see [MCP Security](integrations/mcp.md#security).

## Credential Files on Disk

The add-on stores connection details in plain-text TOML
[connection profiles](connections/profiles.md). Profile files contain
hostnames, ports, usernames, container names, and **paths to private
keys**; they do not contain the keys themselves or any passphrases.

- Keep the profile file readable only by your user
  (`chmod 600 connections.toml` on Unix). The add-on does not enforce
  this.
- When sharing a profile with teammates (check-in, chat, paste), strip
  the `username` and `key_path` fields first. The host + container +
  port fields are usually fine; the identity fields are not.
- The private key file itself should also be mode `600` and owned by
  your user (`chmod 600 ~/.ssh/id_ed25519`). paramiko does not load
  world-readable keys silently, but it does not refuse them either.

## Logs and Artifacts

`server.log` and `progress.log` on the remote capture the solver's
stdout and a structured run log. They can contain:

- Remote file paths under the project directory.
- Scene metadata (object names, frame counts, parameter values).
- Tracebacks from the solver, including absolute paths on the host.

They do not contain your SSH key, Blender credentials, or the raw
geometry. If the remote is shared, review the project directory's
permissions if these logs are sensitive.

The local reload and MCP servers log to Blender's console (the
terminal that launched Blender, or Blender's Python console), not to
disk.

## Quick Checklist Before Connecting to an Untrusted Network

1. Use an Ed25519 key dedicated to the solver host.
2. Make sure the key is either unencrypted-but-safeguarded or loaded
   into `ssh-agent`.
3. Bring up a trusted tunnel from a terminal (`ssh -L
   2222:gpu01.example.com:22 bastion.example.com`) rather than letting
   the add-on dial the hostile network directly.
4. Point the add-on at `localhost:2222`, leaving the weak host-key
   policy applied only to loopback.
5. On the remote, verify the server port is not reachable from the
   outside. With plain SSH the add-on now binds `server.py` to
   `127.0.0.1` so loopback-only is the default; confirm with
   `ss -tlnp | grep 9090` (you should see `127.0.0.1:9090`, not
   `0.0.0.0:9090`).
6. If you use Docker-over-SSH, publish the host-side port as
   `127.0.0.1:9090:9090` (the in-container bind is `0.0.0.0` by
   necessity, but the host mapping can still be loopback-only).
7. Keep the MCP port on `localhost`. Do not `ssh -R` it, do not proxy
   it, do not expose it to an LLM that executes arbitrary network
   tools without review.
