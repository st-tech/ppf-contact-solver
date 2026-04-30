# 🌐 SSH (Direct)

The solver runs **directly** on a remote Linux host reached over SSH,
with no Docker layer in between. For a containerised solver on a remote
host, see [Docker over SSH](docker_over_ssh.md). Two UI modes are
available here: **Custom** (explicit fields) and **Command** (parsed
from a raw `ssh ...` string). Both produce the same connection; once
connected, the rest of the UI behaves identically.

```{figure} ../images/connections/ssh_topology.svg
:alt: Block diagram split into two boxes. The left box (Blender workstation) holds only the add-on. The right box (remote Linux host) holds server.py running directly as a python3 process bound to 127.0.0.1; there is no Docker daemon and no container. A blue solid arrow labelled ssh exec carries lifecycle commands; a purple dashed arrow labelled SSH tunnel (direct-tcpip) to remote localhost:9090 carries server traffic.
:width: 760px

Where each piece lives, and how the add-on reaches it. Blue solid
arrows carry lifecycle commands (start / stop). Purple dashed arrows
carry the TCP connection to `server.py`, which rides an SSH tunnel
into the remote's loopback port. The add-on launches `server.py` with
`--host 127.0.0.1`, so the SSH tunnel is the only path in -- nothing
else on the remote's network can reach the solver port. For the
containerised variant where `server.py` runs inside a Docker container
on the remote, see [Docker over SSH](docker_over_ssh.md).
```

:::{warning}
Not recommended. Setting up the remote Linux host with
[`warmup.py`](https://github.com/st-tech/ppf-contact-solver/blob/main/warmup.py)
is destructive (system package installs via `apt`, CUDA toolkit, shell rc
edits, user-level venv, nvm, rustup, NTP) and is impossible to revert or
uninstall cleanly. Even if you have a remote Linux machine with an NVIDIA
GPU, we strongly recommend the [Docker over SSH](docker_over_ssh.md) backend
instead. SSH (Direct) is intended for the special case where the remote
host is a disposable VM or short-lived environment you are willing to wipe.
:::

## When to Use It

- The GPU lives on a different machine from the user's workstation.
- Multiple users share a lab or cloud solver box.
- You want a persistent remote project that survives Blender restarts.

## Setup - Custom Mode

1. Set **Server Type** to `SSH`.
2. Fill in the fields below.
3. Click **Connect** -> **Start Server**.

```{figure} ../images/connections/ssh.png
:alt: Backend Communicator panel in SSH (Custom) mode
:width: 500px

Backend Communicator with **Server Type** set to `SSH`. **Host**,
**Port**, **User**, **SSH Key**, and **Remote Path** are exposed,
plus the shared **Project Name** field. **Connect** is highlighted.
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| Host | `""` | Hostname or SSH alias from `~/.ssh/config`. |
| Port | `22` | SSH port. |
| User | `""` | Remote user. Leave empty to use SSH config's `User`. |
| SSH Key | `~/.ssh/id_ed25519` or `~/.ssh/id_rsa` | Private key file. |
| Remote Path | `/root/ppf-contact-solver` | Remote solver directory (must contain `server.py`). |

The remote `server.py` port is fixed at `9090` in SSH modes; the panel
does not expose a Server Port field here (it is only editable in
Docker-family modes).

Aliases from your `~/.ssh/config` are resolved automatically, including
entries pulled in via `Include` directives. If the alias's config
supplies a hostname, port, user, or identity file, you can leave those
fields blank in the panel and they will be filled in at connect time.

(supported-ssh-config-options)=
### Supported `ssh_config` Options

The add-on ships its own minimal parser -- it does **not** shell out to
the system `ssh` binary. Only the following keywords are honored:

| Keyword | Supported | Notes |
| ------- | --------- | ----- |
| `Host` | yes | Wildcards `*` and `?` match via `fnmatch`. Multiple patterns per line are allowed. |
| `HostName` | yes | |
| `Port` | yes | |
| `User` | yes | |
| `IdentityFile` | yes | `~` is expanded. Only the first match per host wins; multiple `IdentityFile` lines are not tried in sequence. |
| `Include` | yes | Relative paths resolve against `~/.ssh/`. Globs (`*`, `?`) expand. |

Everything else -- including `ProxyJump`/`ProxyCommand`, `Match`,
`ForwardAgent`, `LocalForward`/`RemoteForward`,
`StrictHostKeyChecking`, `UserKnownHostsFile`,
`PreferredAuthentications`, `IdentitiesOnly`, `CertificateFile`,
`ControlMaster`/`ControlPath`/`ControlPersist`,
`ServerAliveInterval`/`ServerAliveCountMax`, `ConnectTimeout`,
`AddressFamily`, `BindAddress`, `LogLevel`, `PubkeyAcceptedAlgorithms`,
and `SetEnv` -- is silently ignored. If your workflow depends on any of
them (for example, reaching a host only through a bastion with
`ProxyJump`), the add-on cannot connect and there is no workaround
inside the panel. Host-key checking is always `AutoAddPolicy`
regardless of what your config says, and the keepalive interval is
hard-coded to 30 s.

## Setup - Command Mode

Paste a shell-style SSH command and the add-on extracts host, port,
username, and key path from it. This is convenient when you already copy
such a line from a cloud provider or a shared ops doc.

1. Set **Server Type** to `SSH Command`.
2. Paste into **SSH Command**, for example:

   ```text
   ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com
   ```

3. Set **Remote Path** and **Server Port** as above. Click **Connect**.

The parser understands the most common flags: `-p` for port, `-i` for
key path, and `user@host`. Advanced flags like `-o`, `-J`, or
`ProxyCommand` are **not** honored -- if you need those, use Custom
mode and let `~/.ssh/config` provide them.

If the command cannot be parsed (no host token found), the operator
reports an error and aborts.

## SSH Keys

- Ed25519 and RSA keys work out of the box.
- Encrypted (passphrase-protected) keys are **not supported** by the
  add-on's paramiko integration. paramiko raises an exception rather
  than prompting for the passphrase anywhere. Use `ssh-agent` or
  decrypt the key file before connecting.
- PuTTY `.ppk` keys are not supported; convert them to OpenSSH or PKCS#8
  format first.

## Multiple Users on One Solver Host

Sharing a single solver host between multiple users is possible but
not recommended.

## Installing paramiko

The SSH backend requires the `paramiko` Python package. If it is not
present, the main panel shows an **Install Paramiko** button that
installs it into the add-on's private library directory; click it and
wait for the background installer to finish.

:::{admonition} Under the hood
:class: toggle

**Command-mode parser**

The Command-mode parser splits the pasted string with `shlex` and picks
out only these tokens:

- `-p N` -> port
- `-i <path>` -> key path
- `user@host` -> username + host
- the first bare token after `ssh` that is not an option -> host

Every other flag is silently ignored. If no host can be extracted the
operator reports `Failed to parse command. Ensure it includes host.`
and aborts. The parser never invokes the system `ssh` binary; the
parsed fields go straight into paramiko, which is why `-o`, `-J`, and
`ProxyCommand` do not work in Command mode.

**`~/.ssh/config` resolution**

When **Host** looks like an alias instead of a DNS name, the add-on
parses `~/.ssh/config` (first-match, with `Include` directives
resolved) and fills in `HostName`, `Port`, `User`, and `IdentityFile`
for the alias. Later matching entries fill in fields earlier entries
left blank, so a trailing wildcard `Host *` block provides sensible
defaults without overriding explicit blocks. If the config file is
missing or the alias is not found, the alias text is used as the
hostname verbatim. Only the six keywords listed in
{ref}`Supported ssh_config options <supported-ssh-config-options>` are
read; the parser tokenises each non-comment line on whitespace or `=`,
matches the first word case-insensitively, and drops the line if the
keyword isn't one it recognises.

**Host-key policy**

Unknown host keys are accepted silently (paramiko `AutoAddPolicy`).
This is not hardened against MITM attacks and should not be relied on
for untrusted networks.

**Key-loading errors**

`SSHException: not a valid ... key` means the key file is in a format
paramiko cannot read. Typical causes: a PuTTY `.ppk` file (convert
first), or a modern OpenSSH key written with a cipher paramiko was
built without.

**paramiko install path**

The **Install Paramiko** button runs `pip install --target <lib/>`
into the add-on's bundled `lib/` directory on a background thread.
The add-on only imports paramiko from that directory; a system-wide
paramiko installation is not detected or used.
:::
