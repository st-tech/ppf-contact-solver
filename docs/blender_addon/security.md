# 🔒 Security

Four connection types reach the solver over a network:
[SSH](connections/ssh.md),
[SSH Command](connections/ssh.md#setup---command-mode),
[Docker over SSH](connections/docker_over_ssh.md#setup---custom-mode), and
[Docker over SSH Command](connections/docker_over_ssh.md#setup---command-mode).
For those, the connection is only as secure as the SSH endpoint and the
network path to it, because the add-on does not verify the host key (see
[Host-Key Verification](#host-key-verification)). Read this page before you
point the add-on at a host you do not fully control, or at a host reachable
from the internet.

The other connection types need less of this page:

- [Windows Native](connections/windows.md),
  {ref}`macOS Native <macos-native>` and
  [Linux Native](connections/linux.md): the solver is a child process on
  your own machine and no TCP socket leaves it, so the network sections do
  not apply. Read [Code Execution Risk: MCP](#code-execution-risk-mcp) and
  [Credential Files on Disk](#credential-files-on-disk).
- [Docker](connections/docker.md): there is no SSH hop, but the container
  must publish its port on your workstation, and the `docker run` command in
  the project README publishes port 9090 without an address, which is every
  interface. Read
  [Network Exposure of the Solver Port](#network-exposure-of-the-solver-port).

## Where the Trust Boundaries Are

The diagram on the [Connections](connections/index.md) page shows the
transport topology. The trust boundaries that matter for security are:

| Boundary | What crosses it | Protection |
| -------- | --------------- | ---------- |
| Blender workstation to solver host (the four SSH-based types) | SSH session | Public-key authentication and SSH encryption. The host key is not verified (see [Host-Key Verification](#host-key-verification)). |
| SSH session to `ppf-cts-server` on the solver host | A channel inside the SSH session, to the solver host's `localhost:<server_port>` | Stays inside the SSH session; not a separate network hop. |
| Blender workstation to local solver (native types and **Docker**) | Loopback TCP socket | No encryption, and none is needed: the traffic stays on the machine. |
| Blender workstation to local MCP server | Loopback TCP socket, default port `9633` | Bound to `localhost` only. A request with a non-loopback `Origin` header is refused with `403`. No authentication. |
| Blender workstation to local debug server | Loopback TCP socket, default port `8765` | Bound to `localhost` only. No authentication. Off until started. |

With **SSH** and **SSH Command**, the solver port is not reachable from the
network: `ppf-cts-server` binds the solver host's loopback and the add-on
reaches it through the SSH session. The Docker types differ, because the
container must publish its port. See
[Network Exposure of the Solver Port](#network-exposure-of-the-solver-port).

## SSH Authentication

The add-on supports **public-key** authentication only. There is no
password field in the UI, and paramiko is never asked to try password
authentication. Concretely:

- Provide a private key in **SSH Key** (**SSH** and **Docker over SSH**) or
  with `-i` in the command (**SSH Command** and **Docker over SSH Command**).
  Ed25519, ECDSA and RSA keys are supported; PuTTY `.ppk` keys are not.
- The matching public key must be in the remote account's
  `~/.ssh/authorized_keys`.
- Encrypted (passphrase-protected) keys are **not supported**: paramiko is
  invoked without a passphrase argument or prompt callback, so an encrypted
  key file cannot be loaded. Use `ssh-agent` (or `keychain`, `pageant`,
  `gnome-keyring`) to unlock the key once per login session and let
  paramiko pick it up through the agent, or decrypt the key file before
  pointing the add-on at it.
- Use a dedicated key for the solver host where practical, so revoking
  access to the solver does not require revoking access everywhere else
  that key is trusted.

On the server side, the usual SSH hardening rules apply and are not the
add-on's responsibility: disable password authentication in `sshd_config`
(`PasswordAuthentication no`), keep `PermitRootLogin` tight, and limit
`authorized_keys` with `from=`, `command=`, or `restrict` options if the
account is dedicated to the solver.

## Host-Key Verification

:::{warning}
The add-on does not store host keys. On every connection it accepts the key
the server presents, including a key that differs from the one it accepted
last time. A man-in-the-middle is therefore not detected on any connection,
the first or a later one. The add-on does not read `~/.ssh/known_hosts`,
and it ignores `StrictHostKeyChecking` and `UserKnownHostsFile` in
`~/.ssh/config`.
:::

The panel has no setting that makes the add-on reject unknown keys, and the
add-on does not show the key it accepted, so it offers nothing to compare
with the host's fingerprint. To verify a host, use your own `ssh` client:

- **Check the host from a terminal first.** `ssh` shows the host's
  fingerprint and records it in `known_hosts`. Compare the fingerprint with
  the one on the host (for example
  `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`, run on the host).
  This checks the host once, from your terminal. It does not change what
  the add-on accepts on later connections; the tunnel below does.
- **Tunnel through your own `ssh`.** Keep a forward to the solver host's
  SSH port open in a terminal:

  ```bash
  ssh -L 2222:localhost:22 gpu01.example.com
  ```

  Then set **Host** to `localhost` and **Port** to `2222` (or, in a Command
  type, use `ssh -p 2222 alice@localhost`). The terminal's `ssh` verifies
  the solver host. The add-on only connects to `localhost`, and its
  handshake reaches the same host through that tunnel.
- **Verify the bastion too, if one is the only way in.** Use
  `ssh -J bastion.example.com -L 2222:localhost:22 gpu01.internal`. A
  forward that ends at the bastion
  (`ssh -L 2222:gpu01.internal:22 bastion.example.com`) verifies only the
  bastion: the add-on's handshake with the solver host still accepts any
  key.
- **The panel's Proxy Jump is not a substitute.** The add-on dials the jump
  host itself, and it accepts any key from that host as well.
- **Where you cannot verify the host, avoid public networks.** A home or
  office LAN is safer than hotel or airport Wi-Fi.

## Network Exposure of the Solver Port

`ppf-cts-server` listens on TCP, and the Docker image also runs JupyterLab.
Who can reach those sockets depends on how they are deployed, not on the
add-on:

- **SSH and SSH Command.** `ppf-cts-server` runs in the remote user's
  shell, and the add-on launches it with no `--host` flag, so the binary
  uses its default bind address, `127.0.0.1` (`ps` shows no `--host` flag
  for this reason; check the bind with `ss`, as shown below). The socket is
  loopback-only on the solver host. The add-on reaches the server through
  the SSH session, and no other machine can connect, whatever the host
  firewall allows. To make the port reachable from the solver host's wider
  network for some other tool, relaunch `ppf-cts-server --host 0.0.0.0`
  yourself, outside the add-on.
- **Docker, Docker over SSH and Docker over SSH Command.** Inside the
  container the add-on launches `ppf-cts-server --host 0.0.0.0`, because
  `docker -p HOST:CONTAINER` forwards traffic to the container's external
  interface (`eth0`), not to loopback. The `-p` mapping you choose then
  decides who can reach the Docker host's port. A mapping with no address,
  such as `-p 9090:9090` (the form the project README's `docker run` command
  uses, with the port numbers held in variables), binds to **all interfaces**
  of the Docker host, which can expose the solver to any network that
  reaches that host. Use `-p 127.0.0.1:9090:9090` for a loopback-only
  publish. The add-on still reaches it: through the SSH session for the two
  **Docker over SSH** types, and through `localhost` for **Docker**. The
  add-on refuses to continue unless the container publishes the port; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`.

  One exception: the **Docker** type accepts a container on `--network host`,
  which publishes nothing. There the in-container `--host 0.0.0.0` binds
  every interface of the Docker host directly, and no `-p` mapping exists
  to restrict it. Use a bridge network if you need the loopback publish.
- **JupyterLab in the Docker image.** The image starts JupyterLab with no
  token and no password, and inside a container it listens on all
  interfaces. The `docker run` command in the project README publishes its
  port (8080 by default) without an address, which is every interface of the
  Docker host, so anyone who can reach that host can run code in the
  container. Publish it as `-p 127.0.0.1:8080:8080`, or do not publish it.
  Outside Docker it listens on `127.0.0.1` only.
- **Windows Native, macOS Native and Linux Native.** The server binds to
  the local machine only (loopback). No container and no host firewall
  change to worry about.

To check what the server listens on, run this on the solver host (**SSH**
and **SSH Command**) or inside the container (Docker types):

```bash
ss -tlnp | grep 9090
# LISTEN 0 ... 127.0.0.1:9090 ...  loopback only (SSH, SSH Command)
# LISTEN 0 ... 0.0.0.0:9090   ...  all interfaces inside the container; the host-side -p mapping decides who can reach it
```

## SSH Config Options That Affect Security

The bundled SSH parser is minimal, and several options that you might rely
on for security elsewhere are silently ignored. See
{ref}`Supported ssh_config options <supported-ssh-config-options>` for the
authoritative list. The ones worth flagging here:

| Option | Consequence |
| ------ | ----------- |
| `StrictHostKeyChecking` | Ignored. See [Host-Key Verification](#host-key-verification). |
| `UserKnownHostsFile` | Ignored. `known_hosts` is never read. |
| `ProxyJump` | Honored. Each hop is dialed by paramiko and accepts any host key, so a chain adds every jump host to the hosts that are not verified. |
| `ProxyCommand` | Ignored. Bring up the jump yourself with `ssh -L`. |
| `IdentitiesOnly` | Ignored. paramiko tries the configured key, then every key the agent offers, then any `id_rsa`, `id_ecdsa` or `id_ed25519` key it finds in `~/.ssh/`. |
| `PreferredAuthentications` | Ignored. paramiko picks the authentication method. |
| `CertificateFile` | Ignored as an `ssh_config` directive. However, if a matching `<key>-cert.pub` file exists next to a private key (the configured **SSH Key** or an `id_*` key in `~/.ssh/`), paramiko loads and uses it automatically. `CertificateFile` entries in `~/.ssh/config` are not read. |

If your environment depends on certificate-gated authentication or pinned
host keys, bring up `ssh -L` yourself from a terminal (see
[Host-Key Verification](#host-key-verification)). The system `ssh` binary
then verifies the host key, and the add-on only connects to `localhost`.

## Shared Solver Hosts

Two users on the same solver host share a GPU, a filesystem, and (unless
you configure otherwise) a user account.
[SSH](connections/ssh.md#multiple-users-on-one-solver-host) does not
recommend sharing a solver host in the first place. For a safer shared
setup:

- Give each user their own remote Linux account. Separate accounts separate
  their files and processes.
- Give each user a distinct **Project Name**. A solver built from source
  stores project data under the home directory of the account that runs it
  (`~/.local/share/ppf-cts/git-<branch>/<Project Name>`), so users who share
  one account and use the same **Project Name** share a directory, whatever
  their **Remote Path**. A self-contained distribution (and a solver on
  Windows) keeps that data inside its own folder, so a distinct **Remote
  Path** separates it there.
- Give each user a distinct **Remote Path**. Two servers started from one
  folder overwrite each other's `server.log` and `progress.log`.
- Give each user a distinct server port where you can. Distinct ports avoid
  clashes; they do not restrict access. One value, **Docker Port**, serves
  every connection type, so give each entry its own `docker_port` in a
  [connection profile](connections/profiles.md). See
  [Port usage at a glance](connections/index.md#port-usage-at-a-glance) for
  where the port is set.
- Remember that the solver server has no authentication of its own:
  reachability is its only protection. On a host with other accounts, any of
  them can connect to its loopback port.
- Do not share private keys. One key per user means revocation is one line
  in `authorized_keys`.
- Remember that the solver has file-system access to whatever the remote
  account can read. Do not run it under an account that also owns sensitive
  data.

## Code Execution Risk: MCP

The add-on can open two local ports that run code for whoever connects: the
MCP server and the debug server. Neither has a sandbox, an allowlist or
authentication.

### MCP Server

The [MCP server](integrations/mcp.md) exposes `run_python_script` and
`execute_shell_command` by design, and several other tools also act outside
the scene (see below). It does not run until you press **Start
MCP Server on Local** in the **MCP Server** box of the **Solver** panel
(see [Starting the Server](integrations/mcp.md#starting-the-server)). The
default port is `9633`, set in the **Port** field next to the button. A
busy port is replaced by the next free one, which is written back to the
field, and the **MCP Server (Running :<port>)** header shows the live port.
The protections are network scoping and a loopback `Origin` gate:

- The MCP server binds to `localhost` only (see
  [Port usage at a glance](connections/index.md#port-usage-at-a-glance)).
  Do not forward the MCP port over `ssh -R`, `ngrok`, `gh codespaces`, or any
  reverse proxy unless the machine is disposable.
- The HTTP transport refuses any request whose `Origin` header is not
  `localhost`, `127.0.0.1`, or `::1` with a `403 Origin not allowed`. That
  is DNS-rebinding protection: it stops a web page you happen to visit from
  driving `run_python_script` through your browser. A request with no
  `Origin` at all (curl, native MCP clients) is allowed, so this is not an
  authentication check.
- `run_python_script` runs inside Blender, as your account.
  `execute_shell_command` runs on the connected solver host, with that
  connection's account (the SSH user, the container, or your own account for
  the native types). Anything that can reach the MCP port can therefore act
  on the solver host as well while a connection is up.
- Other tools reach beyond the scene. `delete_log_file` deletes the file at
  any path you give it, on the machine that runs Blender, whether or not the
  file is a log. `git_pull_local` runs `git pull` in the add-on's own folder.
  `git_pull_remote` and `compile_project` run `git pull` and a release build
  in the solver folder of the connected host. `install_paramiko` and
  `install_docker` install the `paramiko` and `docker` packages with `pip`
  into Blender's Python. These tools work with **Debug Options** off, and the
  server asks for no confirmation and no authentication, so anything that can
  reach the port can call them as it can call `run_python_script`.
- LLM output wired into `run_python_script` is effectively a shell on your
  machine. Review the agent's tool calls before approving them; see
  [MCP Security](integrations/mcp.md#security).

### Debug Server

The add-on's debug server is a local port, `8765` by default, that runs any
Python it receives inside Blender. It has no `Origin` check and no
authentication.

- It is off in a plain Blender session. It starts when you tick **Debug
  Options** in the **Backend Communicator** panel and press **Start** under
  **Add-on Local Debug Server**. The developer launch script
  `blender_addon/launch.sh` starts it, together with the MCP server, when it
  launches Blender.
- **Reload Add-on Now** and **Full Reload** restart it only if it was
  already running.
- Stop it when you finish debugging.

### Shared Workstations

On a workstation shared with other accounts, every account can reach both
ports and can run Python as you. Start the MCP server and the debug server
only on a workstation you use alone.

## Credential Files on Disk

The add-on stores connection details in two places: plain-text TOML
[connection profiles](connections/profiles.md), and the `.blend` itself.

- A profile file holds host names, ports, usernames, container names, the
  `command` string, and **paths to private keys**. It does not hold the keys
  or any passphrases.
- A saved `.blend` also holds the connection fields currently in the panel
  (**Host**, **User**, the **SSH Key** path, **Proxy Jump**, **SSH
  Command**, **Container** and the paths), never the key itself. Clear those
  fields, or share a copy, before you pass the `.blend` on.
- Keep the profile file readable only by your user
  (`chmod 600 connection_profile.toml` on Unix; on Windows, restrict the
  file to your account in **Properties** > **Security**). The add-on does not
  enforce this.
- **Save Profile** writes every field for the entry, including `command`
  and `key_path` for types that do not use them. When you share a profile
  with teammates (check-in, chat, paste), strip `username`, `key_path`,
  `proxy_jump` and `command` first. `username` and `key_path` identify you.
  `proxy_jump` names your bastion host and can include a `user@`. `command`
  can contain `user@host`, a `-i` key path and a `-J` jump host. The host,
  container and port fields are usually safe to share.
- Keep the private key file at mode `600` and owned by your user
  (`chmod 600 ~/.ssh/id_ed25519`; on Windows, restrict it to your account).
  paramiko does not check file permissions, so it uses a world-readable key
  without any warning.

## Logs

The add-on starts the server with its console output written to `server.log`
in the solver directory, and the server writes `progress.log` beside it. The
solver directory is **Remote Path** (**SSH** and **SSH Command**),
**Container Path** (the Docker types) or **Solver Path** (the three native
types, on your own machine).

- `server.log` is the server's console output: a hardware summary, the
  project directories, the commands that start the build worker and the
  solver, request warnings, and the build worker's error output including
  tracebacks. It can contain absolute paths on the host, and the error
  messages quoted there can name objects in your scene and parameter
  values.
- `progress.log` holds only two markers, `SERVER_STARTING` and
  `SERVER_READY`.

They do not contain your SSH key or the scene geometry. If the solver host
or your workstation is shared, check the permissions of the solver
directory.

The local MCP and debug servers log to Blender's console (the terminal that
launched Blender, or Blender's Python console), not to disk.

## Quick Checklist Before Connecting to an Untrusted Network

1. Use an Ed25519 key dedicated to the solver host.
2. Make sure the key is either unencrypted but safeguarded, or loaded into
   `ssh-agent`.
3. Bring up a tunnel from a terminal
   (`ssh -L 2222:localhost:22 gpu01.example.com`; add
   `-J bastion.example.com` when a bastion is the only way in) rather than
   letting the add-on connect over the untrusted network itself, whether to
   the host or through the panel's **Proxy Jump**.
4. Set **Host** to `localhost` and **Port** to `2222`.
5. With **SSH** or **SSH Command**, confirm that `ppf-cts-server` binds
   `127.0.0.1`: `ss -tlnp | grep 9090` shows `127.0.0.1:9090`, not
   `0.0.0.0:9090`.
6. With a Docker type, publish the host-side port as `127.0.0.1:9090:9090`.
   The in-container bind is `0.0.0.0` by necessity, but the host mapping can
   still be loopback-only. Publish the JupyterLab port the same way
   (`127.0.0.1:8080:8080`) or not at all.
7. Keep the MCP port and the debug port on `localhost`. Do not `ssh -R` or
   proxy them, and do not let an MCP client run tool calls without your
   review.

:::{admonition} Under the hood
:class: toggle

**Host keys.** The add-on creates a new paramiko `SSHClient` for every
connection and for every jump host, and sets `AutoAddPolicy` on it. The
policy adds each key to that client's in-memory table, which is discarded
with the client, and the add-on never loads a known-hosts file.

**Server traffic.** With the SSH-based types, the add-on opens a
`direct-tcpip` channel inside the SSH session to `localhost:<server_port>`
on the solver host. With **Docker**, it connects a plain socket to
`localhost:<server_port>`.
:::
