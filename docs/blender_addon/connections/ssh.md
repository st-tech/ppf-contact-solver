# 🌐 SSH (Direct)

Use **SSH** or **SSH Command** when the solver runs **directly** on a
remote Linux host that you reach over SSH, with no Docker layer in
between. For a solver inside a container on a remote host, see
[Docker over SSH](docker_over_ssh.md). The two types differ only in how
you describe the connection: **SSH** uses separate fields, and **SSH
Command** reads them from a pasted `ssh ...` command. Once connected, the
panel behaves identically.

```{figure} ../images/connections/ssh_topology.svg
:alt: Block diagram split into two boxes. The left box (Blender workstation) holds only the add-on. The right box (remote Linux host) holds ppf-cts-server running directly as a native binary bound to 127.0.0.1; there is no Docker daemon and no container. A blue solid arrow carries start and stop commands over SSH; a purple dashed arrow, an SSH tunnel to the remote host's localhost port 9090, carries server traffic.
:width: 760px

Where each piece lives, and how the add-on reaches it. The blue solid
arrow carries start and stop commands. The purple dashed arrow carries
the connection to `ppf-cts-server` through an SSH tunnel into the remote
host's loopback port (the figure shows the default port, 9090).
The add-on starts `ppf-cts-server` without a `--host` flag, so it listens
on its default address, `127.0.0.1`, and the SSH tunnel is the only path
in: nothing else on the remote host's network can reach the solver port.
For the containerized variant, where `ppf-cts-server` runs inside a Docker
container on the remote host, see [Docker over SSH](docker_over_ssh.md).
```

:::{warning}
Not recommended. Setting up the remote Linux host with
[`warmup.py`](https://github.com/st-tech/ppf-contact-solver/blob/main/warmup.py)
changes the whole machine (system packages through `apt`, the CUDA
toolkit, shell startup files, a user-level Python environment, Node.js
through nvm, Rust through rustup, NTP) and does not undo any of it. Even
if you have a remote Linux machine with an NVIDIA GPU, we recommend the
[Docker over SSH](docker_over_ssh.md) backend instead. Use SSH (Direct)
only when the remote host is a disposable VM or a short-lived environment
you are willing to wipe.
:::

## When to Use It

- The GPU lives on a different machine from your workstation, and that
  machine is a disposable VM or short-lived environment.
- You want a remote project that survives Blender restarts: **Disconnect**
  leaves the server running on the host.

## Before You Start

Every SSH-backed type (**SSH**, **SSH Command**, **Docker over SSH**, and
**Docker over SSH Command**) needs the `paramiko` Python package, and
**Connect** stays unavailable until it is installed. If it is missing, the
main panel shows an **Install Paramiko to Add-on Directory** button.
Click it and wait for the background installer to finish. The installer
downloads paramiko with pip, so the machine running Blender needs network
access.

The solver must already be on the host. Connecting installs nothing there:
**Remote Path** has to name a folder on the host that holds a built solver,
and the panel reports that no server was found otherwise (see
[Troubleshooting](#troubleshooting)). The folder layout is the one
described on [Linux Native](linux.md), and the **Remote Path** row below
lists the accepted locations.

## Setup - Custom Mode

Use this for the **SSH** type.

1. Set **Type** to **SSH**.
2. Fill in the fields below.
3. Click **Connect**, then **Start Server on Remote**.

```{figure} ../images/connections/ssh.png
:alt: Backend Communicator panel with Type set to SSH, showing the Host, Port, User, SSH Key, Proxy Jump, Remote Path, and Project Name fields with the Connect button highlighted
:width: 500px

Backend Communicator with **Type** set to **SSH**. **Host**, **Port**,
**User**, **SSH Key**, **Proxy Jump**, and **Remote Path** are shown,
plus the shared **Project Name** field. **Connect** is highlighted.
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| **Host** | empty | Hostname or an alias from `~/.ssh/config`. Required. |
| **Port** | `22` | SSH port. While it reads `22`, a `Port` set for the alias in `~/.ssh/config` is used instead. |
| **User** | empty | Remote user. Empty uses the alias's `User`. |
| **SSH Key** | `~/.ssh/id_ed25519` if that file exists, otherwise `~/.ssh/id_rsa` | Private key file. See [SSH Keys](#ssh-keys). Clear the field to let the alias's `IdentityFile` apply. |
| **Proxy Jump** | empty | Jump host to tunnel through, written the way `ssh -J` takes it: `[user@]host[:port]`, comma separated for a chain. Empty uses the alias's `ProxyJump` from `~/.ssh/config`. See [Jump Hosts](#jump-hosts). |
| **Remote Path** | empty (for example `/root/ppf-contact-solver`) | The solver root on the host: the directory that holds `ppf-cts-server` under `target/release/`, `target/cuda/release/`, `target/rocm/release/`, or `target/cpu/release/`. Not the `target/release` directory itself. Left empty, the connection opens but the panel reports *The connection names no directory on the solver host*. |

An alias from your `~/.ssh/config`, including entries pulled in with
`Include`, is resolved when you connect, and its `HostName` replaces the
alias. See {ref}`Supported ssh_config options <supported-ssh-config-options>`
for the keywords the add-on reads.

## Setup - Command Mode

Use this for the **SSH Command** type. Paste a shell-style SSH command and
the add-on extracts the host, port, user name, key path, and jump hosts
from it. This is convenient when you already copy such a line from a
cloud provider or a shared ops doc.

1. Set **Type** to **SSH Command**.
2. Paste into **SSH Command**, for example:

   ```text
   ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com
   ```

3. Set **Remote Path**, which the table under
   [Setup - Custom Mode](#setup---custom-mode) describes, and **Project
   Name**. Click **Connect**, then **Start Server on Remote**. The server
   port is described under [The Server Port](#the-server-port), and
   **Connect** is unavailable under the conditions listed in
   [When Connect Is Unavailable](#when-connect-is-unavailable).

The parser reads the destination (`[user@]host`, or an
`ssh://user@host:port` URI) plus `-p` for port, `-i` for key path, `-l`
for login name, `-J` for jump hosts, and those same four settings
written as `-o Port=`, `-o IdentityFile=`, `-o User=`, and
`-o ProxyJump=`. Every other ssh option is accepted and ignored,
`-F` included: the add-on always reads `~/.ssh/config`. A setting given
twice keeps the first value, so `-p` outranks a later `-o Port=`. A
setting the command leaves out falls back to the alias's entry in
`~/.ssh/config`.

Options work as they do in `ssh`: `ssh -p 2222 gpu-alias` connects to
`gpu-alias` on port 2222, an option may follow the destination
(`ssh gpu01.example.com -p 2222`), and anything past a second bare word is
the remote command and is ignored.

If the command cannot be parsed, **Connect** reports why and does not
connect: an unknown option, an option missing its argument, an
unbalanced quote, a port that is not a number from 1 to 65535, a
malformed destination (an empty user name, or a `[` with no closing `]`),
or no host.

## When Connect Is Unavailable

**Connect** stays unavailable while any of these holds:

- paramiko is not installed (see [Before You Start](#before-you-start)).
- **Host** is empty (**SSH**), or **SSH Command** is empty (**SSH
  Command**).
- **Project Name** is empty, or contains anything other than letters,
  digits, `.`, `-`, and `_`.
- **SSH Key** (**SSH** only) or **Remote Path** contains a space or a
  shell special character. The default key path comes from your home
  directory, so an account name with a space makes it unusable: point
  **SSH Key** at a copy of the key in a path without spaces.

{ref}`When Connect is unavailable <connect-unavailable>` lists the message
the panel shows for each case and the characters that are refused.

## After You Connect

This section applies to both **SSH** and **SSH Command**. After
**Connect**, the Connection box shows the **GPU** / **CPU** toggle (the
**Compute Device** setting), **GPU Backend** where the host holds more than
one GPU build, and the **GPU** dropdown with its refresh button. The
builds and GPUs are on the solver host, so the add-on reads them at
**Connect** and applies your choices at **Start Server on Remote**. To move
a running solver onto another build or GPU, click **Stop Server on
Remote**, make the pick, then click **Start Server on Remote**. See
{ref}`Choosing the build <choosing-the-build>` for when each row is drawn
and {ref}`Picking a GPU <gpu-picker>`.

**Disconnect** leaves the server running on the host. **Stop Server on
Remote** ends the server on this connection's port and nothing else.

## The Server Port

The server port is the **Docker Port** setting (default `9090`, range
1024–65535). The panel draws it only for the Docker types, so **SSH** and
**SSH Command** use the value it holds, including one set while a Docker
type was selected. To change it, set **Type** to one of the three Docker
types, edit **Docker Port**, then select **SSH** or **SSH Command** again,
or set `docker_port` in a [connection profile](profiles.md). See
[Port Usage at a Glance](index.md#port-usage-at-a-glance).

(supported-ssh-config-options)=
## Supported `ssh_config` Options

The add-on has its own minimal parser and does **not** run the system
`ssh` binary. It reads only these keywords:

| Keyword | Notes |
| ------- | ----- |
| `Host` | Wildcards `*` and `?` match. Several patterns on one line are allowed. A negated pattern (starting with `!`) is not recognized. |
| `HostName` | Used as written. |
| `Port` | Used as written. |
| `User` | Used as written. |
| `IdentityFile` | `~` is expanded. The first matching `Host` block that sets it wins. With several `IdentityFile` lines in one block, the last one is used; none are tried in sequence. |
| `ProxyJump` | Used; `none` connects directly. See [Jump Hosts](#jump-hosts). |
| `Include` | Relative paths resolve against `~/.ssh/`. Globs (`*`, `?`) expand. |

Everything else, including `ProxyCommand`, `Match`, `ForwardAgent`,
`LocalForward`/`RemoteForward`, `StrictHostKeyChecking`,
`UserKnownHostsFile`, `PreferredAuthentications`, `IdentitiesOnly`,
`CertificateFile`, `ControlMaster`/`ControlPath`/`ControlPersist`,
`ServerAliveInterval`/`ServerAliveCountMax`, `ConnectTimeout`,
`AddressFamily`, `BindAddress`, `LogLevel`, `PubkeyAcceptedAlgorithms`,
and `SetEnv`, is ignored without a message. Whatever your config says,
unknown host keys are always accepted, the keepalive interval is 30
seconds, and each phase of the SSH handshake (TCP connect, protocol
banner, authentication) gives up after 20 seconds.

(jump-hosts)=
## Jump Hosts

A solver host that is only routable from a bastion is reached by naming
the bastion, and the add-on opens the hops itself. Every SSH-backed
server type supports it:

- **SSH** and **Docker over SSH**: type the jump host into **Proxy
  Jump**, in the form `ssh -J` takes it (`[user@]host[:port]`, comma
  separated for a chain, ordered outward from your workstation).
- **SSH Command** and **Docker over SSH Command**: put `-J` (or
  `-o ProxyJump=`) in the command, exactly as you would run it in a
  shell.
- Either way, leaving it empty falls back to the `ProxyJump` entry
  `~/.ssh/config` gives for the host, so a host already configured for
  your terminal needs nothing typed into the panel.

For a bastion in front of `gpu01.internal`, the command is:

```text
ssh -J bastion.example.com gpu01.internal
```

and the **Proxy Jump** field takes:

| Value | Meaning |
| ----- | ------- |
| `bastion.example.com` | One hop. |
| `alice@bastion.example.com:2222,inner.internal` | Two hops: `bastion.example.com` on port 2222 as `alice`, then `inner.internal`. |

Write an IPv6 address in brackets and put the port after them:
`[2001:db8::1]:2222`, or `alice@[2001:db8::1]`. Without brackets, an
address with colons is read as an address with no port.

**Connect** checks the spec before dialing anything, and refuses it with
the reason when:

- a hop is empty (for example `a,,b`, or a trailing comma): *has an empty
  jump host*;
- a hop names no host (for example `alice@`), or has an empty user name
  (`@host`);
- a port is not a number, or lies outside 1 to 65535;
- a `[` has no closing `]`, or text other than `:port` follows the `]`;
- `none` is mixed with other hops. `none` on its own is allowed and
  connects directly, even when `~/.ssh/config` names a `ProxyJump` for the
  host;
- the jump hosts point back at each other: *jump hosts loop back on
  themselves*, followed by the trail.

Each hop is looked up in `~/.ssh/config` the same way the destination is,
so a jump host written as an alias brings its own `HostName`, `Port`,
`User`, and `IdentityFile`. A jump host whose own entry sets `ProxyJump`
has its hops opened first. A user or port written into the spec
overrides what the config says for that alias. A hop with no
`IdentityFile` authenticates with your agent and default keys, the same
way `ssh` would.

A hop that cannot be reached or does not accept your key reports which one
it was (`Jump host bastion.example.com:22 failed: ...`) and closes the hops
already opened.
On **Disconnect** the hops close from the far end inward.

:::{warning}
Every hop gets the same host-key treatment as the destination:
unknown keys are accepted silently. See
[Host-key verification](../security.md#host-key-verification).
:::

## SSH Keys

- Ed25519, ECDSA and RSA keys work.
- The add-on cannot ask for a passphrase, so a passphrase-protected key
  file cannot authenticate on its own. Use a key file without a
  passphrase, or load the key into `ssh-agent` first.
- PuTTY `.ppk` keys are not supported; convert them to OpenSSH or PKCS#8
  format first.

See [SSH Authentication](../security.md#ssh-authentication) for the rest
of the authentication rules.

## Multiple Users on One Solver Host

Sharing one solver host between several users is possible but not
recommended. The users share the host's GPU. Each launch writes its log
files, including `server.log`, into the **Remote Path** folder, so two
servers started from one folder overwrite each other's logs. Every user's
server uses port `9090` unless a connection profile sets another
`docker_port` (see [The Server Port](#the-server-port)): a second server on
a held port is refused with *Server port N is already in use on the remote
host*. The add-on also writes its launch script to the fixed path
`/tmp/start_server.sh` on the host. See
[Shared Solver Hosts](../security.md#shared-solver-hosts) for a safer
setup.

## Troubleshooting

- **`Connection timed out`**: **Connect** did not finish within 60
  seconds. The 60 seconds cover the whole connect: every jump host, the
  SSH handshake, and the listing of the host's GPUs and solver builds.
  **Cancel**, under the status line, ends the attempt sooner.
- **An SSH error message instead of `Connection timed out`**: a host or
  jump host that does not answer ends each handshake phase (TCP connect,
  protocol banner, authentication) after 20 seconds, so an unreachable
  host normally fails with its own error before the 60 seconds are up.
  Check the host name, the port, and that the host is running.
- **`Jump host <host>:<port> failed: ...`**: one hop of the chain failed;
  the text after the colon says why. See [Jump Hosts](#jump-hosts).
- **`not a valid ... key`**: the key file is in a format the add-on cannot
  read, typically a PuTTY `.ppk` file. Convert it to OpenSSH or PKCS#8
  format. See [SSH Keys](#ssh-keys).
- **`The connection names no directory on the solver host, so its solver
  builds could not be listed.`**: **Remote Path** is empty. Enter the
  solver root.
- **`ppf-cts-server not found under <root> on the solver host, in any
  layout`**: **Remote Path** does not hold a built solver. Point it at the
  folder that holds `target/release/`, `target/cuda/release/`,
  `target/rocm/release/`, or `target/cpu/release/`, and build the solver
  there with `cargo build --release` if there is none, so the solver and
  the Python extension are built next to the server.
- **`The solver host holds the CPU build of the solver under <root> and no
  GPU build`** (or the reverse), and **`The solver host holds these GPU
  builds under <root>: cuda, and no rocm build`**: raised at **Start
  Server on Remote**. The host is right and the selection is not. Click
  the build the host holds in the **GPU** / **CPU** toggle, or set **GPU
  Backend** to a build named in the message or to **Automatic**.
- **`Server port N is already in use on the remote host`**: something on
  the host holds the port. The message calls the button *Kill Process*; it
  is **Force Terminate Process**, shown under the error. Click it to end a
  leftover `ppf-cts-server` on this connection's port. If another program
  holds the port, stop it on the host or set another `docker_port` in a
  connection profile (see [The Server Port](#the-server-port)).
- **`Server startup timed out.`**: the server did not answer within 16
  seconds. The panel prints the last 20 lines of `server.log` from the
  **Remote Path** folder on the host.
- **`Server startup failed: <line>`**: the server reported an error while
  starting, and the panel quotes it. Open `server.log` in the **Remote
  Path** folder on the host for the full output.

See [Troubleshooting](../troubleshooting.md) for messages shared by every
connection type.

:::{admonition} Under the hood
:class: toggle

**Command-mode parser**

The Command-mode parser splits the pasted string with `shlex`, then
walks the tokens against ssh's own option list: the letters that take an
argument (`-p`, `-i`, `-l`, `-J`, `-o`, and the rest) and the letters
that stand alone (`-C`, `-v`, `-4`, and the rest). Matching the real
list is what lets an option consume its own argument, so a bare word is
the destination only when no preceding option claimed it. That is why
`ssh -J alice@bastion.example.com gpu01.example.com` reads the first
`alice@...` token as a jump host rather than as the destination.
Single-letter options cluster (`-Cv`), and an argument may be attached
(`-p2222`) or separate (`-p 2222`). A letter outside both lists is
refused rather than skipped, since ssh would not have accepted it either
and the token after it cannot be classified.

Five values are kept: destination, port, login name, identity file, and
jump hosts. Everything else is accepted and dropped. If no host can be
extracted the operator reports `Failed to parse command. Ensure it
includes host.` and aborts; a refused option, a missing argument, or an
unbalanced quote is reported in its own words. The parser never invokes
the system `ssh` binary; the parsed fields go straight into paramiko.

**`~/.ssh/config` resolution**

The add-on parses `~/.ssh/config` (first-match, with `Include` directives
resolved) for the **Host** text, whether it is an alias or a DNS name, and
fills in `HostName`, `Port`, `User`, `IdentityFile`, and `ProxyJump` from
the entries that match it. Later matching entries fill in fields earlier
entries left blank, so a trailing wildcard `Host *` block provides
defaults without overriding explicit blocks, and applies to a DNS name as
well as to an alias. If the config file is missing or no entry matches,
the **Host** text is used as the hostname verbatim. Only the seven
keywords listed in
{ref}`Supported ssh_config options <supported-ssh-config-options>` are
read; the parser tokenizes each non-comment line on whitespace or `=`,
matches the first word case-insensitively, and drops the line if the
keyword isn't one it recognizes.

The panel's values take precedence over the config only where the panel
has a value: **User**, **SSH Key**, and **Proxy Jump** are used when they
are not empty, and **Port** when it is not `22`.

**Jump-host chain**

A jump spec is resolved before the connection is dispatched: a hop that
carries a `ProxyJump` of its own is expanded first, so the result is a
flat list of hops ordered outward from the workstation. A host that
reappears while a hop's own `ProxyJump` entries are being expanded is a
loop and is refused there, so a config that points two hosts at each
other fails with the trail it followed rather than recursing.

paramiko then opens one client per hop. Hop 0 is dialed directly; every
later hop, and finally the solver host, is dialed over a `direct-tcpip`
channel opened on the previous hop, which is the same mechanism ssh uses
for `ProxyJump`. The destination address handed to each channel is the
one this workstation resolved, so the name is resolved once here rather
than depending on what the jump host's resolver would answer. The
keepalive that covers the session covers each hop as well.

**Host-key policy**

Unknown host keys are accepted silently (paramiko `AutoAddPolicy`).
This is not hardened against MITM attacks and should not be relied on
for untrusted networks.

**paramiko install path**

The **Install Paramiko to Add-on Directory** button runs
`pip install --target <dir>` into Blender's user
`scripts/addons/modules` directory on a background thread. That path is
already on `sys.path` and is outside the extension tree, which is why
it is used rather than a directory inside the add-on. The import itself
is a plain `importlib.import_module`, but the presence test looks in
that directory specifically, and it gates the **Connect** button as well
as the banner, so a paramiko installed anywhere else on Blender's
`sys.path` still leaves **Connect** unavailable.
:::
