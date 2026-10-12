# 🐳 Docker over SSH

Use this type when the solver runs inside a Docker container on a **remote**
Linux host. The add-on opens an SSH session to that host and runs `docker`
commands there. Two **Type** entries cover the two ways to provide SSH
credentials: **Docker over SSH** (Custom mode, explicit fields) and
**Docker over SSH Command** (Command mode, parsed from a raw `ssh ...`
string).

```{figure} ../images/connections/docker_ssh_topology.svg
:alt: Block diagram split into two boxes. The left box (Blender workstation) holds only the add-on. The right box (remote Linux host) holds the Docker daemon and the container running ppf-cts-server. Lifecycle commands travel add-on -> SSH -> remote -> docker exec -> container; server traffic rides an SSH tunnel from the add-on to the container's published port on the remote's localhost. A pill states that -p PORT:PORT must be published on the container.
:width: 760px

Where each piece runs, and how the add-on reaches it. Blue solid
arrows carry lifecycle commands (container start/stop, `exec`, port
check). Purple dashed arrows carry the TCP connection to `ppf-cts-server`,
which rides an SSH tunnel into the container's published port on the
remote's localhost. For the local variant, where the daemon and
container run on the Blender machine, see [Docker (Local)](docker.md).
```

## When to Use It

- A GPU host runs Docker for you and you reach it over SSH.
- Your administrator gives you SSH access to a Docker host (your account
  can run `docker`) and a container to use.
- Multiple solvers share one remote GPU host and you want each project
  isolated.

## Before You Start

- The SSH account on the remote host can run `docker`, for example because
  it is in the `docker` group. The add-on runs `docker ps`, `docker inspect`,
  `docker start`, `docker port`, and `docker exec` over the SSH session.
- A container for the solver exists on the remote Docker daemon. The add-on
  starts a stopped container but never creates one. Create it on the remote
  host the way [Docker (Local)](docker.md#setup) describes: `--name
  ppf-contact-solver`, `--gpus all`, and `-p 9090:9090`.
- The container publishes the server port; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`, which also
  covers the network exposure the mapping causes.
- The `paramiko` Python package is installed in Blender; see
  [Installing paramiko](#installing-paramiko).

## Setup - Custom Mode

The SSH fields from the [SSH (Direct)](ssh.md) page are combined with the
Docker fields: the add-on opens an SSH session to the remote host and runs
every Docker command there.

1. Set **Type** to **Docker over SSH**.
2. Fill **Host**, **Port**, **User**, and **SSH Key** as in
   [SSH Custom mode](ssh.md#setup---custom-mode). They locate the remote
   host that runs the Docker daemon. Fill **Proxy Jump** as well if that
   host is only reachable through a bastion (see
   {ref}`Jump Hosts <jump-hosts>`). **SSH Key** must hold no spaces or shell
   special characters. On Windows the default path starts in your user
   folder, which contains a space when your account name does, so point the
   field at a copy of the key in a path without one.
3. Set **Container** to the container name on the *remote* daemon (default
   `ppf-contact-solver`). The add-on runs `docker exec` against this name
   over the SSH session.
4. Set **Container Path** to the solver root **inside** the container: the
   directory that has `target/release/ppf-cts-server` (or one of the other
   layouts) beneath it, for example `/root/ppf-contact-solver`. It is not a
   path on the Blender machine or on the remote host's filesystem, and not
   the `target/release` folder itself.
5. Set **Project Name** to a short name made of letters, digits, `.`, `-`
   and `_`.
6. Set **Docker Port** to the TCP port `ppf-cts-server` listens on inside
   the container (default `9090`).
7. Click **Connect**. The add-on checks that the container exists on the
   remote host and starts it if it is stopped. If **Connect** is grayed
   out, see {ref}`When Connect is unavailable <connect-unavailable>`.
8. Click **Start Server on Remote**. The button reads **Server Starting...**
   while the server starts, and once the server answers (within 16 seconds)
   the status line changes to *Status: Waiting for Data*, for a server that
   holds no scene yet.

**Container**, **Container Path**, **Docker Port**, and **Project Name**
are the same fields as in [Docker (Local)](docker.md#fields), except that
**Container** is looked up on the remote daemon and **Container Path** is
the path seen from inside the remote container.

Up to three more controls appear in the Connection box **once you are
connected**:

- The **GPU** / **CPU** toggle (the **Compute Device** setting).
- The **GPU Backend** dropdown (**Automatic**, **CUDA**, or **ROCm**), drawn
  only with the toggle on **GPU** and where the container holds more than one
  GPU build.
- The **GPU** dropdown, drawn only with the toggle on **GPU**. It lists the
  NVIDIA devices that `nvidia-smi` reports inside the container. The refresh
  button beside it lists the container's GPUs and solver builds again.

The add-on reads the builds and GPUs in the container at **Connect**; the
choices are applied at **Start Server on Remote**. See
{ref}`Choosing the build <choosing-the-build>` and
{ref}`Picking a GPU <gpu-picker>`.

## Setup - Command Mode

Command mode is Custom mode with the SSH parameters taken from a pasted
command instead of separate fields.

1. Set **Type** to **Docker over SSH Command**.
2. Paste the SSH command into **SSH Command**, for example
   `ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com`. To go through
   a bastion, add `-J` to the command (see {ref}`Jump Hosts <jump-hosts>`).
   The [SSH Command section](ssh.md#setup---command-mode) describes the
   parser rules.
3. Set **Container** to the container name on the remote daemon. It is a
   separate field and is never read from the command string.
4. Set **Container Path** to the solver root **inside** that container: the
   directory that has `target/release/ppf-cts-server` (or one of the other
   layouts) beneath it, for example `/root/ppf-contact-solver`, not the
   `target/release` folder itself. It is also a separate field and is never
   derived from the SSH command.
5. Set **Project Name** (letters, digits, `.`, `-`, `_`) and **Docker
   Port**.
6. Click **Connect**, then **Start Server on Remote**.

Only the host, port, user, key path, and jump hosts come from the pasted
command. The panel shows **SSH Command**, **Container**,
**Container Path**, **Project Name**, and **Docker Port**.

```{figure} ../images/connections/docker_over_ssh.png
:alt: Backend Communicator panel with Type set to Docker over SSH Command, showing the SSH Command, Container, Container Path, Project Name and Docker Port fields above the Connect button
:width: 500px

Backend Communicator with **Type** set to **Docker over SSH Command**. The
**SSH Command** field replaces the separate SSH inputs, and **Container** and
**Container Path** locate the solver inside the container.
```

## Installing paramiko

Docker over SSH needs only the `paramiko` Python package: the remote daemon
is driven by `docker` commands run over the SSH session, not by docker-py.
While `paramiko` is missing, the panel shows an **Install Paramiko to Add-on
Directory** button, and **Connect** stays grayed out. The button installs
into Blender's user `scripts/addons/modules` directory. The installer
downloads the package with pip, so Blender needs internet access. The panel
shows *Installing...* while it runs, and any error afterwards.

## Troubleshooting

- **`Error: ...`** after **Connect** - `docker ps` or `docker inspect`
  failed on the remote host, and the host's own message follows. Docker may
  not be installed, its daemon may be stopped, or your SSH account may lack
  permission to run `docker`.
- **`Container 'X' does not exist.`** - the name is wrong or the container
  was removed on the remote. Run `docker ps -a` on the remote to see what is
  actually there.
- **`Error starting container 'X': ...`** - the container exists, but
  `docker start` failed. The text after the colon is the daemon's reason.
- **`Docker port N is not exposed on container 'X' ...`** - the container
  does not publish the server port. Recreate it with `-p N:N`; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`.
- **`Server port N is already in use on the remote host ...`** - the server
  could not bind the port. A stale `ppf-cts-server` in the container is a
  likely cause: press **Force Terminate Process** (the message calls it "Kill
  Process"). To use another port, press **Disconnect**, set a different
  **Docker Port** (the container must publish it), and press **Connect**
  again.
- **`Server reached SERVER_READY but the client cannot reach it ...`** - the
  server started, but the SSH tunnel cannot reach it on the remote host.
  Check that the container publishes the port; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`.
- **`Server startup timed out.`** - the container started but
  `ppf-cts-server` did not become ready within 16 seconds. Check
  `server.log` inside the directory set in **Container Path**; the panel
  prints the last 20 lines automatically.

For SSH-level errors (authentication, jump hosts, key files), see
[Troubleshooting](../troubleshooting.md) and the [SSH (Direct)](ssh.md)
page.

:::{admonition} Under the hood
:class: toggle

**Transport**

The add-on opens one SSH session to the remote host and runs every Docker
command there. Commands for the solver are wrapped as
`docker exec -w <Container Path> <container> ...`. During connect, the
add-on checks that the container exists and starts it if it is stopped; a
missing container aborts the connect. The solver TCP connection rides an
SSH tunnel to `localhost:<Docker Port>` on the remote host.

**Port publication check**

Before **Start Server on Remote** the add-on runs

```sh
docker port <container> <port>
```

on the remote SSH host. A non-zero exit code aborts with the *Docker port N
is not exposed on container 'X'* message.

**Server startup path**

Docker over SSH uses the same Unix server-launch path as the SSH backends
(see {ref}`Connections - Under the hood <connections-under-the-hood>`): a
small script inside the container starts the `ppf-cts-server` that
**Compute Device** and **GPU Backend** resolve to, with `--host 0.0.0.0` on
the configured port, and the UI waits up to 16 seconds for readiness.
:::
