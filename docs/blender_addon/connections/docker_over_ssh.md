# 🐳 Docker over SSH

The solver runs inside a Docker container on a **remote** Linux host,
managed by a Docker daemon reached through SSH. Two UI modes cover the
two ways to provide SSH credentials: **Custom** (explicit fields) and
**Command** (parsed from a raw `ssh ...` string).

```{figure} ../images/connections/docker_ssh_topology.svg
:alt: Block diagram split into two boxes. The left box (Blender workstation) holds only the add-on. The right box (remote Linux host) holds the Docker daemon and the container running server.py. Lifecycle commands travel add-on -> SSH -> remote -> docker exec -> container; server traffic rides an SSH tunnel from the add-on to the container's published port on the remote's localhost. A pill reminds the reader that -p PORT:PORT must be published on the container.
:width: 760px

Where each piece lives, and how the add-on reaches it. Blue solid
arrows carry lifecycle commands (container start/stop, `exec`, port
check). Purple dashed arrows carry the TCP connection to `server.py`,
which rides an SSH tunnel into the container's published port on the
remote's localhost. For the local variant where the daemon and
container run on the Blender machine, see [Docker (Local)](docker.md).
```

## When to Use It

- A GPU host runs Docker for you and you reach it over SSH.
- Your cluster administrator hands you a container instead of shell
  access on the bare host.
- Multiple solvers share one remote GPU host and you want each project
  isolated.

## Setup - Custom Mode

The SSH fields from the [SSH (Direct)](ssh.md) page are combined with
the Docker fields: the add-on opens an SSH session to the remote host
and runs every Docker command there.

1. Set **Server Type** to `Docker over SSH`.
2. Fill **Host** / **Port** / **User** / **SSH Key** as in SSH
   Custom mode -- these locate the remote host that runs the Docker
   daemon.
3. Fill **Container** with the container name on the *remote* daemon
   (e.g. `ppf-dev`). The add-on runs `docker exec` against this name
   over the SSH session.
4. Fill **Container Path** with the working directory **inside that
   container** (default `/root/ppf-contact-solver`). This path
   is not interpreted on the Blender host or on the remote host's
   filesystem -- it is the path seen from inside the container, where
   `server.py` lives.
5. Set **Docker Port** to the TCP port `server.py` listens on inside
   the container.
6. Click **Connect**. The add-on verifies that the container exists on
   the remote host and starts it if it is stopped.

```{figure} ../images/connections/docker_over_ssh.png
:alt: Backend Communicator panel in Docker over SSH Command mode
:width: 500px

Backend Communicator with **Server Type** set to `Docker over SSH
Command`. The **SSH Command** field replaces the per-field SSH inputs
(in Custom mode you would see **Host** / **Port** / **User** / **SSH
Key** instead). **Container**, **Container Path**, and **Docker Port**
are the same fields as in [Docker (Local)](docker.md). **Connect** is
highlighted.
```

| Field | Description |
| ----- | ----------- |
| Container | Container name on the *remote* Docker daemon. Must already exist. |
| Container Path | Working directory **inside** the remote container that holds `server.py`. |
| Docker Port | Port inside the container where `server.py` listens. Must be published on the container with `-p`. |

:::{warning}
The server port must be published on the container (`-p 9090:9090` or
equivalent in your compose file). Before **Start Server**, the add-on
checks the port mapping on the remote host and refuses to continue if
the port is not exposed -- the error text tells you exactly which port
and container failed. You must fix this on the container side; the
add-on cannot publish ports on a container that is already created.
:::

## Setup - Command Mode

Identical to Custom Mode, but the SSH parameters come from a pasted
command instead of separate fields.

1. Set **Server Type** to `Docker over SSH Command`.
2. Paste the SSH command into **SSH Command**, e.g.
   `ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com`. See the
   [SSH Command section](ssh.md#setup---command-mode) for the parser
   rules.
3. Fill **Container** with the container name on the remote daemon
   (the same field as in Custom Mode -- it is *not* parsed from the
   command string).
4. Fill **Container Path** with the working directory **inside** that
   container (where `server.py` lives). This is also a separate field
   and is never derived from the SSH command.
5. Set **Docker Port** and click **Connect**.

The **Container** and **Container Path** fields behave exactly as in
Custom Mode -- only the host / port / user / key path are sourced from
the pasted command.

## Installing paramiko and docker-py

Docker over SSH needs both the `paramiko` and `docker` Python packages.
The main panel shows **Install Paramiko** and **Install Docker** buttons
when either is missing; both install into the add-on's private library
directory.

## Troubleshooting

- **`Container 'X' does not exist.`** - the name is wrong or the
  container was removed on the remote. Run `docker ps -a` on the remote
  to see what is actually there.
- **`Error starting container 'X'`** - the daemon returned a non-zero
  exit or the remote user lacks `docker` group membership.
- **Server startup timed out.** - the container started but `server.py`
  did not become ready within 16 seconds. Check `server.log` inside the
  directory set in **Container Path**; the panel prints the last 20
  lines automatically.

:::{admonition} Under the hood
:class: toggle

**Transport**

An SSH session is opened to the remote host and every Docker command
is wrapped in `docker exec -w <cwd> <container> ...` on that session.
During connect the add-on checks that the container exists and starts
it if it is stopped; a missing container aborts the connect.

**Port publication check**

Before **Start Server** the add-on runs

```sh
docker port <container> <port>
```

on the remote SSH host. A non-zero exit code aborts with:

> Docker port 9090 is not exposed on container 'ppf-dev'. Please expose
> the port with '-p 9090:9090' when starting the container.

Fix this on the container side by re-running `docker run -p 9090:9090`
(or editing your `compose.yaml`); the add-on cannot publish ports on
an existing container.

**Server startup path**

Docker over SSH uses the same Unix server-launch path as the SSH and
Local backends (see {ref}`Connections - Under the hood <connections-under-the-hood>`):
a small script inside the container launches `server.py` on the
configured port and the UI waits up to 16 s for readiness.
:::
