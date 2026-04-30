# 🐳 Docker (Local)

The solver runs inside a Docker container on the **same machine** as
Blender, managed by the local Docker daemon. For a remote container
reached over SSH, see [Docker over SSH](docker_over_ssh.md).

```{figure} ../images/connections/docker_local_topology.svg
:alt: Block diagram showing the Blender add-on, the Docker daemon, and the container running server.py all inside a single Blender workstation box. Lifecycle commands travel add-on -> docker-py -> daemon -> exec -> container; server traffic goes add-on -> socket to localhost:port -> container.
:width: 760px

Where each piece lives, and how the add-on reaches it, in local Docker
mode. Blue solid arrows carry lifecycle commands (container
start/stop, `exec`). Purple dashed arrows carry the TCP connection to
`server.py`. The container must publish the server port -- that is
what lets the socket reach `server.py`. For the remote variant where
the daemon and container live on a different host, see
[Docker over SSH](docker_over_ssh.md).
```

## When to Use It

- The solver depends on CUDA/driver versions you do not want to install
  on the host.
- You want each project isolated on the workstation.

## Setup

1. Set **Server Type** to `Docker`.
2. Fill **Container** with the Docker container name (default
   `ppf-dev`). The container must already exist on the daemon; the
   add-on will start it if it is stopped, but it will not create one.
3. Fill **Container Path** with the working directory **inside** the
   container (default `/root/ppf-contact-solver`). This is the
   directory containing `server.py`, *not* a path on the Blender host.
4. Set **Docker Port** to the TCP port `server.py` listens on inside
   the container (default `9090`).
5. Click **Connect**. If the container exists but is stopped, the
   add-on starts it for you. A missing container is reported as an
   error.

```{figure} ../images/connections/docker.png
:alt: Backend Communicator panel in Docker mode
:width: 500px

Backend Communicator with **Server Type** set to `Docker`.
**Container**, **Container Path**, and **Docker Port** replace the SSH
fields. The **Install Docker-Py** banner appears when the vendored
module is missing. **Connect** is highlighted.
```

### Fields

| Field | Description |
| ----- | ----------- |
| Container | Docker container name (e.g. `ppf-dev`). Must already exist on the daemon. |
| Container Path | Working directory **inside** the container that holds `server.py` (e.g. `/root/ppf-contact-solver`). Not a host path. |
| Docker Port | TCP port inside the container where `server.py` listens (default `9090`). Must be published with `-p` when the container was created. |

## Installing docker-py

The Docker backend requires the `docker` Python package (sometimes
called `docker-py`). When it is missing the main panel shows an
**Install Docker** button that installs the package into the add-on's
private library on a background thread.

## Troubleshooting

- **`Container 'X' does not exist.`** - the name is wrong or the
  container was removed. Run `docker ps -a` to see what is actually
  there.
- **`Error starting container 'X'`** - the daemon returned a non-zero
  exit or the user lacks `docker` group membership.
- **Server startup timed out.** - the container started but `server.py`
  did not become ready within 16 seconds. Check `server.log` inside the
  directory set in **Container Path**; the panel prints the last 20
  lines automatically.
- **`docker-py` not found** - click **Install Docker** on the main
  panel and wait for the background installer to finish.

:::{admonition} Under the hood
:class: toggle

**Transport**

The local backend talks to the Docker daemon through the standard
`docker` Python client and looks up the container by name. A stopped
container is started automatically; a missing container aborts connect
with `Container 'X' does not exist.`

**Server startup path**

Local Docker uses the same Unix server-launch path as the SSH and
Local backends (see {ref}`Connections - Under the hood <connections-under-the-hood>`):
a small script inside the container launches `server.py` on the
configured port and the UI waits up to 16 s for readiness. The local
mode does not perform the explicit `docker port` published-port check
that [Docker over SSH](docker_over_ssh.md) does.
:::
