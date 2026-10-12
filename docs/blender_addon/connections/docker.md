# 🐳 Docker (Local)

Use this type when the solver runs inside a Docker container on the **same
machine** as Blender, managed by the local Docker daemon. For a container
on a remote host, see [Docker over SSH](docker_over_ssh.md).

```{figure} ../images/connections/docker_local_topology.svg
:alt: Block diagram showing the Blender add-on, the Docker daemon, and the container running ppf-cts-server all inside a single Blender workstation box. Lifecycle commands travel add-on -> docker-py -> daemon -> exec -> container; server traffic goes add-on -> socket to localhost:port -> container.
:width: 760px

Where each piece runs, and how the add-on reaches it, in local Docker
mode. Blue solid arrows carry lifecycle commands (container
start/stop, `exec`). Purple dashed arrows carry the TCP connection to
`ppf-cts-server`, which needs the container to publish the server port.
For the remote variant, where the daemon and container run on a
different host, see [Docker over SSH](docker_over_ssh.md).
```

## When to Use It

- The solver depends on CUDA or driver versions you do not want to install
  on this machine.
- You want each project isolated on the workstation.

## Setup

Before you start:

- Docker is installed and running on this machine.
- A container for the solver exists. The add-on starts a stopped container
  but never creates one. Create it with the `docker run` command in the
  Docker section of the
  [project README](https://github.com/st-tech/ppf-contact-solver/blob/main/README.md).
  That command names the container `ppf-contact-solver` with `--name`,
  passes `--gpus all` to give the container access to your GPU, and
  publishes the server port (`MY_BLENDER_PORT`, `9090` by default) with
  `-p`. It also publishes the JupyterLab port and passes `--rm`, so stopping
  the container deletes it and you run the command again next time.
- The `docker` Python package is installed in Blender. If the panel shows
  **Install Docker-Py to Add-on Directory**, see
  [Installing docker-py](#installing-docker-py).

Then, in the Connection box:

1. Set **Type** to **Docker**.
2. Set **Container** to the container name, as `docker ps -a` lists it
   (default `ppf-contact-solver`, the name the README command creates).
3. Set **Container Path** to the solver root **inside** the container: the
   directory that has `target/release/ppf-cts-server` beneath it. The
   default, `/root/ppf-contact-solver`, is right for the image the README
   command runs. It is not a path on this machine, and not the
   `target/release` folder itself.
4. Set **Project Name** to a short name made of letters, digits, `.`, `-`
   and `_`. **Connect** stays grayed out until the name is valid.
5. Set **Docker Port** to the TCP port `ppf-cts-server` listens on inside
   the container (default `9090`). The container must publish this port;
   see {ref}`Publishing the port for Docker <docker-port-publishing>`.
6. Click **Connect**. The add-on starts the container if it is stopped.
   It reports an error if the Docker daemon cannot be reached, if the
   container does not exist, or if the container does not publish the
   server port. If **Connect** is grayed out, see
   {ref}`When Connect is unavailable <connect-unavailable>`.
7. Click **Start Server on Remote**. The button reads **Server Starting...**
   while the server starts, and once the server answers (within 16 seconds)
   the status line changes to *Status: Waiting for Data*, for a server that
   holds no scene yet.

```{figure} ../images/connections/docker.png
:alt: Backend Communicator panel with Type set to Docker. A banner at the top reads Docker-Py needs to be installed and offers the Install Docker-Py to Add-on Directory button. The Connection box shows Container ppf-contact-solver, Container Path /root/ppf-contact-solver, a Project Name, and Docker Port 9090. The Connect button is outlined in red and grayed out.
:width: 500px

Backend Communicator with **Type** set to **Docker**, before docker-py is
installed. **Container**, **Container Path**, and **Docker Port** replace
the SSH fields. The **Install Docker-Py to Add-on Directory** banner
appears while the `docker` module is missing, and **Connect** (outlined in
red) stays grayed out until the install finishes.
```

### Fields

| Field | Default | Description |
| ----- | ------- | ----------- |
| Container | `ppf-contact-solver` | Docker container name. Must already exist on the daemon. |
| Container Path | `/root/ppf-contact-solver` | Solver root **inside** the container: the directory that holds `ppf-cts-server` under `target/release/`, `target/cuda/release/`, `target/rocm/release/`, or `target/cpu/release/`. Not a path on this machine, and not the `target/release` folder itself. Holds no spaces or shell special characters. |
| Project Name | `unnamed` | Letters, digits, `.`, `-` and `_` only. The same field on every connection type. |
| Docker Port | `9090` | TCP port inside the container where `ppf-cts-server` listens, 1024-65535. The container must publish it; see {ref}`Publishing the port for Docker <docker-port-publishing>`. The connection reads it at **Connect**. |

Up to three more controls appear in the Connection box **once you are
connected**:

- The **GPU** / **CPU** toggle (the **Compute Device** setting).
- The **GPU Backend** dropdown (**Automatic**, **CUDA**, or **ROCm**), drawn
  only with the toggle on **GPU** and where the container holds more than one
  GPU build.
- The **GPU** dropdown, drawn only with the toggle on **GPU**. It lists the
  NVIDIA devices that `nvidia-smi` reports inside the container. The refresh
  button beside it lists the container's GPUs and solver builds again.

The image the README command runs holds a GPU build in `target/release/` and
a CPU build in `target/cpu/release/`, so either **Compute Device** setting
works. The choices are applied at **Start Server on Remote**. See
{ref}`Choosing the build <choosing-the-build>` and
{ref}`Picking a GPU <gpu-picker>`.

## Installing docker-py

The **Docker** type needs the `docker` Python package (also called
docker-py). While it is missing, the panel shows the message *Docker-Py
needs to be installed.* and an **Install Docker-Py to Add-on Directory**
button, and **Connect** stays grayed out. The button installs the package
into Blender's user `scripts/addons/modules` directory on a background
thread. The installer downloads the package with pip, so Blender needs
internet access. The panel shows *Installing...* while it runs, and any
error afterwards.

## Troubleshooting

- **`Cannot reach the Docker daemon (...)`** - the full message ends with
  *Check that Docker is running, and on Linux that your user is in the
  'docker' group.* Start Docker (Docker Desktop on Windows), or add your
  user to the `docker` group on Linux and log in again.
- **`Container 'X' does not exist. ...`** - the name is wrong or the
  container was removed. Run `docker ps -a` to see what is actually there.
  Create the container with the README command if it is gone.
- **`Error starting container 'X': ...`** - the container exists, but
  starting it failed. The text after the colon is the daemon's reason.
- **`Container 'X' does not publish port N to this machine ...`** - the
  container was created without `-p N:N`. Recreate it with the flag; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`.
- **`Server port N is already in use on the remote host ...`** - the
  server could not bind the port. Despite the wording, **remote host** means
  the container here. A stale `ppf-cts-server` is a likely cause: press
  **Force Terminate Process** (the message calls it "Kill Process"). To use
  another port, press **Disconnect**, set a different **Docker Port** (the
  container must publish it), and press **Connect** again.
- **`Server reached SERVER_READY but the client cannot reach it ...`** -
  the server started, but this machine cannot reach it on **Docker Port**.
  Check that the container publishes the port; see
  {ref}`Publishing the port for Docker <docker-port-publishing>`.
- **`Server startup timed out.`** - the container started but
  `ppf-cts-server` did not become ready within 16 seconds. Check
  `server.log` inside the directory set in **Container Path**; the panel
  prints the last 20 lines automatically.

For problems outside this list, see [Troubleshooting](../troubleshooting.md).

:::{admonition} Under the hood
:class: toggle

**Transport**

The add-on connects to the Docker daemon through the standard `docker`
Python client and looks up the container by name. A stopped container is started
automatically; a missing container aborts the connect with
`Container 'X' does not exist.` Commands run in the container through the
daemon's exec call, with **Container Path** as the working directory. The
solver TCP connection goes straight to `localhost:<Docker Port>` and does
not pass through the daemon.

**Server startup path**

Local Docker uses the same Unix server-launch path as the SSH backends
(see {ref}`Connections - Under the hood <connections-under-the-hood>`): a
small script inside the container starts the `ppf-cts-server` that
**Compute Device** and **GPU Backend** resolve to, with `--host 0.0.0.0` on
the configured port, and the UI waits up to 16 seconds for readiness. Local
Docker checks the published port at **Connect** rather than before
**Start Server on Remote**, and reads the container's own port map through
docker-py instead of running `docker port`. A container on `--network host`
publishes nothing and is accepted as is.
:::
