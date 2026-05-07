# 🖥️ Local

The solver runs on the same **Linux** machine as Blender, with no SSH or
Docker layer in between. On Windows, use the [Windows Native](windows.md)
backend instead; macOS is not supported as a solver backend (the solver
requires CUDA).

:::{warning}
Not recommended for day-to-day workstations. The Linux installation performed
by [`warmup.py`](https://github.com/st-tech/ppf-contact-solver/blob/main/warmup.py)
is destructive (system package installs via `apt`, CUDA toolkit, shell rc
edits, user-level venv, nvm, rustup, NTP) and is impossible to revert or
uninstall cleanly. Even if you have a Linux machine with an NVIDIA GPU, we strongly
recommend the [Docker](docker.md) backend instead. Local mode is intended for
the special case where Blender and the solver both live on a disposable VM
or short-lived environment that you are willing to wipe.
:::

## When to Use It

- You are on a **disposable** Linux VM, cloud instance, or short-lived
  test rig where you do not care about the host being modified by
  `warmup.py`.
- That same machine happens to run both Blender and the solver, so
  there is no remote host to SSH into and no reason to spin up Docker.
- You are iterating on solver code on such a machine and want the
  fastest possible disconnect/reconnect turnaround, with file transfers
  copying directly on disk instead of going through SSH or Docker.

If your Linux box is your real workstation, use the [Docker](docker.md)
backend instead, even though the GPU is local.

## Setup

1. Set **Server Type** to `Local`.
2. Fill **Path** with the directory that contains the
   `ppf-cts-server` binary (typically `target/release/` inside your
   solver checkout).
3. Set **Project Name** on the main panel.
4. Click **Connect**. The add-on checks that `ppf-cts-server` exists
   at the path you gave it.
5. Click **Start Server**. The panel waits a few seconds for the server
   to report that it is ready.

```{figure} ../images/connections/local.png
:alt: Backend Communicator panel in Local mode
:width: 500px

Backend Communicator with **Server Type** set to `Local`. Only **Path**
and **Project Name** show up; no SSH, Docker, or Windows-native fields.
**Connect** is highlighted.
```

:::{tip}
If you ran the solver's installer, it may have created a Python virtual
environment at `$HOME/.local/share/ppf-cts/venv`. When the add-on finds
that venv it activates it automatically before launching the server --
you do not need to do anything.
:::

## Fields

| Field | Description |
| ----- | ----------- |
| Path | Directory containing the `ppf-cts-server` binary. Default `~/ppf-contact-solver`. |

The local `ppf-cts-server` port is fixed at `9090` in Local mode; the
panel does not expose a Server Port field here (it is only editable in
Docker-family modes).

## Dependencies

Local mode requires neither `paramiko` nor `docker-py`. The main panel's
**Install Paramiko** and **Install Docker** buttons are only relevant for
SSH and Docker modes.

## Troubleshooting

- **"Remote path not found (.../ppf-cts-server)"** - the path you
  entered does not contain the `ppf-cts-server` binary. Point it at the
  directory holding the binary (typically `target/release/`), not the
  checkout root or a source subdirectory.
- **Server startup timed out.** - the solver launched but did not report
  readiness within 16 seconds. Check `server.log` inside the solver
  directory; the panel also prints the last 20 lines when the timeout
  fires.
- **Port already in use.** - another solver (or a stale `ppf-cts-server`
  process) is already bound to the port. Click **Stop Server** first, use
  the **Force Terminate Process** button shown next to the port-in-use
  error, or change the Server Port.

:::{admonition} Under the hood
:class: toggle

**Launch script**

Local mode launches the server with a bash script (`nohup`, `source
.../bin/activate`, `./target/release/ppf-cts-server`). That is the same
launch path the SSH and Docker backends use. The script is `bash`-only,
which is why Local mode is Linux-only; Windows goes through the Windows
Native backend and macOS is not supported (the solver requires CUDA).

**Shared port field**

The **Server Port** field is the same underlying property for every
connection type. The label in profile TOML files is `docker_port` for
historical reasons, even on Local connections.

**Virtual environment activation**

Local mode reuses the same launch script as the SSH and Docker
backends: it sources `$HOME/.local/share/ppf-cts/venv/bin/activate` if
that file exists. The Rust server spawns a Python build worker that
imports the `_ppf_cts_py` PyO3 module from that venv. The solver's
own install scripts are responsible for creating the venv; the add-on
never creates or modifies it.

**File transfer fast path**

On local connections, file transfers copy directly on disk instead of
going through the solver TCP socket: no CBOR-over-TCP overhead, and
much faster than the SSH or Docker paths. The trade-off is cosmetic:
the panel does not display a bandwidth figure while a local transfer
is in progress. That is expected on this backend.
:::
