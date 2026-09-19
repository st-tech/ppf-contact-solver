# File: connection.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection establishment/teardown functions extracted from client.py.

import os
import posixpath
import subprocess
import sys

from .gpu_devices import AUTOMATIC, apply_selection
from .protocol import PROTOCOL_VERSION
from .status import ConnectionInfo


class PortInUseByForeignProcess(Exception):
    """A process is bound to the port the addon wants.

    Raised when the addon's bind attempt fails because the port is
    already taken. The connect path does not kill the holder itself; the
    panel offers Force Terminate Process under this message for the local server
    types, and the user can also end the process by hand before retrying.
    """

    def __init__(self, port: int, detail: str = ""):
        msg = f"Port {port} is in use"
        if detail:
            msg += f" ({detail})"
        msg += (
            "; press Force Terminate Process, or stop the process holding it yourself, "
            "before starting the solver server."
        )
        super().__init__(msg)
        self.port = port
        self.detail = detail


def _port_is_in_use(port: int) -> bool:
    """Return True iff the port is bound on the loopback interface.

    Uses a non-blocking bind probe. Avoids netstat/wmic/PowerShell
    introspection that's flaky on locked-down or modern Windows hosts.
    """
    import socket as _socket

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
    except OSError:
        return True
    finally:
        s.close()
    return False


def _probe_ppf_cts_server(port: int, timeout: float = 1.5) -> bool:
    """True if a ppf-cts-server is alive on *port*."""
    return _query_ppf_cts_server(port, timeout) is not None


def _query_ppf_cts_server(port: int, timeout: float = 1.5, name: str = ""):
    """The status response of a ppf-cts-server on *port*, or ``None``.

    Sends a minimal TCMD ping (length-prefixed empty-args header) and
    requires the response to be valid JSON containing
    ``protocol_version``. That field is on every response shape, so the
    check distinguishes our server from arbitrary other listeners
    (e.g. a Jupyter notebook server that someone parked on 9090).

    Used by the locally-launched connect paths (win_native, mac_native) so a
    Blender restart can re-attach to a still-running server from the previous
    session instead of failing with PortInUseByForeignProcess, and so
    ``check_running_server`` can read which build that server runs.

    *name* is the project the ping names, and it is not inert: the server
    SELECTS whichever project a ping names. The default ``__probe__`` is a
    project nobody uses, and a caller that knows the add-on's own project
    passes it instead, which sends exactly the ping the add-on's first status
    poll sends and so leaves the server's selection where that poll puts it.
    """
    import json as _json
    import socket as _socket

    payload = f"--name {name or '__probe__'}".encode("utf-8")
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect(("127.0.0.1", port))
            s.sendall(b"TCMD")
            s.sendall(len(payload).to_bytes(4, "big"))
            s.sendall(payload)
            buf = b""
            while True:
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
                # Cap so a chatty foreign listener can't keep us reading
                # forever; our real responses are well under this.
                if len(buf) > 64 * 1024:
                    return None
        finally:
            s.close()
    except OSError:
        return None
    if not buf:
        return None
    try:
        resp = _json.loads(buf.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError):
        return None
    if isinstance(resp, dict) and "protocol_version" in resp:
        return resp
    return None


# THE COMPUTE DEVICE A LOCAL NATIVE BACKEND RUNS ON.
#
# The backend is a property of a BUILD rather than a runtime flag: the solver
# links one backend, and `crates/ppf-cts-solver/build.rs` refuses to put a
# second one in a directory that already holds another. So choosing GPU or CPU
# is choosing WHICH BUILD DIRECTORY the server comes from, and a tree that has
# both has them in two: `target/release` and `target/cpu/release`, which is the
# layout `build.rs` prescribes in its own refusal message.
#
# "GPU" is deliberately not spelled "CUDA" or "Metal". The artist is answering
# "accelerated or not"; which accelerator the host has is the host's business.
DEVICE_GPU = "GPU"
DEVICE_CPU = "CPU"

# THE BACKENDS WHOSE SERVER RUNS ON THIS MACHINE AND IS STARTED BY THE ADD-ON,
# mapped to the name their messages call them by.
#
# ONE LIST, because several places have to agree about it and each of them used
# to spell its own: the launch that spawns a child rather than writing a shell
# script, the stop that kills a listener on a loopback port, the direct-disk
# transfer, and the Force Terminate Process button, which works without a
# connection exactly for these. A backend added to one spelling and not another
# is a silent gap rather than an error, which is what a shared list removes.
NATIVE_BACKENDS = {
    "win_native": "Windows Native",
    "mac_native": "macOS Native",
    "linux_native": "Linux Native",
}

# Where each device's server lives, relative to the root the user picked.
#
# ONE TABLE PER PLATFORM, because they do NOT accept the same layouts and
# merging them is a behavior change rather than a tidy-up. Windows accepts a
# `bin/` bundle beside a `target/release` checkout; macOS has only ever
# accepted the checkout, and `bl_mac_native_root_resolve` asserts that a
# `bin/`-only root resolves to NOTHING there. A single shared table gave macOS
# the Windows layout and turned that assertion red.
#
# What IS shared is the DEVICE dimension: every table carries the same keys, so
# a device added here reaches every platform. That is the property worth
# holding, and it is asserted directly rather than implied by one table.
#
# The GPU entries are the historical layouts and MUST stay first and unchanged,
# so a root that has only ever held a GPU build resolves exactly as it always
# did.
#
# A ROW IS WHERE TO LOOK, NOT WHAT WAS FOUND. `--features cpu` links the CPU
# backend into `target/release`, the GPU row, whenever that directory is free
# to take it, so the CPU row is where a SECOND build goes rather than where
# every CPU build goes. `_native_server_binary` reads the backend marker to
# settle which device a directory actually answers to; these rows only decide
# the order the directories are visited in.
_WIN_DEVICE_SUBDIRS = {
    DEVICE_GPU: (("target", "release"), ("bin",)),
    DEVICE_CPU: (("target", "cpu", "release"), ("bin-cpu",)),
}
_MAC_DEVICE_SUBDIRS = {
    DEVICE_GPU: (("target", "release"),),
    DEVICE_CPU: (("target", "cpu", "release"),),
}
# LINUX SHIPS ITS DISTRIBUTION IN THE SAME SHAPE IT BUILDS IN, which is why it
# needs no `bin/` row: `build-linux-native/bundle.sh` copies each backend into
# the `target/<backend>/release` directory it was built in and writes the marker
# there itself, so an extracted distribution and a checkout are the same layout.
# `bin/` in that distribution holds the backend library and ffmpeg, never a
# server, so probing it would accept a directory that cannot launch. The
# per-backend GPU directories are reached through `_GPU_BACKEND_SUBDIRS` below,
# as they are on Windows.
_LINUX_DEVICE_SUBDIRS = {
    DEVICE_GPU: (("target", "release"),),
    DEVICE_CPU: (("target", "cpu", "release"),),
}


# Parent levels ``resolve_win_native_root`` climbs before giving up. The
# deepest legitimate selection is ``<root>/target/release`` (two below the
# root); a small bound keeps the walk from wandering far up the filesystem
# when the user picked something unrelated to a solver root.
_WIN_NATIVE_MAX_ASCEND = 6


# The file `crates/ppf-cts-solver/build.rs` writes beside its artifacts naming
# the backend it just linked, and the backend names it can carry.
#
# THIS IS THE ONLY THING THAT ACTUALLY KNOWS WHAT A DIRECTORY HOLDS. Every
# backend links the same executable name, so a path is a convention rather than
# evidence, and `--features cpu` on a tree whose `target/release` is empty puts
# a CPU build exactly where the GPU one would go. Reading the marker is what
# `frontend.solver_dir` does for the same question, and this is that rule
# reaching the addon.
_BACKEND_MARKER = ".ppf-backend"
# The accelerated backend names, as `frontend._GPU_BACKENDS` lists them. A ROCm
# build's marker says "rocm", and a name missing here resolves that GPU build as
# the CPU device.
_GPU_BACKENDS = ("cuda", "metal", "rocm")


class _Disk:
    """The layouts a root holds, asked of THIS machine's filesystem.

    THE RESOLUTION RULE IS THE SAME WHEREVER THE BUILDS ARE, and this is the
    seam that lets it be written once. A native connection answers these three
    questions with `os.path`; an SSH or Docker connection answers them from a
    listing its own solver host produced (`_Reported`). Everything above this
    seam, which is where all the judgment lives, is shared.
    """

    @staticmethod
    def join(root, parts, name):
        return os.path.join(root, *parts, name)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def parent(path):
        return os.path.dirname(path)

    @staticmethod
    def read_marker(directory):
        try:
            with open(os.path.join(directory, _BACKEND_MARKER)) as handle:
                return handle.read().strip()
        except OSError:
            return None


class _Reported:
    """The layouts a root holds, as the SOLVER HOST itself reported them.

    *listing* maps an absolute POSIX directory on that host to the backend its
    marker records there, ``""`` where it carries none. It is produced once per
    connection by `core.remote_builds`, never while drawing: reaching that host
    means a command over the backend, and a panel redraws many times a second.

    A directory absent from the listing holds no server, which is what the
    probe was asked. That is the same answer `_Disk.exists` gives for a path
    that is not there, and it is why the rule above cannot tell the two apart.
    """

    def __init__(self, listing):
        self._listing = dict(listing or {})

    @staticmethod
    def join(root, parts, name):
        return posixpath.join(root, *parts, name)

    def exists(self, path):
        return posixpath.dirname(path) in self._listing

    @staticmethod
    def parent(path):
        return posixpath.dirname(path)

    def read_marker(self, directory):
        return self._listing.get(directory)


_DISK = _Disk()


def _recorded_device(directory, fs=_DISK):
    """Which device the backend marker in *directory* names, or ``None``.

    ``None`` for a directory carrying no marker, which is not an error and is
    the normal state of a DISTRIBUTED bundle: `build-mac-native/bundle.sh`
    copies the three binaries and not the marker, so a downloaded release has
    a solver and nothing saying which backend it is. The caller falls back to
    the layout in that case, which is the answer that shipped before markers
    were consulted at all.
    """
    found = fs.read_marker(directory)
    if not found:
        return None
    return DEVICE_GPU if found in _GPU_BACKENDS else DEVICE_CPU


# WHICH ACCELERATOR, once the device is GPU. A distribution carries one
# directory per backend, so a root can hold a CUDA build and a ROCm build at
# once and something has to choose between them.
#
# THE RULE IS THE FRONTEND'S, RESTATED HERE RATHER THAN IMPORTED.
# `frontend/_backends_.py` decides the same question for a run, and this add-on
# cannot call it: it runs inside Blender's interpreter and reaches the solver
# through the server rather than through the solver's Python package. The two
# therefore have to agree by test, which `addon_host_tests` and
# `frontend/tests/_backend_resolution_.py` do for the same cases.
GPU_BACKEND_AUTO = "AUTO"
# Where each named GPU build lives under the root, beside the historical rows in
# the device tables above.
_GPU_BACKEND_SUBDIRS = {
    "cuda": ("target", "cuda", "release"),
    "rocm": ("target", "rocm", "release"),
    "metal": ("target", "metal", "release"),
}
# The order the automatic choice prefers, CUDA before ROCm.
_GPU_BACKEND_ORDER = ("cuda", "rocm", "metal")
# A GPU build whose directory carries no marker: the distributed bundle that
# shipped before markers, which cannot say which accelerator it holds.
_GPU_UNNAMED = ""
# `ppf-contact-solver --probe`'s exit status for "no usable device here".
_PROBE_UNUSABLE = 3
_PROBE_TIMEOUT_S = 120


class NoUsableGpuBackend(Exception):
    """No GPU build under this root has a usable device on this machine.

    Carries what each backend reported, so the refusal names the reason rather
    than sending the artist back to the folder they correctly chose.

    THIS REFUSES WHERE `frontend` FALLS BACK, AND THE TWO AGREE. That module's
    automatic rule answers with the CPU backend when no GPU here can run, and
    says so; its rule 1 is that an EXPLICIT choice never falls back. Compute
    Device is an explicit choice, made in the panel and saved in the `.blend`,
    so GPU means GPU: an artist who picked it and got a run 30x slower with a
    line in a console they may not have open is the silent substitution the
    device split exists to prevent. The panel's own CPU row is how they take
    the other answer, and it takes one click.
    """

    def __init__(self, reasons):
        self.reasons = dict(reasons)
        detail = "; ".join(f"{name}: {why}" for name, why in self.reasons.items())
        super().__init__(
            f"No GPU build in this folder has a usable device ({detail}). Set "
            f"Compute Device to CPU to run without a GPU."
        )


def _recorded_backend(directory, fs=_DISK):
    """The backend name the marker in *directory* records, or ``None``."""
    return fs.read_marker(directory) or None


def gpu_builds(root, table, exe_name, fs=_DISK):
    """Every GPU build under *root*, as ``{backend name: server path}``.

    A directory answers under the name its marker records. The unmarked
    historical GPU rows answer under ``_GPU_UNNAMED``, since nothing in them
    says which accelerator they hold, and that is the shape a bundle from
    before markers has.

    A fact about the disk, which is all the panel needs: whether a backend has a
    usable device is `probe_gpu_build`'s question, asked when a server is
    spawned and never while drawing.
    """
    found = {}
    if not root:
        return found
    for name in _GPU_BACKEND_ORDER:
        candidate = fs.join(root, _GPU_BACKEND_SUBDIRS[name], exe_name)
        if not fs.exists(candidate):
            continue
        recorded = _recorded_backend(fs.parent(candidate), fs)
        # THE MARKER WINS OVER THE DIRECTORY'S NAME. A `target/cuda` holding a
        # ROCm build is a ROCm build: the marker is the only evidence of what a
        # directory holds, and answering by the name would offer CUDA and run
        # ROCm. An unmarked one is what its name says, which is all that can be
        # known about it.
        if recorded is None:
            found.setdefault(name, candidate)
        elif recorded in _GPU_BACKENDS:
            found.setdefault(recorded, candidate)
    for parts in table.get(DEVICE_GPU, ()):
        candidate = fs.join(root, parts, exe_name)
        if not fs.exists(candidate):
            continue
        recorded = _recorded_backend(fs.parent(candidate), fs)
        if recorded is None:
            found.setdefault(_GPU_UNNAMED, candidate)
        elif recorded in _GPU_BACKENDS:
            found.setdefault(recorded, candidate)
    return found


def probe_gpu_build(server_binary, root):
    """Ask the solver beside *server_binary* whether it can run on this machine.

    Returns ``(usable, detail)``, where *detail* is the device's name or the
    backend's own reason. Anything that is not an answer, including a solver
    built before ``--probe`` existed, is reported as unusable naming what it
    printed, because a build that cannot say what it is cannot be chosen over
    one that can.
    """
    directory = os.path.dirname(server_binary)
    solver = os.path.join(
        directory,
        "ppf-contact-solver.exe" if sys.platform == "win32" else "ppf-contact-solver",
    )
    if not os.path.exists(solver):
        return False, f"no solver beside the server in {directory}"
    env = os.environ.copy()
    if sys.platform == "win32":
        # The solver imports its backend DLL; a distribution keeps those in
        # `bin`, and a checkout in the CUDA build's own library directory.
        env["PATH"] = ";".join(
            [
                directory,
                os.path.join(root, "bin"),
                os.path.join(root, "crates", "ppf-cts-compute", "cuda", "build", "lib"),
                env.get("PATH", ""),
            ]
        )
    try:
        result = subprocess.run(
            [solver, "--probe"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"its solver could not be asked: {error}"
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key not in fields:
            fields[key] = value.strip()
    if result.returncode == 0 and "device" in fields:
        return True, fields["device"]
    if result.returncode == _PROBE_UNUSABLE and "unusable" in fields:
        return False, fields["unusable"]
    printed = (result.stdout + result.stderr).strip() or "nothing"
    return False, f"its solver did not answer --probe (exit {result.returncode}): {printed}"


def _gpu_server_binary(root, table, exe_name, gpu_backend, probe, fs=_DISK):
    """The GPU server under *root* for *gpu_backend*, or ``None``.

    *gpu_backend* is a backend name or ``GPU_BACKEND_AUTO``, and *probe* takes a
    server path and returns ``(usable, detail)``. A named choice resolves to
    that build or to nothing, never to another one. ``AUTO`` takes the only GPU
    build present; where several are, it asks each in `_GPU_BACKEND_ORDER` and
    takes the first with a usable device, raising `NoUsableGpuBackend` when none
    has one. Without a *probe* (the panel, which must not run a solver while
    drawing) it takes the first in that order.
    """
    builds = gpu_builds(root, table, exe_name, fs)
    if gpu_backend and gpu_backend != GPU_BACKEND_AUTO:
        return builds.get(gpu_backend.lower())
    named = [(name, path) for name, path in builds.items() if name != _GPU_UNNAMED]
    if not named:
        return builds.get(_GPU_UNNAMED)
    if len(named) == 1 or probe is None:
        return named[0][1]
    reasons = {}
    for name, path in named:
        usable, detail = probe(path)
        if usable:
            return path
        reasons[name] = detail
    raise NoUsableGpuBackend(reasons)


def _server_subpaths(table, device, exe_name):
    """Candidate `<root>`-relative paths to a server for *device*.

    An unknown device yields NOTHING rather than falling back to the GPU
    layout. A caller that asked for CPU and silently got the GPU server is the
    split this whole mechanism exists to prevent, and an empty result makes the
    caller's own "not found" message the one the user sees.

    EVERY DEVICE'S LAYOUT IS OFFERED, not just the requested one, paired with
    whether the row is the requested device's OWN. The marker decides which
    device a directory answers to, and a row from another device is admissible
    only when a marker SAYS so, which is what the flag lets the caller enforce.
    The requested device's own rows come FIRST, so a tree holding both builds
    in the prescribed layout resolves exactly as it always did without reading
    a single marker.
    """
    if device not in table:
        return
    for parts in table[device]:
        yield (*parts, exe_name), True
    for other, rows in table.items():
        if other == device:
            continue
        for parts in rows:
            yield (*parts, exe_name), False


def _native_server_binary(root, device, table, exe_name, fs=_DISK):
    """Resolve *root*'s server for *device*, or ``None``.

    A candidate directory answers for the device its MARKER names. So a CPU
    build sitting in `target/release`, which is what a plain
    `cargo build --release --features cpu` produces and what `build.rs`'s own
    banner says it produces, resolves as CPU and NOT as GPU.

    THE SUBSTITUTION THIS PREVENTS RUNS BOTH WAYS. Reading the layout alone,
    such a tree offered "GPU" and spawned the CPU solver, roughly 30x slower
    with nothing reporting the swap, while reporting in the same panel that no
    CPU build existed. That is the silent substitution the device split exists
    to prevent, arriving through the build layout rather than through a
    fallback.

    AN UNMARKED DIRECTORY ANSWERS ONLY FOR ITS OWN LAYOUT ROW, never for
    another device's. A marker is the only evidence that a directory holds
    something other than what its path says, so without one there is nothing to
    override the layout with, and admitting an unmarked directory on another
    device's behalf would reinstate the fallback as a bug. That is also what
    keeps a DISTRIBUTED bundle, which carries no marker, resolving exactly as
    it always did.
    """
    if not root:
        return None
    for parts, own_row in _server_subpaths(table, device, exe_name):
        candidate = fs.join(root, parts[:-1], parts[-1])
        if not fs.exists(candidate):
            continue
        recorded = _recorded_device(fs.parent(candidate), fs)
        accepted = own_row if recorded is None else recorded == device
        if accepted:
            return candidate
    return None


def _holds_any_server(root, table, exe_name, fs=_DISK):
    """Whether *root* holds a server for ANY device, in ANY accepted layout.

    "IS THIS A SOLVER ROOT" IS A DIFFERENT QUESTION FROM "DOES IT HOLD THE
    DEVICE I PICKED", and only the root walk asks the first. A tree built with
    `--features cpu` and nothing else is a perfectly good root, and answering
    the root question with a device-specific probe would have the path resolver
    walk past it and report that the folder holds no solver at all, which is
    both wrong and unactionable: the artist would be told to build the thing
    they had just built.

    THE PER-BACKEND DIRECTORIES COUNT, and leaving them out was the same defect
    arriving through a layout rather than through a device. A distribution
    carrying several GPU builds keeps each in `target/<backend>/release`, which
    is where `build.bat`, `build-linux-native/build.sh` and both bundlers put
    them, and none of those rows is in a device table. So a root whose only
    build sat in `target/cuda/release` was not recognized as a root: Connect
    still worked, because it falls back to the raw selection and
    `<platform>_native_server_binary` does know those directories, while the
    panel's own line called the same folder one with no solver in it, and an
    ascent from a subdirectory of it found nothing to ascend to.

    The rows are the shared `_GPU_BACKEND_SUBDIRS`, so this stays one rule for
    every platform. A table's own rows still bound what else is accepted, which
    is what keeps a `bin/`-only root refused on macOS.
    """
    if not root:
        return False
    for rows in table.values():
        for parts in rows:
            if fs.exists(fs.join(root, parts, exe_name)):
                return True
    for parts in _GPU_BACKEND_SUBDIRS.values():
        if fs.exists(fs.join(root, parts, exe_name)):
            return True
    return False


def win_native_server_binary(
    root, device=DEVICE_GPU, gpu_backend=GPU_BACKEND_AUTO, probe=None
):
    """Return the ``ppf-cts-server.exe`` path under *root*, or ``None``.

    Probes the two shipped layouts (``target/release/`` for a repo checkout,
    ``bin/`` for a distributable bundle) plus the per-backend directories a
    distribution carrying several GPU builds has, and asks each directory it
    finds a server in which backend it holds. A directory is a valid Windows
    Native root exactly when this returns a path, so it is the single source of
    truth for "does this directory hold the solver", shared by the spawn path
    and the panel's live validity label.

    *gpu_backend* names the accelerator to run, or ``GPU_BACKEND_AUTO``, and
    *probe* is what asks a build whether its device is usable; see
    `_gpu_server_binary` for the rule and for why the panel passes no probe.
    """
    if device == DEVICE_GPU:
        named = _gpu_server_binary(
            root, _WIN_DEVICE_SUBDIRS, "ppf-cts-server.exe", gpu_backend, probe
        )
        if named is not None or (gpu_backend and gpu_backend != GPU_BACKEND_AUTO):
            # A NAMED CHOICE RESOLVES TO THAT BUILD OR TO NOTHING. Falling
            # through to the device-level layout would answer a request for one
            # accelerator with another, which is the substitution the device
            # split exists to prevent.
            return named
    return _native_server_binary(
        root, device, _WIN_DEVICE_SUBDIRS, "ppf-cts-server.exe"
    )


def win_native_gpu_builds(root):
    """Every GPU build under a Windows Native *root*, as ``{backend: path}``."""
    return gpu_builds(root, _WIN_DEVICE_SUBDIRS, "ppf-cts-server.exe")


def native_device_choice_open(have, selected) -> bool:
    """Whether a local native root's GPU/CPU selector accepts a change.

    *have* maps DEVICE_GPU and DEVICE_CPU to whether the root holds that build,
    and *selected* is the current Compute Device.

    OPEN when the root holds both builds, the one case with a choice to make.
    Also OPEN while the selection names a build the root does not hold. The
    property defaults to GPU, so a folder with only the CPU build (the Windows
    ARM64 distribution, or a tree built only with ``--features cpu``) would
    otherwise lock the selector on a device Connect can only refuse. CLOSED
    when the selection is the one build present, so it cannot be moved onto
    something that is not there.

    It never changes the selection. Which build runs is the artist's choice,
    and Connect refuses a device the root does not hold by name
    (``_absent_device_message``), never by running the other build.
    """
    return all(have.values()) or not have.get(selected, False)


def _absent_device_message(root, device, resolver):
    """The refusal for a root that holds only the OTHER device's build, else None.

    The generic not-found text tells the artist to point Solver Path somewhere
    else, which is wrong for this root: the folder is right and the Compute
    Device is not. Naming the build that is there makes the refusal
    actionable, and it is still a refusal, never a substitution.
    """
    if device not in (DEVICE_GPU, DEVICE_CPU):
        return None
    other = DEVICE_CPU if device == DEVICE_GPU else DEVICE_GPU
    if resolver(root, other) is None:
        return None
    return (
        f"{root} holds the {other} build of the solver and no {device} build. "
        f"Set Compute Device to {other} to run it, or point Solver Path at a "
        f"folder that has a {device} build."
    )


def win_native_not_found_message(root: str, device: str | None = None) -> str:
    """Text for a Windows Native root that holds no ``ppf-cts-server.exe``.

    Says which directory was examined, which two layouts are accepted, and
    what to do next. The reader of this message is usually an artist who
    downloaded the prebuilt Windows bundle and has no Rust toolchain, so
    "build it" cannot be the only instruction: the first thing to check is
    whether the folder they picked is the extracted bundle root rather than
    the folder they extracted it INTO, which is the selection this probe
    cannot resolve on its own (it walks up to a root, never down into one).

    Given *device*, a root that holds the other device's build is refused
    with the text that names it instead (``_absent_device_message``).
    """
    absent = _absent_device_message(root, device, win_native_server_binary)
    if absent is not None:
        return absent
    return (
        f"ppf-cts-server.exe not found under {root}. Point Solver Path at the "
        f"folder that has target\\release\\ppf-cts-server.exe in it (the "
        f"extracted Windows bundle, or a repo checkout you built), not at the "
        f"folder you extracted the bundle into and not at the .zip. If you "
        f"are building from source, run "
        f"`cargo build --release -p ppf-cts-server` first."
    )


def resolve_win_native_root(selected):
    """Resolve the real Windows Native solver root from a user *selected* path.

    The user is meant to pick the bundle / repo root that holds
    ``ppf-cts-server.exe`` (see :func:`win_native_server_binary`), but community
    users frequently pick a *subdirectory* of it (``target/release``, ``bin``,
    or the embedded ``python`` folder) and hit a confusing "ppf-cts-server.exe
    not found" error. Starting at *selected*, walk up parent directories and
    return the first one that is a valid root: a subdirectory resolves to its
    parent, and an already-correct selection returns itself (the selected path
    is tested before any ascent, so a valid root never over-climbs to an outer
    one).

    Returns ``None`` when neither *selected* nor any parent within
    :data:`_WIN_NATIVE_MAX_ASCEND` levels is a valid root, so the caller can
    fall back to the raw selection and surface its own "not found" error
    against the path the user actually chose.
    """
    if not selected or not selected.strip():
        return None
    current = selected.strip().rstrip("/\\")
    # DIR_PATH yields a directory, but a hand-typed path might name the binary
    # itself; ascend from its containing directory in that case.
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(_WIN_NATIVE_MAX_ASCEND + 1):
        if _holds_any_server(current, _WIN_DEVICE_SUBDIRS, "ppf-cts-server.exe"):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break  # reached the filesystem root
        current = parent
    return None


def _absent_gpu_backend_message(root, gpu_backend, builds):
    """The refusal for a root holding GPU builds but not the chosen one, else None.

    The device-level message tells the artist to point Solver Path elsewhere,
    which is wrong for this root: the folder is right and the GPU Backend is
    not. Naming the builds that are here makes the refusal actionable, and it
    stays a refusal, never a substitution.
    """
    if not gpu_backend or gpu_backend == GPU_BACKEND_AUTO:
        return None
    found = [name for name in builds(root) if name]
    if not found:
        return None
    return (
        f"{root} holds these GPU builds: {', '.join(found)}, and no "
        f"{gpu_backend.lower()} build. Set GPU Backend to one of those or to "
        f"Automatic, or point Solver Path at a folder that has a "
        f"{gpu_backend.lower()} build."
    )


def spawn_win_native_server(
    root,
    port,
    cuda_device=AUTOMATIC,
    cuda_device_uuid="",
    device=DEVICE_GPU,
    gpu_backend=GPU_BACKEND_AUTO,
):
    """Spawn a fresh win_native ``ppf-cts-server.exe`` subprocess and return the Popen.

    Used by both initial connect and the Stop/Start cycle on the win_native
    backend. ``connect_win_native`` is the one-shot init path; this helper
    is what ``WinNativeBackend.start_server`` calls to relaunch after a
    user-issued Stop.

    Args:
        root: Project root the server runs from.
        port: TCP port the server listens on.
        cuda_device: Saved GPU display index, or ``gpu_devices.AUTOMATIC``.
        cuda_device_uuid: Stable GPU identity for ``CUDA_VISIBLE_DEVICES``.

    Returns:
        A ``subprocess.Popen`` for the freshly-launched server, ``None``
        when ``PPF_WIN_NATIVE_NO_SPAWN`` is set (test/CI mode), OR ``None``
        when a ppf-cts-server is already alive on *port* (attach mode,
        e.g. Blender restart while the previous session's server lingers).

    Raises:
        FileNotFoundError: if ``ppf-cts-server.exe`` or the embedded Python is missing.
        PortInUseByForeignProcess: if *port* is bound by something that
            isn't a ppf-cts-server we recognize.
    """
    root = root.rstrip("/\\")

    # Test/CI mode: an external orchestrator (e.g. the headless debug
    # rig) already started the server. Bail out before any path probes
    # so we don't fail on a missing binary the rig will provide.
    if os.environ.get("PPF_WIN_NATIVE_NO_SPAWN"):
        return None

    # If a ppf-cts-server is already running on the port (Blender was
    # restarted while the previous session's server kept going), attach
    # to it instead of erroring out. The probe sends a real TCMD ping and
    # checks the JSON response, so a foreign squatter (e.g. some other tool
    # parked on 9090) still surfaces as PortInUseByForeignProcess below.
    #
    # Attaching means no launch happens, so *cuda_device* reaches nothing: the
    # server keeps whatever GPU it was started on. The panel reports that
    # disagreement from the GPU index the server itself reports, which is the
    # only thing that can be right about a server the add-on did not start.
    # *device* is different: a server running the other BUILD is refused
    # rather than reported, because every solve it ran would use that build.
    if _port_is_in_use(port):
        response = _query_ppf_cts_server(port)
        if response is not None:
            check_running_server(
                response,
                root,
                port,
                device,
                # The SELECTED backend's build is what a run would use, so that
                # is the directory the running server is compared against. No
                # probe: attaching runs no solver, and asking one here would
                # make a refusal depend on a device the server may not use.
                lambda where, which: win_native_server_binary(
                    where, which, gpu_backend
                ),
            )
            return None
        raise PortInUseByForeignProcess(port)

    # Resolve the Rust ``ppf-cts-server.exe`` binary. We look in the dev
    # layout first, then the bundled ``bin/`` so a Windows native bundle
    # can ship the binary alongside the embedded Python. Only required
    # when we actually need to spawn; the attach path above already
    # returned. Resolved before the embedded-Python branch so an
    # all-missing layout surfaces the server.exe error first.
    #
    # THE PROBE IS PASSED HERE AND NOWHERE ELSE IN THIS MODULE: a spawn is the
    # one moment a solver is about to run, so it is the moment to ask which
    # accelerator can run it. `NoUsableGpuBackend` carries what each backend
    # reported and reaches the artist unchanged.
    rust_bin = win_native_server_binary(
        root, device, gpu_backend, lambda path: probe_gpu_build(path, root)
    )
    if rust_bin is None:
        raise FileNotFoundError(
            _absent_gpu_backend_message(root, gpu_backend, win_native_gpu_builds)
            or win_native_not_found_message(root, device)
        )

    build_dir = os.path.join(root, "build-win-native")
    if os.path.exists(os.path.join(build_dir, "python", "python.exe")):
        extra_paths = [
            os.path.join(build_dir, "python"),
            os.path.join(root, "target", "release"),
            # The CUDA backend library, which the solver IMPORTS. build.bat's
            # LIB_DIR puts it here and nowhere else in a checkout, the same
            # directory the session launcher's LIB_PATH_DEV names; a spelling
            # that names another directory adds a dead entry to the DLL
            # search path rather than failing.
            os.path.join(root, "crates", "ppf-cts-compute", "cuda", "build", "lib"),
            os.path.join(build_dir, "cuda", "bin"),
        ]
        cuda_path = os.path.join(build_dir, "cuda")
    elif os.path.exists(os.path.join(root, "python", "python.exe")):
        extra_paths = [
            os.path.join(root, "python"),
            os.path.join(root, "bin"),
            os.path.join(root, "target", "release"),
        ]
        cuda_path = None
    else:
        raise FileNotFoundError(
            f"Embedded Python not found in {build_dir} or {root}"
        )

    env = os.environ.copy()
    # The resolved build's own directory goes FIRST, ahead of the layout
    # guesses above: those name `target\release` unconditionally, so a CPU run
    # would otherwise resolve a DLL out of the GPU build.
    env["PATH"] = ";".join(
        [os.path.dirname(rust_bin)] + extra_paths + [env.get("PATH", "")]
    )
    env["PYTHONPATH"] = root + ";" + env.get("PYTHONPATH", "")
    _apply_target_dir(env, rust_bin)
    if cuda_path and os.path.exists(cuda_path):
        env["CUDA_PATH"] = cuda_path
    # The solver never calls cudaSetDevice, so it runs on device 0 of whatever
    # CUDA can see. Restricting the visible set here is what makes the panel's
    # GPU choice reach it, and the server's own hardware probe reads the same
    # variable, so what the Remote Hardware block reports is the device the
    # solver is on.
    apply_selection(env, cuda_device, cuda_device_uuid)

    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    # Redirect to a real file, NOT subprocess.PIPE. With PIPE the addon
    # owns the read end and never drains it; on Windows the OS pipe
    # buffer is only a few KB, and once the server's log4rs console
    # appender fills it, every subsequent write blocks the tokio worker
    # thread that emitted it. After enough activity (the 21 rapid polls
    # _do_terminate fires are usually what tips it over) every worker
    # ends up blocked in a write syscall, the runtime stops scheduling
    # new tasks, and the server appears wedged: connections accept but
    # never get a response, CPU is 0, all threads in Wait state.
    log_path = os.path.join(root, "server.log")
    log_fp = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [rust_bin, "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    finally:
        log_fp.close()


def connect_win_native(
    root, port, device=DEVICE_GPU, project_name="", gpu_backend=GPU_BACKEND_AUTO
):
    """Connect using Windows native build.

    The *root* path must be the project root directory where
    ``ppf-cts-server.exe`` is located.

    Connecting does not start the server. Start Server does, on this backend
    as on every other, which is what lets a GPU be picked from the panel in
    between: the choice has to be made against a device list, and that list is
    read over the connection.

    Args:
        root: Project root directory (where ppf-cts-server.exe lives).
        port: Port for the solver server.

    Returns:
        A tuple of (ConnectionInfo, None). The second element is the server
        subprocess, which this step does not create.

    Raises:
        FileNotFoundError: if no ``ppf-cts-server.exe`` is under *root* (or
            under an ancestor within the resolver's reach), unless
            ``PPF_WIN_NATIVE_NO_SPAWN`` says an external orchestrator owns
            the server.
    """
    # Community users often point the addon at a subdirectory of the real
    # solver root (target/release, bin, or the embedded python/ folder). Walk
    # up to the actual bundle / repo root so those selections just work; fall
    # back to the raw path when nothing qualifies so the launch's
    # FileNotFoundError still names the path the user chose. Resolving here
    # keeps current_directory / remote_root pointed at the real root, which the
    # backend uses as the transfer cwd and PYTHONPATH base and which
    # start_server spawns from.
    root = resolve_win_native_root(root) or root.rstrip("/\\")
    root = root.rstrip("/\\")

    # Refuse a root that holds no solver, rather than reporting a connection
    # and leaving the failure to Start Server. The panel draws
    # "ppf-cts-server.exe not found" for this directory at the moment it is
    # set, so a connection reported against it puts that line and "Connected"
    # on screen together, and only a second button press settles which one is
    # true. Refusing here keeps the two in agreement.
    #
    # PPF_WIN_NATIVE_NO_SPAWN is the test/CI mode where an external
    # orchestrator owns the server, which is exactly the case where the
    # binary need not be under this root; the spawn path skips its probes for
    # the same reason.
    if not os.environ.get("PPF_WIN_NATIVE_NO_SPAWN"):
        # NO PROBE WHILE CONNECTING. Whether the chosen accelerator has a usable
        # device is asked when a server is spawned; here the question is only
        # whether this root holds the build the panel names, so connecting to a
        # machine whose GPU is busy or absent still reports the root honestly
        # and the refusal arrives from the spawn, naming the device.
        if win_native_server_binary(root, device, gpu_backend) is None:
            raise FileNotFoundError(
                _absent_gpu_backend_message(root, gpu_backend, win_native_gpu_builds)
                or win_native_not_found_message(root, device)
            )

    # A server already answering on the port is what this connection will
    # drive, since Start Server attaches rather than launching, so it has to
    # be the build the Compute Device names. Checked in the test mode as well:
    # an orchestrator-owned server running another build is the same silent
    # substitution. Asked under the add-on's own project name, so the query
    # is the ping the first status poll would send anyway.
    check_running_server(
        _query_ppf_cts_server(port, name=project_name),
        root, port, device,
        lambda where, which: win_native_server_binary(where, which, gpu_backend),
    )

    connection_info = ConnectionInfo()
    connection_info.type = "win_native"
    connection_info.current_directory = root
    connection_info.remote_root = root
    connection_info.instance = "win_native"
    connection_info.server_running = False
    connection_info.container = ""
    connection_info.server_port = port
    return connection_info, None


# Parent levels ``resolve_mac_native_root`` climbs before giving up. The
# deepest legitimate selection is ``<root>/target/release`` (two below the
# root); a small bound keeps the walk from wandering far up the filesystem
# when the user picked something unrelated to a solver root.
_MAC_NATIVE_MAX_ASCEND = 6


def mac_native_server_binary(
    root, device=DEVICE_GPU, gpu_backend=GPU_BACKEND_AUTO, probe=None
):
    """Return the ``ppf-cts-server`` path under *root*, or ``None``.

    Probes the one shipped layout, ``target/release/``, and asks each directory
    it finds a server in which backend it holds. A directory is a valid macOS
    Native root exactly when this returns a path, so it is the single source of
    truth for "does this directory hold the solver", shared by the spawn path
    and the panel's live validity label.

    IT TAKES THE SAME PARAMETERS AS ITS WINDOWS TWIN although macOS has one
    accelerator: the callers are shared, and a signature that differed by
    platform would make every one of them ask which platform it is on.
    """
    if device == DEVICE_GPU:
        named = _gpu_server_binary(
            root, _MAC_DEVICE_SUBDIRS, "ppf-cts-server", gpu_backend, probe
        )
        if named is not None or (gpu_backend and gpu_backend != GPU_BACKEND_AUTO):
            return named
    return _native_server_binary(root, device, _MAC_DEVICE_SUBDIRS, "ppf-cts-server")


def mac_native_gpu_builds(root):
    """Every GPU build under a macOS Native *root*, as ``{backend: path}``."""
    return gpu_builds(root, _MAC_DEVICE_SUBDIRS, "ppf-cts-server")


def native_gpu_backend_choice_open(builds, selected) -> bool:
    """Whether a local native root's GPU Backend selector accepts a change.

    *builds* is `gpu_builds`'s answer and *selected* the saved choice.

    OPEN when the root holds more than one named GPU build, the one case with a
    choice to make, and OPEN while the selection names a backend the root does
    not hold, so a `.blend` carrying ROCm cannot lock the selector on a folder
    that has only CUDA. CLOSED otherwise, so it cannot be moved onto something
    that is not there. It never changes the selection, exactly as
    `native_device_choice_open` does not.
    """
    named = [name for name in builds if name]
    if len(named) > 1:
        return True
    return selected != GPU_BACKEND_AUTO and selected.lower() not in named


def mac_native_not_found_message(root: str, device: str | None = None) -> str:
    """Text for a macOS Native root that holds no ``ppf-cts-server``.

    Says which directory was examined, which layout is accepted, and what to
    do next. The reader of this message is usually an artist who downloaded
    the prebuilt macOS distribution and has no Rust toolchain, so "build it"
    cannot be the only instruction: the first thing to check is whether the
    folder they picked is the extracted distribution root rather than the
    folder they extracted it INTO, which is the selection this probe cannot
    resolve on its own (it walks up to a root, never down into one).

    Given *device*, a root that holds the other device's build is refused
    with the text that names it instead (``_absent_device_message``).
    """
    absent = _absent_device_message(root, device, mac_native_server_binary)
    if absent is not None:
        return absent
    return (
        f"ppf-cts-server not found under {root}. Point Solver Path at the "
        f"folder that has target/release/ppf-cts-server in it (the extracted "
        f"macOS distribution, or a repo checkout you built), not at the folder "
        f"you extracted it into. If you are building from source, run "
        f"`cargo build --release -p ppf-cts-server` first."
    )


def resolve_mac_native_root(selected):
    """Resolve the real macOS Native solver root from a user *selected* path.

    The user is meant to pick the bundle / repo root that holds
    ``ppf-cts-server`` (see :func:`mac_native_server_binary`), but a
    subdirectory of it (``target``, ``target/release``, or the bundle's
    ``bin``) is an easy selection to make and reports only that no solver was
    found. Starting at *selected*, walk up parent directories and return the
    first one that is a valid root: a subdirectory resolves to its parent, and
    an already-correct selection returns itself (the selected path is tested
    before any ascent, so a valid root never over-climbs to an outer one).

    Returns ``None`` when neither *selected* nor any parent within
    :data:`_MAC_NATIVE_MAX_ASCEND` levels is a valid root, so the caller can
    fall back to the raw selection and surface its own "not found" error
    against the path the user actually chose.
    """
    if not selected or not selected.strip():
        return None
    current = selected.strip().rstrip("/")
    # DIR_PATH yields a directory, but a hand-typed path might name the binary
    # itself; ascend from its containing directory in that case.
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(_MAC_NATIVE_MAX_ASCEND + 1):
        if _holds_any_server(current, _MAC_DEVICE_SUBDIRS, "ppf-cts-server"):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break  # reached the filesystem root
        current = parent
    return None


# Parent levels ``resolve_linux_native_root`` climbs before giving up. The
# deepest legitimate selection is ``<root>/target/cuda/release``, three below
# the root, since a Linux distribution keeps every backend in its own target
# directory; a small bound keeps the walk from wandering far up the filesystem
# when the user picked something unrelated to a solver root.
_LINUX_NATIVE_MAX_ASCEND = 6


def linux_native_server_binary(
    root, device=DEVICE_GPU, gpu_backend=GPU_BACKEND_AUTO, probe=None
):
    """Return the ``ppf-cts-server`` path under *root*, or ``None``.

    Probes the checkout layout ``target/release/`` and the per-backend
    directories a distribution carrying several GPU builds has, and asks each
    directory it finds a server in which backend it holds. A directory is a
    valid Linux Native root exactly when this returns a path, so it is the
    single source of truth for "does this directory hold the solver", shared by
    the spawn path and the panel's live validity label.

    A LINUX DISTRIBUTION IS THE SAME SHAPE AS THE TREE IT WAS BUILT FROM, which
    is why there is no bundle row to add: `build-linux-native/bundle.sh` copies
    each backend into the `target/<backend>/release` directory it was built in.
    Its `bin/` holds the backend library and ffmpeg and never a server.

    *gpu_backend* names the accelerator to run, or ``GPU_BACKEND_AUTO``, and
    *probe* is what asks a build whether its device is usable; see
    `_gpu_server_binary` for the rule and for why the panel passes no probe.
    """
    if device == DEVICE_GPU:
        named = _gpu_server_binary(
            root, _LINUX_DEVICE_SUBDIRS, "ppf-cts-server", gpu_backend, probe
        )
        if named is not None or (gpu_backend and gpu_backend != GPU_BACKEND_AUTO):
            # A NAMED CHOICE RESOLVES TO THAT BUILD OR TO NOTHING, as on every
            # other native: falling through to the device-level layout would
            # answer a request for one accelerator with another.
            return named
    return _native_server_binary(
        root, device, _LINUX_DEVICE_SUBDIRS, "ppf-cts-server"
    )


def linux_native_gpu_builds(root):
    """Every GPU build under a Linux Native *root*, as ``{backend: path}``."""
    return gpu_builds(root, _LINUX_DEVICE_SUBDIRS, "ppf-cts-server")


def linux_native_not_found_message(root: str, device: str | None = None) -> str:
    """Text for a Linux Native root that holds no ``ppf-cts-server``.

    Says which directory was examined, which layouts are accepted, and what to
    do next. The reader of this message is usually an artist who downloaded the
    prebuilt Linux distribution and has no Rust toolchain, so "build it" cannot
    be the only instruction: the first thing to check is whether the folder they
    picked is the extracted distribution root rather than the folder they
    extracted it INTO, which is the selection this probe cannot resolve on its
    own (it walks up to a root, never down into one).

    Given *device*, a root that holds the other device's build is refused with
    the text that names it instead (``_absent_device_message``).
    """
    absent = _absent_device_message(root, device, linux_native_server_binary)
    if absent is not None:
        return absent
    return (
        f"ppf-cts-server not found under {root}. Point Solver Path at the "
        f"folder that has target/release/ppf-cts-server in it, or "
        f"target/cuda/release, target/rocm/release or target/cpu/release (the "
        f"extracted Linux distribution, or a repo checkout you built), not at "
        f"the folder you extracted it into. If you are building from source, "
        f"run `cargo build --release -p ppf-cts-server` first."
    )


def resolve_linux_native_root(selected):
    """Resolve the real Linux Native solver root from a user *selected* path.

    The user is meant to pick the distribution / repo root that holds the
    server (see :func:`linux_native_server_binary`), but a subdirectory of it
    (``target``, ``target/release``, ``target/cuda/release``, or the
    distribution's ``bin``) is an easy selection to make and reports only that
    no solver was found. Starting at *selected*, walk up parent directories and
    return the first one that is a valid root: a subdirectory resolves to its
    parent, and an already-correct selection returns itself (the selected path
    is tested before any ascent, so a valid root never over-climbs to an outer
    one).

    Returns ``None`` when neither *selected* nor any parent within
    :data:`_LINUX_NATIVE_MAX_ASCEND` levels is a valid root, so the caller can
    fall back to the raw selection and surface its own "not found" error
    against the path the user actually chose.
    """
    if not selected or not selected.strip():
        return None
    current = selected.strip().rstrip("/")
    # DIR_PATH yields a directory, but a hand-typed path might name the binary
    # itself; ascend from its containing directory in that case.
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(_LINUX_NATIVE_MAX_ASCEND + 1):
        if _holds_any_server(current, _LINUX_DEVICE_SUBDIRS, "ppf-cts-server"):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break  # reached the filesystem root
        current = parent
    return None


def native_target_dir(server_binary):
    """The cargo target directory that owns *server_binary*, or ``None``.

    WHY THE SPAWN HAS TO SAY THIS AT ALL. Choosing a device chooses a build
    DIRECTORY, and the server binary is only one of the three things a run
    takes out of it. The build worker's Python loads the cdylib from whichever
    target directory `frontend._target_dirs` finds first, and
    `frontend.artifact_dir` then writes THAT directory into the session's
    `command.sh` as `SOLVER_PATH`. So without naming the directory here, the
    add-on spawns the CPU server and the solve it drives runs the GPU solver
    out of `target/release`: the two halves of one run come from different
    builds, and nothing anywhere reports the split, because each binary answers
    `--backend` honestly about itself and neither is asked about the other.

    ``None`` for a layout that is not a cargo target directory, which is the
    Windows bundle's ``bin/``. The caller drops any inherited value in that
    case rather than guessing one: a `CARGO_TARGET_DIR` exported in the user's
    own shell is the ONLY directory the frontend searches, so it would send the
    frontend looking for the cdylib somewhere unrelated to the folder the
    artist selected. The distribution's own launcher unsets it for that reason.
    """
    if not server_binary:
        return None
    profile_dir = os.path.dirname(server_binary)
    # Only a `<target>/<profile>` shape names a target directory. The profile
    # is `release` everywhere here: nothing ships or spawns a debug build.
    if os.path.basename(profile_dir) != "release":
        return None
    return os.path.dirname(profile_dir)


def _apply_target_dir(env, server_binary):
    """Name the build directory *server_binary* came out of, in *env*.

    Both natives call it, because the split it prevents is not
    platform-specific: it is a property of the frontend choosing the cdylib at
    import and the session script naming the solver separately.
    """
    target_dir = native_target_dir(server_binary)
    if target_dir:
        env["CARGO_TARGET_DIR"] = target_dir
    else:
        env.pop("CARGO_TARGET_DIR", None)


class NativeServerMismatch(Exception):
    """A server already on the port runs a different build from the one selected."""


def _same_directory(a, b):
    """Whether *a* and *b* name one directory, however each is spelled.

    ``samefile`` compares the directories themselves, so a symlink, a Windows
    junction, a trailing separator or a case difference on a case-insensitive
    volume all compare equal, and a path that does not exist compares unequal
    rather than raising.
    """
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _expected_target_dir(root, server_binary):
    """The target directory a server spawned from *server_binary* runs from.

    The spawn names it in ``CARGO_TARGET_DIR`` when the binary sits in a cargo
    target layout, and removes the variable otherwise, which leaves the
    frontend on ``<root>/target`` (see ``_apply_target_dir``). So this is what a
    server the add-on started for this selection reports as its
    ``solver_target_dir``.
    """
    return native_target_dir(server_binary) or os.path.join(root, "target")


def check_running_server(response, root, port, device, resolver):
    """Refuse a server already answering on *port* that runs another build.

    THE ATTACH PATH IS WHERE A DEVICE SELECTION WAS SILENTLY LOST. Disconnect
    leaves a native server running on purpose (``WinNativeBackend.disconnect``
    says why), and Connect and Start Server then attach to whatever answers on
    the port. So connecting with CPU selected, to a port still held by the GPU
    server an earlier connection started, ran every solve on the GPU build
    while the panel said CPU, and the reverse, with nothing reporting it.

    WHAT IS COMPARED IS THE TARGET DIRECTORY, NOT A BACKEND NAME. The server
    reports the directory its runs take the solver from, which is the thing
    that decides which solver a run executes; a backend name would also pass a
    server from a different tree built for the same device.

    Three cases are not refused. *response* ``None`` means nothing of ours
    answers, so there is nothing to compare. A server on another protocol
    version is left to the handshake, which stops it on its first response. A
    selection with no build under *root* is left to the caller: outside the
    test mode it has already refused that root, and inside it the orchestrator
    owns the server and the root need not hold one.

    Raises:
        NativeServerMismatch: naming both directories and how to get out.
    """
    if response is None:
        return
    if str(response.get("protocol_version")) != PROTOCOL_VERSION:
        return
    expected_bin = resolver(root, device)
    if expected_bin is None:
        return
    expected = _expected_target_dir(root, expected_bin)
    found = str(response.get("solver_target_dir") or "")
    if found and _same_directory(found, expected):
        return
    found_backend = str(response.get("solver_backend") or "")
    if not found:
        what = "a build it does not report"
    elif found_backend:
        what = f"the build in {found} (its solver reports {found_backend})"
    else:
        what = f"the build in {found}"
    # Name the way out. Force Terminate Process ends the running server from this
    # refused state without connecting; when the running server is this
    # root's build for the OTHER device, selecting that device connects to
    # it and Stop Server is reachable as well, which is named too.
    other = next(
        (
            candidate
            for candidate in (DEVICE_GPU, DEVICE_CPU)
            if candidate != device
            and found
            and resolver(root, candidate) is not None
            and _same_directory(
                found, _expected_target_dir(root, resolver(root, candidate))
            )
        ),
        None,
    )
    if other is not None:
        remedy = (
            f"To switch, press Force Terminate Process and Connect again, or set Compute "
            f"Device to {other}, Connect, click Stop Server, then set it back "
            f"to {device} and Connect again."
        )
    else:
        remedy = (
            f"Press Force Terminate Process to end the ppf-cts-server process on port "
            f"{port}, or connect on a different port."
        )
    raise NativeServerMismatch(
        f"A solver server is already running on port {port}, and its runs "
        f"use {what}. Compute Device is set to {device}, whose build is in "
        f"{expected}. {remedy}"
    )


# The developer environment's interpreter, by the convention every
# provisioning path here uses (`warmup.py`'s ``get_venv_path``, which
# ``build-mac-native/warmup.sh`` reads rather than restating).
_DEV_VENV_SUBPATH = (".local", "share", "ppf-cts", "venv", "bin", "python")

# The marker the bundlers write at the root of a packaged distribution, and the
# one ``datamodel::app::is_selfcontained`` reads. It is what separates a
# downloaded release from a developer checkout here.
_SELFCONTAINED_MARKER = ".ppf-selfcontained"

# The extended attribute a browser sets on what it downloads, and that an
# unarchiver carries onto every entry it extracts.
_QUARANTINE_ATTR = "com.apple.quarantine"


def native_build_python(root):
    """The interpreter the build worker should run under for *root*, or ``None``.

    THE INTERPRETER BELONGS TO THE ROOT, NOT TO THE MACHINE. A distribution
    ships its own relocatable CPython at ``<root>/python/bin/python3`` with the
    full frontend dependency set installed INTO it, which is the interpreter
    its own launcher names in ``PPF_CTS_BUILD_PYTHON`` and the one its
    JupyterLab runs on. A developer checkout ships none and uses the venv under
    ``~/.local/share/ppf-cts``. The two are disjoint: a distribution has no venv
    relationship and a checkout has no ``<root>/python``, so asking the ROOT
    which interpreter is its own answers for both without a mode flag.

    ONE ANSWER FOR THE POSIX NATIVES. ``build-mac-native/bundle.sh`` and
    ``build-linux-native/bundle.sh`` put the interpreter in the same place, so
    the question is the root's rather than the platform's and both spawns ask
    it here.

    WHAT THIS PREVENTS. Naming the developer venv unconditionally hands a
    DISTRIBUTION's build worker an interpreter belonging to a checkout
    somewhere else on the machine, or one provisioned long ago, while the
    distribution's own fully-provisioned interpreter sits unused inside the
    folder the artist selected. It presents as a missing frontend dependency
    (``No module named 'pytetwild'``) from a distribution that ships pytetwild
    and runs it fine under its own JupyterLab, which reads as the distribution
    being broken rather than as the add-on naming the wrong Python.
    """
    if root:
        shipped = os.path.join(root, "python", "bin", "python3")
        if os.path.isfile(shipped):
            return shipped
    dev_venv = os.path.join(os.path.expanduser("~"), *_DEV_VENV_SUBPATH)
    if os.path.isfile(dev_venv):
        return dev_venv
    return None


def clear_mac_native_quarantine(root):
    """Clear ``com.apple.quarantine`` from a downloaded distribution at *root*.

    Returns ``(cleared, remaining)``, counted rather than assumed: the walk is
    repeated after the attempt, so a folder that could not be written reports
    what is still marked instead of a success it did not have. Returns ``None``
    when the question does not apply to this root or could not be asked.

    WHY THE ADD-ON DOES THIS AND NOT ONLY THE LAUNCHER. A distribution's
    launcher clears the mark on its own folder at startup, which covers an
    artist who opens JupyterLab. This backend never runs that launcher: it
    spawns ``ppf-cts-server`` out of the same folder directly. So an artist who
    downloads a release and drives it only from Blender would meet Gatekeeper
    at the spawn, with the server dying before it binds a port and the panel
    reporting a connection that never came up.

    ONLY FOR A PACKAGED DISTRIBUTION, decided by the ``.ppf-selfcontained``
    marker the bundlers write. A developer checkout is never downloaded as a
    unit and so is never marked, and it carries a ``target/`` directory whose
    walk would cost far more than the answer is worth. The marker is the same
    one ``datamodel::app::is_selfcontained`` reads to decide where a tree roots
    its data, so this asks a question the tree already answers.

    THE WHOLE FOLDER, NOT THE SERVER BINARY. Every Mach-O is assessed when it
    is loaded, not only when it is started, so clearing the one file about to
    be spawned would get the server running and fail inside it, at the backend
    dylib or at the build worker's first ``import``.

    ``-s`` KEEPS THE WALK INSIDE THE FOLDER. Without it ``xattr`` follows a
    symbolic link, leaving the link's own mark in place and reaching whatever
    it points at, which may be outside the distribution entirely; and a link
    pointing nowhere makes it exit non-zero. With ``-s`` the walk is silent and
    touches nothing the artist did not download.

    IT NEVER RAISES, AND IT NEVER CLAIMS MORE THAN IT DID. A folder that cannot
    be cleared, which is what another user's folder or a read-only volume looks
    like, is a folder the spawn will fail on, and that failure carries the
    better message: it names the binary and the root. Refusing to connect here
    because a subprocess misbehaved would replace a specific error with a vague
    one. What this must not do is report a clearing that did not happen, which
    is why the count comes from a second walk rather than from the first.
    """
    if sys.platform != "darwin" or not root:
        return None
    if not os.path.isfile(os.path.join(root, _SELFCONTAINED_MARKER)):
        return None
    before = _count_quarantined(root)
    if before is None:
        return None
    if before == 0:
        return (0, 0)
    try:
        subprocess.run(
            ["/usr/bin/xattr", "-s", "-d", "-r", _QUARANTINE_ATTR, root],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    after = _count_quarantined(root)
    if after is None:
        return None
    return (before - after, after)


def _count_quarantined(root):
    """How many entries under *root* carry the mark, or ``None`` if unanswerable.

    ``find -xattrname`` is an Apple extension and the only thing that answers
    this without walking the tree in Python. A host whose ``find`` does not
    carry it returns a non-zero status, which is reported as "unanswerable"
    rather than as zero: reading it as zero would turn a check that could not
    run into a clean verdict, which is the one wrong answer available here.
    """
    try:
        found = subprocess.run(
            ["/usr/bin/find", root, "-xattrname", _QUARANTINE_ATTR, "-print"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if found.returncode != 0:
        return None
    listing = found.stdout.decode("utf-8", "replace")
    return len([line for line in listing.splitlines() if line])


def spawn_mac_native_server(root, port, device=DEVICE_GPU):
    """Spawn a fresh mac_native ``ppf-cts-server`` subprocess and return the Popen.

    Used by both initial connect and the Stop/Start cycle on the mac_native
    backend. ``connect_mac_native`` is the one-shot init path; this helper is
    what ``MacNativeBackend.start_server`` calls to relaunch after a
    user-issued Stop.

    No device selection accompanies the launch. The Metal backend opens the
    system default device and offers no way to name another, so there is
    nothing for a device argument to carry.

    Args:
        root: Project root the server runs from.
        port: TCP port the server listens on.

    Returns:
        A ``subprocess.Popen`` for the freshly-launched server, ``None`` when
        ``PPF_MAC_NATIVE_NO_SPAWN`` is set (test/CI mode), OR ``None`` when a
        ppf-cts-server is already alive on *port* (attach mode, e.g. Blender
        restart while the previous session's server lingers).

    Raises:
        FileNotFoundError: if ``ppf-cts-server`` is missing under *root*.
        PortInUseByForeignProcess: if *port* is bound by something that isn't
            a ppf-cts-server we recognize.
    """
    root = root.rstrip("/")

    # Test/CI mode: an external orchestrator (e.g. the headless debug rig)
    # already started the server. Bail out before any path probes so we don't
    # fail on a missing binary the rig will provide.
    if os.environ.get("PPF_MAC_NATIVE_NO_SPAWN"):
        return None

    # If a ppf-cts-server is already running on the port (Blender was
    # restarted while the previous session's server kept going), attach to it
    # instead of erroring out. The probe sends a real TCMD ping and checks the
    # JSON response, so a foreign squatter still surfaces as
    # PortInUseByForeignProcess below.
    #
    # Attaching means no launch happens, so this call reaches nothing and the
    # running server keeps serving. The panel reports what that server says
    # about itself, which is the only thing that can be right about a server
    # the add-on did not start, except for its BUILD, which is refused when it
    # is not the one *device* names.
    if _port_is_in_use(port):
        response = _query_ppf_cts_server(port)
        if response is not None:
            check_running_server(
                response, root, port, device, mac_native_server_binary
            )
            return None
        raise PortInUseByForeignProcess(port)

    rust_bin = mac_native_server_binary(root, device)
    if rust_bin is None:
        raise FileNotFoundError(mac_native_not_found_message(root, device))

    # THE DOWNLOAD MARK, CLEARED BEFORE THE SPAWN AND NOT AFTER IT. Gatekeeper
    # assesses a Mach-O when it is loaded, so a distribution the artist
    # downloaded through a browser would kill this server on exec, or inside
    # it at the backend dylib. Blender is not the launcher, so nothing else on
    # this path would have cleared it. It answers None for a developer
    # checkout, which is never marked, and never raises: see
    # clear_mac_native_quarantine.
    outcome = clear_mac_native_quarantine(root)
    if outcome is not None:
        cleared, remaining = outcome
        if cleared:
            print(
                f"Cleared the macOS download mark from {cleared} entries "
                f"under {root}"
            )
        if remaining:
            print(
                f"WARNING: {remaining} entries under {root} are still marked "
                "com.apple.quarantine and macOS may refuse to load them. That "
                "is a folder owned by another user or on a read-only volume. "
                "Move it somewhere you own, or clear it with: "
                f'xattr -s -d -r com.apple.quarantine "{root}"'
            )

    # No library search path is set here. The solver binary carries an
    # LC_RPATH and the Metal backend dylib an @rpath install name, so each
    # resolves the other on its own, and ppf-cts-server links neither.
    #
    # What the child does need is an interpreter for the build worker
    # ppf-cts-server spawns. The server resolves that as PPF_CTS_BUILD_PYTHON,
    # then $VIRTUAL_ENV/bin/python, then ``python3`` on PATH; macOS ships 3.9
    # as its system python3, which cannot parse the frontend at all, so leaving
    # it to that last step is not an option.
    #
    # NAMED EXPLICITLY RATHER THAN LEFT TO $VIRTUAL_ENV, because only the first
    # step can express "the interpreter that belongs to THIS root", which is
    # what a distribution needs: its own is inside the folder the artist
    # selected. `native_build_python` answers that for both layouts, and
    # this mirrors what the distribution's own `start.sh` exports.
    #
    # AN INHERITED PPF_CTS_BUILD_PYTHON WINS. Someone who set it meant it, and
    # this is also how the add-on is launched from inside a distribution's own
    # environment without the two disagreeing.
    #
    # $VIRTUAL_ENV is still set when the answer IS the developer venv, so a
    # subprocess further down that reads it rather than the explicit variable
    # sees the same environment it always did. Nothing is set at all when no
    # interpreter can be found, and the worker's own resolution then reports
    # what it could not find.
    env = os.environ.copy()
    build_python = native_build_python(root)
    if build_python and not env.get("PPF_CTS_BUILD_PYTHON"):
        env["PPF_CTS_BUILD_PYTHON"] = build_python
        bin_dir = os.path.dirname(build_python)
        env["PATH"] = bin_dir + ":" + env.get("PATH", "")
        # ONE INTERPRETER, NAMED ONCE. $VIRTUAL_ENV is set when the answer IS a
        # venv, so a subprocess reading that instead of the explicit variable
        # sees the same environment; and it is CLEARED when the answer is not,
        # because Blender is commonly started from a shell with some venv
        # active and an inherited one would otherwise sit in the environment
        # contradicting the interpreter actually chosen. A distribution's
        # relocatable interpreter is never a venv: a venv records absolute
        # paths in pyvenv.cfg and would not survive being copied, which is why
        # the distribution installs into the tree instead. So that file is the
        # thing to ask, not the path.
        prefix = os.path.dirname(bin_dir)
        if os.path.isfile(os.path.join(prefix, "pyvenv.cfg")):
            env["VIRTUAL_ENV"] = prefix
        else:
            env.pop("VIRTUAL_ENV", None)
    env["PYTHONPATH"] = root + ":" + env.get("PYTHONPATH", "")
    _apply_target_dir(env, rust_bin)

    # Redirect to a real file, NOT subprocess.PIPE. With PIPE the addon owns
    # the read end and never drains it; once the OS pipe buffer fills, every
    # subsequent write blocks the tokio worker thread that emitted it. After
    # enough activity every worker ends up blocked in a write syscall, the
    # runtime stops scheduling new tasks, and the server appears wedged:
    # connections accept but never get a response.
    log_path = os.path.join(root, "server.log")
    log_fp = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [rust_bin, "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_fp.close()


def connect_mac_native(root, port, device=DEVICE_GPU, project_name=""):
    """Connect using the macOS native build.

    The *root* path must be the project root directory where
    ``ppf-cts-server`` is located.

    Connecting does not start the server. Start Server does, on this backend
    as on every other.

    Args:
        root: Project root directory (where ppf-cts-server lives).
        port: Port for the solver server.

    Returns:
        A tuple of (ConnectionInfo, None). The second element is the server
        subprocess, which this step does not create.

    Raises:
        FileNotFoundError: if no ``ppf-cts-server`` is under *root* (or under
            an ancestor within the resolver's reach), unless
            ``PPF_MAC_NATIVE_NO_SPAWN`` says an external orchestrator owns the
            server.
    """
    # A subdirectory of the real solver root (target/release, target, or the
    # bundle's bin) is an easy selection to make. Walk up to the actual bundle
    # / repo root so those selections just work; fall back to the raw path
    # when nothing qualifies so the launch's FileNotFoundError still names the
    # path the user chose. Resolving here keeps current_directory /
    # remote_root pointed at the real root, which the backend uses as the
    # transfer cwd and PYTHONPATH base and which start_server spawns from.
    root = resolve_mac_native_root(root) or root.rstrip("/")
    root = root.rstrip("/")

    # Refuse a root that holds no solver, rather than reporting a connection
    # and leaving the failure to Start Server. The panel draws
    # "ppf-cts-server not found" for this directory at the moment it is set,
    # so a connection reported against it puts that line and "Connected" on
    # screen together, and only a second button press settles which one is
    # true. Refusing here keeps the two in agreement.
    #
    # PPF_MAC_NATIVE_NO_SPAWN is the test/CI mode where an external
    # orchestrator owns the server, which is exactly the case where the binary
    # need not be under this root; the spawn path skips its probes for the
    # same reason.
    if not os.environ.get("PPF_MAC_NATIVE_NO_SPAWN"):
        if mac_native_server_binary(root, device) is None:
            raise FileNotFoundError(mac_native_not_found_message(root, device))

    # Same contract as connect_win_native: a server already on the port has to
    # be the build the Compute Device names, asked under the add-on's project.
    check_running_server(
        _query_ppf_cts_server(port, name=project_name),
        root, port, device, mac_native_server_binary,
    )

    connection_info = ConnectionInfo()
    connection_info.type = "mac_native"
    connection_info.current_directory = root
    connection_info.remote_root = root
    connection_info.instance = "mac_native"
    connection_info.server_running = False
    connection_info.container = ""
    connection_info.server_port = port
    return connection_info, None


def spawn_linux_native_server(
    root,
    port,
    cuda_device=AUTOMATIC,
    cuda_device_uuid="",
    device=DEVICE_GPU,
    gpu_backend=GPU_BACKEND_AUTO,
):
    """Spawn a fresh linux_native ``ppf-cts-server`` subprocess and return the Popen.

    Used by both initial connect and the Stop/Start cycle on the linux_native
    backend. ``connect_linux_native`` is the one-shot init path; this helper is
    what ``LinuxNativeBackend.start_server`` calls to relaunch after a
    user-issued Stop.

    IT CARRIES A GPU SELECTION AND THE macOS TWIN DOES NOT, because Linux is
    where both halves of that choice mean something: a Linux x86_64
    distribution carries CUDA and ROCm together, so *gpu_backend* picks the
    accelerator, and CUDA numbers its devices, so *cuda_device* picks which
    card. Metal has neither question to answer.

    Args:
        root: Project root the server runs from.
        port: TCP port the server listens on.
        cuda_device: Saved GPU display index, or ``gpu_devices.AUTOMATIC``.
        cuda_device_uuid: Stable GPU identity for ``CUDA_VISIBLE_DEVICES``.
        device: ``DEVICE_GPU`` or ``DEVICE_CPU``, which build to run.
        gpu_backend: The accelerator to run, or ``GPU_BACKEND_AUTO``.

    Returns:
        A ``subprocess.Popen`` for the freshly-launched server, ``None`` when
        ``PPF_LINUX_NATIVE_NO_SPAWN`` is set (test/CI mode), OR ``None`` when a
        ppf-cts-server is already alive on *port* (attach mode, e.g. Blender
        restart while the previous session's server lingers).

    Raises:
        FileNotFoundError: if ``ppf-cts-server`` is missing under *root*.
        NoUsableGpuBackend: if no GPU build under *root* has a usable device.
        PortInUseByForeignProcess: if *port* is bound by something that isn't
            a ppf-cts-server we recognize.
    """
    root = root.rstrip("/")

    # Test/CI mode: an external orchestrator (e.g. the headless debug rig)
    # already started the server. Bail out before any path probes so we don't
    # fail on a missing binary the rig will provide.
    if os.environ.get("PPF_LINUX_NATIVE_NO_SPAWN"):
        return None

    # If a ppf-cts-server is already running on the port (Blender was restarted
    # while the previous session's server kept going), attach to it instead of
    # erroring out. The probe sends a real TCMD ping and checks the JSON
    # response, so a foreign squatter still surfaces as
    # PortInUseByForeignProcess below.
    #
    # Attaching means no launch happens, so *cuda_device* reaches nothing: the
    # server keeps whatever GPU it was started on. *device* is different: a
    # server running the other BUILD is refused rather than reported, because
    # every solve it ran would use that build.
    if _port_is_in_use(port):
        response = _query_ppf_cts_server(port)
        if response is not None:
            check_running_server(
                response,
                root,
                port,
                device,
                # The SELECTED backend's build is what a run would use, so that
                # is the directory the running server is compared against. No
                # probe: attaching runs no solver, and asking one here would
                # make a refusal depend on a device the server may not use.
                lambda where, which: linux_native_server_binary(
                    where, which, gpu_backend
                ),
            )
            return None
        raise PortInUseByForeignProcess(port)

    # THE PROBE IS PASSED HERE AND NOWHERE ELSE: a spawn is the one moment a
    # solver is about to run, so it is the moment to ask which accelerator can
    # run it. `NoUsableGpuBackend` carries what each backend reported and
    # reaches the artist unchanged.
    rust_bin = linux_native_server_binary(
        root, device, gpu_backend, lambda path: probe_gpu_build(path, root)
    )
    if rust_bin is None:
        raise FileNotFoundError(
            _absent_gpu_backend_message(root, gpu_backend, linux_native_gpu_builds)
            or linux_native_not_found_message(root, device)
        )

    # No library search path is set here, and setting one would be a defect
    # rather than a convenience. Every binary this project ships on Linux finds
    # its backend library through its own RPATH, which the loader searches
    # BEFORE LD_LIBRARY_PATH, and `build-linux-native/bundle.sh` writes that
    # RPATH precisely so a distribution is not at the mercy of whatever a
    # developer's shell exports. Adding a search path here would reintroduce
    # what that arrangement exists to prevent.
    #
    # What the child does need is an interpreter for the build worker
    # ppf-cts-server spawns. The server resolves that as PPF_CTS_BUILD_PYTHON,
    # then $VIRTUAL_ENV/bin/python, then ``python3`` on PATH. A distribution
    # carries its own interpreter inside the folder the artist selected, and
    # leaving the choice to PATH would hand its build worker whichever python3
    # the machine happens to have, with none of the frontend's dependencies.
    #
    # AN INHERITED PPF_CTS_BUILD_PYTHON WINS. Someone who set it meant it, and
    # this is also how the add-on is launched from inside a distribution's own
    # environment without the two disagreeing.
    env = os.environ.copy()
    build_python = native_build_python(root)
    if build_python and not env.get("PPF_CTS_BUILD_PYTHON"):
        env["PPF_CTS_BUILD_PYTHON"] = build_python
        bin_dir = os.path.dirname(build_python)
        env["PATH"] = bin_dir + ":" + env.get("PATH", "")
        # ONE INTERPRETER, NAMED ONCE. $VIRTUAL_ENV is set when the answer IS a
        # venv, so a subprocess reading that instead of the explicit variable
        # sees the same environment; and it is CLEARED when the answer is not,
        # because Blender is commonly started from a shell with some venv
        # active and an inherited one would otherwise sit in the environment
        # contradicting the interpreter actually chosen. A distribution's
        # relocatable interpreter is never a venv: a venv records absolute
        # paths in pyvenv.cfg and would not survive being copied, which is why
        # the distribution installs into the tree instead. So that file is the
        # thing to ask, not the path.
        prefix = os.path.dirname(bin_dir)
        if os.path.isfile(os.path.join(prefix, "pyvenv.cfg")):
            env["VIRTUAL_ENV"] = prefix
        else:
            env.pop("VIRTUAL_ENV", None)
    env["PYTHONPATH"] = root + ":" + env.get("PYTHONPATH", "")
    _apply_target_dir(env, rust_bin)
    # The solver never calls cudaSetDevice, so it runs on device 0 of whatever
    # CUDA can see. Restricting the visible set here is what makes the panel's
    # GPU choice reach it, and the server's own hardware probe reads the same
    # variable, so what the Remote Hardware block reports is the device the
    # solver is on.
    apply_selection(env, cuda_device, cuda_device_uuid)

    # Redirect to a real file, NOT subprocess.PIPE. With PIPE the addon owns
    # the read end and never drains it; once the OS pipe buffer fills, every
    # subsequent write blocks the tokio worker thread that emitted it. After
    # enough activity every worker ends up blocked in a write syscall, the
    # runtime stops scheduling new tasks, and the server appears wedged:
    # connections accept but never get a response.
    log_path = os.path.join(root, "server.log")
    log_fp = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [rust_bin, "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_fp.close()


def connect_linux_native(
    root, port, device=DEVICE_GPU, project_name="", gpu_backend=GPU_BACKEND_AUTO
):
    """Connect using a Linux build on this machine.

    The *root* path must be the project root directory where ``ppf-cts-server``
    is located.

    Connecting does not start the server. Start Server does, on this backend as
    on every other, which is what lets a GPU be picked from the panel in
    between: the choice has to be made against a device list, and that list is
    read over the connection.

    Args:
        root: Project root directory (where ppf-cts-server lives).
        port: Port for the solver server.
        device: ``DEVICE_GPU`` or ``DEVICE_CPU``, which build to run.
        project_name: The add-on's project, so the status query is the ping the
            first poll would send anyway.
        gpu_backend: The accelerator to run, or ``GPU_BACKEND_AUTO``.

    Returns:
        A tuple of (ConnectionInfo, None). The second element is the server
        subprocess, which this step does not create.

    Raises:
        FileNotFoundError: if no ``ppf-cts-server`` is under *root* (or under an
            ancestor within the resolver's reach), unless
            ``PPF_LINUX_NATIVE_NO_SPAWN`` says an external orchestrator owns the
            server.
    """
    # A subdirectory of the real solver root (target/release,
    # target/cuda/release, target, or the distribution's bin) is an easy
    # selection to make. Walk up to the actual distribution / repo root so those
    # selections just work; fall back to the raw path when nothing qualifies so
    # the launch's FileNotFoundError still names the path the user chose.
    # Resolving here keeps current_directory / remote_root pointed at the real
    # root, which the backend uses as the transfer cwd and PYTHONPATH base and
    # which start_server spawns from.
    root = resolve_linux_native_root(root) or root.rstrip("/")
    root = root.rstrip("/")

    # Refuse a root that holds no solver, rather than reporting a connection and
    # leaving the failure to Start Server. The panel draws "ppf-cts-server not
    # found" for this directory at the moment it is set, so a connection
    # reported against it puts that line and "Connected" on screen together, and
    # only a second button press settles which one is true. Refusing here keeps
    # the two in agreement.
    #
    # PPF_LINUX_NATIVE_NO_SPAWN is the test/CI mode where an external
    # orchestrator owns the server, which is exactly the case where the binary
    # need not be under this root; the spawn path skips its probes for the same
    # reason.
    if not os.environ.get("PPF_LINUX_NATIVE_NO_SPAWN"):
        # NO PROBE WHILE CONNECTING. Whether the chosen accelerator has a usable
        # device is asked when a server is spawned; here the question is only
        # whether this root holds the build the panel names, so connecting on a
        # machine whose GPU is busy or absent still reports the root honestly
        # and the refusal arrives from the spawn, naming the device.
        if linux_native_server_binary(root, device, gpu_backend) is None:
            raise FileNotFoundError(
                _absent_gpu_backend_message(root, gpu_backend, linux_native_gpu_builds)
                or linux_native_not_found_message(root, device)
            )

    # Same contract as connect_win_native: a server already on the port has to
    # be the build the Compute Device names, asked under the add-on's project.
    check_running_server(
        _query_ppf_cts_server(port, name=project_name),
        root, port, device,
        lambda where, which: linux_native_server_binary(where, which, gpu_backend),
    )

    connection_info = ConnectionInfo()
    connection_info.type = "linux_native"
    connection_info.current_directory = root
    connection_info.remote_root = root
    connection_info.instance = "linux_native"
    connection_info.server_running = False
    connection_info.container = ""
    connection_info.server_port = port
    return connection_info, None


# WHICH RESOLVER, WHICH REFUSAL AND WHICH BUILD LISTING EACH NATIVE ANSWERS
# WITH, keyed by the backend type its connection reports.
#
# Built here, once, rather than branched on at each call site. Three callers ask
# this same question, the panel while drawing, the path check after connecting,
# and the launch, and a chain of `if backend_type == ...` in each is a place a
# platform can be added to two of the three. That gap reports nothing: the
# panel simply draws no line and the path check simply returns, which reads as
# a folder that is fine.
_NATIVE_RESOLVERS = {
    "win_native": (
        win_native_server_binary,
        win_native_not_found_message,
        win_native_gpu_builds,
        resolve_win_native_root,
    ),
    "mac_native": (
        mac_native_server_binary,
        mac_native_not_found_message,
        mac_native_gpu_builds,
        resolve_mac_native_root,
    ),
    "linux_native": (
        linux_native_server_binary,
        linux_native_not_found_message,
        linux_native_gpu_builds,
        resolve_linux_native_root,
    ),
}


def native_resolvers(backend_type):
    """``(server_binary, not_found_message, gpu_builds, resolve_root)`` for a native.

    Raises ``KeyError`` for anything else, which is the intended behavior: a
    caller reaching here with a remote backend has asked a question about a
    filesystem it cannot see, and a default would answer it with this machine's.
    """
    return _NATIVE_RESOLVERS[backend_type]


def native_path_check(
    backend_type, root, device=DEVICE_GPU, gpu_backend=GPU_BACKEND_AUTO
):
    """The refusal for *root* under this native, or ``None`` when it holds the build.

    The same sentence the launch path raises, so a missing binary is described
    once however the artist meets it.
    """
    resolve, message, builds, _ = native_resolvers(backend_type)
    if resolve(root, device, gpu_backend) is not None:
        return None
    return (
        _absent_gpu_backend_message(root, gpu_backend, builds)
        or message(root, device)
    )


# ---------------------------------------------------------------------------
# A server on ANOTHER machine
#
# WHAT IS ASKED IS THE SAME QUESTION; ONLY WHO CAN SEE THE DISK IS DIFFERENT.
# An SSH or Docker connection reaches a solver host this add-on cannot stat, so
# the host is asked once, over the connection, for a listing of what it holds
# (`core.remote_builds`), and every function below resolves that listing with
# the rule the natives resolve a filesystem with. The judgment stays in one
# place: which layouts count, what a marker overrides, which accelerator AUTO
# prefers, and what a refusal says.
#
# THE LAYOUT IS LINUX'S because the far side always is: `_do_launch_server`
# writes a bash script and `_server_join` uses POSIX joins for every backend but
# win_native, and the add-on has never reached a Windows or macOS host remotely.
# ---------------------------------------------------------------------------

REMOTE_EXE = "ppf-cts-server"


def remote_gpu_builds(root, listing):
    """Every GPU build under a remote *root*, as ``{backend: path}``."""
    return gpu_builds(root, _LINUX_DEVICE_SUBDIRS, REMOTE_EXE, _Reported(listing))


def remote_server_binary(
    root, listing, device=DEVICE_GPU, gpu_backend=GPU_BACKEND_AUTO
):
    """The server on the remote host for *device*, or ``None``.

    NO PROBE IS PASSED, AND THAT IS NOT AN OVERSIGHT. `_gpu_server_binary`'s
    probe runs a solver to ask whether its device is usable, and the solver that
    would have to answer is on the other machine: running it means another
    command over the connection, from a path this add-on has not yet decided to
    launch. So where a remote root holds several GPU builds, AUTO takes the
    first in `_GPU_BACKEND_ORDER` rather than the first usable one, and an
    artist whose remote host has both cards names the one they want. The panel
    draws the GPU Backend row for exactly that case.
    """
    fs = _Reported(listing)
    if device == DEVICE_GPU:
        named = _gpu_server_binary(
            root, _LINUX_DEVICE_SUBDIRS, REMOTE_EXE, gpu_backend, None, fs
        )
        if named is not None or (gpu_backend and gpu_backend != GPU_BACKEND_AUTO):
            return named
    return _native_server_binary(
        root, device, _LINUX_DEVICE_SUBDIRS, REMOTE_EXE, fs
    )


def remote_holds_any_server(root, listing):
    """Whether the remote *root* holds a server in any layout."""
    return _holds_any_server(
        root, _LINUX_DEVICE_SUBDIRS, REMOTE_EXE, _Reported(listing)
    )


def remote_target_dir(server_binary):
    """The cargo target directory that owns a remote *server_binary*, or ``None``.

    The POSIX twin of `native_target_dir`, and it exists for the same reason:
    the server binary is one of three things a run takes out of a build
    directory, and naming only the binary leaves the build worker's frontend to
    find the cdylib wherever it likes. Spelled with `posixpath` rather than
    reusing that function because the path belongs to the SOLVER HOST: on a
    Windows client `os.path` would answer about a path it is not looking at.
    """
    if not server_binary:
        return None
    profile_dir = posixpath.dirname(server_binary)
    if posixpath.basename(profile_dir) != "release":
        return None
    return posixpath.dirname(profile_dir)


def remote_not_found_message(root, listing, device=None, gpu_backend=None):
    """Why the remote *root* cannot serve this selection, or ``None`` when it can.

    THE THREE REFUSALS ARE DIFFERENT AND THE ARTIST CAN ACT ON EACH: the folder
    holds no solver at all, it holds the other device's build, or it holds GPU
    builds and not the accelerator that was asked for. A single "not found"
    would send them to change the path in all three cases, and in two of them
    the path is right.
    """
    if remote_server_binary(root, listing, device or DEVICE_GPU, gpu_backend
                            or GPU_BACKEND_AUTO) is not None:
        return None
    if gpu_backend and gpu_backend != GPU_BACKEND_AUTO:
        named = [name for name in remote_gpu_builds(root, listing) if name]
        if named:
            return (
                f"The solver host holds these GPU builds under {root}: "
                f"{', '.join(named)}, and no {gpu_backend.lower()} build. Set "
                f"GPU Backend to one of those or to Automatic."
            )
    if device in (DEVICE_GPU, DEVICE_CPU):
        other = DEVICE_CPU if device == DEVICE_GPU else DEVICE_GPU
        if remote_server_binary(root, listing, other) is not None:
            return (
                f"The solver host holds the {other} build of the solver under "
                f"{root} and no {device} build. Set Compute Device to {other} "
                f"to run it, or point Remote Path at a folder that has a "
                f"{device} build."
            )
    if not remote_holds_any_server(root, listing):
        return (
            f"{REMOTE_EXE} not found under {root} on the solver host, in any "
            f"layout (target/release, target/cuda/release, target/rocm/release "
            f"or target/cpu/release). Point Remote Path at the folder that "
            f"holds a built solver, and build one there with "
            f"`cargo build --release -p ppf-cts-server` if there is none."
        )
    return (
        f"No {device} build of the solver is under {root} on the solver host."
    )
