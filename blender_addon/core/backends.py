# File: backends.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection backend abstraction.
#
# A ``ConnectionBackend`` encapsulates everything needed to talk to a remote
# (or local) solver: opening channels, executing commands, querying the
# server, and sending/receiving data.  Four concrete implementations cover
# SSH, Docker, and the three native modes.
#
# The old ``connection.type`` string checks scattered across protocol.py,
# client.py, and connection.py are replaced by polymorphic dispatch on the
# backend instance.

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from .protocol import (
    DATA_SEND_PICKLE_REJECT,
    DEFAULT_CHUNK_SIZE,
    HEADER_TEXT_CMD,
    _read_ok_response,
    _send_json_header,
    format_traffic,
    socket_data_send,
    socket_data_receive,
    socket_upload_atomic,
)
from .connection import DEVICE_GPU
from .gpu_devices import AUTOMATIC
from .server_kill import KillReport, kill_local_server
from .status import BytesPerSecondCalculator
from ..models.console import console
from ..models.defaults import DEFAULT_SERVER_PORT, DEFAULT_SSH_KEEPALIVE_INTERVAL

# The two scene payloads land at these fixed basenames under the
# project root; data_send refuses them (only upload_atomic writes them)
# and the direct-disk path writes them straight to disk.
DATA_PICKLE = "data.pickle"
PARAM_PICKLE = "param.pickle"

def command_lines(text: str) -> list[str]:
    """One command's output as lines, keeping a LEADING TAB on the first one.

    STRIPPED OF LINE ENDINGS ONLY, never of whitespace in general, and this is
    the one place every backend gets that from.

    The build probe (`core.remote_builds.probe_command`) prints
    `<marker><TAB><directory>` per build directory, and a directory with no
    `.ppf-backend` marker prints its separator with nothing before it. That is
    not a corner case: it is the shape of a downloaded distribution, which the
    probe's own docstring names as expected. `str.strip()` on the whole output
    takes the leading TAB off the FIRST line, the line no longer carries the
    separator, `parse_listing` ignores it, and a host holding exactly one
    unmarked build is reported as holding none. The panel then draws "No solver
    build found under ...", which reads as a wrong path.

    `parse_listing` rstrips line endings only, deliberately and with a comment
    saying why; stripping here defeated that one layer above it. Measured
    against a container serving an unmarked build: the probe printed
    `"\t/root/ppf-contact-solver/target/release"` and the add-on cached `{}`.

    `splitlines` needs no trailing strip of its own: it does not invent a final
    empty line for text that ends in a newline.
    """
    return text.strip("\r\n").splitlines()


# Status-query channels (DockerBackend and the three native backends open a
# raw socket; SSHBackend a paramiko direct-tcpip
# channel) carry a short request/response round-trip, so they must complete in
# well under a second on any backend. None of the open_channel implementations
# set a timeout, so a blocking recv() waits forever if the server accepts the
# connection but never answers -- e.g. its accept loop is momentarily stalled
# while the solver finalizes and writes finished.txt. Because the single I/O
# worker thread runs one operation at a time, one such stuck query wedges the
# whole worker: the addon stops polling and hangs indefinitely with a stale
# solver=RUNNING even though the solve already finished (observed
# intermittently, and made more likely by build-time timing shifts such as a
# newer scipy). Capping the query channel lets the `_query_via_channel` except
# -> (alive=False) path fire on a stall so the next background poll retries on
# a fresh connection, which the now-freed server answers. Both socket and
# paramiko Channel expose settimeout.
_QUERY_CHANNEL_TIMEOUT_S = 30.0

# The data-transfer and upload-notify channels need the same guard as the query
# channel above. Their recv()/send() loops otherwise block forever when the
# co-located server accepts the connection but stalls mid-exchange (never writes
# the OK line or the advertised payload bytes, and never closes), wedging the
# single serial I/O worker the same way a stuck query does. This is a
# per-recv/per-send INACTIVITY bound, not a whole-transfer deadline, so a large
# but healthy transfer (bytes keep moving) is never killed; only a genuine stall
# trips it, and the worker loop's except path (see effect_runner _worker_loop)
# turns that into a loud ErrorOccurred / ConnectionLost instead of a silent hang.
_TRANSFER_CHANNEL_TIMEOUT_S = 30.0


def _force_tcp() -> bool:
    """True when ``PPF_FORCE_TCP_TRANSFER`` is set to a truthy value.

    Co-located backends (``win_native`` / ``mac_native`` / ``linux_native``)
    default to direct disk I/O: they write/read the project pickles
    straight to/from the shared filesystem instead of streaming them
    through the localhost socket. This knob routes them back through
    the wire handlers so the test rig can keep exercising the streamed
    path that SSH/Docker rely on in production. SSH/Docker never
    consult it (they have no disk to share).
    """
    val = os.environ.get("PPF_FORCE_TCP_TRANSFER", "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def _reject_scene_pickles(remote_path: str) -> None:
    """Refuse a data_send aimed at the scene pickles, on any transport.

    The scene payloads only ever land via upload_atomic (atomic + hashed),
    never via data_send. The server enforces this server-side, but the
    streamed path (channel) and the direct-disk path both pre-check here so
    both reject identically and the disk path -- which never contacts the
    server -- is guarded too. The wording is shared with the Rust server via
    ``DATA_SEND_PICKLE_REJECT``.
    """
    basename = os.path.basename(remote_path)
    if basename in (DATA_PICKLE, PARAM_PICKLE):
        raise Exception(DATA_SEND_PICKLE_REJECT.format(basename=basename))


# ---------------------------------------------------------------------------
# Backend protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class ConnectionBackend(Protocol):
    """Abstract interface for all connection types."""

    @property
    def backend_type(self) -> str:
        """The type tag: "ssh", "docker", "win_native", "mac_native" or "linux_native"."""
        ...

    @property
    def current_directory(self) -> str:
        ...

    @property
    def server_port(self) -> int:
        ...

    @property
    def container(self) -> str:
        """Return the Docker container name, or "" if not containerized."""
        return ""

    def open_channel(self) -> Any:
        """Open a socket or SSH channel to the solver server."""
        ...

    def exec_command(
        self,
        command: str,
        *,
        shell: bool = False,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Execute a command on the remote.

        Returns ``{"exit_code": int, "stdout": [str], "stderr": [str]}``.
        """
        ...

    def query(
        self,
        args: dict[str, Any],
        project_name: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> tuple[dict, bool]:
        """Send a query to the solver server.

        Returns ``(response_dict, server_running_bool)``.
        """
        ...

    def send_data(
        self,
        remote_path: str,
        data: bytes,
        project_name: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_cb: Callable | None = None,
        interrupt_cb: Callable | None = None,
        bps_window: float = 3.0,
    ) -> None:
        """Send binary data to the remote."""
        ...

    def upload_atomic(
        self,
        project_root: str,
        data: bytes,
        param: bytes,
        project_name: str,
        *,
        data_hash: str = "",
        param_hash: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_cb: Callable | None = None,
        interrupt_cb: Callable | None = None,
        bps_window: float = 3.0,
    ) -> None:
        """Upload (data.pickle, param.pickle) atomically to *project_root*.

        Either payload may be empty — the server skips that file — but at
        least one must be non-empty. The server mints a fresh upload_id,
        renames both payloads into place under a single transaction, and
        dispatches one UploadLanded event.
        """
        ...

    def receive_data(
        self,
        remote_path: str,
        project_name: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_cb: Callable | None = None,
        interrupt_cb: Callable | None = None,
        bps_window: float = 3.0,
    ) -> bytes:
        """Receive binary data from the remote."""
        ...

    def disconnect(self) -> None:
        """Close the connection and release resources."""
        ...

    def is_alive(self) -> bool:
        """Return True if the connection is still usable."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Last query failure written to the console, so a repeating one is written
# once rather than once per background poll. Cleared by a successful query,
# which is what makes the same failure after a recovery a new report rather
# than a suppressed duplicate.
_last_query_failure = ""


def _clear_query_failure() -> None:
    global _last_query_failure
    _last_query_failure = ""


def _query_via_channel(
    channel_opener: Callable,
    args: dict,
    project_name: str,
    chunk_size: int,
) -> tuple[dict, bool]:
    """Send a text-command query over a channel and parse the JSON response.

    Wire format (TCMD): ``b"TCMD"`` header (4 bytes), then a
    big-endian u32 payload-length prefix, then exactly that many
    payload bytes (the ``--key value`` argument string). The server
    reads the length, then exactly that many bytes, so we never need
    ``shutdown(SHUT_WR)`` to signal end of input. The previous wire did
    rely on the half-close, which on Windows tokio failed to deliver
    EOF to the server's AsyncRead and pinned every query in FIN_WAIT_2
    until the server's task pool drained.
    """
    if project_name is None:
        return {}, False

    args = dict(args)
    args["name"] = project_name
    flattened = ""
    for key, value in args.items():
        flattened += f"--{key} {value} "

    channel = None
    try:
        channel = channel_opener()
        # Bound every send/recv so a wedged server can't block the I/O worker
        # forever; on timeout the except below returns (alive=False) and the
        # caller retries on the next poll. See _QUERY_CHANNEL_TIMEOUT_S.
        channel.settimeout(_QUERY_CHANNEL_TIMEOUT_S)
        payload = flattened.encode()
        channel.sendall(HEADER_TEXT_CMD)
        channel.sendall(len(payload).to_bytes(4, "big"))
        total_sent = 0
        while total_sent < len(payload):
            sent = channel.send(payload[total_sent : total_sent + chunk_size])
            if sent == 0:
                raise RuntimeError("Socket connection broken during send")
            total_sent += sent
        # No half-close: server already knows the exact payload length
        # from the prefix and is expected to write the response and
        # then fully close, which is observable on every platform.
        response_data = b""
        while True:
            chunk = channel.recv(chunk_size)
            if not chunk:
                break
            response_data += chunk
        if not response_data:
            raise Exception("Empty JSON response.")
        response = json.loads(response_data.decode())
        _clear_query_failure()
        return response, True
    except Exception as e:
        # The contract here is (response, alive), so the cause has nowhere to
        # go in the return value, and the background poll stays quiet on a
        # failed query by design: a server that has not finished booting must
        # not trip a connection-lost reset. The console is therefore the only
        # place the cause can be recorded, and it is what separates "not up
        # yet" from a port nothing forwards, which are otherwise the same
        # silence.
        #
        # Written only when it CHANGES. The background poll repeats for as
        # long as the connection is held, so an unreachable server would
        # otherwise put one identical line in the console per tick and bury
        # everything else in it. One line per distinct cause is what a
        # diagnosis needs, and a cause that alternates still shows both.
        global _last_query_failure
        cause = f"{type(e).__name__}: {e}"
        if cause != _last_query_failure:
            _last_query_failure = cause
            console.write(f"Server query failed: {cause}")
        return {}, False
    finally:
        if channel:
            channel.close()


def _send_via_channel(
    channel_opener: Callable,
    remote_path: str,
    data: bytes,
    project_name: str,
    chunk_size: int,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
    bps_window: float,
) -> None:
    """Send data over a channel (socket or SSH)."""
    if project_name is None:
        raise Exception("Project name is not set.")
    if data is None or len(data) == 0:
        if progress_cb:
            progress_cb(1.0, "")
        raise Exception("No data to send.")
    # Pre-check so the streamed path rejects the scene pickles with the
    # same provenance as the disk path, instead of relying on the server.
    _reject_scene_pickles(remote_path)

    if progress_cb:
        progress_cb(0.0, format_traffic(0))

    bps = BytesPerSecondCalculator(bps_window)
    request_data = {
        "request": "data_send",
        "path": remote_path,
        "size": len(data),
        "name": project_name,
    }
    channel = channel_opener()
    channel.settimeout(_TRANSFER_CHANNEL_TIMEOUT_S)
    try:
        socket_data_send(channel, request_data, data, chunk_size,
                         progress_cb, interrupt_cb, bps)
    finally:
        channel.close()
    if progress_cb:
        progress_cb(1.0, "")


def _upload_atomic_via_channel(
    channel_opener: Callable,
    project_root: str,
    data: bytes,
    param: bytes,
    project_name: str,
    chunk_size: int,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
    bps_window: float,
    data_hash: str = "",
    param_hash: str = "",
) -> None:
    """Upload (data.pickle, param.pickle) in a single atomic transaction.

    Either payload may be empty — the server will skip the corresponding
    write — but at least one must be non-empty. The server stamps a new
    upload_id.txt and dispatches a single UploadLanded event once both
    files are renamed into place.
    """
    if project_name is None:
        raise Exception("Project name is not set.")
    if not data and not param:
        raise Exception("upload_atomic requires at least one payload.")

    if progress_cb:
        progress_cb(0.0, format_traffic(0))

    bps = BytesPerSecondCalculator(bps_window)
    request_data = {
        "request": "upload_atomic",
        "path": project_root,
        "name": project_name,
        "data_size": len(data),
        "param_size": len(param),
        "data_hash": data_hash,
        "param_hash": param_hash,
    }
    channel = channel_opener()
    channel.settimeout(_TRANSFER_CHANNEL_TIMEOUT_S)
    try:
        socket_upload_atomic(channel, request_data, data, param,
                             chunk_size, progress_cb, interrupt_cb, bps)
    finally:
        channel.close()
    if progress_cb:
        progress_cb(1.0, "")


def _receive_via_channel(
    channel_opener: Callable,
    remote_path: str,
    project_name: str,
    chunk_size: int,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
    bps_window: float,
) -> bytes:
    """Receive data over a channel (socket or SSH)."""
    if progress_cb:
        progress_cb(0.0, format_traffic(0))

    bps = BytesPerSecondCalculator(bps_window)
    request_data = {
        "request": "data_receive",
        "path": remote_path,
        "name": project_name,
    }
    channel = channel_opener()
    channel.settimeout(_TRANSFER_CHANNEL_TIMEOUT_S)
    try:
        data = socket_data_receive(channel, request_data, chunk_size,
                                   progress_cb, interrupt_cb, bps)
    finally:
        channel.close()
    if progress_cb:
        progress_cb(1.0, "")
    return data


# ---------------------------------------------------------------------------
# Direct-disk helpers (the co-located natives' fast path)
#
# When the addon and server share a filesystem, the payloads never need
# to cross the socket: the addon writes/reads them on disk directly.
# Only upload_atomic still touches the socket, and just for a tiny
# upload_notify control message so the server's state machine advances
# exactly as the streamed path would (UploadLanded -> Data::Uploaded,
# stale-build invalidation).
# ---------------------------------------------------------------------------

def _atomic_write_disk(path: str, data: bytes) -> None:
    """Write *data* to *path* via a sibling tempfile + ``os.replace``.

    ``os.replace`` is atomic on POSIX and on Windows NTFS, so a reader
    (a racing status reconcile, the build worker) never observes a
    half-written file. Mirrors the server's temp+rename staging.
    """
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(
        directory,
        f"{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _send_via_disk(
    remote_path: str,
    data: bytes,
    project_name: str,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
) -> None:
    """Write a generic file straight to disk (data_send equivalent)."""
    if project_name is None:
        raise Exception("Project name is not set.")
    if data is None or len(data) == 0:
        if progress_cb:
            progress_cb(1.0, "")
        raise Exception("No data to send.")
    # The disk path never contacts the server, so it must enforce the
    # scene-pickle invariant itself (only upload_atomic writes them).
    _reject_scene_pickles(remote_path)
    if interrupt_cb and interrupt_cb():
        raise Exception("Data send interrupted.")
    if progress_cb:
        progress_cb(0.0, format_traffic(0))
    parent = os.path.dirname(remote_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _atomic_write_disk(remote_path, data)
    if progress_cb:
        progress_cb(1.0, "")


def _notify_upload_via_channel(channel_opener: Callable, request_data: dict) -> None:
    """Send a payload-free ``upload_notify`` control message and await OK."""
    channel = channel_opener()
    channel.settimeout(_TRANSFER_CHANNEL_TIMEOUT_S)
    try:
        _send_json_header(channel, request_data)
        _read_ok_response(channel)
    finally:
        channel.close()


def _upload_atomic_via_disk(
    channel_opener: Callable,
    project_root: str,
    data: bytes,
    param: bytes,
    project_name: str,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
    data_hash: str,
    param_hash: str,
) -> None:
    """Write the pickles to disk, then notify the co-located server.

    The addon mints the ``upload_id`` and carries it on the notify so
    the server stamps that exact id (the streamed path mints its own
    server-side). The pickles are written atomically; the server's
    ``upload_notify`` handler writes ``upload_id.txt`` + the hash files
    and dispatches the single ``UploadLanded`` event.
    """
    if project_name is None:
        raise Exception("Project name is not set.")
    if not data and not param:
        raise Exception("upload_atomic requires at least one payload.")
    if progress_cb:
        progress_cb(0.0, format_traffic(0))

    os.makedirs(project_root, exist_ok=True)
    upload_id = uuid.uuid4().hex[:12]
    has_data = bool(data)
    has_param = bool(param)

    if has_data:
        if interrupt_cb and interrupt_cb():
            raise Exception("Upload interrupted.")
        _atomic_write_disk(os.path.join(project_root, DATA_PICKLE), data)
    if has_param:
        if interrupt_cb and interrupt_cb():
            raise Exception("Upload interrupted.")
        _atomic_write_disk(os.path.join(project_root, PARAM_PICKLE), param)

    # The big bytes are on disk now; only this tiny control message
    # crosses the socket so the server advances its state machine.
    _notify_upload_via_channel(
        channel_opener,
        {
            "request": "upload_notify",
            "name": project_name,
            "upload_id": upload_id,
            "data_hash": data_hash,
            "param_hash": param_hash,
            "has_data": has_data,
            "has_param": has_param,
        },
    )
    if progress_cb:
        progress_cb(1.0, "")


def _receive_via_disk(
    remote_path: str,
    progress_cb: Callable | None,
    interrupt_cb: Callable | None,
) -> bytes:
    """Read a file straight off disk (data_receive equivalent)."""
    if progress_cb:
        progress_cb(0.0, format_traffic(0))
    if interrupt_cb and interrupt_cb():
        raise Exception("Data receive interrupted.")
    if not os.path.isfile(remote_path):
        raise Exception(f"File not found: {remote_path}")
    with open(remote_path, "rb") as f:
        data = f.read()
    if progress_cb:
        progress_cb(1.0, "")
    return data


# ---------------------------------------------------------------------------
# SSH backend
# ---------------------------------------------------------------------------

class SSHBackend:
    """Connection via paramiko SSH client, optionally forwarded into Docker."""

    def __init__(
        self,
        instance: Any,  # paramiko.SSHClient
        directory: str,
        port: int,
        container: str = "",
        jump_clients: list | None = None,
        device: str = "GPU",
        gpu_backend: str = "AUTO",
    ) -> None:
        self._instance = instance
        self._directory = directory
        self._port = port
        self._container = container
        # The jump hosts the session is tunneled through, ordered outward from
        # this machine. Each one carries the channel the next hop runs over, so
        # the backend owns them for as long as it owns the session and closes
        # them in reverse on disconnect.
        self._jump_clients = list(jump_clients or [])
        # WHICH BUILD ON THE SOLVER HOST A RUN USES, held here for the same
        # reason the natives hold it: the launch happens at Start Server rather
        # than at connect, so a Stop/Start cycle has to make the same two
        # choices the connection did. Which DIRECTORY each answers to is asked
        # of the host, once, by `core.remote_builds`.
        self._device = device
        self._gpu_backend = gpu_backend

    @property
    def backend_type(self) -> str:
        return "ssh"

    @property
    def current_directory(self) -> str:
        return self._directory

    @property
    def server_port(self) -> int:
        return self._port

    @property
    def container(self) -> str:
        return self._container

    def open_channel(self) -> Any:
        transport = self._instance.get_transport()
        return transport.open_channel(
            kind="direct-tcpip",
            dest_addr=("localhost", self._port),
            src_addr=("localhost", 0),
        )

    def exec_command(
        self, command: str, *, shell: bool = False, cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        cwd = cwd or self._directory
        if shell:
            command = f"/bin/sh -c {shlex.quote(command)}"
        if timeout is not None:
            command = f"timeout --signal=KILL {timeout:g}s {command}"
        if self._container and not command.startswith("docker"):
            command = f"docker exec -w {cwd} {self._container} {command}"
        elif not self._container:
            command = f"cd {cwd} && {command}"
        try:
            _stdin, stdout, stderr = self._instance.exec_command(
                command,
                timeout=None if timeout is None else timeout + 1.0,
            )
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode()
            error_output = stderr.read().decode()
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}
        return {
            "exit_code": exit_code,
            "stdout": command_lines(output),
            "stderr": command_lines(error_output),
        }

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        _send_via_channel(self.open_channel, remote_path, data, project_name,
                          chunk_size, progress_cb, interrupt_cb, bps_window)

    def upload_atomic(self, project_root, data, param, project_name, *,
                      data_hash="", param_hash="",
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      progress_cb=None, interrupt_cb=None, bps_window=3.0):
        _upload_atomic_via_channel(
            self.open_channel, project_root, data, param, project_name,
            chunk_size, progress_cb, interrupt_cb, bps_window,
            data_hash=data_hash, param_hash=param_hash,
        )

    def receive_data(self, remote_path, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                     progress_cb=None, interrupt_cb=None, bps_window=3.0):
        return _receive_via_channel(self.open_channel, remote_path, project_name,
                                    chunk_size, progress_cb, interrupt_cb, bps_window)

    def disconnect(self) -> None:
        if self._instance:
            self._instance.close()
            self._instance = None
        # Closed from the far end inward, so a hop is only torn down once
        # nothing is still tunneled over it.
        _close_jump_clients(self._jump_clients)

    def is_alive(self) -> bool:
        if not self._instance:
            return False
        try:
            transport = self._instance.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------

class DockerBackend:
    """Connection via Docker API (local Docker socket)."""

    def __init__(
        self,
        instance: Any,
        directory: str,
        port: int,
        container: str = "",
        device: str = "GPU",
        gpu_backend: str = "AUTO",
    ) -> None:
        self._instance = instance  # docker container object
        self._directory = directory
        self._port = port
        self._container = container
        # The same two answers the SSH backend holds, for the same reason.
        self._device = device
        self._gpu_backend = gpu_backend

    @property
    def backend_type(self) -> str:
        return "docker"

    @property
    def current_directory(self) -> str:
        return self._directory

    @property
    def server_port(self) -> int:
        return self._port

    @property
    def container(self) -> str:
        return self._container

    def open_channel(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", self._port))
        return s

    def exec_command(
        self, command: str, *, shell: bool = False, cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        cwd = cwd or self._directory
        if shell:
            command = f"/bin/sh -c {shlex.quote(command)}"
        if timeout is not None:
            command = f"timeout --signal=KILL {timeout:g}s {command}"
        try:
            # ``demux=True`` keeps the container's two streams apart. Muxed
            # together they cannot both be reported: the caller that reads
            # stdout (the launch loop polling progress.log) would swallow a
            # diagnostic written to stderr, and a caller that reads stderr
            # would swallow the output it asked for. Keeping stdout on a
            # failing command matters for the same reason: a script that
            # printed how far it got before exiting non-zero is the only
            # description of that failure there is, and a Start Server whose
            # container-side script died has nothing else to show for it.
            exit_code, streams = self._instance.exec_run(
                command, workdir=cwd, demux=True
            )
            raw_out, raw_err = streams if isinstance(streams, tuple) else (streams, None)
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}
        return {
            "exit_code": exit_code,
            "stdout": command_lines((raw_out or b"").decode(errors="replace")),
            "stderr": command_lines((raw_err or b"").decode(errors="replace")),
        }

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        _send_via_channel(self.open_channel, remote_path, data, project_name,
                          chunk_size, progress_cb, interrupt_cb, bps_window)

    def upload_atomic(self, project_root, data, param, project_name, *,
                      data_hash="", param_hash="",
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      progress_cb=None, interrupt_cb=None, bps_window=3.0):
        _upload_atomic_via_channel(
            self.open_channel, project_root, data, param, project_name,
            chunk_size, progress_cb, interrupt_cb, bps_window,
            data_hash=data_hash, param_hash=param_hash,
        )

    def receive_data(self, remote_path, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                     progress_cb=None, interrupt_cb=None, bps_window=3.0):
        return _receive_via_channel(self.open_channel, remote_path, project_name,
                                    chunk_size, progress_cb, interrupt_cb, bps_window)

    def disconnect(self) -> None:
        self._instance = None

    def is_alive(self) -> bool:
        # Cheap pre-gate: disconnect() nulls the handle.
        if self._instance is None:
            return False
        # Probe the port so a stopped or crashed container is detected
        # rather than reported alive just because the handle was set.
        from .connection import _probe_ppf_cts_server
        return _probe_ppf_cts_server(self._port)


# ---------------------------------------------------------------------------
# Windows native backend
# ---------------------------------------------------------------------------

class WinNativeBackend:
    """Connection to a locally-launched Windows native solver."""

    def __init__(
        self,
        directory: str,
        port: int,
        process: subprocess.Popen,
        device: str = "GPU",
        gpu_backend: str = "AUTO",
    ) -> None:
        self._directory = directory
        self._port = port
        self._process = process
        # HELD FOR THE RESTART PATH. `start_server` re-spawns after a
        # user-issued Stop, and a restart that forgot the choice would come
        # back on the other build with nothing saying so. The accelerator is
        # held for the same reason: a root carrying CUDA and ROCm would
        # otherwise restart on whichever one the rule picks rather than on the
        # one the artist chose.
        self._device = device
        self._gpu_backend = gpu_backend

    @property
    def backend_type(self) -> str:
        return "win_native"

    @property
    def current_directory(self) -> str:
        return self._directory

    @property
    def server_port(self) -> int:
        return self._port

    def open_channel(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", self._port))
        return s

    def exec_command(
        self, command: str, *, shell: bool = False, cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        cwd = cwd or self._directory
        try:
            process = subprocess.Popen(
                command, shell=shell, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return {
                    "exit_code": 124,
                    "stdout": command_lines(stdout.decode()),
                    "stderr": ["command timed out"],
                }
            return {
                "exit_code": process.returncode,
                "stdout": command_lines(stdout.decode()),
                "stderr": command_lines(stderr.decode()),
            }
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        # Co-located: write straight to the shared filesystem. See
        # SSHBackend.send_data for the PPF_FORCE_TCP_TRANSFER override.
        if _force_tcp():
            _send_via_channel(self.open_channel, remote_path, data, project_name,
                              chunk_size, progress_cb, interrupt_cb, bps_window)
        else:
            _send_via_disk(remote_path, data, project_name, progress_cb, interrupt_cb)

    def upload_atomic(self, project_root, data, param, project_name, *,
                      data_hash="", param_hash="",
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            _upload_atomic_via_channel(
                self.open_channel, project_root, data, param, project_name,
                chunk_size, progress_cb, interrupt_cb, bps_window,
                data_hash=data_hash, param_hash=param_hash,
            )
        else:
            _upload_atomic_via_disk(
                self.open_channel, project_root, data, param, project_name,
                progress_cb, interrupt_cb, data_hash, param_hash,
            )

    def receive_data(self, remote_path, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                     progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            return _receive_via_channel(self.open_channel, remote_path, project_name,
                                        chunk_size, progress_cb, interrupt_cb, bps_window)
        return _receive_via_disk(remote_path, progress_cb, interrupt_cb)

    def stop_server(self) -> KillReport:
        """Terminate the local server subprocess but keep the backend alive.

        Two paths:

        - **Owned** (we spawned it): terminate the Popen handle.
        - **Attach mode** (``_process is None``): the addon adopted a
          pre-existing ``ppf-cts-server.exe`` (Blender restart, addon
          reload, etc). ``kill_local_server`` then ends the listener on
          ``self._port`` with its process tree, which gives the user a
          working Stop button regardless of how the server was started.

        THE ATTACH KILL IS SCOPED TO THIS BACKEND'S PORT, as on macOS: the
        rig runs its Windows workers in parallel, one ``ppf-cts-server.exe``
        each on its own port, and an image-name kill would end them all.
        """
        if self._process and self._process.poll() is None:
            pid = self._process.pid
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            report = KillReport("this machine", self._port, killed=(pid,))
        elif self._process is None:
            report = kill_local_server(self._port)
        else:
            report = KillReport("this machine", self._port)
        self._process = None
        return report

    def start_server(
        self, cuda_device: int = AUTOMATIC, cuda_device_uuid: str = ""
    ) -> bool:
        """Launch ``ppf-cts-server.exe`` on CUDA device *cuda_device*.

        No-op if the process is already alive (Start clicked twice) or in test
        mode where an external orchestrator owns the server.
        ``spawn_win_native_server`` also returns None when a ppf-cts-server is
        already on the port (attach mode), so ``_process`` legitimately stays
        None there.

        Returns True when a server was actually spawned, so the caller can say
        whether the device selection reached anything."""
        if self.is_alive():
            return False
        from .connection import spawn_win_native_server
        self._process = spawn_win_native_server(
            self._directory,
            self._port,
            cuda_device,
            cuda_device_uuid,
            self._device,
            self._gpu_backend,
        )
        return self._process is not None

    def disconnect(self) -> None:
        """Sever the addon's reference to the server without stopping it.

        Matches its sibling natives: the server keeps running so a
        subsequent Connect attaches via the probe path in
        ``spawn_win_native_server`` instead of trying to spawn a new
        ``ppf-cts-server.exe`` and colliding with the still-bound port.

        Why this is a no-op rather than a terminate: on Windows the
        spawned ``ppf-contact-solver.exe`` solver subprocess inherits
        the listen socket from its parent (Rust's ``Command::spawn``
        defaults to ``bInheritHandles=TRUE``, tokio's ``TcpListener``
        doesn't mark sockets non-inheritable). Killing the parent
        leaves the orphan solver squatting on port 9090: netstat shows
        it bound to a non-existent PID and the probe times out
        because no one's accepting, surfacing as ``Port N is in use``
        on the next Connect attempt.

        Explicit teardown (Stop button) still goes through
        ``stop_server`` if the user really wants to terminate the
        server -- this just decouples it from the routine disconnect
        triggered by ``load_pre`` / atexit / the addon's own
        DisconnectRequested flow.
        """
        return

    def is_alive(self) -> bool:
        # Owned process: cheap poll on the Popen handle.
        if self._process is not None:
            return self._process.poll() is None
        # Attach mode (and test mode): we don't own the process, so
        # poll the port instead. A successful TCMD probe is the
        # liveness signal the rest of the backend cares about.
        from .connection import _probe_ppf_cts_server
        return _probe_ppf_cts_server(self._port)


# ---------------------------------------------------------------------------
# macOS native backend
# ---------------------------------------------------------------------------

class MacNativeBackend:
    """Connection to a locally-launched macOS native solver."""

    def __init__(
        self,
        directory: str,
        port: int,
        process: subprocess.Popen,
        device: str = "GPU",
    ) -> None:
        self._directory = directory
        self._port = port
        self._process = process
        # HELD FOR THE RESTART PATH. `start_server` re-spawns after a
        # user-issued Stop, and a restart that forgot the choice would come
        # back on the other build with nothing saying so.
        self._device = device

    @property
    def backend_type(self) -> str:
        return "mac_native"

    @property
    def current_directory(self) -> str:
        return self._directory

    @property
    def server_port(self) -> int:
        return self._port

    def open_channel(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", self._port))
        return s

    def exec_command(
        self, command: str, *, shell: bool = False, cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        cwd = cwd or self._directory
        try:
            process = subprocess.Popen(
                command, shell=shell, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return {
                    "exit_code": 124,
                    "stdout": command_lines(stdout.decode()),
                    "stderr": ["command timed out"],
                }
            return {
                "exit_code": process.returncode,
                "stdout": command_lines(stdout.decode()),
                "stderr": command_lines(stderr.decode()),
            }
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        # Co-located: write straight to the shared filesystem. See
        # SSHBackend.send_data for the PPF_FORCE_TCP_TRANSFER override.
        if _force_tcp():
            _send_via_channel(self.open_channel, remote_path, data, project_name,
                              chunk_size, progress_cb, interrupt_cb, bps_window)
        else:
            _send_via_disk(remote_path, data, project_name, progress_cb, interrupt_cb)

    def upload_atomic(self, project_root, data, param, project_name, *,
                      data_hash="", param_hash="",
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            _upload_atomic_via_channel(
                self.open_channel, project_root, data, param, project_name,
                chunk_size, progress_cb, interrupt_cb, bps_window,
                data_hash=data_hash, param_hash=param_hash,
            )
        else:
            _upload_atomic_via_disk(
                self.open_channel, project_root, data, param, project_name,
                progress_cb, interrupt_cb, data_hash, param_hash,
            )

    def receive_data(self, remote_path, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                     progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            return _receive_via_channel(self.open_channel, remote_path, project_name,
                                        chunk_size, progress_cb, interrupt_cb, bps_window)
        return _receive_via_disk(remote_path, progress_cb, interrupt_cb)

    def stop_server(self) -> KillReport:
        """Terminate the local server subprocess but keep the backend alive.

        Two paths:

        - **Owned** (we spawned it): terminate the Popen handle.
        - **Attach mode** (``_process is None``): the addon adopted a
          pre-existing ``ppf-cts-server`` (Blender restart, addon reload,
          etc). ``kill_local_server`` finds the process listening on
          ``self._port`` and kills that one. SIGTERM first, because the
          server passes a cancel to an in-flight build worker on its own
          shutdown path, then SIGKILL for a survivor.

        THE ATTACH KILL IS SCOPED TO THIS BACKEND'S PORT, and a name match
        would be wrong here. The binary name is unique to this project but not
        unique on the host: the rig runs one ``ppf-cts-server`` per worker slot,
        each on its own port, and every worker sets ``PPF_MAC_NATIVE_NO_SPAWN``,
        so this branch is the only stop path a mac_native connection takes. A
        name-wide kill would reach into another worker's run and end its solve
        mid-flight, which ``bl_server_stop_is_real`` states as the invariant
        both POSIX kill patterns must carry the port. The Windows twin reads
        the listener's pid from ``netstat`` and is scoped the same way.
        """
        if self._process and self._process.poll() is None:
            pid = self._process.pid
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            report = KillReport("this machine", self._port, killed=(pid,))
        elif self._process is None:
            report = kill_local_server(self._port)
        else:
            report = KillReport("this machine", self._port)
        self._process = None
        return report

    def start_server(self) -> bool:
        """Launch ``ppf-cts-server`` from the solver root.

        Takes no device argument: the Metal backend opens the system default
        device and offers no way to name another, so there is no selection to
        deliver.

        No-op if the process is already alive (Start clicked twice) or in test
        mode where an external orchestrator owns the server.
        ``spawn_mac_native_server`` also returns None when a ppf-cts-server is
        already on the port (attach mode), so ``_process`` legitimately stays
        None there.

        Returns True when a server was actually spawned, so the caller can say
        whether it started one or attached to one already running.
        """
        if self.is_alive():
            return False
        from .connection import spawn_mac_native_server
        self._process = spawn_mac_native_server(
            self._directory, self._port, self._device
        )
        return self._process is not None

    def disconnect(self) -> None:
        """Sever the addon's reference to the server without stopping it.

        Matches its sibling natives: the server keeps running so a
        subsequent Connect attaches through the probe path in
        ``spawn_mac_native_server`` instead of spawning a second
        ``ppf-cts-server`` against a port the first one still holds.

        Explicit teardown (Stop button) still goes through ``stop_server`` if
        the user wants the server gone; this only decouples that from the
        routine disconnect triggered by ``load_pre`` / atexit / the addon's
        own DisconnectRequested flow.
        """
        return

    def is_alive(self) -> bool:
        # Owned process: cheap poll on the Popen handle.
        if self._process is not None:
            return self._process.poll() is None
        # Attach mode (and test mode): we don't own the process, so poll the
        # port instead. A successful TCMD probe is the liveness signal the
        # rest of the backend cares about.
        from .connection import _probe_ppf_cts_server
        return _probe_ppf_cts_server(self._port)


# ---------------------------------------------------------------------------
# Linux native backend
# ---------------------------------------------------------------------------

class LinuxNativeBackend:
    """Connection to a locally-launched Linux native solver."""

    def __init__(
        self,
        directory: str,
        port: int,
        process: subprocess.Popen,
        device: str = "GPU",
        gpu_backend: str = "AUTO",
    ) -> None:
        self._directory = directory
        self._port = port
        self._process = process
        # HELD FOR THE RESTART PATH. `start_server` re-spawns after a
        # user-issued Stop, and a restart that forgot the choice would come
        # back on the other build with nothing saying so. The accelerator is
        # held for the same reason: a root carrying CUDA and ROCm would
        # otherwise restart on whichever one the rule picks rather than on the
        # one the artist chose.
        self._device = device
        self._gpu_backend = gpu_backend

    @property
    def backend_type(self) -> str:
        return "linux_native"

    @property
    def current_directory(self) -> str:
        return self._directory

    @property
    def server_port(self) -> int:
        return self._port

    def open_channel(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", self._port))
        return s

    def exec_command(
        self, command: str, *, shell: bool = False, cwd: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        cwd = cwd or self._directory
        try:
            process = subprocess.Popen(
                command, shell=shell, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return {
                    "exit_code": 124,
                    "stdout": command_lines(stdout.decode()),
                    "stderr": ["command timed out"],
                }
            return {
                "exit_code": process.returncode,
                "stdout": command_lines(stdout.decode()),
                "stderr": command_lines(stderr.decode()),
            }
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        # Co-located: write straight to the shared filesystem. See
        # SSHBackend.send_data for the PPF_FORCE_TCP_TRANSFER override.
        if _force_tcp():
            _send_via_channel(self.open_channel, remote_path, data, project_name,
                              chunk_size, progress_cb, interrupt_cb, bps_window)
        else:
            _send_via_disk(remote_path, data, project_name, progress_cb, interrupt_cb)

    def upload_atomic(self, project_root, data, param, project_name, *,
                      data_hash="", param_hash="",
                      chunk_size=DEFAULT_CHUNK_SIZE,
                      progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            _upload_atomic_via_channel(
                self.open_channel, project_root, data, param, project_name,
                chunk_size, progress_cb, interrupt_cb, bps_window,
                data_hash=data_hash, param_hash=param_hash,
            )
        else:
            _upload_atomic_via_disk(
                self.open_channel, project_root, data, param, project_name,
                progress_cb, interrupt_cb, data_hash, param_hash,
            )

    def receive_data(self, remote_path, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                     progress_cb=None, interrupt_cb=None, bps_window=3.0):
        if _force_tcp():
            return _receive_via_channel(self.open_channel, remote_path, project_name,
                                        chunk_size, progress_cb, interrupt_cb, bps_window)
        return _receive_via_disk(remote_path, progress_cb, interrupt_cb)

    def stop_server(self) -> KillReport:
        """Terminate the local server subprocess but keep the backend alive.

        Two paths:

        - **Owned** (we spawned it): terminate the Popen handle.
        - **Attach mode** (``_process is None``): the addon adopted a
          pre-existing ``ppf-cts-server`` (Blender restart, addon reload,
          etc). ``kill_local_server`` finds the process listening on
          ``self._port`` and kills that one. SIGTERM first, because the
          server passes a cancel to an in-flight build worker on its own
          shutdown path, then SIGKILL for a survivor.

        THE ATTACH KILL IS SCOPED TO THIS BACKEND'S PORT, and a name match
        would be wrong here. The binary name is unique to this project but not
        unique on the host: the rig runs one ``ppf-cts-server`` per worker slot,
        each on its own port, and every worker sets ``PPF_LINUX_NATIVE_NO_SPAWN``,
        so this branch is the only stop path a linux_native connection takes. A
        name-wide kill would reach into another worker's run and end its solve
        mid-flight, which ``bl_server_stop_is_real`` states as the invariant
        both POSIX kill patterns must carry the port. The Windows twin reads
        the listener's pid from ``netstat`` and is scoped the same way.
        """
        if self._process and self._process.poll() is None:
            pid = self._process.pid
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            report = KillReport("this machine", self._port, killed=(pid,))
        elif self._process is None:
            report = kill_local_server(self._port)
        else:
            report = KillReport("this machine", self._port)
        self._process = None
        return report

    def start_server(
        self, cuda_device: int = AUTOMATIC, cuda_device_uuid: str = ""
    ) -> bool:
        """Launch ``ppf-cts-server`` from the solver root, on CUDA device *cuda_device*.

        IT TAKES A DEVICE AND THE macOS TWIN DOES NOT. Linux is where the GPU
        picker has something to say: a CUDA machine numbers its cards, so the
        panel's choice has to reach the launch. Metal opens the system default
        device and offers no way to name another.

        No-op if the process is already alive (Start clicked twice) or in test
        mode where an external orchestrator owns the server.
        ``spawn_linux_native_server`` also returns None when a ppf-cts-server is
        already on the port (attach mode), so ``_process`` legitimately stays
        None there.

        Returns True when a server was actually spawned, so the caller can say
        whether the device selection reached anything.
        """
        if self.is_alive():
            return False
        from .connection import spawn_linux_native_server
        self._process = spawn_linux_native_server(
            self._directory,
            self._port,
            cuda_device,
            cuda_device_uuid,
            self._device,
            self._gpu_backend,
        )
        return self._process is not None

    def disconnect(self) -> None:
        """Sever the addon's reference to the server without stopping it.

        Matches its Windows and macOS twins: the server keeps running so a
        subsequent Connect attaches through the probe path in
        ``spawn_linux_native_server`` instead of spawning a second
        ``ppf-cts-server`` against a port the first one still holds.

        Explicit teardown (Stop button) still goes through ``stop_server`` if
        the user wants the server gone; this only decouples that from the
        routine disconnect triggered by ``load_pre`` / atexit / the addon's
        own DisconnectRequested flow.
        """
        return

    def is_alive(self) -> bool:
        # Owned process: cheap poll on the Popen handle.
        if self._process is not None:
            return self._process.poll() is None
        # Attach mode (and test mode): we don't own the process, so poll the
        # port instead. A successful TCMD probe is the liveness signal the
        # rest of the backend cares about.
        from .connection import _probe_ppf_cts_server
        return _probe_ppf_cts_server(self._port)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _close_jump_clients(clients: list) -> None:
    """Close and drop every jump client, from the far end inward.

    A close that fails is not worth reporting: the caller is either tearing
    the session down or already carrying the error that brought it here, and
    a hop left half-closed is dropped with the transport either way.
    """
    while clients:
        try:
            clients.pop().close()
        except Exception:
            pass


def _open_jump_chain(
    paramiko: Any, jumps: list, target_host: str, target_port: int, keepalive: int
) -> tuple[list, Any]:
    """Connect through *jumps* in order and return the clients and the socket.

    Each hop is opened over the channel the previous hop forwards, and the
    returned socket is a channel from the last hop to the target, which is
    what paramiko's ``sock`` argument expects. The chain is torn down before
    the error is re-raised if any hop fails, since a half-open chain would
    otherwise hold sockets open with nothing left to close them.

    The channel destination is the address as this machine resolved it, which
    is what ssh sends a jump host as well: the name is resolved once, here,
    rather than depending on what the bastion's own resolver would answer.
    """
    clients: list = []
    sock = None
    try:
        for index, hop in enumerate(jumps):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=hop["host"],
                port=hop.get("port", 22),
                username=hop.get("username"),
                key_filename=hop.get("key_path"),
                sock=sock,
                compress=True,
            )
            transport = client.get_transport()
            transport.set_keepalive(keepalive)
            clients.append(client)
            if index + 1 < len(jumps):
                next_hop = jumps[index + 1]
                dest = (next_hop["host"], next_hop.get("port", 22))
            else:
                dest = (target_host, target_port)
            sock = transport.open_channel(
                kind="direct-tcpip",
                dest_addr=dest,
                src_addr=("localhost", 0),
            )
    except Exception as exc:
        # Read the hop that failed before the teardown empties the list: it is
        # the one past the last connected client, and the channel to the next
        # hop is opened in the same iteration that connected it.
        failed = jumps[min(len(clients), len(jumps) - 1)]
        _close_jump_clients(clients)
        raise Exception(
            f"Jump host {failed['host']}:{failed.get('port', 22)} failed: {exc}"
        ) from exc
    return clients, sock


def _require_published_port(instance, container: str, port: int) -> None:
    """Raise unless *container* publishes *port* to the host.

    ``NetworkSettings.Ports`` maps a container port to the host bindings
    ``docker run -p`` created, and is empty (or maps to ``None``) for a port
    that was never published. A container reachable through host networking
    publishes nothing and needs nothing, so a container on that mode is
    accepted as is rather than refused on a map it will never fill.
    """
    attrs = getattr(instance, "attrs", None) or {}
    settings = attrs.get("NetworkSettings") or {}
    if (attrs.get("HostConfig") or {}).get("NetworkMode") == "host":
        return
    ports = settings.get("Ports") or {}
    if any(key.split("/")[0] == str(port) and bindings
           for key, bindings in ports.items()):
        return
    raise Exception(
        f"Container '{container}' does not publish port {port} to this "
        f"machine, so the add-on cannot reach the solver inside it. Recreate "
        f"it with '-p {port}:{port}'."
    )


def create_backend(backend_type: str, config: dict) -> ConnectionBackend:
    """Create a ConnectionBackend from a type tag and config dict.

    The *config* dict keys vary by backend_type:

    - ``ssh``: host, port, username, key_path, path, container,
               keepalive_interval, jumps, server_port
    - ``docker``: container, path, server_port
    - ``local``: path, server_port
    - ``win_native``: path, server_port
    - ``mac_native``: path, server_port

    ``jumps`` is the resolved ProxyJump chain: one dict per hop, ordered
    outward from this machine, each holding the same host / port / username /
    key_path keys as the target.
    """
    if backend_type == "ssh":
        from .module import import_module
        paramiko = import_module("paramiko")
        keepalive = config.get("keepalive_interval", DEFAULT_SSH_KEEPALIVE_INTERVAL)
        jump_clients, sock = _open_jump_chain(
            paramiko,
            config.get("jumps") or [],
            config["host"],
            config.get("port", 22),
            keepalive,
        )
        instance = paramiko.SSHClient()
        instance.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            instance.connect(
                hostname=config["host"],
                port=config.get("port", 22),
                username=config.get("username"),
                key_filename=config.get("key_path"),
                sock=sock,
                compress=True,
            )
        except Exception:
            _close_jump_clients(jump_clients)
            raise
        instance.get_transport().set_keepalive(keepalive)

        backend = SSHBackend(
            instance=instance,
            directory=config["path"],
            port=config.get("server_port", DEFAULT_SERVER_PORT),
            container=config.get("container", ""),
            jump_clients=jump_clients,
            device=config.get("device", DEVICE_GPU),
            gpu_backend=config.get("gpu_backend", "AUTO"),
        )

        # If there's a Docker container over SSH, verify it's running
        container = config.get("container", "")
        if container:
            result = backend.exec_command(
                f"docker ps -a --filter 'name={container}' --format '{{{{.Names}}}}'",
            )
            if result["exit_code"] != 0:
                backend.disconnect()
                raise Exception(f"Error: {result['stderr']}")
            if not result["stdout"]:
                backend.disconnect()
                raise Exception(f"Container '{container}' does not exist.")
            result = backend.exec_command(
                f"docker inspect -f '{{{{.State.Running}}}}' {container}",
            )
            if result["exit_code"] != 0:
                backend.disconnect()
                raise Exception(f"Error: {result['stderr']}")
            is_running_str = "\n".join(result["stdout"]).strip()
            if is_running_str != "true":
                start_result = backend.exec_command(f"docker start {container}")
                # exec_command always returns a dict, so the truth of the dict
                # says nothing about the command. The exit code is what says
                # whether the container started.
                if start_result["exit_code"] != 0:
                    detail = "\n".join(start_result.get("stderr", [])).strip()
                    backend.disconnect()
                    raise Exception(
                        f"Error starting container '{container}'"
                        + (f": {detail}" if detail else "")
                    )

        return backend

    elif backend_type == "docker":
        from .module import import_module
        docker = import_module("docker")
        container = config["container"]
        # Every failure below reaches the user as the whole text of
        # "Connection failed: <e>", so each one has to name what is wrong and
        # what to do about it. Left to docker-py these arrive as transport
        # noise: a container that is not there raises
        # ``404 Client Error for http+docker://localhost/v1.54/containers/
        # <name>/json: Not Found ("No such container: <name>")``, and a daemon
        # the user cannot reach raises a urllib connection error. Neither
        # names the field to correct. The wording for a missing container is
        # the same sentence the Docker-over-SSH path already produces, so the
        # two Docker modes report the same condition identically.
        try:
            client = docker.from_env()
        except Exception as e:
            raise Exception(
                f"Cannot reach the Docker daemon ({e}). Check that Docker is "
                f"running, and on Linux that your user is in the 'docker' group."
            ) from e
        try:
            container_instance = client.containers.get(container)
        except docker.errors.NotFound as e:
            raise Exception(
                f"Container '{container}' does not exist. Run 'docker ps -a' "
                f"to list the containers on this daemon, and set Container to "
                f"the name of the one running the solver."
            ) from e
        if container_instance.status != "running":
            try:
                container_instance.start()
            except Exception as e:
                raise Exception(
                    f"Error starting container '{container}': {e}"
                ) from e
            container_instance.reload()
        # The add-on reaches the server through the host's loopback, so the
        # container has to publish the port. Docker-over-SSH already asks
        # `docker port` for this; ask the daemon here for the same answer,
        # which docker-py has already fetched as part of the container's
        # attributes. Without this the connection succeeds, Start Server
        # reports the server ready inside the container, and every query then
        # fails against a port nothing forwards, which is a much harder
        # failure to read than a refusal naming the missing flag.
        _require_published_port(container_instance, container,
                                config.get("server_port", DEFAULT_SERVER_PORT))
        return DockerBackend(
            instance=container_instance,
            directory=config["path"],
            port=config.get("server_port", DEFAULT_SERVER_PORT),
            container=config["container"],
            device=config.get("device", DEVICE_GPU),
            gpu_backend=config.get("gpu_backend", "AUTO"),
        )

    elif backend_type == "win_native":
        from .connection import connect_win_native
        # THE DEVICE IS REMEMBERED ON THE BACKEND, not only used once here.
        # `start_server` re-spawns after a user-issued Stop, and a restart that
        # forgot the choice would silently come back on the other build.
        device = config.get("device", DEVICE_GPU)
        # The accelerator rides beside the device for the same reason: both are
        # answers about WHICH BUILD to run, and a restart has to make the same
        # two choices the connection did.
        gpu_backend = config.get("gpu_backend", "AUTO")
        info, process = connect_win_native(
            config["path"],
            config.get("server_port", DEFAULT_SERVER_PORT),
            device,
            config.get("project_name", ""),
            gpu_backend,
        )
        return WinNativeBackend(
            directory=info.current_directory,
            port=info.server_port,
            process=process,
            device=device,
            gpu_backend=gpu_backend,
        )

    elif backend_type == "mac_native":
        from .connection import connect_mac_native
        # Same contract as win_native above.
        device = config.get("device", DEVICE_GPU)
        info, process = connect_mac_native(
            config["path"],
            config.get("server_port", DEFAULT_SERVER_PORT),
            device,
            config.get("project_name", ""),
        )
        return MacNativeBackend(
            directory=info.current_directory,
            port=info.server_port,
            process=process,
            device=device,
        )

    elif backend_type == "linux_native":
        from .connection import connect_linux_native
        # Same contract as win_native above, and Linux carries both halves of
        # it: a distribution here ships CUDA and ROCm side by side.
        device = config.get("device", DEVICE_GPU)
        gpu_backend = config.get("gpu_backend", "AUTO")
        info, process = connect_linux_native(
            config["path"],
            config.get("server_port", DEFAULT_SERVER_PORT),
            device,
            config.get("project_name", ""),
            gpu_backend,
        )
        return LinuxNativeBackend(
            directory=info.current_directory,
            port=info.server_port,
            process=process,
            device=device,
            gpu_backend=gpu_backend,
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
