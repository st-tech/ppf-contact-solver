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
# SSH, Docker, local-subprocess, and Windows-native modes.
#
# The old ``connection.type`` string checks scattered across protocol.py,
# client.py, and connection.py are replaced by polymorphic dispatch on the
# backend instance.

from __future__ import annotations

import json
import os
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
from .status import BytesPerSecondCalculator
from ..models.defaults import DEFAULT_SERVER_PORT, DEFAULT_SSH_KEEPALIVE_INTERVAL

# The two scene payloads land at these fixed basenames under the
# project root; data_send refuses them (only upload_atomic writes them)
# and the direct-disk path writes them straight to disk.
DATA_PICKLE = "data.pickle"
PARAM_PICKLE = "param.pickle"

# Status-query channels (LocalBackend / DockerBackend / WinNativeBackend open
# a raw socket; SSHBackend a paramiko direct-tcpip channel) carry a short
# request/response round-trip, so they must complete in well under a second on
# any backend. None of the open_channel implementations set a timeout, so a
# blocking recv() waits forever if the server accepts the connection but never
# answers -- e.g. its accept loop is momentarily stalled while the solver
# finalizes and writes finished.txt. Because the single I/O worker thread runs
# one operation at a time, one such stuck query wedges the whole worker: the
# addon stops polling and hangs indefinitely with a stale solver=RUNNING even
# though the solve already finished (observed intermittently, and made more
# likely by build-time timing shifts such as a newer scipy). Capping the query
# channel lets the `_query_via_channel` except -> (alive=False) path fire on a
# stall so the next background poll retries on a fresh connection, which the
# now-freed server answers. Both socket and paramiko Channel expose settimeout.
_QUERY_CHANNEL_TIMEOUT_S = 30.0


def _force_tcp() -> bool:
    """True when ``PPF_FORCE_TCP_TRANSFER`` is set to a truthy value.

    Co-located backends (``local`` / ``win_native``) default to direct
    disk I/O: they write/read the project pickles straight to/from the
    shared filesystem instead of streaming them through the localhost
    socket. This knob routes them back through the wire handlers so the
    test rig can keep exercising the streamed path that SSH/Docker rely
    on in production. SSH/Docker never consult it (they have no disk to
    share).
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
        """Return the type tag: "ssh", "docker", "local", or "win_native"."""
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
        self, command: str, *, shell: bool = False, cwd: str | None = None
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
        return response, True
    except Exception:
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
    try:
        data = socket_data_receive(channel, request_data, chunk_size,
                                   progress_cb, interrupt_cb, bps)
    finally:
        channel.close()
    if progress_cb:
        progress_cb(1.0, "")
    return data


# ---------------------------------------------------------------------------
# Direct-disk helpers (co-located local / win_native fast path)
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
    ) -> None:
        self._instance = instance
        self._directory = directory
        self._port = port
        self._container = container

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

    def exec_command(self, command: str, *, shell: bool = False, cwd: str | None = None) -> dict:
        cwd = cwd or self._directory
        if shell:
            command = f"/bin/sh -c '{command}'"
        if self._container and not command.startswith("docker"):
            command = f"docker exec -w {cwd} {self._container} {command}"
        elif not self._container:
            command = f"cd {cwd} && {command}"
        try:
            _stdin, stdout, stderr = self._instance.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode().strip()
            error_output = stderr.read().decode().strip()
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}
        return {
            "exit_code": exit_code,
            "stdout": output.splitlines(),
            "stderr": error_output.splitlines(),
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

    def __init__(self, instance: Any, directory: str, port: int, container: str = "") -> None:
        self._instance = instance  # docker container object
        self._directory = directory
        self._port = port
        self._container = container

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

    def exec_command(self, command: str, *, shell: bool = False, cwd: str | None = None) -> dict:
        cwd = cwd or self._directory
        if shell:
            command = f"/bin/sh -c '{command}'"
        try:
            exit_code, stdout = self._instance.exec_run(command, workdir=cwd)
            output = stdout.decode().strip() if exit_code == 0 else ""
            error_output = stdout.decode().strip() if exit_code != 0 else ""
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}
        return {
            "exit_code": exit_code,
            "stdout": output.splitlines(),
            "stderr": error_output.splitlines(),
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
# Local backend
# ---------------------------------------------------------------------------

class LocalBackend:
    """Connection to a local solver process via localhost socket."""

    def __init__(self, directory: str, port: int) -> None:
        self._directory = directory
        self._port = port

    @property
    def backend_type(self) -> str:
        return "local"

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

    def exec_command(self, command: str, *, shell: bool = False, cwd: str | None = None) -> dict:
        cwd = cwd or self._directory
        try:
            process = subprocess.Popen(
                command, shell=shell, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode().strip().splitlines(),
                "stderr": stderr.decode().strip().splitlines(),
            }
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        # Co-located: write straight to the shared filesystem. Set
        # PPF_FORCE_TCP_TRANSFER=1 to stream through the socket instead
        # (keeps the wire handlers under test in the rig).
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

    def disconnect(self) -> None:
        pass

    def stop_server(self) -> None:
        """Direct-kill the local ``ppf-cts-server`` bound to our port.

        The local server is launched detached (``nohup bash -c "...; \
        ppf-cts-server --port N" &``) by ``effect_runner``, so there is no
        Popen handle to terminate, and terminating the ``bash`` wrapper would
        orphan the server child. Stop therefore finds the process listening on
        ``self._port`` and kills it (SIGTERM, then SIGKILL), plus any matching
        wrapper. Best-effort and idempotent: a clean no-op when nothing is
        listening. ``effect_runner._do_stop_server`` clears the response cache
        and dispatches ``ServerStopped`` after this returns.
        """
        port = self._port
        # ``lsof -ti tcp:N`` yields the PID(s) bound to the port (the server
        # binds 127.0.0.1:N). Kill those, then sweep any lingering wrapper by
        # command match. Both ``lsof`` and ``pkill`` are present on macOS/Linux
        # where the local backend runs; guards keep it quiet if either is
        # missing or the port is already free.
        self.exec_command(
            f"PIDS=$(lsof -ti tcp:{port} 2>/dev/null); "
            f'[ -n "$PIDS" ] && kill $PIDS 2>/dev/null; '
            f"sleep 0.3; "
            f"PIDS=$(lsof -ti tcp:{port} 2>/dev/null); "
            f'[ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null; '
            f"pkill -f 'ppf-cts-server .*--port {port}' 2>/dev/null; true",
            shell=True,
        )
        return

    def start_server(self) -> None:
        """Symmetric no-op so Stop/Start cycles don't AttributeError.

        ``ServerLaunched`` is dispatched by the caller anyway; the
        external server (rig orchestrator, dev shell) is responsible
        for keeping it up.
        """
        return

    def is_alive(self) -> bool:
        # Probe the port: a crashed local server should report dead so
        # the protocol's is_alive() means actual reachability, matching
        # the SSH and win_native backends.
        from .connection import _probe_ppf_cts_server
        return _probe_ppf_cts_server(self._port)


# ---------------------------------------------------------------------------
# Windows native backend
# ---------------------------------------------------------------------------

class WinNativeBackend:
    """Connection to a locally-launched Windows native solver."""

    def __init__(self, directory: str, port: int, process: subprocess.Popen) -> None:
        self._directory = directory
        self._port = port
        self._process = process

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

    def exec_command(self, command: str, *, shell: bool = False, cwd: str | None = None) -> dict:
        cwd = cwd or self._directory
        try:
            process = subprocess.Popen(
                command, shell=shell, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode().strip().splitlines(),
                "stderr": stderr.decode().strip().splitlines(),
            }
        except Exception as e:
            return {"exit_code": 1, "stdout": [], "stderr": [str(e)]}

    def query(self, args: dict, project_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[dict, bool]:
        return _query_via_channel(self.open_channel, args, project_name, chunk_size)

    def send_data(self, remote_path, data, project_name, *, chunk_size=DEFAULT_CHUNK_SIZE,
                  progress_cb=None, interrupt_cb=None, bps_window=3.0):
        # Co-located: write straight to the shared filesystem. See
        # LocalBackend.send_data for the PPF_FORCE_TCP_TRANSFER override.
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

    def stop_server(self) -> None:
        """Terminate the local server subprocess but keep the backend alive.

        Two paths:

        - **Owned** (we spawned it): terminate the Popen handle.
        - **Attach mode** (``_process is None``): the addon adopted a
          pre-existing ``ppf-cts-server.exe`` (Blender restart, addon
          reload, etc). Fall back to ``taskkill /F /IM ppf-cts-server.exe``.
          The binary name is unique to this project, so killing every
          instance is safe and gives the user a working Stop button
          regardless of how the server was started.
        """
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        elif self._process is None:
            subprocess.run(
                ["taskkill", "/F", "/IM", "ppf-cts-server.exe"],
                capture_output=True, check=False,
            )
        self._process = None

    def start_server(self) -> None:
        """Re-launch ``ppf-cts-server.exe`` after a Stop. No-op if the
        process is already alive (Start clicked twice) or in test mode
        where an external orchestrator owns the server. ``spawn_win_native_server``
        also returns None when a ppf-cts-server is already on the port
        (attach mode), so ``_process`` legitimately stays None there."""
        if self.is_alive():
            return
        from .connection import spawn_win_native_server
        self._process = spawn_win_native_server(self._directory, self._port)

    def disconnect(self) -> None:
        """Sever the addon's reference to the server without stopping it.

        Matches ``LocalBackend.disconnect``: the server keeps running so a
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
# Factory
# ---------------------------------------------------------------------------

def create_backend(backend_type: str, config: dict) -> ConnectionBackend:
    """Create a ConnectionBackend from a type tag and config dict.

    The *config* dict keys vary by backend_type:

    - ``ssh``: host, port, username, key_path, path, container,
               keepalive_interval, server_port
    - ``docker``: container, path, server_port
    - ``local``: path, server_port
    - ``win_native``: path, server_port
    """
    if backend_type == "ssh":
        from .module import import_module
        paramiko = import_module("paramiko")
        instance = paramiko.SSHClient()
        instance.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        instance.connect(
            hostname=config["host"],
            port=config.get("port", 22),
            username=config.get("username"),
            key_filename=config.get("key_path"),
            compress=True,
        )
        keepalive = config.get("keepalive_interval", DEFAULT_SSH_KEEPALIVE_INTERVAL)
        instance.get_transport().set_keepalive(keepalive)

        backend = SSHBackend(
            instance=instance,
            directory=config["path"],
            port=config.get("server_port", DEFAULT_SERVER_PORT),
            container=config.get("container", ""),
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
                if not start_result:
                    backend.disconnect()
                    raise Exception(f"Error starting container '{container}'")

        return backend

    elif backend_type == "docker":
        from .module import import_module
        docker = import_module("docker")
        client = docker.from_env()
        container_instance = client.containers.get(config["container"])
        if container_instance.status != "running":
            container_instance.start()
            container_instance.reload()
        return DockerBackend(
            instance=container_instance,
            directory=config["path"],
            port=config.get("server_port", DEFAULT_SERVER_PORT),
            container=config["container"],
        )

    elif backend_type == "local":
        return LocalBackend(
            directory=config["path"],
            port=config.get("server_port", DEFAULT_SERVER_PORT),
        )

    elif backend_type == "win_native":
        from .connection import connect_win_native
        info, process = connect_win_native(config["path"], config.get("server_port", DEFAULT_SERVER_PORT))
        return WinNativeBackend(
            directory=info.current_directory,
            port=info.server_port,
            process=process,
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
