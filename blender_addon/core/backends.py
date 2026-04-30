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
from typing import Any, Callable, Protocol, runtime_checkable

from .protocol import (
    DEFAULT_CHUNK_SIZE,
    HEADER_TEXT_CMD,
    HEADER_JSON_DATA,
    PROTOCOL_VERSION,
    _shutdown_write,
    format_traffic,
    socket_data_send,
    socket_data_receive,
    socket_upload_atomic,
)
from .status import BytesPerSecondCalculator


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
    """Send a text-command query over a channel and parse the JSON response."""
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
        channel.sendall(HEADER_TEXT_CMD)
        data = flattened.encode()
        total_sent = 0
        while total_sent < len(data):
            sent = channel.send(data[total_sent : total_sent + chunk_size])
            if sent == 0:
                raise RuntimeError("Socket connection broken during send")
            total_sent += sent
        _shutdown_write(channel)
        response_data = b""
        while True:
            data = channel.recv(chunk_size)
            if not data:
                break
            response_data += data
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
        return self._instance is not None


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
        # Go through the server socket so the server's handler runs
        # (atomic rename, event dispatch). Skipping the socket for a
        # direct disk write bypasses upload_id minting and leaves the
        # server unaware of the upload.
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
        pass

    def is_alive(self) -> bool:
        return True


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
        # Go through the server socket so the server's handler runs
        # (atomic rename, event dispatch). See LocalBackend.send_data.
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

    def stop_server(self) -> None:
        """Terminate the local server subprocess but keep the backend alive.

        Primary route is the Popen handle. After an addon reload (handle
        dropped) or a connect that lost its bind race to an orphan
        ``server.py`` squatting on the port (handle is for a dead spawn),
        the handle is unusable and terminating it is a no-op. Fall back
        to the shared port reaper so "Stop Server" actually stops the
        live process. Foreign processes on the port are ignored here —
        we don't reap unrelated third-party listeners on a stop request.
        """
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        from .connection import (
            PortInUseByForeignProcess,
            kill_local_server_on_port,
        )
        try:
            kill_local_server_on_port(self._port)
        except PortInUseByForeignProcess:
            # Something we don't own holds the port. Disconnect should
            # not crash on that — surface only if a subsequent connect
            # tries to bind.
            pass

    def disconnect(self) -> None:
        self.stop_server()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


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
        keepalive = config.get("keepalive_interval", 60)
        instance.get_transport().set_keepalive(keepalive)

        backend = SSHBackend(
            instance=instance,
            directory=config["path"],
            port=config.get("server_port", 9090),
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
            port=config.get("server_port", 9090),
            container=config["container"],
        )

    elif backend_type == "local":
        return LocalBackend(
            directory=config["path"],
            port=config.get("server_port", 9090),
        )

    elif backend_type == "win_native":
        from .connection import connect_win_native
        info, process = connect_win_native(config["path"], config.get("server_port", 9090))
        return WinNativeBackend(
            directory=info.current_directory,
            port=info.server_port,
            process=process,
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
