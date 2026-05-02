# File: connection.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection establishment/teardown functions extracted from client.py.

import os
import subprocess

from ..models.console import console
from .module import import_module
from .status import ConnectionInfo


def connect_ssh(
    host,
    port,
    username,
    key_path,
    path,
    container,
    keepalive_interval,
    server_port,
    exec_fn,
):
    """Establish an SSH connection and return a ConnectionInfo.

    Args:
        host: SSH hostname.
        port: SSH port.
        username: SSH username.
        key_path: Path to the SSH key file.
        path: Remote working directory.
        container: Docker container name (may be empty).
        keepalive_interval: SSH keep-alive interval in seconds.
        server_port: The solver server port.
        exec_fn: Callable ``exec_fn(command, connection=..., shell=..., cwd=...)``
                  used to run commands on the remote; signature matches
                  ``protocol.exec_command``.

    Returns:
        A populated ConnectionInfo instance.
    """
    paramiko = import_module("paramiko")
    instance = paramiko.SSHClient()
    instance.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    instance.connect(
        hostname=host,
        port=port,
        username=username,
        key_filename=key_path,
        compress=True,
    )
    instance.get_transport().set_keepalive(keepalive_interval)

    connection = ConnectionInfo()
    connection.type = "ssh"
    connection.current_directory = path
    connection.instance = instance
    connection.container = container
    connection.server_port = server_port

    if container:
        result = exec_fn(
            f"docker ps -a --filter 'name={container}' --format '{{{{.Names}}}}'",
            connection=connection,
        )
        exit_code = result["exit_code"]
        if exit_code != 0:
            raise Exception(f"Error: {result['stderr']}")
        container_check = result["stdout"]
        if not container_check:
            connection.instance.close()
            connection.instance = None
            raise Exception(f"Container '{container}' does not exist.")
        result = exec_fn(
            f"docker inspect -f '{{{{.State.Running}}}}' {container}",
            connection=connection,
        )
        exit_code = result["exit_code"]
        if exit_code != 0:
            connection.instance.close()
            connection.instance = None
            raise Exception(f"Error: {result['stderr']}")
        is_running_str = result["stdout"]
        if is_running_str != "true":
            start_output = exec_fn(
                f"docker start {container}", connection=connection
            )
            if not start_output:
                connection.instance.close()
                connection.instance = None
                raise Exception(f"error starting container '{container}'")

    return connection


def connect_docker(container, path, server_port):
    """Establish a Docker connection and return a ConnectionInfo.

    Args:
        container: Docker container name.
        path: Working directory inside the container.
        server_port: The solver server port.

    Returns:
        A populated ConnectionInfo instance.
    """
    console.write(f"connecting to Docker container {container}...")
    docker = import_module("docker")
    client = docker.from_env()
    container_instance = client.containers.get(container)

    if container_instance.status != "running":
        container_instance.start()
        container_instance.reload()

    connection_info = ConnectionInfo()
    connection_info.type = "docker"
    connection_info.current_directory = path
    connection_info.instance = container_instance
    connection_info.container = container
    connection_info.server_port = server_port
    return connection_info


def connect_local(path, server_port):
    """Establish a local connection and return a ConnectionInfo.

    Args:
        path: Local working directory.
        server_port: The solver server port.

    Returns:
        A populated ConnectionInfo instance.
    """
    connection_info = ConnectionInfo()
    connection_info.type = "local"
    connection_info.current_directory = path
    connection_info.instance = "local"
    connection_info.server_port = server_port
    return connection_info


class PortInUseByForeignProcess(Exception):
    """A non-server.py process is bound to the port the addon wants.

    Raised by :func:`kill_local_server_on_port` so callers (connect,
    stop_server) can decide how to react. Connect refuses; stop swallows.
    Killing an unrelated third-party process here would be unsafe.
    """

    def __init__(self, port: int, pid: int, info: str):
        super().__init__(
            f"Port {port} is held by PID {pid} ({info}); "
            "stop that process before starting the solver server."
        )
        self.port = port
        self.pid = pid
        self.info = info


def kill_local_server_on_port(port: int) -> int:
    """Reap any orphan ``server.py`` listening on *port* on this host.

    Returns the number of processes killed (0 if the port was free).
    Raises :class:`PortInUseByForeignProcess` when the port is held by
    something that is **not** a ``python.exe`` running ``server.py``.

    Best-effort: returns 0 silently if ``netstat`` / ``wmic`` /
    ``taskkill`` aren't on PATH; the caller's subsequent bind attempt
    will surface the conflict naturally.
    """
    try:
        netstat = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0
    port_token = f":{port}"
    pids: set[int] = set()
    for line in (netstat.stdout or "").splitlines():
        if "LISTENING" not in line or port_token not in line:
            continue
        parts = line.split()
        # Confirm the port match was on the local-address column (a
        # remote-address column ending with ":{port}" would be a false
        # positive — though LISTENING rows have 0.0.0.0:0 there in
        # practice, this is cheap insurance).
        if len(parts) < 2 or not parts[1].endswith(port_token):
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue

    killed = 0
    for pid in pids:
        info = _describe_pid(pid)
        if info is None:
            # Couldn't introspect the PID at all (no wmic, no powershell,
            # taskkill not on PATH). Skip it — the caller's bind attempt
            # will surface the conflict.
            continue
        blob = info.lower()
        if "python" not in blob or "server.py" not in blob:
            raise PortInUseByForeignProcess(port, pid, info)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True, timeout=5,
            )
            killed += 1
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return killed


def _describe_pid(pid: int) -> str | None:
    """Return a short description of *pid* (Name + CommandLine) so the
    reaper can verify it before taskkill. wmic was retired on recent
    Windows builds; fall back to PowerShell's CIM bridge there. Returns
    None when neither tool can be found.
    """
    try:
        wmic = subprocess.run(
            ["wmic", "process", "where", f"ProcessId={pid}",
             "get", "Name,CommandLine", "/format:list"],
            capture_output=True, text=True, timeout=5,
        )
        out = wmic.stdout.strip()
        if out:
            return out
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    try:
        ps = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' "
                "| Select-Object Name,CommandLine | Format-List",
            ],
            capture_output=True, text=True, timeout=10,
        )
        out = ps.stdout.strip()
        if out:
            return out
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def spawn_win_native_server(root, port):
    """Spawn a fresh win_native ``server.py`` subprocess and return the Popen.

    Used by both initial connect and the Stop/Start cycle on the win_native
    backend. ``connect_win_native`` is the one-shot init path; this helper
    is what ``WinNativeBackend.start_server`` calls to relaunch after a
    user-issued Stop.

    Reaps any squatter on ``port`` first so the new bind wins the race.

    Returns:
        A ``subprocess.Popen`` for the freshly-launched server, or ``None``
        when ``PPF_WIN_NATIVE_NO_SPAWN`` is set (test/CI mode).

    Raises:
        FileNotFoundError: if ``server.py`` or the embedded Python is missing.
        PortInUseByForeignProcess: if *port* is held by a non-server.py process.
    """
    root = root.rstrip("/\\")
    server_py = os.path.join(root, "server.py")
    if not os.path.exists(server_py):
        raise FileNotFoundError(f"server.py not found: {server_py}")

    if os.environ.get("PPF_WIN_NATIVE_NO_SPAWN"):
        return None

    kill_local_server_on_port(port)

    build_dir = os.path.join(root, "build-win-native")
    if os.path.exists(os.path.join(build_dir, "python", "python.exe")):
        python_exe = os.path.join(build_dir, "python", "python.exe")
        extra_paths = [
            os.path.join(build_dir, "python"),
            os.path.join(root, "target", "release"),
            os.path.join(root, "src", "cpp", "build", "lib"),
            os.path.join(build_dir, "cuda", "bin"),
        ]
        cuda_path = os.path.join(build_dir, "cuda")
    elif os.path.exists(os.path.join(root, "python", "python.exe")):
        python_exe = os.path.join(root, "python", "python.exe")
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
    env["PATH"] = ";".join(extra_paths + [env.get("PATH", "")])
    env["PYTHONPATH"] = root + ";" + env.get("PYTHONPATH", "")
    if cuda_path and os.path.exists(cuda_path):
        env["CUDA_PATH"] = cuda_path

    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    return subprocess.Popen(
        [python_exe, server_py, "--port", str(port)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )


def connect_win_native(root, port):
    """Connect using Windows native build.

    The *root* path must be the project root directory where ``server.py``
    is located.  The function auto-detects the Python/CUDA environment from
    either ``build-win-native/`` (dev layout) or bundled ``python/`` + ``bin/``
    (dist layout).

    Args:
        root: Project root directory (where server.py lives).
        port: Port for the solver server.

    Returns:
        A tuple of (ConnectionInfo, subprocess.Popen).

    Raises:
        FileNotFoundError: if ``server.py`` is not at *root* or the
            embedded Python is missing.
        PortInUseByForeignProcess: if *port* is bound by a process that
            isn't an addon-spawned ``server.py``. Stop that process
            before retrying — the addon refuses to terminate unrelated
            third-party listeners.
    """
    # Test/CI mode: when ``PPF_WIN_NATIVE_NO_SPAWN`` is set, an external
    # process (typically the debug rig's orchestrator) has already
    # started server.py on ``port``. Skip the embedded-Python detection
    # and the Popen, and return ConnectionInfo with ``process=None`` so
    # WinNativeBackend treats the connection as metadata-only -- the
    # same shape connect_local has on Linux.
    process = spawn_win_native_server(root, port)
    root = root.rstrip("/\\")

    connection_info = ConnectionInfo()
    connection_info.type = "win_native"
    connection_info.current_directory = root
    connection_info.remote_root = root
    connection_info.instance = "win_native"
    connection_info.server_running = True
    connection_info.container = ""
    connection_info.server_port = port
    return connection_info, process


def disconnect_ssh(connection):
    """Close an SSH connection.

    Args:
        connection: A ConnectionInfo with an open SSH instance.
    """
    if connection.instance:
        connection.instance.close()
    connection.clear()


def disconnect_docker(connection):
    """Clear a Docker connection.

    Args:
        connection: A ConnectionInfo for a Docker connection.
    """
    connection.clear()


def disconnect_local(connection):
    """Clear a local connection.

    Args:
        connection: A ConnectionInfo for a local connection.
    """
    connection.clear()


def validate_remote_path(path, exec_fn):
    """Validate that required remote paths exist.

    Args:
        path: The remote working directory (``connection.current_directory``).
        exec_fn: Callable with the same signature as ``protocol.exec_command``.

    Raises:
        Exception: If a required path is not found or the command fails.
    """
    server_script_path = os.path.join(path, "server.py")
    check_list = [
        (server_script_path, "-f"),
    ]
    for check_path, kind in check_list:
        if check_path:
            result = exec_fn(
                f"if [ {kind} {check_path} ]; then echo FOUND; else echo NOT_FOUND; fi",
                shell=True,
                cwd="/",
            )
            exit_code = result["exit_code"]
            output = result["stdout"]
            if exit_code != 0 or not output:
                raise Exception("Error: Command execution failed.")
            if "NOT_FOUND" in output:
                raise Exception(f"Remote path not found ({check_path}).")
