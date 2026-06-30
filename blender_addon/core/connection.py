# File: connection.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection establishment/teardown functions extracted from client.py.

import os
import subprocess

from .status import ConnectionInfo


class PortInUseByForeignProcess(Exception):
    """A process is bound to the port the addon wants.

    Raised when the addon's bind attempt fails because the port is
    already taken. The addon does not try to identify or kill the
    squatter; the user must stop it manually before retrying.
    """

    def __init__(self, port: int, detail: str = ""):
        msg = f"Port {port} is in use"
        if detail:
            msg += f" ({detail})"
        msg += "; stop the process holding it before starting the solver server."
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
    """True if a ppf-cts-server is alive on *port*.

    Sends a minimal TCMD ping (length-prefixed empty-args header) and
    requires the response to be valid JSON containing
    ``protocol_version``. That field is on every response shape, so the
    check distinguishes our server from arbitrary other listeners
    (e.g. a Jupyter notebook server that someone parked on 9090).

    Used by the win_native connect path so a Blender restart can
    re-attach to a still-running server from the previous session
    instead of failing with PortInUseByForeignProcess.
    """
    import json as _json
    import socket as _socket

    payload = b"--name __probe__"
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
                    return False
        finally:
            s.close()
    except OSError:
        return False
    if not buf:
        return False
    try:
        resp = _json.loads(buf.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError):
        return False
    return isinstance(resp, dict) and "protocol_version" in resp


def spawn_win_native_server(root, port):
    """Spawn a fresh win_native ``ppf-cts-server.exe`` subprocess and return the Popen.

    Used by both initial connect and the Stop/Start cycle on the win_native
    backend. ``connect_win_native`` is the one-shot init path; this helper
    is what ``WinNativeBackend.start_server`` calls to relaunch after a
    user-issued Stop.

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
    # to it instead of erroring out. The user's Linux/macOS workflow
    # does this implicitly because the server is started outside the
    # addon; on Windows the addon owns the spawn, so we have to
    # recognize the attach case explicitly. The probe sends a real
    # TCMD ping and checks the JSON response, so a foreign squatter
    # (e.g. some other tool parked on 9090) still surfaces as
    # PortInUseByForeignProcess below.
    if _port_is_in_use(port):
        if _probe_ppf_cts_server(port):
            return None
        raise PortInUseByForeignProcess(port)

    # Resolve the Rust ``ppf-cts-server.exe`` binary. We look in the dev
    # layout first, then the bundled ``bin/`` so a Windows native bundle
    # can ship the binary alongside the embedded Python. Only required
    # when we actually need to spawn; the attach path above already
    # returned. Resolved before the embedded-Python branch so an
    # all-missing layout surfaces the server.exe error first.
    candidates = [
        os.path.join(root, "target", "release", "ppf-cts-server.exe"),
        os.path.join(root, "bin", "ppf-cts-server.exe"),
    ]
    rust_bin = next((p for p in candidates if os.path.exists(p)), None)
    if rust_bin is None:
        raise FileNotFoundError(
            "Rust ppf-cts-server.exe not found in "
            f"{candidates}. Build with `cargo build --release -p ppf-cts-server`."
        )

    build_dir = os.path.join(root, "build-win-native")
    if os.path.exists(os.path.join(build_dir, "python", "python.exe")):
        extra_paths = [
            os.path.join(build_dir, "python"),
            os.path.join(root, "target", "release"),
            os.path.join(root, "src", "cpp", "build", "lib"),
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
    env["PATH"] = ";".join(extra_paths + [env.get("PATH", "")])
    env["PYTHONPATH"] = root + ";" + env.get("PYTHONPATH", "")
    if cuda_path and os.path.exists(cuda_path):
        env["CUDA_PATH"] = cuda_path

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


def connect_win_native(root, port):
    """Connect using Windows native build.

    The *root* path must be the project root directory where
    ``ppf-cts-server.exe`` is located. The function auto-detects the
    Python/CUDA environment from either ``build-win-native/`` (dev layout)
    or bundled ``python/`` + ``bin/`` (dist layout).

    Args:
        root: Project root directory (where ppf-cts-server.exe lives).
        port: Port for the solver server.

    Returns:
        A tuple of (ConnectionInfo, subprocess.Popen).

    Raises:
        FileNotFoundError: if ``ppf-cts-server.exe`` is not at *root* or the
            embedded Python is missing.
        PortInUseByForeignProcess: if *port* is bound by a process that
            isn't an addon-spawned ``ppf-cts-server.exe``. Stop that process
            before retrying; the addon refuses to terminate unrelated
            third-party listeners.
    """
    # Test/CI mode: when ``PPF_WIN_NATIVE_NO_SPAWN`` is set, an external
    # process (typically the debug rig's orchestrator) has already
    # started ppf-cts-server.exe on ``port``. Skip the embedded-Python
    # detection and the Popen, and return ConnectionInfo with
    # ``process=None`` so WinNativeBackend treats the connection as
    # metadata-only, the same shape connect_local has on Linux.
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
