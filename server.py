# File: server.py (rewritten with event-driven state machine)
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Socket server for the physics simulation solver.
# Uses the event-driven state machine from server/ package:
#   - server/state.py: frozen ServerState + enums
#   - server/events.py: typed event classes
#   - server/effects.py: typed effect classes
#   - server/transitions.py: pure transition function
#   - server/engine.py: ServerEngine + EffectExecutor
#   - server/monitor.py: background solver monitor thread

import builtins
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import threading

from server.engine import (
    ServerEngine, _new_upload_id, write_upload_id,
    write_data_hash, write_param_hash,
)
from server.state import Build
from server.events import (
    BuildRequested,
    CancelBuildRequested,
    DeleteRequested,
    ResumeRequested,
    SaveAndQuitRequested,
    StartRequested,
    TerminateRequested,
    UploadLanded,
)
from server.monitor import start_solver_monitor

PROTOCOL_VERSION = "0.03"

# Protocol headers (4 bytes each)
HEADER_TEXT_CMD = b"TCMD"
HEADER_BINARY_DATA = b"BDAT"
HEADER_JSON_DATA = b"JSON"


# ---------------------------------------------------------------------------
# Hardware / system info (cached)
# ---------------------------------------------------------------------------

_hardware_cache = None


def _get_hardware_info():
    global _hardware_cache
    if _hardware_cache is not None:
        return _hardware_cache
    import platform

    gpu_name = "Unknown"
    vram = "Unknown"
    cuda_ver = "Unknown"
    sm_cap = "Unknown"
    cpu_name = "Unknown"
    ram = "Unknown"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    gpu_name = parts[0]
                    vram = f"{int(parts[1]) / 1024:.1f} GB"
                    sm_cap = f"sm_{parts[2].replace('.', '')}"
                    break
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    for part in line.split():
                        try:
                            float(part)
                            cuda_ver = part
                            break
                        except ValueError:
                            continue
                    break
    except Exception:
        pass

    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                cpu_name = result.stdout.strip()
        else:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
        if cpu_name == "Unknown":
            cpu_name = platform.processor() or "Unknown"
    except Exception:
        pass

    try:
        import psutil
        mem = psutil.virtual_memory()
        ram = f"{mem.total / (1024**3):.1f} GB"
    except Exception:
        pass

    _hardware_cache = {
        "GPU": gpu_name, "VRAM": vram,
        "CUDA": cuda_ver, "SM": sm_cap,
        "CPU": cpu_name, "RAM": ram,
    }
    return _hardware_cache


def _get_runtime_usage():
    info = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split("\n")[0].split(",")]
            if len(parts) == 3:
                info["GPU Util"] = f"{parts[0]}%"
                used, total = int(parts[1]), int(parts[2])
                pct = round(100 * used / total) if total > 0 else 0
                info["VRAM Usage"] = f"{pct}% ({used / 1024:.1f}/{total / 1024:.1f} GB)"
    except Exception:
        pass
    try:
        import psutil
        info["CPU Usage"] = f"{psutil.cpu_percent(interval=0)}%"
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        info["RAM Usage"] = f"{mem.percent}% ({used_gb:.1f}/{total_gb:.1f} GB)"
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Git / path helpers
# ---------------------------------------------------------------------------

_GIT_BRANCH_CACHE: str | None = None


def get_git_branch():
    # Cached for the server's lifetime: the branch can't change under
    # us, and on Windows concurrent ``subprocess.check_output(["git",
    # ...])`` calls from multiple TCMD handler threads deadlock on
    # ``communicate()`` (handle-inheritance race), leaving the request
    # path hung. Mint once on first call, reuse forever.
    global _GIT_BRANCH_CACHE
    if _GIT_BRANCH_CACHE is not None:
        return _GIT_BRANCH_CACHE
    branch = "unknown"
    try:
        branch_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".git", "branch_name.txt"
        )
        if os.path.exists(branch_file):
            with open(branch_file, "r") as f:
                v = f.read().strip()
                if v:
                    _GIT_BRANCH_CACHE = v
                    return v
    except Exception:
        pass
    # Skip git invocation entirely when there's no .git dir next to
    # server.py (Windows native bundle layout). Avoids spawning a
    # process that can hang and pollutes the log with "fatal: not a
    # git repository" on every call.
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        _GIT_BRANCH_CACHE = branch
        return branch
    try:
        v = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            text=True,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if v:
            branch = v
    except Exception:
        pass
    _GIT_BRANCH_CACHE = branch
    return branch


def make_root(id_name):
    # PPF_CTS_DATA_ROOT shadows the per-user data dir for isolated test
    # runs (see blender_addon/debug/orchestrator.py). The git-branch
    # subdirectory is preserved so a single shadow root can host runs
    # from multiple branches without collision.
    #
    # On Windows the default base is rooted next to server.py so a Windows
    # native bundle stays self-contained; mirrors frontend/_app_.py's
    # get_data_dirpath. On POSIX it stays at ~/.local/share so existing
    # Linux/macOS layouts are unaffected.
    base = os.environ.get("PPF_CTS_DATA_ROOT")
    if base:
        git_branch = "debug" if "--debug" in sys.argv else get_git_branch()
        root = os.path.join(base, f"git-{git_branch}", id_name)
    else:
        git_branch = get_git_branch()
        if platform.system() == "Windows":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            root = os.path.join(
                base_dir, "local", "share", "ppf-cts",
                f"git-{git_branch}", id_name,
            )
        else:
            root = os.path.join(
                os.path.expanduser("~"),
                ".local", "share", "ppf-cts",
                f"git-{git_branch}", id_name,
            )
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    return root


def easy_parse(argv):
    key, args = None, {}
    for val in argv:
        if val.startswith("--"):
            key = val[2:]
            if key not in args:
                args[key] = None
        else:
            if args[key] is None:
                args[key] = val
            else:
                if isinstance(args[key], list):
                    args[key].append(val)
                else:
                    args[key] = [args[key], val]
    return args


# ---------------------------------------------------------------------------
# Protocol handlers
# ---------------------------------------------------------------------------

def read_protocol_header(conn, addr):
    try:
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                print(f"Client {addr} disconnected while reading protocol header")
                return None
            header += chunk
        return header
    except TimeoutError:
        print(f"Timeout reading protocol header from client {addr}")
        return None
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error reading protocol header from {addr}: {e}")
        return None


def handle_data_transfer(conn, request_data, engine):
    try:
        conn.settimeout(1.0)
        req_type = request_data.get("request", "")
        name = request_data.get("name", "")

        if req_type == "notebook_delete":
            rel = (request_data.get("relative_path") or "").strip().lstrip("/\\")
            if not name or not rel:
                conn.sendall(
                    (json.dumps({"error": "Missing name or relative_path"}) + "\n").encode()
                )
                return False

            notebook_root = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
            )
            target = os.path.abspath(os.path.join(notebook_root, rel))
            if os.path.commonpath([target, notebook_root]) != notebook_root:
                conn.sendall(
                    (json.dumps({"error": f"Path escapes sandbox: {rel}"}) + "\n").encode()
                )
                return False

            if os.path.exists(target):
                try:
                    os.remove(target)
                    print(f"Deleted notebook {target}")
                except OSError as e:
                    conn.sendall(
                        (json.dumps({"error": f"Delete failed: {e}"}) + "\n").encode()
                    )
                    return False
            conn.sendall(b"OK\n")
            return True

        if req_type == "notebook_send":
            # Write a .ipynb (or any file) under <server.py dir>/examples/.
            # The addon sends a sandbox-relative path so the server decides
            # the real location and JupyterLab's contents API is bypassed
            # entirely — Jupyter's workspace/contents machinery was
            # corrupting externally-created files.
            rel = (request_data.get("relative_path") or "").strip().lstrip("/\\")
            size = request_data.get("size", 0)
            if not name or not rel:
                conn.sendall(
                    (json.dumps({"error": "Missing name or relative_path"}) + "\n").encode()
                )
                return False
            if size <= 0:
                conn.sendall((json.dumps({"error": "Invalid size"}) + "\n").encode())
                return False

            notebook_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "examples"
            )
            target = os.path.abspath(os.path.join(notebook_root, rel))
            if os.path.commonpath([target, os.path.abspath(notebook_root)]) != os.path.abspath(notebook_root):
                conn.sendall(
                    (json.dumps({"error": f"Path escapes sandbox: {rel}"}) + "\n").encode()
                )
                return False

            os.makedirs(os.path.dirname(target), exist_ok=True)
            print(f"Expecting to receive {size} bytes for notebook {target}")
            data = b""
            bytes_received = 0
            chunk_size = 32 * 1024
            while bytes_received < size:
                remaining = size - bytes_received
                to_receive = min(chunk_size, remaining)
                try:
                    chunk = conn.recv(to_receive)
                    if not chunk:
                        print(f"Connection closed, received {bytes_received}/{size}")
                        return False
                    data += chunk
                    bytes_received += len(chunk)
                except TimeoutError:
                    print(f"Timeout during receive, got {bytes_received}/{size}")
                    return False
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"Connection error during receive: {e}")
                    return False
            with open(target, "wb") as f:
                f.write(data)
            print(f"Written notebook to {target}")
            conn.sendall(b"OK\n")
            return True

        path = request_data.get("path", "")
        if not path or not name:
            error_response = json.dumps({"error": "Missing path or name"}) + "\n"
            conn.sendall(error_response.encode())
            return False

        if req_type == "upload_atomic":
            # Atomic combined upload of (data.pickle, param.pickle).
            # Protocol: after the JSON header, the client sends *data_size*
            # bytes then *param_size* bytes back-to-back. Either size may be
            # zero (params-only update), but at least one must be present.
            # Everything lands via temp-file + rename so a concurrent reader
            # never sees a partial file. One fresh upload_id is minted and
            # both data/param files plus upload_id.txt are renamed into
            # place. A single UploadLanded event fires once on completion.
            data_size = int(request_data.get("data_size", 0))
            param_size = int(request_data.get("param_size", 0))
            if data_size <= 0 and param_size <= 0:
                conn.sendall((json.dumps({
                    "error": "upload_atomic requires at least one of "
                             "data_size or param_size to be positive",
                }) + "\n").encode())
                return False
            if engine.state.build == Build.BUILDING:
                conn.sendall((json.dumps({
                    "error": "Cannot upload while a build is in progress. "
                             "Abort the build first, then retry the upload.",
                }) + "\n").encode())
                return False

            project_root = path
            os.makedirs(project_root, exist_ok=True)

            def _recv_exact(nbytes: int) -> bytes | None:
                buf = b""
                remaining = nbytes
                cs = 32 * 1024
                while remaining > 0:
                    try:
                        chunk = conn.recv(min(cs, remaining))
                    except TimeoutError:
                        print(f"upload_atomic: timeout receiving {nbytes} bytes")
                        return None
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                        print(f"upload_atomic: connection error: {e}")
                        return None
                    if not chunk:
                        print(f"upload_atomic: connection closed mid-transfer ({remaining} bytes short)")
                        return None
                    buf += chunk
                    remaining -= len(chunk)
                return buf

            import time as _time
            stamp = f"{os.getpid()}.{_time.time_ns()}"
            data_final = os.path.join(project_root, "data.pickle")
            param_final = os.path.join(project_root, "param.pickle")
            data_tmp = f"{data_final}.tmp.{stamp}" if data_size > 0 else None
            param_tmp = f"{param_final}.tmp.{stamp}" if param_size > 0 else None

            try:
                if data_size > 0:
                    payload = _recv_exact(data_size)
                    if payload is None:
                        return False
                    with open(data_tmp, "wb") as f:
                        f.write(payload)
                if param_size > 0:
                    payload = _recv_exact(param_size)
                    if payload is None:
                        return False
                    with open(param_tmp, "wb") as f:
                        f.write(payload)

                if data_tmp is not None:
                    os.replace(data_tmp, data_final)
                    data_tmp = None
                if param_tmp is not None:
                    os.replace(param_tmp, param_final)
                    param_tmp = None

                uid = _new_upload_id()
                write_upload_id(project_root, uid)
                has_data = os.path.exists(data_final)
                has_param = os.path.exists(param_final)
                # Hashes travel in the upload header. Each is persisted
                # only when its own payload was uploaded; the other side
                # is left as-is so partial uploads (param-only) can't
                # accidentally clear the data-side fingerprint.
                upload_data_hash = str(request_data.get("data_hash", "") or "")
                upload_param_hash = str(request_data.get("param_hash", "") or "")
                if data_size > 0:
                    write_data_hash(project_root, upload_data_hash)
                if param_size > 0:
                    write_param_hash(project_root, upload_param_hash)
                engine.dispatch(UploadLanded(
                    upload_id=uid,
                    data_hash=upload_data_hash,
                    param_hash=upload_param_hash,
                    has_data=has_data, has_param=has_param,
                ))
            finally:
                for t in (data_tmp, param_tmp):
                    if t and os.path.exists(t):
                        try:
                            os.remove(t)
                        except OSError:
                            pass

            conn.sendall(b"OK\n")
            return True

        if req_type == "data_send":
            # Generic file-write transport. Used by the debug round-trip
            # tests to ship a dummy payload to the remote. Pickle uploads
            # (data.pickle / param.pickle) go through ``upload_atomic``
            # instead: those must be transactional so the build never
            # observes a partial/mismatched pair.
            basename = os.path.basename(path)
            if basename in ("data.pickle", "param.pickle"):
                conn.sendall((json.dumps({
                    "error": f"data_send no longer accepts {basename}; "
                             f"use upload_atomic for scene uploads.",
                }) + "\n").encode())
                return False
            size = request_data.get("size", 0)
            if size <= 0:
                conn.sendall((json.dumps({"error": "Invalid size"}) + "\n").encode())
                return False
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Expecting to receive {size} bytes for {path}")
            data = b""
            bytes_received = 0
            chunk_size = 32 * 1024
            while bytes_received < size:
                remaining = size - bytes_received
                to_receive = min(chunk_size, remaining)
                try:
                    chunk = conn.recv(to_receive)
                    if not chunk:
                        print(f"Connection closed, received {bytes_received}/{size}")
                        return False
                    data += chunk
                    bytes_received += len(chunk)
                except TimeoutError:
                    print(f"Timeout during receive, got {bytes_received}/{size}")
                    return False
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"Connection error during receive: {e}")
                    return False
            print(f"Received {bytes_received} bytes total")
            # Atomic write: tmp file + rename.
            import time as _time
            tmp_path = f"{path}.tmp.{os.getpid()}.{_time.time_ns()}"
            try:
                with open(tmp_path, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            print(f"Written to {path}")
            conn.sendall(b"OK\n")
            return True

        elif req_type == "data_receive":
            if not os.path.exists(path):
                conn.sendall((json.dumps({"error": "File not found"}) + "\n").encode())
                return False
            file_size = os.path.getsize(path)
            metadata = json.dumps({"size": file_size}) + "\n"
            conn.sendall(metadata.encode())
            chunk_size = 32 * 1024
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    try:
                        conn.sendall(chunk)
                    except (TimeoutError, ConnectionResetError,
                            ConnectionAbortedError, BrokenPipeError) as e:
                        print(f"Connection error during send: {e}")
                        return False
            return True

    except Exception as e:
        print(f"Error in handle_data_transfer: {e}")
        try:
            conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
        except Exception:
            print("Failed to send error response")
        return False
    return False


def handle_json_data(conn, addr, engine):
    try:
        json_line = b""
        while b"\n" not in json_line:
            chunk = conn.recv(1)
            if not chunk:
                print(f"Client {addr} disconnected while reading JSON")
                return False
            json_line += chunk
        try:
            request_data = json.loads(json_line.decode().strip())
        except UnicodeDecodeError as e:
            conn.sendall((json.dumps({"error": f"UTF-8 decode error: {e}"}) + "\n").encode())
            return False
        if isinstance(request_data, dict) and "request" in request_data:
            req_type = request_data["request"]
            if req_type in ("upload_atomic", "data_send", "data_receive",
                            "notebook_send", "notebook_delete"):
                print(f"=== handling {req_type} request from ({addr}) ===")
                return handle_data_transfer(conn, request_data, engine=engine)
            else:
                conn.sendall((json.dumps({"error": f"Unknown request: {req_type}"}) + "\n").encode())
                return False
        else:
            conn.sendall((json.dumps({"error": "Invalid request format"}) + "\n").encode())
            return False
    except (json.JSONDecodeError, ValueError) as e:
        conn.sendall((json.dumps({"error": f"JSON error: {e}"}) + "\n").encode())
        return False
    except (TimeoutError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error handling JSON from {addr}: {e}")
        return False


def handle_binary_data(conn, addr):
    try:
        binary_data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            binary_data += chunk
        print(f"Received {len(binary_data)} bytes of binary data from {addr}")
        conn.sendall(b"BINARY_OK")
        return True
    except Exception as e:
        print(f"Error handling binary data from {addr}: {e}")
        return False


# ---------------------------------------------------------------------------
# Request dispatch via engine (replaces process() + handle_request())
# ---------------------------------------------------------------------------

# Map protocol request strings to event classes.
_REQUEST_EVENT_MAP = {
    "build": BuildRequested,
    "cancel_build": CancelBuildRequested,
    "start": StartRequested,
    "resume": ResumeRequested,
    "terminate": TerminateRequested,
    "save_and_quit": SaveAndQuitRequested,
    "delete": DeleteRequested,
}


def handle_text_command(conn, data, addr, engine):
    """Handle a text command request using the event-driven engine.

    Replaces the old ``process()`` function. Event dispatch + response
    generation are fast; status queries involve zero filesystem I/O.
    """
    try:
        args = easy_parse(data.split())
        name = args.get("name")
        if not name:
            response = {"error": "NO_ID", "protocol_version": PROTOCOL_VERSION}
            conn.sendall(json.dumps(response).encode())
            return

        root = make_root(name)

        # Set project context
        engine.select_project(name, root)

        # Dispatch request event if present
        request = args.get("request")
        if request:
            logging.info(f"=== request from name: {name} ({addr}) ===")
            logging.info(f"args: {args}")
            event_cls = _REQUEST_EVENT_MAP.get(request)
            if event_cls:
                engine.dispatch(event_cls())
            logging.info("=== request handled ===")
        else:
            logging.info(f"ping from ({addr})")

        # Generate and send response (reads cached state, zero I/O)
        response = engine.make_response()
        # Log build state for debugging
        s = engine.state
        logging.info(f"[STATE] build={s.build} solver={s.solver} bp={s.build_progress:.2f} bi={s.build_info!r}")
        response["stdout"] = ""
        response["stderr"] = ""
        conn.sendall(json.dumps(response).encode())

    except Exception as e:
        import traceback
        logging.error(f"Processing error: {e}\n{traceback.format_exc()}")
        try:
            # Error-only response contract (protocol 0.03):
            #   * protocol_version and upload_id are always present so the
            #     client's strict field checks pass,
            #   * ``error`` carries the human-readable message,
            #   * ``status`` is the empty string — the client recognises
            #     ``error != "" AND status == ""`` as "server raised, no
            #     status update" and preserves its current solver/activity.
            # See blender_addon/core/transitions.py:_interpret_response for
            # the client-side handler.
            conn.sendall(json.dumps({
                "error": str(e),
                "protocol_version": PROTOCOL_VERSION,
                "upload_id": "",
                "status": "",
            }).encode())
        except Exception:
            logging.error(f"Failed to send error response to {addr}")


# ---------------------------------------------------------------------------
# Connection handler (runs in its own thread)
# ---------------------------------------------------------------------------

def handle_connection(conn, addr, engine):
    """Handle a single client connection."""
    with conn:
        try:
            conn.settimeout(30.0)
            header = read_protocol_header(conn, addr)
            if not header:
                return

            print(f"Received protocol header: {header} from {addr}")

            if header == HEADER_JSON_DATA:
                result = handle_json_data(conn, addr, engine=engine)
                if result:
                    print(f"JSON data handling completed for {addr}")

            elif header == HEADER_BINARY_DATA:
                handle_binary_data(conn, addr)

            elif header == HEADER_TEXT_CMD:
                remaining_data = b""
                while True:
                    try:
                        data_chunk = conn.recv(4096)
                        if not data_chunk:
                            break
                        remaining_data += data_chunk
                    except TimeoutError:
                        print(f"Timeout reading text command from {addr}")
                        break
                    except (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError) as e:
                        print(f"Connection error from {addr}: {e}")
                        break

                if remaining_data:
                    try:
                        text = remaining_data.decode()
                        print(f"=== received text command from ({addr}) ===")
                        handle_text_command(conn, text, addr, engine)
                    except UnicodeDecodeError as e:
                        print(f"Unicode decode error from {addr}: {e}")
                        conn.sendall(json.dumps(
                            {"error": f"Text decode error: {e}"}
                        ).encode())
            else:
                print(f"Unknown protocol header {header} from {addr}")
                try:
                    conn.sendall(json.dumps(
                        {"error": f"Unknown header: {header.hex()}"}
                    ).encode())
                except Exception:
                    pass

        except (TimeoutError, ConnectionResetError,
                ConnectionAbortedError, BrokenPipeError) as e:
            print(f"Connection error with client {addr}: {e}")
        except Exception as e:
            print(f"Unexpected error handling client {addr}: {e}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def print_to_logging(*args, **__kwargs__):
    """Module-level print replacement so numba/pickle can resolve it by name."""
    message = " ".join(str(arg) for arg in args)
    logging.info(message)


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[logging.FileHandler(log_file_path)],
    )
    builtins.print = print_to_logging


# ---------------------------------------------------------------------------
# Progress file helpers
# ---------------------------------------------------------------------------

def write_progress(message, progress_file="progress.log"):
    with open(progress_file, "a") as f:
        f.write(f"{message}\n")


def clear_progress(progress_file="progress.log"):
    if os.path.exists(progress_file):
        os.remove(progress_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def signal_handler(sig, frame):
    del sig, frame
    print("\nReceived interrupt signal, shutting down gracefully...")
    sys.exit(0)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="ZOZO's Contact Solver Server")
    parser.add_argument("--port", type=int, default=9090,
                        help="Port to listen on (default: 9090)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind on (default: 127.0.0.1, "
                             "loopback only). Pass 0.0.0.0 when running "
                             "inside a Docker container so published-port "
                             "traffic (-p HOST:CONTAINER) can reach the "
                             "server through the container's external "
                             "interface.")
    parser.add_argument("--debug", action="store_true",
                        help="Run with the in-process emulator instead of "
                             "the real CUDA backend. Replaces frontend.Utils "
                             "and frontend.BlenderApp with stand-ins; every "
                             "wire-format obligation (BuildProgress, vert_*.bin, "
                             "finished.txt, save_*.bin, app_state.pickle) is "
                             "honored by the fake. See server/emulator.py.")
    return parser.parse_args()


if __name__ == "__main__":
    PROGRESS_FILE = "progress.log"
    clear_progress(PROGRESS_FILE)
    write_progress("SERVER_STARTING", PROGRESS_FILE)

    args = parse_args()
    HOST = args.host
    PORT = args.port
    write_progress(f"PARSING_ARGS port={PORT}", PROGRESS_FILE)

    setup_logging("server.log")
    write_progress("LOGGING_SETUP", PROGRESS_FILE)

    if args.debug:
        # Inject before any code path imports `frontend`. The executor
        # and monitor both do `from frontend import ...` lazily, so as
        # long as our fake module is in sys.modules first, they bind
        # to it transparently.
        from server.emulator import install as install_emulator
        install_emulator()
        write_progress("EMULATOR_INSTALLED", PROGRESS_FILE)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    write_progress("SIGNAL_HANDLERS_SET", PROGRESS_FILE)

    # Create the engine with hardware/git info
    engine = ServerEngine(
        hardware_info=_get_hardware_info(),
        git_branch=get_git_branch(),
    )

    # Start background solver monitor
    start_solver_monitor(engine)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    write_progress("SOCKET_CREATED", PROGRESS_FILE)

    try:
        write_progress(f"BINDING port={PORT}", PROGRESS_FILE)
        s.bind((HOST, PORT))
        s.listen()
        write_progress("SERVER_READY", PROGRESS_FILE)
        print(f"**** server started at {HOST}:{PORT} ****")

        # Threaded accept loop: each connection gets its own thread
        while True:
            conn, addr = s.accept()
            addr = addr[0]
            print("established connection from", addr)
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, engine),
                daemon=True,
            ).start()

    except KeyboardInterrupt:
        print("\nserver interrupted by user; shutting down...")
    finally:
        s.close()
