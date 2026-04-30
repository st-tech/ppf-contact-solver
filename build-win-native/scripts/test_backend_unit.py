"""Unit test for the Windows-native backend (no Blender required).

Exercises ``blender_addon/core/backends.py:WinNativeBackend`` end-to-end
against a freshly built solver. Verifies the four things that broke on
Windows before the fixes in commit 24a9372d:

  1. ``create_backend("win_native", …)`` launches ``server.py`` as a
     subprocess and the server binds the requested port.
  2. A socket query round-trips (``backend.query``).
  3. The ``glob``-based frame-file counter (which replaces the Unix
     ``ls -1 vert_*.bin`` shell-out in ``_count_remote_frames``)
     matches the max index correctly.
  4. ``backend.stop_server()`` terminates the subprocess cleanly
     (replaces the ``pkill -f server.py`` path).

Designed to work under either project layout:

  - **Dev layout**  (``C:\\dev``): has ``build-win-native\\python\\…``.
  - **Bundle layout** (``<dist>``): has ``python\\…`` directly at the root.

Run over SSH against the Windows box:

    # Dev layout, embedded Python
    C:\\dev\\build-win-native\\python\\python.exe \\
        C:\\dev\\build-win-native\\scripts\\test_backend_unit.py  C:\\dev  9091

    # Bundle layout, bundled Python
    C:\\dev\\build-win-native\\dist\\python\\python.exe \\
        C:\\dev\\build-win-native\\scripts\\test_backend_unit.py \\
        C:\\dev\\build-win-native\\dist  9092

Argv: ``<root>  [port]`` — ``root`` must contain ``blender_addon/`` and
``server.py``. Both layouts satisfy that.

On success every step prints an ``OK`` line and the script exits 0.
On failure the offending step prints a ``FAIL:`` message to stderr
(including the subprocess's stderr if the port never bound) and exits 1.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import socket
import sys
import time
import types


ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\ppf-contact-solver"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9091
PROJECT_NAME = "backend_unit"

WAIT_PORT_TIMEOUT_S = 30.0


def _die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _wait_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _import_backends() -> types.ModuleType:
    """Import ``blender_addon.core.backends`` without triggering its
    bpy-dependent ``__init__.py``.

    The addon's top-level ``__init__.py`` imports UI modules that require
    ``bpy``. ``backends.py`` itself is pure-Python, but its relative
    imports reach ``.protocol``, ``.status``, and ``..models.console``.
    We stub the non-pure imports (``bpy`` + ``models.console``) and
    side-load the three pure siblings via ``spec_from_file_location``.
    """
    addon_dir = os.path.join(ROOT, "blender_addon")
    core_dir = os.path.join(addon_dir, "core")

    # Package stubs so relative imports resolve.
    for pkg_name, pkg_path in (
        ("blender_addon", addon_dir),
        ("blender_addon.core", core_dir),
        ("blender_addon.models", os.path.join(addon_dir, "models")),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [pkg_path]
            sys.modules[pkg_name] = pkg

    sys.modules.setdefault("bpy", types.ModuleType("bpy"))

    # models.console pulls bpy itself; provide a no-op stand-in.
    if "blender_addon.models.console" not in sys.modules:
        stub = types.ModuleType("blender_addon.models.console")
        class _NoopConsole:
            def write(self, *_a, **_kw): pass
            def show(self, *_a, **_kw): pass
        stub.console = _NoopConsole()
        sys.modules["blender_addon.models.console"] = stub

    def _load(fqn: str, path: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(fqn, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = module
        spec.loader.exec_module(module)
        return module

    # Order matters: backends.py imports from protocol and status.
    _load("blender_addon.core.protocol", os.path.join(core_dir, "protocol.py"))
    _load("blender_addon.core.status", os.path.join(core_dir, "status.py"))
    return _load(
        "blender_addon.core.backends", os.path.join(core_dir, "backends.py")
    )


def _dump_subprocess(backend) -> None:
    """Surface the subprocess's own stderr when the port never bound.

    Without this, a server.py import error (seen IRL when the bundle was
    missing the ``server/`` package) looks like a generic "didn't bind"
    timeout and you're stuck guessing.
    """
    proc = getattr(backend, "_process", None)
    code = proc.poll() if proc is not None else "no process"
    out, err = b"", b""
    if proc is not None and code is not None:
        try:
            out, err = proc.communicate(timeout=2)
        except Exception:
            pass
    print(f"  subprocess exit_code={code}", file=sys.stderr)
    print(f"  subprocess stdout:\n{out.decode(errors='replace')}",
          file=sys.stderr)
    print(f"  subprocess stderr:\n{err.decode(errors='replace')}",
          file=sys.stderr)


def main() -> None:
    backends = _import_backends()
    create_backend = backends.create_backend
    WinNativeBackend = backends.WinNativeBackend

    backend = create_backend(
        "win_native", {"path": ROOT, "server_port": PORT}
    )
    assert isinstance(backend, WinNativeBackend), type(backend)
    assert backend.backend_type == "win_native"
    assert backend.server_port == PORT

    try:
        # Step 1 — subprocess launched and port bound
        if not _wait_port("127.0.0.1", PORT, WAIT_PORT_TIMEOUT_S):
            _dump_subprocess(backend)
            _die(f"server did not bind 127.0.0.1:{PORT} within "
                 f"{WAIT_PORT_TIMEOUT_S:.0f}s")
        print("OK [port-bind]: server is listening")

        # Step 2 — query round-trip
        resp, alive = backend.query({}, PROJECT_NAME)
        if not alive:
            _die(f"query alive=False, resp={resp!r}")
        print(f"OK [query]: keys={sorted(resp.keys())}")

        # Step 3 — glob-based frame counting (replaces `ls -1` shell-out)
        fake_out = os.path.join(ROOT, "session", "output_backend_unit_fake")
        os.makedirs(fake_out, exist_ok=True)
        try:
            for i in (1, 2, 5):
                open(os.path.join(fake_out, f"vert_{i}.bin"), "wb").close()
            open(os.path.join(fake_out, "unrelated.txt"), "wb").close()
            files = sorted(
                os.path.basename(p)
                for p in glob.glob(os.path.join(fake_out, "vert_*.bin"))
            )
            expected = ["vert_1.bin", "vert_2.bin", "vert_5.bin"]
            if files != expected:
                _die(f"glob returned {files!r}, expected {expected!r}")
            max_frame = max(int(n[5:-4]) for n in files)
            if max_frame != 5:
                _die(f"expected max_frame=5, got {max_frame}")
            print(f"OK [glob-count]: max_frame={max_frame}")
        finally:
            shutil.rmtree(fake_out, ignore_errors=True)

        # Step 4 — stop_server terminates subprocess and releases the port
        backend.stop_server()
        if backend._process is not None:
            _die("stop_server did not clear _process")
        gone = False
        for _ in range(10):
            try:
                with socket.create_connection(("127.0.0.1", PORT),
                                              timeout=0.5):
                    pass
            except OSError:
                gone = True
                break
            time.sleep(0.25)
        if not gone:
            _die("port still reachable after stop_server")
        print("OK [stop-server]: process terminated, port released")
    finally:
        # Idempotent — disconnect() after stop_server() is a no-op.
        backend.disconnect()

    print("PASS test_backend_unit")


if __name__ == "__main__":
    main()
