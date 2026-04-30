# File: reload_server.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import contextlib
import importlib
import json
import socket
import sys
import threading
import traceback

from typing import Any, Optional

import bpy  # pyright: ignore

from ..models.defaults import DEFAULT_MCP_PORT, DEFAULT_RELOAD_PORT

_reload_server_instance: Optional["ReloadServer"] = None


def _cleanup_stale_servers(port: int) -> None:
    """Find and stop any orphaned ReloadServer instances holding the port."""
    import gc
    gc.collect()
    for obj in gc.get_objects():
        try:
            if (type(obj).__name__ == "ReloadServer"
                    and getattr(obj, "socket", None) is not None
                    and getattr(obj, "port", None) == port):
                obj.running = False
                with contextlib.suppress(OSError):
                    obj.socket.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(OSError):
                    obj.socket.close()
                obj.socket = None
                if getattr(obj, "thread", None):
                    obj.thread.join(timeout=1.0)
                    obj.thread = None
        except Exception:
            pass


class ReloadServer:
    def __init__(self, port: int = DEFAULT_RELOAD_PORT):
        self.port = port
        self.socket = None
        self.running = False
        self.thread = None
        # Track one-shot main-thread timers scheduled by _dispatch so
        # ``stop()`` can cancel any that haven't fired yet. If we don't,
        # a pending reload phase fires into a half-torn-down module.
        self._pending_timers: list = []

    def _schedule(self, fn, first_interval: float) -> None:
        """Schedule a main-thread timer and remember it for stop()."""
        bpy.app.timers.register(fn, first_interval=first_interval)
        self._pending_timers.append(fn)

    def start(self):
        """Start TCP listener for reload packets and Python code execution"""
        global _reload_server_instance

        if self.running:
            return

        # Kill any stale instances left from a previous module reload
        _cleanup_stale_servers(self.port)

        if self.socket:
            with contextlib.suppress(OSError):
                self.socket.close()
            self.socket = None

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("localhost", self.port))
            self.socket.listen(4)
            self.socket.settimeout(1.0)
            self.running = True

            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()

            _reload_server_instance = self
            print(f"Reload server listening on TCP port {self.port}")

        except Exception as e:
            print(f"Failed to start reload server: {e}")
            self.running = False
            if self.socket:
                with contextlib.suppress(OSError):
                    self.socket.close()
                self.socket = None

    def stop(self):
        """Stop the reload server"""
        global _reload_server_instance

        self.running = False
        # Cancel any unfired reload-phase timers so they can't call into
        # an unregistered/freed module state after stop() returns.
        for fn in self._pending_timers:
            if bpy.app.timers.is_registered(fn):
                try:
                    bpy.app.timers.unregister(fn)
                except ValueError:
                    pass
        self._pending_timers.clear()
        if self.socket:
            with contextlib.suppress(OSError):
                self.socket.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                self.socket.close()
            self.socket = None
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        _reload_server_instance = None
        print("Reload server stopped")

    def _listen(self):
        """Accept TCP connections"""
        while self.running:
            try:
                if self.socket is None:
                    break
                try:
                    conn, addr = self.socket.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr),
                    daemon=True,
                ).start()
            except Exception as e:
                if self.running:
                    print(f"Reload server error: {e}")

    def _handle_connection(self, conn: socket.socket, addr):
        """Handle a single TCP connection: read request, dispatch, send reply."""
        try:
            conn.settimeout(10.0)
            chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                # Try to parse — if valid JSON we have the full message
                try:
                    json.loads(b"".join(chunks))
                    break
                except json.JSONDecodeError:
                    continue

            raw = b"".join(chunks).decode("utf-8")
            if not raw:
                return

            try:
                packet = json.loads(raw)
            except json.JSONDecodeError:
                self._send(conn, {"error": f"Invalid JSON: {raw[:200]}"})
                return

            if not isinstance(packet, dict) or "command" not in packet:
                self._send(conn, {"error": "Missing 'command' key"})
                return

            print(f"Debug server: {packet.get('command')} from {addr}")
            self._dispatch(packet, conn)

        except Exception as e:
            if self.running:
                print(f"Connection handler error: {e}")
            self._send(conn, {"error": str(e)})
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def _send(self, conn: socket.socket, message: dict):
        """Send a JSON response over TCP."""
        try:
            data = json.dumps(message).encode("utf-8")
            conn.sendall(data)
        except Exception:
            pass

    def _dispatch(self, packet: dict[str, Any], conn: socket.socket):
        """Route command to handler."""
        command = packet.get("command")

        if command == "reload":
            # Run on main thread and wait so we can report real errors
            # rather than blindly returning "ok" before perform_reload runs.
            result_holder: dict[str, Any] = {}
            done = threading.Event()

            def _run_reload():
                try:
                    self.perform_reload()
                    result_holder["status"] = "ok"
                except Exception as exc:
                    result_holder["status"] = "error"
                    result_holder["error"] = str(exc)
                    traceback.print_exc()
                done.set()

            self._schedule(_run_reload, 0.1)
            if done.wait(timeout=30.0):
                self._send(conn, {**result_holder, "command": "reload"})
            else:
                self._send(conn, {"status": "error", "error": "Reload timed out"})

        elif command == "execute":
            python_code = packet.get("code", "")
            if not python_code:
                self._send(conn, {"error": "No code provided"})
                return
            exec_timeout = packet.get("timeout", 30.0)
            result_holder = {}
            done = threading.Event()

            def _run():
                result_holder.update(self._execute_python_code(python_code))
                done.set()

            self._schedule(_run, 0.1)
            done.wait(timeout=exec_timeout)
            if done.is_set():
                self._send(conn, result_holder)
            else:
                self._send(conn, {"error": f"Execution timed out ({exec_timeout}s)"})

        elif command == "start_mcp":
            port = packet.get("port", DEFAULT_MCP_PORT)
            result_holder = {}
            done = threading.Event()

            def _run():
                result_holder.update(self._start_mcp_server(port))
                done.set()

            self._schedule(_run, 0.1)
            done.wait(timeout=10.0)
            if done.is_set():
                self._send(conn, result_holder)
            else:
                self._send(conn, {"error": "MCP start timed out"})

        elif command == "full_reload":
            # "Full reload" splits disable and enable across two event-loop
            # ticks so Blender fully drops RNA for classes with nested
            # CollectionProperty bindings before the new register() runs.
            # The regular `reload` path does both steps inline in the same
            # timer callback, which silently fails to refresh nested
            # PropertyGroup RNA in some cases (e.g. InvisibleColliderItem
            # with its keyframes CollectionProperty).
            result_holder: dict[str, Any] = {}
            ctx_holder: list[Any] = [None]
            done = threading.Event()

            def _full_phase2():
                try:
                    self._reload_phase2_enable(ctx_holder[0])
                    result_holder["status"] = "ok"
                except Exception as exc:
                    result_holder["status"] = "error"
                    result_holder["error"] = f"enable: {exc}"
                    traceback.print_exc()
                done.set()
                return None

            def _full_phase1():
                try:
                    ctx_holder[0] = self._reload_phase1_disable()
                except Exception as exc:
                    result_holder["status"] = "error"
                    result_holder["error"] = f"disable: {exc}"
                    traceback.print_exc()
                    done.set()
                    return None
                # Yield back to Blender for one full event-loop iteration
                # before re-registering; that's where RNA cleanup happens.
                self._schedule(_full_phase2, 0.3)
                return None

            self._schedule(_full_phase1, 0.1)
            if done.wait(timeout=60.0):
                self._send(conn, {**result_holder, "command": "full_reload"})
            else:
                self._send(
                    conn, {"status": "error", "error": "Full reload timed out"}
                )

        elif command == "ping":
            self._send(conn, {"status": "ok", "command": "ping"})

        else:
            self._send(conn, {"error": f"Unknown command: {command}"})

    def perform_reload(self):
        """Perform add-on reload using license-safe custom approach.

        Correctly handles the case where the addon is loaded under BOTH the
        legacy top-level name (``ppf_contact_solver``) and the Blender 5.x
        extension alias (``bl_ext.<vendor>.ppf_contact_solver``). Every
        matching ``sys.modules`` entry is deleted and importer caches are
        invalidated before re-enable so fresh source is executed.
        """
        current_module = __name__
        parts = current_module.split(".")

        # Derive the short addon id (leaf package name), independent of
        # whether the addon is loaded legacy-style (``ppf_contact_solver.*``)
        # or via the extension system (``bl_ext.<vendor>.ppf_contact_solver.*``).
        if parts[0] == "bl_ext" and len(parts) >= 3:
            short_addon_id = parts[2]
            bl_ext_prefix = ".".join(parts[:3])
        else:
            short_addon_id = parts[0]
            bl_ext_prefix = None

        # Collect every plausible addon-module name.
        possible_addon_names: set[str] = {short_addon_id}
        if bl_ext_prefix:
            possible_addon_names.add(bl_ext_prefix)

        # Also accept whatever Blender currently lists in preferences that
        # *ends* with the short addon id, so the extension alias is picked
        # up even when the reload server itself is imported via the legacy
        # path (or vice versa).
        enabled_modules = [a.module for a in bpy.context.preferences.addons]
        for mod in enabled_modules:
            if mod == short_addon_id or mod.endswith("." + short_addon_id):
                possible_addon_names.add(mod)

        # Pick the name that Blender actually has registered. Prefer the
        # longest (most specific) match.
        actual_addon_name = None
        for candidate in sorted(possible_addon_names, key=len, reverse=True):
            if candidate in enabled_modules:
                actual_addon_name = candidate
                break
        if not actual_addon_name:
            actual_addon_name = short_addon_id

        # Prefixes we must purge from sys.modules. Includes the resolved
        # addon name PLUS any alias seen in sys.modules so both legacy and
        # bl_ext namespaces are flushed together.
        purge_prefixes: set[str] = {actual_addon_name}
        for mod_name in list(sys.modules):
            mod_parts = mod_name.split(".")
            if mod_parts[0] == short_addon_id:
                purge_prefixes.add(short_addon_id)
            elif (
                mod_parts[0] == "bl_ext"
                and len(mod_parts) >= 3
                and mod_parts[2] == short_addon_id
            ):
                purge_prefixes.add(".".join(mod_parts[:3]))

        try:
            ctx = self._reload_phase1_disable(
                short_addon_id=short_addon_id,
                possible_addon_names=possible_addon_names,
                actual_addon_name=actual_addon_name,
                purge_prefixes=purge_prefixes,
            )
            self._reload_phase2_enable(ctx)
        except Exception as e:
            print(f"Reload failed: {e}")
            traceback.print_exc()

    # -- full-reload phases ------------------------------------------------
    # The fast `perform_reload` runs both phases in the same timer tick.
    # `perform_full_reload` (driven by the dispatcher for the "full_reload"
    # command) splits them across two ticks so Blender's event loop can run
    # RNA cleanup between disable and enable.

    def _reload_phase1_disable(
        self,
        short_addon_id: str | None = None,
        possible_addon_names: set | None = None,
        actual_addon_name: str | None = None,
        purge_prefixes: set | None = None,
    ) -> dict:
        """Unregister the addon and purge its modules. Returns a context
        dict consumed by _reload_phase2_enable."""
        # When called from perform_full_reload we don't get the precomputed
        # names; derive them here so the method stands alone.
        if actual_addon_name is None:
            current_module = __name__
            parts = current_module.split(".")
            if parts[0] == "bl_ext" and len(parts) >= 3:
                short_addon_id = parts[2]
                bl_ext_prefix = ".".join(parts[:3])
            else:
                short_addon_id = parts[0]
                bl_ext_prefix = None
            possible_addon_names = {short_addon_id}
            if bl_ext_prefix:
                possible_addon_names.add(bl_ext_prefix)
            enabled_modules = [a.module for a in bpy.context.preferences.addons]
            for mod in enabled_modules:
                if mod == short_addon_id or mod.endswith("." + short_addon_id):
                    possible_addon_names.add(mod)
            actual_addon_name = None
            for candidate in sorted(possible_addon_names, key=len, reverse=True):
                if candidate in enabled_modules:
                    actual_addon_name = candidate
                    break
            if not actual_addon_name:
                actual_addon_name = short_addon_id
            purge_prefixes = {actual_addon_name}
            for mod_name in list(sys.modules):
                mod_parts = mod_name.split(".")
                if mod_parts[0] == short_addon_id:
                    purge_prefixes.add(short_addon_id)
                elif (
                    mod_parts[0] == "bl_ext"
                    and len(mod_parts) >= 3
                    and mod_parts[2] == short_addon_id
                ):
                    purge_prefixes.add(".".join(mod_parts[:3]))

        # Save which servers are running so register() can restart them
        _save_server_state_before_reload()

        print(f"Reloading add-on: {actual_addon_name}")
        disable_result = bpy.ops.preferences.addon_disable(
            module=actual_addon_name
        )
        if "CANCELLED" in disable_result:
            print(
                f"addon_disable returned CANCELLED for "
                f"{actual_addon_name!r}; trying fallbacks"
            )
            for candidate in sorted(
                possible_addon_names, key=len, reverse=True
            ):
                if candidate == actual_addon_name:
                    continue
                try:
                    res = bpy.ops.preferences.addon_disable(module=candidate)
                except Exception:
                    continue
                if "CANCELLED" not in res:
                    actual_addon_name = candidate
                    purge_prefixes.add(candidate)
                    break

        # Snapshot the keys so we never mutate sys.modules while iterating.
        addon_module_names = [
            name
            for name in list(sys.modules)
            if any(
                name == p or name.startswith(p + ".") for p in purge_prefixes
            )
        ]
        # Delete deepest first so parents see all children already gone.
        addon_module_names.sort(key=lambda x: x.count("."), reverse=True)

        invalidated_modules = []
        for module_name in addon_module_names:
            if module_name in sys.modules:
                invalidated_modules.append(module_name)
                del sys.modules[module_name]

        # Drop finder caches so __pycache__ lookups re-stat source files.
        # Note: we deliberately do not delete .pyc files here — doing so
        # raced with Blender's OpenEXR thread pool and crashed the process.
        # If stale bytecode is suspected, delete __pycache__ manually and
        # restart Blender.
        importlib.invalidate_caches()

        print(
            f"Invalidated {len(invalidated_modules)} modules "
            f"(prefixes: {sorted(purge_prefixes)})"
        )
        return {"actual_addon_name": actual_addon_name}

    def _reload_phase2_enable(self, ctx: dict) -> None:
        """Re-enable the addon after phase1. `ctx` carries the resolved
        addon module name."""
        bpy.ops.preferences.addon_enable(module=ctx["actual_addon_name"])
        print(f"Successfully reloaded {ctx['actual_addon_name']}")

    def _execute_python_code(self, code: str) -> dict:
        """Execute arbitrary Python code in Blender context and return output."""
        import builtins
        import io

        output_buf = io.StringIO()

        def _capture_print(*args, **kwargs):
            builtins.print(*args, **kwargs)
            kwargs.pop("file", None)
            builtins.print(*args, file=output_buf, **kwargs)

        exec_globals = {
            "bpy": bpy,
            "__builtins__": builtins,
            "print": _capture_print,
        }
        try:
            exec(code, exec_globals)
            return {"status": "ok", "output": output_buf.getvalue()}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e), "output": output_buf.getvalue()}

    def _start_mcp_server(self, port: int = DEFAULT_MCP_PORT) -> dict:
        """Start the MCP server, return status dict."""
        try:
            from ..mcp.mcp_server import is_mcp_running, start_mcp_server

            if is_mcp_running():
                return {"status": "ok", "message": "MCP server already running"}

            start_mcp_server(port)

            if is_mcp_running():
                return {"status": "ok", "message": f"MCP server started on port {port}"}
            else:
                return {"status": "error", "error": "MCP server failed to start"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}


# Utility functions for UI integration
def get_reload_server_status() -> bool:
    """Check if reload server is running"""
    return _reload_server_instance is not None and _reload_server_instance.running


def start_reload_server(port: int = DEFAULT_RELOAD_PORT):
    """Start the reload server"""
    global _reload_server_instance
    if _reload_server_instance and _reload_server_instance.running:
        _reload_server_instance.stop()
    server = ReloadServer(port)
    server.start()


def stop_reload_server():
    """Stop the reload server"""
    if _reload_server_instance:
        _reload_server_instance.stop()


def trigger_reload_now():
    """Trigger an addon reload via the reload server instance.

    Schedules through bpy.app.timers so the calling operator's execute()
    unwinds before addon_disable tears down its class — running
    perform_reload inline from a UI button frees the operator while its
    Python frame is still on the stack, which crashes Blender."""
    if not (_reload_server_instance and _reload_server_instance.running):
        raise RuntimeError("Reload server is not running")

    server = _reload_server_instance

    def _run():
        try:
            server.perform_reload()
        except Exception as exc:
            print(f"Reload failed: {exc}")
            traceback.print_exc()
        return None

    server._schedule(_run, 0.1)


def trigger_full_reload_now():
    """Trigger a full (two-phase) addon reload. Splits disable/enable across
    two event-loop ticks so Blender rebuilds RNA for nested
    CollectionProperty schemas. Schedules via bpy.app.timers so the button
    click returns immediately."""
    if not (_reload_server_instance and _reload_server_instance.running):
        raise RuntimeError("Reload server is not running")

    server = _reload_server_instance
    ctx_holder: list = [None]

    def _phase2():
        try:
            server._reload_phase2_enable(ctx_holder[0])
        except Exception as exc:
            print(f"Full reload phase2 failed: {exc}")
            traceback.print_exc()
        return None

    def _phase1():
        try:
            ctx_holder[0] = server._reload_phase1_disable()
        except Exception as exc:
            print(f"Full reload phase1 failed: {exc}")
            traceback.print_exc()
            return None
        server._schedule(_phase2, 0.3)
        return None

    server._schedule(_phase1, 0.1)




_RESTART_STATE_KEY = "_ppf_restart_servers"


def _save_server_state_before_reload():
    """Save which servers are running so register() can restore them.
    Uses bpy.app.driver_namespace (a plain dict designed for cross-reload
    state) — Blender rejects arbitrary attribute writes on bpy.app itself."""
    state = {"debug": get_reload_server_status()}
    try:
        from ..mcp.mcp_server import is_mcp_running
        state["mcp"] = is_mcp_running()
    except Exception:
        state["mcp"] = False
    bpy.app.driver_namespace[_RESTART_STATE_KEY] = state


def get_restart_server_state():
    """Pop saved server state (if any). Returns dict or None."""
    return bpy.app.driver_namespace.pop(_RESTART_STATE_KEY, None)
