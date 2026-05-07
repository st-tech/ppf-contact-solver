# File: scenarios/bl_rust_binary_protocol.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Lock in the launcher to Rust-binary handshake. Distinct from
# bl_connect_local in that it (a) asserts the binary on disk is the
# Rust port (target/release/ppf-cts-server, not the retired server.py)
# and (b) reads the binary's --version output to verify the reported
# PROTOCOL_VERSION matches what the addon negotiates against. The
# in-Blender driver still goes through the LOCAL connect path so the
# whole chain (launcher exec, TCP handshake, addon transition) is
# exercised.

from __future__ import annotations

import os
import re
import subprocess

from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Linux runs the Rust binary directly via the LOCAL backend. macOS
# usually goes through SSH/Docker; Windows uses bl_connect_win_native
# which probes ppf-cts-server.exe at a different code path.
PLATFORMS = ("linux",)

# Pinned in crates/ppf-cts-server/src/lib.rs::PROTOCOL_VERSION. If a
# future schema bump lands, update both places in lockstep.
EXPECTED_PROTOCOL = "0.04"


_DRIVER_TEMPLATE = """
import sys, time, traceback
try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    root = groups.get_addon_data(bpy.context.scene)
    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = <<LOCAL_PATH_REPR>>
    root.ssh_state.docker_port = <<SERVER_PORT>>

    com = client.communicator
    com.connect_local(root.ssh_state.local_path,
                      server_port=root.ssh_state.docker_port)

    deadline = time.time() + 20.0
    while time.time() < deadline:
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE":
            break
        time.sleep(0.2)

    s = facade.engine.state
    result["phase"] = s.phase.name
    result["server"] = s.server.name
    result["solver"] = s.solver.name
    result["connected"] = bool(com.is_connected())
    result["version_ok"] = bool(s.version_ok)
except Exception as exc:
    result["errors"].append(f"driver: {type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH_REPR>>", repr(repo_root))
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def _binary_path() -> str:
    """Resolve the production server binary on this host."""
    name = "ppf-cts-server.exe" if os.name == "nt" else "ppf-cts-server"
    return os.path.join(REPO_ROOT_POSIX, "target", "release", name)


def _probe_protocol_version(binary: str, timeout: float = 10.0) -> str | None:
    """Run ``ppf-cts-server --version`` and extract the protocol token.

    The clap-generated output is a single line shaped
    ``ppf-cts-server 0.1.0 (protocol v0.03, schema v1)``. We grep for
    the ``protocol v...`` substring rather than parsing positionally so
    a future version-line tweak doesn't false-fail.
    """
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    blob = (out.stdout or "") + (out.stderr or "")
    m = re.search(r"protocol\s+v?([\w.\-]+)", blob)
    return m.group(1) if m else None


def run(ctx: r.ScenarioContext) -> dict:
    import blender_harness as bh

    binary = _binary_path()
    if not os.path.isfile(binary):
        return r.failed(
            [f"Rust solver binary not built: {binary}"],
            notes=[
                "build with `cargo build --release -p ppf-cts-server` "
                "before running this scenario",
            ],
        )

    reported = _probe_protocol_version(binary)
    if reported is None:
        return r.failed(
            [f"could not parse `{binary} --version` output for "
             f"protocol token"],
        )
    if reported != EXPECTED_PROTOCOL:
        return r.failed(
            [f"binary reports protocol v{reported}; "
             f"expected v{EXPECTED_PROTOCOL}"],
            notes=["update EXPECTED_PROTOCOL or revert the version bump"],
        )

    bspec = ctx.artifacts.get("blender_spec")
    proc = ctx.artifacts.get("blender_proc")
    if bspec is None or proc is None:
        return r.failed(["no Blender process attached to context"])

    try:
        result = bh.wait_for_result(bspec, proc, timeout=max(ctx.timeout, 90.0))
    except TimeoutError as e:
        return r.failed(
            [str(e)],
            notes=[
                f"stdout (tail): {open(bspec.stdout_path).read()[-1500:]!r}",
                f"stderr (tail): {open(bspec.stderr_path).read()[-1500:]!r}",
            ],
        )

    violations: list[str] = list(result.get("errors") or [])
    if not result.get("scenario_done"):
        violations.append("driver did not run to completion")
    if not result.get("connected"):
        violations.append("addon never reached connected=True")
    if result.get("phase") != "ONLINE":
        violations.append(
            f"phase did not reach ONLINE: {result.get('phase')!r}"
        )
    if not result.get("version_ok"):
        violations.append(
            "addon set version_ok=False: protocol negotiation rejected the "
            "server response (binary reported "
            f"v{reported}, addon expected v{EXPECTED_PROTOCOL})"
        )

    notes = [
        f"binary={binary} (protocol v{reported})",
        f"phase={result.get('phase')}, server={result.get('server')}, "
        f"solver={result.get('solver')}, version_ok={result.get('version_ok')}",
    ]
    if violations:
        return r.failed(violations, notes=notes)
    return r.passed(notes=notes)
