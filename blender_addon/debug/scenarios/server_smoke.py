# File: scenarios/server_smoke.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Smallest end-to-end exercise: a debug server is up; a client connects;
# the engine reports the expected initial NO_DATA / NO_BUILD / IDLE state.

from __future__ import annotations

import importlib.util
import os

from . import _runner as r


# Server-only plumbing (no Blender, no physics); runs on the real-GPU
# jobs too, where it talks to the real ppf-cts-server.
BACKENDS = ("emulated", "real")


def _load_protocol_version() -> str:
    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "core", "protocol.py")
    )
    spec = importlib.util.spec_from_file_location("_protocol_for_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROTOCOL_VERSION


PROTOCOL_VERSION = _load_protocol_version()


def run(ctx: r.ScenarioContext) -> dict:
    client = r.ProtoClient(ctx.host, ctx.server_port, timeout=ctx.timeout)
    ctx.log(f"server_smoke: querying status of project {ctx.project_name!r}")

    response = client.text_cmd({"name": ctx.project_name})

    violations: list[str] = []

    if response.get("protocol_version") != PROTOCOL_VERSION:
        violations.append(
            f"unexpected protocol_version: {response.get('protocol_version')!r}"
        )

    # A fresh project the server has never seen lands in NO_DATA: no
    # data.pickle on disk, no app loaded.
    status = response.get("status", "")
    data = response.get("data", "")
    if status != "NO_DATA":
        violations.append(f"expected status=NO_DATA, got {status!r}")
    if data != "NO_DATA":
        violations.append(f"expected data=NO_DATA, got {data!r}")

    if response.get("upload_id") != "":
        violations.append(
            f"expected empty upload_id on fresh project, got "
            f"{response.get('upload_id')!r}"
        )

    if violations:
        return r.failed(violations, notes=[f"raw response: {response}"])
    return r.passed(notes=["status reached NO_DATA on first ping"])
