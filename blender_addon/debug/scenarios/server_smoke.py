# File: scenarios/server_smoke.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Smallest end-to-end exercise: a debug server is up; a client connects;
# the engine reports the expected initial NO_DATA / NO_BUILD / IDLE state.

from __future__ import annotations

from . import _runner as r


def run(ctx: r.ScenarioContext) -> dict:
    client = r.ProtoClient(ctx.host, ctx.server_port, timeout=ctx.timeout)
    ctx.log(f"server_smoke: querying status of project {ctx.project_name!r}")

    response = client.text_cmd({"name": ctx.project_name})

    violations: list[str] = []

    if response.get("protocol_version") not in ("0.02", "0.03"):
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
