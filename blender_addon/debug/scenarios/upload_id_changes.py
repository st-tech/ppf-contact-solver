# File: scenarios/upload_id_changes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Two atomic uploads back-to-back: the upload_id must change, but the
# server's existing build (if any) must remain valid for the *previous*
# upload_id until a fresh build is requested. Tests the
# upload_id-as-build-anchor contract that the addon's communicator pins
# to.

from __future__ import annotations

from . import _runner as r


def run(ctx: r.ScenarioContext) -> dict:
    client = r.ProtoClient(ctx.host, ctx.server_port, timeout=ctx.timeout)

    upload_a = client.upload_atomic(
        ctx.project_name,
        r.make_minimal_data(),
        r.make_minimal_param(frames=4, fps=10),
        ctx.project_root,
    )
    if not upload_a.get("ok"):
        return r.failed([f"first upload failed: {upload_a}"])

    s_a = client.text_cmd({"name": ctx.project_name})
    id_a = s_a.get("upload_id", "")
    if not id_a:
        return r.failed([f"first upload yielded no upload_id; status={s_a}"])

    upload_b = client.upload_atomic(
        ctx.project_name,
        r.make_minimal_data(),
        r.make_minimal_param(frames=4, fps=10),
        ctx.project_root,
    )
    if not upload_b.get("ok"):
        return r.failed([f"second upload failed: {upload_b}"])

    s_b = client.text_cmd({"name": ctx.project_name})
    id_b = s_b.get("upload_id", "")
    if not id_b:
        return r.failed([f"second upload yielded no upload_id; status={s_b}"])

    violations: list[str] = []
    if id_a == id_b:
        violations.append(
            f"upload_id did not change across two atomic uploads: {id_a!r}"
        )
    if len(id_b) != len(id_a):
        # Format invariant: 12 hex chars per server.engine._new_upload_id.
        violations.append(
            f"upload_id length changed: {len(id_a)} -> {len(id_b)}"
        )

    # Status after second upload must be NO_BUILD: the prior build (if
    # any) is no longer valid for the new upload, so the client knows
    # it has to re-trigger build.
    if s_b.get("status") != "NO_BUILD":
        violations.append(
            f"status after re-upload should be NO_BUILD, got "
            f"{s_b.get('status')!r}"
        )

    if violations:
        return r.failed(violations)
    return r.passed(notes=[f"id_a={id_a} id_b={id_b}"])
