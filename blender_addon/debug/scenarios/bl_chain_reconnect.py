# Reconnect cycles: assert the addon/server can survive multiple
# disconnect/reconnect rounds without losing state-on-disk hashes,
# and that runs continue to work each time.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "run", "fetch", "verify_pc2",
    "disconnect", "reconnect", "verify_idle",
    "run", "fetch", "verify_pc2",
    "disconnect", "reconnect", "verify_idle",
    "clear_animation", "verify_no_pc2",
    "run", "fetch", "verify_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_reconnect",
        mesh_name="ChainReconnectMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
