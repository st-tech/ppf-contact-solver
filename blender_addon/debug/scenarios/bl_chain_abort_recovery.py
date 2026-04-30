# Abort cycles: confirm the addon and server come back to a usable
# state after aborts at different stages (mid-build and during run).

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "abort_midbuild", "verify_idle",
    "transfer", "verify_idle",
    "run", "fetch", "verify_pc2",
    "clear_animation",
    "abort_during_run", "verify_idle",
    "run", "fetch", "verify_pc2",
    "abort_midbuild", "verify_idle",
    "transfer", "verify_idle",
    "run", "fetch", "verify_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_abort_recovery",
        mesh_name="ChainAbortMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
