# Lifecycle chain: connect → transfer → run → fetch → clear → rerun
# → fetch → clear. Targets stale state in the run/fetch/clear cycle
# that the matrix scenarios miss because they only do one cycle.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "run", "fetch", "verify_pc2",
    "clear_animation", "verify_no_pc2",
    "run", "fetch", "verify_pc2",
    "clear_animation", "verify_no_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_lifecycle",
        mesh_name="ChainLifecycleMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
