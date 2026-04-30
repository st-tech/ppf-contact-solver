# Data-evolution chain: each ``data_retransfer`` adds a vertex
# (topology change) and re-uploads. Catches missed mesh-hash
# invalidations and stale runner._fetched lists across rebuilds.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "run", "fetch", "verify_pc2", "clear_animation",
    "data_retransfer", "run", "fetch", "verify_pc2", "clear_animation",
    "data_retransfer", "run", "fetch", "verify_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_data_evolution",
        mesh_name="ChainDataMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
