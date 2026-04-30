# Param-edit cycle: catch param-hash drift / re-encode bugs across
# multiple update_params cycles. Each param_retransfer mutates the
# pin op's delta; the simulation result must reflect the latest
# value (verified by fetch + PC2 inspection in the chain harness).

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "run", "fetch", "verify_pc2", "clear_animation",
    "param_retransfer", "run", "fetch", "verify_pc2", "clear_animation",
    "param_retransfer", "run", "fetch", "verify_pc2", "clear_animation",
    "param_retransfer", "run", "fetch", "verify_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_param_repeat",
        mesh_name="ChainParamMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
