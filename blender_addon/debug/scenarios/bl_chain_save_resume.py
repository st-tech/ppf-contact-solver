# Save/resume chain: catch state leak across the save_and_quit /
# resume boundary. Save mid-run, resume, fetch, clear, run again.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "save_and_quit", "verify_resumable",
    "resume", "fetch", "verify_pc2",
    "clear_animation", "verify_no_pc2",
    "run", "fetch", "verify_pc2",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_save_resume",
        mesh_name="ChainSaveResumeMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
