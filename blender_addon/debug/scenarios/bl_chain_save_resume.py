# Save/resume chain: catch state leak across the save_and_quit /
# resume boundary. Save mid-run, resume, fetch, clear, run again.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

# AN OBSERVATION WINDOW, for the same reason the resume scenarios ask for
# one. This scenario samples a solve WHILE IT RUNS, and a real backend
# finishes a scene this size faster than the addon polls: the failure is
# `saw_running: false`, or a mid-run step finding the state already
# READY, which reads like a broken transition and is a race lost. The
# delay is per STEP, so the run lasts substeps times this; what has to be
# long is the RUN, not any budget.
KNOBS = {"PPF_STEP_DELAY_MS": "1000"}

# RUNS ON THE REAL BACKEND, established by RUNNING it. The sweep that had
# failed it loaded a DIFFERENT tree's addon through the shared extension
# symlink, so that verdict was about other code; against this tree it
# passes unchanged.
BACKENDS = ("real",)

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
