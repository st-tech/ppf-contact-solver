"""Simulation control handlers (transfer, run, resume, fetch, etc.)."""

import bpy  # pyright: ignore

from ...core.client import communicator as com
from ...core import services
from ..decorators import (
    MCPError,
    mcp_handler,
    simulation_handler,
)


def _status_detail() -> str:
    """Return a compact status/error suffix for MCP-facing failures."""
    detail = [f"status={com.info.status.value}"]
    error = (
        com.info.error
        or com.info.server_error
        or com.error
        or com.server_error
        or com.info.response.get("error", "")
    )
    if error:
        detail.append(f"error={error}")
    return ", ".join(detail)


def _require_operator_poll(op_cls, action: str) -> None:
    """Validate the exact Blender operator poll used by the UI operator."""
    try:
        allowed = bool(op_cls.poll(bpy.context))
    except Exception as exc:
        raise MCPError(
            f"Cannot {action}: operator poll failed ({type(exc).__name__}: {exc})"
        ) from exc
    if not allowed:
        raise MCPError(
            f"Cannot {action}: operator conditions not met ({_status_detail()})"
        )


@simulation_handler
def transfer_data():
    """Transfer data to the solver."""
    from ...ui.solver import SOLVER_OT_Transfer

    _require_operator_poll(SOLVER_OT_Transfer, "transfer data")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.transfer()
    return {
        "message": "Data transfer initiated",
        "current_status": com.info.status.value,
    }


@simulation_handler
def run_simulation():
    """Start simulation."""
    from ...ui.solver import SOLVER_OT_Run

    _require_operator_poll(SOLVER_OT_Run, "start simulation")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.run()
    return {
        "message": "Simulation started",
        "current_status": com.info.status.value,
    }


@simulation_handler
def resume_simulation():
    """Resume paused simulation."""
    from ...ui.solver import SOLVER_OT_Resume

    _require_operator_poll(SOLVER_OT_Resume, "resume simulation")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.resume()
    return {
        "message": "Simulation resumed",
        "current_status": com.info.status.value,
    }


@simulation_handler
def terminate_simulation():
    """Force terminate simulation."""
    services.terminate()
    return {
        "message": "Simulation termination initiated",
        "current_status": com.info.status.value,
    }


@simulation_handler
def save_and_quit_simulation():
    """Save and quit simulation gracefully."""
    services.save_and_quit()
    return {
        "message": "Save and quit initiated",
        "current_status": com.info.status.value,
    }


@simulation_handler
def update_params():
    """Update the parameters of the solver."""
    from ...ui.solver import SOLVER_OT_UpdateParams

    _require_operator_poll(SOLVER_OT_UpdateParams, "update parameters")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.update_params()
    return {
        "message": "Parameter update initiated",
        "current_status": com.info.status.value,
    }


@simulation_handler
def delete_remote_data():
    """Delete data on the remote server."""
    from ...ui.solver import SOLVER_OT_DeleteRemoteData

    _require_operator_poll(SOLVER_OT_DeleteRemoteData, "delete remote data")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.delete_remote_data()
    return {
        "message": "Remote data deletion initiated",
        "current_status": com.info.status.value,
    }


@simulation_handler
def fetch_animation():
    """Fetch simulation results from server."""
    from ...ui.solver import SOLVER_OT_FetchData

    _require_operator_poll(SOLVER_OT_FetchData, "fetch animation")
    # Use bpy.ops for the modal timer loop
    bpy.ops.solver.fetch_remote_data()
    return {
        "message": "Animation fetch initiated",
        "current_status": com.info.status.value,
    }


@mcp_handler
def clear_local_animation():
    """Clear local animation data and keyframes."""
    from ...ui.solver import SOLVER_OT_ClearAnimation

    _require_operator_poll(SOLVER_OT_ClearAnimation, "clear animation")
    # Use bpy.ops since the operator contains complex clear logic
    bpy.ops.solver.clear_animation()
    return "Local animation data cleared"


@mcp_handler
def list_checkpoint_frames():
    """List resumable checkpoint frames saved on the server.

    Returns the saved-state frames (Blender 1-based) a resume can continue
    from, read from the latest solver status response. Empty until at least
    one checkpoint has been saved (via Save Checkpoints, Auto Save, or Save
    State on Finish). Use resume_simulation to continue from the latest one.
    """
    frames = [int(f) for f in com.saved_state_frames()]
    return {"checkpoint_frames": frames, "count": len(frames)}


@mcp_handler
def resume_simulation_from(frame: int):
    """Resume the simulation from a specific saved checkpoint frame.

    Continues the run already on the server from the chosen checkpoint
    (Blender 1-based) without re-uploading or rebuilding: frames before the
    checkpoint are kept, the rest are overwritten. Refuses if the geometry
    has drifted (transfer_data + run_simulation instead) or the parameters
    have changed (update_params first). Use list_checkpoint_frames to see the
    available frames; resume_simulation continues from the latest one.

    Args:
        frame: Saved checkpoint frame to resume from (Blender 1-based).
    """
    from ...models.groups import get_addon_data
    from ...core.facade import engine
    from ...core.encoder.mesh import compute_data_hash
    from ...core.encoder.params import compute_param_hash
    from ...ui.solver import SOLVER_OT_ResumeFrom, _check_project_name_sync

    _require_operator_poll(SOLVER_OT_ResumeFrom, "resume from a checkpoint")
    context = bpy.context

    saved = [int(f) for f in com.saved_state_frames()]
    target = int(frame)
    if target not in saved:
        raise MCPError(
            f"Frame {target} is not a saved checkpoint. Available frames: {saved}"
        )

    # Mirror SOLVER_OT_ResumeFrom.invoke()'s drift guards. Resume never
    # re-uploads or rebuilds, so refuse when the live encoding has drifted
    # from what the server last echoed; EXEC_DEFAULT below skips invoke, so
    # these run here instead.
    error = _check_project_name_sync(context)
    if error:
        raise MCPError(error)
    try:
        local_data = compute_data_hash(context)
    except ValueError as e:
        raise MCPError(str(e))
    local_param = compute_param_hash(context)
    if engine.state.server_data_hash and local_data != engine.state.server_data_hash:
        raise MCPError(
            "Geometry has changed; resume is not possible. Transfer and run "
            "for a fresh simulation."
        )
    if engine.state.server_param_hash and local_param != engine.state.server_param_hash:
        raise MCPError(
            "Parameters have changed since the last transfer. Call update_params "
            "before resuming."
        )

    # Populate the checkpoint-picker state the operator's execute() reads,
    # selecting the requested frame, then drive execute() (com.resume +
    # modal) directly, bypassing the interactive dialog.
    state = get_addon_data(context.scene).state
    state.checkpoint_frames.clear()
    target_index = -1
    for i, saved_frame in enumerate(saved):
        item = state.checkpoint_frames.add()
        item.frame = saved_frame
        if saved_frame == target:
            target_index = i
    state.checkpoint_frames_index = target_index
    bpy.ops.solver.resume_from("EXEC_DEFAULT")
    return {
        "message": f"Resuming simulation from checkpoint frame {target}",
        "current_status": com.info.status.value,
    }


def _export_sim_cache(op, filepath: str, kind: str) -> dict:
    """Shared body for the two cache exporters.

    The preconditions are checked here rather than left to the operator
    because bpy.ops returns only a status set: a CANCELLED export would
    otherwise reach the caller with no reason attached, and the reasons
    (a run still in flight, unfetched frames, nothing simulated yet) each
    call for a different next step.
    """
    import os

    from ...ui.dynamics.export_ops import _excluded_sim_curves, preflight_error

    context = bpy.context
    error = preflight_error(context)
    if error:
        raise MCPError(f"Cannot export {kind}: {error}")

    if not filepath:
        raise MCPError("filepath is required")
    resolved = bpy.path.abspath(filepath)
    parent = os.path.dirname(resolved) or "."
    if not os.path.isdir(parent):
        raise MCPError(f"Destination directory does not exist: {parent}")

    excluded_curves = _excluded_sim_curves(context)
    try:
        status = op("EXEC_DEFAULT", filepath=resolved)
    except RuntimeError as exc:
        # The operator reports an ERROR before returning CANCELLED, and
        # bpy.ops raises an operator's ERROR report as this RuntimeError, so
        # the reason arrives here and not in the status set below. Restating
        # it as an MCPError keeps the failure inside this module's error type
        # (ValueError through the Python API) instead of a bare RuntimeError.
        raise MCPError(f"{kind} export failed: {exc}") from exc
    if "FINISHED" not in status:
        raise MCPError(
            f"{kind} export did not complete ({', '.join(sorted(status))}). "
            f"Check the Blender console via get_console_lines."
        )

    result = {
        "message": f"Exported the simulated mesh sequence to {resolved}",
        "filepath": resolved,
        "format": kind,
    }
    if excluded_curves:
        result["excluded_curves"] = excluded_curves
        result["message"] += (
            f". {len(excluded_curves)} rod/curve object(s) were not exported, "
            "which this cache format does not carry"
        )
    return result


@mcp_handler
def export_usd(filepath: str):
    """Export the simulated mesh sequence as a USD cache.

    A lighter alternative to baking shape keys: the deformation is sampled per
    frame from the solver cache into a file other DCC tools can play back.
    Requires every frame to be fetched first; call fetch_animation and wait for
    it to finish. Rod and curve objects are not carried by this format.

    Args:
        filepath: Destination path, used as given once a leading "//"
            blend-relative prefix is resolved. The suffix is what picks the
            USD flavor and it is never rewritten, so pass one of .usdc
            (crate), .usda (ASCII), .usd or .usdz (package). The parent
            directory must already exist.
    """
    return _export_sim_cache(bpy.ops.solver.export_usd, filepath, "USD")


@mcp_handler
def export_alembic(filepath: str):
    """Export the simulated mesh sequence as an Alembic (ABC) cache.

    A lighter alternative to baking shape keys: the deformation is sampled per
    frame from the solver cache into a file other DCC tools can play back.
    Requires every frame to be fetched first; call fetch_animation and wait for
    it to finish. Rod and curve objects are not carried by this format.

    Args:
        filepath: Destination .abc path, used as given once a leading "//"
            blend-relative prefix is resolved. The parent directory must
            already exist.
    """
    return _export_sim_cache(bpy.ops.solver.export_alembic, filepath, "Alembic")


@mcp_handler
def get_fetch_status():
    """Report which simulated frames have been fetched back into Blender.

    Fetching is a modal operation: `fetch_animation` returns as soon as it has
    started, so a caller needs a separate way to see how far it got. The export
    tools refuse while any frame is still unfetched, and `bpy.ops` hands back
    only a canceled status without the reason, so this reports the export
    preflight verdict alongside the frame list.

    `fetched_frames` is what landed locally, which is a different question from
    `list_checkpoint_frames` (frames saved on the remote), from
    `get_remote_status` (how the run itself is doing), and from
    `get_modal_job_status` (whether a bake or a deformation capture is still
    running inside Blender).
    """
    from ...models.groups import get_addon_data
    from ...ui.dynamics.export_ops import preflight_error

    scene = bpy.context.scene
    if scene is None:
        raise MCPError("No active Blender scene")

    state = get_addon_data(scene).state
    fetched = sorted(state.convert_fetched_frames_to_list())
    blocker = preflight_error(bpy.context)

    return {
        "fetched_frames": fetched,
        "fetched_count": len(fetched),
        "expected_frame_count": state.frame_count,
        "export_ready": blocker is None,
        "export_blocked_reason": blocker,
    }


# ---------------------------------------------------------------------------
# Long-running modal jobs: the keyframe bake, the STATIC deformation capture
# and the pin deformation capture.
#
# Each of the three keeps running on a timer after the tool that started it
# has returned, so an agent that starts one needs a way to ask whether it is
# still going and a way to stop it. An abort operator only raises its job's
# abort flag; the modal loop reads that flag on its next tick and unwinds
# there, which is why no tool below reports a job as already stopped.
#
# The job dictionaries themselves are private to the three ui.dynamics
# modules. What these adapters read is the public view of them: ``is_*_running``
# and ``*_progress_snapshot`` for the state, and the abort operator's own poll,
# which is true while the job is active and its abort flag is still clear, for
# whether that flag is already raised. An abort operator is not the only writer
# of the flag: both deformation captures raise it themselves when a per-frame
# sample fails, so a raised flag reports that the job is unwinding and not that
# a tool necessarily asked it to.
# ---------------------------------------------------------------------------

_BAKE_JOB = "bake"
_STATIC_CAPTURE_JOB = "static_deformation_capture"
_PIN_CAPTURE_JOB = "pin_deformation_capture"


def _job_state(key: str, abort_tool: str, is_running, snapshot, abort_cls) -> dict:
    """One modal job's state, reported through the owning module's public API.

    Progress is filled in only while the job is running: the counters are
    cleared once a job ends, so a zero read from a job that is not running
    says nothing about what that job did.
    """
    if not is_running():
        return {
            "job": key,
            "running": False,
            "abort_requested": False,
            "abort_tool": abort_tool,
            "frames_done": None,
            "frames_total": None,
            "item_count": None,
            "status_line": None,
        }
    frames_done, frames_total, status_line, item_count = snapshot()
    return {
        "job": key,
        "running": True,
        # The abort operator polls true only while the job is active and its
        # abort flag is still clear, so a running job whose poll is false is
        # one whose abort flag is raised, by an abort tool or by the job
        # itself on a frame it could not sample.
        "abort_requested": not bool(abort_cls.poll(bpy.context)),
        "abort_tool": abort_tool,
        "frames_done": int(frames_done),
        "frames_total": int(frames_total),
        "item_count": int(item_count),
        "status_line": str(status_line),
    }


def _bake_job_state() -> dict:
    """State of the keyframe bake job."""
    from ...ui.dynamics.bake_ops import (
        SOLVER_OT_BakeAbort,
        bake_progress_snapshot,
        is_bake_running,
    )

    return _job_state(
        _BAKE_JOB,
        "abort_bake",
        is_bake_running,
        bake_progress_snapshot,
        SOLVER_OT_BakeAbort,
    )


def _static_capture_job_state() -> dict:
    """State of the STATIC deformation capture job."""
    from ...ui.dynamics.static_deform_ops import (
        SOLVER_OT_CaptureAbort,
        capture_progress_snapshot,
        is_capture_running,
    )

    return _job_state(
        _STATIC_CAPTURE_JOB,
        "abort_static_deformation_capture",
        is_capture_running,
        capture_progress_snapshot,
        SOLVER_OT_CaptureAbort,
    )


def _pin_capture_job_state() -> dict:
    """State of the pin deformation capture job."""
    from ...ui.dynamics.pin_capture_ops import (
        SOLVER_OT_PinCaptureAbort,
        is_pin_capture_running,
        pin_capture_progress_snapshot,
    )

    return _job_state(
        _PIN_CAPTURE_JOB,
        "abort_pin_deformation_capture",
        is_pin_capture_running,
        pin_capture_progress_snapshot,
        SOLVER_OT_PinCaptureAbort,
    )


def _request_abort(state: dict, operator, label: str) -> dict:
    """Raise a running job's abort flag, or refuse and name the reason.

    The operator's return value is checked rather than assumed: it is the
    only thing bpy.ops hands back, and a job left running while this reported
    success would leave the caller polling for a stop that was never asked
    for.
    """
    if not state["running"]:
        raise MCPError(
            f"No {label} is running, so there is nothing to abort. "
            "Call get_modal_job_status for the jobs running now."
        )
    if state["abort_requested"]:
        raise MCPError(
            f"The {label} is already unwinding: its abort flag is raised, "
            "either by an earlier abort call or by the job itself on a frame "
            "it could not sample. The job stops on its next timer tick; poll "
            "get_modal_job_status until it reports running false. A job that "
            "raised the flag on its own reports the reason through Blender's "
            "own operator report, which this tool cannot read back."
        )
    result = operator("EXEC_DEFAULT")
    if "FINISHED" not in result:
        raise MCPError(
            f"The abort operator for the {label} returned "
            f"{', '.join(sorted(result))} instead of FINISHED, so the job was "
            "not asked to stop."
        )
    return {
        "message": (
            f"Requested an abort of the {label}. It takes effect on the job's "
            "next timer tick; poll get_modal_job_status until the job reports "
            "running false."
        ),
        "job": state["job"],
        "operator_result": sorted(result),
    }


@mcp_handler
def abort_bake():
    """Stop a running keyframe bake and undo what it has written so far.

    Baking an animation runs as a modal job that keeps going after
    bake_group_animation or bake_all_animation returns. This raises that job's
    abort flag. The job stops on its next timer tick and rolls back what it
    wrote: the shape keys and F-curves it added are removed and the curve
    handle types it changed are restored, leaving the PC2 caches, the
    ContactSolverCache modifiers and group membership as they were before the
    bake started.

    bake_group_single_frame and bake_all_single_frame start no such job. Each
    bakes its one frame inline and is complete when its own tool call returns,
    so a single-frame bake is never in flight, there is nothing to poll for and
    nothing here to abort.

    Refused when no bake is running, and refused again while an abort of the
    same bake is already in flight. Call get_modal_job_status for the jobs
    running now, and poll it afterwards until the bake reports running false.

    This stops the bake inside Blender. abort_operation stops an operation on
    the solver server, which is a different job.
    """
    return _request_abort(_bake_job_state(), bpy.ops.solver.bake_abort, "bake")


@mcp_handler
def abort_static_deformation_capture():
    """Stop a running STATIC collider deformation capture.

    capture_static_deformation and recapture_all_deformations start a modal job
    that steps the timeline and samples the shape of each deforming STATIC
    collider. This raises the job's abort flag. The job stops on its next timer
    tick, restores the frame it started from, and re-enables the
    ContactSolverCache modifiers it suspended for the sampling.

    A capture writes an object's result only once every frame of that object is
    sampled, so the frames taken before the abort are discarded and each object
    keeps the deformation cache it already had. get_static_deformation_status
    reports what is on an object; run the capture again to record it.

    Inside a recapture_all_deformations run the static phase runs first and the
    pin phase is queued behind it, so aborting here cancels the whole run: the
    pin phase never starts and those pins keep the captures they already had.
    Capture them with capture_pin_deformation, or start
    recapture_all_deformations again.

    Refused when no static capture is running, and refused again while an abort
    of it is already in flight. Call get_modal_job_status for the jobs running
    now, and poll it afterwards until this job reports running false.
    """
    return _request_abort(
        _static_capture_job_state(),
        bpy.ops.solver.capture_abort,
        "STATIC deformation capture",
    )


@mcp_handler
def abort_pin_deformation_capture():
    """Stop a running pin deformation capture.

    capture_pin_deformation and recapture_all_deformations start a modal job
    that steps the timeline and samples the moving pin vertices of each
    animated pin. This raises the job's abort flag. The job stops on its next
    timer tick, restores the frame it started from, and re-enables the
    ContactSolverCache modifiers it suspended for the sampling.

    A capture writes a pin's result only once every frame of that pin is
    sampled, so the frames taken before the abort are discarded and each pin
    keeps the capture it already had. get_pin_deformation_status reports what is
    on a pin; run the capture again to record it.

    Refused when no pin capture is running, and refused again while an abort of
    it is already in flight. Call get_modal_job_status for the jobs running now,
    and poll it afterwards until this job reports running false.
    """
    return _request_abort(
        _pin_capture_job_state(),
        bpy.ops.solver.pin_capture_abort,
        "pin deformation capture",
    )


@mcp_handler
def get_modal_job_status():
    """Report which long-running bake or capture job is running right now.

    Three jobs run on a timer inside Blender and outlive the tool call that
    started them, so a caller that starts one has no other way to tell whether
    it is still going: the keyframe bake, the STATIC collider deformation
    capture, and the pin deformation capture. This reports all three in one
    call, each with the frames it has processed and the tool that stops it.

    ``jobs`` carries one entry per job. ``bake`` is started by
    bake_group_animation or bake_all_animation and stopped by abort_bake.
    ``static_deformation_capture`` is started by capture_static_deformation or
    recapture_all_deformations and stopped by
    abort_static_deformation_capture. ``pin_deformation_capture`` is started by
    capture_pin_deformation or recapture_all_deformations and stopped by
    abort_pin_deformation_capture.

    bake_group_single_frame and bake_all_single_frame start none of the three.
    Each bakes its one frame inline and is complete when its own tool call
    returns, so ``bake`` stays running false throughout and there is nothing to
    poll for after one of them.

    ``abort_requested`` is true once the job is unwinding, whether an abort tool
    asked for it or the job hit an internal failure: the two deformation
    captures raise the same flag on a frame they could not sample. Either way
    the job has not yet reached the tick that stops it. ``frames_done``,
    ``frames_total``, ``item_count`` (objects for the bake and the STATIC
    capture, pins for the pin capture) and ``status_line`` are null while a job
    is not running, because those counters are cleared when a job ends.

    This covers the jobs running inside Blender. get_fetch_status reports how
    much of a solve has been fetched back into Blender and whether an export
    would be accepted, and get_remote_status reports the run on the server.
    """
    jobs = [
        _bake_job_state(),
        _static_capture_job_state(),
        _pin_capture_job_state(),
    ]
    running = [job["job"] for job in jobs if job["running"]]
    return {
        "jobs": jobs,
        "running_jobs": running,
        "any_running": bool(running),
    }
