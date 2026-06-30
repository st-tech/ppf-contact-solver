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
