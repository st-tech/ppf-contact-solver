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
