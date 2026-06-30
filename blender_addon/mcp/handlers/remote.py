"""Remote management handlers (scene parameters, install, abort)."""

from typing import Optional

import bpy  # pyright: ignore

from ...core import services
from ...core.client import communicator as com
from ...models.groups import get_addon_data
from ..decorators import (
    MCPError,
    mcp_handler,
    remote_handler,
)


@remote_handler
def abort_operation():
    """Abort the current in-progress operation."""
    services.abort()
    return "Operation abort initiated"


@remote_handler
def install_paramiko():
    """Install the Paramiko library."""
    from ...core.module import get_installing_status

    if get_installing_status() is not False:
        raise MCPError("Cannot install Paramiko: another installation in progress")
    # Use bpy.ops for the modal timer loop
    bpy.ops.ssh.install_paramiko()
    return "Paramiko installation initiated"


@remote_handler
def install_docker():
    """Install the Docker library."""
    from ...core.module import get_installing_status

    if get_installing_status() is not False:
        raise MCPError("Cannot install Docker-Py: another installation in progress")
    # Use bpy.ops for the modal timer loop
    bpy.ops.ssh.install_docker()
    return "Docker-Py installation initiated"


@mcp_handler
def set_scene_parameters(
    step_size: Optional[float] = None,
    min_newton_steps: Optional[int] = None,
    frame_count: Optional[int] = None,
    frame_rate: Optional[int] = None,
    gravity: Optional[list[float]] = None,
    wind_direction: Optional[list[float]] = None,
    wind_strength: Optional[float] = None,
    air_density: Optional[float] = None,
    air_friction: Optional[float] = None,
    vertex_air_damp: Optional[float] = None,
    inactive_momentum_frames: Optional[int] = None,
    contact_nnz: Optional[int] = None,
    line_search_max_t: Optional[float] = None,
    constraint_ghat: Optional[float] = None,
    cg_max_iter: Optional[int] = None,
    cg_tol: Optional[float] = None,
    include_face_mass: Optional[bool] = None,
    disable_contact: Optional[bool] = None,
    auto_save: Optional[bool] = None,
    auto_save_interval: Optional[int] = None,
    save_state_on_finish: Optional[bool] = None,
    keep_states: Optional[int] = None,
    precond: Optional[str] = None,
    schwarz_levels: Optional[int] = None,
    use_frame_rate_in_output: Optional[bool] = None,
    project_name: Optional[str] = None,
):
    """Set global scene parameters for physics simulation.

    Args:
        step_size: Simulation step size (seconds)
        min_newton_steps: Minimum Newton iterations per step
        frame_count: Number of simulation frames
        frame_rate: Frame rate for simulation
        gravity: Gravity acceleration vector [x, y, z] m/s^2
        wind_direction: Wind direction vector [x, y, z]
        wind_strength: Wind speed magnitude (m/s)
        air_density: Air density (kg/m^3)
        air_friction: Tangential/normal air friction ratio
        vertex_air_damp: Vertex-level air damping factor
        inactive_momentum_frames: Inactive momentum frame count
        contact_nnz: Max contact non-zero entries
        line_search_max_t: CCD TOI extension factor
        constraint_ghat: Boundary constraint gap distance
        cg_max_iter: PCG max iterations
        cg_tol: PCG relative tolerance
        include_face_mass: Include shell face mass for solids' surface elements
        disable_contact: Disable all contact detection
        auto_save: Enable auto-save
        auto_save_interval: Auto-save interval (frames)
        save_state_on_finish: Save a resumable state when the simulation finishes
        keep_states: Number of most-recent saved states to retain (0 = keep all)
        precond: PCG preconditioner, "BLOCK_JACOBI" (default) or "SCHWARZ"
        schwarz_levels: Number of additive Schwarz levels, 1 (single-level
            smoother) or 2 (two-level coarse correction, default). Only used
            when precond is "SCHWARZ".
        use_frame_rate_in_output: Use frame rate in output
        project_name: Project name used for remote session directory
    """
    scene = bpy.context.scene
    state = get_addon_data(scene).state

    from ...core.utils import check_vec3
    if gravity is not None:
        gravity = check_vec3("gravity", gravity, MCPError)
    if wind_direction is not None:
        wind_direction = check_vec3("wind_direction", wind_direction, MCPError)
    if precond is not None:
        precond = precond.upper()
        if precond not in {"SCHWARZ", "BLOCK_JACOBI"}:
            raise MCPError("precond must be 'SCHWARZ' or 'BLOCK_JACOBI'")
    if schwarz_levels is not None and schwarz_levels not in (1, 2):
        raise MCPError("schwarz_levels must be 1 or 2")

    param_map = {
        "step_size": step_size,
        "min_newton_steps": min_newton_steps,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "gravity_3d": gravity,
        "wind_direction": wind_direction,
        "wind_strength": wind_strength,
        "air_density": air_density,
        "air_friction": air_friction,
        "vertex_air_damp": vertex_air_damp,
        "inactive_momentum_frames": inactive_momentum_frames,
        "contact_nnz": contact_nnz,
        "line_search_max_t": line_search_max_t,
        "constraint_ghat": constraint_ghat,
        "cg_max_iter": cg_max_iter,
        "cg_tol": cg_tol,
        "include_face_mass": include_face_mass,
        "disable_contact": disable_contact,
        "auto_save": auto_save,
        "auto_save_interval": auto_save_interval,
        "save_state_on_finish": save_state_on_finish,
        "keep_states": keep_states,
        "precond": precond,
        "schwarz_levels": (
            f"LEVEL_{schwarz_levels}" if schwarz_levels is not None else None
        ),
        "use_frame_rate_in_output": use_frame_rate_in_output,
        "project_name": project_name,
    }

    updated_params = {}
    for param_name, value in param_map.items():
        if value is not None:
            setattr(state, param_name, value)
            updated_params[param_name] = value

    # Keep the runner's cached project name in lockstep with the UI field.
    # Without this, transfers after an MCP rename would still target the
    # name captured at Connect time.
    if project_name is not None:
        com.set_project_name(project_name)

    return {
        "message": f"Updated {len(updated_params)} scene parameters",
        "updated_parameters": updated_params,
    }


@mcp_handler
def get_scene_parameters():
    """Get current scene parameters."""
    scene = bpy.context.scene
    state = get_addon_data(scene).state

    return {
        "parameters": {
            "step_size": state.step_size,
            "min_newton_steps": state.min_newton_steps,
            "frame_count": state.frame_count,
            "frame_rate": state.frame_rate,
            "gravity": list(state.gravity_3d),
            "wind_direction": list(state.wind_direction),
            "wind_strength": state.wind_strength,
            "air_density": state.air_density,
            "air_friction": state.air_friction,
            "vertex_air_damp": state.vertex_air_damp,
            "inactive_momentum_frames": state.inactive_momentum_frames,
            "contact_nnz": state.contact_nnz,
            "line_search_max_t": state.line_search_max_t,
            "constraint_ghat": state.constraint_ghat,
            "cg_max_iter": state.cg_max_iter,
            "cg_tol": state.cg_tol,
            "include_face_mass": state.include_face_mass,
            "disable_contact": state.disable_contact,
            "auto_save": state.auto_save,
            "auto_save_interval": state.auto_save_interval,
            "save_state_on_finish": state.save_state_on_finish,
            "keep_states": state.keep_states,
            "precond": state.precond,
            "schwarz_levels": int(state.schwarz_levels.split("_")[1]),
            "use_frame_rate_in_output": state.use_frame_rate_in_output,
            "project_name": state.project_name,
        }
    }


@mcp_handler
def set_save_checkpoint_frames(frames: list[int]):
    """Set the explicit frames at which to save a resumable checkpoint.

    Replaces the current Save Checkpoints list. Frames are de-duplicated,
    clamped to Blender's 1-based minimum, and sorted ascending. These are
    the frames the Resume dialog offers, in addition to Auto Save and Save
    State on Finish.

    Args:
        frames: Frame indices (1-based) to save checkpoints at.
    """
    state = get_addon_data(bpy.context.scene).state
    cleaned = sorted({max(1, int(f)) for f in frames})
    state.save_checkpoint_frames.clear()
    for f in cleaned:
        item = state.save_checkpoint_frames.add()
        item.frame = f
    state.save_checkpoint_frames_index = len(cleaned) - 1 if cleaned else -1
    return {"message": f"Set {len(cleaned)} checkpoint frames", "frames": cleaned}


@mcp_handler
def clear_save_checkpoint_frames():
    """Clear all explicit Save Checkpoints frames."""
    state = get_addon_data(bpy.context.scene).state
    count = len(state.save_checkpoint_frames)
    state.save_checkpoint_frames.clear()
    state.save_checkpoint_frames_index = -1
    return {"message": f"Cleared {count} checkpoint frames"}


@mcp_handler
def list_save_checkpoint_frames():
    """List the explicit Save Checkpoints frames configured for the next run."""
    state = get_addon_data(bpy.context.scene).state
    frames = [int(item.frame) for item in state.save_checkpoint_frames]
    return {"frames": frames, "count": len(frames)}
