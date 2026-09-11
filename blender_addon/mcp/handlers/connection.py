"""Connection management handlers (SSH, Docker, local)."""

from typing import Optional

import bpy  # pyright: ignore

from ...core.client import communicator as com
from ...core import services
from ...models.defaults import DEFAULT_SERVER_PORT
from ...models.groups import get_addon_data
from ..decorators import (
    MCPError,
    connection_handler,
    mcp_handler,
)


def _get_connection_state():
    """Get scene state and SSH properties for connection setup."""
    scene = bpy.context.scene
    addon_data = get_addon_data(scene)
    state = addon_data.state
    state.project_name = state.project_name or "default"
    return state, addon_data.ssh_state


def _require_offline():
    """Raise MCPError unless a connection can actually be started.

    The state machine accepts a connect request only from the offline phase,
    so one issued while a connection is up, or while an earlier attempt is
    still handshaking, is dropped and changes nothing. Reporting that as an
    initiated connection would send a caller on to a transfer or a run
    against a host it never reached, so both states refuse here and the
    message names the one the session is in. The refusal comes before the
    connection settings are written, so a refused call leaves the panel
    holding the settings of the connection that is actually in play.
    """
    if com.is_connected():
        raise MCPError("Cannot initiate connection: already connected")
    if com.is_connecting():
        raise MCPError(
            "Cannot initiate connection: an earlier attempt is still connecting"
        )


@mcp_handler
def connect_ssh(
    host: str,
    username: str,
    key_path: str,
    remote_path: str,
    port: int = 22,
    container: Optional[str] = None,
    proxy_jump: Optional[str] = None,
):
    """Establish SSH connection to remote server for contact solver.

    Args:
        host: SSH hostname or IP address
        username: SSH username
        key_path: Path to SSH private key
        remote_path: Remote working directory path
        port: SSH port
        container: Docker container name (optional)
        proxy_jump: Jump host(s) to tunnel through, as ssh -J takes them,
            "[user@]host[:port]" comma separated (optional). Left unset, the
            ProxyJump entry in ~/.ssh/config for the host applies.
    """
    _require_offline()

    # Set connection parameters in scene state
    _, props = _get_connection_state()

    # Configure SSH connection parameters
    props.server_type = "CUSTOM"
    props.host = host
    props.username = username
    props.key_path = key_path
    props.ssh_remote_path = remote_path
    props.port = port
    props.proxy_jump = proxy_jump or ""
    if container:
        props.container = container
        props.server_type = "DOCKER_SSH"

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    return {
        "message": f"SSH connection initiated to {username}@{host}:{port}",
        "connection_type": "ssh",
        "host": host,
        "port": port,
        "container": container,
        "proxy_jump": proxy_jump,
    }


@mcp_handler
def connect_docker(container: str, path: str, port: int = DEFAULT_SERVER_PORT):
    """Establish Docker connection for contact solver.

    Args:
        container: Docker container name
        path: Working directory path in container
        port: Port the solver server listens on inside the container.
            Must be within the range the port field itself accepts.
    """
    _require_offline()

    # Set connection parameters in scene state
    _, props = _get_connection_state()

    # The port field has its own hard range and Blender clamps an assignment
    # outside it without saying so, which would report a successful connection
    # on a port the caller never asked for.
    port_range = props.bl_rna.properties["docker_port"]
    if not (port_range.hard_min <= port <= port_range.hard_max):
        raise MCPError(
            f"port must be within [{port_range.hard_min}, "
            f"{port_range.hard_max}], got {port}"
        )

    # Configure Docker connection parameters
    props.server_type = "DOCKER"
    props.container = container
    props.docker_path = path
    props.docker_port = port

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    return {
        "message": f"Docker connection initiated to container '{container}'",
        "connection_type": "docker",
        "container": container,
        "path": path,
    }


@mcp_handler
def connect_local(path: str):
    """Establish local connection for contact solver.

    Args:
        path: Local working directory path
    """
    _require_offline()

    # Set connection parameters in scene state
    _, props = _get_connection_state()

    # Configure local connection parameters
    props.server_type = "LOCAL"
    props.local_path = path

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    return {
        "message": f"Local connection initiated to path '{path}'",
        "connection_type": "local",
        "path": path,
    }


@mcp_handler
def connect_win_native(path: str, port: int = DEFAULT_SERVER_PORT):
    """Establish Windows native connection for contact solver.

    Args:
        path: Path to the Windows native build or distribution directory
        port: Port for the solver server
    """
    _require_offline()

    _, props = _get_connection_state()

    # Configure Windows native connection parameters
    props.server_type = "WIN_NATIVE"
    props.win_native_path = path
    props.docker_port = port

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    return {
        "message": f"Windows native connection initiated to path '{path}'",
        "connection_type": "win_native",
        "path": path,
        "port": port,
    }


@mcp_handler
def disconnect():
    """Disconnect from the solver host, or cancel a connection still in progress."""
    if com.info.status.abortable():
        raise MCPError("Cannot disconnect: an abortable operation is in progress")
    services.disconnect()
    return "Disconnected from server"


@mcp_handler
def connect():
    """Connect using current connection settings, mimicking the connect button press."""
    _require_offline()

    # Get current scene and connection state
    state, props = _get_connection_state()

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    # Return current connection info
    return {
        "message": "Connection initiated using current settings",
        "connection_type": props.server_type,
        "project_name": state.project_name,
        "status": "connecting",
    }


@connection_handler
def start_remote_server():
    """Start the remote server process."""
    if not com.is_connected() or com.is_server_running() is not False or com.busy():
        raise MCPError("Cannot start remote server: operator conditions not met")
    # Use bpy.ops for the modal timer loop
    bpy.ops.ssh.start_server()
    return "Remote server start initiated"


@connection_handler
def stop_remote_server():
    """Stop the remote server process."""
    if (
        not com.is_connected()
        or not com.is_server_running()
        or com.info.status.abortable()
    ):
        raise MCPError("Cannot stop remote server: operator conditions not met")
    # Use bpy.ops for the modal timer loop
    bpy.ops.ssh.stop_server()
    return "Remote server stop initiated"


@connection_handler
def is_remote_server_running():
    """Check if remote server is running."""
    running = com.is_server_running()
    return {
        "server_running": running,
        "message": f"Remote server is {'running' if running else 'not running'}",
    }


@connection_handler
def get_remote_status():
    """Get detailed remote server status."""
    response = com.response
    return {
        "server_running": com.is_server_running(),
        "server_response": response,
        "current_status": com.info.status.value,
        "progress": com.info.progress,
        "message": com.info.message,
        "error": com.info.error,
        "server_error": com.info.server_error,
    }


@connection_handler
def update_remote_status():
    """Update remote server status."""
    services.update_status()
    return "Status update initiated"


@mcp_handler
def get_connection_info():
    """Get detailed connection information."""
    scene = bpy.context.scene
    ssh_state = get_addon_data(scene).ssh_state
    state = get_addon_data(scene).state
    connection = com.connection
    info = com.info

    # Determine connection type in snake_case
    connection_type = "unknown"
    if connection.type == "ssh":
        connection_type = "docker_over_ssh" if connection.container else "ssh"
    elif connection.type == "docker":
        connection_type = "docker_local"

    # Build SSH configuration info - show if there's any SSH config set
    ssh_config = {}
    if (
        connection.type == "ssh"
        or ssh_state.server_type in ["CUSTOM", "DOCKER_SSH", "DOCKER_SSH_COMMAND"]
        or ssh_state.host
    ):  # Show SSH config if host is set
        # Only include non-empty values
        if ssh_state.host:
            ssh_config["host"] = ssh_state.host
        if ssh_state.port:
            ssh_config["port"] = ssh_state.port
        if ssh_state.username:
            ssh_config["username"] = ssh_state.username
        if ssh_state.key_path:
            ssh_config["key_path"] = ssh_state.key_path
        if ssh_state.proxy_jump:
            ssh_config["proxy_jump"] = ssh_state.proxy_jump

        # Use the correct remote path based on server type
        if ssh_state.server_type in ["CUSTOM", "COMMAND"]:
            # Pure SSH connections use ssh_remote_path
            if ssh_state.ssh_remote_path:
                ssh_config["remote_path"] = ssh_state.ssh_remote_path
        else:
            # Docker over SSH connections use docker_path
            if ssh_state.docker_path:
                ssh_config["remote_path"] = ssh_state.docker_path

        # Add SSH command if using command mode and command is set
        if (
            ssh_state.server_type in ["COMMAND", "DOCKER_SSH_COMMAND"]
            and ssh_state.command
        ):
            ssh_config["ssh_command"] = ssh_state.command

    # Build Docker configuration info - show if there's any Docker config set
    docker_config = {}
    if (
        connection.container
        or connection.type == "docker"
        or ssh_state.server_type in ["DOCKER", "DOCKER_SSH", "DOCKER_SSH_COMMAND"]
        or ssh_state.container
    ):  # Show Docker config if container is set
        container_name = connection.container or ssh_state.container
        if container_name:
            docker_config["container"] = container_name
        if ssh_state.docker_path:
            docker_config["docker_path"] = ssh_state.docker_path

    # Build project info
    project_info = {
        "project_name": state.project_name,
    }

    # Build connection status
    connection_status = {
        "connected": com.is_connected(),
        "server_running": connection.server_running,
        "type": connection_type,
        "current_directory": connection.current_directory,
        "status": info.status.value,
    }

    return {
        "ssh_config": ssh_config,
        "docker_config": docker_config,
        "project_info": project_info,
        "connection_status": connection_status,
    }


# ---------------------------------------------------------------------------
# Solver GPU selection
# ---------------------------------------------------------------------------


@mcp_handler
def list_solver_gpus():
    """List the GPUs on the solver host, and which one is selected.

    The list is a cache filled by `refresh_solver_gpus`, which reads it from
    the solver host over the active connection. Before the first refresh the
    list is empty, which reports as `probed: false` rather than as a host with
    no GPUs.

    A selection is stored as both an index and a stable UUID, and the UUID
    wins: a .blend saved against one host and opened against another must not
    silently resolve to a different physical device.
    """
    from ...core import gpu_devices

    _, props = _get_connection_state()
    devices = gpu_devices.cached_gpu_devices()
    selected = gpu_devices.selected_device(
        props.solver_gpu_index, props.solver_gpu_uuid, devices
    )
    return {
        "probed": gpu_devices.has_probed(),
        "probe_error": gpu_devices.gpu_probe_error() or None,
        "devices": [
            {"index": device.index, "uuid": device.uuid, "name": device.name}
            for device in devices
        ],
        "selected_index": props.solver_gpu_index,
        "selected_uuid": props.solver_gpu_uuid or None,
        "selected_name": selected.name if selected else None,
    }


@connection_handler
def refresh_solver_gpus():
    """Re-read the GPU list from the solver host.

    Requires an active connection: the list is produced by a command run on
    the host, so there is nowhere to read it from otherwise. The refreshed
    list is available from `list_solver_gpus`.
    """
    com.refresh_solver_host_gpus()
    return "Solver GPU list refresh requested"


@mcp_handler
def set_solver_gpu(uuid: Optional[str] = None, index: Optional[int] = None):
    """Choose which GPU on the solver host runs the simulation.

    Pass `uuid` to name a device stably, which is what the add-on stores and
    prefers. `index` alone selects by CUDA index and is only reliable while
    the host's device set does not change. Passing neither clears the
    selection back to automatic.

    The selection is validated against the cached device list when one has
    been probed; with no list there is no evidence to contradict the request,
    so it is honored as given.

    Args:
        uuid: Stable device UUID, from list_solver_gpus
        index: CUDA device index, used when no uuid is given
    """
    from ...core import gpu_devices

    _, props = _get_connection_state()

    if uuid is None and index is None:
        props.solver_gpu_index = 0
        props.solver_gpu_uuid = ""
        return {"message": "Solver GPU selection cleared to automatic"}

    if uuid is not None:
        device = gpu_devices.find_device_by_uuid(uuid)
        if device is None and gpu_devices.has_probed():
            raise MCPError(
                f"No GPU with uuid {uuid!r} on the solver host. "
                "Call refresh_solver_gpus, then list_solver_gpus."
            )
        resolved_index = device.index if device else (index if index is not None else 0)
        props.solver_gpu_index = resolved_index
        props.solver_gpu_uuid = uuid
    else:
        try:
            gpu_devices.validate_selection(index)
        except Exception as exc:
            raise MCPError(str(exc)) from exc
        device = gpu_devices.find_device(index)
        props.solver_gpu_index = index
        props.solver_gpu_uuid = device.uuid if device else ""

    return {
        "message": "Solver GPU selected",
        "selected_index": props.solver_gpu_index,
        "selected_uuid": props.solver_gpu_uuid or None,
    }
