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


def _require_not_connected():
    """Raise MCPError if already connected."""
    if com.is_connected():
        raise MCPError("Cannot initiate connection: already connected")


@mcp_handler
def connect_ssh(
    host: str,
    username: str,
    key_path: str,
    remote_path: str,
    port: int = 22,
    container: Optional[str] = None,
):
    """Establish SSH connection to remote server for contact solver.

    Args:
        host: SSH hostname or IP address
        username: SSH username
        key_path: Path to SSH private key
        remote_path: Remote working directory path
        port: SSH port
        container: Docker container name (optional)
    """
    # Set connection parameters in scene state
    state, props = _get_connection_state()

    # Configure SSH connection parameters
    props.server_type = "CUSTOM"
    props.host = host
    props.username = username
    props.key_path = key_path
    props.ssh_remote_path = remote_path
    props.port = port
    if container:
        props.container = container
        props.server_type = "DOCKER_SSH"

    _require_not_connected()

    # Use bpy.ops for the modal timer loop required by connection lifecycle
    bpy.ops.ssh.run_command()

    return {
        "message": f"SSH connection initiated to {username}@{host}:{port}",
        "connection_type": "ssh",
        "host": host,
        "port": port,
        "container": container,
    }


@mcp_handler
def connect_docker(container: str, path: str):
    """Establish Docker connection for contact solver.

    Args:
        container: Docker container name
        path: Working directory path in container
    """
    # Set connection parameters in scene state
    state, props = _get_connection_state()

    # Configure Docker connection parameters
    props.server_type = "DOCKER"
    props.container = container
    props.docker_path = path

    _require_not_connected()

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
    # Set connection parameters in scene state
    state, props = _get_connection_state()

    # Configure local connection parameters
    props.server_type = "LOCAL"
    props.local_path = path

    _require_not_connected()

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
    state, props = _get_connection_state()

    # Configure Windows native connection parameters
    props.server_type = "WIN_NATIVE"
    props.win_native_path = path
    props.docker_port = port

    _require_not_connected()

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
    """Disconnect from remote server."""
    if com.info.status.abortable():
        raise MCPError("Cannot disconnect: an abortable operation is in progress")
    services.disconnect()
    return "Disconnected from server"


@mcp_handler
def connect():
    """Connect using current connection settings, mimicking the connect button press."""
    # Get current scene and connection state
    state, props = _get_connection_state()

    _require_not_connected()

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
