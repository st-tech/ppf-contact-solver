"""Debug and testing handlers (commands, data transfer tests, git, compile)."""

import bpy  # pyright: ignore

from ...core.client import communicator as com
from ...core import services
from ...models.groups import get_addon_data
from ..decorators import (
    MCPError,
    debug_handler,
    mcp_handler,
)


@mcp_handler
def debug_data_send(data_size_mb: int = 1):
    """Send test data to remote server for debugging data transfer.

    Args:
        data_size_mb: Size of test data in megabytes (default: 1MB)
    """
    # Set data size in scene state first
    scene = bpy.context.scene
    get_addon_data(scene).state.data_size = data_size_mb

    if not com.is_server_running() or com.busy():
        raise MCPError("Cannot send test data: operator conditions not met")
    # Use bpy.ops for the modal timer loop
    bpy.ops.debug.data_send()
    return {
        "message": f"Test data send initiated ({data_size_mb}MB)",
        "data_size_mb": data_size_mb,
    }


@mcp_handler
def debug_data_receive():
    """Receive test data from remote server and verify integrity.

    This function should be called after debug_data_send to test the
    complete round-trip data transfer functionality.
    """
    if not com.is_server_running() or com.busy():
        raise MCPError(
            "Cannot receive test data: operator conditions not met (system busy or no test data sent)"
        )
    # Use bpy.ops for the modal timer loop
    bpy.ops.debug.data_receive()
    return {
        "message": "Test data receive initiated",
        "note": "Check Blender info/error messages for transfer verification results",
    }


@debug_handler
def execute_server_command(server_script: str):
    """Execute a server command/script.

    Args:
        server_script: Server script command to execute
    """
    # Set the server script in scene state
    scene = bpy.context.scene
    get_addon_data(scene).state.server_script = server_script

    if not com.is_connected() or not com.is_server_running() or com.busy():
        raise MCPError("Cannot execute server command: operator conditions not met")
    # Use bpy.ops for the modal timer loop
    bpy.ops.debug.execute_server()
    return {
        "message": "Server command execution initiated",
        "server_script": server_script,
    }


@debug_handler
def execute_shell_command(shell_command: str, use_shell: bool = True):
    """Execute a shell command on remote server.

    Args:
        shell_command: Shell command to execute
        use_shell: Whether to use shell execution
    """
    # Set the shell command in scene state
    scene = bpy.context.scene
    get_addon_data(scene).state.shell_command = shell_command
    get_addon_data(scene).state.use_shell = use_shell

    services.execute_shell(shell_command, shell=use_shell)
    return {
        "message": "Shell command execution initiated",
        "shell_command": shell_command,
        "use_shell": use_shell,
    }


@debug_handler
def git_pull_remote():
    """Pull the latest changes from the Git repository on remote server."""
    services.git_pull()
    return {
        "message": "Git pull on remote server initiated",
    }


@debug_handler
def compile_project():
    """Compile the project on remote server."""
    services.compile_project()
    return {
        "message": "Project compilation initiated",
    }


@debug_handler
def delete_log_file(log_file_path: str):
    """Delete the specified log file.

    Args:
        log_file_path: Path to the log file to delete
    """
    # Set the log file path in scene state
    scene = bpy.context.scene
    get_addon_data(scene).state.log_file_path = log_file_path

    services.delete_log(log_file_path)
    return {
        "message": f"Log file deleted: {log_file_path}",
        "log_file_path": log_file_path,
    }


@debug_handler
def git_pull_local():
    """Pull the latest changes from the local Git repository."""
    # Use bpy.ops for the modal subprocess loop
    bpy.ops.debug.git_pull_local()
    return {
        "message": "Local git pull initiated",
    }
