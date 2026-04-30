# File: services.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Service layer providing canonical entry points for operations.
# Called by both MCP handlers and UI operators.  These functions
# contain the real logic and validation so that neither layer
# depends on the other.

import os

from ..models.defaults import DEFAULT_SERVER_PORT
from .client import communicator as com


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect_ssh(
    host, port, username, key_path, path, container=None, server_port=DEFAULT_SERVER_PORT
):
    """Initiate SSH connection."""
    if com.is_connected():
        raise RuntimeError("Already connected")
    com.connect_ssh(
        host=host,
        port=port,
        username=username,
        key_path=key_path,
        path=path,
        container=container,
        server_port=server_port,
    )


def connect_docker(container, path, server_port=DEFAULT_SERVER_PORT):
    """Initiate Docker connection."""
    if com.is_connected():
        raise RuntimeError("Already connected")
    com.connect_docker(container, path, server_port=server_port)


def connect_local(path, server_port=DEFAULT_SERVER_PORT):
    """Initiate local connection."""
    if com.is_connected():
        raise RuntimeError("Already connected")
    com.connect_local(path, server_port=server_port)


def disconnect():
    """Disconnect current connection."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    com.disconnect()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server():
    """Start remote server."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.is_server_running():
        raise RuntimeError("Server already running")
    com.start_server()


def stop_server():
    """Stop remote server."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if not com.is_server_running():
        raise RuntimeError("Server not running")
    com.stop_server()


# ---------------------------------------------------------------------------
# Solver control (simple, non-modal)
# ---------------------------------------------------------------------------


def terminate():
    """Terminate solver."""
    com.terminate()


def save_and_quit():
    """Save and quit solver."""
    com.save_and_quit()


def update_status():
    """Query server status."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.busy():
        raise RuntimeError("Communicator is busy")
    com.query(message="Updating Status...")


def abort():
    """Abort current operation."""
    com.abort()


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def show_console():
    """Show the console window."""
    from ..models.console import console

    console.show()


# ---------------------------------------------------------------------------
# Remote shell / maintenance
# ---------------------------------------------------------------------------


def execute_shell(command, shell=True):
    """Execute shell command on remote."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.busy():
        raise RuntimeError("Communicator is busy")
    com.exec(command, shell=shell)


def git_pull():
    """Pull latest changes on remote."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.busy():
        raise RuntimeError("Communicator is busy")
    com.exec("git pull")


def compile_project():
    """Compile project on remote."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.busy():
        raise RuntimeError("Communicator is busy")
    com.exec("/root/.cargo/bin/cargo build --release")


def delete_log(log_path):
    """Delete a local log file."""
    if not log_path or not os.path.exists(log_path):
        raise RuntimeError("Log file does not exist")
    os.remove(log_path)
