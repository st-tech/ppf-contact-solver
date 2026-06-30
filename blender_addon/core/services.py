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

from .facade import communicator as com


def _require_connected_and_idle():
    """Raise if not connected or if the communicator is busy."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    if com.busy():
        raise RuntimeError("Communicator is busy")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def disconnect():
    """Disconnect current connection."""
    if not com.is_connected():
        raise RuntimeError("Not connected")
    com.disconnect()


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
    _require_connected_and_idle()
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
    _require_connected_and_idle()
    com.exec(command, shell=shell)


def git_pull():
    """Pull latest changes on remote."""
    _require_connected_and_idle()
    com.exec("git pull")


def compile_project():
    """Compile project on remote."""
    _require_connected_and_idle()
    com.exec("/root/.cargo/bin/cargo build --release")


def delete_log(log_path):
    """Delete a local log file."""
    if not log_path or not os.path.exists(log_path):
        raise RuntimeError("Log file does not exist")
    os.remove(log_path)
