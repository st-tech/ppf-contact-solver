# File: console.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Blender text editor integration for the console.
# Console class and singleton live in models.console; this module
# provides the Blender timer callback and registration.

import bpy  # pyright: ignore

from ..core.utils import get_timer_wait_time
from ..models.console import console


def timer_callback():
    """Blender timer callback to process messages and MCP tasks."""
    console.process_messages()

    # Also process MCP tasks in the main thread
    try:
        from ..mcp.task_system import process_mcp_tasks

        process_mcp_tasks()
    except ImportError:
        # MCP server module not available, skip
        pass
    except Exception as e:
        # Log error but don't crash the timer
        console.write(f"MCP task processing error: {e}")

    return get_timer_wait_time()


def register():
    """Register the console timer."""
    if not bpy.app.timers.is_registered(timer_callback):
        bpy.app.timers.register(timer_callback, persistent=True)


def unregister():
    """Unregister the console timer."""
    if bpy.app.timers.is_registered(timer_callback):
        bpy.app.timers.unregister(timer_callback)
