"""Console and status handlers."""

from ...core.client import communicator as com
from ...core import services
from ...models.console import console
from ..decorators import (
    mcp_handler,
)


@mcp_handler
def get_console_lines():
    """Get current console text lines."""
    console_text = console.get_or_create()
    lines = [line.body for line in console_text.lines]
    return {"console_lines": lines, "line_count": len(lines)}


@mcp_handler
def get_latest_error():
    """Get latest error from both local and remote."""
    local_error = com.error
    remote_error = com.server_error

    return {
        "local_error": local_error if local_error else None,
        "remote_error": remote_error if remote_error else None,
        "has_errors": bool(local_error or remote_error),
    }


@mcp_handler
def show_console():
    """Show console window."""
    services.show_console()
    return "Console window shown"
