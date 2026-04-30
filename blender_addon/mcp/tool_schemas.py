"""MCP tool schema definitions."""


def get_tools_list():
    """Return the list of available MCP tools with their schemas."""
    from .integration import get_integrated_tools_list

    return get_integrated_tools_list()
