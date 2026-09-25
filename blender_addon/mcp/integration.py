"""Integration module for the decorator-based MCP system."""

from collections.abc import Callable
from typing import Any

from .decorators import get_handler_registry


def get_integrated_tools_list() -> list[dict[str, Any]]:
    """Return every registered tool schema, as each handler declares it.

    A tool's description and input schema are its whole reference: they come
    from the handler that implements the tool, so they cannot describe a tool
    other than the one that runs.
    """
    return [dict(info["schema"]) for info in get_handler_registry().values()]


def get_integrated_handlers() -> dict[str, Callable]:
    """Get handler mapping from decorator-based handlers.

    Returns:
        Dictionary mapping handler names to handler functions
    """
    return {name: info["func"] for name, info in get_handler_registry().items()}


def initialize_integrated_system():
    """Initialize the integrated MCP system."""
    try:
        from . import blender_handlers  # noqa: F401  # pyright: ignore[reportUnusedImport]
        from .handlers import (  # noqa: F401  # pyright: ignore[reportUnusedImport]
            connection,
            console,
            debug,
            dyn_params,
            group,
            material_maps,
            object_ops,
            presets,
            remote,
            scene,
            simulation,
            statistics,
        )
    except ImportError as e:
        print(f"Could not load handlers: {e}")

    handlers = get_integrated_handlers()
    tools = get_integrated_tools_list()
    return {"handlers": handlers, "tools": tools, "registry": get_handler_registry()}
