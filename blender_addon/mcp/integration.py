"""Integration module for the decorator-based MCP system."""

from collections.abc import Callable
from typing import Any

from .decorators import get_handler_registry


# Each handler source module maps to the LLM resource(s) a client should read
# for usage context. The key matches the module basename (``handlers/<name>.py``
# -> ``<name>``; ``blender_handlers.py`` -> ``blender``).
_LLM_REFS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "connection": ("connections",),
    "group": ("scene",),
    "object_ops": ("constraints", "scene"),
    "simulation": ("simulation",),
    "scene": ("constraints",),
    "dyn_params": ("parameters",),
    "remote": ("connections",),
    "console": ("debug",),
    "debug": ("debug",),
    "blender": ("integrations",),
}
_DEFAULT_REFS: tuple[str, ...] = ("integrations",)

# Per-tool overrides for handlers whose natural doc lives outside their
# module's default. Keeps the common case declarative.
_LLM_REFS_BY_TOOL: dict[str, tuple[str, ...]] = {
    # run_python_script and execute_shell_command live on the Blender
    # escape-hatch surface; `integrations` has the scope table and the
    # "use MCP tools, not raw Python" rules.
    "run_python_script": ("integrations", "scene"),
    "execute_shell_command": ("connections", "debug"),
    # Mesh-resolution helpers are explained in the MCP scene-setup section
    # of integrations (1-3% edge-length window).
    "get_average_edge_length": ("integrations",),
    "get_object_bounding_box_diagonal": ("integrations",),
    # Material / scene parameters belong to the parameters doc.
    "set_group_material_properties": ("parameters",),
    "set_scene_parameters": ("parameters",),
    "get_scene_parameters": ("parameters",),
}


def _infer_module_key(func: Callable) -> str:
    module = getattr(func, "__module__", "") or ""
    if ".handlers." in module:
        return module.rsplit(".", 1)[-1]
    if module.endswith(".blender_handlers"):
        return "blender"
    return "default"


def _enrich_description(tool_name: str, description: str, func: Callable) -> str:
    refs = _LLM_REFS_BY_TOOL.get(tool_name)
    if refs is None:
        refs = _LLM_REFS_BY_MODULE.get(_infer_module_key(func), _DEFAULT_REFS)
    uris = ", ".join(f"llm://{r}" for r in refs)
    base = (description or "").rstrip()
    if base and not base.endswith((".", "!", "?")):
        base += "."
    return f"{base}\n\nDocs: read {uris} with resources/read for usage and examples."


def get_integrated_tools_list() -> list[dict[str, Any]]:
    """Return every registered tool schema with an `llm://` docs pointer appended.

    The pointer tells MCP clients which `resources/read` URI to fetch for
    usage context when they scan tool descriptions.
    """
    enriched: list[dict[str, Any]] = []
    for name, info in get_handler_registry().items():
        schema = dict(info["schema"])
        schema["description"] = _enrich_description(
            name, schema.get("description", ""), info["func"]
        )
        enriched.append(schema)
    return enriched


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
            object_ops,
            remote,
            scene,
            simulation,
        )
    except ImportError as e:
        print(f"Could not load handlers: {e}")

    handlers = get_integrated_handlers()
    tools = get_integrated_tools_list()
    return {"handlers": handlers, "tools": tools, "registry": get_handler_registry()}
