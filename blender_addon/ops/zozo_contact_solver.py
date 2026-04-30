"""Auto-generated bpy.ops.zozo_contact_solver.* operators from MCP handler registry.

This module dynamically creates Blender operator classes for every registered
MCP handler, making them callable as bpy.ops.zozo_contact_solver.<handler_name>().
"""

import inspect
import json
from typing import Any, Union, get_type_hints

import bpy  # pyright: ignore
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty  # pyright: ignore

# All dynamically generated operator classes
_generated_classes: list[type] = []


def _get_base_type(python_type):
    """Unwrap Optional[T] and return the base type."""
    if hasattr(python_type, "__origin__") and python_type.__origin__ is Union:
        args = python_type.__args__
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return python_type


def _make_property(python_type, default, description):
    """Create a Blender property from a Python type annotation."""
    base = _get_base_type(python_type)

    if base is int:
        kwargs = {"name": "", "description": description}
        if default is not inspect.Parameter.empty and default is not None:
            kwargs["default"] = default
        return IntProperty(**kwargs)
    elif base is float:
        kwargs = {"name": "", "description": description}
        if default is not inspect.Parameter.empty and default is not None:
            kwargs["default"] = default
        return FloatProperty(**kwargs)
    elif base is bool:
        kwargs = {"name": "", "description": description}
        if default is not inspect.Parameter.empty and default is not None:
            kwargs["default"] = default
        return BoolProperty(**kwargs)
    else:
        # str, list, dict, and anything else → StringProperty (JSON-encoded for complex types)
        kwargs = {"name": "", "description": description}
        if default is not inspect.Parameter.empty:
            if isinstance(default, str):
                kwargs["default"] = default
            elif default is None:
                kwargs["default"] = ""
            else:
                kwargs["default"] = json.dumps(default)
        return StringProperty(**kwargs)


def _is_json_property(python_type):
    """Check if this type should be JSON-decoded before passing to handler."""
    base = _get_base_type(python_type)
    return base in (list, dict) or (
        hasattr(base, "__origin__") and base.__origin__ in (list, dict)
    )


def _build_operator_class(name: str, handler_info: dict[str, Any]) -> type:
    """Build a bpy.types.Operator subclass for a single MCP handler."""
    wrapper = handler_info["func"]
    schema = handler_info["schema"]
    original_func = getattr(wrapper, "_original_func", None)

    # Extract parameter info from the original function
    params_info = {}  # name -> (python_type, default, is_json)
    if original_func:
        sig = inspect.signature(original_func)
        try:
            hints = get_type_hints(original_func)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            python_type = hints.get(param_name, str)
            params_info[param_name] = (
                python_type,
                param.default,
                _is_json_property(python_type),
            )

    # Get param descriptions from schema
    schema_props = schema.get("inputSchema", {}).get("properties", {})
    required_params = set(schema.get("inputSchema", {}).get("required", []))

    # Build class attributes
    annotations = {}
    class_attrs = {
        "bl_idname": f"zozo_contact_solver.{name}",
        "bl_label": name.replace("_", " ").title(),
        "bl_description": schema.get("description", ""),
        "bl_options": {"REGISTER"},
    }

    # Add properties — Blender 4.x requires property descriptors in __annotations__
    for param_name, (python_type, default, is_json) in params_info.items():
        desc = schema_props.get(param_name, {}).get("description", param_name)
        prop = _make_property(python_type, default, desc)
        annotations[param_name] = prop

    # Capture in closure
    _wrapper = wrapper
    _params_info = params_info
    _required_params = required_params

    def execute(self, context):
        # Build args dict from operator properties
        args = {}
        for p_name, (p_type, p_default, p_is_json) in _params_info.items():
            value = getattr(self, p_name)
            # Handle empty strings for optional params
            if (
                isinstance(value, str)
                and value == ""
                and p_name not in _required_params
                and p_default is inspect.Parameter.empty
            ):
                continue  # Skip unset optional params
            if (
                isinstance(value, str)
                and value == ""
                and p_default is not inspect.Parameter.empty
                and p_default is None
            ):
                continue  # Skip None-defaulted optional params left empty
            # JSON decode for list/dict types
            if p_is_json and isinstance(value, str) and value:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    self.report({"ERROR"}, f"Invalid JSON for parameter '{p_name}': {value}")
                    return {"CANCELLED"}
            args[p_name] = value

        result = _wrapper(args)

        if isinstance(result, dict) and result.get("status") == "error":
            self.report({"ERROR"}, result.get("message", "Unknown error"))
            return {"CANCELLED"}

        return {"FINISHED"}

    class_attrs["execute"] = execute
    class_attrs["__annotations__"] = annotations

    # Create the class
    cls = type(f"ZOZO_CTS_OT_{name}", (bpy.types.Operator,), class_attrs)
    return cls


def register():
    """Register all auto-generated operators from MCP handler registry."""
    global _generated_classes

    # Import handlers to ensure they're registered
    from ..mcp import blender_handlers  # noqa: F401
    from ..mcp.handlers import connection, console, debug, group, remote, scene, simulation  # noqa: F401
    from ..mcp.decorators import get_handler_registry

    registry = get_handler_registry()

    for name, info in registry.items():
        try:
            cls = _build_operator_class(name, info)
            bpy.utils.register_class(cls)
            _generated_classes.append(cls)
        except Exception as e:
            print(f"Failed to register zozo_contact_solver.{name}: {e}")

    # Register state_ops operators
    from . import state_ops

    state_ops.register()

    # Make 'from zozo_contact_solver import solver' work in scripts
    import sys

    from . import api

    sys.modules["zozo_contact_solver"] = api

    print(f"Registered {len(_generated_classes)} zozo_contact_solver.* operators")


def unregister():
    """Unregister all auto-generated operators."""
    global _generated_classes

    # Unregister state_ops first
    try:
        from . import state_ops

        state_ops.unregister()
    except Exception:
        pass

    import sys

    sys.modules.pop("zozo_contact_solver", None)

    for cls in reversed(_generated_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _generated_classes.clear()
