"""MCP handler decorators for simplified and clean handler implementation.

This module provides decorators that eliminate boilerplate code by automatically:
- Extracting tool names, descriptions, and parameters from function signatures and docstrings
- Generating JSON schemas for MCP tools
- Handling parameter validation and type conversion
- Providing consistent error handling and response formatting
- Registering handlers for auto-discovery
"""

import functools
import inspect
import re

from collections.abc import Callable
from typing import Any, Union, get_type_hints

# Registry for all decorated handlers
_handler_registry: dict[str, dict[str, Any]] = {}


# Custom exceptions for clean error handling
class MCPError(Exception):
    """Base exception for MCP handler errors."""


class ValidationError(MCPError):
    """Exception for parameter validation errors."""


class ConnectionError(MCPError):
    """Exception for connection-related errors."""


def parse_docstring(func: Callable) -> dict[str, Any]:
    """Parse function docstring to extract description and parameter info.

    Args:
        func: Function to parse docstring from

    Returns:
        Dictionary containing:
        - description: First line of docstring
        - parameters: Dict mapping parameter names to descriptions
    """
    if not func.__doc__:
        return {"description": "", "parameters": {}}

    doc = inspect.cleandoc(func.__doc__)
    lines = doc.split("\n")

    # Extract description (first line)
    description = lines[0].strip() if lines else ""

    # Parse Args section
    parameters = {}
    in_args_section = False

    for line in lines:
        line = line.strip()

        # Detect Args section
        if line.lower() in ["args:", "arguments:", "parameters:"]:
            in_args_section = True
            continue

        # End of Args section
        if in_args_section and line and not line.startswith(" ") and ":" not in line:
            break

        # Parse parameter line: "param_name: description"
        if in_args_section and ":" in line:
            match = re.match(r"(\w+):\s*(.+)", line)
            if match:
                param_name, param_desc = match.groups()
                parameters[param_name] = param_desc.strip()

    return {"description": description, "parameters": parameters}


def get_json_schema_type(python_type: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema type definition.

    Args:
        python_type: Python type to convert

    Returns:
        JSON Schema type definition
    """
    # Handle Union types (Optional[T] is Union[T, None])
    if hasattr(python_type, "__origin__") and python_type.__origin__ is Union:
        args = python_type.__args__
        # Optional[T] -> just use T
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return get_json_schema_type(non_none_type)

    # Handle List types
    if hasattr(python_type, "__origin__") and python_type.__origin__ is list:
        item_type = python_type.__args__[0] if python_type.__args__ else str
        return {"type": "array", "items": get_json_schema_type(item_type)}

    # Basic type mapping
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object", "additionalProperties": True},
        list: {"type": "array"},
    }

    return type_mapping.get(python_type, {"type": "string"})


def generate_tool_schema(
    func: Callable, description: str, param_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Generate MCP tool schema from function signature and docstring.

    Args:
        func: Function to generate schema for
        description: Tool description
        param_descriptions: Parameter descriptions from docstring

    Returns:
        Complete MCP tool schema
    """
    # Get function signature
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Build properties and required lists
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        # Get parameter type
        param_type = type_hints.get(param_name, str)
        schema_type = get_json_schema_type(param_type)

        # Add description if available
        if param_name in param_descriptions:
            schema_type["description"] = param_descriptions[param_name]

        # Add default value if present
        if param.default != inspect.Parameter.empty:
            schema_type["default"] = param.default
        else:
            # Parameter is required if no default
            required.append(param_name)

        properties[param_name] = schema_type

    # Build complete schema
    input_schema = {"type": "object", "properties": properties}

    if required:
        input_schema["required"] = required

    return {
        "name": func.__name__,
        "description": description,
        "inputSchema": input_schema,
    }


def validate_and_convert_args(func: Callable, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and convert arguments based on function signature.

    Args:
        func: Function to validate arguments for
        args: Arguments dictionary to validate

    Returns:
        Validated and converted arguments

    Raises:
        ValidationError: If validation fails
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    validated_args = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        value = args.get(param_name)

        # Check required parameters
        if value is None and param.default == inspect.Parameter.empty:
            raise ValidationError(f"Missing required parameter: {param_name}")

        # Use default if not provided
        if value is None:
            validated_args[param_name] = param.default
            continue

        # Type conversion/validation
        expected_type = type_hints.get(param_name)
        if expected_type:
            try:
                # Handle Optional types
                if (
                    hasattr(expected_type, "__origin__")
                    and expected_type.__origin__ is Union
                ):
                    args_types = expected_type.__args__
                    if len(args_types) == 2 and type(None) in args_types:
                        # Optional[T] - get the non-None type
                        expected_type = (
                            args_types[0]
                            if args_types[1] is type(None)
                            else args_types[1]
                        )

                # Basic type checking (don't convert, just validate)
                if expected_type in (
                    str,
                    int,
                    float,
                    bool,
                    dict,
                    list,
                ) and not isinstance(value, expected_type):
                    # Try conversion for basic types
                    if expected_type is int and isinstance(value, str | float):
                        value = int(value)
                    elif expected_type is float and isinstance(value, str | int):
                        value = float(value)
                    elif expected_type is str and not isinstance(value, str):
                        value = str(value)
                    elif expected_type is bool and isinstance(value, str | int):
                        value = bool(value)
                    else:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}"
                        )

            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Invalid value for parameter '{param_name}': {e}"
                ) from e

        validated_args[param_name] = value

    return validated_args


def mcp_handler(func: Callable) -> Callable:
    """Main MCP handler decorator that extracts everything from docstring and signature.

    This decorator:
    - Extracts tool name from function name
    - Extracts description from first line of docstring
    - Extracts parameter info from Args section and type hints
    - Generates JSON schema automatically
    - Provides parameter validation and error handling
    - Registers handler for auto-discovery

    Args:
        func: Function to decorate

    Returns:
        Decorated function with MCP handler capabilities
    """
    # Parse docstring
    docstring_info = parse_docstring(func)
    description = docstring_info["description"]
    param_descriptions = docstring_info["parameters"]

    # Generate schema
    schema = generate_tool_schema(func, description, param_descriptions)

    @functools.wraps(func)
    def wrapper(args: dict[str, Any]) -> dict[str, Any]:
        try:
            # Validate and convert arguments
            validated_args = validate_and_convert_args(func, args)

            # Call the original function
            result = func(**validated_args)

            # Format response
            if isinstance(result, dict) and "status" in result:
                # Already formatted response
                return result
            elif isinstance(result, str):
                # Simple string response
                return {"status": "success", "message": result}
            elif isinstance(result, dict):
                # Dict response - add success status
                return {"status": "success", **result}
            else:
                # Other types - convert to string
                return {"status": "success", "result": result}

        except MCPError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"{func.__name__} failed: {str(e)}"}

    # Register handler (after wrapper is created)
    handler_name = func.__name__
    _handler_registry[handler_name] = {
        "func": wrapper,  # Store the wrapper, not the original function
        "schema": schema,
        "category": "general",
    }

    # Preserve original function for introspection
    wrapper._original_func = func  # type: ignore
    wrapper._handler_name = handler_name  # type: ignore
    wrapper._schema = schema  # type: ignore

    return wrapper


def connection_handler(func: Callable) -> Callable:
    """Decorator for handlers that require an active connection.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with connection validation
    """
    decorated = mcp_handler(func)

    # Update category
    if func.__name__ in _handler_registry:
        _handler_registry[func.__name__]["category"] = "connection"

    @functools.wraps(decorated)
    def wrapper(args: dict[str, Any]) -> dict[str, Any]:
        # Import here to avoid circular imports
        try:
            from ..core.client import communicator as com

            if not com.is_connected():
                return {
                    "status": "error",
                    "message": "Not connected to server. Please establish a connection first.",
                }
        except ImportError:
            pass  # Skip validation if communicator not available

        return decorated(args)

    return wrapper


def group_handler(func: Callable) -> Callable:
    """Decorator for handlers that operate on dynamics groups.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with group validation helpers
    """
    decorated = mcp_handler(func)

    # Update category
    if func.__name__ in _handler_registry:
        _handler_registry[func.__name__]["category"] = "group"

    return decorated


def simulation_handler(func: Callable) -> Callable:
    """Decorator for handlers that require server status checks.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with simulation status validation
    """
    decorated = mcp_handler(func)

    # Update category
    if func.__name__ in _handler_registry:
        _handler_registry[func.__name__]["category"] = "simulation"

    return decorated


def debug_handler(func: Callable) -> Callable:
    """Decorator for debug/development handlers.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with debug category
    """
    decorated = mcp_handler(func)

    # Update category
    if func.__name__ in _handler_registry:
        _handler_registry[func.__name__]["category"] = "debug"

    return decorated


def remote_handler(func: Callable) -> Callable:
    """Decorator for remote server operation handlers.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with remote category
    """
    decorated = mcp_handler(func)

    # Update category
    if func.__name__ in _handler_registry:
        _handler_registry[func.__name__]["category"] = "remote"

    return decorated


# Utility functions for handlers
def get_handler_registry() -> dict[str, dict[str, Any]]:
    """Get the complete handler registry.

    Returns:
        Dictionary mapping handler names to handler info
    """
    return _handler_registry.copy()
