"""General Blender operation handlers using the decorator system.

This module contains MCP handlers for basic Blender operations like scene management,
object creation, material handling, and other general Blender functionality.
"""

import io
import sys

from typing import Optional

import bmesh  # pyright: ignore
import bpy  # pyright: ignore

from mathutils import Vector  # pyright: ignore

from ..models.groups import get_addon_data
from .decorators import MCPError, ValidationError, mcp_handler


@mcp_handler
def run_python_script(code: str):
    """Execute arbitrary Python code in Blender with access to bpy, bmesh, and mathutils modules.

    Args:
        code: Python code to execute in Blender context
    """
    # Capture stdout to return any printed output
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        # Create a safe execution environment with common Blender modules
        exec_globals = {
            "__builtins__": __builtins__,
            "bpy": bpy,
            "bmesh": bmesh,
            "mathutils": __import__("mathutils"),
            "Vector": Vector,
        }

        # Execute the code
        exec(code, exec_globals)

        # Get any printed output
        output = captured_output.getvalue()

        return {
            "message": "Python script executed successfully",
            "output": output if output else "No output",
            "success": True,
        }

    except Exception as e:
        return {
            "message": f"Python script execution failed: {str(e)}",
            "output": captured_output.getvalue(),
            "success": False,
            "error": str(e),
        }

    finally:
        # Restore stdout
        sys.stdout = old_stdout


@mcp_handler
def capture_viewport_image(filepath: str, max_size: int = 800):
    """Capture a screenshot of the current 3D viewport and save it to specified file path.

    Args:
        filepath: File path where to save the screenshot
        max_size: Maximum size in pixels for the largest dimension
    """
    # Find the 3D viewport area
    viewport_area = None
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            viewport_area = area
            break

    if not viewport_area:
        raise MCPError("No 3D viewport found")

    # Set up render settings for viewport screenshot
    scene = bpy.context.scene
    original_filepath = scene.render.filepath
    original_resolution_x = scene.render.resolution_x
    original_resolution_y = scene.render.resolution_y

    try:
        # Set render settings
        scene.render.filepath = filepath

        # Calculate resolution maintaining aspect ratio
        current_x = viewport_area.width
        current_y = viewport_area.height

        if current_x > current_y:
            new_x = min(max_size, current_x)
            new_y = int((current_y / current_x) * new_x)
        else:
            new_y = min(max_size, current_y)
            new_x = int((current_x / current_y) * new_y)

        scene.render.resolution_x = new_x
        scene.render.resolution_y = new_y

        # Override context to render from 3D viewport
        with bpy.context.temp_override(area=viewport_area):
            bpy.ops.render.opengl(write_still=True)

        return {
            "message": f"Viewport screenshot saved to '{filepath}'",
            "filepath": filepath,
            "resolution": [new_x, new_y],
            "max_size": max_size,
        }

    finally:
        # Restore original render settings
        scene.render.filepath = original_filepath
        scene.render.resolution_x = original_resolution_x
        scene.render.resolution_y = original_resolution_y


# ============================================================================
# UI Element Status Functions
# ============================================================================


@mcp_handler
def get_ui_element_status(
    element_type: str = "all",
    element_name: Optional[str] = None,
    category: Optional[str] = None,
):
    """Get status of Blender addon UI elements - poll results for operators, values for properties.

    Args:
        element_type: Type of elements to check ("operator", "property", "all")
        element_name: Specific element name to check (optional)
        category: Filter by category ("solver", "dynamics", "client", "debug")
    """
    result = {"operators": [], "properties": []}

    # Validate element_type
    if element_type not in ["operator", "property", "all"]:
        raise ValidationError("element_type must be 'operator', 'property', or 'all'")

    try:
        # Get operators if requested
        if element_type in ["operator", "all"]:
            operators = _get_operator_status(element_name, category)
            result["operators"] = operators

        # Get properties if requested
        if element_type in ["property", "all"]:
            properties = _get_property_status(element_name, category)
            result["properties"] = properties

        return {
            "elements": result,
            "total_operators": len(result["operators"]),
            "total_properties": len(result["properties"]),
        }

    except Exception as e:
        raise MCPError(f"Failed to get UI element status: {str(e)}") from e


def _get_operator_status(
    element_name: Optional[str], category: Optional[str]
) -> list[dict]:
    """Get status of all operators with poll method results."""
    operators = []

    # Import UI modules to discover operators
    try:
        from ..ui import dynamics, solver

        # Define operator mappings with categories
        operator_modules = {"solver": solver, "dynamics": dynamics}

        for module_name, module in operator_modules.items():
            # Skip if category filter doesn't match
            if category and category != module_name:
                continue

            # Get all operator classes from the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's an operator class
                if hasattr(attr, "__bases__") and any(
                    "Operator" in base.__name__ for base in attr.__bases__
                ):
                    # Skip if element_name filter doesn't match
                    if element_name and attr_name != element_name:
                        continue

                    try:
                        # Get operator info
                        op_info = {
                            "name": attr_name,
                            "bl_idname": getattr(attr, "bl_idname", "unknown"),
                            "description": attr.__doc__.split("\n")[0].strip()
                            if attr.__doc__
                            else "",
                            "category": module_name,
                            "clickable": False,
                        }

                        # Call poll method if it exists
                        if hasattr(attr, "poll") and callable(attr.poll):
                            try:
                                context = bpy.context
                                poll_result = attr.poll(context)
                                op_info["clickable"] = bool(poll_result)
                            except Exception as poll_error:
                                op_info["clickable"] = False
                                op_info["poll_error"] = str(poll_error)

                        operators.append(op_info)

                    except Exception:
                        # If we can't process this operator, skip it
                        continue

    except Exception as e:
        raise MCPError(f"Failed to discover operators: {str(e)}") from e

    return operators


def _get_property_status(
    element_name: Optional[str], category: Optional[str]
) -> list[dict]:
    """Get current values of scene properties by dynamically discovering them."""
    properties = []

    try:
        scene = bpy.context.scene

        # Dynamically discover State properties (solver/debug categories)
        addon_data = get_addon_data(scene)
        if addon_data and hasattr(addon_data, "state"):
            state_properties = _discover_property_group_properties(
                addon_data.state, "state", _categorize_state_property
            )
            properties.extend(
                _filter_properties(state_properties, element_name, category)
            )

        # Dynamically discover SSH State properties (client category)
        if addon_data and hasattr(addon_data, "ssh_state"):
            ssh_properties = _discover_property_group_properties(
                addon_data.ssh_state, "ssh_state", lambda name: "client"
            )
            properties.extend(
                _filter_properties(ssh_properties, element_name, category)
            )

        # Dynamically discover Object group properties (dynamics category)
        if category in [None, "dynamics"]:
            from ..models.groups import iterate_active_object_groups

            for i, group in enumerate(iterate_active_object_groups(scene)):
                # Read-only: never call ensure_uuid() here; this runs in
                # a timer context where ID-property writes are forbidden.
                group_uuid = group.uuid if group.uuid else f"idx{i}"
                prefix = f"group_{group_uuid}"
                group_properties = _discover_property_group_properties(
                    group, prefix, lambda name: "dynamics"
                )
                properties.extend(
                    _filter_properties(group_properties, element_name, category)
                )

    except Exception as e:
        raise MCPError(f"Failed to get property values: {str(e)}") from e

    return properties


def _discover_property_group_properties(
    prop_group, prefix: str, categorize_func
) -> list[dict]:
    """Dynamically discover properties in a Blender PropertyGroup using introspection."""
    properties = []

    try:
        # Use Blender's bl_rna.properties to get property definitions
        if hasattr(prop_group, "bl_rna") and hasattr(prop_group.bl_rna, "properties"):
            bl_properties = prop_group.bl_rna.properties

            for attr_name in bl_properties:
                # Skip private/internal attributes
                if attr_name.startswith("_") or attr_name in [
                    "bl_rna",
                    "rna_type",
                    "name",
                ]:
                    continue

                try:
                    # Get property value
                    attr_value = getattr(prop_group, attr_name)

                    # Skip methods and functions
                    if callable(attr_value):
                        continue

                    # Get property info from bl_rna
                    prop_info = bl_properties[attr_name]

                    prop_name = (
                        f"{prefix}_{attr_name}" if prefix != attr_name else attr_name
                    )
                    prop_type = _get_blender_rna_property_type(prop_info)
                    prop_category = categorize_func(attr_name)

                    # Serialize the value to ensure JSON compatibility
                    serialized_value = _serialize_property_value(attr_value)

                    properties.append(
                        {
                            "name": prop_name,
                            "value": serialized_value,
                            "type": prop_type,
                            "category": prop_category,
                        }
                    )

                except Exception:
                    # If we can't get property info, skip it
                    continue

    except Exception:
        # If RNA introspection fails, fall back to simple attribute iteration
        for attr_name in dir(prop_group):
            if (
                not attr_name.startswith("_")
                and attr_name not in ["bl_rna", "rna_type"]
                and not callable(getattr(prop_group, attr_name, None))
            ):
                try:
                    attr_value = getattr(prop_group, attr_name)
                    prop_name = (
                        f"{prefix}_{attr_name}" if prefix != attr_name else attr_name
                    )
                    prop_type = _infer_type_from_value(attr_value)
                    prop_category = categorize_func(attr_name)

                    # Serialize the value to ensure JSON compatibility
                    serialized_value = _serialize_property_value(attr_value)

                    properties.append(
                        {
                            "name": prop_name,
                            "value": serialized_value,
                            "type": prop_type,
                            "category": prop_category,
                        }
                    )
                except Exception:
                    continue

    return properties


def _get_blender_rna_property_type(prop_info) -> str:
    """Get the type string for a Blender RNA property info."""
    try:
        # Get the property type from RNA info
        if hasattr(prop_info, "type"):
            prop_type = prop_info.type

            # Check for specific subtypes first
            if hasattr(prop_info, "subtype") and prop_info.subtype != "NONE":
                subtype = prop_info.subtype
                if subtype == "FILE_PATH":
                    return "file_path"
                elif subtype == "COLOR":
                    return "color"

            # Map RNA types to our type strings
            type_mapping = {
                "STRING": "string",
                "INT": "int",
                "FLOAT": "float",
                "BOOLEAN": "bool",
                "ENUM": "enum",
                "FLOAT_VECTOR": "float_vector",
                "COLLECTION": "collection",
                "POINTER": "pointer",
            }
            return type_mapping.get(prop_type, "unknown")

    except Exception:
        pass

    return "unknown"


def _serialize_property_value(value):
    """Serialize property value to ensure JSON compatibility."""
    try:
        # Handle basic types that are already JSON-serializable
        if isinstance(value, str | int | float | bool | type(None)):
            return value

        # Handle Blender collections and special objects
        if hasattr(value, "__len__") and not isinstance(value, str):
            # Try to convert collections to lists, handling Blender PropertyGroup items
            try:
                serialized_list = []
                for item in value:
                    # Handle Blender PropertyGroup items (like AssignedObject)
                    if hasattr(item, "bl_rna"):
                        # For PropertyGroup items, extract key properties
                        entry = {"type": str(type(item).__name__)}
                        if hasattr(item, "name"):
                            entry["name"] = item.name
                        if hasattr(item, "uuid") and item.uuid:
                            entry["uuid"] = item.uuid
                        serialized_list.append(entry)
                    else:
                        # For regular items, try direct serialization
                        try:
                            serialized_list.append(item)
                        except Exception:
                            serialized_list.append(str(item))
                return serialized_list
            except (TypeError, ValueError):
                # If collection conversion fails, return a string representation
                return str(value)

        # For any other type, convert to string
        return str(value)

    except Exception:
        # If all else fails, return a safe string representation
        return "<unserializable>"


def _infer_type_from_value(value) -> str:
    """Infer property type from its current value."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "string"
    elif hasattr(value, "__len__") and not isinstance(value, str):
        return "collection"
    else:
        return "unknown"


def _categorize_state_property(prop_name: str) -> str:
    """Categorize state properties into solver or debug categories."""
    debug_props = {
        "server_script",
        "shell_command",
        "mcp_port",
        "max_console_lines",
        "use_shell",
        "data_size",
        "log_file_path",
    }
    return "debug" if prop_name in debug_props else "solver"


def _filter_properties(
    properties: list[dict], element_name: Optional[str], category: Optional[str]
) -> list[dict]:
    """Filter properties based on element_name and category filters."""
    filtered = []

    for prop in properties:
        # Skip if category filter doesn't match
        if category and prop["category"] != category:
            continue

        # Skip if element_name filter doesn't match
        if element_name and prop["name"] != element_name:
            continue

        filtered.append(prop)

    return filtered


# ============================================================================
# UI Management Functions
# ============================================================================


@mcp_handler
def get_average_edge_length(object_name: str):
    """Compute the average edge length of a mesh object.

    Args:
        object_name: Name of the mesh object to analyze
    """
    # Get the object from the scene
    obj = bpy.data.objects.get(object_name)

    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")

    if obj.type != "MESH":
        raise MCPError(
            f"Object '{object_name}' is not a mesh object (type: {obj.type})"
        )

    from ..core.uuid_registry import get_object_uuid
    obj_uuid = get_object_uuid(obj)  # read-only: timer context forbids ID writes
    mesh = obj.data

    if len(mesh.edges) == 0:
        return {
            "object_name": object_name,
            "object_uuid": obj_uuid,
            "average_edge_length": 0.0,
            "total_edges": 0,
            "message": "Object has no edges",
        }

    # Calculate edge lengths in world space
    world_mat = obj.matrix_world
    total_edge_length = 0.0

    for edge in mesh.edges:
        v1_world = world_mat @ mesh.vertices[edge.vertices[0]].co
        v2_world = world_mat @ mesh.vertices[edge.vertices[1]].co
        edge_length = (v2_world - v1_world).length
        total_edge_length += edge_length

    average_length = total_edge_length / len(mesh.edges)

    return {
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "average_edge_length": average_length,
        "total_edges": len(mesh.edges),
        "total_vertices": len(mesh.vertices),
        "message": f"Average edge length computed for '{object_name}'",
    }


@mcp_handler
def get_object_bounding_box_diagonal(object_name: str):
    """Compute the bounding box of an object and return the largest diagonal distance.

    Args:
        object_name: Name of the object to analyze
    """
    # Get the object from the scene
    obj = bpy.data.objects.get(object_name)

    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")

    from ..core.uuid_registry import get_object_uuid
    obj_uuid = get_object_uuid(obj)  # read-only: timer context forbids ID writes

    # Get bounding box in local coordinates
    bbox_corners = [Vector(corner) for corner in obj.bound_box]

    # Transform to world coordinates
    world_mat = obj.matrix_world
    world_bbox_corners = [world_mat @ corner for corner in bbox_corners]

    # Find min and max coordinates for each axis
    min_x = min(corner.x for corner in world_bbox_corners)
    max_x = max(corner.x for corner in world_bbox_corners)
    min_y = min(corner.y for corner in world_bbox_corners)
    max_y = max(corner.y for corner in world_bbox_corners)
    min_z = min(corner.z for corner in world_bbox_corners)
    max_z = max(corner.z for corner in world_bbox_corners)

    # Calculate bounding box dimensions
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z

    # Calculate the main diagonal (3D diagonal of the bounding box)
    diagonal_distance = (Vector((width, height, depth))).length

    return {
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "bounding_box": {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
            "center": [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2],
            "dimensions": [width, height, depth],
        },
        "largest_diagonal_distance": diagonal_distance,
        "message": f"Bounding box diagonal computed for '{object_name}'",
    }


@mcp_handler
def refresh_ui():
    """Refresh all UI areas in Blender to reflect recent changes.

    This is useful when programmatic changes need to be reflected in the UI,
    such as after starting/stopping servers or updating addon state.
    """
    import bpy  # pyright: ignore

    # Refresh all areas in the current screen
    for area in bpy.context.screen.areas:
        area.tag_redraw()

    return "UI refresh completed successfully"
