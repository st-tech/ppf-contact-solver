# File: scenarios/bl_mcp_hair_pin_weights.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The per-point weights of Hair Curves, through the MCP server (issue #146).
#
# ``add_pin_vertex_group`` takes ``weights`` beside ``indices`` on a Hair
# Curves object and stores the pin as a float point attribute, which is what
# makes its pull fade along a strand. A tool's schema is closed, so an
# argument it does not declare is refused before the handler runs: reaching
# the handler at all with ``weights`` is the first thing to prove. A ROD
# group also takes a spatial material map, on what a rod segment reads.
#
# A refusal is asserted on the tool payload (``status`` "error" and a
# message), and has to leave the object as it was.
#
# Subtests:
#   A. ``weights_is_a_declared_argument`` -- tools/list shows ``weights`` on
#      add_pin_vertex_group, an array of numbers, and not among the required.
#   B. ``weighted_pin_is_stored_and_listed`` -- the call writes a float point
#      attribute holding exactly the weights given, list_pins reports that pin
#      as weighted and the derived Roots pin as not, and the pin is listed
#      after Roots, which is the order a caller has to change for the roots
#      to stay exact.
#   C. ``weights_are_refused_where_they_mean_nothing`` -- on a mesh, without
#      indices, with a zero weight and with one weight too few, each by name
#      and with nothing written.
#   D. ``a_rod_group_takes_a_map`` -- list_material_maps offers a ROD group
#      the seven keys a segment reads, add_material_map stores one on an
#      attribute source, and a key only a triangle reads is refused.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

PLATFORMS = ("linux", "win32")
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

HAIR_NAME = "McpHair"
MESH_NAME = "McpSheet"
ROD_KEYS = [
    "young-mod", "bend", "friction", "deformation-damping",
    "bending-damping", "strain-limit", "bend-plasticity",
]


def build_hair(name):
    # Three strands of five points each, along +X.
    curves = bpy.data.hair_curves.new(name + "Data")
    curves.add_curves([5, 5, 5])
    points = [
        (0.05 * i, 0.0, 0.1 * c) for c in range(3) for i in range(5)
    ]
    curves.attributes["position"].data.foreach_set(
        "vector", np.array(points, dtype=np.float32).ravel())
    obj = bpy.data.objects.new(name, curves)
    bpy.context.collection.objects.link(obj)
    return obj


def build_sheet(name):
    coords = [(col * 0.5, 2.0 + row * 0.5, 0.0)
              for row in range(3) for col in range(3)]
    faces = [
        (row * 3 + col, row * 3 + col + 1, row * 3 + col + 4, row * 3 + col + 3)
        for row in range(2) for col in range(2)
    ]
    mesh = bpy.data.meshes.new(name + "Data")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj.vertex_groups.new(name="Row").add([0, 1, 2], 1.0, "REPLACE")
    return obj


def float_attribute(obj, name):
    attribute = obj.data.attributes.get(name)
    if attribute is None or attribute.data_type != "FLOAT":
        return None
    values = np.empty(len(attribute.data), dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values.tolist()


try:
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    hair = build_hair(HAIR_NAME)
    sheet = build_sheet(MESH_NAME)
    result["phases"].append((time.time(), "scene_built"))

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    next_id = [10]

    def tool(name, arguments):
        next_id[0] += 1
        payload, _raw = mcp_tool(pkg, url, name, arguments, request_id=next_id[0])
        return payload

    # ----- A. the argument is declared ----------------------------
    env, resp = mcp_call(pkg, url, "tools/list", request_id=1)
    schemas = {
        entry.get("name"): entry
        for entry in ((env.get("result") or {}).get("tools") or [])
    }
    schema = (schemas.get("add_pin_vertex_group") or {}).get("inputSchema") or {}
    weights_schema = (schema.get("properties") or {}).get("weights") or {}
    mcp_check(
        result, "A_weights_is_a_declared_argument",
        "weights" in (schema.get("properties") or {})
        and "array" in json.dumps(weights_schema)
        and "number" in json.dumps(weights_schema)
        and "weights" not in (schema.get("required") or [])
        and schema.get("additionalProperties") is False,
        {"weights": weights_schema, "required": schema.get("required"),
         "additionalProperties": schema.get("additionalProperties")},
    )

    tool("delete_all_groups", {})
    rod = tool("create_group", {"name": "Hair", "type": "ROD"})
    rod_uuid = rod.get("group_uuid") or ""
    cloth = tool("create_group", {"name": "Cloth", "type": "SHELL"})
    cloth_uuid = cloth.get("group_uuid") or ""
    tool("add_objects_to_group",
         {"group_uuid": rod_uuid, "object_names": [HAIR_NAME]})
    tool("add_objects_to_group",
         {"group_uuid": cloth_uuid, "object_names": [MESH_NAME]})

    # ----- B. a weighted pin, stored and listed -------------------
    tool("add_pin_vertex_group", {
        "group_uuid": rod_uuid,
        "vertex_group_identifier": HAIR_NAME + "::Roots",
    })
    added = tool("add_pin_vertex_group", {
        "group_uuid": rod_uuid,
        "vertex_group_identifier": HAIR_NAME + "::Goal",
        "indices": [0, 1, 2, 3, 10, 11, 12],
        "weights": [1.0, 1.0, 0.5, 0.25, 1.0, 0.75, 0.125],
    })
    stored = float_attribute(hair, "Goal")
    expected = [0.0] * 15
    for index, weight in zip(
            [0, 1, 2, 3, 10, 11, 12], [1.0, 1.0, 0.5, 0.25, 1.0, 0.75, 0.125]):
        expected[index] = weight
    listed = tool("list_pins", {"group_uuid": rod_uuid})
    pins = {
        entry.get("vertex_group_name"): entry
        for entry in listed.get("pins") or []
    }
    order = [entry.get("vertex_group_name") for entry in listed.get("pins") or []]
    mcp_check(
        result, "B_weighted_pin_is_stored_and_listed",
        added.get("status") == "success"
        and stored == expected
        and (pins.get("Goal") or {}).get("weighted") is True
        and (pins.get("Roots") or {}).get("weighted") is False
        and order == ["Roots", "Goal"],
        {"added": added, "stored": stored, "order": order,
         "weighted": {name: entry.get("weighted") for name, entry in pins.items()}},
    )

    # ----- C. where weights mean nothing --------------------------
    on_mesh = tool("add_pin_vertex_group", {
        "group_uuid": cloth_uuid,
        "vertex_group_identifier": MESH_NAME + "::Row",
        "weights": [1.0, 0.5, 0.25],
    })
    no_indices = tool("add_pin_vertex_group", {
        "group_uuid": rod_uuid,
        "vertex_group_identifier": HAIR_NAME + "::Goal",
        "weights": [1.0],
    })
    zero = tool("add_pin_vertex_group", {
        "group_uuid": rod_uuid,
        "vertex_group_identifier": HAIR_NAME + "::Zeroed",
        "indices": [5, 6],
        "weights": [1.0, 0.0],
    })
    short = tool("add_pin_vertex_group", {
        "group_uuid": rod_uuid,
        "vertex_group_identifier": HAIR_NAME + "::Short",
        "indices": [5, 6],
        "weights": [1.0],
    })
    cloth_pins = tool("list_pins", {"group_uuid": cloth_uuid})
    mcp_check(
        result, "C_weights_are_refused_where_they_mean_nothing",
        on_mesh.get("status") == "error"
        and "MESH" in (on_mesh.get("message") or "")
        and cloth_pins.get("pin_count") == 0
        and no_indices.get("status") == "error"
        and "no indices were given" in (no_indices.get("message") or "")
        and zero.get("status") == "error"
        and "weight of zero" in (zero.get("message") or "")
        and "Zeroed" not in hair.data.attributes
        and short.get("status") == "error"
        and "1 weights were given for 2" in (short.get("message") or "")
        and "Short" not in hair.data.attributes
        and float_attribute(hair, "Goal") == expected,
        {"on_mesh": on_mesh.get("message"), "no_indices": no_indices.get("message"),
         "zero": zero.get("message"), "short": short.get("message"),
         "cloth_pin_count": cloth_pins.get("pin_count")},
    )

    # ----- D. a rod group takes a map -----------------------------
    offered = tool("list_material_maps", {"group_uuid": rod_uuid})
    mapped = tool("add_material_map", {
        "group_uuid": rod_uuid,
        "parameter": "bend",
        "source_type": "ATTRIBUTE",
        "source_name": "Goal",
        "target_value": 50.0,
    })
    refused = tool("add_material_map", {
        "group_uuid": rod_uuid,
        "parameter": "bend-warp",
        "source_type": "ATTRIBUTE",
        "source_name": "Goal",
        "target_value": 1.0,
    })
    after = tool("list_material_maps", {"group_uuid": rod_uuid})
    rows = after.get("maps") or []
    mcp_check(
        result, "D_a_rod_group_takes_a_map",
        offered.get("object_type") == "ROD"
        and sorted(offered.get("available_parameters") or []) == sorted(ROD_KEYS)
        and mapped.get("status") == "success"
        and refused.get("status") == "error"
        and "ROD" in (refused.get("message") or "")
        and after.get("map_count") == 1
        and (rows[0] if rows else {}).get("parameter") == "bend",
        {"available": offered.get("available_parameters"),
         "mapped": mapped.get("status"), "refused": refused.get("message"),
         "map_count": after.get("map_count")},
    )

    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
