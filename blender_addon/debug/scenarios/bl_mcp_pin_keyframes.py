# File: scenarios/bl_mcp_pin_keyframes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP keyframed-pin-animation surface, against a real Blender.
#
# A pin draws its motion from EITHER the parametric operations
# (add_pin_operation, Move/Spin/Scale/Torque) OR vertex position keyframes,
# never both, and each surface refuses a pin the other one owns. That mutual
# exclusion is what this scenario is written around: add_pin_operation is
# refused while the pin carries keys, delete_pin_keyframes is what frees it,
# and add_pin_keyframe is refused again once a parametric op is back on the
# pin. The two handlers wrap the panel's Make Keyframe and Delete All
# Keyframes operators, which own the fcurve layout and the EMBEDDED_MOVE
# marker op that tells the encoder a per-vertex track is to be spliced in, so
# the marker is checked alongside the curves it stands for.
#
# The keys are ordinary Blender keyframes on the MESH, one per pinned vertex
# on a "vertices[i].co" fcurve, so every check reads them back off the mesh's
# own action rather than trusting the handler's report of what it wrote.
# Blender 5.x keeps those curves under action.layers[].strips[].channelbag,
# and the walker below reads the flat action.fcurves layout as well.
#
# The refusals are asserted on the tool payload rather than on the JSON-RPC
# envelope. A handler that ran and rejected its arguments is a well-formed
# result carrying isError, whose payload has status "error" and a message; a
# protocol error is reserved for a call that never reached a tool.
#
# Assertions:
#   A. ``keyframe_tools_are_registered`` -- add_pin_keyframe and
#      delete_pin_keyframes both take the group and the pin identifier, only
#      the deleting one is annotated destructive, and the fresh pin carries no
#      operation and no key.
#   B. ``first_key_lands_at_the_current_frame`` -- add_pin_keyframe keys every
#      pinned vertex at the frame the scene is on, reports that frame and the
#      track it now holds, writes LINEAR keys carrying the positions the mesh
#      held, and puts one EMBEDDED_MOVE marker at the head of the pin's ops.
#   C. ``second_key_reports_the_whole_track`` -- keying a second frame reports
#      both frames, records the pose the mesh holds at that frame, and leaves
#      exactly one marker rather than a second one.
#   D. ``operation_is_refused_while_keyframed`` -- add_pin_operation on the
#      keyframed pin is refused, names the conflict and the op families it
#      covers, and leaves both the track and the marker as they were.
#   E. ``delete_reports_what_it_removed`` -- delete_pin_keyframes reports the
#      frames and the marker it dropped, and the mesh carries no vertex-co key
#      afterwards.
#   F. ``operation_is_accepted_once_keys_are_gone`` -- the same
#      add_pin_operation call now succeeds and the op is the pin's only one.
#   G. ``keyframe_is_refused_while_operations_exist`` -- the mirror refusal:
#      add_pin_keyframe names the op families and the two tools that clear
#      them, and writes no key.
#   H. ``frame_below_one_is_refused`` -- a key under frame 1 is dropped when
#      the pin track is read, so the scene's frame is refused by number before
#      the operator writes one, and no marker is left behind.
#   I. ``empty_vertex_group_is_refused`` -- a pin whose vertex group holds no
#      vertex has nothing to key, and is refused instead of being marked
#      animated.
#   J. ``missing_vertex_group_is_refused`` -- a pin whose vertex group is gone
#      from the object is refused by name, naming both the group and the
#      object.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")
# No solver, no build and no connection: bpy datablock calls, two panel
# operators and the add-on's own HTTP server, so the assertions hold
# identically on either backend.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

import re

GROUP_NAME = "KeyedCloth"
MESH_NAME = "PinnedSheet"
EMPTY_MESH = "EmptiedSheet"
LOST_MESH = "LostGroupSheet"
VG_NAME = "PinRow"
EMPTY_VG = "EmptiedRow"
LOST_VG = "GoneRow"
PIN_INDICES = [0, 1, 2]
FIRST_FRAME = 5
SECOND_FRAME = 9
POSED_Z = 0.75
Z_AXIS = 2

VERTEX_CO_RGX = re.compile(r"vertices\[(\d+)\]\.co$")


def build_grid(name):
    # A 3 by 3 vertex sheet of four quads, built from the datablock API so the
    # vertex indices this scenario pins are the ones from_pydata assigned.
    coords = [(col * 0.5, row * 0.5, 0.0) for row in range(3) for col in range(3)]
    faces = [
        (row * 3 + col, row * 3 + col + 1, row * 3 + col + 4, row * 3 + col + 3)
        for row in range(2)
        for col in range(2)
    ]
    mesh = bpy.data.meshes.new(name + "Data")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


def vertex_curves(obj):
    # Every vertex-co fcurve on the mesh's action, as
    # {"<vertex>:<axis>": [[frame, value, interpolation], ...]}. Read from
    # Blender rather than through the add-on, so what a handler reports is
    # measured against what the mesh actually carries. Blender 5.x keeps the
    # curves under action.layers[].strips[].channelbag(slot) and leaves
    # action.fcurves empty; the flat layout is read too so the walk does not
    # depend on the version.
    curves = {}
    anim = obj.data.animation_data
    action = anim.action if anim is not None else None
    if action is None:
        return curves

    def consume(fcurve):
        match = VERTEX_CO_RGX.match(fcurve.data_path)
        if match is None or not 0 <= fcurve.array_index < 3:
            return
        curves["%d:%d" % (int(match.group(1)), fcurve.array_index)] = [
            [
                int(round(point.co[0])),
                round(float(point.co[1]), 4),
                point.interpolation,
            ]
            for point in fcurve.keyframe_points
        ]

    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in layer.strips:
                for slot in action.slots:
                    channelbag = strip.channelbag(slot)
                    if channelbag is None:
                        continue
                    for fcurve in channelbag.fcurves:
                        consume(fcurve)
    elif hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            consume(fcurve)
    return curves


def keyed_frames(obj):
    frames = set()
    for points in vertex_curves(obj).values():
        for frame, _value, _interp in points:
            frames.add(frame)
    return sorted(frames)


def keyed_vertices(obj):
    return sorted({int(key.split(":")[0]) for key in vertex_curves(obj)})


def keyed_value(obj, vertex_index, axis, frame):
    # The value one key holds, or None when that vertex carries no key there.
    for point_frame, value, _interp in vertex_curves(obj).get(
        "%d:%d" % (vertex_index, axis), []
    ):
        if point_frame == frame:
            return value
    return None


def key_interpolations(obj):
    return sorted(
        {interp for points in vertex_curves(obj).values() for _f, _v, interp in points}
    )


def pose_pinned_vertices(obj, height):
    # Lift the pinned vertices so the second key records a different pose from
    # the first, which is what makes the keyed VALUES worth reading back.
    for index in PIN_INDICES:
        vertex = obj.data.vertices[index]
        vertex.co = (vertex.co[0], vertex.co[1], height)
    obj.data.update()


try:
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    mesh_obj = build_grid(MESH_NAME)
    empty_obj = build_grid(EMPTY_MESH)
    lost_obj = build_grid(LOST_MESH)
    scene = bpy.context.scene
    scene.frame_set(FIRST_FRAME)
    result["phases"].append((time.time(), "scene_built"))

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    request_id = {"next": 1}

    def call(name, arguments):
        # A refusal is data, not a transport failure: a handler that ran and
        # refused answers with isError and a payload whose status is "error",
        # so both halves are returned to the caller.
        request_id["next"] += 1
        return mcp_tool(pkg, url, name, arguments, request_id=request_id["next"])

    def must(name, arguments):
        payload, _raw = call(name, arguments)
        if payload.get("status") != "success":
            raise RuntimeError("%s: %r" % (name, payload))
        return payload

    def pin_id(object_name, vertex_group_name):
        return object_name + "::" + vertex_group_name

    def key_pin(identifier):
        return call(
            "add_pin_keyframe",
            {"group_uuid": group_uuid, "vertex_group_identifier": identifier},
        )

    def move_op(identifier):
        return call(
            "add_pin_operation",
            {
                "group_uuid": group_uuid,
                "vertex_group_identifier": identifier,
                "op_type": "MOVE_BY",
                "delta": [0.0, 0.0, 1.0],
                "frame_start": 1,
                "frame_end": 10,
            },
        )

    def op_types(identifier):
        listed = must(
            "list_pin_operations",
            {"group_uuid": group_uuid, "vertex_group_identifier": identifier},
        )
        return [entry.get("op_type") for entry in listed.get("operations") or []]

    # ----- the scene the assertions run against -------------------
    group_uuid = must(
        "create_group", {"name": GROUP_NAME, "type": "SHELL"}
    ).get("group_uuid") or ""
    if not group_uuid:
        raise RuntimeError("create_group returned no uuid")
    must(
        "add_objects_to_group",
        {
            "group_uuid": group_uuid,
            "object_names": [MESH_NAME, EMPTY_MESH, LOST_MESH],
        },
    )
    for object_name, vertex_group_name in (
        (MESH_NAME, VG_NAME),
        (EMPTY_MESH, EMPTY_VG),
        (LOST_MESH, LOST_VG),
    ):
        must(
            "create_vertex_group",
            {
                "object_name": object_name,
                "name": vertex_group_name,
                "indices": PIN_INDICES,
            },
        )
        must(
            "add_pin_vertex_group",
            {
                "group_uuid": group_uuid,
                "vertex_group_identifier": pin_id(object_name, vertex_group_name),
            },
        )

    identifier = pin_id(MESH_NAME, VG_NAME)
    empty_identifier = pin_id(EMPTY_MESH, EMPTY_VG)
    lost_identifier = pin_id(LOST_MESH, LOST_VG)

    # One pin keeps its vertex group but loses every member, the other loses
    # the vertex group itself. Both are states the add-on has to survive being
    # asked to key, and neither is reachable through the MCP surface, so they
    # are made here with plain bpy calls.
    empty_obj.vertex_groups[EMPTY_VG].remove(PIN_INDICES)
    lost_obj.vertex_groups.remove(lost_obj.vertex_groups[LOST_VG])
    result["phases"].append((time.time(), "pins_built"))

    # ----- A. the two tools, and a pin with nothing on it ---------
    env, _resp = mcp_call(pkg, url, "tools/list", request_id=1)
    schemas = {
        tool.get("name"): tool
        for tool in ((env.get("result") or {}).get("tools") or [])
    }
    required = {}
    annotations = {}
    for tool_name in ("add_pin_keyframe", "delete_pin_keyframes"):
        schema = schemas.get(tool_name) or {}
        required[tool_name] = sorted(
            (schema.get("inputSchema") or {}).get("required") or []
        )
        annotations[tool_name] = schema.get("annotations") or {}
    expected_required = ["group_uuid", "vertex_group_identifier"]
    mcp_check(
        result, "A_keyframe_tools_are_registered",
        required["add_pin_keyframe"] == expected_required
        and required["delete_pin_keyframes"] == expected_required
        # The annotation is derived from the handler name, and only the
        # deleting half destroys state.
        and annotations["add_pin_keyframe"] == {}
        and annotations["delete_pin_keyframes"] == {"destructiveHint": True}
        and op_types(identifier) == []
        and keyed_frames(mesh_obj) == [],
        {
            "required": required,
            "annotations": annotations,
            "operations": op_types(identifier),
            "keyed_frames": keyed_frames(mesh_obj),
        },
    )

    # ----- B. the first key, at the frame the scene is on ---------
    first, _raw_first = key_pin(identifier)
    first_curves = vertex_curves(mesh_obj)
    mcp_check(
        result, "B_first_key_lands_at_the_current_frame",
        first.get("status") == "success"
        and first.get("frame") == FIRST_FRAME
        and first.get("keyframed_frames") == [FIRST_FRAME]
        and first.get("keyframed_vertex_count") == len(PIN_INDICES)
        and first.get("vertex_group_name") == VG_NAME
        and first.get("object_name") == MESH_NAME
        and first.get("operator_result") == ["FINISHED"]
        # Three axes on each pinned vertex, and nothing on any other vertex.
        and keyed_vertices(mesh_obj) == sorted(PIN_INDICES)
        and len(first_curves) == 3 * len(PIN_INDICES)
        and keyed_frames(mesh_obj) == [FIRST_FRAME]
        # The key records the position the mesh held, and is LINEAR so the
        # sparse track reads the way the solver treats it.
        and keyed_value(mesh_obj, PIN_INDICES[0], Z_AXIS, FIRST_FRAME) == 0.0
        and key_interpolations(mesh_obj) == ["LINEAR"]
        # The marker the encoder reads as "this pin has a keyframed track".
        and op_types(identifier) == ["EMBEDDED_MOVE"],
        {
            "payload": first,
            "keyed_vertices": keyed_vertices(mesh_obj),
            "curve_count": len(first_curves),
            "keyed_frames": keyed_frames(mesh_obj),
            "first_key_z": keyed_value(
                mesh_obj, PIN_INDICES[0], Z_AXIS, FIRST_FRAME
            ),
            "interpolations": key_interpolations(mesh_obj),
            "operations": op_types(identifier),
        },
    )

    # ----- C. a second frame, and the track it reports ------------
    scene.frame_set(SECOND_FRAME)
    pose_pinned_vertices(mesh_obj, POSED_Z)
    second, _raw_second = key_pin(identifier)
    second_z = keyed_value(mesh_obj, PIN_INDICES[0], Z_AXIS, SECOND_FRAME)
    mcp_check(
        result, "C_second_key_reports_the_whole_track",
        second.get("status") == "success"
        and second.get("frame") == SECOND_FRAME
        and second.get("keyframed_frames") == [FIRST_FRAME, SECOND_FRAME]
        and second.get("keyframed_vertex_count") == len(PIN_INDICES)
        and keyed_frames(mesh_obj) == [FIRST_FRAME, SECOND_FRAME]
        # Each key holds the pose its own frame was posed in.
        and keyed_value(mesh_obj, PIN_INDICES[0], Z_AXIS, FIRST_FRAME) == 0.0
        and second_z == POSED_Z
        # Re-keying finds the marker already there and adds no second one.
        and op_types(identifier) == ["EMBEDDED_MOVE"],
        {
            "payload": second,
            "keyed_frames": keyed_frames(mesh_obj),
            "first_key_z": keyed_value(
                mesh_obj, PIN_INDICES[0], Z_AXIS, FIRST_FRAME
            ),
            "second_key_z": second_z,
            "operations": op_types(identifier),
        },
    )

    # ----- D. the parametric surface, on a keyframed pin ----------
    refused_op, raw_op = move_op(identifier)
    op_message = refused_op.get("message") or ""
    mcp_check(
        result, "D_operation_is_refused_while_keyframed",
        refused_op.get("status") == "error"
        and raw_op.get("isError") is True
        and "keyframed" in op_message
        and "Move/Spin/Scale/Torque" in op_message
        # The refusal leaves the pin exactly as it was, marker included.
        and op_types(identifier) == ["EMBEDDED_MOVE"]
        and keyed_frames(mesh_obj) == [FIRST_FRAME, SECOND_FRAME],
        {
            "payload": refused_op,
            "isError": raw_op.get("isError"),
            "message": op_message,
            "operations": op_types(identifier),
            "keyed_frames": keyed_frames(mesh_obj),
        },
    )

    # ----- E. dropping the whole track ---------------------------
    deleted, _raw_deleted = call(
        "delete_pin_keyframes",
        {"group_uuid": group_uuid, "vertex_group_identifier": identifier},
    )
    mcp_check(
        result, "E_delete_reports_what_it_removed",
        deleted.get("status") == "success"
        and deleted.get("removed_frames") == [FIRST_FRAME, SECOND_FRAME]
        and deleted.get("removed_marker_ops") == 1
        and deleted.get("operation_count") == 0
        and deleted.get("object_name") == MESH_NAME
        and deleted.get("operator_result") == ["FINISHED"]
        # The curves are gone from the mesh, not merely emptied of keys.
        and vertex_curves(mesh_obj) == {}
        and op_types(identifier) == [],
        {
            "payload": deleted,
            "curves_after": vertex_curves(mesh_obj),
            "operations": op_types(identifier),
        },
    )

    # ----- F. the same op call, on the freed pin ------------------
    accepted_op, _raw_accepted = move_op(identifier)
    mcp_check(
        result, "F_operation_is_accepted_once_keys_are_gone",
        accepted_op.get("status") == "success"
        and accepted_op.get("op_type") == "MOVE_BY"
        and accepted_op.get("operation_index") == 0
        and accepted_op.get("operation_count") == 1
        and op_types(identifier) == ["MOVE_BY"],
        {"payload": accepted_op, "operations": op_types(identifier)},
    )

    # ----- G. and the refusal in the other direction --------------
    refused_key, raw_key = key_pin(identifier)
    key_message = refused_key.get("message") or ""
    mcp_check(
        result, "G_keyframe_is_refused_while_operations_exist",
        refused_key.get("status") == "error"
        and raw_key.get("isError") is True
        and "Move/Spin/Scale/Torque operations" in key_message
        and "remove_pin_operation" in key_message
        and "clear_pin_operations" in key_message
        and keyed_frames(mesh_obj) == []
        and op_types(identifier) == ["MOVE_BY"],
        {
            "payload": refused_key,
            "isError": raw_key.get("isError"),
            "message": key_message,
            "keyed_frames": keyed_frames(mesh_obj),
            "operations": op_types(identifier),
        },
    )

    # ----- H. a frame the pin track would never be read at --------
    must(
        "clear_pin_operations",
        {"group_uuid": group_uuid, "vertex_group_identifier": identifier},
    )
    scene.frame_set(0)
    refused_frame, raw_frame = key_pin(identifier)
    frame_message = refused_frame.get("message") or ""
    mcp_check(
        result, "H_frame_below_one_is_refused",
        refused_frame.get("status") == "error"
        and raw_frame.get("isError") is True
        and "frame 0" in frame_message
        and "frame 1" in frame_message
        # Refused before the operator ran, so neither a key nor the marker
        # that would claim the pin is animated was written.
        and keyed_frames(mesh_obj) == []
        and op_types(identifier) == [],
        {
            "payload": refused_frame,
            "isError": raw_frame.get("isError"),
            "message": frame_message,
            "keyed_frames": keyed_frames(mesh_obj),
            "operations": op_types(identifier),
            "scene_frame": int(scene.frame_current),
        },
    )
    scene.frame_set(FIRST_FRAME)

    # ----- I. a pin whose vertex group holds no vertex ------------
    refused_empty, raw_empty = key_pin(empty_identifier)
    empty_message = refused_empty.get("message") or ""
    mcp_check(
        result, "I_empty_vertex_group_is_refused",
        refused_empty.get("status") == "error"
        and raw_empty.get("isError") is True
        and empty_identifier in empty_message
        and "covers no vertex" in empty_message
        and keyed_frames(empty_obj) == []
        and op_types(empty_identifier) == [],
        {
            "payload": refused_empty,
            "isError": raw_empty.get("isError"),
            "message": empty_message,
            "keyed_frames": keyed_frames(empty_obj),
            "operations": op_types(empty_identifier),
        },
    )

    # ----- J. a pin whose vertex group is gone --------------------
    refused_lost, raw_lost = key_pin(lost_identifier)
    lost_message = refused_lost.get("message") or ""
    mcp_check(
        result, "J_missing_vertex_group_is_refused",
        refused_lost.get("status") == "error"
        and raw_lost.get("isError") is True
        and LOST_VG in lost_message
        and LOST_MESH in lost_message
        and keyed_frames(lost_obj) == []
        and op_types(lost_identifier) == [],
        {
            "payload": refused_lost,
            "isError": raw_lost.get("isError"),
            "message": lost_message,
            "keyed_frames": keyed_frames(lost_obj),
            "operations": op_types(lost_identifier),
        },
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
