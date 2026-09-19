# File: scenarios/bl_mcp_geometry_repair.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP triangulate_degenerate_faces surface, against a real Blender.
#
# This is the targeted repair the Transfer refusal names. A quad whose corner
# sits on, or very near, the straight edge between its neighbors splits along
# the diagonal that puts three nearly collinear vertices in one triangle, and
# the solver has no usable rest shape for that triangle. The repair re-splits
# exactly those faces and leaves every other face of the mesh as the artist
# authored it. That is what separates it from its two neighbors on the same
# panel: triangulate_for_solver rewrites every quad and n-gon of every object
# it is given, and symmetric_triangulate pokes every face and adds a vertex to
# each. Check C measures the narrow scope on a mesh carrying one flagged quad
# among three sound ones, and check F runs triangulate_for_solver over a mesh
# this repair refuses outright and watches all four quads become triangles, so
# the two scopes are measured against each other rather than described.
#
# The tool takes no object argument, because the operator underneath offers no
# property to narrow its scope: it repairs the included meshes of every active
# object group except SAND, one entry per mesh datablock. Every phase below
# therefore establishes the scope by deleting the active groups and creating
# one holding exactly the objects that phase is about.
#
# Three outcomes are reachable and only one of them is a success, so each is
# asserted separately: a repair that cleared at least one face reports what it
# split, a scan that finds nothing flagged is refused rather than reported as
# a zero-face success, and a scan whose flagged faces have no sound split is
# refused with the remedy that does apply. A refusal here is a normal
# tools/call result carrying isError, so the checks below read it through
# mcp_call and assert on the message rather than only on the failure.
#
# Assertions:
#   A. ``repair_takes_no_object_argument`` -- tools/list gives this tool an
#      empty input schema and no required member, while triangulate_for_solver
#      requires object_names, which is the scope difference in the schema.
#   B. ``repair_reports_the_faces_it_split`` -- one flagged quad among three
#      sound ones is reported as one face repaired and one face added, with
#      the before and after counts, the group object type, and a polygon count
#      the mesh itself agrees with.
#   C. ``only_the_flagged_face_is_split`` -- the mesh comes back with the
#      flagged quad as two triangles and the other three still quads, the
#      vertex count unchanged so no cache is invalidated, and nothing flagged.
#   D. ``repaired_mesh_reports_nothing_left`` -- running it again on the same
#      scene is refused, naming the scope it examined, rather than claiming a
#      second success, and it changes nothing.
#   E. ``sound_mesh_is_refused`` -- a mesh that never carried a flagged face
#      takes the same refusal, not a success reporting zero.
#   F. ``whole_mesh_tool_rewrites_every_quad`` -- triangulate_for_solver on
#      that same refused mesh turns all four quads into eight triangles, which
#      is the change the targeted repair does not make.
#   G. ``unrepairable_face_is_refused_by_name`` -- a quad with two coincident
#      corners has no sound split, so the call is refused with the object, the
#      face count, and the two tools that do apply, and the mesh is untouched.
#   H. ``partial_repair_names_what_is_left`` -- a mesh carrying both a
#      repairable quad and an already degenerate triangle reports one face
#      repaired and the triangle under still_degenerate, and its message
#      withholds the invitation to transfer again.
#   I. ``non_mesh_object_is_refused`` -- a curve assigned to an active ROD
#      group is never examined, so the call is refused, and naming that curve
#      to triangulate_for_solver is refused as a non-mesh object.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
import bmesh

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

SLIVER = "SliverPanel"
CLEAN = "SoundPanel"
STUCK = "CoincidentPanel"
PARTIAL = "PartialPanel"
CURVE = "RodCurve"

# One flagged quad, in the shape issue #144 reported: corner 1 sits 1e-06 off
# the straight line from corner 0 to corner 2, so the 0-2 diagonal puts three
# nearly collinear vertices in one triangle and the 1-3 diagonal does not. Its
# conditioning is 0.4 * 1e-06, three orders below sqrt(float32 eps), and the
# BEAUTY re-split clears it, so this is the face the repair targets.
SLIVER_QUAD = [
    (0.0, 0.0, 0.0),
    (1.0, 1.0e-06, 0.0),
    (2.0, 0.0, 0.0),
    (1.0, -1.0, 0.0),
]

# Already a triangle, so it is its own only triangulation and no re-split
# reaches it: corner 1 sits on the line between the other two.
COLLINEAR_TRI = [
    (0.0, 3.0, 0.0),
    (1.0, 3.0 + 1.0e-07, 0.0),
    (2.0, 3.0, 0.0),
]

# Two coincident corners, so the zero-length boundary edge they form belongs
# to a degenerate triangle of every triangulation.
COINCIDENT_QUAD = [
    (0.0, -3.0, 0.0),
    (0.0, -3.0, 0.0),
    (2.0, -3.0, 0.0),
    (0.0, -1.0, 0.0),
]


def sound_quad(x):
    # A unit square: conditioning 1, which no gate flags.
    return [
        (x, 0.0, 0.0),
        (x + 1.0, 0.0, 0.0),
        (x + 1.0, 1.0, 0.0),
        (x, 1.0, 0.0),
    ]


try:
    utils = __import__(
        pkg + ".core.utils", fromlist=["find_degenerate_tessellation"]
    )

    def new_mesh_object(name, polygons):
        # Each face owns its corners, so no two faces share a vertex and a
        # per-face count reads directly off the polygon list.
        me = bpy.data.meshes.new(name + "Mesh")
        obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(obj)
        bm = bmesh.new()
        for corners in polygons:
            bm.faces.new([bm.verts.new(c) for c in corners])
        bm.to_mesh(me)
        bm.free()
        me.update()
        return obj

    def poly_sizes(obj):
        # Corner count per face, sorted. This is what separates a repair that
        # split one quad from one that rewrote the whole mesh.
        return sorted(len(p.vertices) for p in obj.data.polygons)

    def flagged(obj):
        # The add-on's own scan, so the report is measured against the gate
        # the repair and the Transfer both read.
        found = utils.find_degenerate_tessellation(obj)
        return {
            "count": int(found["count"]),
            "polygons": list(found["polygons"]),
            "repairable_polygons": list(found["repairable_polygons"]),
            "triangle_polygons": list(found["triangle_polygons"]),
        }

    # ----- scene: nothing but what each phase builds ---------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    request_ids = [0]

    def next_id():
        request_ids[0] += 1
        return request_ids[0]

    def call_any(name, arguments=None):
        # Returns (payload, tool result). A handler that ran and refused
        # answers with isError and a payload whose status is "error", which is
        # data rather than a transport failure, so both halves are returned.
        envelope, _resp = mcp_call(
            pkg, url, "tools/call",
            {"name": name, "arguments": arguments or {}},
            request_id=next_id(),
        )
        return mcp_tool_payload(envelope), (envelope.get("result") or {})

    def call_ok(name, arguments=None):
        payload, tool_result = call_any(name, arguments)
        if payload.get("status") != "success":
            raise RuntimeError(
                "%s: %r (isError=%r)"
                % (name, payload, tool_result.get("isError"))
            )
        return payload

    def fresh_group(name, group_type, object_names):
        # The repair reads every active group, so each phase starts from no
        # group at all and assigns exactly the objects it is about. The delete
        # is tolerant: its operator does not poll with no group present, which
        # is the state the first phase starts in.
        call_any("delete_all_groups")
        created = call_ok(
            "create_group", {"name": name, "type": group_type}
        )
        group_uuid = created.get("group_uuid") or ""
        if not group_uuid:
            raise RuntimeError("create_group returned no uuid: %r" % (created,))
        added = call_ok(
            "add_objects_to_group",
            {"group_uuid": group_uuid, "object_names": object_names},
        )
        return group_uuid, added

    # ----- A. the schema says this tool names no object ------------
    env_a, _resp_a = mcp_call(pkg, url, "tools/list", request_id=next_id())
    tools = {
        entry.get("name"): entry
        for entry in ((env_a.get("result") or {}).get("tools") or [])
    }
    targeted_schema = (tools.get("triangulate_degenerate_faces") or {}).get(
        "inputSchema"
    ) or {}
    whole_schema = (tools.get("triangulate_for_solver") or {}).get(
        "inputSchema"
    ) or {}
    mcp_check(
        result, "A_repair_takes_no_object_argument",
        targeted_schema.get("type") == "object"
        and targeted_schema.get("properties") == {}
        and not targeted_schema.get("required")
        and whole_schema.get("required") == ["object_names"]
        and "symmetric_triangulate" in tools,
        {
            "targeted_schema": targeted_schema,
            "whole_mesh_required": whole_schema.get("required"),
            "neighbors_registered": sorted(
                name
                for name in tools
                if "triangulate" in (name or "")
            ),
        },
    )

    # ----- B. one flagged quad among three sound ones --------------
    sliver = new_mesh_object(
        SLIVER, [SLIVER_QUAD, sound_quad(4.0), sound_quad(6.0), sound_quad(8.0)]
    )
    fresh_group("SliverCloth", "SHELL", [SLIVER])
    before_b = flagged(sliver)
    sizes_before_b = poly_sizes(sliver)
    verts_before_b = len(sliver.data.vertices)
    payload_b = call_ok("triangulate_degenerate_faces")
    entry_b = (payload_b.get("objects") or [{}])[0]
    message_b = payload_b.get("message") or ""
    mcp_check(
        result, "B_repair_reports_the_faces_it_split",
        payload_b.get("faces_repaired") == 1
        and payload_b.get("faces_added") == 1
        and payload_b.get("objects_repaired") == [SLIVER]
        and payload_b.get("still_degenerate") == []
        and payload_b.get("operator_status") == ["FINISHED"]
        and len(payload_b.get("objects") or []) == 1
        and entry_b.get("object_name") == SLIVER
        and entry_b.get("group_object_types") == ["SHELL"]
        and entry_b.get("faces_repaired") == 1
        and entry_b.get("degenerate_faces_before") == 1
        and entry_b.get("degenerate_faces_after") == 0
        and entry_b.get("degenerate_triangles_before") >= 1
        and entry_b.get("degenerate_triangles_after") == 0
        and entry_b.get("polygons_before") == 4
        and entry_b.get("polygons_after") == 5
        # The aggregate and the per-object numbers describe one edit.
        and entry_b.get("polygons_after") - entry_b.get("polygons_before")
        == payload_b.get("faces_added")
        # And the report is measured against the mesh, not only against itself.
        and len(sliver.data.polygons) == entry_b.get("polygons_after")
        and "Triangulated 1 face(s)" in message_b
        and "Run Transfer again." in message_b,
        {
            "message": message_b,
            "faces_repaired": payload_b.get("faces_repaired"),
            "faces_added": payload_b.get("faces_added"),
            "objects_repaired": payload_b.get("objects_repaired"),
            "still_degenerate": payload_b.get("still_degenerate"),
            "operator_status": payload_b.get("operator_status"),
            "entry": entry_b,
            "scan_before": before_b,
            "polygons_now": len(sliver.data.polygons),
        },
    )

    # ----- C. and it split nothing else ----------------------------
    after_c = flagged(sliver)
    sizes_after_c = poly_sizes(sliver)
    mcp_check(
        result, "C_only_the_flagged_face_is_split",
        sizes_before_b == [4, 4, 4, 4]
        and before_b["polygons"] == [0]
        and before_b["repairable_polygons"] == [0]
        # The flagged quad is now two triangles; the other three are quads.
        and sizes_after_c == [3, 3, 4, 4, 4]
        # A re-split moves no vertex, so no capture or display cache is stale.
        and len(sliver.data.vertices) == verts_before_b
        and after_c["count"] == 0
        and after_c["polygons"] == [],
        {
            "sizes_before": sizes_before_b,
            "sizes_after": sizes_after_c,
            "vertices_before": verts_before_b,
            "vertices_after": len(sliver.data.vertices),
            "scan_before": before_b,
            "scan_after": after_c,
        },
    )

    # ----- D. a second run has nothing left to do ------------------
    payload_d, tool_d = call_any("triangulate_degenerate_faces")
    message_d = payload_d.get("message") or ""
    mcp_check(
        result, "D_repaired_mesh_reports_nothing_left",
        payload_d.get("status") == "error"
        and tool_d.get("isError") is True
        and message_d.startswith("Nothing to repair")
        # The refusal names the scope it examined, so a caller can tell a
        # clean scene from a mesh this repair never looks at.
        and "SAND" in message_d
        and "base cage" in message_d
        and poly_sizes(sliver) == [3, 3, 4, 4, 4]
        and flagged(sliver)["count"] == 0,
        {
            "status": payload_d.get("status"),
            "is_error": tool_d.get("isError"),
            "message": message_d,
            "sizes_now": poly_sizes(sliver),
        },
    )

    # ----- E. a mesh that never carried a flagged face -------------
    clean = new_mesh_object(
        CLEAN,
        [sound_quad(0.0), sound_quad(2.0), sound_quad(4.0), sound_quad(6.0)],
    )
    fresh_group("SoundCloth", "SHELL", [CLEAN])
    scan_e = flagged(clean)
    payload_e, tool_e = call_any("triangulate_degenerate_faces")
    message_e = payload_e.get("message") or ""
    mcp_check(
        result, "E_sound_mesh_is_refused",
        scan_e["count"] == 0
        and payload_e.get("status") == "error"
        and tool_e.get("isError") is True
        and message_e.startswith("Nothing to repair")
        # A refusal and nothing else: no faces_repaired, no objects, no
        # per-object entry reporting a zero-face success.
        and sorted(payload_e) == ["message", "status"]
        and poly_sizes(clean) == [4, 4, 4, 4],
        {
            "scan": scan_e,
            "status": payload_e.get("status"),
            "is_error": tool_e.get("isError"),
            "message": message_e,
            "payload_keys": sorted(payload_e),
            "sizes_now": poly_sizes(clean),
        },
    )

    # ----- F. what the whole-mesh neighbor does to it --------------
    payload_f = call_ok("triangulate_for_solver", {"object_names": [CLEAN]})
    changed_f = (payload_f.get("changed") or [{}])[0]
    mcp_check(
        result, "F_whole_mesh_tool_rewrites_every_quad",
        payload_f.get("changed_count") == 1
        and changed_f.get("object_name") == CLEAN
        and changed_f.get("polygons_before") == 4
        and changed_f.get("polygons_after") == 8
        and changed_f.get("vertices_before") == changed_f.get("vertices_after")
        # Every quad became two triangles, on a mesh the targeted repair
        # refused to touch at all.
        and poly_sizes(clean) == [3] * 8,
        {
            "changed": changed_f,
            "changed_count": payload_f.get("changed_count"),
            "operator_status": payload_f.get("operator_status"),
            "sizes_now": poly_sizes(clean),
        },
    )

    # ----- G. a flagged face no split can rescue -------------------
    stuck = new_mesh_object(STUCK, [COINCIDENT_QUAD])
    fresh_group("StuckCloth", "SHELL", [STUCK])
    scan_g = flagged(stuck)
    payload_g, tool_g = call_any("triangulate_degenerate_faces")
    message_g = payload_g.get("message") or ""
    mcp_check(
        result, "G_unrepairable_face_is_refused_by_name",
        scan_g["count"] >= 1
        and scan_g["repairable_polygons"] == []
        and payload_g.get("status") == "error"
        and tool_g.get("isError") is True
        and "No flagged face here can be repaired by triangulating" in message_g
        and STUCK in message_g
        and "merge_by_distance" in message_g
        and "dissolve_degenerate_faces" in message_g
        # Splitting it would edit the mesh and leave the defect, so it is left
        # exactly as authored.
        and poly_sizes(stuck) == [4]
        and flagged(stuck)["count"] == scan_g["count"],
        {
            "scan": scan_g,
            "status": payload_g.get("status"),
            "is_error": tool_g.get("isError"),
            "message": message_g,
            "sizes_now": poly_sizes(stuck),
        },
    )

    # ----- H. one of each in the same mesh -------------------------
    partial = new_mesh_object(
        PARTIAL, [SLIVER_QUAD, COLLINEAR_TRI, sound_quad(4.0)]
    )
    fresh_group("PartialCloth", "SHELL", [PARTIAL])
    scan_h = flagged(partial)
    payload_h = call_ok("triangulate_degenerate_faces")
    entry_h = (payload_h.get("objects") or [{}])[0]
    still_h = payload_h.get("still_degenerate") or []
    message_h = payload_h.get("message") or ""
    mcp_check(
        result, "H_partial_repair_names_what_is_left",
        scan_h["polygons"] == [0, 1]
        and scan_h["repairable_polygons"] == [0]
        and scan_h["triangle_polygons"] == [1]
        and payload_h.get("faces_repaired") == 1
        and payload_h.get("faces_added") == 1
        and entry_h.get("degenerate_faces_before") == 2
        and entry_h.get("degenerate_faces_after") == 1
        and still_h == [
            {
                "object_name": PARTIAL,
                "degenerate_faces": 1,
                "already_triangles": 1,
            }
        ]
        and "Still degenerate: %s (1)" % PARTIAL in message_h
        and "dissolve_degenerate_faces" in message_h
        # Transfer still refuses this scene, so the message does not invite it.
        and "Run Transfer again." not in message_h
        # The repairable quad became two triangles; the flagged triangle and
        # the sound quad are as they were.
        and poly_sizes(partial) == [3, 3, 3, 4],
        {
            "scan_before": scan_h,
            "message": message_h,
            "entry": entry_h,
            "still_degenerate": still_h,
            "faces_repaired": payload_h.get("faces_repaired"),
            "faces_added": payload_h.get("faces_added"),
            "sizes_now": poly_sizes(partial),
            "scan_after": flagged(partial),
        },
    )

    # ----- I. an object that is not a mesh -------------------------
    bpy.ops.curve.primitive_bezier_curve_add(location=(0.0, 6.0, 0.0))
    curve_obj = bpy.context.active_object
    curve_obj.name = CURVE
    _uuid_i, added_i = fresh_group("RodGroup", "ROD", [CURVE])
    payload_i, tool_i = call_any("triangulate_degenerate_faces")
    message_i = payload_i.get("message") or ""
    payload_j, tool_j = call_any(
        "triangulate_for_solver", {"object_names": [CURVE]}
    )
    message_j = payload_j.get("message") or ""
    mcp_check(
        result, "I_non_mesh_object_is_refused",
        curve_obj.type == "CURVE"
        # A ROD group accepts a curve, so the repair's group walk reaches one.
        and [obj.get("name") for obj in (added_i.get("added_objects") or [])]
        == [CURVE]
        # It is skipped rather than examined, so the scan comes back empty.
        and payload_i.get("status") == "error"
        and tool_i.get("isError") is True
        and message_i.startswith("Nothing to repair")
        # The whole-mesh neighbor takes object names, and refuses this one by
        # name and by type.
        and payload_j.get("status") == "error"
        and tool_j.get("isError") is True
        and "Not mesh object(s)" in message_j
        and "%s (CURVE)" % CURVE in message_j
        and "MESH objects only" in message_j,
        {
            "curve_type": curve_obj.type,
            "added": added_i.get("added_objects"),
            "warnings": added_i.get("warnings"),
            "repair_status": payload_i.get("status"),
            "repair_message": message_i,
            "neighbor_status": payload_j.get("status"),
            "neighbor_message": message_j,
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
