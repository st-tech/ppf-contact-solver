# File: scenarios/bl_resolve_intersections.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resolve Intersections, the Intersection Correction entry of the Utility
# Tools panel, driven on real Blender objects.
#
# Transfer refuses a scene whose surfaces start through each other, and a
# chain converted from Geometry Nodes arrives with a shallow overlap at most
# of its links (public issue #157). The tool pushes SOLID and PDRD objects
# apart as rigid bodies, writing object transforms only
# (`mesh_ops/intersection_ops.py`, `core/intersection_correction.py`).
#
# Whether two surfaces cross is measured here with Blender's own
# triangle-triangle test on the TRUE surfaces, never read from the tool's
# report, so a tool that reported a clean result it did not produce fails.
#
# Two scenes go through Transfer, each refused before and READY after:
#   `A_overlapping_chain_refused_then_builds`: six PDRD links, each tube
#       4 mm inside the next, one of them inside-out. The five overlapping
#       pairs are reported, none is left, and no link moved farther than the
#       overlap could ask for.
#   `A2_refused_overlay_comes_down_and_stays_down`: the refused Transfer of A
#       leaves its violations drawn over the scene as it was uploaded. A run
#       that moves nothing keeps them. The run that moves the links takes
#       them down, and they stay down through the status polls that follow,
#       in which the server reports the refused upload's violations again.
#       They also stay down on every tick of the next Transfer: a server
#       that kept the old violations past the new upload sent them under the
#       new upload's id, and they were drawn again until the build ended.
#   `B_contact_offsets_refused_then_build`: two PDRD cubes that do not cross
#       but sit closer than the sum of their contact offsets, which Transfer
#       refuses as well. They end at least the offsets plus half a clearance
#       apart.
#
# The rest need no solver:
#   `C_pinned_and_animated_are_held`: a pinned link and a link with location
#       keys stay where they are, are named with the reason, and the links
#       between them take the whole correction. The chain between two held
#       ends cannot change length, so it is laid out at a pitch that fits and
#       the overlaps come from links shifted along it.
#   `D_open_mesh_is_left_out_by_name`: a cube missing a face has no inside.
#   `E_shell_and_ungrouped_are_ignored`: a SHELL sheet and a mesh in no
#       group, both crossing a link, are neither moved nor counted.
#   `F_reported_distance_is_the_distance_moved`: for every object, the
#       largest displacement of its own vertices is what the report says.
#   `G_selected_only_moves_the_selection`: of two overlapping SOLID cubes
#       only the selected one moves, and with nothing selected nothing can.
#   `H_parent_and_child_both_land`: a child and its parent are pushed in
#       different directions and each ends where it was solved to. Writing
#       the child first would carry it along with the parent afterwards.
#   `I_panel_draws_every_state`: the panel entry with too few objects, with
#       enough, and with each kind of last-run report, through a recording
#       layout; the button carries the three settings.
#   `J_mcp_tool_and_api`: the MCP tool is registered, narrows to the named
#       objects, refuses bad arguments, and the scripting API reaches it. Two
#       objects the call holds still overlap afterwards, and the result says
#       so instead of calling the scene clear.
#   `K_squeezed_object_is_reported_left`: a block between two held ones that
#       push it opposite ways cannot be freed. Nothing moves, both pairs are
#       reported as left, and the run stops after the pass that moved nothing.
#   `L_small_chain_far_from_origin`: 5 mm links thirty units out, with a
#       contact gap of 85 nanometers, ask for a clearance far below what
#       Blender's single-precision tree resolves there (3.6 micrometers).
#       The clearance is raised to what the tree resolves, and the room left
#       at each joint is measured: left as asked, a joint ends within the
#       tree's rounding of touching, on either side of it.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import math
import traceback

import bmesh
import numpy as np
from mathutils.bvhtree import BVHTree

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT = "resolve_intersections"
BUILD_TIMEOUT = 120.0

# Major and minor radius of a link. Threaded links, alternately in the xy and
# xz planes, clear each other up to a pitch of 2 * (RING - TUBE) = 0.08. At
# 0.084 each tube sits 4 mm inside the next.
RING = 0.05
TUBE = 0.01
PITCH = 0.084
OVERLAP = PITCH - 2.0 * (RING - TUBE)


def fresh():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.api.solver.clear()
    ops.clear_last_report()


def add_link(name, k, flip=False, pitch=PITCH, shift=0.0, scale=1.0,
             origin=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=RING * scale, minor_radius=TUBE * scale,
        major_segments=24, minor_segments=12,
        location=(origin[0] + (k * pitch + shift) * scale, origin[1],
                  origin[2]),
        rotation=(math.pi / 2 if k % 2 else 0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    if flip:
        obj.data.flip_normals()
    return obj


def add_cube(name, size, location):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def world_verts(obj):
    bpy.context.view_layer.update()
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices])


def crossing_pairs(objs):
    # Blender's triangle-triangle test on the true world-space surfaces.
    trees = []
    for obj in objs:
        obj.data.calc_loop_triangles()
        trees.append(BVHTree.FromPolygons(
            [tuple(p) for p in world_verts(obj)],
            [tuple(t.vertices) for t in obj.data.loop_triangles],
            all_triangles=True))
    return sorted(
        (objs[i].name, objs[j].name)
        for i in range(len(objs)) for j in range(i + 1, len(objs))
        if trees[i].overlap(trees[j]))


def snapshot(objs):
    return {o.name: world_verts(o) for o in objs}


def displacement(obj, before):
    return float(np.linalg.norm(world_verts(obj) - before[obj.name], axis=1).max())


def resolve(**arguments):
    # What the panel's button does: the three settings, passed to the
    # operator.
    state = dh.groups.get_addon_data(bpy.context.scene).state
    call = {"max_iterations": state.intersection_fix_iterations,
            "clearance": state.intersection_fix_clearance,
            "selected_only": state.intersection_fix_selected_only}
    call.update(arguments)
    status = bpy.ops.object.resolve_intersections(**call)
    return status, ops.get_last_report()


def build(message, first=False):
    if first:
        dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                   project_name=PROJECT)
    return dh.build_verdict(message, timeout=BUILD_TIMEOUT)


def transfer_state():
    root = dh.configure_state(project_name=PROJECT, frame_count=10,
                              frame_rate=60, step_size=0.01)
    root.state.disable_contact = False
    return root


class RecordingLayout:
    # A sidebar draw does not reliably fire in the rig's hidden window, so
    # the draw function is driven with a layout that records what it is
    # given. `enabled` is read back after the draw, since the panel sets it
    # on a column before or after filling it.
    def __init__(self, sink):
        self.sink = sink
        self.enabled = True
        self.alert = False

    def box(self):
        return RecordingLayout(self.sink)

    def column(self, **kwargs):
        return RecordingLayout(self.sink)

    def row(self, **kwargs):
        return RecordingLayout(self.sink)

    def label(self, **kwargs):
        self.sink.append(("label", kwargs.get("text", ""), self))

    def prop(self, *args, **kwargs):
        self.sink.append(("prop", args[1] if len(args) > 1 else "", self))

    def operator(self, idname, **kwargs):
        class Settings:
            pass
        settings = Settings()
        self.sink.append(("operator", idname, self, settings))
        return settings


def draw_panel():
    sink = []
    ops.draw_intersection_correction(RecordingLayout(sink), bpy.context)
    labels = [entry[1] for entry in sink if entry[0] == "label"]
    props = [entry[1] for entry in sink if entry[0] == "prop"]
    buttons = [entry for entry in sink if entry[0] == "operator"]
    return labels, props, buttons


try:
    dh = DriverHelpers(pkg, result)
    ops = __import__(pkg + ".mesh_ops.intersection_ops",
                     fromlist=["get_last_report"])
    mc = __import__(pkg + ".mcp.handlers.mesh_cleaning",
                    fromlist=["resolve_intersections"])
    dh.log("setup_start")

    # ----- A: an overlapping chain, through Transfer -------------------
    fresh()
    links = [add_link(f"Link{k}", k, flip=(k == 3)) for k in range(6)]
    transfer_state()
    group = dh.api.solver.create_group("Links", "PDRD")
    group.add(*[o.name for o in links])
    expected = [(f"Link{k}", f"Link{k + 1}") for k in range(5)]
    crossing_before = crossing_pairs(links)
    before = snapshot(links)
    verdict_before, error_before = build("overlapping chain", first=True)

    def overlay():
        # What the viewport draws, after two seconds of status polls. The
        # second number is what the server's latest response still carries.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            time.sleep(0.1)
        return (len(dh.com.info.violations),
                len(dh.com.info.response.get("violations") or []))

    overlay_refused = overlay()
    bpy.ops.object.select_all(action="DESELECT")
    status_idle, _report = resolve(selected_only=True)
    overlay_idle = overlay()
    status, report = resolve()
    overlay_resolved = overlay()
    crossing_after = crossing_pairs(links)
    moved = {o.name: displacement(o, before) for o in links}

    def build_watching_the_overlay(message):
        # `dh.build_verdict`, reading the overlay on every tick on the way:
        # the largest number of violation records the viewport was given
        # between the dispatch and this build's own verdict, and the upload
        # ids they were seen under.
        data_hash = dh.encoder_mesh.compute_data_hash(bpy.context)
        param_hash = dh.encoder_param.compute_param_hash(bpy.context)
        data_bytes, param_bytes = dh.encode_payload()
        held = dh.facade.engine.state.server_upload_id
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes,
            data_hash=data_hash, param_hash=param_hash, message=message))
        most, seen_under = 0, set()
        deadline = time.time() + BUILD_TIMEOUT
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            s = dh.facade.engine.state
            if s.violations:
                most = max(most, len(s.violations))
                seen_under.add(s.server_upload_id == held)
            if (s.activity.name == "IDLE"
                    and s.solver.name in ("READY", "RESUMABLE", "FAILED")
                    and s.server_upload_id != held):
                return s.solver.name, (s.server_error or ""), most, sorted(seen_under)
            time.sleep(0.02)
        raise RuntimeError(f"build {message!r} gave no verdict")

    verdict_after, error_after, flashed, flashed_under_old_id = (
        build_watching_the_overlay("resolved chain"))
    overlay_built = overlay()
    dh.record(
        "A2_refused_overlay_comes_down_and_stays_down",
        overlay_refused[0] > 0
        and status_idle == {"CANCELLED"} and overlay_idle[0] > 0
        and status == {"FINISHED"} and overlay_resolved[0] == 0
        and flashed == 0
        and overlay_built == (0, 0),
        {"refused": overlay_refused, "nothing_moved": overlay_idle,
         "resolved": overlay_resolved, "built": overlay_built,
         "flashed_during_the_next_transfer": flashed,
         "flashed_under_the_old_upload_id": flashed_under_old_id,
         "status_idle": sorted(status_idle)},
    )
    dh.record(
        "A_overlapping_chain_refused_then_builds",
        crossing_before == expected
        and verdict_before == "FAILED"
        and status == {"FINISHED"}
        and report is not None
        and [tuple(p) for p in report["initial_pairs"]] == expected
        and report["remaining_pairs"] == []
        and crossing_after == []
        and report["objects"] == 6
        and not report["held"] and not report["skipped"]
        # The chain is drawn too far out and has to close up by 4 mm at each
        # of five joints, about its middle: every link moves, and an end link
        # by two and a half joints.
        and all(0.0 < d < 3 * OVERLAP for d in moved.values())
        and verdict_after == "READY",
        {"crossing_before": crossing_before, "crossing_after": crossing_after,
         "verdict_before": verdict_before, "error_before": error_before[:400],
         "verdict_after": verdict_after, "error_after": error_after[:400],
         "status": sorted(status), "moved": moved,
         "iterations": (report or {}).get("iterations")},
    )

    # ----- B: closer than the contact offsets, through Transfer --------
    # The cubes face each other squarely, so the smallest motion that opens
    # them is a translation along x and the room between them can be read
    # off their x extents.
    SIDE, ROOM, GAP, OFFSET = 0.1, 0.004, 0.001, 0.005
    fresh()
    left = add_cube("Left", SIDE, (0.0, 0.0, 0.0))
    right = add_cube("Right", SIDE, (SIDE + ROOM, 0.0, 0.0))
    transfer_state()
    cubes = dh.api.solver.create_group("Cubes", "PDRD")
    cubes.add("Left", "Right")
    cubes.param.use_group_bounding_box_diagonal = False
    cubes.param.contact_gap = GAP
    cubes.param.contact_offset = OFFSET

    def room():
        return float(world_verts(right)[:, 0].min() - world_verts(left)[:, 0].max())

    room_before = room()
    crossing_before = crossing_pairs([left, right])
    verdict_before, error_before = build("inside the offsets")
    status, report = resolve()
    room_after = room()
    verdict_after, error_after = build("outside the offsets")
    dh.record(
        "B_contact_offsets_refused_then_build",
        crossing_before == []
        and verdict_before == "FAILED"
        and status == {"FINISHED"}
        and [tuple(p) for p in report["initial_pairs"]] == [("Left", "Right")]
        and report["remaining_pairs"] == []
        # At least the offsets and half a clearance, which is where a pair
        # stops counting as overlapping, and no more than the offsets and a
        # whole clearance, which is what it is opened to.
        and 2 * OFFSET + 0.5 * GAP <= room_after <= 2 * OFFSET + 1.01 * GAP
        and verdict_after == "READY",
        {"room_before": room_before, "room_after": room_after,
         "offsets": 2 * OFFSET, "clearance": GAP,
         "verdict_before": verdict_before, "error_before": error_before[:400],
         "verdict_after": verdict_after, "error_after": error_after[:400]},
    )

    # ----- C, D, E, F: who takes part -----------------------------------
    # A pitch of 0.078 leaves every joint 2 mm of room. Shifting the four
    # inner links by 4 mm, alternately back and forth, takes the second and
    # fourth joints to a pitch of 0.086, 6 mm past where the tubes meet,
    # while the two held ends stay a distance apart that the chain can span.
    FIT = 0.078
    SHIFT = (0.0, -0.004, 0.004, -0.004, 0.004)
    fresh()
    links = [add_link(f"Link{k}", k, pitch=FIT, shift=SHIFT[k])
             for k in range(5)]
    keyed = add_link("Keyed", 5, pitch=FIT)
    keyed.keyframe_insert(data_path="location", frame=1)
    keyed.keyframe_insert(data_path="location", frame=20)
    # A cube missing a face, around the far side of the first link's tube.
    bowl = add_cube("Bowl", 0.03, (-RING, 0.0, 0.0))
    bm = bmesh.new()
    bm.from_mesh(bowl.data)
    bm.faces.ensure_lookup_table()
    bm.faces.remove(bm.faces[0])
    bm.to_mesh(bowl.data)
    bm.free()
    bowl.data.update()
    # A sheet cutting the third link, and a mesh in no group around the top
    # of the fourth.
    bpy.ops.mesh.primitive_plane_add(
        size=0.3, location=(2 * FIT + SHIFT[2], 0.0, 0.0),
        rotation=(0.0, math.pi / 2, 0.0))
    sheet = bpy.context.active_object
    sheet.name = "Sheet"
    stray = add_cube("Stray", 0.03, (3 * FIT + SHIFT[3], 0.0, RING))
    group = dh.api.solver.create_group("Links", "PDRD")
    group.add(*[o.name for o in links], "Keyed", "Bowl")
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add("Sheet")
    hold = links[0].vertex_groups.new(name="hold")
    hold.add(list(range(len(links[0].data.vertices))), 1.0, "REPLACE")
    group.create_pin("Link0", "hold")

    everything = links + [keyed, bowl, sheet, stray]
    chain = links + [keyed]
    crossing_before = crossing_pairs(chain)
    bystanders_before = crossing_pairs(everything)
    before = snapshot(everything)
    status, report = resolve()
    moved = {o.name: displacement(o, before) for o in everything}
    crossing_after = crossing_pairs(chain)
    dh.record(
        "C_pinned_and_animated_are_held",
        status == {"FINISHED"}
        and crossing_before == [("Link1", "Link2"), ("Link3", "Link4")]
        and crossing_after == []
        and [tuple(p) for p in report["initial_pairs"]] == crossing_before
        and report["remaining_pairs"] == []
        and report["held"] == {"Link0": "pinned", "Keyed": "driven"}
        and moved["Link0"] == 0.0 and moved["Keyed"] == 0.0
        # Both ends are held, so the four links between them take the two
        # overlaps among themselves.
        and all(moved[f"Link{k}"] > 0.0 for k in range(1, 5)),
        {"held": report["held"], "crossing_before": crossing_before,
         "crossing_after": crossing_after, "moved": moved,
         "iterations": report["iterations"]},
    )
    dh.record(
        "D_open_mesh_is_left_out_by_name",
        report["skipped"] == ["Bowl"] and moved["Bowl"] == 0.0
        and "Bowl" not in report["moved"],
        {"skipped": report["skipped"], "moved": moved["Bowl"]},
    )
    dh.record(
        "E_shell_and_ungrouped_are_ignored",
        # Both really do cross a link, before and after: the tool neither
        # moved them nor moved a link out of their way.
        ("Link2", "Sheet") in bystanders_before
        and ("Link3", "Stray") in bystanders_before
        and moved["Sheet"] == 0.0 and moved["Stray"] == 0.0
        and report["objects"] == 6
        and not any("Sheet" in p or "Stray" in p
                    for p in report["initial_pairs"]),
        {"bystanders_before": bystanders_before, "objects": report["objects"],
         "initial_pairs": report["initial_pairs"]},
    )
    mismatch = {
        name: (moved[name], report["moved"].get(name, 0.0))
        for name in moved
        if abs(moved[name] - report["moved"].get(name, 0.0))
        > 1e-4 * max(moved[name], 1e-3)
    }
    dh.record(
        "F_reported_distance_is_the_distance_moved",
        not mismatch and set(report["moved"]) == {f"Link{k}" for k in range(1, 5)},
        {"mismatch": mismatch, "reported": report["moved"]},
    )
    labels_held, _props, _buttons = draw_panel()

    # ----- G: Selected Only ---------------------------------------------
    fresh()
    first = add_cube("First", 0.1, (0.0, 0.0, 0.0))
    second = add_cube("Second", 0.1, (0.09, 0.02, 0.01))
    solid = dh.api.solver.create_group("Blocks", "SOLID")
    solid.add("First", "Second")
    before = snapshot([first, second])
    bpy.ops.object.select_all(action="DESELECT")
    status_none, report_none = resolve(selected_only=True)
    moved_none = {o.name: displacement(o, before) for o in (first, second)}
    second.select_set(True)
    status, report = resolve(selected_only=True)
    moved = {o.name: displacement(o, before) for o in (first, second)}
    dh.record(
        "G_selected_only_moves_the_selection",
        status_none == {"CANCELLED"} and report_none is None
        and moved_none == {"First": 0.0, "Second": 0.0}
        and status == {"FINISHED"}
        and report["held"] == {"First": "unselected"}
        and moved["First"] == 0.0 and moved["Second"] > 0.0
        and crossing_pairs([first, second]) == []
        and report["remaining_pairs"] == [],
        {"status_none": sorted(status_none), "status": sorted(status),
         "held": (report or {}).get("held"), "moved": moved},
    )

    # ----- H: a parent and its child ------------------------------------
    # The parent is pushed along x by one block and its child along y by
    # another, so the two are solved to different places.
    fresh()
    parent = add_cube("Parent", 0.1, (0.0, 0.0, 0.0))
    child = add_cube("Child", 0.1, (0.0, 0.5, 0.0))
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    pusher_x = add_cube("PusherX", 0.1, (-0.09, 0.0, 0.0))
    pusher_y = add_cube("PusherY", 0.1, (0.0, 0.59, 0.0))
    family = [parent, child, pusher_x, pusher_y]
    solid = dh.api.solver.create_group("Blocks", "SOLID")
    solid.add(*[o.name for o in family])
    for name in ("PusherX", "PusherY"):
        obj = bpy.data.objects[name]
        pin = obj.vertex_groups.new(name="hold")
        pin.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        solid.create_pin(name, "hold")
    before = snapshot(family)
    status, report = resolve()
    moved = {o.name: displacement(o, before) for o in family}
    mismatch = {
        name: (moved[name], report["moved"].get(name, 0.0))
        for name in moved
        if abs(moved[name] - report["moved"].get(name, 0.0))
        > 1e-4 * max(moved[name], 1e-3)
    }
    dh.record(
        "H_parent_and_child_both_land",
        status == {"FINISHED"}
        and set(report["moved"]) == {"Parent", "Child"}
        and not mismatch
        and crossing_pairs(family) == []
        and report["remaining_pairs"] == [],
        {"moved": moved, "reported": report["moved"], "mismatch": mismatch,
         "held": report["held"]},
    )

    # ----- I: the panel entry -------------------------------------------
    labels_clean, props, buttons = draw_panel()
    state = dh.groups.get_addon_data(bpy.context.scene).state
    state.intersection_fix_iterations = 7
    state.intersection_fix_clearance = 2.5
    state.intersection_fix_selected_only = True
    _labels, _props, buttons_set = draw_panel()
    settings = buttons_set[0][3]
    state.intersection_fix_iterations = 50
    state.intersection_fix_clearance = 1.0
    state.intersection_fix_selected_only = False
    fresh()
    lone = add_cube("Lone", 0.1, (0.0, 0.0, 0.0))
    solo = dh.api.solver.create_group("Blocks", "SOLID")
    solo.add("Lone")
    labels_few, _props, buttons_few = draw_panel()
    dh.record(
        "I_panel_draws_every_state",
        props == ["intersection_fix_iterations", "intersection_fix_clearance",
                  "intersection_fix_selected_only"]
        and [b[1] for b in buttons] == ["object.resolve_intersections"]
        and buttons[0][2].enabled is True
        and (settings.max_iterations, settings.clearance,
             settings.selected_only) == (7, 2.5, True)
        and any("4 SOLID or PDRD object(s)" in t for t in labels_clean)
        and any("Last run: 2 overlapping pair(s) opened" in t
                for t in labels_clean)
        and any("2 object(s) moved, by at most" in t for t in labels_clean)
        and any("2 object(s) held in place" in t for t in labels_clean)
        and any("1 object(s) left out: surface not closed" in t
                for t in labels_held)
        and any("Scan them with Mesh Cleaning above." in t for t in labels_held)
        and buttons_few[0][2].enabled is False
        and any("Needs two or more objects in SOLID or PDRD groups" in t
                for t in labels_few)
        and not any(t.startswith("Last run") for t in labels_few),
        {"props": props, "labels_clean": labels_clean,
         "labels_held": labels_held, "labels_few": labels_few,
         "settings": [getattr(settings, k, None) for k in
                      ("max_iterations", "clearance", "selected_only")]},
    )

    # ----- J: the MCP tool and the scripting API ------------------------
    dec = __import__(pkg + ".mcp.decorators", fromlist=["get_handler_registry"])
    handlers_pkg = __import__(pkg + ".mcp.handlers",
                              fromlist=["resolve_intersections"])
    registry = dec.get_handler_registry()
    schema = registry.get("resolve_intersections", {}).get(
        "schema", {}).get("inputSchema", {})

    fresh()
    first = add_cube("First", 0.1, (0.0, 0.0, 0.0))
    second = add_cube("Second", 0.1, (0.09, 0.02, 0.01))
    third = add_cube("Third", 0.1, (0.18, 0.04, 0.02))
    blocks = [first, second, third]
    solid = dh.api.solver.create_group("Blocks", "SOLID")
    solid.add("First", "Second", "Third")
    bpy.ops.object.select_all(action="DESELECT")
    first.select_set(True)
    bpy.context.view_layer.update()
    home = {o.name: o.matrix_world.copy() for o in blocks}
    before = snapshot(blocks)
    refused_passes = mc.resolve_intersections({"max_iterations": 0})
    refused_clearance = mc.resolve_intersections({"clearance": 0.0})
    refused_name = mc.resolve_intersections({"object_names": ["Nobody"]})
    untouched = {o.name: displacement(o, before) for o in blocks}
    named = mc.resolve_intersections({"object_names": ["Third"]})
    moved_named = {o.name: displacement(o, before) for o in blocks}
    selection_kept = [o.name for o in blocks if o.select_get()]
    crossing_named = crossing_pairs(blocks)

    for obj in blocks:
        obj.matrix_world = home[obj.name]
    bpy.context.view_layer.update()
    before = snapshot(blocks)
    via_api = dh.api.solver.resolve_intersections()
    moved_api = {o.name: displacement(o, before) for o in blocks}
    api_error = ""
    try:
        dh.api.solver.resolve_intersections("Nobody")
    except ValueError as exc:
        api_error = str(exc)
    dh.record(
        "J_mcp_tool_and_api",
        "resolve_intersections" in registry
        and hasattr(handlers_pkg, "resolve_intersections")
        and set(schema.get("properties", {})) == {
            "object_names", "max_iterations", "clearance"}
        and not schema.get("required")
        and refused_passes.get("status") == "error"
        and refused_clearance.get("status") == "error"
        and refused_name.get("status") == "error"
        and "Nobody" in str(refused_name.get("message", ""))
        and untouched == {"First": 0.0, "Second": 0.0, "Third": 0.0}
        and named.get("status") == "success"
        and named.get("held") == {"First": "unselected", "Second": "unselected"}
        and [m["object_name"] for m in named.get("moved", [])] == ["Third"]
        and moved_named["First"] == 0.0 and moved_named["Second"] == 0.0
        and moved_named["Third"] > 0.0
        and selection_kept == ["First"]
        and named.get("initial_pair_count") == 2
        # The two objects the call held still overlap, and the result names
        # them.
        and crossing_named == [("First", "Second")]
        and named.get("remaining_pairs") == [["First", "Second"]]
        and "still overlap" in named.get("message", "")
        and via_api.get("remaining_pairs") == []
        and via_api.get("initial_pair_count") == 2
        and all(d > 0.0 for d in moved_api.values())
        and crossing_pairs(blocks) == []
        and "Nobody" in api_error,
        {"schema": schema, "refused": [refused_passes.get("message"),
                                       refused_clearance.get("message"),
                                       refused_name.get("message")],
         "named": named, "moved_named": moved_named,
         "selection_kept": selection_kept, "crossing_named": crossing_named,
         "via_api": via_api, "moved_api": moved_api, "api_error": api_error},
    )

    # ----- K: an object no motion can free -----------------------------
    # The wedge is narrower than the walls, so its side faces pass through
    # each wall's face and no two faces are coplanar.
    fresh()
    wall_a = add_cube("WallA", 0.1, (0.0, 0.0, 0.0))
    wall_b = add_cube("WallB", 0.1, (0.15, 0.0, 0.0))
    wedge = add_cube("Wedge", 0.08, (0.075, 0.0, 0.0))
    trio = [wall_a, wall_b, wedge]
    solid = dh.api.solver.create_group("Blocks", "SOLID")
    solid.add("WallA", "WallB", "Wedge")
    for wall in (wall_a, wall_b):
        pin = wall.vertex_groups.new(name="hold")
        pin.add(list(range(len(wall.data.vertices))), 1.0, "REPLACE")
        solid.create_pin(wall.name, "hold")
    squeezed = [("WallA", "Wedge"), ("WallB", "Wedge")]
    before = snapshot(trio)
    status, report = resolve(max_iterations=20)
    moved = {o.name: displacement(o, before) for o in trio}
    labels_left, _props, _buttons = draw_panel()
    dh.record(
        "K_squeezed_object_is_reported_left",
        status == {"FINISHED"}
        and [tuple(p) for p in report["initial_pairs"]] == squeezed
        and [tuple(p) for p in report["remaining_pairs"]] == squeezed
        and crossing_pairs(trio) == squeezed
        and report["iterations"] == 1
        and moved == {"WallA": 0.0, "WallB": 0.0, "Wedge": 0.0}
        and report["moved"] == {}
        and any("Last run: 2 of 2 pair(s) still overlap" in t
                for t in labels_left),
        {"status": sorted(status), "moved": moved,
         "iterations": report["iterations"],
         "remaining": report["remaining_pairs"], "labels": labels_left},
    )

    # ----- L: a clearance below what the tree resolves ----------------
    # The chain of check A at a twentieth of the size, thirty units from the
    # origin, with the gap ratio a hundredth of its default.
    FAR = (30.0, 20.0, 10.0)
    SMALL = 0.05
    fresh()
    links = [add_link(f"Link{k}", k, scale=SMALL, origin=FAR) for k in range(6)]
    group = dh.api.solver.create_group("Links", "PDRD")
    group.add(*[o.name for o in links])
    group.param.use_group_bounding_box_diagonal = True
    group.param.contact_gap_rat = 1e-5
    entry = next(iter(ops.volumetric_entries(bpy.context.scene)))
    lengths = __import__(pkg + ".core.encoder.params",
                         fromlist=["object_contact_lengths"])
    gap, _offset = lengths.object_contact_lengths(entry[0], entry[1], 1.0)
    resolution = float(np.finfo(np.float32).eps) * max(FAR)

    def rooms():
        # The room at each joint, read off the links' positions. Each link's
        # tube meets the next one's at a vertex of both meshes on the chain's
        # axis, where the room is the pitch that just touches less the pitch
        # the two links are at. The chain is symmetric about that axis, so
        # the links only slide along it.
        bpy.context.view_layer.update()
        x = [o.matrix_world.translation.x for o in links]
        return [2.0 * (RING - TUBE) * SMALL - (x[k + 1] - x[k])
                for k in range(5)]

    rooms_before = rooms()
    crossing_before = crossing_pairs(links)
    status, report = resolve()
    rooms_after = rooms()
    dh.record(
        "L_small_chain_far_from_origin",
        gap < 0.1 * resolution
        and len(crossing_before) == 5
        and all(room < 0.0 for room in rooms_before)
        and status == {"FINISHED"}
        and len(report["initial_pairs"]) == 5
        and report["remaining_pairs"] == []
        and crossing_pairs(links) == []
        # Raised, each clearance is 16 times the resolution, a joint counts
        # as overlapping below half of that and is opened to the whole of
        # it. Left at 85 nanometers, a joint ends within a few times the
        # resolution of touching.
        and all(4 * resolution <= room <= 20 * resolution
                for room in rooms_after),
        {"gap": gap, "resolution": resolution,
         "rooms_before": rooms_before, "rooms_after": rooms_after,
         "rooms_after_in_resolutions": [r / resolution for r in rooms_after],
         "iterations": report["iterations"],
         "remaining": report["remaining_pairs"]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
