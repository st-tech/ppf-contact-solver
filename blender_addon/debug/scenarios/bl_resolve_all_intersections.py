# File: scenarios/bl_resolve_all_intersections.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resolve All Intersections, the second entry of Intersection Correction in
# the Mesh Repair panel, driven on real Blender objects.
#
# Resolve Intersections moves SOLID and PDRD objects as rigid bodies. This
# entry does that first and then deforms SHELL meshes until no triangle of one
# is inside the contact offsets of a triangle of another or of itself, holding
# the SOLID, PDRD and STATIC objects where the rigid step left them, and it
# writes the cloth's vertex coordinates (`mesh_ops/shell_intersection_ops.py`,
# `core/shell_untangle.py`).
#
# Whether two sheets cross is measured here with Blender's own
# triangle-triangle test on the stored meshes, and whether Transfer accepts the
# scene by sending it to the solver build, never by reading the tool's report.
#
# Three scenes go through Transfer, each refused before and READY after:
#   `A_two_sheets_through_each_other`: wavy sheets laid one through the other,
#       their stored coordinates changed, vertices only.
#   `B_a_sheet_folded_through_itself`: one mesh with a flap passing through
#       its own sheet, which no rigid motion and no second object can show.
#   `C_rigid_step_then_cloth_over_a_cube`: two overlapping cubes are
#       pushed apart, and a sheet that a cube pokes through is deformed over
#       the cube where it ended. The cubes are PDRD: see the check.
#
# The rest need the solver only for the first:
#   `D_pins_and_unwritable_meshes_are_held`: a pinned vertex keeps its very
#       value; a mesh with shape keys, a mesh another object shares, and a mesh
#       a modifier deforms at the starting frame are held whole and named with
#       the reason, and the free sheet crossing all of them goes around them.
#   `E_objects_with_a_transform_land_in_world_space`: a parented, rotated and
#       scaled sheet and a scaled one, crossing in the world, end clear of each
#       other, and a vertex the run did not move keeps its stored value.
#   `F_selected_only_moves_the_selection`.
#   `G_undo_takes_back_the_rigid_step_and_the_cloth`.
#   `H_a_child_process_gives_what_the_operator_writes`: the scene read for the
#       operator, run in the child process the button starts, then canceled.
#       The child is a process because a second Python thread beside Blender's
#       interface corrupts the interpreter (`core/shell_untangle.py`).
#   `I_panel_draws_every_state`, and `I2_...`: with a run going both buttons are
#       disabled (the polls refuse too) and the panel draws its step, a bar of
#       the pairs cleared, the iteration, the seconds elapsed and an Abort
#       button, in both panels and only while a run goes.
#   `J_mcp_tool_and_api`.
#   `K_nothing_to_do_and_clean_scenes_write_nothing`.
#   `N_the_interface_stays_usable_and_a_run_can_be_aborted`: the modal lets
#       every event but its own timer through (Esc included), the Abort
#       operator asks the run to stop at its next tick, and a mesh edited while
#       a run goes is left as the artist has it.
#   `L_a_mesh_triangulated_differently_after_a_move_is_noticed`: Blender splits
#       a quad along the diagonal its shape asks for, so a run that bent quads
#       can leave a mesh with other triangles than the ones it kept clear, and
#       Transfer ships the new ones. The operator compares them after writing
#       and solves again (`passes` in the report); scene D needs a second solve
#       on most runs, and is judged by Blender's triangle test on the final mesh.
#   `M_a_retriangulated_write_asks_for_another_solve`: the step between two
#       solves writes, asks for another when a written mesh is triangulated
#       differently, adds the solves up, and does not repeat one that left pairs.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import math
import time
import traceback

import numpy as np
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT = "resolve_all_intersections"
BUILD_TIMEOUT = 120.0

# Absolute contact lengths of every group, in metres.
GAP = 0.002
OFFSET = 0.002
CLEAR = 0.5 * GAP


def fresh():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    dh.api.solver.clear()
    ops.clear_last_report()
    rigid_ops.clear_last_report()


def absolute(group):
    group.param.use_group_bounding_box_diagonal = False
    group.param.contact_gap = GAP
    group.param.contact_offset = OFFSET
    return group


def sheet(name, n, surface, size=1.0, matrix=None, parent=None):
    # An n x n grid of quads whose point (u, v) in [0, size]^2 is surface(u, v),
    # a 3-vector. The stored coordinates are in the object's own space.
    us = np.linspace(0.0, size, n + 1)
    verts = [tuple(float(c) for c in surface(u, v)) for u in us for v in us]
    faces = [
        (i * (n + 1) + j, (i + 1) * (n + 1) + j,
         (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1)
        for i in range(n) for j in range(n)
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    if matrix is not None:
        obj.matrix_basis = matrix
    bpy.context.view_layer.update()
    return obj


def cube(name, size, location):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def local_coords(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def world_verts(obj):
    bpy.context.view_layer.update()
    m = np.array(obj.matrix_world)
    return local_coords(obj).astype(np.float64) @ m[:3, :3].T + m[:3, 3]


def tree(obj):
    # The surface Transfer ships: the evaluated mesh, which is the stored one
    # unless a modifier deforms it.
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        m = np.array(evaluated.matrix_world)
        points = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    finally:
        evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(
        [tuple(p) for p in points], triangles, all_triangles=True)


def crossing_pairs(objs):
    # Blender's triangle-triangle test between different objects.
    trees = [tree(o) for o in objs]
    return sorted(
        (objs[i].name, objs[j].name)
        for i in range(len(objs)) for j in range(i + 1, len(objs))
        if trees[i].overlap(trees[j]))


def snapshot(objs):
    return {o.name: local_coords(o).copy() for o in objs}


def resolve(**arguments):
    # What the panel's button does: the three settings, passed to the
    # operator, run in line.
    state = dh.groups.get_addon_data(bpy.context.scene).state
    call = {"max_iterations": state.intersection_fix_iterations,
            "clearance": state.intersection_fix_clearance,
            "selected_only": state.intersection_fix_selected_only}
    call.update(arguments)
    status = bpy.ops.object.resolve_all_intersections("EXEC_DEFAULT", **call)
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

    def progress(self, **kwargs):
        self.sink.append(("progress", kwargs.get("text", ""), self, kwargs.get("factor")))

    def operator(self, idname, **kwargs):
        class Settings:
            pass
        settings = Settings()
        self.sink.append(("operator", idname, self, settings))
        return settings


def draw_panel():
    sink = []
    ops.draw_shell_intersection_correction(RecordingLayout(sink), bpy.context)
    labels = [entry[1] for entry in sink if entry[0] == "label"]
    props = [entry[1] for entry in sink if entry[0] == "prop"]
    buttons = [entry for entry in sink if entry[0] == "operator"]
    return labels, props, buttons


def wave(phase, lift, fx=5.0, fy=4.0):
    return lambda u, v: (u, v, lift + 0.02 * math.sin(fx * u + phase) * math.cos(fy * v))


try:
    dh = DriverHelpers(pkg, result)
    ops = __import__(pkg + ".mesh_ops.shell_intersection_ops",
                     fromlist=["get_last_report"])
    rigid_ops = __import__(pkg + ".mesh_ops.intersection_ops",
                           fromlist=["get_last_report"])
    mc = __import__(pkg + ".mcp.handlers.mesh_cleaning",
                    fromlist=["resolve_all_intersections"])
    su = __import__(pkg + ".core.shell_untangle", fromlist=["Run"])
    dh.log("setup_start")

    # ----- A: two sheets laid through each other, through Transfer ------
    fresh()
    under = sheet("Under", 12, wave(0.0, 0.0))
    over = sheet("Over", 12, wave(0.6, 0.004, 4.0, 5.0))
    transfer_state()
    cloth = absolute(dh.api.solver.create_group("Cloth", "SHELL"))
    cloth.add("Under", "Over")
    sheets = [under, over]
    crossing_before = crossing_pairs(sheets)
    before = snapshot(sheets)
    verdict_before, error_before = build("sheets through each other", first=True)
    status, report = resolve()
    crossing_after = crossing_pairs(sheets)
    after = snapshot(sheets)
    shell = (report or {}).get("shell") or {}
    verdict_after, error_after = build("sheets apart")
    changed = {n: int(np.any(after[n] != before[n], axis=1).sum()) for n in before}
    dh.record(
        "A_two_sheets_through_each_other",
        crossing_before == [("Under", "Over")]
        and verdict_before == "FAILED"
        and status == {"FINISHED"}
        and shell.get("converged") and shell.get("remaining") == 0
        and shell.get("initial", 0) > 0
        and crossing_after == []
        and verdict_after == "READY"
        # Vertices only: the transforms are as they were, and both sheets moved.
        and all(np.allclose(np.array(o.matrix_world), np.eye(4)) for o in sheets)
        and all(c > 0 for c in changed.values())
        and set(shell.get("moved", {})) == {"Under", "Over"}
        and all(0.0 < d < 0.2 for d in shell["moved"].values()),
        {"crossing_before": crossing_before, "crossing_after": crossing_after,
         "verdict_before": verdict_before, "error_before": error_before[:400],
         "verdict_after": verdict_after, "error_after": error_after[:400],
         "status": sorted(status), "changed_vertices": changed,
         "shell": {k: v for k, v in shell.items() if k != "message"},
         "message": shell.get("message")},
    )

    # ----- B: a flap of one sheet through the sheet ----------------------
    def folded(u, v):
        hinge = 0.65
        x, z = u, 0.02 * math.sin(6.0 * u) * math.cos(5.0 * v)
        if u > hinge:
            s = u - hinge
            x, z = hinge - 0.998 * s, 0.06 * s + 0.004
        return (x, v, z)

    fresh()
    flap = sheet("Flap", 12, folded)
    transfer_state()
    cloth = absolute(dh.api.solver.create_group("Cloth", "SHELL"))
    cloth.add("Flap")
    before = snapshot([flap])
    verdict_before, error_before = build("sheet folded through itself")
    status, report = resolve()
    shell = (report or {}).get("shell") or {}
    verdict_after, error_after = build("sheet unfolded")
    dh.record(
        "B_a_sheet_folded_through_itself",
        verdict_before == "FAILED"
        and status == {"FINISHED"}
        and shell.get("converged") and shell.get("remaining") == 0
        and shell.get("initial", 0) > 0
        and verdict_after == "READY"
        and np.any(local_coords(flap) != before["Flap"]),
        {"verdict_before": verdict_before, "error_before": error_before[:400],
         "verdict_after": verdict_after, "error_after": error_after[:400],
         "shell": {k: v for k, v in shell.items() if k != "message"},
         "message": shell.get("message")},
    )

    # ----- C: the rigid step, then the cloth over what it left ----------
    fresh()
    left = cube("Left", 0.2, (0.0, 0.0, -0.3))
    right = cube("Right", 0.2, (0.196, 0.0, -0.3))
    lid = sheet("Lid", 12, lambda u, v: (u - 0.5, v - 0.5, 0.0), size=1.0)
    # Left's top face is at z = -0.2; lift Left so that it pokes through the
    # sheet, and let the pair overlap by 4 mm.
    left.location.z = -0.05
    right.location.z = -0.05
    bpy.context.view_layer.update()
    transfer_state()
    # PDRD and not SOLID: a SOLID cube is meshed by a tetrahedralizer that is not
    # deterministic, and with an offset of 2 mm Transfer refuses even two cubes
    # far apart, by pairs of the cube's own triangles near its edges, in a run
    # in five. The rigid step and the cloth see the same surface either way.
    solids = absolute(dh.api.solver.create_group("Blocks", "PDRD"))
    solids.add("Left", "Right")
    cloth = absolute(dh.api.solver.create_group("Cloth", "SHELL"))
    cloth.add("Lid")
    objs = [left, right, lid]
    crossing_before = crossing_pairs(objs)
    verdict_before, error_before = build("cubes overlapping, sheet through a cube")
    status, report = resolve()
    rigid = (report or {}).get("rigid") or {}
    shell = (report or {}).get("shell") or {}
    crossing_after = crossing_pairs(objs)
    verdict_after, error_after = build("cubes apart, sheet over them")
    violations_after = [repr(v)[:400] for v in list(dh.com.info.violations)[:4]]
    dh.record(
        "C_rigid_step_then_cloth_over_a_cube",
        ("Left", "Right") in crossing_before
        and any("Lid" in pair for pair in crossing_before)
        and verdict_before == "FAILED"
        and status == {"FINISHED"}
        and [tuple(p) for p in rigid.get("initial_pairs", [])] == [("Left", "Right")]
        and rigid.get("remaining_pairs") == []
        and shell.get("converged") and shell.get("remaining") == 0
        and set(shell.get("moved", {})) == {"Lid"}
        and shell.get("colliders") == 2
        and crossing_after == []
        and verdict_after == "READY",
        {"crossing_before": crossing_before, "crossing_after": crossing_after,
         "verdict_before": verdict_before, "error_before": error_before[:400],
         "verdict_after": verdict_after, "error_after": error_after[:400],
         "rigid": rigid, "shell": {k: v for k, v in shell.items() if k != "message"},
         "message": shell.get("message"), "violations_after": violations_after},
    )

    # ----- D: who is held ------------------------------------------------
    # A long free sheet at z = 0 and four small sheets that stand mostly below
    # it and poke their upper edge through it, each held for a different
    # reason (or pinned along its low edge).
    def across(x0):
        return lambda u, v: (x0 + 0.5 * u, v, -0.15 + 0.17 * u / 0.9)

    fresh()
    probe = sheet("Probe", 24, lambda u, v: (4.0 * u, v, 0.0), size=1.0)
    pinned = sheet("Pinned", 8, across(0.4), size=0.9)
    keyed = sheet("Keyed", 8, across(1.4), size=0.9)
    twin_a = sheet("TwinA", 8, across(2.4), size=0.9)
    bent = sheet("Bent", 8, across(3.4), size=0.9)
    keyed.shape_key_add(name="Basis")
    keyed.shape_key_add(name="Lift")
    shift = bent.modifiers.new(name="Shift", type="DISPLACE")
    shift.strength = 0.02
    bpy.context.view_layer.update()
    low = min(v.co.x for v in pinned.data.vertices)
    edge_ids = [i for i, v in enumerate(pinned.data.vertices) if v.co.x == low]
    pin_group = pinned.vertex_groups.new(name="Edge")
    pin_group.add(edge_ids, 1.0, "REPLACE")
    cloth = absolute(dh.api.solver.create_group("Cloth", "SHELL"))
    cloth.add("Probe", "Pinned", "Keyed", "TwinA", "Bent")
    cloth.create_pin("Pinned", "Edge")
    # Assigning a linked duplicate is refused, so the twin is made after: an
    # object in no group that shares TwinA's mesh, which a write would change.
    twin_b = bpy.data.objects.new("TwinB", twin_a.data)
    bpy.context.scene.collection.objects.link(twin_b)
    twin_b.location = (0.0, 5.0, 0.0)
    everything = [probe, pinned, keyed, twin_a, bent]
    before = snapshot(everything + [twin_b])
    crossing_before = crossing_pairs(everything)
    status, report = resolve()
    shell = (report or {}).get("shell") or {}
    after = snapshot(everything + [twin_b])
    crossing_after = crossing_pairs(everything)
    pin_rows = np.array(edge_ids)
    moved_vertices = {
        n: int(np.any(after[n] != before[n], axis=1).sum()) for n in before}
    dh.record(
        "D_pins_and_unwritable_meshes_are_held",
        len(crossing_before) == 4
        and status == {"FINISHED"}
        and shell.get("held") == {
            "Keyed": "shape keys", "TwinA": "shared mesh", "Bent": "deformed"}
        and shell.get("pinned") == {"Pinned": len(edge_ids)}
        and np.array_equal(after["Pinned"][pin_rows], before["Pinned"][pin_rows])
        and moved_vertices["Keyed"] == 0 and moved_vertices["TwinA"] == 0
        and moved_vertices["TwinB"] == 0 and moved_vertices["Bent"] == 0
        and moved_vertices["Probe"] > 0 and moved_vertices["Pinned"] > 0
        and shell.get("converged")
        and (shell.get("remaining") or 0) == 0
        and crossing_after == [],
        {"crossing_before": crossing_before, "crossing_after": crossing_after,
         "held": shell.get("held"), "pinned": shell.get("pinned"),
         "moved_vertices": moved_vertices, "passes": shell.get("passes"),
         "message": shell.get("message")},
    )

    # ----- E: parented, rotated and scaled -------------------------------
    fresh()
    holder = bpy.data.objects.new("Holder", None)
    bpy.context.scene.collection.objects.link(holder)
    holder.location = (3.0, -2.0, 1.0)
    holder.rotation_euler = (0.2, -0.3, 0.9)
    holder.scale = (1.4, 1.4, 1.4)
    bpy.context.view_layer.update()
    first = sheet("First", 12, wave(0.0, 0.0), parent=holder)
    second = sheet("Second", 12, wave(0.7, 0.004, 4.0, 5.0),
                   matrix=Matrix.Diagonal((1.0, 0.8, 1.0, 1.0)), parent=holder)
    scene_sheets = [first, second]
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("First", "Second")
    crossing_before = crossing_pairs(scene_sheets)
    world_before = {o.name: world_verts(o) for o in scene_sheets}
    local_before = snapshot(scene_sheets)
    status, report = resolve()
    shell = (report or {}).get("shell") or {}
    crossing_after = crossing_pairs(scene_sheets)
    world_after = {o.name: world_verts(o) for o in scene_sheets}
    untouched = {}
    for o in scene_sheets:
        same = np.all(local_coords(o) == local_before[o.name], axis=1)
        untouched[o.name] = int(same.sum())
    reported = {n: float(np.linalg.norm(world_after[n] - world_before[n], axis=1).max())
                for n in world_before}
    dh.record(
        "E_objects_with_a_transform_land_in_world_space",
        crossing_before == [("First", "Second")]
        and status == {"FINISHED"}
        and shell.get("converged") and crossing_after == []
        and all(abs(reported[n] - shell["moved"][n]) < 1e-4 for n in reported)
        and all(shell["moved"][n] > 0.0 for n in reported),
        {"crossing_before": crossing_before, "crossing_after": crossing_after,
         "reported": shell.get("moved"), "measured": reported,
         "unmoved_vertices": untouched, "message": shell.get("message")},
    )

    # ----- F: Selected Only ---------------------------------------------
    fresh()
    one = sheet("One", 10, wave(0.0, 0.0))
    two = sheet("Two", 10, wave(0.6, 0.004, 4.0, 5.0))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("One", "Two")
    bpy.ops.object.select_all(action="DESELECT")
    one.select_set(True)
    before = snapshot([one, two])
    status, report = resolve(selected_only=True)
    shell = (report or {}).get("shell") or {}
    after = snapshot([one, two])
    dh.record(
        "F_selected_only_moves_the_selection",
        status == {"FINISHED"}
        and shell.get("held") == {"Two": "unselected"}
        and set(shell.get("moved", {})) == {"One"}
        and np.array_equal(after["Two"], before["Two"])
        and np.any(after["One"] != before["One"])
        and crossing_pairs([one, two]) == [],
        {"held": shell.get("held"), "moved": shell.get("moved"),
         "message": shell.get("message")},
    )

    # ----- G: undo -------------------------------------------------------
    # Everything the operator writes is Blender data (object transforms and
    # mesh coordinates), so an undo step restores it, and the operator asks for
    # one. A run through `bpy.ops` from this driver is not pushed as a step of
    # its own (the click path is, and was read off a live session), so the
    # steps around it are pushed here.
    fresh()
    top = cube("Top", 0.2, (0.0, 0.0, 0.0))
    side = cube("Side", 0.2, (0.196, 0.0, 0.0))
    mat = sheet("Mat", 10, lambda u, v: (u - 0.5, v - 0.5, 0.05), size=1.0)
    absolute(dh.api.solver.create_group("Blocks", "SOLID")).add("Top", "Side")
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("Mat")
    cube_before = {n: np.array(bpy.data.objects[n].matrix_world) for n in ("Top", "Side")}
    mat_before = local_coords(mat).copy()
    bpy.ops.ed.undo_push(message="before Resolve All Intersections")
    status, report = resolve()
    bpy.ops.ed.undo_push(message="after Resolve All Intersections")
    cube_moved = {n: bool(np.any(np.array(bpy.data.objects[n].matrix_world) != cube_before[n]))
                  for n in cube_before}
    mat_moved = bool(np.any(local_coords(bpy.data.objects["Mat"]) != mat_before))
    undo_error, undos, restored = "", 0, False
    try:
        while undos < 2 and not restored:
            bpy.ops.ed.undo()
            undos += 1
            if all(n in bpy.data.objects for n in ("Top", "Side", "Mat")):
                restored = (
                    all(np.array_equal(np.array(bpy.data.objects[n].matrix_world), cube_before[n])
                        for n in cube_before)
                    and np.array_equal(local_coords(bpy.data.objects["Mat"]), mat_before))
    except RuntimeError as exc:
        undo_error = str(exc)
    dh.record(
        "G_undo_takes_back_the_rigid_step_and_the_cloth",
        "UNDO" in ops.OBJECT_OT_ResolveAllIntersections.bl_options
        and status == {"FINISHED"} and all(cube_moved.values()) and mat_moved
        and not undo_error and restored,
        {"bl_options": sorted(ops.OBJECT_OT_ResolveAllIntersections.bl_options),
         "cube_moved": cube_moved, "mat_moved": mat_moved,
         "undo_error": undo_error, "undos": undos, "restored": restored},
    )

    # ----- H: the child process the button starts -----------------------
    fresh()
    sheet("One", 10, wave(0.0, 0.0))
    sheet("Two", 10, wave(0.6, 0.004, 4.0, 5.0))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("One", "Two")
    context = bpy.context
    scene_in = ops.collect_scene(context, selected_only=False, clearance=1.0)
    st = su.Settings(clearance=scene_in.clearance)
    expected, expected_report = su.untangle(scene_in.surfaces, st)
    run = su.Run(scene_in.surfaces, st)
    run.start()
    finished = run.wait(240.0)
    out, rep = run.result()
    same = all(np.array_equal(a, b) for a, b in zip(out, expected))
    written = ops.write_back(scene_in, out)
    stopper = su.Run(scene_in.surfaces, st)
    stopper.cancel()
    stopper.start()
    stopper.wait(240.0)
    canceled_out, canceled_rep = stopper.result()
    dh.record(
        "H_a_child_process_gives_what_the_operator_writes",
        finished and same and rep.converged and set(written) == {"One", "Two"}
        and stopper.canceled and not canceled_rep.converged
        and all(np.array_equal(a, s.vert)
                for a, s in zip(canceled_out, scene_in.surfaces))
        and crossing_pairs([bpy.data.objects["One"], bpy.data.objects["Two"]]) == [],
        {"finished": finished, "same": same, "written": written,
         "canceled": stopper.canceled,
         "cancel_message": canceled_rep.message},
    )

    # ----- I: the panel --------------------------------------------------
    state = dh.groups.get_addon_data(bpy.context.scene).state
    fresh()
    labels_empty, props, buttons_empty = draw_panel()
    one = sheet("One", 10, wave(0.0, 0.0))
    two = sheet("Two", 10, wave(0.6, 0.004, 4.0, 5.0))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("One", "Two")
    state.intersection_fix_iterations = 7
    state.intersection_fix_clearance = 2.5
    state.intersection_fix_selected_only = True
    labels_ready, _props, buttons_ready = draw_panel()
    settings = buttons_ready[0][3]
    state.intersection_fix_iterations = 50
    state.intersection_fix_clearance = 1.0
    state.intersection_fix_selected_only = False
    status, report = resolve()
    labels_done, _props, _buttons = draw_panel()
    ops._last_report = {"rigid": None, "shell": {
        "objects": 2, "colliders": 0, "initial": 40, "remaining": 6,
        "converged": False, "canceled": False, "moved": {}, "max_move": 0.0,
        "mean_move": 0.0, "held": {"Two": "shape keys"}, "pinned": {},
        "skipped": [], "held_pairs": 3, "seconds": 1.0, "message": "x"}}
    labels_failed, _props, _buttons = draw_panel()
    ops._last_report = {"rigid": None, "shell": {
        "objects": 2, "colliders": 0, "initial": 0, "remaining": 0,
        "converged": False, "canceled": True, "moved": {}, "max_move": 0.0,
        "mean_move": 0.0, "held": {}, "pinned": {}, "skipped": [],
        "held_pairs": 0, "seconds": 1.0, "message": "canceled"}}
    labels_canceled, _props, _buttons = draw_panel()
    # While a run is going both buttons are disabled and the panel shows where it is.
    def sinks():
        new_sink, old_sink = [], []
        ops.draw_shell_intersection_correction(RecordingLayout(new_sink), bpy.context)
        rigid_ops.draw_intersection_correction(RecordingLayout(old_sink), bpy.context)
        return new_sink, old_sink

    def button(sink, idname):
        return next(e for e in sink if e[0] == "operator" and e[1] == idname)[2].enabled

    fresh()
    sheet("One", 10, wave(0.0, 0.0))
    sheet("Two", 10, wave(0.6, 0.004, 4.0, 5.0))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("One", "Two")
    idle_new, idle_old = sinks()
    poll_idle = (ops.OBJECT_OT_ResolveAllIntersections.poll(bpy.context),
                 rigid_ops.OBJECT_OT_ResolveIntersections.poll(bpy.context))
    started = time.monotonic() - 7.0
    rigid_ops.set_busy({"stage": "untangle", "iteration": 12, "pairs": 30,
                        "initial": 120, "started": started, "canceling": False})
    first_new, first_old = sinks()
    poll_busy = (ops.OBJECT_OT_ResolveAllIntersections.poll(bpy.context),
                 rigid_ops.OBJECT_OT_ResolveIntersections.poll(bpy.context))
    rigid_ops.set_busy({"stage": "recover", "iteration": 140, "pairs": 0,
                        "initial": 120, "started": started, "canceling": False})
    second_new, _ = sinks()
    rigid_ops.set_busy({"stage": "untangle", "iteration": 3, "pairs": 40,
                        "initial": 120, "started": started, "canceling": True})
    canceling_new, _ = sinks()
    rigid_ops.set_busy(None)
    after_new, after_old = sinks()

    def texts(sink, kind="label"):
        return [e[1] for e in sink if e[0] == kind]

    bars = lambda sink: [(e[1], round(e[3], 3)) for e in sink if e[0] == "progress"]
    dh.record(
        "I2_buttons_are_disabled_and_progress_is_drawn_while_a_run_goes",
        button(idle_new, "object.resolve_all_intersections") is True
        and button(idle_old, "object.resolve_intersections") is False  # one sheet pair, no solids
        and poll_idle == (True, True) and poll_busy == (False, False)
        and button(first_new, "object.resolve_all_intersections") is False
        and any("Resolve All Intersections is running" in t for t in texts(first_old))
        and any("Step 1 of 2: untangling" in t for t in texts(first_new))
        and bars(first_new) == [("Pairs cleared: 75%", 0.75)]
        and any("Iteration 12, 30 pair(s) inside their margin" in t for t in texts(first_new))
        and any(t == "7 s elapsed" for t in texts(first_new))
        and not any("Esc" in t for t in texts(first_new) + texts(first_old))
        and button(first_new, "object.resolve_all_intersections_abort") is True
        and button(first_old, "object.resolve_all_intersections_abort") is True
        and not any(e[1] == "object.resolve_all_intersections_abort"
                    for e in idle_new + idle_old + after_new + after_old
                    if e[0] == "operator")
        and button(canceling_new, "object.resolve_all_intersections_abort") is False
        and any("Step 2 of 2: restoring the shape" in t for t in texts(second_new))
        and bars(second_new) == [("Pairs cleared: 100%", 1.0)]
        and any("Iteration 140" in t for t in texts(second_new))
        and any("Aborting..." in t for t in texts(canceling_new))
        and button(after_new, "object.resolve_all_intersections") is True
        and poll_busy != (True, True) and rigid_ops.get_busy() is None
        and not any(e[0] == "progress" for e in after_new),
        {"idle_polls": poll_idle, "busy_polls": poll_busy,
         "first": texts(first_new), "bars": bars(first_new), "second": texts(second_new),
         "canceling": texts(canceling_new), "after": texts(after_new)},
    )
    dh.record(
        "I_panel_draws_every_state",
        props == ["intersection_fix_iterations", "intersection_fix_clearance",
                  "intersection_fix_selected_only"]
        and [b[1] for b in buttons_ready] == ["object.resolve_all_intersections"]
        and buttons_empty[0][2].enabled is False
        and buttons_ready[0][2].enabled is True
        and any("Needs a SHELL object" in t for t in labels_empty)
        and any("2 SHELL and 0 SOLID or PDRD object(s)" in t for t in labels_ready)
        and (settings.max_iterations, settings.clearance,
             settings.selected_only) == (7, 2.5, True)
        and any("overlapping pair(s) opened in" in t for t in labels_done)
        and any("2 shell object(s) moved, by at most" in t for t in labels_done)
        and any("Last run: 6 of 40 pair(s) still inside their margin" in t
                for t in labels_failed)
        and any("1 shell object(s) held in place" in t for t in labels_failed)
        and any("3 pair(s) of held triangles overlap" in t for t in labels_failed)
        and any("Last run: aborted" in t for t in labels_canceled),
        {"props": props, "empty": labels_empty, "ready": labels_ready,
         "done": labels_done, "failed": labels_failed,
         "canceled": labels_canceled,
         "settings": [getattr(settings, k, None) for k in
                      ("max_iterations", "clearance", "selected_only")]},
    )

    # ----- J: the MCP tool and the scripting API ------------------------
    dec = __import__(pkg + ".mcp.decorators", fromlist=["get_handler_registry"])
    registry = dec.get_handler_registry()
    schema = registry.get("resolve_all_intersections", {}).get(
        "schema", {}).get("inputSchema", {})
    fresh()
    a = sheet("A", 10, wave(0.0, 0.0))
    b = sheet("B", 10, wave(0.6, 0.004, 4.0, 5.0))
    c = sheet("C", 10, wave(0.0, 0.0), matrix=Matrix.Translation((0.0, 3.0, 0.0)))
    d = sheet("D", 10, wave(0.6, 0.004, 4.0, 5.0),
              matrix=Matrix.Translation((0.0, 3.0, 0.0)))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("A", "B", "C", "D")
    refused_passes = mc.resolve_all_intersections({"max_iterations": 0})
    refused_clearance = mc.resolve_all_intersections({"clearance": 0.0})
    refused_name = mc.resolve_all_intersections({"object_names": ["Nobody"]})
    before = snapshot([a, b, c, d])
    named = mc.resolve_all_intersections({"object_names": ["C"]})
    after = snapshot([a, b, c, d])
    via_api = dh.api.solver.resolve_all_intersections()
    api_error = ""
    try:
        dh.api.solver.resolve_all_intersections("Nobody")
    except ValueError as exc:
        api_error = str(exc)
    dh.record(
        "J_mcp_tool_and_api",
        "resolve_all_intersections" in registry
        and set(schema.get("properties", {})) == {
            "object_names", "max_iterations", "clearance"}
        and not schema.get("required")
        and refused_passes.get("status") == "error"
        and refused_clearance.get("status") == "error"
        and refused_name.get("status") == "error"
        and "Nobody" in str(refused_name.get("message", ""))
        and named.get("status") == "success"
        and named["shell"]["held"] == {"A": "unselected", "B": "unselected",
                                       "D": "unselected"}
        and [m["object_name"] for m in named["shell"]["moved"]] == ["C"]
        and np.array_equal(after["A"], before["A"])
        and np.array_equal(after["B"], before["B"])
        and np.array_equal(after["D"], before["D"])
        and np.any(after["C"] != before["C"])
        and via_api["shell"]["resolved"] is True
        and via_api["shell"]["initial_pair_count"] > 0
        and crossing_pairs([a, b]) == [] and crossing_pairs([c, d]) == []
        and "Nobody" in api_error,
        {"schema": schema,
         "refused": [refused_passes.get("message"),
                     refused_clearance.get("message"),
                     refused_name.get("message")],
         "named": named, "via_api": via_api, "api_error": api_error},
    )

    # ----- K: nothing to do ---------------------------------------------
    fresh()
    status_none, report_none = resolve()
    left = sheet("Left", 10, wave(0.0, 0.0))
    far = sheet("Far", 10, wave(0.0, 0.5))
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("Left", "Far")
    clean_before = snapshot([left, far])
    status_clean, report_clean = resolve()
    clean_after = snapshot([left, far])
    shell_clean = (report_clean or {}).get("shell") or {}
    dh.record(
        "K_nothing_to_do_and_clean_scenes_write_nothing",
        status_none == {"CANCELLED"} and report_none is None
        and status_clean == {"FINISHED"}
        and shell_clean.get("initial") == 0 and shell_clean.get("converged")
        and shell_clean.get("moved") == {}
        and all(np.array_equal(clean_after[n], clean_before[n]) for n in clean_before),
        {"status_none": sorted(status_none), "status_clean": sorted(status_clean),
         "shell": {k: v for k, v in shell_clean.items() if k != "message"}},
    )

    # ----- L: a quad's diagonal follows its shape -------------------------
    fresh()
    quad = sheet("Quad", 2, lambda u, v: (u, v, 0.0), size=1.0)
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("Quad")
    scene_l = ops.collect_scene(bpy.context, selected_only=False, clearance=1.0)
    untouched = ops._retriangulated(scene_l)
    quad.data.vertices[4].co = (0.5005, 0.5005, 0.0)
    quad.data.update()
    nudged = ops._retriangulated(scene_l)
    # Blender splits a quad along its first-third diagonal unless the two
    # triangles that makes face opposite ways. Vertex 1 is the first quad's
    # fourth corner; across the diagonal from the first corner to vertex 4
    # the quad is folded, and the other diagonal is the split.
    quad.data.vertices[1].co = (0.6, 0.1, 0.0)
    quad.data.update()
    bent = ops._retriangulated(scene_l)
    dh.record(
        "L_a_mesh_triangulated_differently_after_a_move_is_noticed",
        not untouched and not nudged and bent,
        {"untouched": untouched, "nudged": nudged, "bent": bent},
    )

    # ----- M: a solve is followed by another on the new triangles -----------
    # `_write` is the step between two solves: it writes, and asks for another
    # when Blender triangulates a written mesh differently. It reads only the
    # operator's two settings, so a stand-in carries them.
    fresh()
    quad = sheet("Quad", 2, lambda u, v: (u, v, 0.0), size=1.0)
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("Quad")
    scene_m = ops.collect_scene(bpy.context, selected_only=False, clearance=1.0)
    tri_input = scene_m.surfaces[0].tri.copy()
    plan = ops.Plan(None, scene_m, su.Settings(clearance=scene_m.clearance))
    stand_in = type("StandIn", (), {"selected_only": False, "clearance": 1.0})()
    write = ops.OBJECT_OT_ResolveAllIntersections._write
    folded = scene_m.surfaces[0].vert.copy()
    folded[1] = (0.6, 0.1, 0.0)
    again = write(stand_in, bpy.context, plan,
                  ([folded], su.Report(converged=True, initial=7, seconds=2.0)), False)
    passes_after_first = plan.passes
    new_triangles = not np.array_equal(plan.scene.surfaces[0].tri, tri_input)
    held_still = plan.scene.surfaces[0].vert.copy()
    again_2 = write(stand_in, bpy.context, plan,
                    ([held_still], su.Report(converged=True, initial=0, seconds=1.0)), False)
    total, moved_total = plan.totals()
    stopped_unresolved = write(
        stand_in, bpy.context,
        ops.Plan(None, ops.collect_scene(bpy.context, selected_only=False, clearance=1.0),
                 su.Settings()),
        ([plan.scene.surfaces[0].vert.copy()],
         su.Report(converged=False, remaining=3, initial=5, seconds=1.0)), False)
    expected = float(np.linalg.norm(np.array([0.6, 0.1, 0.0]) - np.array([0.0, 0.5, 0.0])))
    dh.record(
        "M_a_retriangulated_write_asks_for_another_solve",
        again and passes_after_first == 2 and new_triangles
        and not again_2 and plan.passes == 2
        and total.initial == 7 and abs(total.seconds - 3.0) < 1e-9
        and set(moved_total) == {"Quad"} and abs(moved_total["Quad"] - expected) < 1e-5
        and abs(total.max_move - expected) < 1e-5
        and not stopped_unresolved,
        {"again": again, "again_2": again_2, "passes": plan.passes,
         "new_triangles": new_triangles, "initial": total.initial,
         "seconds": total.seconds, "moved": moved_total, "max_move": total.max_move,
         "expected": expected, "stopped_unresolved": stopped_unresolved},
    )

    # ----- N: the interface stays usable, Abort, and edits under a run ----
    fresh()
    sheet("One", 10, wave(0.0, 0.0))
    sheet("Two", 10, wave(0.6, 0.004, 4.0, 5.0))
    moved_one = sheet("Moving", 4, lambda u, v: (u + 3.0, v, 0.0), size=1.0)
    absolute(dh.api.solver.create_group("Cloth", "SHELL")).add("One", "Two", "Moving")

    class FakeRun:
        def __init__(self):
            self.canceled, self.finished, self.cancels = False, False, 0

        def cancel(self):
            self.canceled = True
            self.cancels += 1

    stand_in = type("StandIn", (), {})()
    stand_in._run = FakeRun()
    stand_in._publish = lambda context: None
    modal = ops.OBJECT_OT_ResolveAllIntersections._modal

    def event(kind, value="PRESS"):
        return type("Event", (), {"type": kind, "value": value})()

    ops.clear_abort()
    passes = {kind: modal(stand_in, bpy.context, event(kind, value))
              for kind, value in (("MOUSEMOVE", "NOTHING"), ("WHEELUPMOUSE", "PRESS"),
                                  ("LEFTMOUSE", "PRESS"), ("ESC", "PRESS"), ("X", "PRESS"))}
    idle_tick = modal(stand_in, bpy.context, event("TIMER", "NOTHING"))
    cancels_before_abort = stand_in._run.cancels
    # The Abort button: refused with no run, otherwise it sets the flag and
    # grays itself out at once; the modal acts on the flag at its next tick.
    poll_no_run = ops.OBJECT_OT_ResolveAllIntersectionsAbort.poll(bpy.context)
    rigid_ops.set_busy({"stage": "untangle", "iteration": 1, "pairs": 9, "initial": 9,
                        "started": time.monotonic(), "canceling": False})
    poll_run = ops.OBJECT_OT_ResolveAllIntersectionsAbort.poll(bpy.context)
    abort_status = bpy.ops.object.resolve_all_intersections_abort("EXEC_DEFAULT")
    flag_set, busy_canceling = ops.abort_requested(), rigid_ops.get_busy()["canceling"]
    poll_after = ops.OBJECT_OT_ResolveAllIntersectionsAbort.poll(bpy.context)
    tick = modal(stand_in, bpy.context, event("TIMER", "NOTHING"))
    tick_again = modal(stand_in, bpy.context, event("TIMER", "NOTHING"))
    rigid_ops.set_busy(None)
    ops.clear_abort()
    # An edit under a run: the mesh keeps what the artist did, the others take the result.
    scene_n = ops.collect_scene(bpy.context, selected_only=False, clearance=1.0)
    unchanged = ops._changed_target(scene_n)
    plan_n = ops.Plan(None, scene_n, su.Settings(clearance=scene_n.clearance))
    moved_one.data.vertices[3].co.z += 0.25
    moved_one.data.update()
    changed_vertex = ops._changed_target(scene_n)
    moved_one.data.vertices[3].co.z -= 0.25
    moved_one.data.update()
    restored = ops._changed_target(scene_n)
    moved_one.location.x += 1.0
    bpy.context.view_layer.update()
    changed_matrix = ops._changed_target(scene_n)
    before_write = {n: local_coords(bpy.data.objects[n]).copy() for n in ("One", "Two", "Moving")}
    offered = [s.vert + 0.5 for s in scene_n.surfaces]
    wrote = ops.OBJECT_OT_ResolveAllIntersections._write(
        type("StandIn", (), {"selected_only": False, "clearance": 1.0})(), bpy.context,
        plan_n, (offered, su.Report(converged=True, initial=3, seconds=1.0)), False)
    unwritten = all(np.array_equal(local_coords(bpy.data.objects[n]), before_write[n])
                    for n in before_write)
    # A pose read at the starting frame is not the pose the scene is in now, and
    # that difference alone is no edit.
    moved_one.location.x -= 1.0
    for frame, x in ((1, 3.0), (10, 5.0)):
        bpy.context.scene.frame_set(frame)
        moved_one.location.x = x
        moved_one.keyframe_insert(data_path="location", index=0)
    bpy.context.scene.frame_set(7)
    animated = ops._changed_target(
        ops.collect_scene(bpy.context, selected_only=False, clearance=1.0))
    bpy.context.scene.frame_set(1)
    dh.record(
        "N_the_interface_stays_usable_and_a_run_can_be_aborted",
        all(v == {"PASS_THROUGH"} for v in passes.values()) and cancels_before_abort == 0
        and idle_tick == {"RUNNING_MODAL"}
        and not poll_no_run and poll_run and abort_status == {"FINISHED"}
        and flag_set and busy_canceling and not poll_after
        and tick == {"RUNNING_MODAL"} and stand_in._run.cancels == 1
        and tick_again == {"RUNNING_MODAL"} and stand_in._run.cancels == 1
        and unchanged is None and changed_vertex == "Moving" and restored is None
        and changed_matrix == "Moving"
        and wrote is False and unwritten and plan_n.refused
        and "Moving" in plan_n.refused and plan_n.final is None
        and animated is None,
        {"passes": {k: sorted(v) for k, v in passes.items()}, "idle_tick": sorted(idle_tick), "poll_no_run": poll_no_run,
         "poll_run": poll_run, "abort_status": sorted(abort_status), "flag_set": flag_set,
         "busy_canceling": busy_canceling, "poll_after": poll_after,
         "cancels": stand_in._run.cancels, "unchanged": unchanged,
         "changed_vertex": changed_vertex, "restored": restored,
         "changed_matrix": changed_matrix, "wrote": wrote, "unwritten": unwritten,
         "refused": plan_n.refused, "animated": animated},
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
