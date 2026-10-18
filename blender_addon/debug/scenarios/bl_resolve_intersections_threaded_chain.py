# File: scenarios/bl_resolve_intersections_threaded_chain.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Intersection Correction on a chain whose links are threaded TIGHT through
# each other, as a Geometry Nodes chain arrives (public issue #157, the
# reporter's second file): eleven oval PDRD links 6 mm long with a 0.54 mm
# wire, pitched shorter than the taut length so neighbors cross, the two end
# links pinned, and a contact gap of 1 mm, which asks for more room on every
# side of a wire than the hole it is threaded through has.
#
# Two things unlink such a chain while the report calls the scene clear: one
# way out per PAIR of objects, which sums the two pieces of overlap on either
# side of a hole into a direction out of neither, and an unbounded pass, which
# carries a wire through the other link's wire. What is
# checked here is the property the user needs, measured on the result and
# never read from the report: every link is still threaded through its
# neighbors (the Gauss linking number of the two centerlines, which a link
# passing through a wire changes), no two surfaces cross by Blender's own
# triangle test, and no link moved farther than the room it had.
#
#   `A_threaded_chain_stays_linked`: the Resolve Intersections button at the
#       panel's defaults. The clearance cannot fit, so the run lowers it, and
#       the report and the panel say by how much.
#   `B_a_clearance_that_fits_is_used_as_set`: the same chain at a tenth of
#       the gap, which fits, is solved at that clearance.
#   `C_resolve_all_keeps_the_chain_linked`: the Resolve All Intersections
#       button runs the same rigid step first, with no SHELL in the scene.
#   `D_a_rod_through_a_block_leaves_sideways`: a rod passing clean through a
#       held block crosses its surface twice, and the two crossings bound ONE
#       piece of overlap (the side of the rod between them is inside the
#       block), so they are not given two ways out along the rod; and the
#       pair already crosses, so its exit is not cut into passes that drift.
#   `E_a_compressed_chain_is_cleared_by_clicking_again`: another chain of
#       the same links, crossing more, needs more than one click at the
#       panel's defaults; clicked until clear, it stays threaded.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


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

# An oval link: two straights of STRAIGHT joined by half circles of radius
# BEND, swept by a wire of radius WIRE. Its outer length is 6 mm, and a link
# threaded through the next has 0.23 mm on each side of its wire.
BEND = 1.31e-3
STRAIGHT = 2.3e-3
WIRE = 0.54e-3
ALONG, AROUND = 40, 8
COUNT = 11
GAP = 1.0e-3
# A meter out, as the reporter's chain is.
ORIGIN = np.array([-0.19, 0.0, 1.0])
# The taut pitch is 2 * (STRAIGHT / 2 + BEND - WIRE) = 3.84 mm.
PITCH = (2.6e-3, 3.0e-3)


def centerline(count):
    total = 2.0 * STRAIGHT + 2.0 * math.pi * BEND
    out, tangent = [], []
    for k in range(count):
        s = total * k / count
        if s < STRAIGHT:
            p, t = (-STRAIGHT / 2 + s, -BEND), (1.0, 0.0)
        elif s < STRAIGHT + math.pi * BEND:
            a = (s - STRAIGHT) / BEND - math.pi / 2
            p = (STRAIGHT / 2 + BEND * math.cos(a), BEND * math.sin(a))
            t = (-math.sin(a), math.cos(a))
        elif s < 2.0 * STRAIGHT + math.pi * BEND:
            p, t = (STRAIGHT / 2 - (s - STRAIGHT - math.pi * BEND), BEND), (-1.0, 0.0)
        else:
            a = (s - 2.0 * STRAIGHT - math.pi * BEND) / BEND + math.pi / 2
            p = (-STRAIGHT / 2 + BEND * math.cos(a), BEND * math.sin(a))
            t = (-math.sin(a), math.cos(a))
        out.append((p[0], p[1], 0.0))
        tangent.append((t[0], t[1], 0.0))
    return np.array(out), np.array(tangent)


def link_geometry():
    line, tangent = centerline(ALONG)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(tangent, up)
    angle = 2.0 * math.pi * np.arange(AROUND) / AROUND
    vert = (line[:, None] + WIRE * (np.cos(angle)[None, :, None] * side[:, None]
                                    + np.sin(angle)[None, :, None] * up)).reshape(-1, 3)
    faces = []
    for i in range(ALONG):
        for j in range(AROUND):
            a = i * AROUND + j
            b = ((i + 1) % ALONG) * AROUND + j
            c = ((i + 1) % ALONG) * AROUND + (j + 1) % AROUND
            d = i * AROUND + (j + 1) % AROUND
            faces.append((a, b, c, d))
    return vert, faces, centerline(64)[0]


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def fresh():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.api.solver.clear()
    ops.clear_last_report()
    shell_ops.clear_last_report()


def build_chain(seed=1):
    # The chain as Geometry Nodes would leave it: every link its own object,
    # its mesh in world coordinates and its transform the identity.
    vert, faces, line = link_geometry()
    rng = np.random.default_rng(seed)
    links, lines, x = [], [], 0.0
    for k in range(COUNT):
        turn = (rotation([1, 0, 0], (math.pi / 2 if k % 2 else 0.0)
                         + math.radians(rng.normal(0.0, 15.0)))
                @ rotation([0, 0, 1], math.radians(rng.normal(0.0, 7.5))))
        at = ORIGIN + np.array([x, 0.0, 0.0])
        mesh = bpy.data.meshes.new(f"Link{k}")
        mesh.from_pydata([tuple(p) for p in vert @ turn.T + at], [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"Link{k}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        links.append(obj)
        lines.append(line @ turn.T + at)
        x += rng.uniform(*PITCH)
    group = dh.api.solver.create_group("Chain", "PDRD")
    group.add(*[o.name for o in links])
    group.param.use_group_bounding_box_diagonal = False
    group.param.contact_gap = GAP
    group.param.contact_offset = 0.0
    for end in (links[0], links[-1]):
        hold = end.vertex_groups.new(name="hold")
        hold.add(list(range(len(end.data.vertices))), 1.0, "REPLACE")
        group.create_pin(end.name, "hold")
    return links, lines


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


def linking(c1, c2):
    # The Gauss linking integral of two closed polylines: +-1 for two links
    # threaded through each other, 0 for two apart. Midpoint rule on the
    # segments refined eight times, which is within a few hundredths here.
    def refine(c):
        step = np.roll(c, -1, axis=0) - c
        t = np.arange(8) / 8.0
        return (c[:, None] + t[None, :, None] * step[:, None]).reshape(-1, 3)
    a, b = refine(c1), refine(c2)
    da, db = np.roll(a, -1, 0) - a, np.roll(b, -1, 0) - b
    d = (a + 0.5 * da)[:, None] - (b + 0.5 * db)[None]
    cross = np.cross(da[:, None], db[None])
    return float(np.sum(np.einsum("ijk,ijk->ij", d, cross)
                        / np.linalg.norm(d, axis=2) ** 3) / (4.0 * math.pi))


def threaded(links, lines):
    # The linking number of each pair of neighbors, its centerline carried by
    # the transform the run wrote.
    out = []
    for k in range(len(links) - 1):
        placed = []
        for obj, line in ((links[k], lines[k]), (links[k + 1], lines[k + 1])):
            m = np.array(obj.matrix_world)
            placed.append(line @ m[:3, :3].T + m[:3, 3])
        out.append(round(abs(linking(*placed)), 3))
    return out


def displacement(obj, before):
    return float(np.linalg.norm(world_verts(obj) - before[obj.name], axis=1).max())


class RecordingLayout:
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
        self.sink.append(("label", kwargs.get("text", "")))

    def prop(self, *args, **kwargs):
        self.sink.append(("prop", args[1] if len(args) > 1 else ""))

    def operator(self, idname, **kwargs):
        class Settings:
            pass
        self.sink.append(("operator", idname))
        return Settings()


def panel_labels(draw):
    sink = []
    draw(RecordingLayout(sink), bpy.context)
    return [entry[1] for entry in sink if entry[0] == "label"]


def judge(name, links, lines, before, status, report):
    linked = threaded(links, lines)
    moved = {o.name: displacement(o, before) for o in links}
    left = crossing_pairs(links)
    ok = (
        status == {"FINISHED"}
        and report["initial_pairs"] != []
        and report["remaining_pairs"] == []
        and left == []
        and all(abs(v - 1.0) < 0.05 for v in linked)
        and moved[links[0].name] == 0.0
        and moved[links[-1].name] == 0.0
        # Half a link: the reporter's links were carried a whole link and
        # more when they came unlinked.
        and max(moved.values()) < 3.0e-3
    )
    details = {"status": sorted(status), "linking": linked,
               "crossing_after": left, "moved_mm": {k: v * 1e3 for k, v in moved.items()},
               "iterations": report["iterations"],
               "clearance_scale": report["clearance_scale"],
               "initial": len(report["initial_pairs"]),
               "remaining": report["remaining_pairs"]}
    return ok, details


try:
    dh = DriverHelpers(pkg, result)
    ops = __import__(pkg + ".mesh_ops.intersection_ops",
                     fromlist=["get_last_report"])
    shell_ops = __import__(pkg + ".mesh_ops.shell_intersection_ops",
                           fromlist=["get_last_report"])
    dh.log("setup_start")
    state = dh.groups.get_addon_data(bpy.context.scene).state

    # ----- A: the panel's defaults, which the chain cannot fit ----------
    fresh()
    links, lines = build_chain()
    crossing_before = crossing_pairs(links)
    linked_before = threaded(links, lines)
    before = {o.name: world_verts(o) for o in links}
    status = bpy.ops.object.resolve_intersections(
        max_iterations=state.intersection_fix_iterations,
        clearance=state.intersection_fix_clearance,
        selected_only=False)
    report = ops.get_last_report()
    ok, details = judge("A", links, lines, before, status, report)
    labels = panel_labels(ops.draw_intersection_correction)
    note = [t for t in labels if t.startswith("Clearance lowered to")]
    dh.record(
        "A_threaded_chain_stays_linked",
        ok
        and len(crossing_before) > 0
        and all(abs(v - 1.0) < 0.05 for v in linked_before)
        and state.intersection_fix_clearance == 1.0
        and report["clearance_scale"] < 1.0
        and note == [ops.clearance_note(report)],
        dict(details, crossing_before=len(crossing_before),
             linking_before=linked_before, note=note),
    )

    # ----- B: a clearance that fits is not lowered ---------------------
    # A run that spends its budget is repeated at half the clearance as
    # well, so the budget is the operator's soft maximum: what is checked is
    # that a clearance with room is kept, not how many passes it takes.
    fresh()
    links, lines = build_chain()
    before = {o.name: world_verts(o) for o in links}
    status = bpy.ops.object.resolve_intersections(
        max_iterations=200, clearance=0.1, selected_only=False)
    report = ops.get_last_report()
    ok, details = judge("B", links, lines, before, status, report)
    labels = panel_labels(ops.draw_intersection_correction)
    dh.record(
        "B_a_clearance_that_fits_is_used_as_set",
        ok
        and report["clearance_scale"] == 1.0
        and ops.clearance_note(report) is None
        and not any(t.startswith("Clearance lowered to") for t in labels),
        details,
    )

    # ----- C: Resolve All Intersections, whose first step is the same ---
    fresh()
    links, lines = build_chain()
    before = {o.name: world_verts(o) for o in links}
    status = bpy.ops.object.resolve_all_intersections(
        "EXEC_DEFAULT", max_iterations=50, clearance=1.0, selected_only=False)
    rigid = (shell_ops.get_last_report() or {}).get("rigid")
    ok, details = judge("C", links, lines, before, status, rigid)
    labels = panel_labels(shell_ops.draw_shell_intersection_correction)
    dh.record(
        "C_resolve_all_keeps_the_chain_linked",
        ok
        and rigid["clearance_scale"] < 1.0
        and any(t == ops.clearance_note(rigid) for t in labels),
        dict(details, labels=labels),
    )

    # ----- D: a rod through a held block --------------------------------
    fresh()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    block = bpy.context.active_object
    block.name = "Block"
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=0.1, depth=3.0, location=(0.2, 0.0, 0.0))
    rod = bpy.context.active_object
    rod.name = "Rod"
    solid = dh.api.solver.create_group("Blocks", "SOLID")
    solid.add("Block", "Rod")
    hold = block.vertex_groups.new(name="hold")
    hold.add(list(range(len(block.data.vertices))), 1.0, "REPLACE")
    solid.create_pin("Block", "hold")
    status = bpy.ops.object.resolve_intersections(
        max_iterations=50, clearance=1.0, selected_only=False)
    report = ops.get_last_report()
    where = tuple(rod.matrix_world.translation)
    dh.record(
        "D_a_rod_through_a_block_leaves_sideways",
        status == {"FINISHED"}
        and [tuple(p) for p in report["initial_pairs"]] == [("Block", "Rod")]
        and report["remaining_pairs"] == []
        and crossing_pairs([block, rod]) == []
        # Out through the near side: the rod spans x from 0.1 to 0.3 and the
        # block's face is at 0.5, so the rod moves by a little over 0.4 in x
        # and nowhere else: not along its own axis, which a push from the
        # samples' nearest faces would give it, and not along the block's
        # face, where a step cut into passes drifted it by 3 cm.
        and 0.6 < where[0] < 0.65
        and abs(where[1]) < 1.0e-3
        and abs(where[2]) < 1.0e-3,
        {"status": sorted(status), "rod": where,
         "remaining": report["remaining_pairs"],
         "iterations": report["iterations"]},
    )

    # ----- E: a chain pushed a quarter shorter than its taut length -----
    # More than one click's budget at the panel's defaults: each click keeps
    # what it did, so the next goes on from there, and none of them may
    # carry a link through another on the way.
    fresh()
    links, lines = build_chain(seed=2)
    crossing_before = crossing_pairs(links)
    before = {o.name: world_verts(o) for o in links}
    clicks = []
    for _click in range(4):
        status = bpy.ops.object.resolve_intersections(
            max_iterations=state.intersection_fix_iterations,
            clearance=state.intersection_fix_clearance,
            selected_only=False)
        report = ops.get_last_report()
        clicks.append(len(report["remaining_pairs"]))
        if not report["remaining_pairs"]:
            break
    ok, details = judge("E", links, lines, before, status, report)
    dh.record(
        "E_a_compressed_chain_is_cleared_by_clicking_again",
        ok and len(crossing_before) > 0,
        dict(details, crossing_before=len(crossing_before), clicks=clicks),
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
