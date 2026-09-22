# File: scenarios/_allowance_matrix.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared driver library for the `bl_intersection_allowance_matrix_*`
# scenarios: the four intersection allowances, authored in Blender through the
# add-on's own data model, encoded, decoded and SIMULATED on the real solver,
# across every combination of group types.
#
# `rig_intersection_allowance_contact*` settle what an allowed pair does in
# the solver, but they author their scenes through `frontend` directly, so they
# cannot see whether the ADD-ON gets a checkbox, an "Apply to All Objects"
# switch, a narrowed object list or a pin flag onto the right objects of the
# right groups. `bl_intersection_allowances` reads the encoded files, which is
# a statement about bytes. This family is the statement about the SCENE: a
# checkbox ticked in Blender either lets one object pass through another in
# the simulated result or it does not.
#
# THE SHAPE. Every cell is one scene, built and run on its own:
#
#   HELD     an object lying flat with its top at z = 0, about 1 m wide, held
#            in place. A dynamic held object is fully pinned WITH a tiny
#            MOVE_BY (1 mm over the clip), which keeps it in the solved
#            namespace; a fully pinned object with no operation is promoted to
#            a rest-pose collider, which is a different namespace with a
#            different policy. A STATIC held object is either rest-pose (a
#            contact-only collision mesh) or animated by the same 1 mm
#            MOVE_BY as a static op (a pin shell in the solved namespace).
#   FALLING  a smaller object whose lowest point starts at z = 0.1, released
#            under gravity.
#   FLOOR    an invisible wall at z = -0.3, normal +z. It is an analytic
#            collider, which no allowance reaches, so every pass-through cell
#            also proves the allowance stayed inside the pairs it names.
#
# Where the pair is allowed the falling object drops through the held one and
# comes to rest on the floor; where it is not, it comes to rest on the held
# object. The verdict reads the falling object's REST HEIGHT, the mean z of
# its vertices on the last frame minus the height its mean sat above its own
# lowest point on the first frame. That puts every type on one scale whatever
# its thickness: about 0 resting on the held object, about -0.3 on the floor.
# The two outcomes are 0.3 apart and the threshold is their midpoint, so the
# verdict needs no tolerance tuned to a backend or a type.
#
# Layouts:
#
#   pair        held and falling objects in two groups of their own types.
#   same_group  held and falling objects in ONE group, so an inter-group
#               allowance must NOT cover them.
#   self        ONE object holding a pinned lower part and a free upper part,
#               so only a self allowance covers the pair. The verdict reads
#               the upper part's vertices only.
#   narrowed    the pair layout with a bystander object added to one group,
#               and that group's allowance narrowed ("Apply to All Objects"
#               off) to a list naming either the object in the pair or the
#               bystander.
#
# Every cell also requires the run to reach the last frame without the solver
# failing, and the falling object never to go through the floor.
#
# RUNNING ONE CELL. `--knob PPF_ALLOWANCE_MATRIX_CELLS=<a>,<b>` runs only the
# cells whose names contain one of the comma-separated substrings. A filtered
# run records `Z_matrix_complete` as FAILED, naming the cells it skipped, so a
# subset can never be read as a pass of the scenario.

from __future__ import annotations

import json

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


# The allowance keys a cell names, as the operators and the models spell them
# (`models/intersection_allowances.py`).
SELF = "self"
INTER_OBJECT = "inter_object"
INTER_GROUP = "inter_group"

# Falling types every held type is paired with.
FALLING_TYPES = ("SHELL", "SOLID", "ROD", "SAND", "PDRD")


def pair_cells(held: str) -> list[dict]:
    """The a) .. g) variants for one held type against every falling type.

    `held` is SHELL, SOLID, STATIC_REST or STATIC_ANIM. A rest-pose STATIC
    collider is a contact-only collision mesh that carries no policy of its
    own, so its OWN checkboxes cannot let anything through (c and e expect
    held), while the falling object's flags still do (b and d). The per-pin
    flag (g) exists only where the held object carries a pin item: a STATIC
    group draws "Transform" operations instead of pins, and the encoder skips
    STATIC pins, so g is not authored for either STATIC held type.
    """
    rest_static = held == "STATIC_REST"
    cells = []
    for falling in FALLING_TYPES:
        base = "held_%s.fall_%s." % (held, falling)
        variants = [
            ("a_no_flag", {}, {}, False, "held"),
            ("b_inter_object_on_falling", {}, {"falling": [INTER_OBJECT]},
             False, "pass"),
            ("c_inter_object_on_held", {"held": [INTER_OBJECT]}, {}, False,
             "held" if rest_static else "pass"),
            ("d_inter_group_on_falling", {}, {"falling": [INTER_GROUP]},
             False, "pass"),
            ("e_inter_group_on_held", {"held": [INTER_GROUP]}, {}, False,
             "held" if rest_static else "pass"),
            ("f_self_on_both", {"held": [SELF]}, {"falling": [SELF]}, False,
             "held"),
        ]
        if held in ("SHELL", "SOLID"):
            variants.append(("g_pin_allow_on_held", {}, {}, True, "pass"))
        for tag, held_flags, falling_flags, pin_allow, expect in variants:
            flags = {}
            flags.update(held_flags)
            flags.update(falling_flags)
            cells.append({
                "name": base + tag,
                "layout": "pair",
                "held": held,
                "falling": falling,
                "flags": flags,
                "pin_allow": pin_allow,
                "expect": expect,
            })
    return cells


def same_group_cells(kind: str) -> list[dict]:
    """Held and falling objects of one type in ONE group."""
    base = "same_group_%s." % kind
    return [
        {"name": base + "no_flag", "layout": "same_group", "held": kind,
         "falling": kind, "flags": {}, "pin_allow": False, "expect": "held"},
        {"name": base + "inter_group_still_collides", "layout": "same_group",
         "held": kind, "falling": kind, "flags": {"group": [INTER_GROUP]},
         "pin_allow": False, "expect": "held"},
        {"name": base + "inter_object_passes", "layout": "same_group",
         "held": kind, "falling": kind, "flags": {"group": [INTER_OBJECT]},
         "pin_allow": False, "expect": "pass"},
    ]


def self_cells(kind: str) -> list[dict]:
    """One object of `kind`, a pinned lower part under a free upper part."""
    base = "self_%s." % kind
    return [
        {"name": base + "allow_self_passes", "layout": "self", "held": kind,
         "falling": kind, "flags": {"group": [SELF]}, "pin_allow": False,
         "expect": "pass"},
        {"name": base + "no_flag", "layout": "self", "held": kind,
         "falling": kind, "flags": {}, "pin_allow": False, "expect": "held"},
        {"name": base + "inter_object_only_is_held", "layout": "self",
         "held": kind, "falling": kind, "flags": {"group": [INTER_OBJECT]},
         "pin_allow": False, "expect": "held"},
        {"name": base + "inter_group_only_is_held", "layout": "self",
         "held": kind, "falling": kind, "flags": {"group": [INTER_GROUP]},
         "pin_allow": False, "expect": "held"},
    ]


def narrowed_cells() -> list[dict]:
    """Inter-group narrowed to a named object, on either side of the pair.

    The held and falling objects are SHELL grids in two SHELL groups, and the
    narrowed group also holds a bystander far away. Listing the object in the
    pair lets the pair through; listing the bystander, with the checkbox
    still on, must not.
    """
    cells = []
    for side in ("falling", "held"):
        for listed, expect in (("member", "pass"), ("bystander", "held")):
            cells.append({
                "name": "narrowed.inter_group_on_%s.lists_%s" % (side, listed),
                "layout": "narrowed",
                "held": "SHELL",
                "falling": "SHELL",
                "flags": {side: [INTER_GROUP]},
                "pin_allow": False,
                "narrow": {"side": side, "allowance": INTER_GROUP,
                           "listed": listed},
                "expect": expect,
            })
    return cells


MATRIX_LIB = r'''
import json
import os
import time
import traceback

import numpy as np

# Geometry and time base, shared by every cell.
FLOOR = -0.3          # the invisible wall, normal +z
HELD_TOP = 0.0        # top surface of every held object
DROP = 0.1            # lowest point of every falling object at rest
MIDPOINT = 0.5 * (FLOOR + HELD_TOP)
FLOOR_TOL = 0.02      # how far below the floor a vertex may read and still
                      # be "on" it: the SOLID and PDRD surface is written back
                      # through a mapping, and a grain center sits a radius up
HOLD_DELTA = (0.0, 0.0, 0.001)   # the tiny move that keeps a pinned object
                                 # dynamic
# 13 Blender frames at 20 fps is 0.6 s of simulated time: the falling object
# reaches the floor 0.29 s in and has settled well before the end.
FRAME_COUNT = 13
FRAME_RATE = 20
STEP = 0.01
# SAND follows the granular scene invariants: grain spacing over
# 2 * radius, a contact gap of one radius, and a small step, because there is
# no point-point CCD and a grain must never step over the barrier.
SAND_STEP = 0.002
SAND_R = 0.02
BUILD_TIMEOUT = 600.0
RUN_TIMEOUT = 900.0

CELL_FILTER = [s for s in os.environ.get(
    "PPF_ALLOWANCE_MATRIX_CELLS", "").split(",") if s.strip()]


# ---------------------------------------------------------------- geometry --

def am_grid(size, n, z, cx=0.0, cy=0.0):
    h = 0.5 * size
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((cx - h + size * i / n, cy - h + size * j / n, z))
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    return verts, [], faces


def am_box(sx, sy, z0, z1, cx=0.0, cy=0.0):
    x0, x1 = cx - 0.5 * sx, cx + 0.5 * sx
    y0, y1 = cy - 0.5 * sy, cy + 0.5 * sy
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    # Outward normals, so the tetrahedralizer and PDRD's volume see a
    # positively oriented closed surface.
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, [], faces


def am_rod(width, n_lines, n_pts, z, along, cx=0.0, cy=0.0):
    # One CONNECTED polyline: n_lines parallel runs joined end to end, so the
    # rod covers a square patch without being several disconnected strands.
    verts = []
    for k in range(n_lines):
        t = -0.5 * width + width * k / (n_lines - 1)
        run = [-0.5 * width + width * i / (n_pts - 1) for i in range(n_pts)]
        if k % 2:
            run = run[::-1]
        for s in run:
            if along == "x":
                verts.append((cx + s, cy + t, z))
            else:
                verts.append((cx + t, cy + s, z))
    edges = [(i, i + 1) for i in range(len(verts) - 1)]
    return verts, edges, []


def am_sand(n, layers, spacing, z0, cx=0.0, cy=0.0):
    # z0 is the LOWEST grain center.
    verts = []
    for k in range(layers):
        for j in range(n):
            for i in range(n):
                verts.append((cx + (i - 0.5 * (n - 1)) * spacing,
                              cy + (j - 0.5 * (n - 1)) * spacing,
                              z0 + k * spacing))
    return verts, [], []


def am_held_geometry(kind, cx=0.0, cy=0.0):
    # A held object of `kind` with its top at HELD_TOP.
    if kind in ("SHELL", "STATIC_REST", "STATIC_ANIM"):
        return am_grid(1.0, 8, HELD_TOP, cx, cy)
    if kind == "SOLID":
        return am_box(1.0, 1.0, HELD_TOP - 0.1, HELD_TOP, cx, cy)
    if kind == "PDRD":
        return am_box(0.8, 0.8, HELD_TOP - 0.1, HELD_TOP, cx, cy)
    if kind == "ROD":
        # Thirteen runs along x, 5 cm apart: a falling rod whose runs are
        # along y crosses them and cannot slip between.
        return am_rod(0.6, 13, 7, HELD_TOP, "x", cx, cy)
    if kind == "SAND":
        # One DENSE layer, 2.2 radii apart: a grain cannot drop through the
        # hole at the center of a grid square (1.56 radii from each of
        # its corners, against the 2 radii contact needs), while the start
        # is still penetration-free.
        return am_sand(9, 1, 2.2 * SAND_R, HELD_TOP - SAND_R, cx, cy)
    raise ValueError("no held geometry for %r" % (kind,))


def am_falling_geometry(kind, cx=0.0, cy=0.0):
    # A falling object of `kind` with its lowest point at DROP.
    if kind == "SHELL":
        return am_grid(0.5, 5, DROP, cx, cy)
    if kind in ("SOLID", "PDRD"):
        return am_box(0.2, 0.2, DROP, DROP + 0.2, cx, cy)
    if kind == "ROD":
        return am_rod(0.3, 4, 6, DROP, "y", cx, cy)
    if kind == "SAND":
        return am_sand(3, 2, 3.0 * SAND_R, DROP + SAND_R, cx, cy)
    raise ValueError("no falling geometry for %r" % (kind,))


def am_link(name, verts, edges, faces):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([tuple(float(c) for c in v) for v in verts],
                     list(edges), list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if obj.name != name:
        # Every lookup below is by name; a suffixed duplicate would make the
        # cell measure an object from an earlier cell.
        raise RuntimeError("object %r was created as %r: the scene was not "
                           "cleared" % (name, obj.name))
    return obj


def am_mark_particles(obj):
    # A SAND group admits only a committed particle mesh. The Convert
    # operator (`ui/dynamics/sand_ops.build_and_commit_particle_mesh`) seeds
    # grains at random and stamps these three properties; the cells need a
    # REGULAR grid instead (the spacing invariant above and a held layer dense
    # enough that no grain drops through its holes), so the grid is authored
    # here and stamped the same way. The render-only "Particle Mesh"
    # geometry-nodes modifier is left off: nothing the solver receives
    # depends on it.
    obj["particle_mesh"] = 1
    obj["seed_count"] = len(obj.data.vertices)
    obj["grain_radius"] = SAND_R


def am_group_type(kind):
    return "STATIC" if kind.startswith("STATIC") else kind


# ------------------------------------------------------------------ scene --

def am_reset_scene(dh):
    # Groups first, while their members still exist, then every object and
    # the meshes they leave behind, so the next cell's names are free.
    # The operator behind delete_all_groups polls False on a scene with no
    # group, so it is asked only when there is one to delete.
    if dh.api.solver.get_groups():
        dh.api.solver.delete_all_groups()
    if dh.api.solver.get_groups():
        raise RuntimeError("groups survived delete_all_groups: %r"
                           % [g.name for g in dh.api.solver.get_groups()])
    dh.api.solver.clear_invisible_colliders()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def am_group_pg(dh, group):
    pg = dh.groups.get_group_by_uuid(bpy.context.scene, group.uuid)
    if pg is None:
        raise RuntimeError("group %r has no property group" % (group.name,))
    return pg


def am_slot(dh, group):
    groups_mod = __import__(dh.pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    return groups_mod.get_group_slot_index(bpy.context.scene, group.uuid)


def am_set_flags(pg, keys):
    specs = __import__(pkg + ".models.intersection_allowances",
                       fromlist=["allowance_by_key"])
    for key in keys:
        spec = specs.allowance_by_key(key)
        setattr(pg, spec.enable_prop, True)


def am_pin(group, obj, indices, hold_move, allow):
    # Pin `indices` of `obj` in `group`, optionally with the tiny MOVE_BY and
    # the per-pin allowance. Returns the pin item, read back off the group.
    obj.vertex_groups.new(name="Hold").add(list(indices), 1.0, "REPLACE")
    pin = group.create_pin(obj.name, "Hold")
    if hold_move:
        pin.move_by(delta=HOLD_DELTA, frame_start=1, frame_end=FRAME_COUNT)
    _, item = pin._find_pin_item()
    if item is None:
        raise RuntimeError("no pin item for %r" % (obj.name,))
    if allow:
        item.allow_intersection = True
    return item


def am_static_move(dh, group, obj):
    # A STATIC collider animated by a UI static op, the same tiny MOVE_BY the
    # dynamic held objects carry.
    pg = am_group_pg(dh, group)
    uuid_registry = __import__(dh.pkg + ".core.uuid_registry",
                               fromlist=["get_or_create_object_uuid"])
    obj_uuid = uuid_registry.get_or_create_object_uuid(obj)
    for assigned in pg.assigned_objects:
        if assigned.uuid == obj_uuid:
            op = assigned.static_ops.add()
            op.op_type = "MOVE_BY"
            op.delta = HOLD_DELTA
            op.frame_start = 1
            op.frame_end = FRAME_COUNT
            op.transition = "LINEAR"
            return
    raise RuntimeError("%r is not assigned to %r" % (obj.name, group.name))


def am_configure_sand(dh, group):
    pg = am_group_pg(dh, group)
    pg.contact_gap = SAND_R
    pg.sand_friction = 0.5


def am_narrow(dh, group, allowance, obj):
    # "Apply to All Objects" off and the list set to `obj` alone, through the
    # panel's own operator.
    specs = __import__(pkg + ".models.intersection_allowances",
                       fromlist=["allowance_by_key"])
    spec = specs.allowance_by_key(allowance)
    pg = am_group_pg(dh, group)
    setattr(pg, spec.all_objects_prop, False)
    slot = am_slot(dh, group)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    got = bpy.ops.object.add_intersection_allowance_objects(
        group_index=slot, allowance=allowance)
    listed = [item.name for item in getattr(pg, spec.objects_prop)]
    if got != {"FINISHED"} or listed != [obj.name]:
        raise RuntimeError("narrowing %r to %r gave %r, listed %r"
                           % (allowance, obj.name, got, listed))
    return listed


def am_author(dh, cell):
    # Author one cell's scene. Returns (measured object, measured vertex
    # indices, notes).
    layout = cell["layout"]
    flags = cell.get("flags", {})
    notes = {}
    sand = cell["held"] == "SAND" or cell["falling"] == "SAND"

    if layout == "self":
        kind = cell["held"]
        hv, he, hf = am_held_geometry(kind)
        fv, fe, ff = am_falling_geometry(kind)
        n_low = len(hv)
        verts = list(hv) + list(fv)
        edges = list(he) + [(a + n_low, b + n_low) for a, b in fe]
        faces = list(hf) + [tuple(i + n_low for i in f) for f in ff]
        obj = am_link("Both", verts, edges, faces)
        if kind == "SAND":
            am_mark_particles(obj)
        group = dh.api.solver.create_group("Self", kind)
        group.add(obj.name)
        if kind == "SAND":
            am_configure_sand(dh, group)
        # The lower part is pinned and the upper part is free, so the object
        # is never fully pinned and stays dynamic without a move.
        am_pin(group, obj, range(n_low), False, False)
        am_set_flags(am_group_pg(dh, group), flags.get("group", []))
        notes["lower_vertices"] = n_low
        notes["upper_vertices"] = len(fv)
        return obj, list(range(n_low, len(verts))), notes, sand

    held_kind = cell["held"]
    fall_kind = cell["falling"]
    held = am_link("Held", *am_held_geometry(held_kind))
    falling = am_link("Falling", *am_falling_geometry(fall_kind))
    for obj, kind in ((held, held_kind), (falling, fall_kind)):
        if kind == "SAND":
            am_mark_particles(obj)

    if layout == "same_group":
        group = dh.api.solver.create_group("Both", am_group_type(held_kind))
        group.add(held.name)
        group.add(falling.name)
        held_group = fall_group = group
    else:
        held_group = dh.api.solver.create_group(
            "HeldGroup", am_group_type(held_kind))
        held_group.add(held.name)
        fall_group = dh.api.solver.create_group(
            "FallingGroup", am_group_type(fall_kind))
        fall_group.add(falling.name)
    for group, kind in ((held_group, held_kind), (fall_group, fall_kind)):
        if kind == "SAND":
            am_configure_sand(dh, group)

    if held_kind == "STATIC_ANIM":
        am_static_move(dh, held_group, held)
    elif held_kind != "STATIC_REST":
        am_pin(held_group, held, range(len(held.data.vertices)), True,
               cell.get("pin_allow", False))

    if layout == "same_group":
        am_set_flags(am_group_pg(dh, held_group), flags.get("group", []))
    else:
        am_set_flags(am_group_pg(dh, held_group), flags.get("held", []))
        am_set_flags(am_group_pg(dh, fall_group), flags.get("falling", []))

    if layout == "narrowed":
        narrow = cell["narrow"]
        side_group = held_group if narrow["side"] == "held" else fall_group
        member = held if narrow["side"] == "held" else falling
        # A bystander of the same type in the narrowed group, far from the
        # pair. It is free and falls to the floor on its own.
        bystander = am_link(
            "Bystander", *am_falling_geometry(fall_kind, cx=3.0))
        side_group.add(bystander.name)
        target = member if narrow["listed"] == "member" else bystander
        notes["listed"] = am_narrow(dh, side_group, narrow["allowance"],
                                    target)
    return falling, list(range(len(falling.data.vertices))), notes, sand


# -------------------------------------------------------------- pipeline --

def am_build(dh, message, first, local_path, server_port, project_name):
    # One build, waited on until the server echoes THIS payload's hashes.
    # An upload invalidates the build the server held, so once both hashes
    # are this payload's, a terminal solver state is this build's verdict and
    # not the previous cell's.
    data_hash = dh.encoder_mesh.compute_data_hash(bpy.context)
    param_hash = dh.encoder_param.compute_param_hash(bpy.context)
    data_bytes = dh.encoder_mesh.encode_obj(bpy.context)
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    if first:
        dh.connect(local_path=local_path, server_port=server_port,
                   project_name=project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=data_hash, param_hash=param_hash, message=message,
    ))
    deadline = time.time() + BUILD_TIMEOUT
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")
                and s.server_data_hash == data_hash
                and s.server_param_hash == param_hash):
            return s.solver.name
        time.sleep(0.1)
    s = dh.facade.engine.state
    raise RuntimeError(
        "build %r did not produce its own verdict within %ss: solver=%s "
        "activity=%s data_hash echoed=%s param_hash echoed=%s error=%r"
        % (message, BUILD_TIMEOUT, s.solver.name, s.activity.name,
           s.server_data_hash == data_hash, s.server_param_hash == param_hash,
           s.error))


def am_measure(arr, idx):
    first = arr[0][idx]
    last = arr[-1][idx]
    lift = float(first[:, 2].mean() - first[:, 2].min())
    return {
        "rest_z": round(float(last[:, 2].mean()) - lift, 4),
        "mean_z_last": round(float(last[:, 2].mean()), 4),
        "min_z_last": round(float(last[:, 2].min()), 4),
        "min_z_first": round(float(first[:, 2].min()), 4),
        "finite": bool(np.all(np.isfinite(arr))),
    }


def am_verdict(expect, m):
    if not m["finite"]:
        return False
    if m["min_z_last"] < FLOOR - FLOOR_TOL:
        # Through the floor: an allowance reached the analytic wall.
        return False
    if expect == "pass":
        # Below the midpoint AND touching the floor, so an object that
        # stopped in mid-air (a stalled solve, a lost frame) is not a pass.
        return m["rest_z"] < MIDPOINT and m["min_z_last"] < FLOOR + 0.05
    return MIDPOINT < m["rest_z"] < DROP + 0.05


def am_keep_session(cell_name, project_name):
    # Every cell builds into the same session directory, so the next cell
    # overwrites a failure's evidence. Copy it aside under the worker
    # directory, where `--keep-all` or a failing verdict keeps it, and return
    # where it went plus the tail of the solver's own error log.
    import shutil
    workspace = os.path.dirname(PROBE_DIR)
    session = os.path.join(workspace, "project", project_name, "session")
    if not os.path.isdir(session):
        return {"session": "absent: " + session}
    dest = os.path.join(workspace, "failed_cells", cell_name)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(session, dest)
    tail = ""
    err = os.path.join(session, "error.log")
    if os.path.isfile(err):
        with open(err, "rb") as f:
            tail = f.read()[-800:].decode("utf-8", "replace")
    return {"session_copy": dest, "error_log_tail": tail}


def am_run_cell(dh, cell, first, local_path, server_port, project_name):
    t0 = time.time()
    details = {"expect": cell["expect"], "layout": cell["layout"],
               "held": cell["held"], "falling": cell["falling"],
               "flags": cell.get("flags", {}),
               "pin_allow": cell.get("pin_allow", False)}
    am_reset_scene(dh)
    obj, idx, notes, sand = am_author(dh, cell)
    details.update(notes)
    root = dh.configure_state(project_name=project_name,
                              frame_count=FRAME_COUNT, frame_rate=FRAME_RATE,
                              step_size=SAND_STEP if sand else STEP,
                              gravity=(0.0, 0.0, -9.8))
    root.state.disable_contact = False
    dh.api.solver.add_wall(position=(0.0, 0.0, FLOOR), normal=(0.0, 0.0, 1.0))
    details["step_size"] = SAND_STEP if sand else STEP

    solver = am_build(dh, "allowance-matrix:" + cell["name"], first,
                      local_path, server_port, project_name)
    details["build"] = solver
    details["build_s"] = round(time.time() - t0, 1)
    if solver == "FAILED":
        details["error"] = build_failure_message(dh.facade, dh.com)[:1500]
        details["seconds"] = round(time.time() - t0, 1)
        return False, details

    t_run = time.time()
    saw_running = dh.run_and_wait(timeout=RUN_TIMEOUT)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=60.0)
    s = dh.facade.engine.state
    details["run_s"] = round(time.time() - t_run, 1)
    details["saw_running"] = saw_running
    details["solver_after_run"] = s.solver.name
    details["frame"] = int(s.frame)
    if s.solver.name == "FAILED":
        details["error"] = build_failure_message(
            dh.facade, dh.com, prefix="run failed")[:1500]
    ran_to_end = (saw_running and s.solver.name in ("READY", "RESUMABLE")
                  and int(s.frame) >= FRAME_COUNT - 1)

    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    details["fetched"] = [applied, total]
    pc2_path = dh.find_pc2_for(obj)
    if not pc2_path or not os.path.isfile(pc2_path):
        details["error"] = details.get("error", "") + (
            " no PC2 for %r (path=%r)" % (obj.name, pc2_path))
        details["seconds"] = round(time.time() - t0, 1)
        return False, details
    arr = dh.read_pc2(pc2_path).copy()
    details["pc2_samples"] = int(arr.shape[0])
    details["pc2_vertices"] = int(arr.shape[1])
    details["mesh_vertices"] = len(obj.data.vertices)
    complete = (arr.shape[0] >= FRAME_COUNT
                and arr.shape[1] == len(obj.data.vertices))
    m = am_measure(arr, idx)
    details.update(m)
    ok = ran_to_end and complete and am_verdict(cell["expect"], m)
    details["seconds"] = round(time.time() - t0, 1)
    return ok, details


def am_selected(cell):
    if not CELL_FILTER:
        return True
    return any(s.strip() in cell["name"] for s in CELL_FILTER)


def am_run_matrix(dh, cells, local_path, server_port, project_name):
    started = time.time()
    first = True
    skipped = []
    for cell in cells:
        if not am_selected(cell):
            skipped.append(cell["name"])
            continue
        dh.log("cell_start %s" % cell["name"])
        try:
            ok, details = am_run_cell(dh, cell, first, local_path,
                                      server_port, project_name)
        except Exception as exc:
            ok = False
            details = {"expect": cell["expect"],
                       "exception": "%s: %s" % (type(exc).__name__, exc),
                       "traceback": traceback.format_exc()[-1500:]}
        first = False
        if not ok:
            try:
                details.update(am_keep_session(cell["name"], project_name))
            except Exception as exc:
                details["session_copy_error"] = "%s: %s" % (
                    type(exc).__name__, exc)
        dh.record(cell["name"], ok, details)
        dh.log("cell_end %s ok=%s rest_z=%s s=%s"
               % (cell["name"], ok, details.get("rest_z"),
                  details.get("seconds")))
    dh.record("Z_matrix_complete", not skipped,
              {"cells": len(cells), "ran": len(cells) - len(skipped),
               "skipped": skipped, "filter": CELL_FILTER,
               "wall_s": round(time.time() - started, 1)})
'''


_DRIVER_BODY = r'''
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
CELLS = json.loads(<<CELLS>>)

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start cells=%d" % len(CELLS))
    am_reset_scene(dh)
    dh.save_blend(PROBE_DIR, "allowance_matrix.blend")
    am_run_matrix(dh, CELLS, LOCAL_PATH, SERVER_PORT, PROJECT_NAME)
except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
'''


def build_driver(cells: list[dict], ctx: r.ScenarioContext) -> str:
    """The Blender-side driver for `cells`."""
    return (
        dl.DRIVER_LIB + MATRIX_LIB + _DRIVER_BODY
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
        .replace("<<CELLS>>", repr(json.dumps(cells)))
    )


# A cell on the CPU backend is a build plus a run of 60 solver steps (300 for
# SAND); the budget is generous so a slow host fails a CELL on its own
# timeouts rather than the whole scenario on this one.
PER_CELL_BUDGET_S = 300.0


def run(ctx: r.ScenarioContext, cells: list[dict]) -> dict:
    timeout = max(ctx.timeout, 120.0 + PER_CELL_BUDGET_S * len(cells))
    result, err = r.wait_blender_result(ctx, timeout=timeout)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}),
                                 label="allowance matrix cells",
                                 max_violations=12)
