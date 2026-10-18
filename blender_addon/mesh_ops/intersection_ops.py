# File: intersection_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Intersection Correction, an entry of the "Mesh Repair" panel.
#
# Transfer refuses a scene whose surfaces start through each other, and a
# chain converted from Geometry Nodes can arrive with hundreds of shallow
# overlaps between its links. This pushes the overlapping objects apart
# without the solver: each object moves as a rigid body, by the smallest
# motions of all of them that open every overlap
# (`core/intersection_correction.py`).
#
# WHICH OBJECTS. Only the included mesh objects of active SOLID and PDRD
# groups take part: those are the types whose surface encloses a volume, so
# "inside" is defined for them. A shell has no inside, and its overlaps are
# not this tool's.
#
# WHAT IS WRITTEN. An object's world transform and nothing else: no mesh,
# vertex group, pin or cache is touched, so everything that addresses the
# mesh by vertex index stays valid. A run that moves anything also takes down
# the violation overlay a refused Transfer left, which shows the scene as it
# was uploaded.
#
# WHAT IS HELD IN PLACE, decided from the scene rather than from a setting.
# A held object still pushes the others, which then take the whole
# correction. Two held objects that overlap each other are reported as still
# overlapping and left as they are.
#   - An object one of its group's pins holds: the pin is where the artist
#     anchored it.
#   - An object whose transform something else drives (its own curves, a
#     driver, an NLA strip or a constraint, on it or on a parent): a value
#     written to it would be overwritten on the next frame change.
#   - An object linked from another file, whose transform that file owns.
#   - With "Selected Only", every object that is not selected.
#
# WHAT IS LEFT OUT, by name. An object whose surface is open, non-manifold or
# inconsistently wound has no inside to push out of. Mesh Cleaning above
# reports the same defects and repairs the winding.

import numpy as np

import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore
from bpy.props import BoolProperty, FloatProperty, IntProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from mathutils import Matrix  # pyright: ignore

from ..core.client import communicator as com
from ..core.intersection_correction import Body, resolve
from ..core.utils import _may_move
from ..core.uuid_registry import resolve_assigned, resolve_pin
from ..models.groups import get_addon_data, iterate_active_object_groups
from .cleaning_ops import find_surface_defects

VOLUMETRIC_TYPES = ("SOLID", "PDRD")

HELD_PINNED = "pinned"
HELD_DRIVEN = "driven"
HELD_LINKED = "linked"
HELD_UNSELECTED = "unselected"

# The last run's outcome, for the panel. Cleared when the add-on unregisters.
_last_report = None

# What a running Resolve All Intersections is doing, for the panel and for the
# polls of both operators, or None when nothing runs (see
# `shell_intersection_ops`). The run is a modal operator that ends on every
# path out of it, and clears this as it ends.
_busy = None


def get_last_report():
    """The outcome of the last Resolve Intersections run, or None."""
    return _last_report


def clear_last_report():
    global _last_report
    _last_report = None


def get_busy():
    """The state of the run in progress (see `shell_intersection_ops`), or None."""
    return _busy


def set_busy(info):
    global _busy
    _busy = info


# Set by the Abort button, read by the modal operator that owns the run.
_abort = False


def request_abort():
    """Ask the run in progress to stop at its next tick."""
    global _abort
    _abort = True


def abort_requested() -> bool:
    return _abort


def clear_abort():
    global _abort
    _abort = False


def push_undo_before(message: str) -> None:
    """Make the scene as it is now an undo step of its own.

    An operator's undo step holds the scene AFTER it ran, so one undo goes back
    to the step before. That is the state before the button only when the
    state before was pushed, which a scene built by a script, or edited from the
    Python console, never did: one undo then lands on whatever was last pushed,
    which can be the startup scene. Pushing here, before anything is written,
    makes "undo" mean "the state before the button" whatever came before it.
    """
    bpy.ops.ed.undo_push(message=message)


def volumetric_entries(scene):
    """``(group, assigned, object)`` for every object that takes part."""
    for group in iterate_active_object_groups(scene):
        if group.object_type not in VOLUMETRIC_TYPES:
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is not None and obj.type == "MESH" and obj.data is not None:
                yield group, assigned, obj


def _is_pinned(group, obj) -> bool:
    for pin_item in group.pin_vertex_groups:
        # A malformed pin (an empty object reference) raises in resolve_pin.
        # It holds nothing, and the pin encoder is where it gets reported.
        try:
            if resolve_pin(pin_item) == obj:
                return True
        except ValueError:
            continue
    return False


def _parent_depth(obj) -> int:
    depth = 0
    while obj.parent is not None:
        obj = obj.parent
        depth += 1
    return depth


def held_reason(group, obj, selected_only: bool):
    """Why *obj* is held in place, or None when it is free to move."""
    if _is_pinned(group, obj):
        return HELD_PINNED
    if _may_move(obj):
        return HELD_DRIVEN
    if obj.library is not None:
        return HELD_LINKED
    if selected_only and not obj.select_get():
        return HELD_UNSELECTED
    return None


def collect_bodies(context, *, selected_only: bool, clearance: float):
    """The bodies of the scene, in the pose Transfer would ship.

    Returns ``(bodies, objects, held, skipped)``: the bodies for
    `resolve`, the Blender object behind each by name, ``{name: reason}`` for
    the ones held in place, and the names left out for having no closed,
    consistently wound surface.
    """
    from ..core.encoder.mesh import _shipped_local_verts, evaluate_at_start_frame
    from ..core.encoder.params import object_contact_lengths

    scene = context.scene
    state = get_addon_data(scene).state
    bodies, objects, held, skipped = [], {}, {}, []
    with evaluate_at_start_frame(context, state):
        for group, assigned, obj in volumetric_entries(scene):
            defects = find_surface_defects(obj.data)
            if defects["count"]:
                skipped.append(obj.name)
                continue
            mesh = obj.data
            rest = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", rest)
            local = np.asarray(
                _shipped_local_verts(
                    obj, context, state, group, rest.reshape(-1, 3)
                ),
                dtype=np.float64,
            )
            world = np.array(obj.matrix_world, dtype=np.float64)
            vert = local @ world[:3, :3].T + world[:3, 3]
            mesh.calc_loop_triangles()
            if not len(mesh.loop_triangles):
                skipped.append(obj.name)
                continue
            tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tri)
            reason = held_reason(group, obj, selected_only)
            if reason is not None:
                held[obj.name] = reason
            # The object's contact gap, which is the distance its barrier
            # engages at, and its contact offset, which Transfer refuses two
            # surfaces for being inside of. Both in Blender units, which is
            # what the geometry is in.
            gap, offset = object_contact_lengths(group, assigned, 1.0)
            bodies.append(
                Body(
                    obj.name, vert, tri.reshape(-1, 3),
                    movable=reason is None, offset=offset,
                    clearance=clearance * gap,
                )
            )
            objects[obj.name] = obj
    return bodies, objects, held, skipped


# Why `run_rigid` ran nothing.
STOPPED_FEW = "few"
STOPPED_HELD = "held"


def run_rigid(context, *, selected_only: bool, clearance: float,
              max_iterations: int):
    """Push apart the SOLID and PDRD objects of the scene and write the result.

    Returns ``(report, stopped)``. ``stopped`` is None when the passes ran, and
    ``report`` is then the dict `get_last_report` serves; otherwise ``report``
    is None and ``stopped`` is `STOPPED_FEW` (fewer than two closed objects) or
    `STOPPED_HELD` (every one of them is held in place). Raises ValueError when
    a pose cannot be shipped (`collect_bodies`).
    """
    bodies, objects, held, skipped = collect_bodies(
        context, selected_only=selected_only, clearance=clearance,
    )
    if len(bodies) < 2:
        return None, STOPPED_FEW
    if not any(body.movable for body in bodies):
        return None, STOPPED_HELD

    wm = context.window_manager
    wm.progress_begin(0, max_iterations)
    try:
        outcome = resolve(
            bodies, max_iterations=max_iterations,
            progress=lambda done, _left: wm.progress_update(done),
        )
    finally:
        wm.progress_end()

    # Every target is formed from the transforms the bodies were read at,
    # and a parent is written before its children: writing a world
    # transform solves for the object's own channels against its parent's
    # world transform, which has to be the parent's final one by then.
    targets = {
        body.name: Matrix(body.matrix().tolist()) @ objects[body.name].matrix_world
        for body in bodies
        if body.name in outcome["moved"]
    }
    for name in sorted(targets, key=lambda n: _parent_depth(objects[n])):
        objects[name].matrix_world = targets[name]
    context.view_layer.update()
    # A Transfer that was refused left its violations drawn over the
    # scene as it was uploaded. Those highlights now mark where the
    # surfaces were, so they come down; the next Transfer reports what is
    # left, if anything is.
    if targets:
        com.dismiss_violations()

    return {
        "iterations": outcome["iterations"],
        "objects": len(bodies),
        "initial_pairs": outcome["initial_pairs"],
        "remaining_pairs": outcome["remaining_pairs"],
        "moved": outcome["moved"],
        "clearance_scale": outcome["clearance_scale"],
        "held": held,
        "skipped": skipped,
    }, None


def clearance_note(report: dict) -> str | None:
    """The line saying the Clearance was lowered so the objects fit, or None
    when the run used the setting as it is (`resolve`)."""
    scale = report["clearance_scale"]
    if scale >= 1.0:
        return None
    return iface_(
        "Clearance lowered to {percent}% of the setting so the objects fit"
    ).format(percent=f"{100.0 * scale:.3g}")


class OBJECT_OT_ResolveIntersections(Operator):
    """Push apart SOLID and PDRD objects that start out overlapping, so
    Transfer accepts the scene. Each object moves as a rigid body, by the
    smallest motions that open every overlap; only object transforms change.
    Pinned objects, and objects whose transform is animated, constrained or
    linked from another file, stay where they are and the others move around
    them"""

    bl_idname = "object.resolve_intersections"
    bl_label = "Resolve Intersections"
    bl_options = {"REGISTER", "UNDO"}

    max_iterations: IntProperty(
        name="Max Iterations",
        description=(
            "How many passes to spend. Each pass finds the overlaps that are "
            "left and moves the objects once"
        ),
        default=50,
        min=1,
        soft_max=200,
    )  # pyright: ignore
    clearance: FloatProperty(
        name="Clearance",
        description=(
            "How far apart to leave surfaces that overlap, beyond their "
            "contact offsets, in multiples of each object's contact gap. "
            "Surfaces closer than half of this count as overlapping"
        ),
        default=1.0,
        min=0.1,
        soft_max=10.0,
    )  # pyright: ignore
    selected_only: BoolProperty(
        name="Selected Only",
        description=(
            "Move only the selected objects. The others stay where they are "
            "and the selected ones move around them"
        ),
        default=False,
    )  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and _busy is None

    def invoke(self, context, event):
        push_undo_before("Resolve Intersections (before)")
        return self.execute(context)

    def execute(self, context):
        global _last_report
        try:
            report, stopped = run_rigid(
                context, selected_only=self.selected_only,
                clearance=self.clearance, max_iterations=self.max_iterations,
            )
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if stopped == STOPPED_FEW:
            self.report(
                {"WARNING"},
                iface_("Needs two or more closed objects in SOLID or PDRD groups"),
            )
            return {"CANCELLED"}
        if stopped == STOPPED_HELD:
            self.report(
                {"WARNING"},
                iface_("Every SOLID or PDRD object is held in place"),
            )
            return {"CANCELLED"}

        _last_report = report
        moved, left = len(report["moved"]), len(report["remaining_pairs"])
        note = clearance_note(report)
        if not report["initial_pairs"]:
            self.report({"INFO"}, iface_("No overlapping objects found"))
        elif left:
            self.report(
                {"WARNING"},
                iface_(
                    "Moved {moved} object(s); {left} pair(s) still overlap "
                    "after {passes} pass(es)"
                ).format(moved=moved, left=left, passes=report["iterations"])
                + (f". {note}" if note else ""),
            )
        else:
            self.report(
                {"INFO"},
                iface_(
                    "Moved {moved} object(s) in {passes} pass(es); no overlap "
                    "is left. Run Transfer again"
                ).format(moved=moved, passes=report["iterations"])
                + (f". {note}" if note else ""),
            )
        return {"FINISHED"}


def draw_intersection_correction(layout, context):
    """Draw the Intersection Correction entry of the Mesh Repair panel.

    The button is drawn unconditionally and grayed out when its precondition
    fails, with a status label saying why. Nothing here reads geometry: the
    count is of group entries, and the outcome shown is the last run's.
    """
    scene = context.scene
    state = get_addon_data(scene).state
    box = layout.box()
    box.label(text="Intersection Correction")
    box.label(text="Push apart SOLID and PDRD objects that overlap.")

    col = box.column(align=True)
    col.prop(state, "intersection_fix_iterations")
    col.prop(state, "intersection_fix_clearance")
    box.prop(state, "intersection_fix_selected_only")

    count = sum(1 for _entry in volumetric_entries(scene))
    col = box.column()
    col.enabled = context.mode == "OBJECT" and count >= 2 and _busy is None
    op = col.operator(
        "object.resolve_intersections",
        text="Resolve Intersections",
        icon="MOD_PHYSICS",
    )
    op.max_iterations = state.intersection_fix_iterations
    op.clearance = state.intersection_fix_clearance
    op.selected_only = state.intersection_fix_selected_only

    if _busy is not None:
        box.label(text="Resolve All Intersections is running", icon="TIME")
        row = box.row()
        row.enabled = not _busy["canceling"]
        row.operator("object.resolve_all_intersections_abort", text="Abort", icon="CANCEL")
    elif context.mode != "OBJECT":
        box.label(text="Switch to Object Mode", icon="INFO")
    elif count < 2:
        box.label(
            text="Needs two or more objects in SOLID or PDRD groups", icon="INFO"
        )
    else:
        box.label(
            text=iface_("{count} SOLID or PDRD object(s)").format(count=count),
            icon="CHECKMARK",
        )

    report = get_last_report()
    if report is None:
        return
    result = box.box()
    if not report["initial_pairs"]:
        result.label(text="Last run: no overlapping objects found", icon="CHECKMARK")
    elif report["remaining_pairs"]:
        result.label(
            text=iface_("Last run: {left} of {total} pair(s) still overlap").format(
                left=len(report["remaining_pairs"]),
                total=len(report["initial_pairs"]),
            ),
            icon="ERROR",
        )
    else:
        result.label(
            text=iface_(
                "Last run: {total} overlapping pair(s) opened in {passes} pass(es)"
            ).format(
                total=len(report["initial_pairs"]), passes=report["iterations"]
            ),
            icon="CHECKMARK",
        )
    note = clearance_note(report)
    if note:
        result.label(text=note, icon="INFO")
    if report["moved"]:
        result.label(
            text=iface_("{count} object(s) moved, by at most {distance}").format(
                count=len(report["moved"]),
                distance=f"{max(report['moved'].values()):.4g}",
            )
        )
    if report["held"]:
        result.label(
            text=iface_("{count} object(s) held in place").format(
                count=len(report["held"])
            )
        )
    if report["skipped"]:
        result.label(
            text=iface_("{count} object(s) left out: surface not closed").format(
                count=len(report["skipped"])
            ),
            icon="INFO",
        )
        result.label(text="Scan them with Mesh Cleaning above.")


classes = (OBJECT_OT_ResolveIntersections,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    clear_last_report()
    set_busy(None)
