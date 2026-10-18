# File: shell_intersection_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resolve All Intersections, the second entry of Intersection Correction in the
# "Mesh Repair" panel.
#
# Resolve Intersections (`intersection_ops.py`) pushes SOLID and PDRD objects
# apart as rigid bodies and writes object transforms only. A SHELL has no inside
# to push out of and a rigid motion cannot open a fold of one sheet through
# itself, so this entry deforms the cloth instead
# (`core/shell_untangle.py`). It does both, in order:
#
#   1. The rigid pass of Resolve Intersections, for the SOLID and PDRD objects.
#   2. The SHELL objects, deformed until no triangle of one is inside the
#      contact offsets of a triangle of another, or of itself. The SOLID, PDRD
#      and STATIC objects are held where they are, in the pose step 1 left,
#      and push the cloth.
#
# WHAT IS WRITTEN. The vertex coordinates of the SHELL objects that moved, in
# the object's own space, and the transforms step 1 wrote. No other mesh data
# is touched, so vertex indices, groups, pins and UVs stay valid. The operator
# is registered with UNDO, so one Ctrl+Z takes back both steps.
#
# WHAT IS HELD IN PLACE, decided from the scene. A held surface still pushes the
# cloth around it. A pin holds exactly the vertices of its vertex group, and
# every vertex of an object that is held as a whole:
#   - a vertex a pin of its group holds;
#   - an object whose mesh cannot be written: linked from another file, shared
#     with another object, carrying shape keys, or deformed at the starting
#     frame by a modifier stack (the pose Transfer ships is not the mesh, so
#     there is no coordinate to write it to);
#   - with "Selected Only", every object that is not selected.
#
# THE RUN. Setup and the rigid pass take seconds. The shell pass grows with the
# mesh, from a few seconds to a few minutes at ten thousand vertices, so a
# button press runs it in a child process (`shell_untangle.Run`) under a modal
# operator that shows progress and ends on its Abort button. It is a process
# and not a thread because a second Python thread beside Blender's interface
# corrupts the interpreter with this add-on loaded (`Run`). A call from a script
# or the MCP tool runs it in line. Abort in the first stage leaves every shell as
# it was; in the second it keeps the embedded state reached so far.
#
# The interface stays usable while the run goes: the operator lets every event
# but its own timer through. So the scene can be edited under the run, and the
# result is written only over a mesh that still holds the coordinates and matrix
# it was read with (`_changed_target`); an edited one is left as the artist has
# it, and the run says so.
#
# The modal operator exists only while the run does, so no modal handler stays
# attached for the session (Blender will not auto-save while one is).

import math
import time

from dataclasses import dataclass, field, replace

import bpy  # pyright: ignore
import numpy as np

from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore
from bpy.props import BoolProperty, FloatProperty, IntProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..core.client import communicator as com
from ..core.encoder.pin import pin_vertex_indices
from ..core.intersection_correction import resolvable_clearance
from ..core.shell_untangle import Report, Run, Settings, Surface, untangle
from ..core.uuid_registry import resolve_assigned, resolve_pin
from ..models.groups import (
    decode_vertex_group_identifier,
    get_addon_data,
    iterate_active_object_groups,
)
from .intersection_ops import (
    HELD_LINKED,
    HELD_UNSELECTED,
    abort_requested,
    clear_abort,
    clearance_note,
    get_busy,
    push_undo_before,
    request_abort,
    run_rigid,
    set_busy,
    volumetric_entries,
)

SHELL_TYPE = "SHELL"
# Held surfaces: they take no part in the deformation and push the cloth.
COLLIDER_TYPES = ("STATIC", "SOLID", "PDRD")

HELD_SHARED = "shared mesh"
HELD_SHAPE_KEYS = "shape keys"
HELD_DEFORMED = "deformed"
HELD_TRANSFORM = "transform"
HELD_PINNED = "pinned"

# Half a unit of single-precision resolution at a coordinate of magnitude 1. A
# vertex that moved by less than this times the largest coordinate cannot be
# written as a different value, and is left alone, bit for bit.
_HALF_ULP = 0.5 * float(np.finfo(np.float32).eps)

# How often the modal operator looks at the run, in seconds.
_POLL_SECONDS = 0.25

# How many solves one press may take. Blender triangulates a quad from its shape,
# so a solve that bent quads can leave a mesh triangulated differently from the
# input it was given, and what Transfer ships is the new triangles
# (`_retriangulated`). The next solve starts from an almost clear scene.
_MAX_PASSES = 4

# The last run's outcome, for the panel. Cleared when the add-on unregisters.
_last_report = None


def get_last_report():
    """The outcome of the last Resolve All Intersections run, or None."""
    return _last_report


def clear_last_report():
    global _last_report
    _last_report = None


@dataclass
class Target:
    """A SHELL mesh the result is written to."""

    obj: object
    # The 4x4 that takes the world position the run works in to the object's
    # own space, at the starting frame the pose was read at.
    inverse: np.ndarray
    # The stored coordinates and the world matrix, as the scene holds them when
    # the run is given its input. The result is written over a mesh that still
    # holds them.
    co: np.ndarray
    matrix: np.ndarray | None


@dataclass
class Scene:
    """What the shell pass is given, and what is needed to write it back.

    Attributes:
        surfaces: Every SHELL, SOLID, PDRD and STATIC mesh of the active groups,
            in world space at the starting frame.
        targets: ``{surface name: Target}`` for the SHELL meshes that can be
            written.
        held: ``{name: reason}`` for the SHELL objects held as a whole.
        pinned: ``{name: count}`` of the vertices a pin holds, for the SHELL
            objects that are otherwise free.
        colliders: Number of SOLID, PDRD and STATIC meshes in ``surfaces``.
        skipped: SHELL objects with no triangles.
        clearance: The distance asked on top of the contact offsets, in Blender
            units.
        movable: True when some vertex may move.
    """

    surfaces: list
    targets: dict
    held: dict
    pinned: dict
    colliders: int
    skipped: list
    clearance: float
    movable: bool


@dataclass
class Plan:
    """The rigid pass's outcome and the shell scene read after it.

    One press can take several solves (`_MAX_PASSES`). The scene and settings
    are those of the solve going now; the rest adds the solves up, so that the
    report measures from where the cloth was when the press began.
    """

    rigid: dict | None
    scene: Scene
    settings: Settings
    passes: int = 1
    # World positions at the very start, one array per surface.
    origin: list = field(init=False)
    # Positions and report of the last solve, the pairs inside their margin at
    # the start, the seconds spent and the names of the meshes written so far.
    final: list | None = field(init=False, default=None)
    rep: object = field(init=False, default=None)
    initial: int | None = field(init=False, default=None)
    spent: float = field(init=False, default=0.0)
    touched: set = field(init=False, default_factory=set)
    # Set when a solve was not written because a mesh had been edited meanwhile.
    refused: str = field(init=False, default="")

    def __post_init__(self):
        self.origin = [surface.vert.copy() for surface in self.scene.surfaces]

    def accept(self, positions, rep, moved: dict):
        """Take in a solve that was written."""
        self.final = [np.asarray(p) for p in positions]
        self.rep = rep
        if self.initial is None:
            self.initial = rep.initial
        self.spent += rep.seconds
        self.touched.update(moved)

    def totals(self):
        """``(report, {name: largest displacement})`` over all the solves.

        The report is the last solve's, with the pairs at the very start and the
        distances measured from where the cloth began.
        """
        if self.rep is None:
            # Nothing was written: the first solve was refused.
            return Report(message=self.refused), {}
        shifts, moved = [], {}
        for surface, start, end in zip(
            self.scene.surfaces, self.origin, self.final, strict=True
        ):
            shift = np.linalg.norm(end - start, axis=1)
            if surface.name in self.touched:
                moved[surface.name] = float(shift.max())
            free = np.broadcast_to(np.asarray(surface.movable, bool), shift.shape)
            shifts.append(shift[free])
        rep = replace(self.rep, initial=self.initial, seconds=self.spent)
        if self.refused:
            rep.converged, rep.message = False, self.refused
        if self.passes > 1:
            every = np.concatenate(shifts)
            rep.max_move = float(every.max()) if len(every) else 0.0
            rep.mean_move = float(every.mean()) if len(every) else 0.0
        return rep, moved


def _triangles(mesh):
    mesh.calc_loop_triangles()
    count = len(mesh.loop_triangles)
    if not count:
        return None
    tri = np.empty(count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri)
    return tri.reshape(-1, 3)


def _retriangulated(scene: Scene) -> bool:
    """Whether Blender now triangulates a written mesh differently from the input.

    A quad is split along the diagonal its current shape asks for, so moving its
    vertices can swap the diagonal. The triangles the solve kept clear are then
    not the ones Transfer ships, and two surfaces it separated can cross along
    the new one.
    """
    for surface in scene.surfaces:
        target = scene.targets.get(surface.name)
        if target is None:
            continue
        tri = _triangles(target.obj.data)
        if tri is None or not np.array_equal(tri, surface.tri):
            return True
    return False


def _pinned_vertices(scene) -> dict:
    """``{object name: set of vertex indices}`` the pins of the SHELL groups hold.

    A pin whose object or vertex group is gone holds nothing here: Transfer
    refuses it by name, and this is not where that is reported.
    """
    pinned: dict = {}
    for group in iterate_active_object_groups(scene):
        if group.object_type != SHELL_TYPE:
            continue
        for pin_item in group.pin_vertex_groups:
            try:
                obj = resolve_pin(pin_item)
                if obj is None or obj.type != "MESH":
                    continue
                _, vg_name = decode_vertex_group_identifier(pin_item.name)
                indices = pin_vertex_indices(obj, vg_name, pin_item.name, group.name)
            except ValueError:
                continue
            pinned.setdefault(obj.name_full, set()).update(indices)
    return pinned


def _same_pose(rest: np.ndarray, shipped: np.ndarray) -> bool:
    """Whether the pose Transfer ships is the stored mesh, to single precision.

    A rest-pose armature or an identity lattice evaluates a few units of
    resolution off the stored coordinates, which is not a deformation.
    """
    if rest.shape != shipped.shape:
        return False
    if not len(rest):
        return True
    extent = float(np.abs(rest).max())
    return float(np.abs(shipped - rest).max()) <= 64 * 2 * _HALF_ULP * extent


def _hold_reason(obj, mesh, rest, shipped, world, selected_only):
    """Why a SHELL object is held as a whole, or None when it may deform."""
    if selected_only and not obj.select_get():
        return HELD_UNSELECTED
    if obj.library is not None or getattr(mesh, "library", None) is not None:
        return HELD_LINKED
    if mesh.users - (1 if mesh.use_fake_user else 0) > 1:
        return HELD_SHARED
    if mesh.shape_keys is not None:
        return HELD_SHAPE_KEYS
    if not _same_pose(rest, shipped):
        return HELD_DEFORMED
    linear = world[:3, :3]
    if not np.all(np.isfinite(world)) or abs(np.linalg.det(linear)) <= (
        1e-12 * float(np.abs(linear).max()) ** 3
    ):
        return HELD_TRANSFORM
    return None


def collect_scene(context, *, selected_only: bool, clearance: float) -> Scene:
    """Read the shell pass's input in the pose Transfer would ship.

    ``clearance`` is the Clearance setting, in multiples of the contact gap of
    the smallest SHELL object that may deform. Two surfaces count as
    overlapping when they are closer than their contact offsets plus half of
    it, as in the rigid pass, and the run opens them to that distance. Raises
    ValueError when a pose cannot be shipped (`_shipped_local_verts`).
    """
    from ..core.encoder.mesh import _shipped_local_verts, evaluate_at_start_frame
    from ..core.encoder.params import object_contact_lengths

    blender_scene = context.scene
    state = get_addon_data(blender_scene).state
    pinned = _pinned_vertices(blender_scene)
    surfaces, targets, held, partial, skipped, gaps = [], {}, {}, {}, [], []
    colliders = 0
    seen = set()
    with evaluate_at_start_frame(context, state):
        for group in iterate_active_object_groups(blender_scene):
            is_shell = group.object_type == SHELL_TYPE
            if not (is_shell or group.object_type in COLLIDER_TYPES):
                continue
            for assigned in group.assigned_objects:
                if not assigned.included:
                    continue
                obj = resolve_assigned(assigned)
                if obj is None or obj.type != "MESH" or obj.data is None:
                    continue
                if obj.name_full in seen:
                    continue
                seen.add(obj.name_full)
                mesh = obj.data
                tri = _triangles(mesh)
                if tri is None:
                    if is_shell:
                        skipped.append(obj.name_full)
                    continue
                rest = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", rest)
                rest = rest.reshape(-1, 3)
                shipped = np.asarray(
                    _shipped_local_verts(obj, context, state, group, rest),
                    dtype=np.float64,
                )
                world = np.array(obj.matrix_world, dtype=np.float64)
                vert = shipped @ world[:3, :3].T + world[:3, 3]
                gap, offset = object_contact_lengths(group, assigned, 1.0)
                name = obj.name_full
                if not is_shell:
                    surfaces.append(Surface(name, vert, tri, False, offset))
                    colliders += 1
                    continue
                reason = _hold_reason(obj, mesh, rest, shipped, world, selected_only)
                free = np.ones(len(vert), dtype=bool)
                if reason is None and name in pinned:
                    free[sorted(i for i in pinned[name] if i < len(vert))] = False
                    if not free.any():
                        reason = HELD_PINNED
                    elif not free.all():
                        partial[name] = int((~free).sum())
                if reason is not None:
                    held[name] = reason
                    surfaces.append(Surface(name, vert, tri, False, offset))
                    continue
                surfaces.append(
                    Surface(name, vert, tri, free if not free.all() else True, offset)
                )
                targets[name] = Target(obj, np.linalg.inv(world), rest.copy(), world.copy())
                gaps.append(gap)

    # What `_changed_target` compares is the pose as the scene is now, which is
    # not the pose of the starting frame the run works in.
    view_layer = context.view_layer
    view_layer.update()
    for target in targets.values():
        target.matrix = np.array(target.obj.matrix_world, dtype=np.float64)

    movable = bool(targets)
    base = 0.5 * clearance * min(gaps) if gaps else 0.0
    if movable:
        extent = max(
            float(np.abs(s.vert).max()) for s in surfaces if s.name in targets
        )
        base = resolvable_clearance(base, extent)
    return Scene(
        surfaces, targets, held, partial, colliders, skipped, base, movable
    )


def write_back(scene: Scene, positions: list) -> dict:
    """Write the deformed SHELL vertices to their meshes.

    Returns ``{name: largest displacement}`` of each object that changed. Only a
    vertex that moved by more than single precision resolves is written, so a
    held or unmoved vertex keeps the very value it had.
    """
    moved = {}
    for surface, new in zip(scene.surfaces, positions, strict=True):
        target = scene.targets.get(surface.name)
        if target is None:
            continue
        shift = np.linalg.norm(new - surface.vert, axis=1)
        tolerance = _HALF_ULP * float(np.abs(surface.vert).max())
        changed = np.flatnonzero(shift > tolerance)
        if not len(changed):
            continue
        mesh = target.obj.data
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        homogeneous = np.concatenate([new[changed], np.ones((len(changed), 1))], axis=1)
        co[changed] = (homogeneous @ target.inverse.T)[:, :3]
        mesh.vertices.foreach_set("co", co.reshape(-1))
        mesh.update()
        moved[surface.name] = float(shift.max())
    return moved


def _changed_target(scene: Scene) -> str | None:
    """The name of a written mesh that was edited since it was read, or None.

    The interface is free while a run goes, so the artist can move or edit an
    object under it, or undo past it, and the result would be written over that.
    A mesh counts as edited when its stored coordinates or its world matrix are
    no longer the ones the run was given; an object that an undo freed counts
    too.
    """
    for name, target in scene.targets.items():
        try:
            mesh = target.obj.data
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            matrix = np.array(target.obj.matrix_world, dtype=np.float64)
        except (ReferenceError, RuntimeError):
            return name
        if co.shape != target.co.reshape(-1).shape or not np.array_equal(
            co, target.co.reshape(-1)
        ):
            return name
        if not np.array_equal(matrix, target.matrix):
            return name
    return None


def shell_entries(scene):
    """``(group, assigned, object)`` for every included mesh of a SHELL group."""
    for group in iterate_active_object_groups(scene):
        if group.object_type != SHELL_TYPE:
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is not None and obj.type == "MESH" and obj.data is not None:
                yield group, assigned, obj


def _shell_report(scene: Scene, rep, moved: dict, canceled: bool, passes: int = 1) -> dict:
    return {
        "objects": len(scene.targets),
        "colliders": scene.colliders,
        "passes": passes,
        "initial": rep.initial,
        "remaining": rep.remaining,
        "converged": rep.converged,
        "canceled": canceled,
        "moved": moved,
        "max_move": rep.max_move,
        "mean_move": rep.mean_move,
        "held": dict(scene.held),
        "pinned": dict(scene.pinned),
        "skipped": list(scene.skipped),
        "held_pairs": rep.held_pairs,
        "seconds": rep.seconds,
        "message": rep.message,
    }


class OBJECT_OT_ResolveAllIntersections(Operator):
    """Open every overlap in the scene so Transfer accepts it. SOLID and PDRD
    objects are pushed apart first, as rigid bodies, and then SHELL meshes are
    deformed until no surface is inside another's contact offsets or its own.
    The vertex coordinates of the cloth are edited; Ctrl+Z takes all of it
    back. Pinned vertices, and objects whose mesh cannot be edited, stay where
    they are and the rest moves around them"""

    bl_idname = "object.resolve_all_intersections"
    bl_label = "Resolve All Intersections"
    bl_options = {"UNDO"}

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
        return context.mode == "OBJECT" and get_busy() is None

    # -- Setup ---------------------------------------------------------------

    def _begin(self, context):
        """The rigid pass, then the shell scene it leaves.

        Returns ``(plan, status)``: ``plan`` is None when there is nothing more
        to run and ``status`` is then what the operator returns.
        """
        global _last_report
        rigid, stopped = None, None
        try:
            rigid, stopped = run_rigid(
                context, selected_only=self.selected_only,
                clearance=self.clearance, max_iterations=self.max_iterations,
            )
            scene = collect_scene(
                context, selected_only=self.selected_only,
                clearance=self.clearance,
            )
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            # A rigid pass that already ran has written transforms, which only
            # an undo step can take back.
            return None, {"FINISHED"} if rigid else {"CANCELLED"}
        if not scene.movable:
            if stopped is not None:
                self.report(
                    {"WARNING"},
                    iface_(
                        "Needs a SHELL object that may deform, or two or more "
                        "closed objects in SOLID or PDRD groups"
                    ),
                )
                return None, {"CANCELLED"}
            _last_report = {"rigid": rigid, "shell": None}
            self.report({"INFO"}, self._rigid_line(rigid))
            return None, {"FINISHED"}
        settings = Settings(clearance=scene.clearance)
        return Plan(rigid, scene, settings), None

    def _rigid_line(self, rigid):
        moved, left = len(rigid["moved"]), len(rigid["remaining_pairs"])
        if not rigid["initial_pairs"]:
            return iface_("No overlapping objects found")
        note = clearance_note(rigid)
        if left:
            line = iface_(
                "Moved {moved} object(s); {left} pair(s) still overlap "
                "after {passes} pass(es)"
            ).format(moved=moved, left=left, passes=rigid["iterations"])
        else:
            line = iface_(
                "Moved {moved} object(s) in {passes} pass(es); no overlap is "
                "left. Run Transfer again"
            ).format(moved=moved, passes=rigid["iterations"])
        return line + (f". {note}" if note else "")

    # -- Finish --------------------------------------------------------------

    def _write(self, context, plan: Plan, result, canceled: bool) -> bool:
        """Write one solve to the meshes. True when another solve follows.

        The next solve is asked for when this one cleared its scene but Blender
        now triangulates a written mesh differently (`_retriangulated`): the
        plan then holds the scene read again and its settings. A solve that did
        not clear its scene is not repeated, since the same time would be spent
        on the same pairs.
        """
        positions, rep = result
        edited = _changed_target(plan.scene)
        if edited is not None:
            plan.refused = iface_(
                "{name} was edited while the run was going, so the result was "
                "not written. Run again"
            ).format(name=edited)
            return False
        moved = write_back(plan.scene, positions)
        context.view_layer.update()
        plan.accept(positions, rep, moved)
        if (
            canceled
            or not rep.converged
            or rep.remaining
            or plan.passes >= _MAX_PASSES
            or not _retriangulated(plan.scene)
        ):
            return False
        scene = collect_scene(
            context, selected_only=self.selected_only, clearance=self.clearance
        )
        if [s.name for s in scene.surfaces] != [s.name for s in plan.scene.surfaces]:
            return False
        plan.scene, plan.settings = scene, Settings(clearance=scene.clearance)
        plan.passes += 1
        return True

    def _finish(self, context, plan: Plan, canceled: bool):
        global _last_report
        rep, moved = plan.totals()
        # A Transfer that was refused left its violations drawn over the scene
        # as it was uploaded; they mark where the surfaces were.
        if moved or (plan.rigid and plan.rigid["moved"]):
            com.dismiss_violations()
        shell = _shell_report(plan.scene, rep, moved, canceled, plan.passes)
        _last_report = {"rigid": plan.rigid, "shell": shell}

        if plan.refused:
            self.report({"WARNING"}, plan.refused)
        elif canceled:
            self.report(
                {"INFO"},
                iface_("Aborted: {count} shell object(s) changed").format(
                    count=len(moved)
                ),
            )
        elif not rep.converged or rep.remaining:
            self.report(
                {"WARNING"},
                iface_("Shells not resolved: {message}").format(
                    message=rep.message or iface_("pairs are still inside their margin")
                ),
            )
        elif not rep.initial:
            if plan.rigid and plan.rigid["initial_pairs"]:
                self.report({"INFO"}, self._rigid_line(plan.rigid))
            else:
                self.report({"INFO"}, iface_("No overlapping surfaces found"))
        else:
            self.report(
                {"INFO"},
                iface_(
                    "Opened {pairs} overlapping pair(s); {count} shell "
                    "object(s) moved by at most {distance}. Run Transfer again"
                ).format(
                    pairs=rep.initial,
                    count=len(moved),
                    distance=f"{max(moved.values(), default=0.0):.4g}",
                ),
            )
        return {"FINISHED"}

    # -- In line (scripts, the MCP tool) -------------------------------------

    def _run_in_line(self, context, plan: Plan):
        again = True
        while again:
            try:
                result = untangle(plan.scene.surfaces, plan.settings)
                again = self._write(context, plan, result, canceled=False)
            except ValueError as error:
                self.report({"ERROR"}, str(error))
                written = plan.rigid or plan.passes > 1 or plan.final is not None
                return {"FINISHED"} if written else {"CANCELLED"}
        return self._finish(context, plan, canceled=False)

    def execute(self, context):
        plan, status = self._begin(context)
        if plan is None:
            return status
        return self._run_in_line(context, plan)

    # -- In a child process (the button) ------------------------------------

    def invoke(self, context, event):
        window = context.window
        if window is None:
            return self.execute(context)
        push_undo_before("Resolve All Intersections (before)")
        plan, status = self._begin(context)
        if plan is None:
            return status
        try:
            self._run = Run(plan.scene.surfaces, plan.settings)
            self._run.start()
        except (OSError, RuntimeError) as error:
            self.report(
                {"WARNING"},
                iface_(
                    "Could not start a background process ({error}); "
                    "running in the foreground"
                ).format(error=error),
            )
            return self._run_in_line(context, plan)
        self._plan = plan
        self._started = time.monotonic()
        self._initial = 0
        self._place = None
        clear_abort()
        wm = context.window_manager
        self._timer = wm.event_timer_add(_POLL_SECONDS, window=window)
        wm.modal_handler_add(self)
        self._publish(context)
        return {"RUNNING_MODAL"}

    def _publish(self, context):
        """Show where the run is: the status bar, and the panel through `set_busy`."""
        run = self._run
        stage, pairs, place = run.stage, run.pairs, run.status
        key = (place.get("round"), place.get("group"))
        if key != self._place:
            # A scene solved a place at a time: each place starts its own count.
            self._place, self._initial = key, 0
        if stage == "untangle":
            # What the share of cleared pairs is measured against.
            self._initial = max(self._initial, pairs)
        set_busy({
            "stage": stage,
            "iteration": run.iteration,
            "pairs": pairs,
            "initial": self._initial,
            "started": self._started,
            "canceling": run.canceled,
            "place": place,
            "passes": self._plan.passes,
        })
        if run.canceled:
            text = iface_("Aborting shell intersection correction...")
        else:
            words = {
                "": iface_("setting up"),
                "untangle": iface_("untangling"),
                "recover": iface_("restoring the shape"),
            }[stage]
            text = iface_(
                "Resolving shell intersections: {stage}, iteration {iteration}, "
                "{pairs} pair(s) inside their margin"
            ).format(stage=words, iteration=run.iteration, pairs=pairs)
        context.workspace.status_text_set(text)
        _redraw(context)

    def _release(self, context):
        context.workspace.status_text_set(None)
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        # Both buttons are enabled again, whichever way the run ended.
        set_busy(None)
        clear_abort()
        _redraw(context)

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except BaseException:
            # The buttons are gated on the run, so no way out of it may leave
            # the run marked as going.
            self._run.kill()
            self._release(context)
            raise

    def _modal(self, context, event):
        if event.type == "TIMER":
            if abort_requested() and not self._run.canceled:
                self._run.cancel()
            if not self._run.finished:
                self._publish(context)
                return {"RUNNING_MODAL"}
            plan, canceled = self._plan, self._run.canceled
            try:
                again = self._write(context, plan, self._run.result(), canceled)
            except (ValueError, RuntimeError) as error:
                self._release(context)
                self.report({"ERROR"}, str(error))
                written = plan.rigid or plan.passes > 1 or plan.final is not None
                return {"FINISHED"} if written else {"CANCELLED"}
            if again:
                # Blender triangulated a written mesh differently: the same
                # button solves again on the new triangles.
                try:
                    self._run = Run(plan.scene.surfaces, plan.settings)
                    self._run.start()
                except (OSError, RuntimeError):
                    self._release(context)
                    return self._run_in_line(context, plan)
                self._place, self._initial = None, 0
                self._publish(context)
                return {"RUNNING_MODAL"}
            self._release(context)
            return self._finish(context, plan, canceled)
        # The interface stays usable: the result is written only over meshes
        # that were not edited meanwhile (`_changed_target`).
        return {"PASS_THROUGH"}

    def cancel(self, context):
        # Blender ended the modal from outside (the file was closed or loaded
        # over): nothing will read the result.
        self._run.kill()
        self._release(context)


def _redraw(context):
    """Ask the sidebar of every 3D viewport to redraw, and nothing else.

    The run reports four times a second, and redrawing the viewport that often
    would redraw the cloth with it, which a large garment makes slow.
    """
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "UI":
                    region.tag_redraw()


def _draw_progress(box, busy):
    """The live state of the run, below the button."""
    stage = busy["stage"]
    if busy["canceling"]:
        title = iface_("Aborting...")
    elif stage == "recover":
        title = iface_("Step 2 of 2: restoring the shape")
    elif stage == "untangle":
        title = iface_("Step 1 of 2: untangling")
    else:
        title = iface_("Setting up")
    box.label(text=title, icon="TIME")
    if busy.get("passes", 1) > 1:
        box.label(
            text=iface_(
                "Solve {number}: Blender triangulated the cloth differently after the move"
            ).format(number=busy["passes"])
        )
    place = busy.get("place") or {}
    if place.get("groups"):
        box.label(
            text=iface_("Round {round}, place {group} of {groups}").format(
                round=place["round"], group=place["group"], groups=place["groups"]
            )
        )
    # The share of the pairs that were inside their margin and are not now. It is
    # whole once the second step begins: that step has no length known in advance.
    if stage == "recover":
        share = 1.0
    elif busy["initial"]:
        share = max(0.0, min(1.0, 1.0 - busy["pairs"] / busy["initial"]))
    else:
        share = 0.0
    box.progress(
        factor=share,
        # Rounded down, so that 100% means no pair is left.
        text=iface_("Pairs cleared: {percent}%").format(percent=math.floor(100.0 * share + 1e-9)),
        translate=False,
        type="BAR",
    )
    if stage == "recover":
        box.label(text=iface_("Iteration {iteration}").format(iteration=busy["iteration"]))
    elif stage == "untangle":
        box.label(
            text=iface_("Iteration {iteration}, {pairs} pair(s) inside their margin").format(
                iteration=busy["iteration"], pairs=busy["pairs"]
            )
        )
    box.label(
        text=iface_("{seconds} s elapsed").format(
            seconds=int(time.monotonic() - busy["started"])
        )
    )
    row = box.row()
    row.enabled = not busy["canceling"]
    row.operator("object.resolve_all_intersections_abort", text="Abort", icon="CANCEL")


def draw_shell_intersection_correction(layout, context):
    """Draw the Resolve All Intersections entry of the Mesh Repair panel.

    The button is drawn unconditionally and grayed out when its precondition
    fails, with a status label saying why. Nothing here reads geometry: the
    counts are of group entries, and the outcome shown is the last run's.
    """
    scene = context.scene
    state = get_addon_data(scene).state
    box = layout.box()
    box.label(text="Resolve All Intersections")
    box.label(text="Push apart SOLID and PDRD objects, then deform SHELL meshes.")
    box.label(text="The cloth's vertex coordinates change. Ctrl+Z undoes it.")

    col = box.column(align=True)
    col.prop(state, "intersection_fix_iterations")
    col.prop(state, "intersection_fix_clearance")
    box.prop(state, "intersection_fix_selected_only")

    shells = sum(1 for _entry in shell_entries(scene))
    volumetric = sum(1 for _entry in volumetric_entries(scene))
    busy = get_busy()
    col = box.column()
    col.enabled = context.mode == "OBJECT" and (shells >= 1 or volumetric >= 2) and busy is None
    op = col.operator(
        "object.resolve_all_intersections",
        text="Resolve All Intersections",
        icon="MOD_CLOTH",
    )
    op.max_iterations = state.intersection_fix_iterations
    op.clearance = state.intersection_fix_clearance
    op.selected_only = state.intersection_fix_selected_only

    if busy is not None:
        _draw_progress(box, busy)
    elif context.mode != "OBJECT":
        box.label(text="Switch to Object Mode", icon="INFO")
    elif shells == 0 and volumetric < 2:
        box.label(
            text="Needs a SHELL object, or two or more SOLID or PDRD objects",
            icon="INFO",
        )
    else:
        box.label(
            text=iface_("{shells} SHELL and {solids} SOLID or PDRD object(s)").format(
                shells=shells, solids=volumetric
            ),
            icon="CHECKMARK",
        )

    report = get_last_report()
    if report is None:
        return
    result = box.box()
    rigid, shell = report["rigid"], report["shell"]
    if rigid is not None and rigid["initial_pairs"]:
        result.label(
            text=iface_("Rigid: {count} object(s) moved, {left} pair(s) still overlap").format(
                count=len(rigid["moved"]), left=len(rigid["remaining_pairs"])
            ),
            icon="ERROR" if rigid["remaining_pairs"] else "CHECKMARK",
        )
        note = clearance_note(rigid)
        if note:
            result.label(text=note, icon="INFO")
    if shell is None:
        return
    if shell["canceled"]:
        result.label(text="Last run: aborted", icon="INFO")
    elif not shell["initial"]:
        result.label(text="Last run: no overlapping surfaces found", icon="CHECKMARK")
    elif shell["remaining"] or not shell["converged"]:
        result.label(
            text=iface_("Last run: {left} of {total} pair(s) still inside their margin").format(
                left=shell["remaining"], total=shell["initial"]
            ),
            icon="ERROR",
        )
    else:
        result.label(
            text=iface_("Last run: {total} overlapping pair(s) opened in {seconds:.1f} s").format(
                total=shell["initial"], seconds=shell["seconds"]
            ),
            icon="CHECKMARK",
        )
    if shell["moved"]:
        result.label(
            text=iface_("{count} shell object(s) moved, by at most {distance}").format(
                count=len(shell["moved"]),
                distance=f"{max(shell['moved'].values()):.4g}",
            )
        )
    if shell["held"]:
        result.label(
            text=iface_("{count} shell object(s) held in place").format(
                count=len(shell["held"])
            )
        )
    if shell["held_pairs"]:
        result.label(
            text=iface_("{count} pair(s) of held triangles overlap and cannot be moved").format(
                count=shell["held_pairs"]
            ),
            icon="ERROR",
        )


class OBJECT_OT_ResolveAllIntersectionsAbort(Operator):
    """Stop Resolve All Intersections. In the first step the shells stay as they
    were; in the second step the surfaces keep the state reached so far"""

    bl_idname = "object.resolve_all_intersections_abort"
    bl_label = "Abort"
    # The run in progress is not an undo step of its own, and neither is this.
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        busy = get_busy()
        return busy is not None and not busy["canceling"]

    def execute(self, context):
        request_abort()
        busy = get_busy()
        if busy is not None:
            # The button grays out at once; the run is told at its next tick.
            set_busy({**busy, "canceling": True})
            _redraw(context)
        return {"FINISHED"}


classes = (OBJECT_OT_ResolveAllIntersections, OBJECT_OT_ResolveAllIntersectionsAbort)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    clear_last_report()
