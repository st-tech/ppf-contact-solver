# File: cleaning_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Mesh Cleaning: find the geometry defects that make a Transfer fail or make
# the solver abort, and repair the ones that can be repaired safely.
#
# Two kinds of check live here, and the distinction matters:
#
#   * Mirrors of Transfer-time validators. The encoder already rejects
#     isolated vertices, hanging seam vertices, duplicate faces and linked
#     duplicates (encoder.mesh._build_obj_data), but only once the artist
#     presses Transfer. Reporting them here calls the SAME helpers, so a
#     clean scan and a passing Transfer cannot disagree.
#   * Checks nothing else performs: near-coincident vertices, degenerate
#     faces, an open or non-manifold surface, inconsistent winding, and faces
#     Blender may re-split. Each breaks a documented part of the pipeline;
#     see the per-detector docstrings for the failure each one produces.
#
# Scanning never writes to a mesh. Every repair is a separate user-invoked
# operator, so mesh writes only happen from an explicit button press, and the
# repairs that change the vertex count additionally require a confirmation
# dialog because that count is what pins and caches are keyed on.
#
# Tool parameters live on the OPERATORS, not on the saved State, following
# the destructive-tool precedent in ui/dynamics/sand_ops.py: a threshold is
# transient tool configuration rather than authored scene data, and keeping
# it off the PropertyGroup avoids a permanent saved-RNA commitment.

from __future__ import annotations

import math

import bmesh  # pyright: ignore
import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_
from bpy.types import Operator  # pyright: ignore
from mathutils.kdtree import KDTree  # pyright: ignore

from ..core.utils import count_duplicate_faces, find_linked_duplicate_siblings


#: Default weld / degeneracy distance in LOCAL mesh units. Matches Blender's
#: own Merge by Distance default, so the tool agrees with what the artist
#: would get from Mesh > Merge > By Distance in Edit Mode.
DEFAULT_MERGE_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# Scan report cache
# ---------------------------------------------------------------------------
#
# Keyed by object name. Deliberately module-level and NOT a PropertyGroup: a
# scan is transient diagnostic state, not authored data that belongs in the
# .blend. It also keeps the panel draw free of mesh traversal, since the draw
# only formats an already-computed report.
_scan_cache: dict[str, dict] = {}


def get_scan_report(obj_name: str) -> dict | None:
    """Return the cached scan report for *obj_name*, or None if none is valid.

    A report is discarded when the mesh's vertex or polygon count no longer
    matches the scanned counts, which covers every repair in this module since
    each one changes at least one of the two.

    It does NOT detect a count-preserving edit: moving a vertex in Edit Mode
    leaves both counts intact, so a report can survive an edit that changes
    the answer (a hand-welded pair, or a quad folded past the flip boundary).
    Re-scan after editing. Keying on the counts is deliberate: the alternative
    is hashing the coordinates, which is O(verts) work on a path the panel
    draw calls once per selected object per redraw.
    """
    report = _scan_cache.get(obj_name)
    if report is None:
        return None
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH" or obj.data is None:
        _scan_cache.pop(obj_name, None)
        return None
    if (
        len(obj.data.vertices) != report["n_verts"]
        or len(obj.data.polygons) != report["n_polys"]
    ):
        _scan_cache.pop(obj_name, None)
        return None
    return report


def clear_scan_report(obj_name: str | None = None) -> None:
    """Drop one object's cached report, or every report when *obj_name* is None."""
    if obj_name is None:
        _scan_cache.clear()
    else:
        _scan_cache.pop(obj_name, None)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
#
# Every detector returns a dict carrying at least ``count``; ``count == 0``
# means the mesh is clean for that defect. Extra keys are formatted by the
# panel. None of them mutates the mesh.


def find_near_duplicate_vertices(mesh, threshold: float) -> dict:
    """Vertex pairs closer together than *threshold*, in local mesh units.

    Nothing else in the pipeline detects these, and they have the worst
    failure mode of anything in this module. A surviving pair sits far inside
    the contact gap, and the cubic barrier's dynamic stiffness carries a
    ``mass / gap^2`` term, so a separation of a few nanometers contributes
    Hessian entries many orders of magnitude larger than the rest of the row.
    In fp32 the assembled Newton matrix then loses rank and the solver stops
    on its SPD guard with ``p^T A p <= 0`` at the first PCG iteration, naming
    no geometry, which makes the cause very hard to find from the log alone.

    fTetWild masks the problem because its envelope remesh welds such pairs
    away, so a mesh can pass for a long time and fail only once the artist
    switches the tetrahedralizer to TetGen, whose surface-preserving mode
    carries the input through unchanged.

    Both distances are reported: the threshold is a local-space quantity
    (matching Blender's Merge by Distance and the ``bmesh`` operator that
    repairs it), while the magnitude that matters to the solver is world
    space, which for a scaled object is a very different number.

    Returns ``count`` (pairs), ``min_dist`` (local units), ``preview`` (up to
    8 index pairs), and ``verts`` (every vertex index involved).
    """
    n = len(mesh.vertices)
    if n < 2 or threshold <= 0.0:
        return {"count": 0}

    kd = KDTree(n)
    for i, v in enumerate(mesh.vertices):
        kd.insert(v.co, i)
    kd.balance()

    pairs: list[tuple[int, int]] = []
    involved: set[int] = set()
    min_dist = math.inf
    for i, v in enumerate(mesh.vertices):
        for _co, j, dist in kd.find_range(v.co, threshold):
            # find_range returns the query vertex itself and both orderings
            # of every pair; keep one direction only.
            if j <= i:
                continue
            pairs.append((i, j))
            involved.update((i, j))
            if dist < min_dist:
                min_dist = dist
    if not pairs:
        return {"count": 0}
    return {
        "count": len(pairs),
        "min_dist": min_dist,
        "preview": pairs[:8],
        "verts": sorted(involved),
    }


def find_degenerate_faces(mesh, area_eps: float) -> dict:
    """Faces whose area is at or below *area_eps*, in local units squared.

    A zero-area triangle has no defined normal, so both the contact normal
    and the bending hinge built on it are undefined. Unlike a duplicate face
    this is not rejected at Transfer, so it reaches the solver.
    """
    count = 0
    preview: list[int] = []
    for poly in mesh.polygons:
        if poly.area <= area_eps:
            count += 1
            if len(preview) < 8:
                preview.append(poly.index)
    if count == 0:
        return {"count": 0}
    return {"count": count, "preview": preview}


def _edge_direction_in_face(edge, face) -> bool:
    """True when *face* traverses *edge* from ``verts[0]`` toward ``verts[1]``."""
    a, b = edge.verts
    for loop in face.loops:
        if loop.vert is a:
            return loop.link_loop_next.vert is b
    # An edge always belongs to its linked faces, so this is unreachable for
    # a well-formed BMesh; return the value that reads as "no disagreement".
    return True


def find_surface_defects(mesh) -> dict:
    """Boundary edges, non-manifold edges, and inconsistently wound faces.

    Tetrahedralization needs a closed, manifold, consistently wound surface
    to tell inside from outside: TetGen refuses such input outright (the
    frontend re-raises it as needing a clean, closed, manifold mesh) and
    fTetWild accepts it but resamples it into something the artist did not
    author. A SHELL is legitimately open, so the panel reports these as
    information rather than errors, and offers no blanket repair. Closing a
    boundary is a modeling decision, not a cleanup, and this module will not
    guess at one.

    Winding is the one part that IS repairable, via Recalculate Outside. It
    is detected by walking each manifold edge: two consistently wound
    neighbors traverse their shared edge in opposite directions.
    """
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        boundary = 0
        non_manifold = 0
        bad_winding = 0
        for edge in bm.edges:
            n_faces = len(edge.link_faces)
            if n_faces == 1:
                boundary += 1
            elif n_faces > 2:
                non_manifold += 1
            elif n_faces == 2:
                fa, fb = edge.link_faces
                if _edge_direction_in_face(edge, fa) == _edge_direction_in_face(
                    edge, fb
                ):
                    bad_winding += 1
    finally:
        bm.free()

    return {
        "count": boundary + non_manifold + bad_winding,
        "boundary": boundary,
        "non_manifold": non_manifold,
        "bad_winding": bad_winding,
    }


def _quad_fold_angle(mesh, poly) -> float | None:
    """Angle between the two triangles of *poly*'s default 0-2 split, degrees.

    This is the quantity Blender's flip test thresholds at 90 degrees. For a
    face with more than 4 corners it measures the first four corners, which
    is indicative rather than exact, because n-gons are ear-clipped rather
    than split on a single diagonal.
    """
    vids = list(poly.vertices)
    if len(vids) < 4:
        return None
    p0 = mesh.vertices[vids[0]].co
    p1 = mesh.vertices[vids[1]].co
    p2 = mesh.vertices[vids[2]].co
    p3 = mesh.vertices[vids[3]].co
    n_a = (p1 - p0).cross(p2 - p0)
    n_b = (p2 - p0).cross(p3 - p0)
    if n_a.length == 0.0 or n_b.length == 0.0:
        return None
    return math.degrees(n_a.angle(n_b))


def find_resplittable_faces(mesh) -> dict:
    """Faces with more than 3 corners, whose displayed triangulation can drift.

    Blender picks a quad's split diagonal from the CURRENT vertex positions
    every time it tessellates: it builds the (0,1,2) + (0,2,3) split, then
    flips to (0,1,3) + (1,2,3) when the two candidate triangles' normals have
    separated by more than 90 degrees. That decision has no hysteresis and no
    reference to the rest pose.

    The solver, by contrast, ships ONE triangulation captured from the rest
    topology at Transfer time and keeps it for the whole simulation. So as a
    quad deforms, the displayed surface can be split along a different
    diagonal than the simulated surface, and the two then disagree by the
    fold depth of that quad. In the viewport this reads as a thin
    penetration, one the solver's own intersection test cannot see because it
    is not present in the simulated state at all. Triangulating up front
    removes the possibility: an explicit triangle has no diagonal left to
    re-pick.

    ``max_fold_deg`` is the largest current normal separation across the 0-2
    diagonal, and ``past_flip`` counts faces already beyond 90 degrees.
    """
    count = 0
    past_flip = 0
    max_fold = 0.0
    for poly in mesh.polygons:
        if poly.loop_total <= 3:
            continue
        count += 1
        fold = _quad_fold_angle(mesh, poly)
        if fold is None:
            continue
        if fold > max_fold:
            max_fold = fold
        if fold > 90.0:
            past_flip += 1
    if count == 0:
        return {"count": 0}
    return {"count": count, "max_fold_deg": max_fold, "past_flip": past_flip}


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def pinned_vertices_for(obj) -> set:
    """Vertices of *obj* held by a pin in any active group it belongs to.

    Pinned vertices are exempt from the isolated / hanging-seam checks for
    the same reason the encoder exempts them: a pin is a Dirichlet condition,
    so a face-less pinned vertex (a sewn curtain hook) is valid rather than
    stray. Reusing the encoder's own resolver keeps this scan's verdict and
    the Transfer's verdict identical.
    """
    from ..core.encoder.mesh import _group_pinned_vertex_indices
    from ..core.uuid_registry import resolve_assigned
    from ..models.groups import iterate_object_groups

    pinned: set = set()
    for group in iterate_object_groups(bpy.context.scene):
        if not group.active:
            continue
        for assigned in group.assigned_objects:
            resolved = resolve_assigned(assigned)
            if resolved is None or resolved.name != obj.name:
                continue
            try:
                pinned.update(_group_pinned_vertex_indices(group, obj))
            except Exception:
                # A malformed pin must not take down the whole scan; the
                # Transfer path still reports it explicitly.
                continue
    return pinned


def is_particle_mesh(obj) -> bool:
    """True when *obj* is a committed SAND particle mesh.

    A particle mesh is a faceless cloud of loose vertices (the grain
    centers), so EVERY vertex is legitimately in no face. Running the
    isolated / hanging-vertex checks on one would report the whole cloud as
    broken, which is why the encoder skips its own face-based validators for
    SAND. Same predicate as the SAND assignment guard in
    ui/dynamics/group_ops.py, so the two cannot disagree about what counts as
    a particle mesh.
    """
    return bool(
        obj.type == "MESH"
        and obj.data is not None
        and obj.get("ppf_particle_mesh")
        and len(obj.data.polygons) == 0
        and len(obj.data.edges) == 0
    )


def scan_object(obj, *, merge_threshold: float, area_eps: float) -> dict:
    """Build the defect report for one mesh object. Never mutates the mesh."""
    from ..core.encoder.mesh import (
        detect_hanging_stitch_vertices,
        detect_isolated_vertices,
    )

    mesh = obj.data
    particles = is_particle_mesh(obj)

    # A particle mesh has no faces by construction, so the face-based checks
    # do not apply to it; only the near-coincident scan is meaningful, and it
    # is the one that matters most for grains (two grain centers welded
    # together are exactly the pathological contact pair).
    #
    # Resolve the pin exemption LAZILY. pinned_vertices_for walks every pin
    # group through get_vertices_in_group, which is an O(verts x groups)
    # Python loop over RNA: measured at 336 ms on a 6.4k-vertex character
    # carrying 70 vertex groups, which is most of a scan on a rigged body.
    # A pin can only ever REMOVE entries from these two lists, so when both
    # come back empty the pinned set cannot change the answer and does not
    # need to be resolved at all. That is the common case.
    raw_isolated = [] if particles else detect_isolated_vertices(mesh)
    raw_hanging = [] if particles else detect_hanging_stitch_vertices(mesh)
    if raw_isolated or raw_hanging:
        pinned = pinned_vertices_for(obj)
        isolated = [i for i in raw_isolated if i not in pinned]
        hanging = [i for i in raw_hanging if i not in pinned]
    else:
        isolated = []
        hanging = []
    siblings = find_linked_duplicate_siblings(obj)

    defects = {
        "near_duplicates": find_near_duplicate_vertices(mesh, merge_threshold),
        "isolated_verts": {
            "count": len(isolated),
            "preview": isolated[:8],
            "verts": isolated,
        },
        "hanging_verts": {
            "count": len(hanging),
            "preview": hanging[:8],
            "verts": hanging,
        },
        "degenerate_faces": find_degenerate_faces(mesh, area_eps),
        "duplicate_faces": {"count": count_duplicate_faces(obj)},
        "surface": (
            {"count": 0, "boundary": 0, "non_manifold": 0, "bad_winding": 0}
            if particles
            else find_surface_defects(mesh)
        ),
        "resplittable": find_resplittable_faces(mesh),
        "linked_duplicate": {"count": len(siblings), "siblings": siblings},
    }

    # World-space scale of the smallest near-coincident gap, for context: the
    # threshold is local but the solver works in world units.
    near = defects["near_duplicates"]
    if near["count"]:
        scale = obj.matrix_world.to_scale()
        near["min_dist_world"] = near["min_dist"] * max(abs(s) for s in scale)

    # Severity split. Getting this wrong would make the tool cry wolf: an
    # ordinary cloth panel is an OPEN surface made of QUADS, so counting
    # boundary edges and quads as problems would flag every SHELL ever
    # authored and bury the defects that actually stop a run.
    n_errors = (
        defects["near_duplicates"]["count"]
        + defects["isolated_verts"]["count"]
        + defects["hanging_verts"]["count"]
        + defects["duplicate_faces"]["count"]
        + defects["degenerate_faces"]["count"]
        + defects["linked_duplicate"]["count"]
        + defects["surface"]["bad_winding"]
    )
    n_notes = (
        defects["surface"]["boundary"]
        + defects["surface"]["non_manifold"]
        + defects["resplittable"]["count"]
    )

    return {
        "object": obj.name,
        "n_verts": len(mesh.vertices),
        "n_polys": len(mesh.polygons),
        "merge_threshold": merge_threshold,
        "area_eps": area_eps,
        "defects": defects,
        "n_errors": n_errors,
        "n_notes": n_notes,
        "total": sum(d["count"] for d in defects.values()),
        # Resolved HERE rather than at draw time: it walks every group's pins
        # and lazy-loads the capture cache from disk, which would be a file
        # open on every mouse move if the panel computed it.
        "dependents": vertex_count_dependents(obj),
    }


# ---------------------------------------------------------------------------
# What a vertex-count change invalidates
# ---------------------------------------------------------------------------


def vertex_count_dependents(obj) -> list[str]:
    """Names of what a vertex-count change on *obj* affects.

    Two different kinds of consequence, and conflating them would mislead:

    * The caches are INVALIDATED. A captured STATIC deformation and a written
      PC2 display cache are both sized by vertex count, so afterward the
      encoder rejects the capture and the PC2 replays the wrong points. These
      are derived data and ``clear_stale_vertex_caches`` deletes them.
    * The pin groups stay VALID but their membership can SHIFT. Blender remaps
      its own vertex groups through a merge, so no pin is stranded, but when a
      pinned vertex merges with an unpinned one the survivor's membership
      follows Blender's rules rather than the artist's original pick. Nothing
      here deletes a pin: that is authored work, and it is not broken.

    The panel names these before such a repair runs, rather than letting the
    mismatch surface at the next Transfer.
    """
    from ..core.pc2 import get_static_deform_cache
    from ..core.uuid_registry import resolve_assigned
    from ..models.groups import iterate_object_groups

    dependents: list[str] = []

    if any(m.type == "MESH_CACHE" for m in obj.modifiers):
        dependents.append(iface_("Display cache (PC2)"))

    try:
        if get_static_deform_cache(obj) is not None:
            dependents.append(iface_("Capture Deformation cache"))
    except Exception:
        # A missing / unreadable cache file is not this tool's problem to
        # report; treat it as absent.
        pass

    n_pins = 0
    for group in iterate_object_groups(bpy.context.scene):
        if not group.active:
            continue
        in_group = False
        for assigned in group.assigned_objects:
            resolved = resolve_assigned(assigned)
            if resolved is not None and resolved.name == obj.name:
                in_group = True
                break
        if in_group:
            n_pins += len(group.pin_vertex_groups)
    if n_pins:
        dependents.append(
            iface_("{count} pin group(s): membership may shift").format(
                count=n_pins
            )
        )

    return dependents


# ---------------------------------------------------------------------------
# Shared operator plumbing
# ---------------------------------------------------------------------------


def can_clean(context) -> bool:
    """Precondition shared by the panel's grayed-out state and every poll().

    Factored into one predicate so the disabled column and the operator poll
    can never disagree (the pattern used by sand_ops._is_convertible_solid_mesh).
    """
    return context.mode == "OBJECT" and any(
        o.type == "MESH" and o.data for o in context.selected_objects
    )


def selected_meshes(context) -> list:
    return [o for o in context.selected_objects if o.type == "MESH" and o.data]


def _unique_mesh_targets(objects) -> list:
    """Objects with distinct mesh datablocks, so a shared mesh is edited once.

    Objects sharing one datablock (linked duplicates) would otherwise be
    repaired once per user, compounding the edit.
    """
    seen: set[str] = set()
    targets = []
    for obj in objects:
        if obj.data.name in seen:
            continue
        seen.add(obj.data.name)
        targets.append(obj)
    return targets


def _edit_bmesh(obj):
    """Context manager yielding a BMesh bound to *obj*'s mesh datablock."""

    class _Ctx:
        def __enter__(self):
            self.bm = bmesh.new()
            self.bm.from_mesh(obj.data)
            return self.bm

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is None:
                    self.bm.to_mesh(obj.data)
                    obj.data.update()
            finally:
                self.bm.free()
            return False

    return _Ctx()


class _RepairBase(Operator):
    """Shared execute() shape for the repair operators.

    Subclasses implement ``repair(obj) -> int`` (how many elements were
    affected) and set ``report_verb``. A subclass that changes the vertex
    count sets ``changes_vertex_count = True``, which adds a confirmation
    dialog listing what the change invalidates and requires the artist to
    tick an acknowledgement before anything is written.
    """

    bl_options = {"REGISTER", "UNDO"}

    changes_vertex_count = False
    #: Past-tense verb for the completion report, e.g. "Merged 4 on Cube".
    report_verb = "Cleaned"

    @classmethod
    def poll(cls, context):
        return can_clean(context)

    def repair(self, obj) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- confirmation for count-changing repairs ---------------------------

    def _dependents(self, context) -> list[str]:
        names: list[str] = []
        for obj in _unique_mesh_targets(selected_meshes(context)):
            for dep in vertex_count_dependents(obj):
                entry = f"{obj.name}: {dep}"
                if entry not in names:
                    names.append(entry)
        return names

    def invoke(self, context, event):
        if not self.changes_vertex_count:
            return self.execute(context)
        self._pending = self._dependents(context)
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text="This repair changes the vertex count.", icon="ERROR")
        pending = getattr(self, "_pending", [])
        if pending:
            box = layout.box()
            box.alert = True
            box.label(text="Applying it invalidates:")
            for entry in pending:
                box.label(text=entry, icon="DOT")
            box.label(text="Re-run Transfer afterward. Pins are kept.")
        else:
            layout.label(
                text="No capture cache, display cache, or pin is affected.",
                icon="INFO",
            )
        for prop in self._dialog_props():
            layout.prop(self, prop)
        if pending:
            layout.prop(self, "clear_stale_caches")
        layout.prop(self, "acknowledge")

    def _dialog_props(self) -> tuple:
        """Extra operator properties to expose in the confirmation dialog."""
        return ()

    # -- execution ---------------------------------------------------------

    def execute(self, context):
        # bmesh from_mesh / to_mesh read and write the mesh datablock, which
        # is stale while the object is in Edit Mode; leave it first.
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass

        targets = _unique_mesh_targets(selected_meshes(context))
        if not targets:
            self.report({"WARNING"}, iface_("Select one or more mesh objects"))
            return {"CANCELLED"}

        if self.changes_vertex_count and not self.acknowledge:
            self.report(
                {"ERROR"},
                iface_(
                    "This repair changes the vertex count. Confirm the "
                    "acknowledgement to apply it."
                ),
            )
            return {"CANCELLED"}

        total = 0
        parts = []
        cleared: list[str] = []
        for obj in targets:
            before_verts = len(obj.data.vertices)
            affected = self.repair(obj)
            total += affected
            if affected:
                parts.append(f"{obj.name} ({affected})")
            # Only clear when this object's count actually moved: a repair can
            # legitimately affect nothing on one of several selected objects,
            # and that object's caches are still valid.
            if self.changes_vertex_count and len(obj.data.vertices) != before_verts:
                # Always re-stamp the pin hashes, regardless of the cache
                # opt-out: a mismatched hash is a per-redraw cost, not user
                # data, so there is nothing to opt out of.
                refresh_pin_vg_hashes(obj)
                if getattr(self, "clear_stale_caches", False):
                    for name in clear_stale_vertex_caches(obj):
                        entry = f"{obj.name}: {name}"
                        if entry not in cleared:
                            cleared.append(entry)
            clear_scan_report(obj.name)

        if total == 0:
            self.report(
                {"INFO"}, iface_("Nothing to repair on the selected meshes.")
            )
            return {"CANCELLED"}

        # A repaired mesh may have been what failed the last Transfer; clear
        # the stale error so the panel stops showing it.
        from ..core.client import communicator as com
        from ..core.utils import redraw_all_areas

        com.set_error("")
        redraw_all_areas(context)

        self.report(
            {"INFO"},
            iface_("{verb} {count} on {objects}. Re-scan to confirm.").format(
                verb=iface_(self.report_verb),
                count=total,
                objects=", ".join(parts),
            ),
        )
        if cleared:
            self.report(
                {"WARNING"},
                iface_("Cleared invalidated: {items}. Re-run Transfer.").format(
                    items=", ".join(cleared)
                ),
            )
        return {"FINISHED"}


def refresh_pin_vg_hashes(obj) -> int:
    """Re-stamp the stored content hash of every pin vertex group on *obj*.

    Each pin item saves a hash of its vertex group's index list so a later
    RENAME can be followed by content. Welding or deleting vertices rewrites
    that index list, so the stored hash stops matching even though the group is
    untouched and correctly named.

    Leaving that mismatch behind is not cosmetic. ``resolve_pin`` asks
    ``resolve_vg_name`` to reconcile the hash on every draw of the Groups
    panel, and a hash that nothing on the object can satisfy is the worst input
    for it: see ``resolve_vg_name`` for why an unsatisfiable hash is kept off
    the scanning path. Since the hash is saved in the .blend, a mismatch left
    here outlives the session.

    Re-stamping is only correct because the repair itself is the authorized
    change: the artist asked for the weld, so the group's new contents ARE the
    intended contents. Called from an operator's ``execute``, never from a draw
    or handler, since it writes to saved state.

    Returns how many pin items were re-stamped.
    """
    from ..core.uuid_registry import compute_vg_hash, resolve_assigned
    from ..models.groups import (
        decode_vertex_group_identifier,
        iterate_object_groups,
    )

    n = 0
    for group in iterate_object_groups(bpy.context.scene):
        if not group.active:
            continue
        in_group = False
        for assigned in group.assigned_objects:
            resolved = resolve_assigned(assigned)
            if resolved is not None and resolved.name == obj.name:
                in_group = True
                break
        if not in_group:
            continue
        for pin_item in group.pin_vertex_groups:
            obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
            if obj_name != obj.name or not vg_name:
                continue
            fresh = str(compute_vg_hash(obj, vg_name))
            if pin_item.vg_hash != fresh:
                pin_item.vg_hash = fresh
                n += 1
    return n


def clear_stale_vertex_caches(obj) -> list[str]:
    """Delete the per-vertex caches that a vertex-count change invalidates.

    Both are DERIVED data sized by vertex count, so after a count change they
    no longer describe the mesh: a PC2 display cache replays the wrong points,
    and the encoder rejects a captured STATIC deformation whose width no longer
    matches. Blender maintains its own vertex groups across such an edit, so
    pins are NOT touched here; deleting authored pin groups would destroy work
    the artist cannot get back, and they remain valid.

    Leaving mismatched caches in place is not harmless: the viewport overlay's
    pin builder reads them, and a builder that raises is retried whenever the
    scene state changes (see ``_rebuild_cached`` in ``ui/dynamics/overlay.py``),
    so a mismatch that persists in the saved file keeps costing frame time.

    Returns the human-readable names of what was cleared.
    """
    from ..core.pc2 import (
        cleanup_mesh_cache,
        get_static_deform_cache,
        remove_static_deform_pc2,
    )

    cleared: list[str] = []

    if any(m.type == "MESH_CACHE" for m in obj.modifiers):
        cleanup_mesh_cache(obj)
        cleared.append(iface_("Display cache (PC2)"))

    try:
        had_capture = get_static_deform_cache(obj) is not None
    except Exception:
        had_capture = False
    if had_capture:
        remove_static_deform_pc2(obj)
        cleared.append(iface_("Capture Deformation cache"))

    return cleared


class _CountChangingRepair(_RepairBase):
    """A repair that changes the vertex count, so it must be acknowledged."""

    changes_vertex_count = True

    acknowledge: bpy.props.BoolProperty(
        name="I understand, apply anyway",
        description=(
            "Confirm that the listed caches may need to be re-created after "
            "this repair changes the vertex count"
        ),
        default=False,
        options={"SKIP_SAVE"},
    )  # pyright: ignore

    clear_stale_caches: bpy.props.BoolProperty(
        name="Clear invalidated caches",
        description=(
            "Delete this object's display cache and captured deformation, "
            "which are sized for the old vertex count and describe the wrong "
            "mesh afterward. Leaving them makes the viewport overlay read "
            "stale data. Regenerate with Transfer, and with Capture "
            "Deformation for a moving collider"
        ),
        default=True,
        options={"SKIP_SAVE"},
    )  # pyright: ignore


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class MESH_OT_PPFScanMeshes(Operator):
    """Scan the selected mesh objects for geometry the solver rejects or
    chokes on, and report what was found. Reads the meshes only, nothing is
    modified. Adjust the thresholds with Adjust Last Operation (F9)"""

    bl_idname = "object.ppf_scan_mesh_defects"
    bl_label = "Scan Selected Meshes"
    bl_options = {"REGISTER"}

    merge_threshold: bpy.props.FloatProperty(
        name="Merge Distance",
        description=(
            "Vertices closer together than this (in local mesh units) are "
            "reported as near-coincident. Matches Blender's Merge by "
            "Distance default"
        ),
        default=DEFAULT_MERGE_THRESHOLD,
        min=0.0,
        soft_max=0.1,
        precision=6,
    )  # pyright: ignore

    area_eps: bpy.props.FloatProperty(
        name="Degenerate Area",
        description=(
            "Faces with an area at or below this (in local units squared) "
            "are reported as degenerate. Zero reports only exactly "
            "zero-area faces"
        ),
        default=0.0,
        min=0.0,
        soft_max=1e-6,
        precision=9,
    )  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return can_clean(context)

    def execute(self, context):
        objects = selected_meshes(context)
        if not objects:
            self.report({"WARNING"}, iface_("Select one or more mesh objects"))
            return {"CANCELLED"}

        n_dirty = 0
        for obj in objects:
            report = scan_object(
                obj,
                merge_threshold=self.merge_threshold,
                area_eps=self.area_eps,
            )
            _scan_cache[obj.name] = report
            # Only error-class findings count as needing attention; an open
            # quad surface is normal for cloth and must not read as a problem.
            if report["n_errors"]:
                n_dirty += 1

        from ..core.utils import redraw_all_areas

        redraw_all_areas(context)

        if n_dirty == 0:
            self.report(
                {"INFO"},
                iface_(
                    "Scanned {count} mesh object(s): no defects found."
                ).format(count=len(objects)),
            )
        else:
            self.report(
                {"WARNING"},
                iface_(
                    "Scanned {count} mesh object(s): {dirty} need attention."
                ).format(count=len(objects), dirty=n_dirty),
            )
        return {"FINISHED"}


class MESH_OT_PPFMergeByDistance(_CountChangingRepair):
    """Weld vertices closer together than the merge distance. Removes the
    near-coincident pairs whose tiny gap makes the contact barrier's
    mass/gap^2 stiffness destroy the conditioning of the solver's fp32 Newton
    matrix. Changes the vertex count"""

    bl_idname = "object.ppf_merge_by_distance"
    bl_label = "Merge by Distance"
    report_verb = "Merged"

    merge_threshold: bpy.props.FloatProperty(
        name="Merge Distance",
        description=(
            "Weld vertices closer together than this, in local mesh units"
        ),
        default=DEFAULT_MERGE_THRESHOLD,
        min=0.0,
        soft_max=0.1,
        precision=6,
    )  # pyright: ignore

    def _dialog_props(self):
        return ("merge_threshold",)

    def repair(self, obj) -> int:
        before = len(obj.data.vertices)
        with _edit_bmesh(obj) as bm:
            bmesh.ops.remove_doubles(
                bm, verts=bm.verts[:], dist=self.merge_threshold
            )
        return before - len(obj.data.vertices)


class MESH_OT_PPFRemoveLooseVertices(_CountChangingRepair):
    """Delete vertices that belong to no face, together with their loose
    edges. The solver averages a vertex's contact parameters over its
    incident faces and aborts when a vertex has none. Pinned vertices are
    kept, since a pin holds them regardless. Changes the vertex count"""

    bl_idname = "object.ppf_remove_loose_vertices"
    bl_label = "Remove Loose Vertices"
    report_verb = "Removed"

    def repair(self, obj) -> int:
        from ..core.encoder.mesh import (
            detect_hanging_stitch_vertices,
            detect_isolated_vertices,
        )

        if is_particle_mesh(obj):
            # Every grain center is legitimately face-less; deleting them
            # would delete the whole SAND body.
            return 0
        # Same lazy pin resolution as scan_object: a pin can only shrink this
        # set, so skip the expensive vertex-group walk when it is empty.
        doomed = set(detect_isolated_vertices(obj.data))
        doomed.update(detect_hanging_stitch_vertices(obj.data))
        if not doomed:
            return 0
        doomed -= pinned_vertices_for(obj)
        if not doomed:
            return 0
        with _edit_bmesh(obj) as bm:
            bm.verts.ensure_lookup_table()
            n = len(bm.verts)
            geom = [bm.verts[i] for i in sorted(doomed) if 0 <= i < n]
            if geom:
                bmesh.ops.delete(bm, geom=geom, context="VERTS")
        return len(doomed)


class MESH_OT_PPFDissolveDegenerate(_CountChangingRepair):
    """Collapse zero-area faces and zero-length edges. A face with no area
    has no defined normal, so the contact normal and the bending hinge built
    on it are both undefined. Changes the vertex count"""

    bl_idname = "object.ppf_dissolve_degenerate"
    bl_label = "Dissolve Degenerate"
    report_verb = "Dissolved"

    merge_threshold: bpy.props.FloatProperty(
        name="Collapse Distance",
        description=(
            "Collapse edges shorter than this, in local mesh units"
        ),
        default=DEFAULT_MERGE_THRESHOLD,
        min=0.0,
        soft_max=0.1,
        precision=6,
    )  # pyright: ignore

    def _dialog_props(self):
        return ("merge_threshold",)

    def repair(self, obj) -> int:
        before_verts = len(obj.data.vertices)
        before_polys = len(obj.data.polygons)
        with _edit_bmesh(obj) as bm:
            bmesh.ops.dissolve_degenerate(
                bm, dist=self.merge_threshold, edges=bm.edges[:]
            )
        # Report whichever count moved: a collapse can remove faces, or only
        # merge vertices, depending on how the degeneracy is shaped.
        return max(
            before_verts - len(obj.data.vertices),
            before_polys - len(obj.data.polygons),
        )


class MESH_OT_PPFDeleteDuplicateFaces(_RepairBase):
    """Delete faces that share their full vertex set with another face. The
    solver builds a degenerate bending element from coincident faces and
    aborts, and the encoder already refuses to transfer them. Leaves the
    vertex count unchanged, so pins and caches survive"""

    bl_idname = "object.ppf_delete_duplicate_faces"
    bl_label = "Delete Duplicate Faces"
    report_verb = "Deleted"

    def repair(self, obj) -> int:
        with _edit_bmesh(obj) as bm:
            seen: set[tuple[int, ...]] = set()
            doomed = []
            for face in bm.faces:
                key = tuple(sorted(v.index for v in face.verts))
                if key in seen:
                    doomed.append(face)
                else:
                    seen.add(key)
            if doomed:
                bmesh.ops.delete(bm, geom=doomed, context="FACES_ONLY")
            return len(doomed)


class MESH_OT_PPFTriangulate(_RepairBase):
    """Triangulate every face with more than three corners, so Blender has no
    split diagonal left to re-pick from the deformed shape. Removes the drift
    between the displayed surface and the simulated one. Leaves the vertex
    count unchanged, so pins and caches survive"""

    bl_idname = "object.ppf_triangulate_for_solver"
    bl_label = "Triangulate"
    report_verb = "Triangulated"

    def repair(self, obj) -> int:
        with _edit_bmesh(obj) as bm:
            targets = [f for f in bm.faces if len(f.verts) > 3]
            if targets:
                bmesh.ops.triangulate(bm, faces=targets)
            return len(targets)


class MESH_OT_PPFRecalcNormals(_RepairBase):
    """Make face winding consistent and outward. Tetrahedralization needs a
    consistently wound surface to tell inside from outside. Leaves the vertex
    count unchanged, so pins and caches survive"""

    bl_idname = "object.ppf_recalc_normals_outside"
    bl_label = "Recalculate Outside"
    report_verb = "Rewound"

    def repair(self, obj) -> int:
        before = find_surface_defects(obj.data)["bad_winding"]
        with _edit_bmesh(obj) as bm:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        after = find_surface_defects(obj.data)["bad_winding"]
        # Report the edge disagreements resolved. A mesh that was already
        # consistent reports 0 and the base class cancels, which is the
        # honest outcome: nothing needed rewinding.
        return max(0, before - after)


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
#
# The report rendering lives here rather than in ui/dynamics/panels.py so the
# defect table sits next to the detectors that fill it: adding a check means
# editing one file, and the row spec cannot drift from the detector's keys.


def _fmt_distance(value: float) -> str:
    """Format a small length so nanometer-scale gaps stay readable."""
    if value == 0.0:
        return "0"
    if value < 1e-6:
        return f"{value * 1e9:.3g} nm"
    if value < 1e-3:
        return f"{value * 1e6:.3g} um"
    if value < 1.0:
        return f"{value * 1e3:.4g} mm"
    return f"{value:.4g}"


def _fix_button(box, operator, text, icon, report, *, pass_threshold=False):
    """Draw a repair button, handing it the distance the scan actually used.

    Without this the dialog would open on the operator's own default, so a
    scan run at a custom distance could be followed by a merge at a different
    one, silently repairing more or less than the report described.
    """
    op = box.operator(operator, text=text, icon=icon)
    if pass_threshold:
        op.merge_threshold = report["merge_threshold"]
    return op


def _draw_near_duplicates(box, defect, report):
    box.label(
        text=iface_("{count} near-coincident vertex pair(s)").format(
            count=defect["count"]
        ),
        icon="ERROR",
    )
    sub = box.column(align=True)
    sub.label(
        text=iface_("smallest gap {local} local, {world} world").format(
            local=_fmt_distance(defect["min_dist"]),
            world=_fmt_distance(defect.get("min_dist_world", defect["min_dist"])),
        )
    )
    sub.label(
        text=iface_("scanned at merge distance {value}").format(
            value=f"{report['merge_threshold']:.6g}"
        )
    )
    _fix_button(
        box,
        "object.ppf_merge_by_distance",
        "Merge by Distance",
        "AUTOMERGE_ON",
        report,
        pass_threshold=True,
    )


def _draw_surface(box, defect, report):
    """Surface integrity: three sub-counts, only winding is repairable."""
    if defect["bad_winding"]:
        box.label(
            text=iface_("{count} inconsistently wound edge(s)").format(
                count=defect["bad_winding"]
            ),
            icon="ERROR",
        )
        box.operator(
            "object.ppf_recalc_normals_outside",
            text="Recalculate Outside",
            icon="NORMALS_FACE",
        )
    if defect["boundary"] or defect["non_manifold"]:
        sub = box.column(align=True)
        if defect["boundary"]:
            sub.label(
                text=iface_("{count} boundary edge(s), surface is open").format(
                    count=defect["boundary"]
                ),
                icon="INFO",
            )
        if defect["non_manifold"]:
            sub.label(
                text=iface_("{count} non-manifold edge(s)").format(
                    count=defect["non_manifold"]
                ),
                icon="INFO",
            )
        sub.label(text="Fine for a SHELL; a SOLID needs a closed surface.")
        sub.label(text="Not repaired automatically: closing a boundary is modeling.")


def _draw_resplittable(box, defect, report):
    box.label(
        text=iface_("{count} face(s) with more than 3 corners").format(
            count=defect["count"]
        ),
        icon="INFO",
    )
    sub = box.column(align=True)
    sub.label(
        text=iface_("largest fold {angle} deg, {count} past the flip boundary").format(
            angle=f"{defect.get('max_fold_deg', 0.0):.1f}",
            count=defect.get("past_flip", 0),
        )
    )
    sub.label(text="Blender may split these differently than the simulation.")
    box.operator(
        "object.ppf_triangulate_for_solver",
        text="Triangulate",
        icon="MOD_TRIANGULATE",
    )


def _draw_linked_duplicate(box, defect, report):
    box.label(
        text=iface_("Linked Duplicate of {name}").format(
            name=defect["siblings"][0]
        ),
        icon="ERROR",
    )
    sub = box.column(align=True)
    sub.label(text="Transfer refuses a shared mesh datablock.")
    sub.label(text="Use Object > Relations > Make Single User > Object & Data.")


def _simple_row(label_key, icon, operator, op_text, op_icon, *, pass_threshold=False):
    """Build a renderer for a defect that is one count plus one fix button."""

    def _draw(box, defect, report):
        box.label(
            text=iface_(label_key).format(count=defect["count"]), icon=icon
        )
        if operator is not None:
            _fix_button(
                box, operator, op_text, op_icon, report,
                pass_threshold=pass_threshold,
            )

    return _draw


#: Ordered defect rows: (report key, renderer, changes_vertex_count).
#: Order is worst-first, so the defect most likely to abort a run is read
#: before the merely cosmetic ones.
_DEFECT_ROWS = (
    ("linked_duplicate", _draw_linked_duplicate, False),
    ("near_duplicates", _draw_near_duplicates, True),
    (
        "isolated_verts",
        _simple_row(
            "{count} isolated vertex(es), in no face",
            "ERROR",
            "object.ppf_remove_loose_vertices",
            "Remove Loose Vertices",
            "X",
        ),
        True,
    ),
    (
        "hanging_verts",
        _simple_row(
            "{count} hanging seam vertex(es)",
            "ERROR",
            "object.ppf_remove_loose_vertices",
            "Remove Loose Vertices",
            "X",
        ),
        True,
    ),
    (
        "duplicate_faces",
        _simple_row(
            "{count} duplicate face(s)",
            "ERROR",
            "object.ppf_delete_duplicate_faces",
            "Delete Duplicate Faces",
            "X",
        ),
        False,
    ),
    (
        "degenerate_faces",
        _simple_row(
            "{count} degenerate (zero-area) face(s)",
            "ERROR",
            "object.ppf_dissolve_degenerate",
            "Dissolve Degenerate",
            "X",
            pass_threshold=True,
        ),
        True,
    ),
    ("surface", _draw_surface, False),
    ("resplittable", _draw_resplittable, False),
)


def _draw_object_report(layout, obj, report):
    obj_box = layout.box()
    obj_box.label(text=obj.name, icon="MESH_DATA")

    if report["n_errors"] == 0:
        obj_box.label(text="No defects found", icon="CHECKMARK")
        if report["n_notes"] == 0:
            return
        # Notes still render below: an open, quad-built SHELL is entirely
        # normal, but the artist may still want to know before a SOLID
        # transfer or a deforming run.

    needs_ack = False
    for key, renderer, changes_count in _DEFECT_ROWS:
        defect = report["defects"].get(key)
        if not defect or not defect["count"]:
            continue
        renderer(obj_box.column(), defect, report)
        needs_ack = needs_ack or changes_count

    if needs_ack:
        # Read from the report; recomputing here would hit the disk on every
        # redraw (see the note in scan_object).
        dependents = report.get("dependents", [])
        warn = obj_box.box()
        warn.alert = bool(dependents)
        if dependents:
            warn.label(text="Changing the vertex count invalidates:", icon="ERROR")
            for entry in dependents:
                warn.label(text=entry, icon="DOT")
            warn.label(text="Each fix asks for confirmation first.")
        else:
            warn.label(
                text="No cache or pin depends on this vertex count.",
                icon="INFO",
            )


def draw_mesh_cleaning(layout, context):
    """Draw the Mesh Cleaning entry of the Utility Tools panel.

    The Scan button is drawn unconditionally and grayed out when its
    precondition fails, with a status label explaining why, so the feature
    stays discoverable rather than vanishing from the panel.
    """
    box = layout.box()
    box.label(text="Mesh Cleaning")
    box.label(text="Find geometry the solver rejects or chokes on.")

    objects = selected_meshes(context)
    col = box.column()
    col.enabled = can_clean(context)
    col.operator(
        "object.ppf_scan_mesh_defects",
        text="Scan Selected Meshes",
        icon="VIEWZOOM",
    )

    if context.mode != "OBJECT":
        box.label(text="Switch to Object Mode", icon="INFO")
        return
    if not objects:
        box.label(text="Select one or more mesh objects", icon="INFO")
        return

    box.label(
        text=iface_("{count} mesh object(s) selected").format(count=len(objects)),
        icon="CHECKMARK",
    )

    unscanned = 0
    for obj in objects:
        report = get_scan_report(obj.name)
        if report is None:
            unscanned += 1
            continue
        _draw_object_report(box, obj, report)

    if unscanned:
        box.label(
            text=iface_("{count} object(s) not scanned yet").format(
                count=unscanned
            ),
            icon="INFO",
        )


classes = (
    MESH_OT_PPFScanMeshes,
    MESH_OT_PPFMergeByDistance,
    MESH_OT_PPFRemoveLooseVertices,
    MESH_OT_PPFDissolveDegenerate,
    MESH_OT_PPFDeleteDuplicateFaces,
    MESH_OT_PPFTriangulate,
    MESH_OT_PPFRecalcNormals,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    clear_scan_report()
