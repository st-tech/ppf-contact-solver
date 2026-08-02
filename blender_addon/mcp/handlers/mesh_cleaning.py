"""Mesh cleaning and mesh utility handlers (scan, repair, triangulate).

These wrap the Utility Tools panel's Mesh Cleaning operators. Two shapes are
deliberate here.

Every tool takes explicit ``object_names`` rather than reading the current
selection. The operators underneath work on the selection, which is the right
affordance for an artist with the meshes highlighted in the viewport, but an
MCP caller has no view of what happens to be selected and these edits are
destructive. Naming the targets makes the call self-describing and repeatable.
Every named object has to be reachable and selectable in the active view
layer, and a call naming one that is not fails whole rather than repairing the
rest, so a result always describes every object the caller asked about.

The three repairs that change the vertex count require ``acknowledge=True``,
mirroring the confirmation dialog the panel raises. A vertex-count change
invalidates a captured deformation and the display cache, both of which have
to be re-taken, and can shift which vertices a pin group holds;
``scan_meshes`` reports exactly what each object would invalidate under
``dependents``, so the caller can look before it commits.
"""

import contextlib

import bpy  # pyright: ignore

from ...mesh_ops.cleaning_ops import (
    find_surface_defects,
    get_scan_report,
    vertex_count_dependents,
)
from ..decorators import (
    MCPError,
    ValidationError,
    mcp_handler,
)


def _resolve_meshes(object_names: list[str]) -> list:
    """Return the named mesh objects, raising on anything unusable.

    Resolved up front and as a whole, so a typo in the third name cannot leave
    the first two already repaired.
    """
    if not object_names:
        raise ValidationError("object_names must name at least one mesh object")

    objects, missing, wrong_type = [], [], []
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
        elif obj.type != "MESH" or obj.data is None:
            wrong_type.append(f"{name} ({obj.type})")
        else:
            objects.append(obj)

    if missing:
        raise ValidationError(f"No such object(s): {', '.join(missing)}")
    if wrong_type:
        raise ValidationError(
            f"Not mesh object(s): {', '.join(wrong_type)}. Mesh cleaning "
            "applies to MESH objects only."
        )
    return objects


@contextlib.contextmanager
def _selected_exactly(objects):
    """Select exactly *objects* for the duration, then restore.

    The cleaning operators read ``context.selected_objects``, and their poll
    additionally requires Object Mode, so both are established here.

    The selection is verified rather than assumed. ``Object.select_set(True)``
    is a no-op on an object the view layer marks unselectable, which covers
    one hidden in the viewport, one hidden by its collection, and one carrying
    Disable Selection. The operators would then run over a shorter list than
    the caller named, and the result would describe a mesh nothing touched, so
    the flag is read back and the whole call is refused before any mesh is
    edited. Reading the flag back rather than testing the settings that
    produce it keeps this in step with whichever combination of those settings
    the view layer counts as unselectable.

    Everything the block changes is put back: the selection, the active
    object, and the mode. The restore is tolerant of an object that no longer
    resolves (a repair can leave the datablock intact but the pointer stale),
    and it runs while an error is propagating, so raising there would replace
    the reason the call failed.
    """
    view_layer = bpy.context.view_layer

    # Settle the view layer before reading anything off it. Both the membership
    # list below and the selection flag further down live on a base, and a base
    # is built for an object when the view layer resyncs, so an object linked
    # into a collection earlier in this same tick is absent from
    # ``view_layer.objects`` and reads as unselected however reachable and
    # selectable it is. Without this both checks refuse an object that is
    # perfectly usable, which is the opposite of the failure they exist to
    # catch, and a caller that creates a mesh and cleans it in one script meets
    # it every time.
    view_layer.update()

    # Resolved before anything is touched, so this refusal leaves the scene
    # exactly as it was found.
    unreachable = [o.name for o in objects if o.name not in view_layer.objects]
    if unreachable:
        raise ValidationError(
            f"Object(s) not in the active view layer: {', '.join(unreachable)}. "
            "Mesh cleaning can only reach objects the current scene shows."
        )

    saved_active = view_layer.objects.active
    saved_mode = saved_active.mode if saved_active is not None else "OBJECT"
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError as exc:
            raise MCPError(f"Could not leave the current mode: {exc}") from exc

    saved = []
    for obj in view_layer.objects:
        try:
            saved.append((obj, obj.select_get()))
        except RuntimeError:
            pass

    try:
        for obj, _ in saved:
            with contextlib.suppress(RuntimeError):
                obj.select_set(False)
        for obj in objects:
            with contextlib.suppress(RuntimeError):
                obj.select_set(True)

        refused = []
        for obj in objects:
            selected = False
            with contextlib.suppress(RuntimeError, ReferenceError):
                selected = obj.select_get()
            if not selected:
                refused.append(obj.name)
        if refused:
            raise ValidationError(
                "Object(s) not selectable in the active view layer: "
                f"{', '.join(refused)}. An object hidden in the viewport, "
                "hidden by its collection, or carrying Disable Selection "
                "cannot be selected, and these tools drive operators that "
                "read the selection, so such an object would be skipped and "
                "the result would not describe it. Make it visible and "
                "selectable, then call again."
            )

        with contextlib.suppress(RuntimeError, ReferenceError):
            view_layer.objects.active = objects[0]
        yield
    finally:
        for obj, was_selected in saved:
            with contextlib.suppress(RuntimeError, ReferenceError):
                obj.select_set(was_selected)
        with contextlib.suppress(RuntimeError, ReferenceError):
            view_layer.objects.active = saved_active
        if saved_mode != "OBJECT":
            with contextlib.suppress(RuntimeError, ReferenceError):
                bpy.ops.object.mode_set(mode=saved_mode)


def _counts(objects) -> dict[str, dict[str, int]]:
    """Element counts: the quantities a repair that adds or removes geometry moves."""
    return {
        obj.name: {
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
        }
        for obj in objects
    }


def _winding(objects) -> dict[str, dict[str, int]]:
    """Inconsistently wound edges: the quantity a rewind moves.

    ``recalc_face_normals`` reorders each face's loops and leaves both element
    counts alone, so the count of manifold edges whose two faces traverse them
    in the same direction is what records the repair. It is the same quantity
    ``MESH_OT_PPFRecalcNormals.repair`` returns, so the tool's report and the
    operator's own verdict are computed from one measurement.
    """
    return {
        obj.name: {"bad_winding": find_surface_defects(obj.data)["bad_winding"]}
        for obj in objects
    }


def _dependents(objects) -> dict[str, list[str]]:
    """Per object, what a vertex-count change on it would invalidate."""
    return {obj.name: vertex_count_dependents(obj) for obj in objects}


def _apply(op, objects, *, verb: str, measure=_counts, **kwargs) -> dict:
    """Run one repair operator over *objects* and report what moved.

    *measure* names the quantities this repair is able to move and reads them
    per object; an object where any of them differs across the call lands in
    ``changed``, carrying every measured quantity as a ``<name>_before`` /
    ``<name>_after`` pair. It is measured here rather than read out of the
    operator's status report, which bpy.ops does not hand back to a caller. A
    repair that rewrites a mesh in place moves no element count, so it passes
    the quantity it does move and the report follows.

    Both readings are taken inside the selection block, which is where the
    switch to Object Mode happens. While an object is in Edit Mode its mesh
    datablock does not carry the pending edits, since the BMesh is written
    back when the mode is left, so a reading taken outside the block would be
    the pre-edit mesh on one side and the written-back mesh on the other and
    the difference between the two would land in the report as a change this
    repair made.
    """
    with _selected_exactly(objects):
        before = measure(objects)
        status = op("EXEC_DEFAULT", **kwargs)
        after = measure(objects)

    changed = []
    for name, fields in before.items():
        if all(after[name][key] == value for key, value in fields.items()):
            continue
        entry = {"object_name": name}
        for key, value in fields.items():
            entry[f"{key}_before"] = value
            entry[f"{key}_after"] = after[name][key]
        changed.append(entry)

    # Reported as ``operator_status``, never ``status``: mcp_handler treats a
    # returned dict that already carries a ``status`` key as a fully formed
    # response and passes it through untouched, so naming this one ``status``
    # would replace the envelope's "success" with the operator's return set and
    # leave every caller unable to tell a success from a failure.
    if not changed:
        return {
            "message": f"Nothing to repair on {len(objects)} mesh object(s).",
            "changed": [],
            "changed_count": 0,
            "operator_status": sorted(status),
        }
    return {
        "message": (
            f"{verb} on {len(changed)} of {len(objects)} mesh object(s). "
            "Run Transfer again before the next simulation."
        ),
        "changed": changed,
        "changed_count": len(changed),
        "operator_status": sorted(status),
    }


def _apply_count_changing(op, objects, *, verb: str, **kwargs) -> dict:
    """Run a count-changing repair and report which caches it deleted.

    ``clear_stale_caches`` decides whether the invalidated caches go with the
    repair, and a caller has no other view of what it took. So the dependents
    are read on both sides of the call and the ones that are gone afterward
    come back as ``cleared_caches``, in the same wording the refusal message
    and ``scan_meshes`` use. A pin group is never among them: a count change
    can shift its membership, and nothing here deletes one.

    ``vertex_count_dependents`` reads the modifier stack, the capture cache on
    disk, and the group's pin list, never the mesh datablock, so unlike the
    element counts in ``_apply`` these two readings do not have to be taken
    inside the selection block to agree with each other.
    """
    before = _dependents(objects)
    result = _apply(op, objects, verb=verb, **kwargs)
    after = _dependents(objects)
    result["cleared_caches"] = [
        f"{name}: {dep}"
        for name, deps in before.items()
        for dep in deps
        if dep not in after[name]
    ]
    return result


def _require_acknowledgement(objects, acknowledge: bool) -> None:
    """Refuse a vertex-count change until the caller has accepted the cost."""
    if acknowledge:
        return
    dependents = []
    for name, deps in _dependents(objects).items():
        dependents += [f"{name}: {dep}" for dep in deps]
    detail = (
        f" It would invalidate: {'; '.join(dependents)}."
        if dependents
        else " No capture cache, display cache, or pin is affected."
    )
    raise ValidationError(
        "This repair changes the vertex count, so it needs "
        f"acknowledge=true.{detail}"
    )


def _compact(report: dict) -> dict:
    """Strip the per-vertex index lists from a scan report.

    ``scan_object`` carries the full offending index list for the panel's
    select-vertices buttons, which on a dense mesh is far larger than the rest
    of the payload combined. The counts and the eight-index previews are what
    a caller acts on.
    """
    defects = {}
    for key, defect in report["defects"].items():
        defects[key] = {k: v for k, v in defect.items() if k != "verts"}
    return {**report, "defects": defects}


@mcp_handler
def scan_meshes(
    object_names: list[str],
    merge_threshold: float = 1e-4,
    area_eps: float = 0.0,
):
    """Scan meshes for geometry the solver rejects, without modifying anything.

    Reports per object, split into errors (near-coincident vertices, isolated
    and hanging vertices, duplicate and degenerate faces, linked duplicates,
    inconsistent winding) and notes (boundary edges, non-manifold edges,
    re-splittable quads). Notes are normal for cloth: an open quad panel is not
    a defect. Each report also carries ``dependents``, what a vertex-count
    change on that object would invalidate.

    Args:
        object_names: Mesh objects to scan
        merge_threshold: Vertices closer than this (local units) count as
            near-coincident. Matches Blender's Merge by Distance default
        area_eps: Faces at or below this area (local units squared) count as
            degenerate. Zero reports only exactly zero-area faces
    """
    objects = _resolve_meshes(object_names)

    with _selected_exactly(objects):
        bpy.ops.object.ppf_scan_mesh_defects(
            "EXEC_DEFAULT",
            merge_threshold=merge_threshold,
            area_eps=area_eps,
        )

    reports, n_errors, n_notes = [], 0, 0
    unscanned = []
    for obj in objects:
        report = get_scan_report(obj.name)
        if report is None:
            unscanned.append(obj.name)
            continue
        reports.append(_compact(report))
        n_errors += report["n_errors"]
        n_notes += report["n_notes"]

    # The operator writes one report per selected mesh and _selected_exactly
    # has already established that every named object is selected, so a
    # missing report means the scan did not reach that object. Reporting the
    # rest as a success would present a mesh nobody looked at as clean.
    if unscanned:
        raise MCPError(
            f"No scan report for: {', '.join(unscanned)}. The scan writes one "
            f"report per selected mesh and {len(reports)} of {len(objects)} "
            "named object(s) produced one, so this call cannot say whether "
            "the rest are clean."
        )

    dirty = [r["object"] for r in reports if r["n_errors"]]
    return {
        "message": (
            f"Scanned {len(reports)} mesh object(s): "
            + (
                f"{len(dirty)} need attention ({n_errors} error(s), "
                f"{n_notes} note(s))"
                if dirty
                else f"no defects found ({n_notes} note(s))"
            )
        ),
        "reports": reports,
        "objects_needing_attention": dirty,
        "total_errors": n_errors,
        "total_notes": n_notes,
    }


@mcp_handler
def merge_by_distance(
    object_names: list[str],
    merge_threshold: float = 1e-4,
    acknowledge: bool = False,
    clear_stale_caches: bool = True,
):
    """Weld near-coincident vertices. Changes the vertex count.

    A pair of vertices separated by a tiny gap drives the contact barrier's
    mass/gap^2 stiffness through the conditioning of the solver's fp32 Newton
    matrix, so welding them is what makes such a mesh simulable.

    Args:
        object_names: Mesh objects to repair
        merge_threshold: Weld vertices closer together than this, local units
        acknowledge: Must be true. Confirms the vertex-count change and the
            caches it invalidates, which scan_meshes reports as dependents
        clear_stale_caches: Delete the capture and display caches the change
            invalidates, which the result reports as cleared_caches. True by
            default, the same value the panel's dialog opens with. Pass false
            to keep them, and expect the viewport overlay to read data sized
            for the old vertex count until Transfer rewrites it
    """
    objects = _resolve_meshes(object_names)
    _require_acknowledgement(objects, acknowledge)
    return _apply_count_changing(
        bpy.ops.object.ppf_merge_by_distance,
        objects,
        verb="Merged vertices",
        merge_threshold=merge_threshold,
        acknowledge=True,
        clear_stale_caches=clear_stale_caches,
    )


@mcp_handler
def remove_loose_vertices(
    object_names: list[str],
    acknowledge: bool = False,
    clear_stale_caches: bool = True,
):
    """Delete vertices that belong to no face. Changes the vertex count.

    A faceless vertex carries no elastic energy, so the solver has nothing to
    hold it with. Pinned vertices are exempt and are never removed.

    Args:
        object_names: Mesh objects to repair
        acknowledge: Must be true. Confirms the vertex-count change and the
            caches it invalidates, which scan_meshes reports as dependents
        clear_stale_caches: Delete the capture and display caches the change
            invalidates, which the result reports as cleared_caches. True by
            default, the same value the panel's dialog opens with. Pass false
            to keep them, and expect the viewport overlay to read data sized
            for the old vertex count until Transfer rewrites it
    """
    objects = _resolve_meshes(object_names)
    _require_acknowledgement(objects, acknowledge)
    return _apply_count_changing(
        bpy.ops.object.ppf_remove_loose_vertices,
        objects,
        verb="Removed loose vertices",
        acknowledge=True,
        clear_stale_caches=clear_stale_caches,
    )


@mcp_handler
def dissolve_degenerate_faces(
    object_names: list[str],
    merge_threshold: float = 1e-4,
    acknowledge: bool = False,
    clear_stale_caches: bool = True,
):
    """Collapse zero-area and slivered faces. Changes the vertex count.

    A face with no area has no well-defined normal, which is what the contact
    and bending terms are built on.

    Args:
        object_names: Mesh objects to repair
        merge_threshold: Edges shorter than this (local units) are collapsed
        acknowledge: Must be true. Confirms the vertex-count change and the
            caches it invalidates, which scan_meshes reports as dependents
        clear_stale_caches: Delete the capture and display caches the change
            invalidates, which the result reports as cleared_caches. True by
            default, the same value the panel's dialog opens with. Pass false
            to keep them, and expect the viewport overlay to read data sized
            for the old vertex count until Transfer rewrites it
    """
    objects = _resolve_meshes(object_names)
    _require_acknowledgement(objects, acknowledge)
    return _apply_count_changing(
        bpy.ops.object.ppf_dissolve_degenerate,
        objects,
        verb="Dissolved degenerate faces",
        merge_threshold=merge_threshold,
        acknowledge=True,
        clear_stale_caches=clear_stale_caches,
    )


@mcp_handler
def delete_duplicate_faces(object_names: list[str]):
    """Delete faces that repeat an existing face's vertex set.

    Two faces on the same vertices contribute their contact and elastic terms
    twice. The vertex count is unchanged, so no cache is invalidated.

    Args:
        object_names: Mesh objects to repair
    """
    objects = _resolve_meshes(object_names)
    return _apply(
        bpy.ops.object.ppf_delete_duplicate_faces,
        objects,
        verb="Deleted duplicate faces",
    )


@mcp_handler
def triangulate_for_solver(object_names: list[str]):
    """Triangulate n-gons and quads with a single diagonal per face.

    The vertex count is unchanged, so no cache is invalidated. Transfer
    triangulates on its own at encode time; use this when the triangulation
    has to be visible and stable in the viewport. For a mesh whose symmetry
    matters under bending, prefer symmetric_triangulate.

    Args:
        object_names: Mesh objects to triangulate
    """
    objects = _resolve_meshes(object_names)
    return _apply(
        bpy.ops.object.ppf_triangulate_for_solver,
        objects,
        verb="Triangulated",
    )


@mcp_handler
def recalculate_normals_outside(object_names: list[str]):
    """Make face winding consistent and outward.

    Inconsistent winding flips the normal a face contributes, which the
    contact and inflate terms read. The vertex count is unchanged, so no cache
    is invalidated, and the repair is reported as the ``bad_winding_before`` /
    ``bad_winding_after`` edge counts rather than as an element delta.

    Args:
        object_names: Mesh objects to repair
    """
    objects = _resolve_meshes(object_names)
    return _apply(
        bpy.ops.object.ppf_recalc_normals_outside,
        objects,
        verb="Recalculated normals",
        measure=_winding,
    )


@mcp_handler
def symmetric_triangulate(object_names: list[str]):
    """Triangulate by poking each face, keeping the mesh mirror-symmetric.

    A single-diagonal triangulation breaks a symmetric mesh's symmetry, which
    shows up as a lopsided drape under bending. Poking inserts a center vertex
    and fans the face into triangles instead, so it ADDS one vertex per face
    and therefore invalidates a captured deformation and the display cache,
    exactly as the count-changing repairs do. It is a Utility Tools operation
    rather than a repair, so it takes no acknowledgement and deletes nothing
    it invalidates: the vertex deltas come back in the result, and Transfer
    and Capture Deformation are what re-take the stale caches.

    Args:
        object_names: Mesh objects to triangulate
    """
    objects = _resolve_meshes(object_names)
    return _apply(
        bpy.ops.object.ppf_symmetric_triangulate,
        objects,
        verb="Symmetric-triangulated",
    )
