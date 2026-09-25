# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os

import bpy  # pyright: ignore

from ..models.groups import decode_vertex_group_identifier, iterate_active_object_groups
from .transform import world_matrix


def get_category_name():
    """Get the category name for the add-on."""
    return "ZOZO's Contact Solver"


# Characters a shell would reinterpret in a connection path that is
# interpolated into a command (``cd {path} && ...``,
# ``docker exec -w {path} ...``).  ``~`` is intentionally NOT included:
# ``~/work`` is a common, safe remote path and the shell expands it fine.  The
# usual path components (``/``, ``\``, ``:``, ``.``, ``-``, ``_``) are also
# allowed so Windows drive letters and separators pass.
_SHELL_UNSAFE_PATH_CHARS = frozenset(
    "&|;<>()$`\"'"     # shell metacharacters
    "*?[]{}#"          # glob, brace expansion, comment
    "!%^"              # history expansion, env, caret
)

# Whitespace on top of the above. A path that IS interpolated into a remote
# shell command has to refuse both; one that never reaches a shell is held to
# ``find_shell_unsafe_path_char`` instead.
_INVALID_PATH_CHARS = _SHELL_UNSAFE_PATH_CHARS | frozenset(" \t\n\r")


def find_shell_unsafe_path_char(path: str) -> str | None:
    """Return the first shell or glob metacharacter in *path*, else ``None``.

    The same set as :func:`find_invalid_path_char` MINUS whitespace. It is
    what a path is held to when it never reaches a shell: the Windows Native
    root is only ever an ``os.path.join`` base, a ``subprocess.Popen`` argv
    element, and that Popen's ``cwd``, none of which parse it. Whitespace is
    therefore harmless there, and it is ordinary: ``C:\\Users\\First
    Last\\Downloads\\...`` is what a Windows account with a two-word name
    gives, and holding it to the whitespace rule refuses that path.

    The metacharacters stay refused even here. They are illegal in a Windows
    filename anyway, so nothing legitimate is rejected, and the check keeps
    holding if this path is ever handed to something that does parse it.
    """
    for ch in path.strip():
        if ch in _SHELL_UNSAFE_PATH_CHARS:
            return ch
    return None


def find_invalid_path_char(path: str) -> str | None:
    """Return the first space or shell-unsafe character in *path*, else ``None``.

    Used to warn about, and refuse, connection paths that would break (or be
    misinterpreted) when interpolated into a shell command.  Returns ``None``
    for an empty or whitespace-only path, so callers can chain
    ``find_invalid_path_char(p) is None`` without rejecting an intentionally
    blank optional path; emptiness is validated separately per connection type.
    """
    for ch in path.strip():
        if ch in _INVALID_PATH_CHARS:
            return ch
    return None


def resolve_local_path(path: str) -> str:
    """Return *path* as an absolute path on the machine Blender runs on.

    The three connection paths that name a directory on the CLIENT machine
    (``local_path``, ``win_native_path`` and ``mac_native_path``) are
    ``DIR_PATH`` properties, so Blender's directory picker writes them in
    whatever form the user's
    ``Preferences > File Paths > Relative Paths`` setting asks for. That
    setting ships enabled, so once the .blend has been saved the picker
    stores a ``//``-prefixed path relative to the .blend
    (``//../ppf-contact-solver-win64``). ``//`` is Blender's own notation and
    means nothing to ``os.path``: probing it finds no solver, and the panel
    reports the folder the user just picked as one that holds no
    ``ppf-cts-server``. Expanding it here is what makes a picked folder mean
    the folder that was picked.

    ``bpy.path.abspath`` resolves the ``//`` form against the current .blend
    and leaves an already-absolute path alone, so this is safe to apply to
    every value the property can hold, typed or picked. A blank path stays
    blank: emptiness is validated per connection type, and turning it into
    the .blend's directory would make an unset field look set.

    Only client-side paths go through this. A REMOTE path (``ssh_remote_path``,
    ``docker_path``) names a directory on the solver host, where the client's
    .blend location has no meaning, so those are used verbatim.
    """
    if not path or not path.strip():
        return path
    # ``normpath`` after ``abspath`` because ``bpy.path.abspath`` only splices
    # the .blend's directory onto the tail: ``//../bundle`` next to a project
    # at ``/work/project`` comes back as ``/work/project/../bundle``. That
    # opens and probes correctly, but it is longer than the path it names, so
    # the Windows MAX_PATH projection would measure the wrong length, and it
    # is what the panel and every refusal would quote back at the user.
    return os.path.normpath(bpy.path.abspath(path.strip()))


# Windows refuses to open a path of 260 characters or more (the classic
# MAX_PATH limit) unless system-wide long-path support is enabled. The build
# pipeline writes cache files several directories below the solver root, so a
# long root pushes the deepest of them past the limit and the Transfer dies
# with a bare ``FileNotFoundError`` that names a path nobody recognizes. The
# deepest file is the tetrahedralize cache; its full server-side path mirrors
# ``datamodel/app.rs`` ``compose_data_dir`` plus the cache filename:
#
#   <root>\local\share\ppf-cts\git-<branch>\<project>\.cash\
#   <64-hex>__<64-hex>_tetrahedralize_<16-hex>.npz.npz
#
# The git branch is resolved on the server and is ``unknown`` for a packaged
# Windows build (no ``.git``), which is the case this guards.
WINDOWS_MAX_PATH = 260

# Width of the digest the server puts in a tetrahedralize cache name when the
# object carries tetrahedralizer overrides. The composer is
# ``datamodel/mesh.rs`` ``tetra_cache_name``, which digests any argument set
# to this fixed width, so one number covers every object a scene can hold.
# The default, argument-free name carries no digest and is 16 characters
# shorter, which is why the projection below is an upper bound.
TETRA_ARG_DIGEST_CHARS = 16


def projected_windows_cache_path_len(base_path: str, project_name: str) -> int:
    """Length of the longest cache file path the build pipeline writes under
    *base_path* for *project_name* on a Windows server.

    Mirrors the server-side layout described in the ``WINDOWS_MAX_PATH`` note,
    so the value covers every cache path the build can write and the guard
    warns as soon as any of them would reach the limit. A project whose SOLID
    objects all use default tetrahedralizer settings writes paths
    ``TETRA_ARG_DIGEST_CHARS`` shorter than this, so such a project can be
    warned while still fitting.
    """
    root = base_path.strip().rstrip("/\\")
    hex64 = "f" * 64  # a SHA-256 hex digest is 64 chars
    digest = "f" * TETRA_ARG_DIGEST_CHARS
    cache_file = f"{hex64}__{hex64}_tetrahedralize_{digest}.npz.npz"
    tail = "\\".join(
        [
            "local", "share", "ppf-cts", "git-unknown",
            project_name.strip() or "unnamed", ".cash", cache_file,
        ]
    )
    return len(root) + len("\\") + len(tail)


def windows_path_too_long(base_path: str, project_name: str) -> int | None:
    """Return the projected deepest cache-path length when it reaches the
    Windows ``MAX_PATH`` limit, else ``None``.

    This is a pure measurement of the path length. It does NOT consider
    whether long-path support is enabled on the host; callers that warn the
    user should also consult :func:`windows_long_paths_enabled`, since with
    long paths on the limit no longer applies and the warning is just noise.

    Returns ``None`` for an empty/whitespace-only path so callers can chain
    ``windows_path_too_long(p, n) is None`` next to
    :func:`find_invalid_path_char` without rejecting a blank field.
    """
    if not base_path.strip():
        return None
    projected = projected_windows_cache_path_len(base_path, project_name)
    return projected if projected >= WINDOWS_MAX_PATH else None


# Cache for windows_long_paths_enabled(). The registry value is honored per
# process at startup (the long-path opt-in is read once), so it won't change
# within a Blender session, and the panel's draw() would otherwise re-read it
# on every redraw. ``None`` means "not yet queried".
_windows_long_paths_enabled: bool | None = None


def _query_windows_long_paths_enabled() -> bool:
    """Read the system-wide long-path flag from the Windows registry.

    ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled``
    is the DWORD that opts the whole system out of the ``MAX_PATH`` limit.
    Returns ``False`` on non-Windows hosts and whenever the value is missing
    or unreadable (i.e. assume the limit applies unless we can prove it
    doesn't).
    """
    import sys

    if sys.platform != "win32":
        return False
    try:
        import winreg  # Windows-only stdlib module.

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return int(value) == 1
    except (OSError, ValueError):
        return False


def windows_long_paths_enabled() -> bool:
    """True when Windows long-path support is enabled system-wide, so paths
    past ``MAX_PATH`` no longer fail.

    ``False`` on non-Windows hosts and whenever the registry flag is unset or
    unreadable. Cached after the first read (see ``_windows_long_paths_enabled``).
    """
    global _windows_long_paths_enabled
    if _windows_long_paths_enabled is None:
        _windows_long_paths_enabled = _query_windows_long_paths_enabled()
    return _windows_long_paths_enabled


def find_invalid_name_char(name: str) -> str | None:
    """Return the first character in *name* that is not filename-safe, else ``None``.

    A project name becomes a single directory-name component on the server and
    is interpolated into commands, so it is held to a stricter rule than a
    path: only letters, digits, ``.``, ``-`` and ``_`` are allowed. That
    rejects spaces, shell/glob metacharacters, AND path separators (``/``,
    ``\\``, ``:``), none of which belong in a single name component. Returns
    ``None`` for an empty or whitespace-only name; emptiness is validated
    separately.
    """
    for ch in name.strip():
        if not (ch.isalnum() or ch in "._-"):
            return ch
    return None


def count_duplicate_faces(obj) -> int:
    """Return how many triangles share their full vertex set with an
    earlier triangle once *obj*'s mesh is tessellated the way the encoder
    tessellates it (Blender's ``loop_triangles``). ``0`` means every
    triangle is unique.

    Two coincident triangles (the same three vertices, in any winding)
    make the solver's bending-hinge builder produce a degenerate
    element and abort the simulation at startup. They almost always come
    from doubled geometry welded with Merge by Distance, common in
    airbag / inflate setups, so the dynamics pipeline rejects them
    up-front and names the object rather than silently dropping faces
    the user may have placed on purpose.

    Returns ``0`` for non-mesh objects (curves, etc.), which have no
    polygons in the Blender mesh sense.

    Used by:
      * ``OBJECT_OT_AddObjectsToGroup`` to refuse doubled meshes at
        assignment time, so the user sees an explicit error popup.
      * ``encoder.mesh._build_obj_data`` to fail the Transfer if a
        doubled mesh got assigned through a path that bypasses the
        operator (older saves, MCP scripts).
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return 0
    from .numpy_mesh_utils import loop_triangle_indices
    seen: set[tuple[int, ...]] = set()
    duplicates = 0
    for tri in loop_triangle_indices(obj.data):
        key = tuple(sorted(int(v) for v in tri))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


class DegenerateTessellationError(ValueError):
    """A Transfer refused because an object tessellates into triangles with no
    usable rest shape.

    Subclasses ``ValueError`` so every existing ``except ValueError`` around the
    encoder keeps catching it; the added fields are what lets the UI offer the
    repair instead of only printing the sentence. ``repairable`` is
    ``find_degenerate_tessellation``'s ``all_repairable_by_triangulation``, which is the
    question "would triangulating help?", so a dialog must not offer the
    Triangulate button when it is False: those faces are degenerate themselves
    and no triangulation of them avoids the problem.
    """

    def __init__(self, message, *, object_name, group_name, polygons, repairable):
        super().__init__(message)
        self.object_name = object_name
        self.group_name = group_name
        self.polygons = list(polygons)
        self.repairable = bool(repairable)


def triangulate_degenerate_faces(obj, found=None) -> tuple[int, int]:
    """Triangulate exactly the polygons of *obj* whose tessellation has no
    usable rest shape, leaving every other face of the mesh alone.

    Returns ``(split, unrepairable)``: how many polygons were split, and how
    many were left because no triangulation of them can help (a flagged polygon
    that is ALREADY a triangle is its own only triangulation, and one carrying a
    zero-length boundary edge forces a degenerate triangle into every
    triangulation).

    Scoped to the offending polygons on purpose. Face > Triangulate Faces over
    a selection converts every quad in it, which is a far larger change to the
    artist's mesh than the defect warrants, and the rest of the mesh has nothing
    wrong with it.

    The caller must be in Object Mode: bmesh reads and writes the mesh
    datablock, which is stale while the object is in Edit Mode.

    `found` is a :func:`find_degenerate_tessellation` result for `obj`, so a
    caller that has already scanned does not pay for the pass twice. That scan
    measures the BASE cage, deliberately: re-splitting a polygon is all this
    can change, and a deform-evaluated pose is not something a split moves.
    """
    import bmesh

    if found is None:
        found = find_degenerate_tessellation(obj)
    if not found["polygons"]:
        return 0, 0
    if not found["repairable_polygons"]:
        return 0, len(found["polygons"])

    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    n = len(bm.faces)
    # The repairable SUBSET, not every flagged polygon. A polygon with no
    # sound triangulation is left exactly as it is: splitting it would edit
    # the artist's mesh and still leave a degenerate triangle behind.
    splittable = [bm.faces[i] for i in found["repairable_polygons"] if 0 <= i < n]
    unrepairable = len(found["polygons"]) - len(found["repairable_polygons"])
    if splittable:
        bmesh.ops.triangulate(
            bm, faces=splittable, quad_method="BEAUTY", ngon_method="BEAUTY"
        )
        bm.to_mesh(me)
    bm.free()
    if splittable:
        me.update()
    return len(splittable), unrepairable


def min_rest_condition() -> float:
    """The smallest rest-shape conditioning the solver can carry, computed
    rather than written down: ``sqrt(float32 eps)``.

    A shell face's rest matrix is inverted once at scene build and the elastic
    Hessian is quadratic in that inverse, so a face conditioned at ratio ``r``
    holds its Hessian entries to a relative precision of about
    ``float32 eps / r**2``. At ``r = sqrt(float32 eps)`` that reaches 1 and no
    correct digit is left. The solver's own gate,
    ``builder::REST_SHAPE_MIN_CONDITION``, is the same number and carries the
    full derivation; both gates have to grant the same set, so neither side
    rounds it.
    """
    import numpy as np

    return float(np.sqrt(np.finfo(np.float32).eps))


def _triangle_conditioning(corners):
    """Per-triangle ``smin / smax`` of the rest tangent matrix, given an
    ``(N, 3, 3)`` array of triangle corner positions.

    This is the exact quantity ``builder::invert_rest_or_panic2`` tests, formed
    without an SVD. The two rest edge vectors ``e0 = c1 - c0`` and
    ``e1 = c2 - c0`` make a 3x2 matrix ``D`` whose singular values are what the
    solver's tangent-frame projection preserves. ``D.T @ D`` is 2x2, its
    determinant is ``(s0 * s1) ** 2`` and its larger eigenvalue is ``s0 ** 2``,
    so the ratio is ``sqrt(det) / s0 ** 2`` in closed form.

    A degenerate triangle returns ``0.0``: zero area gives ``det == 0``, and
    three coincident corners give a zero matrix, which the guarded divide sends
    to zero rather than to NaN.
    """
    import numpy as np

    e0 = corners[:, 1] - corners[:, 0]
    e1 = corners[:, 2] - corners[:, 0]
    g00 = np.einsum("ij,ij->i", e0, e0)
    g01 = np.einsum("ij,ij->i", e0, e1)
    g11 = np.einsum("ij,ij->i", e1, e1)
    trace = g00 + g11
    det = np.maximum(g00 * g11 - g01 * g01, 0.0)
    disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
    smax_sq = 0.5 * (trace + disc)
    ratio = np.zeros(len(corners), dtype=np.float64)
    np.divide(np.sqrt(det), smax_sq, out=ratio, where=smax_sq > 0.0)
    return ratio


def _fill_is_sound(tris, loop_co_world) -> bool:
    """Whether every triangle of one fill clears :func:`min_rest_condition`.

    Measured on the WORLD-linear corners, which is what the gate flags and what
    the solver's rest shape is built from. The FILL itself was chosen from the
    local corners, the space the repair runs bmesh in.
    """
    import numpy as np

    if not tris:
        return False
    corners = np.array(
        [[loop_co_world[i] for i in t] for t in tris], dtype=np.float64
    )
    return bool(_triangle_conditioning(corners).min() >= min_rest_condition())


def _beauty_triangulations(polygons_local):
    """BEAUTY fills for many polygons, in one scratch bmesh.

    Returns one list of index triples per input polygon, in order, with ``[]``
    for a polygon Blender refuses to rebuild. A polygon of fewer than four
    corners is its own only triangulation and gets ``[]``, since no re-split
    exists to offer.
    """
    import bmesh
    from mathutils import Vector

    bm = bmesh.new()
    try:
        faces, offsets = [], []
        for loop_co in polygons_local:
            if len(loop_co) < 4:
                faces.append(None)
                offsets.append(None)
                continue
            base = len(bm.verts)
            verts = [bm.verts.new(Vector(c)) for c in loop_co]
            try:
                faces.append(bm.faces.new(verts))
            except (ValueError, RuntimeError):
                faces.append(None)
            offsets.append(base)
        if not any(f is not None for f in faces):
            return [[] for _ in polygons_local]
        # An n-gon fill is projected through the face normal, and a face built
        # by hand carries (0, 0, 0) until the mesh is asked for one.
        bm.normal_update()
        bm.verts.index_update()
        owner = {}
        for i, face in enumerate(faces):
            if face is not None:
                for v in face.verts:
                    owner[v.index] = i
        bmesh.ops.triangulate(
            bm,
            faces=[f for f in faces if f is not None],
            quad_method="BEAUTY",
            ngon_method="BEAUTY",
        )
        bm.verts.index_update()
        out = [[] for _ in polygons_local]
        for tri in bm.faces:
            indices = [v.index for v in tri.verts]
            which = owner.get(indices[0])
            if which is None or offsets[which] is None:
                continue
            out[which].append(tuple(i - offsets[which] for i in indices))
        return out
    finally:
        bm.free()


def find_degenerate_tessellation(obj, local_verts=None, conditioning=True) -> dict:
    """Locate triangles with no usable rest shape in the tessellation the
    encoder ships.

    Returns ``{"count", "polygons", "repairable_polygons", "triangle_polygons",
    "all_repairable_by_triangulation"}``: how many of ``loop_triangles`` fall
    below :func:`min_rest_condition`, the source polygon indices they came from
    (sorted, deduplicated), which of those polygons re-splitting rescues, which
    of them are ALREADY triangles, and whether re-splitting rescues all of them.

    Three cases reach here and they take three different remedies, which is why
    the last two keys are separate:

      * every flagged polygon is repairable. The polygons are sound and only
        Blender's split of them is degenerate. A quad carrying a vertex on, or
        very near, the straight edge between its neighbors splits along the
        diagonal that produces a collinear triangle, while Face > Triangulate
        Faces (Ctrl+T) picks the other one. Nothing about the mesh changes.
      * a flagged polygon is already a TRIANGLE. It is its own only
        triangulation, so no split can help and none is offered. Its own
        geometry is the defect: a corner sits on the line between the other
        two. The vertex has to move, or the triangle be dissolved into a
        neighbor.
      * a flagged polygon of four or more corners has no sound split. A
        zero-area polygon has none at all, and neither does one with a
        zero-length boundary edge: every boundary edge belongs to exactly one
        triangle of any triangulation, so a pair of coincident consecutive
        vertices forces a degenerate triangle into all of them. That geometry
        has to be merged or dissolved.

    Shipping such a triangle costs the artist the run, in one of two ways that
    look nothing alike. An exactly zero-area one aborts at scene build with
    ``degenerate face {i}: area is zero (collinear or duplicate vertex
    indices)`` from ``triutils::face_areas``. A merely near-collinear one clears
    that assertion, inverts to a finite but enormous ``inv_rest``, and reaches
    the linear solve as a non-finite Hessian, which reports
    ``p^T A p is not-a-number at iter 0`` and names no geometry (issue #144).
    Both abort only after an upload and a build, and both name an index into
    the solver's own concatenated mesh, so the dynamics pipeline refuses the
    Transfer here instead, where the object still has a name and the artist has
    a repair.

    The test is the conditioning of the rest matrix, not an area threshold: it
    is the quantity that decides whether the inverse rest shape survives fp32,
    and a merely thin triangle above it is legitimate geometry the solver
    handles. The measurement is float64 through the object's world transform,
    the same positions the solver builds its rest shape from.

    This gate and ``builder::invert_rest_or_panic2`` use the same threshold and
    agree for a group with isotropic shrink, which is the default. They can
    diverge when ``shrink-x`` and ``shrink-y`` differ: the solver tests the
    matrix it actually inverts, after the UV rotation and the per-axis shrink,
    and this gate measures raw corner positions and never sees either. The
    solver's gate is the authoritative one, and refuses what it must.

    `local_verts` is the local-space positions the encoder will actually ship,
    which is the starting frame's deform-evaluated pose. Passing them is what
    makes this gate judge the geometry the solver receives rather than the base
    cage; the repair scan deliberately passes nothing, because a base-mesh
    re-split is all a repair can change.

    Returns an empty result for non-mesh objects (curves, etc.), which have no
    polygons in the Blender mesh sense.
    """
    empty = {
        "count": 0,
        "polygons": [],
        "repairable_polygons": [],
        "triangle_polygons": [],
        "all_repairable_by_triangulation": True,
    }
    if obj is None or obj.type != "MESH" or obj.data is None:
        return empty
    mesh = obj.data
    mesh.calc_loop_triangles()
    if not len(mesh.loop_triangles):
        return empty

    import numpy as np

    if local_verts is not None and len(local_verts) == len(mesh.vertices):
        # The positions the encoder will ship: the starting frame's
        # deform-evaluated pose. Judging the base cage instead lets a shape key
        # or an armature move a corner onto the line between its neighbors
        # after the gate has already passed it.
        co = np.asarray(local_verts, dtype=np.float64).reshape(-1, 3)
    else:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
    # The LOCAL corners are kept: a candidate triangulation is chosen in the
    # space the repair runs bmesh in, not in the measurement space below.
    co_local = co
    # Only the linear part of the transform: a translation cancels in the edge
    # vectors, while a non-uniform scale genuinely changes every triangle's
    # aspect ratio and so belongs in the measurement.
    linear = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
    co = co @ linear.T

    n_tri = len(mesh.loop_triangles)
    tri = np.empty(n_tri * 3, dtype=np.uint32)
    mesh.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(n_tri, 3)
    # `conditioning=False` is for triangles that never reach a rest-shape
    # inversion: a stationary STATIC collider and an fTetWild SOLID's surface
    # both go to `make_collision_mesh`, whose only per-triangle check is that
    # the area is positive. Refusing those on conditioning would reject
    # geometry the solver accepts.
    ratios = _triangle_conditioning(co[tri])
    floor = min_rest_condition() if conditioning else 0.0
    bad = np.flatnonzero(ratios <= floor if not conditioning else ratios < floor)
    if not len(bad):
        return empty

    poly_index = np.empty(n_tri, dtype=np.int32)
    mesh.loop_triangles.foreach_get("polygon_index", poly_index)
    polygons = sorted({int(p) for p in poly_index[bad]})

    # One scratch bmesh for every flagged polygon rather than one each: this
    # runs on the blocking main thread at Transfer, and a mesh with many
    # offenders paid an allocate and free per polygon.
    loops = [list(mesh.polygons[p].vertices) for p in polygons]
    fills = _beauty_triangulations([co_local[loop] for loop in loops])
    repairable = [
        p
        for p, loop, fill in zip(polygons, loops, fills)
        if _fill_is_sound(fill, co[loop])
    ]
    triangles = [p for p in polygons if len(mesh.polygons[p].vertices) == 3]
    return {
        "count": int(len(bad)),
        "polygons": polygons,
        # The subset re-splitting rescues, so a repair splits exactly those and
        # leaves the rest alone. Splitting a polygon that has no sound
        # triangulation changes the mesh without fixing anything: every
        # triangulation of it still contains a degenerate triangle.
        "repairable_polygons": repairable,
        # Already triangles, so no split exists to offer. Reported separately
        # because the remedy is a change to the geometry, not to its split.
        "triangle_polygons": triangles,
        "all_repairable_by_triangulation": len(repairable) == len(polygons),
    }


def find_linked_duplicate_siblings(obj) -> list[str]:
    """Return the names of other Blender objects that share *obj*'s data
    block (i.e. ``obj`` is a Linked Duplicate / shallow copy, typically
    created by Alt-D).

    The simulator works under the assumption that each object owns its
    own mesh data: shared data means a vertex coordinate written for
    one assigned object would silently propagate to its sibling, which
    in turn would cause the encoder to ship inconsistent geometry,
    corrupt PC2 playback, and break the topology hash. The dynamics
    pipeline rejects these objects up-front.

    Returns an empty list when ``obj`` has its own data block, or when
    ``obj.data`` is None (curves with no spline yet, etc.).
    """
    if obj is None or obj.data is None:
        return []
    # ``users`` counts every reference to the data block, including
    # the active object itself. A solo owner has ``users == 1``.
    if obj.data.users <= 1:
        return []
    return [
        o.name
        for o in bpy.data.objects
        if o is not obj and o.data is obj.data
    ]


def get_timer_wait_time():
    """Get the wait time for the timer."""
    return 0.25


def redraw_all_areas(context):
    """Tag all screen areas for redraw."""
    for area in context.screen.areas:
        area.tag_redraw()


def redraw_all_windows(area_type: str | None = None):
    """Tag areas across all windows for redraw, optionally by type.

    Iterates ``window_manager.windows`` (not ``context.screen``) so it is
    safe to call from the public Python API and from ``bpy.app.timers``
    callbacks where the active screen is unreliable or None. Pass
    ``area_type`` (e.g. ``"VIEW_3D"``) to limit the redraw to one editor type,
    or leave it None to tag every area.
    """
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area_type is None or area.type == area_type:
                area.tag_redraw()


def check_vec3(name: str, v, error_cls) -> tuple[float, float, float]:
    """Coerce `v` to a length-3 tuple of floats, raising `error_cls` on failure.

    Callers plug in their layer's exception type (MCPError, ValidationError,
    MutationError, ...) so the message vocabulary is shared but the error
    class stays specific to the boundary that raised.
    """
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise error_cls(f"{name} must be a length-3 list/tuple, got {v!r}")
    try:
        return tuple(float(x) for x in v)
    except (TypeError, ValueError) as e:
        raise error_cls(f"{name} components must be numeric: {e}")


def parse_vertex_index(data_path: str) -> int | None:
    """Parse the vertex index from a data path string."""
    start = data_path.find("[") + 1
    end = data_path.find("]")
    if start >= 0 and end > start:
        try:
            return int(data_path[start:end])
        except ValueError:
            pass
    return None


def _get_fcurves(action):
    """Get fcurves from an action (Blender 5.0+ layered API)."""
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                if bag.fcurves:
                    return bag.fcurves
    return []


_TRANSFORM_PATHS = (
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
)


def has_transform_fcurves(obj) -> bool:
    """True if *obj* has any object-level transform fcurve (loc/rot/scale).

    Used by the static-ops UI and encoder to enforce mutual exclusion:
    a static object with Blender keyframe animation cannot also use
    UI-assigned move/spin/scale ops (only one source of motion at a
    time).
    """
    if obj is None or not hasattr(obj, "animation_data"):
        return False
    for fc in get_id_fcurves(obj):
        path = getattr(fc, "data_path", "") or ""
        if any(path == p or path.endswith(f".{p}") for p in _TRANSFORM_PATHS):
            return True
    return False


# Modifier types whose evaluation can move mesh vertices off the
# rest pose. Picked by inspecting Blender's modifier categories; if
# any of these is in the stack AND enabled for the depsgraph, the
# evaluated mesh may differ from obj.data.vertices and a STATIC
# collider needs a Capture Deformation pass to feed the solver.
#
# ``NODES`` is included, but a Geometry Nodes modifier is NOT taken
# at face value: ``has_deforming_modifier_stack`` only counts it when
# its node group actually writes vertex positions (see
# ``_nodes_modifier_can_deform``). A Geometry Nodes modifier can do
# anything from full procedural deformation to a pure normal recompute
# (Blender 4.1+'s default "Smooth by Angle" node group, auto-added on
# most meshes, is the most common case and moves no vertices).
# Counting every NODES modifier as a deformer forces Capture
# Deformation on objects that don't need it; the position-write scan
# excludes those, and ``is_deforming_static_object`` keeps a
# depsgraph-sampling backstop for groups whose writers the scan
# doesn't enumerate.
_DEFORMING_MODIFIER_TYPES = frozenset({
    "ARMATURE",
    "CAST",
    "CLOTH",
    "CORRECTIVE_SMOOTH",
    "CURVE",
    "DISPLACE",
    "HOOK",
    "LAPLACIANDEFORM",
    "LAPLACIANSMOOTH",
    "LATTICE",
    "MESH_DEFORM",
    # Geometry Nodes can displace existing vertices (e.g. a Set Position
    # wave driven by Scene Time). It may also change topology, in which
    # case it is additionally treated as generative for MESH_CACHE
    # placement (see ``_GENERATIVE_MODIFIER_TYPES`` in ``core/pc2.py``);
    # the two classifications are independent. A position-writing GN is
    # treated as a deformer here (gated by ``_nodes_modifier_can_deform``)
    # so the pin overlay, Capture Deformation, and the encoder follow
    # GN-driven motion the same way they follow Armature/Lattice.
    "NODES",
    "SHRINKWRAP",
    "SIMPLE_DEFORM",
    "SMOOTH",
    "SOFT_BODY",
    "SURFACE_DEFORM",
    "VOLUME_DISPLACE",
    "WARP",
    "WAVE",
})


def _has_shape_key_animation(obj) -> bool:
    """True if *obj* has any shape-key value fcurve.

    Shape keys with animated `.value` deform the evaluated mesh even
    though the modifier list looks empty. The encoder must not silently
    ignore them.
    """
    if obj is None or obj.type != "MESH":
        return False
    sk = getattr(obj.data, "shape_keys", None)
    if sk is None:
        return False
    return bool(get_id_fcurves(sk))


# Geometry Nodes node types that can move existing mesh vertices off
# their rest position. A GN modifier whose tree contains none of these
# cannot deform the mesh (it may still recompute normals, set shade-
# smooth flags, assign attributes, ...), so it must not be treated as a
# deformer. Instance-transform nodes (Translate/Rotate/Scale Instances,
# Set Instance Transform) are intentionally excluded: they move
# instances, not the realized collider mesh's own vertices.
_GN_POSITION_WRITING_NODES = frozenset({
    "GeometryNodeSetPosition",
    "GeometryNodeTransform",
    "GeometryNodeDeformCurvesOnSurface",
})


def _nodes_modifier_can_deform(mod) -> bool:
    """True if a Geometry Nodes *mod* can move mesh vertices.

    Scans the modifier's node group, recursing into nested node
    groups, for any node that writes geometry position (see
    ``_GN_POSITION_WRITING_NODES``). A group with no such node, e.g.
    Blender's auto-added "Smooth by Angle" (which only sets shade-
    smooth flags), leaves every vertex at rest and is not a deformer.

    Conservative on failure: if the node group can't be introspected,
    returns True so the caller errs toward treating it as deforming
    (``is_deforming_static_object``'s depsgraph backstop then settles
    it per-frame for the one path that hard-fails on a missing cache).
    """
    ng = getattr(mod, "node_group", None)
    if ng is None:
        return False
    try:
        seen = set()
        stack = [ng]
        while stack:
            tree = stack.pop()
            if tree is None or tree.as_pointer() in seen:
                continue
            seen.add(tree.as_pointer())
            for node in tree.nodes:
                if node.bl_idname in _GN_POSITION_WRITING_NODES:
                    return True
                nested = getattr(node, "node_tree", None)
                if nested is not None:
                    stack.append(nested)
        return False
    except Exception:
        return True


def has_deforming_modifier_stack(obj) -> bool:
    """True if *obj*'s modifier stack contains any vertex-moving deformer.

    Cheap declarative check (no depsgraph round-trip, except a node-tree
    scan for Geometry Nodes modifiers). Only modifiers enabled for
    ``show_viewport`` count — a muted deformer doesn't contribute to what
    Blender or the depsgraph sees.

    A Geometry Nodes modifier counts only when its node group actually
    writes vertex positions: the ubiquitous auto-added "Smooth by Angle"
    group moves nothing and must not be mistaken for a deformer.
    """
    if obj is None or obj.type != "MESH":
        return False
    if not hasattr(obj, "modifiers"):
        return False
    for mod in obj.modifiers:
        if not getattr(mod, "show_viewport", True):
            continue
        if mod.type not in _DEFORMING_MODIFIER_TYPES:
            continue
        if mod.type == "NODES" and not _nodes_modifier_can_deform(mod):
            continue
        return True
    return False


def eval_deform_local_positions(obj, context=None, exclude_modifier_name=None,
                                exclude_modifier_names=()):
    """Return ``(N, 3)`` float32 local-space vertex positions of *obj* in the
    pose the solver starts from, evaluated at the current frame, or ``None``
    when no such pose keeps the base mesh's vertex count.

    Honors Geometry Nodes / Armature / Lattice deforms so callers can
    capture the shape the artist sees, instead of the undeformed rest
    cage. ``exclude_modifier_names`` (see
    ``pc2.display_only_modifier_names``) are temporarily hidden during
    evaluation, and ``exclude_modifier_name`` hides one more the same way:
    the ``ContactSolverCache`` MESH_CACHE among them replays prior solver
    output with OVERWRITE, so reading it back would feed the output into
    the next input.

    When the stack changes the vertex count, it is evaluated again with the
    stack cut where the ContactSolverCache is placed: the first
    topology-changing modifier and every modifier after it are hidden too
    (``pc2.modifiers_after_cache_boundary``). On display the cache replaces
    the output of everything in front of it, and the modifiers after it run
    on top of the solver output, so the cut stack is exactly the pose the
    solver's output stands in for. An Armature or Lattice in front of a
    Subdivision therefore reaches the result, while one after it does not,
    since it deforms the simulated mesh on display. A Geometry Nodes
    modifier in a stack that changes the vertex count is a cut point, as it
    is for the cache placement.

    Returns ``None`` in two cases: *obj* is not a mesh, or the cut stack
    still changes the vertex count, which takes a modifier in front of the
    cut that changes it without being one the cache placement counts as
    topology-changing (a Fluid domain, an Ocean in Generate mode, a Mesh
    Sequence Cache whose topology differs from the mesh). A base
    triangulation or vertex-group index would not line up with the result
    in either case. Anything that fails during evaluation propagates: every
    caller runs where the depsgraph can be evaluated.
    """
    import numpy as np

    from .pc2 import modifiers_after_cache_boundary

    if obj is None or obj.type != "MESH":
        return None
    if context is None:
        context = bpy.context
    n_base = len(obj.data.vertices)
    excluded = set(exclude_modifier_names)
    if exclude_modifier_name:
        excluded.add(exclude_modifier_name)
    toggled = []

    def hide(modifiers):
        for m in modifiers:
            if m.show_viewport:
                m.show_viewport = False
                toggled.append(m)

    def evaluate():
        eval_obj = obj.evaluated_get(context.evaluated_depsgraph_get())
        eval_mesh = eval_obj.to_mesh()
        try:
            n = len(eval_mesh.vertices)
            if n != n_base:
                return None
            co = np.empty(n * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", co)
            return co.reshape(n, 3)
        finally:
            eval_obj.to_mesh_clear()

    try:
        hide(m for m in obj.modifiers if m.name in excluded)
        co = evaluate()
        if co is None:
            hide(modifiers_after_cache_boundary(obj))
            co = evaluate()
        return co
    finally:
        for m in toggled:
            m.show_viewport = True


def validate_bend_reference(source_obj, ref_obj, context=None, group_type="SHELL"):
    """Validate that *ref_obj* is a positions-only topological copy of
    *source_obj*, usable as a bending rest-angle reference.

    The check matches how each group type ships geometry:

    * SHELL and mesh ROD: evaluate ``ref_obj`` through its full modifier /
      geometry-nodes stack and compare the result against ``source_obj``'s
      base mesh. The evaluated reference must have the same vertex count and
      identical connectivity (faces for SHELL, edges for ROD); only vertex
      positions may differ.
    * Curve ROD: sample both curves the way the encoder does (control-point
      level, ``sample_curve``) and compare sampled vertex count + edges.
      Curve modifiers / geometry nodes are not sampled, so a curve reference
      must move its control points (directly, or via a modifier baked into
      the control points), mirroring how the source curve rod is shipped.

    Returns ``(True, "")`` on success, or ``(False, message)`` with a
    user-facing error describing the first mismatch found.
    """
    import numpy as np

    if context is None:
        context = bpy.context
    if source_obj is None:
        return False, "Source object not found."
    if ref_obj is None:
        return False, "Reference object not found."
    if ref_obj == source_obj:
        return False, "The reference object must be different from the object itself."
    if ref_obj.type != source_obj.type:
        return False, (
            f"Reference '{ref_obj.name}' is a {ref_obj.type.title()} but "
            f"'{source_obj.name}' is a {source_obj.type.title()}; the "
            f"reference must be the same object type."
        )

    # Curve rod: compare sampled rod vertices (control-point level).
    if group_type == "ROD" and source_obj.type == "CURVE":
        from mathutils import Matrix  # pyright: ignore
        from .curve_rod import sample_curve
        src_v, src_e, _ = sample_curve(source_obj, Matrix.Identity(4))
        ref_v, ref_e, _ = sample_curve(ref_obj, Matrix.Identity(4))
        if len(ref_v) != len(src_v):
            return False, (
                f"Reference '{ref_obj.name}' samples to {len(ref_v)} rod "
                f"vertices but '{source_obj.name}' samples to {len(src_v)}. A "
                f"reference curve must have the same spline structure (only "
                f"control-point positions may change)."
            )
        if not np.array_equal(np.asarray(ref_e), np.asarray(src_e)):
            return False, (
                f"Reference '{ref_obj.name}' has different rod connectivity "
                f"than '{source_obj.name}'. Only control-point positions may "
                f"change in a reference curve."
            )
        return True, ""

    if source_obj.type != "MESH":
        return False, (
            f"'{source_obj.name}' is neither a mesh nor a curve rod; "
            f"reference rest angles are not supported for it."
        )

    is_rod = group_type == "ROD"
    src_mesh = source_obj.data
    n_src = len(src_mesh.vertices)
    if is_rod:
        src_conn = [tuple(sorted(e.vertices)) for e in src_mesh.edges]
        conn_label = "edge"
    else:
        src_conn = [tuple(p.vertices) for p in src_mesh.polygons]
        conn_label = "face"

    deps = context.evaluated_depsgraph_get()
    eval_obj = ref_obj.evaluated_get(deps)
    eval_mesh = eval_obj.to_mesh()
    try:
        n_ref = len(eval_mesh.vertices)
        if n_ref != n_src:
            return False, (
                f"Reference '{ref_obj.name}' has {n_ref} vertices after "
                f"evaluating its modifiers / geometry nodes, but "
                f"'{source_obj.name}' has {n_src}. A reference must be a "
                f"topological copy with only vertex positions changed."
            )
        if is_rod:
            ref_conn = [tuple(sorted(e.vertices)) for e in eval_mesh.edges]
        else:
            ref_conn = [tuple(p.vertices) for p in eval_mesh.polygons]
    finally:
        eval_obj.to_mesh_clear()

    if ref_conn != src_conn:
        return False, (
            f"Reference '{ref_obj.name}' has different {conn_label} "
            f"connectivity than '{source_obj.name}' after evaluation. Only "
            f"vertex positions may change in a reference object."
        )
    return True, ""


def eval_reference_local_positions(ref_obj, context=None):
    """Return ``(N, 3)`` float32 local-space vertex positions of *ref_obj*
    with its FULL modifier / geometry-nodes stack evaluated at the current
    frame, or ``None`` when evaluation isn't available.

    Unlike :func:`eval_deform_local_positions`, this does NOT gate on the
    evaluated count matching the object's own base mesh: a bending
    reference is validated against the SOURCE object's count by the
    caller (see :func:`validate_bend_reference`), so the count check
    belongs there, not here.
    """
    import numpy as np

    if ref_obj is None or ref_obj.type != "MESH":
        return None
    if context is None:
        context = bpy.context
    deps = context.evaluated_depsgraph_get()
    eval_obj = ref_obj.evaluated_get(deps)
    eval_mesh = eval_obj.to_mesh()
    try:
        n = len(eval_mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        eval_mesh.vertices.foreach_get("co", co)
        return co.reshape(n, 3)
    finally:
        eval_obj.to_mesh_clear()


def _depsgraph_mesh_differs_across_range(obj, context) -> bool:
    """Compare depsgraph-evaluated mesh *shape* at ``frame_start`` and
    ``frame_end``. Returns True if any vertex moves in the object's
    LOCAL space between the two samples.

    Local space is the right comparison: the goal is to detect mesh
    SHAPE changes (drivers poking vertex coords, geometry-node
    deformation that fell through the modifier-name list) and let
    rigid object-transform animation be handled separately by
    ``transform_animation``. A previous version of this function
    compared world-space coords, which conflated own-transform
    fcurves with deformation and forced Capture Deformation on
    rigid-only STATIC objects.
    """
    import numpy as np

    from .pc2 import resume_mesh_cache_display, suspend_mesh_cache_display

    scene = context.scene
    if scene.frame_end <= scene.frame_start:
        return False
    saved = scene.frame_current
    # The addon's own ContactSolverCache (a MESH_CACHE with
    # deform_mode='OVERWRITE') replays the PREVIOUS solver output, so with it
    # enabled the evaluated mesh appears to change across the timeline even when
    # the object is rigid. Suspend it for the duration of the two samples so
    # this measures genuine deformer output only. Otherwise a moving STATIC
    # collider that carries a cache (e.g. one previously simulated, or just
    # replaying results) is misread as deforming and wrongly forced to Capture
    # Deformation. Restored in the finally below.
    cache_prior = suspend_mesh_cache_display(obj)
    try:
        def _sample(f):
            scene.frame_set(int(f))
            dg = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(dg)
            eval_mesh = eval_obj.to_mesh()
            try:
                n = len(eval_mesh.vertices)
                if n == 0:
                    return None
                co = np.empty((n, 3), dtype=np.float32)
                eval_mesh.vertices.foreach_get("co", co.ravel())
                return co
            finally:
                eval_obj.to_mesh_clear()

        a = _sample(scene.frame_start)
        b = _sample(scene.frame_end)
    finally:
        scene.frame_set(saved)
        resume_mesh_cache_display(obj, cache_prior)
    if a is None or b is None or a.shape != b.shape:
        return False
    return bool(np.any(np.abs(a - b) > 1e-6))


def _matrix_world_differs_without_own_fcurves(obj, context) -> bool:
    """Detect motion that comes from a parent / constraint / driver
    rather than the object's own location/rotation/scale fcurves.

    ``transform_animation`` samples ``obj.matrix_world`` only at the
    object's own fcurve keyframes; if there are no own fcurves but
    the world matrix still changes across the timeline, the encoder
    would silently drop the motion. Surface that case so the user
    knows to Capture Deformation (which bakes the evaluated motion
    from every source).

    Returns False when the object DOES have its own loc/rot/scale
    fcurves: the encoder's transform-keyframe sampler covers those,
    so the rigid path is sufficient.
    """
    import numpy as np

    if has_transform_fcurves(obj):
        return False
    scene = context.scene
    if scene.frame_end <= scene.frame_start:
        return False
    saved = scene.frame_current
    try:
        def _mw(f):
            scene.frame_set(int(f))
            dg = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(dg)
            return np.array(eval_obj.matrix_world, dtype=np.float64)
        a = _mw(scene.frame_start)
        b = _mw(scene.frame_end)
    finally:
        scene.frame_set(saved)
    return bool(np.any(np.abs(a - b) > 1e-6))


def _has_nonfcurve_motion_source(obj) -> bool:
    """Cheap (no depsgraph, no writes) check for a motion source the rigid
    own-fcurve ``transform_animation`` path can't capture: a parent, a
    constraint, a transform driver, or an NLA track. Used by the UI to keep
    the Capture Deformation button reachable without running the depsgraph
    sampler on every redraw; the encoder's full check is the authoritative
    gate."""
    if obj is None:
        return False
    if obj.parent is not None:
        return True
    if len(getattr(obj, "constraints", ())) > 0:
        return True
    ad = getattr(obj, "animation_data", None)
    if ad is not None:
        for d in ad.drivers:
            dp = getattr(d, "data_path", "") or ""
            if any(t in dp for t in ("location", "rotation", "scale")):
                return True
        if len(ad.nla_tracks) > 0:
            return True
    return False


def _mesh_shape_could_animate(obj) -> bool:
    """Cheap (no depsgraph, no writes) over-approximation of whether *obj*'s
    evaluated mesh SHAPE can change across the timeline.

    Returns True whenever some source could move this mesh's vertices frame
    to frame: any modifier (a deform/generative modifier, or one whose
    parameters are animated or driven), shape keys (their values can be
    keyed or driven), mesh-level animation data (drivers on vertex coords),
    or a parent (parent-relative armature/lattice deform). A fully inert
    rigid mesh, none of the above, returns False; in that case the
    evaluated mesh shape is provably frame-invariant, so the caller can
    skip ``_depsgraph_mesh_differs_across_range`` and its two whole-scene
    frame evaluations.

    Own loc/rot/scale animation (``obj.animation_data`` action fcurves on a
    mesh with no modifier/shape-key/parent) is deliberately NOT a trigger:
    it moves the object transform, not the mesh shape, and is handled by
    the rigid ``transform_animation`` path, not the deform path.
    """
    if obj is None or obj.type != "MESH":
        return False
    if len(getattr(obj, "modifiers", ())) > 0:
        return True
    if obj.parent is not None:
        return True
    mesh = obj.data
    if mesh is not None:
        if getattr(mesh, "shape_keys", None) is not None:
            return True
        if getattr(mesh, "animation_data", None) is not None:
            return True
    return False


def static_mesh_deforms(obj, context, allow_eval: bool = True) -> bool:
    """True if *obj*'s mesh changes SHAPE over the timeline.

    What the encoder requires a Capture Deformation for: a rigid transform
    cannot carry it. Three-tier detection:
      1. Declarative: deforming modifier stack.
      2. Shape-key animation.
      3. Local-space mesh shape change across the timeline (catches
         driver-only and geometry-node deformation).

    Motion of the object as a whole, from its own curves or from a parent,
    constraint, driver or NLA, is not deformation: the encoder ships it as
    the world matrix sampled at every solve frame
    (:func:`sample_transform_animation`).

    ``allow_eval`` gates tier 3, which samples the depsgraph (it calls
    ``scene.frame_set`` and temporarily toggles the ContactSolverCache
    modifier). That mutates scene state and must NOT run from a restricted
    context such as a UI ``draw()`` handler, where Blender forbids ID writes
    and per-redraw frame stepping would be unusable. Pass ``allow_eval=False``
    from draw to get the cheap declarative tiers only; the encoder leaves it
    True for the full, authoritative gate.
    """
    if obj is None or obj.type != "MESH":
        return False
    if has_deforming_modifier_stack(obj):
        return True
    if _has_shape_key_animation(obj):
        return True
    if context is None or not allow_eval:
        return False
    # Tier 3 steps the timeline twice (``scene.frame_set`` at frame_start and
    # frame_end) and re-evaluates the whole-scene depsgraph, so it is gated
    # behind a cheap, no-depsgraph pre-check that is a strict SUPERSET of the
    # motion it can detect: it reports a LOCAL mesh-shape change only if some
    # animatable source can move this mesh's verts
    # (``_mesh_shape_could_animate``). If the pre-check is False the sampler
    # is provably False, so skipping it changes no result, only cost.
    return bool(
        _mesh_shape_could_animate(obj)
        and _depsgraph_mesh_differs_across_range(obj, context)
    )


def is_deforming_static_object(obj, context, allow_eval: bool = True) -> bool:
    """True if Capture Deformation can record motion *obj* has.

    Either its mesh changes shape (:func:`static_mesh_deforms`), which the
    encoder requires a capture for, or the object as a whole is moved by
    something other than its own curves: its ``matrix_world`` changes across
    the timeline and a parent, constraint, transform driver or NLA strip
    can move it. The second ships without a capture as sampled world
    matrices, and a capture of it is accepted in their place.

    ``allow_eval`` is :func:`static_mesh_deforms`'s: False keeps to the
    declarative tiers, for a ``draw()`` handler.
    """
    if static_mesh_deforms(obj, context, allow_eval):
        return True
    if obj is None or obj.type != "MESH" or context is None or not allow_eval:
        return False
    # Sampled only behind the cheap pre-check: a world-matrix change from a
    # non-own-fcurve source (parent / constraint / transform driver / NLA) is
    # exactly what ``_has_nonfcurve_motion_source`` reports, so a False there
    # proves the sampler False.
    return bool(
        _has_nonfcurve_motion_source(obj)
        and _matrix_world_differs_without_own_fcurves(obj, context)
    )


def get_vertices_in_group(obj, vg) -> list[int]:
    """Return vertex indices belonging to the given vertex group.

    For MESH objects, reads from Blender vertex groups.
    For CURVE objects, reads from custom property ``_pin_{vg.name}``.

    Args:
        obj: Blender object (MESH or CURVE).
        vg: Blender vertex group (or object with .name for curve lookup).

    Returns:
        List of vertex indices that belong to *vg*.
    """
    if obj.type == "CURVE":
        import json
        key = f"_pin_{vg.name}"
        raw = obj.get(key)
        if raw:
            return json.loads(raw)
        return []
    indices = []
    if not hasattr(obj.data, "vertices"):
        return indices
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                indices.append(v.index)
                break
    return indices


def pin_covers_all_vertices(obj, vg_name) -> bool:
    """True when the pin's vertex group includes EVERY vertex of the mesh.

    "Track Rest-Pose Deformation" requires such a full pin: with every vertex
    captured, the rest pose IS the captured deformation (the solver drives all
    sim vertices), so no partial-pin reconstruction is needed. A partial pin
    would leave the unpinned region at the undeformed rest and tear the
    boundary, so the feature is gated off for it. Mirrors the decoder's
    ``full_pin`` test (``len(pinned) == n_blender``).

    Implemented as a plain count match: the group's member count equals the
    mesh vertex count. Blender exposes no O(1) vertex-group count, so this scans
    the mesh (O(n)). The panel does not call this every redraw; the Refresh
    button next to the rest-pose toggle runs it on demand and caches the result
    on the pin (full_pin_checked / full_pin_cached). The encoder still calls it
    directly at encode time as the source-of-truth gate.
    """
    if obj is None or getattr(obj, "type", None) != "MESH" or not vg_name:
        return False
    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "vertices"):
        return False
    vg = obj.vertex_groups.get(vg_name)
    if vg is None:
        return False
    n_total = len(data.vertices)
    return n_total > 0 and len(get_vertices_in_group(obj, vg)) == n_total


def set_linear_interpolation(action):
    """Set LINEAR interpolation on all keyframe points in *action*.

    Args:
        action: Blender action containing fcurves.
    """
    for fc in _get_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def get_moving_vertex_indices(obj, exclude=None) -> list[int]:
    from .pc2 import has_mesh_cache

    if exclude is None:
        exclude = []
    # MESH_CACHE modifier means all vertices are animated
    if obj and obj.type == "MESH" and has_mesh_cache(obj):
        return [i for i in range(len(obj.data.vertices)) if i not in exclude]
    return []


def get_pin_vertex_indices(obj, context, frame: int | None = None) -> list[int]:
    """List vertex indices that are pinned (active) at the given frame.

    Args:
        obj: Blender mesh object.
        context: Blender context.
        frame: Current frame number. If given, pins with duration that have
            expired by this frame are excluded (their vertices are no longer
            considered pinned). If ``None``, all pin vertices are returned
            regardless of duration.
    """
    indices = set()
    if obj and hasattr(obj, "vertex_groups") and hasattr(obj.data, "vertices"):
        pin_vg_names = set()

        from .uuid_registry import get_object_uuid
        _obj_uid = get_object_uuid(obj)
        for group in iterate_active_object_groups(context.scene):
            if hasattr(group, "pin_vertex_groups"):
                from .uuid_registry import resolve_pin
                for pin_item in group.pin_vertex_groups:
                    resolve_pin(pin_item)
                    if pin_item.object_uuid != _obj_uid:
                        continue
                    _, vg_name = decode_vertex_group_identifier(pin_item.name)
                    if vg_name:
                        # Pull pins are not hard-pinned — exclude them
                        if pin_item.use_pull:
                            continue
                        # Pins with explicit operations (spin/scale/move_by)
                        # move during simulation — exclude them
                        if any(op.op_type in ("SPIN", "SCALE", "MOVE_BY", "TORQUE") for op in pin_item.operations):
                            continue
                        # If frame is given, skip expired duration-limited pins
                        if frame is not None and pin_item.use_pin_duration:
                            if frame > pin_item.pin_duration:
                                continue
                        pin_vg_names.add(vg_name)

        for vg_name in pin_vg_names:
            vg = obj.vertex_groups.get(vg_name)
            if vg:
                for idx in get_vertices_in_group(obj, vg):
                    indices.add(idx)

    return list(indices)


def get_id_fcurves(id_block) -> list:
    """The F-curves animating *id_block*, read from the slot it is assigned.

    Blender 5.x keeps curves under ``action.layers[].strips[].channelbag(slot)``,
    and one action can hold a bag per slot, each animating a different ID. The
    bag that animates *id_block* is the one for ``animation_data.action_slot``;
    taking the first non-empty bag instead reads another ID's curves off a
    shared action. A legacy (unslotted) action keeps them on the action.
    """
    ad = getattr(id_block, "animation_data", None)
    action = getattr(ad, "action", None) if ad is not None else None
    if action is None:
        return []
    out = []
    slot = getattr(ad, "action_slot", None)
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot) if slot is not None else None
            if bag is not None:
                out.extend(bag.fcurves)
    out.extend(getattr(action, "fcurves", []) or [])
    return out


def _may_move(obj) -> bool:
    """True when anything could change *obj*'s world transform over time.

    Its own transform curves, a driver, an NLA track or a constraint, on it or
    on any parent up its chain. A collider carried by an animated parent moves
    in the viewport without a curve of its own, and treating only its own
    curves as motion shipped it to the solver frozen at its rest pose.
    """
    seen = set()
    o = obj
    while o is not None and o.name not in seen:
        seen.add(o.name)
        ad = getattr(o, "animation_data", None)
        if ad is not None and (
            get_id_fcurves(o) or list(ad.nla_tracks) or list(ad.drivers)
        ):
            return True
        if len(getattr(o, "constraints", ())) > 0:
            return True
        o = o.parent
    return False


def sample_transform_animation(objs, context, start_frame: int,
                               frame_count: int) -> dict:
    """Every STATIC object's world transform at every frame of the solve.

    Returns ``{obj.name: {"frame_offset", "translation", "quaternion",
    "scale", "segments"}}`` for each object in *objs* whose transform
    actually changes over the solve, sampled at frame offsets ``0 ..
    frame_count - 1`` from *start_frame* with a LINEAR segment between
    consecutive samples. An object that could move but does not is left out.

    WHY EVERY FRAME RATHER THAN THE KEYS. Blender evaluates each channel with
    its own interpolation, easing and handles, applies parents, constraints,
    drivers and NLA, and places keys between frames; the solver evaluates one
    shared interpolation between the samples it is sent. Sampling Blender's
    evaluated world matrix at every solve frame makes the two agree at every
    frame the solve reaches, whatever the animation is built from. The one
    residual limit is a turn of more than 180 degrees within a single frame,
    which no sampling can tell from the shorter turn the other way.

    One ``scene.frame_set`` per frame serves every object, and the scene's
    current frame is restored afterward.
    """
    objs = list(objs)
    if not objs:
        return {}
    for obj in objs:
        data = getattr(obj, "data", None)
        if data is not None and get_id_fcurves(data):
            raise RuntimeError(
                f"STATIC object '{obj.name}' has mesh-level animation "
                "(shape keys). Only object-level transform animation is "
                "supported for STATIC objects; use Capture Deformation."
            )
    from mathutils import Matrix  # pyright: ignore

    scene = context.scene
    current_frame = scene.frame_current
    n = max(1, int(frame_count))
    samples = {obj.name: ([], [], [], []) for obj in objs}
    sheared = {}
    try:
        for k in range(n):
            scene.frame_set(int(start_frame) + k)
            for obj in objs:
                matrix = world_matrix(obj)
                loc, quat, scale = matrix.decompose()
                # A location, a rotation and a scale are all the solver can
                # carry. A rotated parent with a non-uniform scale gives a
                # sheared world matrix, which decompose() would silently
                # approximate; it is refused below if the object moves.
                if obj.name not in sheared:
                    rebuilt = Matrix.LocRotScale(loc, quat, scale)
                    size = max(1.0, max(abs(v) for row in matrix for v in row))
                    if max(
                        abs(a - b)
                        for ra, rb in zip(matrix, rebuilt)
                        for a, b in zip(ra, rb)
                    ) > 1e-5 * size:
                        sheared[obj.name] = int(start_frame) + k
                times, translations, quaternions, scales = samples[obj.name]
                times.append(float(k))
                translations.append([float(loc.x), float(loc.y), float(loc.z)])
                quaternions.append(
                    [float(quat.w), float(quat.x), float(quat.y), float(quat.z)]
                )
                scales.append([float(scale.x), float(scale.y), float(scale.z)])
    finally:
        scene.frame_set(current_frame)

    out = {}
    for name, (times, translations, quaternions, scales) in samples.items():
        first = (translations[0], quaternions[0], scales[0])
        moves = any(
            (t, q, s) != first
            for t, q, s in zip(translations, quaternions, scales)
        )
        if not moves:
            continue
        if name in sheared:
            raise ValueError(
                f"STATIC object '{name}' is sheared at frame {sheared[name]} "
                "(a rotated parent with a non-uniform scale), which a moving "
                "collider's location, rotation and scale cannot carry. Click "
                "'Capture Deformation' on its row, or give the parent a "
                "uniform scale."
            )
        out[name] = {
            "frame_offset": times,
            "translation": translations,
            "quaternion": quaternions,
            "scale": scales,
            "segments": [
                {
                    "interpolation": "LINEAR",
                    "handle_right": [1.0 / 3.0, 0.0],
                    "handle_left": [2.0 / 3.0, 1.0],
                }
                for _ in range(len(times) - 1)
            ],
        }
    return out


def get_transform_keyframes(
    obj, context, start_frame: int = 1, frame_count: int | None = None,
) -> dict | None:
    """One STATIC object's transform animation over the solve, or None.

    :func:`sample_transform_animation` for a single object that
    :func:`_may_move`, sampled at every frame offset from *start_frame* over
    *frame_count* frames (the scene's Frame Count when omitted). ``None`` when
    nothing moves it. Offsets are FRAME offsets; the decoder derives seconds
    from the Param payload's fps, keeping the data payload timing-free.
    """
    if obj is None or not _may_move(obj):
        return None
    if frame_count is None:
        from ..models.groups import get_addon_data
        frame_count = get_addon_data(context.scene).state.frame_count
    return sample_transform_animation(
        [obj], context, start_frame, frame_count,
    ).get(obj.name)
