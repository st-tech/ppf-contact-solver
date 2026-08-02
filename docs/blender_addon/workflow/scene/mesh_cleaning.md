# 🧹 Mesh Cleaning

**Mesh Cleaning** reads the meshes you have selected, reports the geometry
that makes a **Transfer** fail or a solve stop, and offers a targeted
repair for each finding. Scanning never writes to a mesh; every repair is
a separate button press on top of it.

Reach for it before the first **Transfer** on geometry you did not author
yourself: an imported character, a downloaded prop, a garment exported
from another application, or anything that has been through a boolean, a
mirror, or a merge. The [**Accepted geometry**](index.md) note describes
what the solver takes as input; this page is how you get a mesh there.

This page covers:

- [Where it lives](#where-it-lives)
- [Scanning a selection](#scanning-a-selection): the read-only pass and
  its two thresholds
- [What the scan reports](#what-the-scan-reports): errors, notes, and why
  an open quad panel is a note
- [The repairs](#the-repairs): six buttons, what each one fixes
- [Symmetric Triangulate](#symmetric-triangulate)
- [What a vertex-count change invalidates](#what-a-vertex-count-change-invalidates):
  read this before running one of the three repairs that move the count
- [Python / MCP API](#python--mcp-api)

## Where It Lives

Open the sidebar (<kbd>N</kbd>) in the 3D viewport, switch to the add-on
tab, and expand the **Utility Tools** panel. It sits between **Snap and
Merge** and **Visualization** and is **collapsed by default**. The panel
holds two boxes: **Mesh Cleaning**, and
[**Symmetric Triangulate**](#symmetric-triangulate) below it.

Everything here works in **Object Mode** on the **selected mesh objects**.
In Edit Mode, or with no mesh selected, the buttons are grayed out and a
label under them names the missing precondition (*"Switch to Object
Mode"*, *"Select one or more mesh objects"*). With a valid selection the
label reads *"N mesh object(s) selected"*.

:::{important}
**Every repair button applies to the whole selection, not just the object
whose report it sits under.** The report is drawn one box per selected
object, so it is easy to read a fix button as belonging to the box above
it. It does not: pressing it repairs every selected mesh. When you want a
fix on one object only, select that object alone first.
:::

## Scanning a Selection

Select the mesh objects you want to check and press **Scan Selected
Meshes**. The status line reports either *"Scanned N mesh object(s): no
defects found"* or *"Scanned N mesh object(s): M need attention"*, and the
panel fills with one box per object listing what was found. Objects that
are selected but have not been scanned are counted in a *"N object(s) not
scanned yet"* line at the bottom.

Only [error-class](#what-the-scan-reports) findings count toward *need
attention*. An object with notes and no errors reports **No defects
found** and still lists its notes below.

### The Two Thresholds

Two numbers control the scan. They live on the operator rather than in the
scene, so you set them the way you set any Blender operator's options:
run the scan, then press <kbd>F9</kbd> (**Adjust Last Operation**) and
edit them in the popup. The scan re-runs as you change them.

| Field               | Default  | Unit                     | What it does                                                                                     |
| ------------------- | -------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Merge Distance**  | `0.0001` | local mesh units         | Vertex pairs closer together than this are reported as near-coincident. Matches Blender's own **Merge by Distance** default, so the scan agrees with what **Mesh > Merge > By Distance** would weld. |
| **Degenerate Area** | `0`      | local units squared      | Faces at or below this area are reported as degenerate. At the default, only exactly zero-area faces are reported. |

A fix button drawn inside a report carries the distance that scan used, so
the repair covers exactly what the report described rather than reverting
to the operator's own default.

:::{note}
**A report is dropped when a fix button runs on the object, and whenever
the object's vertex or polygon count no longer matches the counts the
scan was taken at.** Between them those two rules cover every repair on
this page, including the ones that leave both counts alone.

What they do not cover is a count-preserving hand edit. Dragging a vertex
in Edit Mode leaves both counts intact, so the report stays on screen
even though it may no longer be right: two vertices you nudged together
are a near-coincident pair the standing report does not list. Re-scan
after you edit a shape by hand.
:::

## What the Scan Reports

The report is split by severity, and the split is the reason the tool is
usable on cloth at all.

**Errors** are geometry that stops a **Transfer** or a solve. Each one is
drawn with a warning icon and, where a safe repair exists, its fix button:

| Finding                                     | What it is                                                                  | Fix offered              | Vertex count |
| ------------------------------------------- | --------------------------------------------------------------------------- | ------------------------ | ------------ |
| **Linked Duplicate of `<name>`**            | The object shares its mesh data with another object (an <kbd>Alt</kbd>+<kbd>D</kbd> copy). | none, see below          | n/a          |
| **Near-coincident vertex pair(s)**          | Two vertices closer than the **Merge Distance**, reported with the smallest gap in both local and world units. | **Merge by Distance**    | changes      |
| **Isolated vertex(es), in no face**         | Points that belong to no face, usually left behind in an imported model.    | **Remove Loose Vertices** | changes      |
| **Hanging seam vertex(es)**                 | Points sitting partway along a sewing (loose) edge, belonging to no face.   | **Remove Loose Vertices** | changes      |
| **Duplicate face(s)**                       | Two faces sharing the same full vertex set.                                 | **Delete Duplicate Faces** | unchanged  |
| **Degenerate (zero-area) face(s)**          | Faces at or below the **Degenerate Area**.                                  | **Dissolve Degenerate**  | changes      |
| **Inconsistently wound edge(s)**            | Neighboring faces that traverse a shared edge in the same direction, so one of them faces the wrong way. | **Recalculate Outside**  | unchanged    |

A **Linked Duplicate** is the one error with no button. The solver assumes
each object owns its own mesh data, and repairing shared data here would
edit every object using it, so the report points you at **Object >
Relations > Make Single User > Object & Data** instead.

**Notes** are geometry worth knowing about that is not by itself a
problem. They carry an info icon and never make an object read as needing
attention:

| Finding                                | What it means                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Boundary edge(s), surface is open**  | The surface has an open border. Fine for a **Shell**; a **Solid** needs a closed surface to tell inside from outside. |
| **Non-manifold edge(s)**               | More than two faces meet along one edge.                                                                       |
| **Face(s) with more than 3 corners**   | Quads and n-gons, reported with the largest fold angle across the quad's diagonal and how many are past the flip boundary. Offers **Triangulate**. |

:::{note}
**An open panel made of quads is ordinary cloth, and the scan says so.** A
flat garment panel, a curtain, a flag: every one of these is an open
surface built from quads, so the boundary-edge row and the
more-than-3-corners row appear on almost every **Shell** you will ever
simulate. They are reported as notes precisely so they do not bury the
errors that actually stop a run, and there is nothing to fix. Closing a
boundary is a modeling decision, not a cleanup, so the tool offers no
button for it.
:::

Four of the checks (isolated vertices, hanging seam vertices, duplicate
faces, linked duplicates) are the same checks **Transfer** runs, so a
clean scan and a passing Transfer cannot disagree about them. The rest
(near-coincident vertices, degenerate faces, surface integrity, quad
re-splitting) are performed nowhere else in the pipeline.

:::{note}
**A Sand particle mesh is scanned for near-coincident vertices only.** A
committed particle mesh is a cloud of loose grain centers, so every vertex
legitimately belongs to no face and the face-based checks do not apply to
it. Near-coincident grain centers are still worth catching, and are in
fact the finding that matters most for grains.
:::

## The Repairs

Each repair acts on the selected meshes and reports what it did per
object in the status line (*"Merged 12 on Shirt (12). Re-scan to
confirm."*). A repair that finds nothing to do reports *"Nothing to
repair on the selected meshes."* and cancels without touching anything.
Objects sharing one mesh datablock are repaired once, not once per object
using it.

Three of the six change the vertex count and ask for confirmation before
they write; see
[What a vertex-count change invalidates](#what-a-vertex-count-change-invalidates).

### Merge by Distance

Welds vertex pairs closer together than the **Merge Distance** into one
vertex. **Changes the vertex count.**

Two vertices separated by a hair are the worst geometry you can hand this
solver: the pair sits far inside the contact gap, the contact response
becomes enormously stiff there, and the solve stops on an internal
consistency check that names no geometry at all, which makes the cause
very hard to find from the log. Welding them is what makes such a mesh
simulable.

The confirmation dialog exposes the **Merge Distance** so you can widen or
narrow it before applying. It arrives pre-filled with the distance the
scan used.

:::{note}
**A mesh can pass for a long time and then fail when you switch
tetrahedralizers.** [fTetWild](https://github.com/wildmeshing/fTetWild),
the tolerant default for **Solid** groups, resamples the surface and welds
near-coincident pairs away as a side effect.
[TetGen](https://www.wias-berlin.de/software/tetgen/) preserves the input
surface exactly, so it carries the pair straight through. If a Solid that
worked stops working the day you switch to TetGen, scan it.
:::

### Remove Loose Vertices

Deletes vertices that belong to no face, together with their loose edges.
**Changes the vertex count.**

The solver builds a vertex's mass and contact settings by averaging over
the faces it belongs to. A vertex with no faces has nothing to average, so
a **Static** collider carrying one is rejected at build time (see
[the isolated-vertex entry](../../troubleshooting.md) in
Troubleshooting), and a vertex hanging partway along a sewing seam ends up
with no mass at all, which makes the simulation stop before the first
frame.

Two kinds of vertex are deliberately kept:

- **Pinned vertices.** A pin holds a vertex at a prescribed position
  regardless of what it is attached to, so a face-less pinned point (a
  sewn curtain hook, for instance) is valid geometry rather than a stray
  point. Pinned vertices are exempt here and are not reported by the scan
  either.
- **Sand particle meshes.** Every grain center belongs to no face by
  construction, so running this on one would delete the whole body. It is
  skipped, and the repair reports nothing to do.

### Dissolve Degenerate

Collapses zero-area faces and edges shorter than the **Collapse Distance**.
**Changes the vertex count.**

A face with no area has no defined normal, so both the contact normal and
the bending hinge built on that face are undefined. Unlike a duplicate
face, a degenerate one is not caught at Transfer, so it reaches the
solver.

The scan and the repair measure different quantities. The scan flags a
face by its **area**, and the repair works from a distance, collapsing
edges shorter than the **Collapse Distance** and the zero-area faces that
go with them; that field arrives pre-filled from the scan's **Merge
Distance**. If a reported face is still listed after the repair, raise the
**Collapse Distance** to the size of the sliver you want gone and run it
again.

### Delete Duplicate Faces

Deletes faces that share their full vertex set with another face.
**Leaves the vertex count unchanged**, so pins and caches survive.

Coincident faces make the solver build a degenerate bending element and
stop at startup, and the encoder already refuses to transfer them, so this
one is a hard blocker with a one-click fix. Doubled geometry welded
together is the usual source, which makes this a common companion to
**Merge by Distance**.

### Triangulate

Triangulates every face with more than three corners. **Leaves the vertex
count unchanged**, so pins and caches survive.

Blender chooses which diagonal to split a quad along from the *current*
vertex positions, and re-chooses it as the quad deforms. The simulation
uses one triangulation, fixed when you press **Transfer**. As a quad
folds, the surface you see can therefore be split along a different
diagonal than the surface being simulated, and the two disagree by the
fold depth of that quad. In the viewport this reads as a thin
penetration that the solver's own checks cannot see, because it is not
present in the simulated state at all.

Triangulating up front removes the possibility: an explicit triangle has
no diagonal left to re-pick. The scan tells you how close your quads are
to flipping (*"largest fold 47.3 deg, 0 past the flip boundary"*), so you
can judge whether it is worth doing on a given mesh.

### Recalculate Outside

Makes face winding consistent and outward. **Leaves the vertex count
unchanged**, so pins and caches survive.

Tetrahedralization needs a consistently wound surface to tell inside from
outside, so this matters most for **Solid** groups. On a mesh that is
already consistent the repair reports nothing to do and cancels.

:::{note}
Winding is captured when you press **Transfer**, so run this before
transferring, and transfer again if you run it afterward.
:::

## Symmetric Triangulate

The second box in **Utility Tools** holds a single button, **Symmetric
Triangulate Selected**. It pokes every face: a center vertex is inserted
and the face is fanned into triangles around it, so a quad becomes four
triangles instead of two.

Use it when a symmetric mesh needs to fold symmetrically. A
single-diagonal triangulation is not mirror-symmetric, and the bending
hinges inherit that bias, so a symmetric rest shape buckles to one side.
A center fan has no such bias.

:::{warning}
**Symmetric Triangulate adds one vertex per face, and it does not ask
first.** It changes the vertex count like the three repairs above, but it
has no confirmation dialog and clears nothing on your behalf. Everything
in
[What a vertex-count change invalidates](#what-a-vertex-count-change-invalidates)
applies: re-run **Transfer**, and re-run **Capture Deformation** on any
collider that had one, after you use it.
:::

## What a Vertex-Count Change Invalidates

**Merge by Distance**, **Remove Loose Vertices**, **Dissolve Degenerate**
and **Symmetric Triangulate** all change how many vertices a mesh has.
Several things in the add-on are keyed on that number, and the scan report
names the affected ones for each object before you run anything, under
*"Changing the vertex count invalidates:"*. When nothing depends on it,
the panel says that instead.

| What                          | Why the count matters                                                                                     | How to get it back                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Display cache (PC2)**       | The fetched animation stores one position per vertex per frame, so after a count change it replays the wrong points. | **Transfer**, **Run**, **Fetch** again.                                                       |
| **Capture Deformation cache** | A recorded collider or animated-pin deformation is one full mesh shape per frame, sized by the vertex count at the moment it was recorded, and the encoder rejects it once that count differs. | [**Capture Deformation**](static_objects.md#the-capture-deformation-button) again on that object. |
| **Pin vertex groups**         | These are **not** deleted and stay valid. Blender remaps its own vertex groups through the edit, so no pin is stranded. | Nothing, but see the note below.                                                              |

:::{important}
**Re-run Transfer after any of these repairs.** The add-on tracks the
topology it last transferred, and fetching across a mismatch would bind a
cache to a mesh of a different vertex count. If you skip it, the add-on
warns *"Mesh topology changed since last transfer"* and names the groups
that diverged, so the divergence is visible rather than silently
simulated. See [Troubleshooting](../../troubleshooting.md).
:::

:::{note}
**A merge can move a pin's membership, even though it never deletes one.**
When a pinned vertex is welded to an unpinned one, the survivor's vertex
group membership follows Blender's own merge rules rather than the pick
you originally made. Nothing here removes authored pin groups, but it is
worth a look at your pins after a large weld. See
[Pins and Operations](../constraints/pins.md).
:::

### The Confirmation Dialog

The three repairs that change the count open a dialog before writing
anything. It contains:

1. The line **This repair changes the vertex count.**
2. A list of what that invalidates on the selected objects, one line per
   item, or the line *"No capture cache, display cache, or pin is
   affected."* when nothing does.
3. The repair's own distance field, where it has one (**Merge Distance**
   for Merge by Distance, **Collapse Distance** for Dissolve Degenerate;
   Remove Loose Vertices has none).
4. **Clear invalidated caches**, shown only when something is
   invalidated, and **on by default**.
5. **I understand, apply anyway**, off by default. Without it the repair
   cancels and nothing is written.

Leave **Clear invalidated caches** on. A cache that no longer matches the
mesh is not inert: the viewport overlay reads it on every redraw, and
because the mismatch is saved in the `.blend`, it outlives the session.
With the box ticked, the add-on deletes the display cache and the captured
deformation for each object whose count actually moved, and reports what
it cleared. Objects in the selection whose count did not move keep theirs.

## Python / MCP API

The whole surface is available to an MCP client as eight tools:
`scan_meshes`, `merge_by_distance`, `remove_loose_vertices`,
`dissolve_degenerate_faces`, `delete_duplicate_faces`,
`triangulate_for_solver`, `recalculate_normals_outside` and
`symmetric_triangulate`. Each takes an explicit list of object names
rather than acting on the viewport selection, and `scan_meshes` returns
the same per-object report the panel draws, including the list of what a
vertex-count change would invalidate.

The same eight are on the scripting API as methods on `solver`, where
`scan_meshes` returns the report as data. The three that change the vertex
count take an explicit acknowledgement, since a script gets no confirmation
dialog:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

report = solver.scan_meshes("Cloth")
solver.merge_by_distance("Cloth", acknowledge=True)
```

:::{admonition} Under the hood
:class: toggle

**Why a hair-thin gap is worse than a large defect**

The contact barrier's dynamic stiffness carries a `mass / gap^2` term, so
a pair of vertices a few nanometers apart contributes Hessian entries many
orders of magnitude larger than the rest of the row. All GPU compute runs
in single precision, so the assembled Newton matrix loses rank and the
solve stops on its positive-definiteness guard at the first iteration,
naming no geometry. That is why **Merge by Distance** sits at the top of
the error list even though a near-coincident pair is invisible in the
viewport.

**Why a face-less vertex is fatal rather than ignored**

Vertex mass is aggregated from faces, rod edges and tets. A sewing (loose)
edge contributes none, so a vertex sitting partway along a seam ends up
with zero mass, its inertia block is zero, and the linear solve goes
singular. A pinned vertex is exempt because a pin is an exact boundary
condition: the solver prescribes its position, so it carries no free
momentum and cannot make the system singular. For a **Static** collider
the same aggregation divides by zero and the build stops instead.

**Why the displayed quad and the simulated quad can disagree**

Blender builds a quad's `(0,1,2) + (0,2,3)` split and flips to
`(0,1,3) + (1,2,3)` once the two candidate triangles' normals have
separated by more than 90 degrees. That decision is re-made from the
current positions on every tessellation, with no hysteresis and no
reference to the rest pose. The solver ships one triangulation captured
from the rest topology at Transfer time and keeps it for the whole
simulation. The scan reports the current largest separation across the
`0-2` diagonal, and how many faces are already past 90 degrees.

**Why winding is an error while an open boundary is a note**

Tetrahedralization needs a consistently wound surface to tell inside from
outside: TetGen refuses such input outright, and fTetWild accepts it and
resamples it into something other than what you authored. So winding is a
**Solid** concern first. It is listed with the errors because it has a
safe one-click repair that leaves the vertex count alone, where closing a
boundary would be a modeling decision with no single right answer.
Winding is detected by walking each manifold edge: two consistently wound
neighbors traverse their shared edge in opposite directions.

**Why the fan and not the diagonal**

A single-diagonal triangulation of a quad is not mirror-symmetric under
either in-plane axis, and the bending hinges built on those triangles
inherit the asymmetry, so a symmetric sheet develops its fold to one side.
Poking each face into a center fan is symmetric under both axes, so a
symmetric rest shape folds symmetrically.

**Report caching**

A scan report is cached per object and discarded when the mesh's vertex or
polygon count stops matching the scanned counts. The panel draw only
formats an already-computed report, so it performs no mesh traversal. A
successful repair also clears the last Transfer error from the panel,
since the mesh it named may be the one just fixed.
:::
