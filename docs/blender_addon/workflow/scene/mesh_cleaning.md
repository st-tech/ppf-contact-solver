# 🧹 Mesh Cleaning

**Mesh Cleaning** scans the meshes you have selected for geometry that
stops a **Transfer** or a solve, and offers a repair for each finding that
has a safe one. Scanning never changes a mesh; every repair is a separate
button press.

Use it before the first **Transfer** of geometry you did not author
yourself: an imported character, a downloaded prop, a garment exported
from another application, or anything that has been through a boolean, a
mirror, or a merge. {ref}`Accepted geometry <accepted-geometry>` describes
what the solver takes as input; this page is how you get a mesh there.

This page covers:

- [Where it lives](#where-it-lives)
- [Scanning a selection](#scanning-a-selection): the read-only pass and
  its two thresholds
- [What the scan reports](#what-the-scan-reports): errors, notes, and which
  errors **Transfer** also refuses
- [The repairs](#the-repairs): six buttons, what each one fixes, and the
  **Triangulate Faces** button in the refused-Transfer dialog
- [Symmetric Triangulate](#symmetric-triangulate)
- [Intersection Correction](#intersection-correction): push overlapping **Solid** and **Pdrd** objects apart, and
  reshape **Shell** cloth that passes through itself or other surfaces with
  [Resolve All Intersections](#resolve-all-intersections)
- [What a vertex-count change invalidates](#what-a-vertex-count-change-invalidates):
  read this before running one of the three repairs that move the count
- [Python / MCP API](#python--mcp-api)

## Where It Lives

Open the sidebar (`N`) in the 3D viewport, switch to the add-on tab, and
expand the **Mesh Repair** panel. It sits between **Snap and Merge** and
**Visualization** and is **collapsed by default**. The panel holds four
boxes: **Mesh Cleaning**,
[**Symmetric Triangulate**](#symmetric-triangulate) below it,
[**Intersection Correction**](#intersection-correction) below that, and
[**Resolve All Intersections**](#resolve-all-intersections) at the bottom.

Everything here works in **Object Mode**. **Mesh Cleaning** and **Symmetric
Triangulate** act on the **selected mesh objects**; **Intersection Correction**
acts on the objects of your **Solid** and **Pdrd** groups, and **Resolve All
Intersections** also on those of your **Shell** groups, or on the selected ones
only when **Selected Only** is on. In Edit Mode, or with no mesh selected,
the **Mesh Cleaning** and **Symmetric Triangulate** buttons are grayed out and a
label under them names the missing precondition (*Switch to Object Mode*,
*Select one or more mesh objects*). With a valid selection the label reads
*N mesh object(s) selected*.

:::{important}
**Every repair button applies to the whole selection, not just the object
whose report it sits under.** The report is drawn one box per selected
object, so it is easy to read a fix button as belonging to the box above
it. It does not: pressing it repairs every selected mesh. When you want a
fix on one object only, select that object alone first.
:::

## Scanning a Selection

1. In **Object Mode**, select the mesh objects you want to check.
2. Press **Scan Selected Meshes**.

The status line reports either *Scanned N mesh object(s): no defects
found* or *Scanned N mesh object(s): M need attention*, and the panel fills
with one box per object listing what was found. Objects that are selected
but not yet scanned are counted in an *N object(s) not scanned yet* line at
the bottom.

Only [error-class](#what-the-scan-reports) findings count toward *need
attention*. An object with notes and no errors reports **No defects
found** and still lists its notes below.

### The Two Thresholds

Two numbers control the scan. The panel button scans at the defaults below.
To scan at other values, use a script; see
[Python / MCP API](#python--mcp-api).

| Field               | Default  | Unit                | What it does |
| ------------------- | -------- | ------------------- | ------------ |
| **Merge Distance**  | `0.0001` | local mesh units    | Vertex pairs closer together than this are reported as near-coincident. Matches Blender's own **Merge by Distance** default, so the scan agrees with what **Mesh > Merge > By Distance** would weld. |
| **Degenerate Area** | `0`      | local units squared | Faces at or below this area are reported as degenerate. At the default, only exactly zero-area faces are reported. |

A fix button drawn inside a report carries the distance that scan used, so
the repair covers exactly what the report described rather than reverting
to the operator's own default.

:::{note}
**A report is dropped when a fix button runs on the object, and whenever
the object's vertex or polygon count differs from the counts the
scan was taken at.** Between them those two rules cover every repair on
this page, including the ones that leave both counts alone.

What they do not cover is a count-preserving hand edit. Dragging a vertex
in Edit Mode leaves both counts intact, so the report stays on screen
even though it may be out of date: two vertices you nudged together
are a near-coincident pair the standing report does not list. Re-scan
after you edit a shape by hand.
:::

## What the Scan Reports

The report is split by severity. **Errors** are geometry that can stop a
**Transfer** or a solve. **Notes** are geometry worth knowing about that is
not a problem by itself. An ordinary cloth mesh is not flagged: an open
panel built from quads is normal, so it produces notes and no errors.

**Errors** are drawn with a warning icon and, where a safe repair exists, a
fix button:

| Finding                              | What it is | Fix offered | Vertex count |
| ------------------------------------ | ---------- | ----------- | ------------ |
| **Linked Duplicate of `<name>`**     | The object shares its mesh data with another object (an `Alt`-`D` copy). | none, see below | n/a |
| **Near-coincident vertex pair(s)**   | Two vertices closer than the **Merge Distance**, reported with the smallest gap in both local and world units. | **Merge by Distance** | changes |
| **Isolated vertex(es), in no face**  | Points that belong to no face, usually left behind in an imported model. | **Remove Loose Vertices** | changes |
| **Hanging seam vertex(es)**          | Points sitting partway along a sewing (loose) edge, belonging to no face. | **Remove Loose Vertices** (cuts the seam, see below) | changes |
| **Duplicate face(s)**                | Two triangles of the tessellated mesh that share the same three vertices, usually because a face is doubled. A doubled quad counts as two. | **Delete Duplicate Faces** | unchanged |
| **Degenerate (zero-area) face(s)**   | Faces at or below the **Degenerate Area**. | **Dissolve Degenerate** | changes |
| **Inconsistently wound edge(s)**     | Neighboring faces that traverse a shared edge in the same direction, so one of them faces the wrong way. | **Recalculate Outside** | unchanged |

A **Linked Duplicate** is the one error in the table with no button. Each
object must have its own mesh data, and repairing shared data here would
edit every object using it, so the report points you at
**Object > Relations > Make Single User > Object & Data** instead.

:::{note}
**One error counts without a row: degenerate tessellation.** A face whose
triangles are too thin to simulate (a corner sits on, or very near, the
straight line between its neighbors) is one **Transfer** refuses. The scan
asks the same question and adds what it finds to the *need attention*
total, but it draws no row and no fix button, so an object whose only
defect is this one is counted in the status line while its box below stays
empty. The repair is the **Triangulate Faces** button in the **Cannot
Transfer This Mesh** dialog, described under [Triangulate Faces](#triangulate-faces).
:::

**Notes** carry an info icon and never make an object read as needing
attention:

| Finding                                | What it means |
| -------------------------------------- | ------------- |
| **Boundary edge(s), surface is open**  | The surface has an open border. Fine for a **Shell**; a **Solid** or a **PDRD** body needs a closed surface (a **Solid** to tell inside from outside, a **PDRD** body to compute its volume). |
| **Non-manifold edge(s)**               | More than two faces meet along one edge. |
| **Face(s) with more than 3 corners**   | Quads and n-gons, reported with the largest fold angle across the quad's diagonal and how many are past the flip boundary. Offers **Triangulate**. |

:::{note}
**An open panel made of quads is ordinary cloth, and the scan says so.** A
flat garment panel, a curtain, a flag: each is an open surface built from
quads, so the boundary-edge row and the more-than-3-corners row appear on
almost every **Shell** you will simulate. They are notes so that they do
not hide the errors that stop a run, and there is nothing to fix. Closing a
boundary is a modeling decision, not a cleanup, so the tool offers no
button for it.
:::

:::{note}
**A Sand particle mesh is scanned for near-coincident vertices and shared
mesh data.** A particle mesh (made with **Convert To Solid Particle Mesh**)
is a cloud of loose grain centers, so every vertex legitimately belongs to
no face and the face-based checks are skipped on it. Near-coincident grain
centers are still worth catching, and are in fact the finding that matters
most for grains. The linked-duplicate check runs too, since sharing a mesh
datablock is a problem whatever the geometry is.
:::

:::{warning}
**Mesh Cleaning expects faces.** A Sand particle mesh is the only faceless
mesh the tool recognizes. The edge-only mesh of a **Rod** group has no
faces either, so the scan reports all of its unpinned vertices as isolated
and hanging seam errors, and **Remove Loose Vertices** deletes them. Do not
press **Remove Loose Vertices** on a **Rod** mesh.
:::

### Which Errors Transfer Also Refuses

The scan reports every error above on any mesh you select, whatever group
the mesh is in. **Transfer** refuses only some of them, and only for some
group types. A clean scan is therefore not a promise that **Transfer**
passes, and a scan error is not always a **Transfer** error.

| Finding | **Transfer** refuses it |
| ------- | ----------------------- |
| Linked Duplicate | On any assigned object. |
| Duplicate faces | On any assigned object except in a **Sand** group. |
| Isolated vertices | On a **Static** collider only. A **Shell** or **Solid** is not refused; the solver drops the vertex. A **Rod** mesh has no faces by construction and is not refused either. |
| Hanging seam vertices | On a **Shell** or **Solid** group only. On a **Shell**, a pinned vertex is exempt. On a **Solid**, no pin exempts it. A **Static** collider is refused for such a vertex as an isolated vertex. |
| Degenerate tessellation and zero-area faces | On every group except **Sand**. A **Shell**, or a **Solid** meshed with TetGen, is refused for any triangle too thin to simulate. Every other type is refused only for a triangle of exactly zero area. The scan applies the stricter test to every mesh, so it can flag an object that **Transfer** accepts. |
| Near-coincident vertex pairs, inconsistent winding, boundary and non-manifold edges, quad fold | Not by **Transfer**'s own checks. Only the scan reports them. |

The build that starts after **Transfer** has checks of its own, which this
table does not cover. A **Solid** meshed with TetGen, for example, needs a
closed, manifold, consistently wound surface (see
{ref}`Accepted geometry <accepted-geometry>`), so an open or inconsistently
wound mesh is rejected there.

**Transfer** also checks things that are not geometry, such as an object
assigned to two groups. See
[Transfer or build rejects the mesh](../../troubleshooting.md#transfer-or-build-rejects-the-mesh).

## The Repairs

Each repair acts on the selected meshes and reports what it did per
object in the status line (*Merged 12 on Shirt (12). Re-scan to
confirm.*). A repair that finds nothing to do reports *Nothing to repair on
the selected meshes.* and cancels. Objects sharing one mesh datablock are
repaired once, not once per object using it.

Three of the six repairs change the vertex count and ask for confirmation
before they write; see
[What a vertex-count change invalidates](#what-a-vertex-count-change-invalidates).

### Merge by Distance

Welds vertex pairs closer together than the **Merge Distance** into one
vertex. **Changes the vertex count.**

A near-coincident pair lies far inside the contact gap, which makes the
contact response very stiff there. The solve then stops on a check that names
no geometry, so the cause is hard to find from the log (the numbers are in
the *Under the hood* block at the end of this page). Welding the pair
removes it.

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

Deletes vertices that belong to no face, together with the edges attached
to them. **Changes the vertex count.**

The solver builds a vertex's mass and contact settings by averaging over
the faces it belongs to. A vertex with no faces has nothing to average. A
**Static** collider carrying one is refused at **Transfer** (see
[the isolated-vertex entry in Troubleshooting](../../troubleshooting.md#object--has-n-isolated-vertexes-the-message-says-isolated-vert)),
and a vertex hanging partway along a sewing seam has no mass at all, which
stops the simulation before the first frame.

:::{important}
**For a hanging seam vertex, this button deletes the seam edges attached to
it, which cuts the seam.** To keep the seam, dissolve the vertex in Edit
Mode instead (**Mesh > Delete > Dissolve Vertices**), so that each seam
is a single edge between two vertices that belong to faces, then re-scan.
:::

Two kinds of vertex are deliberately kept:

- **Pinned vertices.** A pin holds a vertex at a prescribed position
  regardless of what it is attached to, so a face-less pinned point (a
  sewn curtain hook, for instance) is valid geometry on a **Shell** rather
  than a stray point. The scan does not report pinned vertices and this
  repair keeps them. On a **Solid**, a pinned vertex that lies on a sewing
  edge but in no face is still refused by **Transfer**, because a Solid is
  rebuilt from its faces alone and no pin reaches such a vertex. The scan
  and this repair leave it alone, so remove or dissolve it yourself.
- **Sand particle meshes.** Every grain center belongs to no face by
  construction, so running this on one would delete the whole body. It is
  skipped, and the repair reports nothing to do.

### Dissolve Degenerate

Collapses zero-area faces and edges shorter than the **Collapse Distance**.
**Changes the vertex count.**

A face with no area has no defined normal, so both the contact normal and
the bending hinge built on that face are undefined. **Transfer** refuses a
triangle of exactly zero area, and on a **Shell**, or a **Solid** meshed with
TetGen, also a triangle too thin to simulate. It does not look at short edges
or at a **Degenerate Area** above zero; this repair and the scan do.

The scan and the repair measure different quantities. The scan flags a
face by its **area**, and the repair works from a distance, collapsing
edges shorter than the **Collapse Distance** and the zero-area faces that
go with them; that field arrives pre-filled from the scan's **Merge
Distance**. If a reported face is still listed after the repair, raise the
**Collapse Distance** to the size of the sliver you want gone and run it
again.

A sliver that is thin without being degenerate is worth removing too. On
a **Shell** group, or a **Solid** meshed with TetGen, which keeps the input
surface, the [Local Gap Limit](../params/material.md#local-gap-limit)
measures the smallest triangle height around each point, so one sliver lowers
the contact gap of every triangle that shares a point with it. A **Solid**
meshed with fTetWild is measured on its resampled surface instead.

### Delete Duplicate Faces

Deletes faces that share their full vertex set with another face.
**Leaves the vertex count unchanged**, so pins and caches survive. It
changes the polygon count, so run **Transfer** again afterward (see
[the note below](#what-a-vertex-count-change-invalidates)).

Coincident faces make the solver build a degenerate bending element and
stop at startup, and **Transfer** refuses them, so this is a hard blocker
with a one-click fix. Doubled geometry welded together is the usual source,
which makes this a common companion to **Merge by Distance**.

### Triangulate

Triangulates every face with more than three corners. **Leaves the vertex
count unchanged**, so pins and caches survive. It changes the polygon
count, so run **Transfer** again afterward.

Blender chooses which diagonal to split a quad along from the *current*
vertex positions, and re-chooses it as the quad deforms. The simulation
uses one triangulation, fixed when you press **Transfer**. As a quad
folds, the surface you see can therefore be split along a different
diagonal than the surface being simulated, and the two disagree where the
quad folds. In the viewport this can read as a thin penetration that the
solver's own checks cannot see, because it is not present in the simulated
state at all.

Triangulating up front removes the possibility: an explicit triangle has
no diagonal left to re-pick. The scan tells you how close your quads are
to flipping (*"largest fold 47.3 deg, 0 past the flip boundary"*), so you
can judge whether it is worth doing on a given mesh.

### Recalculate Outside

Makes face winding consistent and outward. **Leaves the vertex count
unchanged**, so pins and caches survive.

Tetrahedralization needs a consistently wound surface to tell inside from
outside, so this matters most for **Solid** groups. On a mesh whose winding
is already consistent the repair reports nothing to do and cancels.

The scan and the repair count only edges whose two faces disagree. A closed
mesh whose faces are consistently wound but all point inward therefore has
no finding in the scan. **Recalculate Outside** still turns such a mesh
outward, and it reports *Nothing to repair on the selected meshes.* because
no inconsistent edge was fixed.

:::{note}
Winding is captured when you press **Transfer**, so run this before
transferring, and transfer again if you run it afterward.
:::

### Triangulate Faces

This button is not in the **Mesh Repair** panel. When **Transfer** finds
a face whose triangles are too thin to simulate, it refuses and opens the
**Cannot Transfer This Mesh** dialog, which names the object and the faces.
Where a different split of those faces fixes the problem, the dialog
carries a **Triangulate Faces** button. Press it, then **Transfer** again.

The button splits exactly the offending faces and leaves the rest of the
mesh alone. It repairs every included mesh of every active group except
**Sand** (**Static** colliders included), not only the object named in the
dialog. It applies the same strict test as the scan to every one of them, so
it can also split faces on a **Static** collider, a **Rod**, a **PDRD** body or
a **Solid** meshed with fTetWild, which **Transfer** accepts as authored. It
judges each mesh as modeled, so a corner that a shape key, an armature, or
another deformer moves onto the line between its neighbors is not seen by it.

The dialog draws the button only when a different split can fix every
flagged face of the refused object. A face that is already a triangle has
no other triangulation, and a face with no area or a zero-length edge has
no sound split at all; the dialog then only states the refusal. Move the
offending vertex off the line, weld coincident vertices with **Merge by
Distance**, or collapse the face with **Dissolve Degenerate**.

## Symmetric Triangulate

The second box in **Mesh Repair** holds a single button, **Symmetric
Triangulate Selected**. It pokes every face: a center vertex is inserted
and the face is fanned into triangles around it, so a quad becomes four
triangles instead of two and a triangle becomes three.

When it finishes, the status bar reads *Symmetric-triangulated N object(s) (M
mesh data-block(s))*. M is smaller than N when several objects share one mesh,
because shared mesh data is changed once.

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

## Intersection Correction

**Intersection Correction** is a third box in **Mesh Repair**, below
**Symmetric Triangulate**. **Transfer** refuses a scene whose surfaces start
through each other, and a chain converted from Geometry Nodes can arrive with
hundreds of shallow overlaps between its links. It also refuses surfaces that
start closer together than their contact offsets allow, and this tool treats
those pairs as overlapping too. The tool opens the overlaps without running the
solver, so it needs no connection: it moves each overlapping object, as a
whole, by the smallest motions that separate every overlapping pair. Press
**Transfer** again afterward.

1. Switch to Object Mode.
2. Optionally adjust the three settings below.
3. Click **Resolve Intersections**. The mouse cursor shows a progress count
   while the passes repeat, and the status bar then says how many objects moved
   and whether any pair still overlaps.

| Setting | Default | What it does |
| ------- | ------- | ------------ |
| **Max Iterations** | `50` | The number of passes to spend. Each pass finds the overlaps that are left and moves the objects once. The minimum is 1. |
| **Clearance** | `1.0` | How far apart to leave surfaces that overlap, beyond their contact offsets, in multiples of each object's contact gap. Surfaces closer than half of this count as overlapping. The minimum is 0.1. |
| **Selected Only** | off | Moves only the selected objects. The others stay where they are, and the selected ones move around them. |

The button is grayed out outside Object Mode, and when fewer than two objects
are in **Solid** or **Pdrd** groups. The panel then says *Switch to Object
Mode* or *Needs two or more objects in SOLID or PDRD groups*. With two or more
it reads *N SOLID or PDRD object(s)*.

After a run, a box under that label summarizes it:

| Line | What it means |
| ---- | ------------- |
| *Last run: no overlapping objects found* | Nothing needed to move. |
| *Last run: N overlapping pair(s) opened in P pass(es)* | Every overlapping pair that was found is separated. |
| *Last run: M of N pair(s) still overlap* | `M` of the `N` pairs found are still overlapping. |
| *N object(s) moved, by at most D* | `D` is the farthest any vertex of a moved object traveled, in scene units. |
| *N object(s) held in place* | Objects that stayed where they were, for one of the reasons under **What stays in place**. |
| *N object(s) left out: surface not closed* | Objects skipped for the reason under **What it leaves out**. |

**What takes part.** The included mesh objects of active **Solid** and
**Pdrd** groups, because those are the types whose surface encloses a volume.
A **Shell** has no inside, so **Resolve Intersections** leaves its overlaps
alone; [**Resolve All Intersections**](#resolve-all-intersections) handles them.

**What it changes.** Only the object's position and rotation in the scene. No
mesh, vertex group, pin or cache is touched, so everything that refers to
vertices stays valid. When a run moves anything, it also clears the violation
highlights that a refused **Transfer** left in the viewport, because they mark
where the surfaces used to be.

**What stays in place.** These objects do not move, but they still push the
others away:

- An object that one of its group's pins holds.
- An object whose position is driven by something else: its own animation
  curves, a driver, an NLA strip or a constraint, on it or on a parent.
- An object linked from another file.
- With **Selected Only**, every object that is not selected.

If every free object is held, the tool reports *Every SOLID or PDRD object is
held in place*. Two held objects that overlap each other are reported as still
overlapping and left as they are.

**What it leaves out.** An object whose surface is open, non-manifold or
inconsistently wound has no inside to push out of, so it is skipped. Scan it
with **Mesh Cleaning** above, and repair the winding with **Recalculate
Outside**. An object that lies entirely inside another crosses no surface and
is not found. An object that passes completely through another has no single
way out, and it is separated correctly only when the overlap is shallow.

The result is one of three messages: *No overlapping objects found*, *Moved N
object(s) in P pass(es); no overlap is left. Run Transfer again*, or *Moved N
object(s); M pair(s) still overlap after P pass(es)*. When pairs still overlap,
click **Resolve Intersections** again, which continues from where the last run
stopped, raise **Max Iterations**, or free a held object. An object squeezed
between two held objects that push it opposite ways cannot be freed, and both
of its pairs stay reported as overlapping.

**Undo** (`Ctrl+Z`) puts the objects back where they were. It does not bring
back the highlights that the run cleared; the next **Transfer** marks whatever
it refuses.

### Resolve All Intersections

**Resolve All Intersections** is a fourth box in **Mesh Repair**, below
**Intersection Correction**. It does what **Resolve Intersections** does for the
**Solid** and **Pdrd** objects, and then reshapes **Shell** cloth that passes
through itself, through another cloth, or through a **Solid**, **Pdrd** or
**Static** object. **Transfer** refuses all of these, and moving a whole object
cannot free a sleeve that is caught in its own body. Press **Transfer** again
afterward.

1. Switch to Object Mode.
2. Optionally adjust **Max Iterations**, **Clearance** and **Selected Only**.
   They are the three settings above, and both buttons read them. **Max
   Iterations** limits only the first step, the one that moves **Solid** and
   **Pdrd** objects.
3. Click **Resolve All Intersections**.

**What it does.** The **Solid** and **Pdrd** objects move apart first, as rigid
bodies, exactly as **Resolve Intersections** moves them. Then the vertex
coordinates of the **Shell** meshes are edited until no triangle is inside
another's contact offsets, or inside its own. **Solid**, **Pdrd** and **Static**
objects stay where the first step left them, and the cloth goes around them. A
second step pulls the cloth back toward its original shape as far as it can
without overlapping again. A mesh keeps its vertex order, vertex groups, pins
and UVs, so nothing that refers to a vertex becomes invalid.

**While it runs.** The work happens in a separate process, so Blender stays
usable. Both **Resolve** buttons are grayed out, and the box shows the step
(*Step 1 of 2: untangling*, then *Step 2 of 2: restoring the shape*), a bar of
the overlapping pairs cleared so far, the iteration, the seconds elapsed, and
an **Abort** button. A scene made of separate tangles, or a very large one, is
solved one part at a time, and the box then also says *Round R, place K of N*.
Press **Abort** to stop: work that is already finished is kept, a part still in
step 1 stays as it was, and a part in step 2 keeps the state it had reached. Do
not edit a cloth object while the run goes. A mesh that changed meanwhile is
left as you made it, and the run says *NAME was edited while the run was going,
so the result was not written. Run again*.

**Results.** The status bar says *Opened N overlapping pair(s); K shell
object(s) moved by at most D. Run Transfer again*, or *Shells not resolved:*
followed by the reason. A box under the button summarizes the last run:

| Line | What it means |
| ---- | ------------- |
| *Last run: no overlapping surfaces found* | Nothing needed to move. |
| *Last run: N overlapping pair(s) opened in S s* | Every overlapping pair that was found is separated. |
| *Last run: M of N pair(s) still inside their margin* | `M` of the `N` pairs found are still closer than their contact offsets allow. |
| *Last run: aborted* | You pressed **Abort**. |
| *Rigid: N object(s) moved, M pair(s) still overlap* | What the first step did for the **Solid** and **Pdrd** objects. |
| *N shell object(s) moved, by at most D* | `D` is the farthest any vertex traveled, in scene units. |
| *N shell object(s) held in place* | Cloth objects that stayed where they were, for a reason under **What stays in place**. |
| *N pair(s) of held triangles overlap and cannot be moved* | Two parts that both stay put overlap each other, for example one collider through another. Move one of them yourself. |

**What stays in place.** These do not move, and they still push the cloth away:

- Every vertex that a pin of its group holds.
- A whole cloth object that is linked from another file, shares its mesh with
  another object, has Shape Keys, is deformed at the starting frame by a
  modifier such as an Armature, or has a transform that cannot be inverted.
- With **Selected Only**, every object that is not selected.

The box counts the held objects, and the MCP result names the reason for each.

**What it leaves out.** A **Shell** object with no triangles, and every **Rod**
and **Sand** object. Objects that are in no active group take no part. The
*Allow Intersections* settings are ignored: every overlap is opened.

**How long it takes.** A small scene takes seconds to a minute, and a garment of
a hundred thousand triangles takes several minutes.

**When pairs remain.** A scene with a single tangle that cannot be cleared stays
as it was. A scene of separate tangles, or a very large one, keeps what was
cleared, and the status bar says *Shells not resolved: N pair(s) are still
inside their margin*. Click the button again to continue from the state the last
run left.

**Undo** (`Ctrl+Z`) puts everything back in one step: the transforms that the
first step changed and the cloth's vertex coordinates. As with **Resolve
Intersections**, it does not bring back the highlights that the run cleared.

## What a Vertex-Count Change Invalidates

**Merge by Distance**, **Remove Loose Vertices**, **Dissolve Degenerate**
and **Symmetric Triangulate** all change how many vertices a mesh has.
Several things in the add-on are keyed on that number. Before you run
anything, the scan report names the affected ones for each object under
*"Changing the vertex count invalidates:"*. When nothing depends on it,
the panel says *"No cache or pin depends on this vertex count."* instead.

| What | Why the count matters | How to get it back |
| ---- | --------------------- | ------------------ |
| **Display cache (PC2)** | The fetched animation stores one position per vertex per frame, so after a count change it replays the wrong points. | **Transfer**, **Run**, **Fetch All Animation** again. |
| **Capture Deformation cache** | A recorded collider or animated-pin deformation is one full mesh shape per frame, sized by the vertex count at the moment it was recorded, and **Transfer** refuses it once that count differs. | [**Capture Deformation**](static_objects.md#the-capture-deformation-button) again on that object. |
| **N pin group(s): membership may shift** | Pin vertex groups are **not** deleted and stay valid. Blender remaps its own vertex groups through the edit, so no pin is stranded. | Nothing, but see the pin note below. |

:::{important}
**Run Transfer again after any repair on this page.** The add-on records the
topology it last transferred (vertex and polygon counts, among other
things). **Merge by Distance**, **Remove Loose Vertices**, **Dissolve
Degenerate**, **Delete Duplicate Faces**, **Triangulate**, and **Symmetric
Triangulate** change those counts. If you skip **Transfer**, the add-on
warns *"Mesh topology changed since last transfer"* and names the groups
that diverged when you press **Run**, **Resume**, or **Fetch All
Animation**, so the divergence is visible rather than silently simulated.
**Recalculate Outside** changes no count, but the winding is captured at
**Transfer**, so it needs a new **Transfer** too. See
[Troubleshooting](../../troubleshooting.md#mesh-topology-changed-since-last-transfer).
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

Leave **Clear invalidated caches** on. A cache that does not match the
mesh is not inert: the viewport reads the stale cache and can slow down,
and because the mismatch is saved in the `.blend`, it outlives the
session. With the box ticked, the add-on deletes the display cache and the
captured deformation for each object whose count actually moved, and
reports what it cleared. Objects in the selection whose count did not move
keep theirs.

## Python / MCP API

The whole surface is available to an MCP client as eleven tools. Eight of
them take an explicit list of object names rather than acting on the
viewport selection: `scan_meshes`, `merge_by_distance`,
`remove_loose_vertices`, `dissolve_degenerate_faces`,
`delete_duplicate_faces`, `triangulate_for_solver`,
`recalculate_normals_outside` and `symmetric_triangulate`.
`scan_meshes` returns the same per-object report the panel draws,
including the list of what a vertex-count change would invalidate. Over MCP
the three repairs that change the vertex count stand in for the panel's
confirmation dialog, so each takes `acknowledge: true`; without it the call
is refused, and the refusal names what the change would have invalidated.

The `defects` table of a report has an entry for every finding in the tables
above, and one more, `degenerate_tessellation`, for the error that draws no
row: its `count`, the `polygons` whose triangles are too thin to simulate, and
`repairable_polygons`, the subset that **Triangulate Faces** can fix. That
entry counts toward `n_errors`, which is why an object can read as needing
attention while every other count is zero.

The ninth tool, `triangulate_degenerate_faces`, is the counterpart of the
**Triangulate Faces** button. It takes no object list: it repairs every
included mesh of every active group except **Sand**, and a call that finds
nothing it can repair is refused instead of reported as a success.

The tenth tool, `resolve_intersections`, is the counterpart of **Resolve
Intersections** under [Intersection Correction](#intersection-correction). Over
MCP it takes `object_names` (the objects allowed to move; omit it to let every
free object move), `max_iterations` (default `50`, at least 1) and `clearance`
(default `1.0`, at least 0.1). On the scripting API it is
`solver.resolve_intersections(*object_names, max_iterations=50,
clearance=1.0)`, with the names as positional arguments. Both return the same
result: `moved` (each object's name and its largest vertex displacement),
`remaining_pairs` (the pairs that still overlap), `held` (each held object's
name and the reason: `pinned`, `driven`, `linked` or `unselected`), `skipped`
(the objects left out), `iterations` and the counts. A call is refused when a
name is missing or is not a mesh, when an argument is out of range, or when
there are not two closed **Solid** or **Pdrd** objects with one of them free to
move.

The eleventh tool, `resolve_all_intersections`, is the counterpart of **Resolve
All Intersections** under
[Resolve All Intersections](#resolve-all-intersections). It takes the same
arguments as `resolve_intersections`: `object_names` (omit it to let every free
object move; with names given, every other object is held), `max_iterations`
(default `50`, at least 1) and `clearance` (default `1.0`, at least 0.1). On the
scripting API it is `solver.resolve_all_intersections(*object_names,
max_iterations=50, clearance=1.0)`. Both return `rigid` (as
`resolve_intersections` does) and `shell`: `resolved`, `moved` (each object's
name and its largest vertex displacement), `held` (each held object's name and
the reason: `pinned`, `linked`, `shared mesh`, `shape keys`, `deformed`,
`transform` or `unselected`), `pinned_vertices` (how many vertices a pin holds,
per object), `skipped`, the pair counts, `held_pair_count`, `seconds` and
`detail`, which says why a run that did not resolve the cloth did not. Unlike
the button, a call runs in the foreground and holds Blender until it finishes,
and it cannot be aborted. A call is refused when a name is missing or is not a
mesh, when an argument is out of range, or when there is no **Shell** object
that may deform and not two closed **Solid** or **Pdrd** objects with one of
them free to move.

The same eight object-list tools are on the scripting API as methods on
`solver`, where `scan_meshes` returns the report as data. There the call
itself is the confirmation, so the three take no `acknowledge` argument.
What both surfaces take is `clear_stale_caches` (on by default), which
decides what becomes of the caches the count change invalidates, the same
choice the panel's dialog offers. The example assumes `solver` is imported
as shown under [Import](../../integrations/python_api.md#import):

```python
report = solver.scan_meshes("Cloth")
solver.merge_by_distance("Cloth")

# Scan at non-default thresholds.
report = solver.scan_meshes("Cloth", merge_threshold=1e-3, area_eps=1e-8)

# Keep the caches instead, and expect the viewport overlay to read data
# sized for the old vertex count until the next Transfer rewrites it.
solver.merge_by_distance("Cloth", clear_stale_caches=False)
```

`scan_meshes` takes `merge_threshold` and `area_eps` over MCP as well. From
Blender's Python console,
`bpy.ops.object.scan_mesh_defects(merge_threshold=..., area_eps=...)` takes
the same two values.

:::{admonition} Under the hood
:class: toggle

**Why a hair-thin gap is worse than a large defect**

The contact barrier's dynamic stiffness carries a `mass / gap^2` term, so
a pair of vertices a few nanometers apart contributes Hessian entries many
orders of magnitude larger than the rest of the row. In single precision
the assembled Newton matrix then loses rank and the solve stops on its
positive-definiteness guard at the first iteration, naming no geometry.
That is why a near-coincident pair is an error even though it is invisible
in the viewport.

**Why a face-less vertex matters**

Vertex mass is aggregated from faces, rod edges and tets. A sewing (loose)
edge contributes none, so a vertex sitting partway along a seam ends up
with zero mass, its inertia block is zero, and the linear solve goes
singular. A pinned vertex is exempt because a pin is an exact boundary
condition: the solver prescribes its position, so it carries no free
momentum and cannot make the system singular. For a **Static** collider
the same aggregation divides by zero and the build stops instead. On a
**Shell** or **Solid**, a face-less vertex that is not on a seam is
dropped.

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

**How Intersection Correction moves the objects**

Each object moves as a rigid body, and one pass solves for the motions of all
of them at once: the smallest translations and rotations that open every
overlapping pair, with a rotation counted by how far it moves the object's own
vertices. That is why a chain that overlaps at every joint closes up in a
single pass, where pushing pairs apart one at a time would take a pass per
link. Whether two surfaces overlap is decided on copies of them grown outward
by the contact offset plus a quarter of the clearance. A pair therefore counts
as overlapping when it is closer than its contact offsets plus half the
clearance, is opened toward the offsets plus the whole clearance, and ends at
least half a clearance beyond its offsets.

**Why a very small clearance is raised**

Blender stores positions in single precision, which resolves about one part
in eight million of an object's distance from the world origin. A clearance
below sixteen times that resolution cannot be delivered, so for such an object
it is raised to that amount. A 5 mm chain link 30 m from the origin with a
contact gap of 85 nm is one: its clearance becomes about 0.06 mm.

**How Resolve All Intersections reshapes the cloth**

The first step minimizes three terms together: a repulsion between surfaces
that overlap, a penalty on every pair of triangles closer than their margin, and
a shape term that resists changing the cloth's triangles from their original
shape. Where one surface lies mostly on one side of another, which is the case
for a part that has poked through a body, that part is sent back to its own
side, since pushing it further out would only deepen the tangle. Where no side
is right, as for two sheets laid through each other, whole surfaces are pulled
apart. The second step pulls the cloth back toward its original shape while a
guard band around each pair's margin keeps it from overlapping again.

**Separate tangles are solved separately**

Cloth objects whose bounding boxes, each grown by half of the object's own
diagonal, do not meet cannot take part in the same tangle. Each group of
objects that could meet is solved as a scene of its own, so one that cannot be
cleared does not hold back the others. A scene of tens of thousands of triangles
is solved a place at a time instead: only the vertices within a few rings of an
overlap may move at first, and what is left is solved again with the rings
doubled, up to three more times.

**Blender may split a quad the other way after it moved**

Blender splits a quad into two triangles along the diagonal its current shape
asks for, and **Transfer** ships whatever triangles the mesh has at that moment.
A quad that the run folded can therefore be split along the other diagonal
afterward, and two surfaces that were kept apart along one diagonal can cross
along the other. After writing, the button compares the triangles with the ones
it solved for and, when they differ, solves again on the new ones, up to four
solves in all. The box then says *Solve N: Blender triangulated the cloth
differently after the move*.

**Report caching**

A scan report is cached per object and discarded when the mesh's vertex or
polygon count stops matching the scanned counts. The panel draw only
formats an already-computed report, so it performs no mesh traversal. A
successful repair also clears the last Transfer error from the panel,
since the mesh it named may be the one just fixed.
:::
