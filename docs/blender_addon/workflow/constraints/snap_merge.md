# 🧲 Snap and Merge

Two separate steps that work together:

- **Snap** is a one-shot alignment: it translates object A in world space
  so that its closest vertex lines up with the nearest point on object B's
  surface (the nearest triangle for a mesh or solid target, the nearest
  segment for a rod), leaving just enough contact gap to avoid
  interpenetration. In the same pass it also captures stitch anchors for
  **every A-vertex within reach of B**, not just the single closest one.
- **Merge** is a solver-side soft stitch between two objects. At transfer
  time it becomes a cross-object stitch force that holds the two meshes
  together during the simulation. It is a force, not a weld: stitched
  vertices are pulled toward their targets but never share the same point,
  so a small contact gap always remains.

You typically snap *and* merge in sequence (the snap operator creates the
merge pair automatically), but the two concepts are independent and can be
used separately.

:::{important}
**Snap is a set-level operation, not a single-vertex operation.**  The
feature is designed for meshes whose geometry already coincides (fully or
partly) so that *many* A-vertices have a matching counterpart on B.  A
canonical example is duplicating a subdivided square patch and snapping
the copy onto the original: every edge (or interior) vertex on the copy
has a one-to-one counterpart on the original, so the whole set stitches
in a single operation.

If only one isolated vertex on A happens to be close to B while the rest
of the mesh is far away, only that one pair is recorded. The solver
won't magically stitch meshes whose topologies don't align.  Author the
meshes so the intended stitch region coincides before snapping.
:::

```{figure} ../../images/snap_merge/snap_set_level.svg
:alt: Two side-by-side panels. Left, "Edges aligned": a 4x4 patch A in blue on the left and patch B in green on the right, with A's right edge and B's left edge at the same four row heights; four short red dashed lines pair the seam vertices; a badge reads "4 / 4 edge pairs recorded". Right, "Edges misaligned": the same two patches with B shifted down so only A's bottom-right corner shares a row with B's top-left; one red dashed line connects that pair, three A-edge vertices are labeled as having no B match, and a badge reads "1 / 4 edge pairs recorded". A footer reminds the reader that snap is a geometric alignment step, not a topology repair.
:width: 820px

Snap records a stitch pair for every A-vertex already within B's
contact gap, and nothing else. Left: matching subdivided patches
produce a full set of stitches in one pass. Right: a patch whose only
one corner coincides with B yields a single stitch; the rest of the
mesh stays unattached even after **Merge**. Author the meshes so the
intended stitch region coincides before snapping.
```

## Snap and Merge Panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab.
The **Snap and Merge** panel sits below **Scene Configuration** and
**Dynamics Groups**, and is **collapsed by default**. Click the header
to expand it.

When expanded, the panel contains a single box titled **Snap To Nearby
Vertices**:

- An **Object A** dropdown (the source; this is the object that will
  move) with an eyedropper button on the right for picking from the
  viewport.
- An **Object B** dropdown (the target; this stays put) with its own
  eyedropper.
- A **Snap A to B** button with a snap icon.

To snap two objects together, pick A and B with the dropdowns or
eyedroppers, then press **Snap A to B**. The add-on finds the source
vertex whose closest point on the target's surface is globally nearest,
translates A (in world space, parent-safe) so that pair lines up,
applies the contact-gap rules below, and records per-vertex barycentric
anchor data for a later stitch. It also registers a merge pair
automatically.

```{figure} ../../images/snap_merge/panel.png
:alt: Snap and Merge panel with Object A set to PatchA (moves) and Object B set to PatchB (target), the Snap A to B button, and an empty Merge Pairs box ready to receive the new pair.
:width: 500px

Snap and Merge panel with the two plane sheets below picked as A
and B. **PatchA** is the one that will move (magenta); **PatchB** is
the target that stays put (blue). Clicking **Snap A to B** runs the
operator on whatever pair you've set here.
```

```{figure} ../../images/snap_merge/snap_before.png
:alt: Before Snap A to B. Two identically-subdivided square shell patches in the 3D viewport with PatchB (blue) at the origin and PatchA (magenta) shifted along the +X axis only, same Y and same Z plane. A clear gap is visible along the X axis between PatchB's right edge and PatchA's left edge; the patches have not yet been joined.
:width: 520px

**Before.** Two identically-subdivided shell patches assigned to Shell
groups: **PatchB** (blue, at the origin, stays put) and **PatchA**
(magenta, shifted along the **+X axis only**; same Y, same Z). There
is a clear gap along the shared edge; the two patches are not yet
joined.
```

```{figure} ../../images/snap_merge/snap_after.png
:alt: After Snap A to B. The magenta PatchA has translated along the X axis and now sits a small contact gap from the blue PatchB, left-edge to right-edge, in the same plane. Together they read as a single 2x1 rectangle, with each seam vertex on PatchA tied to its counterpart on PatchB rather than merged into one point.
:width: 520px

**After `Snap A to B`.** PatchA has translated along the X axis in
world space until its nearest vertex sits a small contact gap from
PatchB. The two patches are not welded: every seam vertex on PatchA
is tied to its counterpart on PatchB by a recorded soft stitch, and
the stitch force together with the contact barrier settle into
balance across that small gap, so the patches read as a single 2x1
rectangle while staying separate meshes. The merge pair is
registered automatically and the per-vertex stitch anchors are
captured behind the scenes.
```

As soon as at least one merge pair exists, a second box labeled
**Merge Pairs** appears below the snap box, containing:

- A UIList showing each pair, with both object names per row.
- A **Re-snap** button below the list (disabled unless a row is
  selected). It re-runs the snap on the selected pair and rebuilds its
  stitch anchors. When the selected pair cannot stitch, the panel says
  why in a line with an error icon, the same reason **Transfer** gives
  (see [Merge Pairs Without Snapping](#merge-pairs-without-snapping)).
- A **Remove** button below that, likewise disabled unless a row is
  selected.
- A **Stitch Stiffness** slider, shown for **every supported pair**.
  All stitches are soft, so the slider applies to Shell-Shell and
  Rod-Rod pairs the same way it applies to the Solid-involved pairs.
  Raise it to hold a seam together more firmly.

A separate **Visualization** panel further down the sidebar exposes a
**Hide all snaps** toggle that hides or shows the merge-pair / stitch
overlay in the viewport.

### Gap Rules

The solver needs a small separation between the two meshes at rest,
otherwise contact barriers start flagging penetration on frame 1. Because
every stitch is a soft force rather than a weld, the snap operator always
leaves a positive gap; stitched vertices are never coincident. The same
rule applies to **all supported pairs**:

| A type ↔ B type          | Applied gap                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Any supported pair       | The sum of both groups' **Contact Gap** and **Contact Offset** values, plus a small safety margin so the closest pair starts just outside the contact band. |

The gap is in scene units and is read from the two groups' current
settings at the moment you click **Snap A to B**: the absolute
**Contact Gap** and **Contact Offset**, or, on a group that uses
**Use Group Bounding Box Diagonal**, the two ratios times that group's
bounding-box diagonal. **World Scaling** does not change it, and
neither does anything transferred earlier.

### Cross-Stitch Anchors

For every snap, the add-on also stores **per-vertex barycentric anchor
data** on the resulting merge pair.  Every A-vertex that ends up within
a small reach-threshold of B after alignment becomes its own anchor, so a
single snap typically produces **many stitches, one per coincident
vertex pair**, not just one.  Conceptually each source vertex is tied to
a target triangle (or a single target vertex for rod pairs) with
barycentric weights, so the stitch survives later mesh edits until the
topology itself changes.

The reach-threshold is derived from the applied gap (a small multiple of
it), which is why coinciding geometries work best: vertices that are
already on top of each other are well within threshold, while distant
vertices are ignored.  You can see the captured set as **yellow dots
connected by thin yellow lines** in the viewport overlay (toggle with
**Hide all snaps** in the Visualization panel).

```{figure} ../../images/snap_merge/stitch_overlay.png
:alt: Yellow stitch-pair overlay between the two subdivided plane patches. PatchA (magenta) has been pulled straight down in Z after the snap, exposing vertical yellow stitch lines connecting every seam vertex of PatchA to the matching seam vertex of PatchB (blue). The two patches were originally shifted along the X axis only, so every vertex along PatchB's right edge has a counterpart on PatchA's left edge.
:width: 720px

After `Snap A to B`, every A-vertex within reach-threshold of B
becomes a stitch anchor. For this X-axis-only shift, the entire
shared seam qualifies, so the overlay draws one yellow stitch per
seam-vertex pair. **PatchA has been pulled straight down in Z after
the snap** to make the pairs visible; the stitches themselves stay
tied to the original seam positions, separated by the small contact
gap rather than coincident.
```

### Stitching to a Static collider

A dynamic **Shell**, **Solid**, or **Rod** can be stitched onto a
**Static** collider, which lets you pin part of a moving mesh to a piece
of fixed geometry (for example tacking a sleeve to a static mannequin
panel). Pick the dynamic object as one side and the Static as the other,
then snap as usual.

The Static side is treated as the target that stays put: snap never moves
a Static collider, even if you set it as **Object A**. Instead, the
dynamic object is the one translated into place, and the contact gap is
left between them just as for any other pair. The stitch force then holds
the dynamic mesh against the fixed surface during the solve.

### Post Snap Exactly

**Post Snap Exactly** is a global toggle (not per-pair), found below the
**Merge Pairs** box. It is **on by default**.

Because stitches are soft forces, the simulated result keeps a small
contact gap at every seam. When **Post Snap Exactly** is on, fetching a
frame moves each stitched vertex exactly onto its stitch target, so seams
read as fully joined in the viewport. When it is off, the raw simulated
gap is kept, showing the true soft-stitch separation between the parts.

The toggle applies to **all stitch pairs at once**, and it only affects
what you see after fetching; it does not change how the solver computes
the simulation.

## Merge Pairs Without Snapping

A merge pair on its own only *registers* two objects; it does not yet
carry the stitch anchors the solver sews with. Those are captured by
**Snap A to B** (or **Re-snap** on an existing pair). A pair whose
stitch cannot reach the solver is kept as it is, with its stored stitch
untouched: **Transfer** stops with *Merge pair(s) that cannot stitch*,
naming each such pair and its reason, **Update Params on Remote**
refuses the same pairs, and the **Merge Pairs** box shows the same
reason under the selected pair. The reasons are:

- **No stitch points**: the pair was added without ever snapping, or
  the snap recorded none. Click **Re-snap** or remove the pair.
- **A mesh changed since the snap**: an object's vertex count differs
  from when the pair was snapped. Click **Re-snap**.
- **The stored stitch cannot be read**, or its rows are malformed.
  Click **Re-snap**.
- **A Solid side snapped before it recorded its stitch points**: a
  **Solid** side is placed on its tetrahedralized surface from points
  recorded at snap time, and a pair snapped without them cannot be.
  Click **Re-snap**.
- **An object is in no active dynamics group**: one of the pair's
  objects is not assigned to any existing group. Add it to a group
  again or remove the pair.

A merge pair is removed at one moment only: when one of its objects
leaves its group, through **Remove Object**, **Delete Group**, or by
being deleted from the scene. Nothing scans for pairs and removes them
later. Each pair has:

| UI label              | Python / TOML key   | Description                                                                                                                               |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Object A / B**      | `object_a` / `object_b` | The two mesh objects.                                                                                                                 |
| **Stitch Stiffness**  | `stitch_stiffness`  | Per-pair stiffness of the soft stitch force. Shown for every supported pair; raise it to hold the seam together more firmly. |
| **Show Stitch**       | `show_stitch`       | Overlay toggle for the viewport stitch preview.                                                                                           |

## Blender Python API

The same workflow is available from Python:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# Make the shirt hug the mannequin at its closest pair of verts, with the
# right contact gap for the SHELL <-> STATIC pairing. Also registers a
# merge pair with cross-stitch anchor data.
solver.snap("Shirt", "Mannequin")

# Or register a merge pair without moving anything. Snap / Re-snap must
# still run before Transfer, or the pair is refused (no stitch points).
solver.add_merge_pair("Shirt", "Mannequin")

# Iterate existing pairs.
for a, b in solver.get_merge_pairs():
    print(f"{a} <-> {b}")

# Remove a specific pair, or all of them.
solver.remove_merge_pair("Shirt", "Mannequin")
solver.clear_merge_pairs()
```

:::{note}
Snapping and merging both reject library-linked (non-writable) objects
because the solver needs to persist UUIDs on them. Make them local first if
you hit that error.
:::

:::{admonition} Under the hood
:class: toggle

**What Snap does**

Snap is a one-shot alignment:

1. Finds the source vertex whose closest point on the other object's
   nearest triangle (or rod segment) is globally minimal.
2. Translates the moving object in world space (parent-safe) along the
   approach direction so the closest pair ends a small positive gap apart:
   the summed `contact_gap + contact_offset` of both groups, scaled by a
   safety factor (1.1x) plus a tiny floor so the pair starts just outside
   the contact barrier's activation band and never coincident. A Static
   collider is never the moved object; the dynamic side moves instead.
3. Records per-vertex barycentric anchor data for every A-vertex close
   enough to B to participate in a stitch (within a small multiple of the
   applied gap).

**Cross-stitch anchor data**

Each merge pair carries the captured anchor payload:

- Source and target object UUIDs.
- Per source vertex, a six-wide row: indices `[si, si, si, t0, t1, t2]`
  (a degenerate source barycentric plus the target triangle) and weights
  `[1.0, 0.0, 0.0, α, β, γ]`.
- Target positions at snap time.
- Vertex counts at snap time, so stale entries can be detected if you
  edit the topology later.

The full barycentric anchor is what reaches the solver: each stitch row
keeps its three target vertices and their weights, remapped into the
assembled scene's index space. Only a rod *target* degenerates to a
single vertex with weight 1, which is the rod-to-rod case; a rod paired
with a shell or a solid is the source side, and still anchors to a
target triangle.

**Merge-pair encoding**

Merge pairs are tracked by UUID, so renaming either object preserves the
link. The **Transfer** check, the parameter encoder, and the panel line
all ask one function why a pair cannot stitch, so none of them can pass
a pair another refuses, and none of them writes to the pair. A pair
with a **Solid** side needs the six-wide rows and the snap-time surface
points, so a pair stored with four-wide rows is upgraded to six-wide
rows at encode time only when neither side is a **Solid**.
:::
