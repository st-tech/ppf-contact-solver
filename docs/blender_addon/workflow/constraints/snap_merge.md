# 🧲 Snap and Merge

Two separate steps that work together:

- **Snap** is a one-shot alignment: it translates object A in world space
  so that its closest vertex lines up with the nearest vertex on object B,
  leaving just enough contact gap to avoid interpenetration. In the same
  pass it also captures stitch anchors for **every A-vertex within reach
  of B**, not just the single closest one.
- **Merge** is a solver-side stitch constraint between two objects. At
  transfer time it becomes a cross-object constraint that keeps the two
  meshes joined during the simulation.

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
:alt: Two side-by-side panels. Left, Matching topology: a 4x4 grid patch B in green with its copy A in blue offset slightly; every A-vertex is connected to its B counterpart by a short red dashed line; 16 of 16 pairs recorded. Right, Partial overlap: the same B patch at the bottom with a separate 4x4 A patch floating above and to the right such that only A's bottom-left corner vertex coincides with B's top-right corner vertex; a single red dashed line connects that one pair; the other 15 A-vertices are labeled "no pair"; 1 of 16 pairs recorded. A footer reminds the reader that snap is a geometric alignment step, not a topology repair.
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
eyedroppers, then press **Snap A to B**. The add-on finds the closest
pair of vertices between A and B, translates A (in world space,
parent-safe) so they line up, applies the contact-gap rules below, and
records per-vertex barycentric anchor data for a later stitch. It also
registers a merge pair automatically.

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
:alt: After Snap A to B. The magenta PatchA has translated along the X axis and now sits flush against the blue PatchB, left-edge to right-edge, in the same plane. Together they read as a single seamless 2x1 rectangle, with every vertex on the shared seam coincident.
:width: 520px

**After `Snap A to B`.** PatchA has translated along the X axis in
world space until its nearest vertex lands on PatchB. Because both
patches are **Shell**, the gap rule is "merge exactly": PatchA's
left edge now coincides with PatchB's right edge and every seam
vertex is shared, so the two patches read as a single seamless 2x1
rectangle. The merge pair is registered automatically and the
per-vertex stitch anchors are captured behind the scenes.
```

As soon as at least one merge pair exists, a second box labeled
**Merge Pairs** appears below the snap box, containing:

- A UIList showing each pair, with both object names per row.
- A **Remove Merge Pair** button below the list (disabled unless a row
  is selected).
- A **Stitch Stiffness** slider, **only shown when the selected pair
  involves a Solid**. Sheet-sheet (Shell-Shell) and rod-rod pairs
  merge vertices exactly, so stiffness has no meaning for them.

A separate **Visualization** panel further down the sidebar exposes a
**Hide all snaps** toggle that hides or shows the merge-pair / stitch
overlay in the viewport.

### Gap Rules

The solver needs a small separation between the two meshes at rest,
otherwise contact barriers start flagging penetration on frame 1. The snap
operator picks the gap based on the group types of A and B:

| A type ↔ B type          | Applied gap                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Shell ↔ Shell            | **No gap**. Vertices merge exactly.                                                     |
| Rod ↔ Rod                | **No gap**. Vertices merge exactly.                                                     |
| Any other pair           | The larger of the two groups' **Contact Gap** values plus both groups' **Contact Offset**. |

### Cross-Stitch Anchors

For every snap, the add-on also stores **per-vertex barycentric anchor
data** on the resulting merge pair.  Every A-vertex that ends up within
a small reach-threshold of B after alignment becomes its own anchor, so a
single snap typically produces **many stitches, one per coincident
vertex pair**, not just one.  Conceptually each source vertex is tied to
a target triangle (or a single target vertex for rod pairs) with
barycentric weights, so the stitch survives later mesh edits until the
topology itself changes.

The reach-threshold is derived from the applied gap (roughly `2 × gap`),
which is why coinciding geometries work best: vertices that are already
on top of each other are well within threshold, while distant vertices
are ignored.  You can see the captured set as **yellow dots connected by
thin yellow lines** in the viewport overlay (toggle with **Hide all
snaps** in the Visualization panel).

```{figure} ../../images/snap_merge/stitch_overlay.png
:alt: Yellow stitch-pair overlay between the two subdivided plane patches. PatchA (magenta) has been pulled straight down in Z after the snap, exposing vertical yellow stitch lines connecting every seam vertex of PatchA to the matching seam vertex of PatchB (blue). The two patches were originally shifted along the X axis only, so every vertex along PatchB's right edge has a counterpart on PatchA's left edge.
:width: 720px

After `Snap A to B`, every A-vertex within reach-threshold of B
becomes a stitch anchor. For this X-axis-only shift, the entire
shared seam qualifies, so the overlay draws one yellow stitch per
seam-vertex pair. **PatchA has been pulled straight down in Z after
the snap** to make the pairs visible; the stitches themselves remain
attached to the original coincident seam positions.
```

## Merge Pairs Without Snapping

A merge pair alone (without snapping) registers two objects as stitched
during the solve. Each pair has:

| UI label              | Python / TOML key   | Description                                                                                                                               |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Object A / B**      | `object_a` / `object_b` | The two mesh objects.                                                                                                                 |
| **Stitch Stiffness**  | `stitch_stiffness`  | Per-pair stiffness. **Only shown for pairs involving a Solid.** Sheet-sheet (Shell-Shell) and rod-rod pairs merge vertices exactly, so stiffness has no meaning. |
| **Show Stitch**       | `show_stitch`       | Overlay toggle for the viewport stitch preview.                                                                                           |

Merge pairs referencing deleted or unassigned objects are cleaned up
automatically on the next depsgraph update.

## Blender Python API

The same workflow is available from Python:

```python
from zozo_contact_solver import solver

# Make the shirt hug the mannequin at its closest pair of verts, with the
# right contact gap for the SHELL <-> STATIC pairing. Also registers a
# merge pair with cross-stitch anchor data.
solver.snap("Shirt", "Mannequin")

# Or register a merge pair without moving anything.
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

1. Finds the closest pair of vertices between objects A and B.
2. Translates A in world space (parent-safe) along the approach
   direction so the two vertices line up, plus the gap-rule distance and
   a small float32 safety margin.
3. Records per-vertex barycentric anchor data for every A-vertex close
   enough to B to participate in a stitch.

For the no-gap pairings (Shell-Shell, Rod-Rod) the anchor-capture
threshold falls back to `max(gap_a, gap_b)` so nearby vertices still
enter the stitch even though the final separation is zero.

**Cross-stitch anchor data**

Each merge pair carries the captured anchor payload:

- Source and target object UUIDs.
- Per source vertex: target triangle indices and barycentric weights
  `[1.0, α, β, γ]`.
- Target positions at snap time.
- Vertex counts at snap time, so stale entries can be detected if you
  edit the topology later.

When the pair is sent to the solver, the target vertex with the highest
barycentric weight is picked per stitch. For rod-to-rod / rod-to-shell
the target degenerates to a single vertex with weight 1.

**Merge-pair encoding**

Merge pairs are tracked by UUID, so renaming either object preserves the
link. Pairs referencing an object that has never been snapped or merged
(no UUID yet) are skipped at transfer. An empty cross-stitch payload
means snap has not run or the pair is not eligible for a stitch.
:::
