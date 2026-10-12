# 🧲 Snap and Merge

**Snap A to B** moves one object onto another and joins the two with a soft
stitch. Use it to sew a garment panel to a body, to join two cloth pieces
along a seam, or to attach a rod to a cloth edge. One click:

1. Moves object A next to object B, so that the closest pair ends the
   applied gap apart (see [Gap Rules](#gap-rules)). When A is a **Static**
   collider, object B moves instead. Objects whose closest points already
   coincide are not moved.
2. Records a stitch for every vertex of the source object (see
   [Which Pairs Can Be Stitched](#which-pairs-can-be-stitched)) that ends
   up within reach of the other object (see
   [What a Snap Records](#what-a-snap-records)).
3. Registers the two objects as a **merge pair**.

At **Transfer**, the solver turns the pair's stitches into a soft spring
force between the two objects. It is a spring, not a weld: its rest length
comes from the two objects' **Contact Gap**, and the two objects stay
separate meshes.

The merge pair is what the solver turns into stitch forces, and a snap is
what gives the pair its stitch points. A pair registered from Python or MCP
without a snap has no stitch points and is refused at **Transfer** (see
[Merge Pairs Without Snapping](#merge-pairs-without-snapping)).

:::{important}
**Snap is a set-level operation, not a single-vertex operation.** It
stitches every vertex of the source object that is within reach of the
other object, so it works best on meshes whose geometry already lines up
along the seam. For example, take two identically subdivided square
patches that lie side by side with a gap between them: every vertex along
the edge of one patch has a counterpart on the other patch, and the whole
seam is stitched in one click.

If only one vertex is within reach, only that one is stitched. A vertex is
stitched when the other object's surface is within reach, whether or not a
vertex of the other mesh lies there. A rod target is the exception: it is
measured to the nearest sampled point of the rod, so rod points must line
up. Author the meshes so that the intended seam lines up before you snap.
:::

```{figure} ../../images/snap_merge/snap_set_level.svg
:alt: Two side-by-side panels. Left, "Edges aligned": a 4x4 patch A in blue on the left and patch B in green on the right, with A's right edge and B's left edge at the same four row heights; four short red dashed lines pair the seam vertices; a badge reads "4 / 4 edge pairs recorded". Right, "Edges misaligned": the same two patches with B shifted down so only A's bottom-right corner shares a row with B's top-left; one red dashed line connects that pair, three A-edge vertices are labeled as having no B match, and a badge reads "1 / 4 edge pairs recorded". A footer states that snap is a geometric alignment step, not a topology repair.
:width: 820px

Snap records a stitch for every source vertex within reach of the other
object, and for no other vertex. Left: matching subdivided patches produce
a full set of stitches in one pass. Right: a patch with only one corner
lining up with the other patch yields a single stitch, and the rest of the
seam stays unattached. In this schematic A is blue and B is green. The
reach is defined under [What a Snap Records](#what-a-snap-records).
```

## Snap and Merge Panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab.
The **Snap and Merge** panel sits below **Scene Configuration** and
**Dynamics Groups**, and is **collapsed by default**. Click the header to
expand it. The panel has three parts:

- A **Snap To Nearby Vertices** box with the **Object A (moves):** and
  **Object B (target):** pickers and the **Snap A to B** button.
- A **Merge Pairs** box, drawn once at least one pair exists.
- The **Post Snap Exactly** checkbox, always drawn below the boxes.

### Snap Two Objects

Before you snap:

- Add both objects to dynamics groups whose types form a supported pair
  (see [Which Pairs Can Be Stitched](#which-pairs-can-be-stitched)). The
  pickers list a mesh once it has been added to a group, and a curve only
  while it is in a **Rod** group. A pair of unsupported types, or an object
  in no group, is refused with *This object pair does not support
  stitching*.
- Use writable objects. Snap and merge reject library-linked objects,
  because the add-on cannot write to them. Make them local first.
- Put the timeline on the **Starting Frame**. Snap reads each object's
  shape at the current frame, with its modifiers and rig applied, while
  **Transfer** reads it at the starting frame. Snapping there records the
  stitch against the pose the solver starts from.

To snap:

1. Choose **Object A (moves):** in the first dropdown. Alternatively,
   select the object in the viewport or the Outliner and click the
   eyedropper button beside the dropdown. The button sets the field to the
   active object. If the active object is not in the dropdown, the add-on
   reports *'name' is not a snap-eligible object*.
2. Choose **Object B (target):** the same way.
3. Click **Snap A to B**. The button stays disabled until two different
   objects are chosen. The status bar reports, for example, *Stitched PatchA
   to PatchB (was 0.5000 apart, positioned at gap 0.00221)*. The message
   appears even when no stitch was recorded.
4. Check the stitch lines in the viewport and the **Merge Pairs** box. If
   the box shows a line with an error icon saying the pair has no stitch
   points, the snap found no vertex of the source within reach of the
   target (for a rod target, of a sampled point of the rod). Edit the
   geometry so that a vertex of the source lies closer to the target, then
   click **Re-snap**, or remove the pair.
5. Click **Transfer**. The snap moved one of the two objects, and the
   solver needs the moved geometry together with the new pair.

Snap moves an object by changing its location, so the mesh data is
untouched, and it also works when the object has a parent.

```{figure} ../../images/snap_merge/panel.png
:alt: Snap and Merge panel expanded. The Snap To Nearby Vertices box shows Object A (moves) set to PatchA and Object B (target) set to PatchB, each with an eyedropper button, above the Snap A to B button.
:width: 500px

The **Snap To Nearby Vertices** box with **PatchA** chosen as Object A
(the object that moves) and **PatchB** as Object B (the target).
Clicking **Snap A to B** runs the snap on the pair set here.
```

```{figure} ../../images/snap_merge/snap_before.png
:alt: Before Snap A to B. Two identically subdivided square shell patches in the 3D viewport. PatchB (blue) sits at the origin and PatchA (magenta, selected) is shifted along the X axis only, with the same Y and the same Z plane. A clear gap separates PatchB from PatchA, and the patches are not joined.
:width: 520px

**Before.** Two identically subdivided patches, each in a **Shell**
group: **PatchB** (blue, at the origin, stays put) and **PatchA**
(magenta, shifted along the X axis only, same Y and same Z). A clear gap
separates them.
```

```{figure} ../../images/snap_merge/snap_after.png
:alt: After Snap A to B. The magenta PatchA has moved along the X axis and lies edge to edge with the blue PatchB in the same plane, so the two read as one 2x1 rectangle. The gap between them is too small to see at this scale.
:width: 520px

**After Snap A to B.** PatchA has moved along the X axis until its nearest
vertex sits the applied gap from PatchB (see [Gap Rules](#gap-rules)). The
patches are not welded: every seam vertex on PatchA is stitched to its
counterpart on PatchB by a recorded soft stitch, and the two stay separate
meshes. The gap is too small to see at this scale.
```

### The Merge Pairs Box

The **Merge Pairs** box appears below the snap box once at least one pair
exists. It contains:

- A list with one row per pair, showing the two object names as
  *A ↔ B*. The eye icon at the right end of a row is **Show Stitch**: it
  shows or hides that pair's stitch lines. An error icon in front of the
  names means one of the two objects is missing from the file.
- **Re-snap**, below the list. It repeats the whole snap on the selected
  pair: it moves the pair's Object A (or the non-Static object) to the
  applied gap again and records the stitch points again. Put the objects
  where you want them before you click it. Use it after you edit either
  mesh, or when the box reports a problem (see
  [Pairs That Cannot Stitch](#pairs-that-cannot-stitch)). It is disabled
  until a row is selected.
- **Remove**, below that. It deletes the selected pair and moves nothing.
  It is disabled until a row is selected.
- **Stitch Stiffness**, a numeric field for the selected pair. It is drawn
  when the pair's two types form a supported pair.
- When the selected pair cannot stitch, a line with an error icon that
  gives the reason, the same reason **Transfer** gives.

| UI control | Key | Default | Description |
| --- | --- | --- | --- |
| **Stitch Stiffness** field | `stitch_stiffness` | 1.0 | Scales the pair's soft stitch force directly. Minimum 0 (the pair is kept but applies no force), soft maximum 10000. Raise it to hold the seam together more firmly. Separate from the group's **Stitch Stiffness** in [Material Parameters](../params/material.md), which applies to loose edges inside one object. Send a change with **Update Params on Remote**. |
| Eye icon in the pair's row (**Show Stitch**) | `show_stitch` | on | Shows or hides this pair's stitch lines in the viewport. It does not affect the solve. |

The keys are the Blender property names, and the MCP tool
`set_merge_pair_properties` takes the same names. It and
`resnap_merge_pair` also take `object_a` and `object_b`, the names of the
pair's two objects in either order (see
[Blender Python API and MCP](#blender-python-api-and-mcp)).

The overlay draws each stitch as a yellow line from the stitched vertex
(orange dot) to the point on the other object it is stitched to (yellow
dot). When a fetched result is shown, the lines follow the fetched
positions. A pair's lines are drawn only while its eye icon is on and
**Hide all snaps** is off. That toggle is in the **Visualization** panel
and hides the lines of every pair at once.

```{figure} ../../images/snap_merge/stitch_overlay.png
:alt: Yellow stitch lines between the two subdivided patches. PatchA (magenta) has been moved straight down in Z after the snap, which stretches vertical yellow lines from every seam vertex of PatchA to the matching point on PatchB (blue). The patches were first shifted along the X axis only, so every vertex along PatchB's right edge has a counterpart on PatchA's left edge.
:width: 720px

The stitch overlay after **Snap A to B**. **PatchA** was moved straight
down in Z afterward so the lines are long enough to see. Each stitch stays
attached to the same vertex of PatchA and the same point on PatchB, so
moving PatchA stretches the lines.
```

A pair is removed automatically when either of its objects leaves its
group: through **Remove Object**, **Delete Group**, **Delete All Groups**,
**Bake Animation** or **Bake Single Frame** (baking moves the baked
objects out of their groups), or by deleting the object. Otherwise a pair
stays until you click **Remove**.

### Post Snap Exactly

**Post Snap Exactly** is a global checkbox (not per pair), drawn below the
boxes. It is **on by default**. It only changes what you see after
fetching and does not change how the solver computes the simulation.

- **On:** fetching a frame moves each stitched vertex exactly onto its
  stitch target, so seams read as fully joined in the viewport. The same
  toggle closes the loose-edge seams inside a **Shell** or **Solid**
  object: the two ends of a seam that are closer than the object's
  **Contact Offset** plus twice its **Contact Gap** are moved to their
  midpoint.
  **Rod** groups are not affected by this part.
- **Off:** fetched frames show the simulated positions, including the
  separation the soft stitch leaves.

From a script or an MCP client, set it as `post_snap_exactly` (see
[Blender Python API and MCP](#blender-python-api-and-mcp)).

## Snap Rules

### Which Pairs Can Be Stitched

The supported pairs are **Shell** and **Shell**, **Solid** and **Solid**,
**Rod** and **Rod**, **Shell** and **Solid**, **Rod** and **Shell**,
**Rod** and **Solid**, and a **Static** collider with a **Shell**, a
**Solid** or a **Rod**. **PDRD** and **Sand** groups, two **Static**
colliders, and objects that are in no group cannot be stitched.

The group types decide which object moves, which object supplies the
stitched vertices (the source), and which one they are stitched to (the
target):

| Object A and Object B | Moves | Source (its vertices are stitched) | Target |
| --- | --- | --- | --- |
| **Shell** and **Shell** | A | A | B |
| **Solid** and **Solid** | A | A | B |
| **Rod** and **Rod** | A | A | B |
| **Shell** and **Solid**, either order | A | the Shell | the Solid |
| **Rod** and **Shell**, either order | A | the Rod | the Shell |
| **Rod** and **Solid**, either order | A | the Rod | the Solid |
| **Shell**, **Solid** or **Rod** with a **Static**, either order | the object that is not Static | the object that is not Static | the Static |

Object A moves unless it is a **Static** collider, in which case object B
moves. When both objects have the same type, Object A is the source and
Object B the target. Otherwise the group types decide, as the table shows.

### Gap Rules

The snap moves the closest pair to the applied gap, so the pair starts
outside the combined **Contact Gap** and **Contact Offset** of the two
objects. The one exception is a pair whose closest points already coincide
(closer than 1e-9 Blender units): the snap does not move it and applies no
gap, so separate such objects slightly before you snap. The applied gap is
the same for every supported pair: 1.1 times the sum of both objects'
**Contact Gap** and **Contact Offset**, plus 0.00001, in Blender units. The
factor puts the closest pair strictly outside that range, and the added
floor prevents the objects from coinciding when both values are 0.

The values are read from the two groups when you click **Snap A to B**:
the absolute **Contact Gap** and **Contact Offset**, or, on a group with
**Use Object Bounding Box Diagonal** on, **Contact Gap Ratio** and
**Contact Offset Ratio** times that object's own bounding-box diagonal.
**World Scaling** does not change them, because the snap works in Blender
units. A group's [Local Gap Limit](../params/material.md#local-gap-limit),
which can lower an element's gap on a fine mesh when the scene is built,
does not change them either: the snap uses the gap you authored.

### What a Snap Records

After the move, the snap stitches every source vertex that lies within the
**stitch reach** of the other object: 2.5 times the applied gap, and at
least 0.0001 Blender units. The distance is measured to the nearest point
on the target's surface. For a rod target it is measured to the nearest
sampled point of the rod. A mesh target needs no vertex where the source
vertex lands: the stitch is attached to the nearest point of a target
triangle. Vertices that coincide with the other object are well inside the
reach, and distant vertices are ignored, which is why coinciding geometry
works best (see the figure at the top of the page).

Each stitch stores the point on the other object that its vertex is
attached to, so the stitch stays attached to the same vertices when the
objects move. It stays valid while both meshes keep their vertex count.
After you add or delete vertices on either object, click **Re-snap**.
Reordering vertices keeps the count and is not detected, so click
**Re-snap** after that too. The count is checked for mesh objects only.

### Stitching to a Static Collider

A dynamic **Shell**, **Solid** or **Rod** can be stitched to a **Static**
collider, which holds part of a dynamic mesh against the collider (for
example a sleeve tacked to a mannequin panel). Choose the dynamic object as
one side and the Static as the other, then snap as usual.

The Static is always the target, and snap never moves it, even when it is
Object A. The dynamic object moves into place and keeps the applied gap
from the Static. The stitch force then holds the dynamic mesh against the
Static during the solve. If the Static is animated, it keeps following its
own motion and the stitch follows it.

## Merge Pairs Without Snapping

A pair registered without a snap (from Python or MCP) has no stitch
points, and **Transfer** refuses it. Click **Re-snap** on the pair (see
[The Merge Pairs Box](#the-merge-pairs-box)) to record its stitch points,
or remove the pair. A pair that has stitch points can also be unable to
stitch, for the reasons listed next.

### Pairs That Cannot Stitch

The same refusal covers every state in which a pair's stitch cannot reach
the solver. The pair and its stored stitch stay as they are. **Transfer**
stops with *Merge pair(s) that cannot stitch*, naming each pair and its
reason, **Update Params on Remote** refuses the same pairs, and the
**Merge Pairs** box shows the reason under the selected pair. The reasons
are:

- **No stitch points**: the pair was added without a snap, or the snap
  found no vertex of the source within reach of the target. A rod target
  is measured to the nearest sampled point of the rod, so a rod whose
  points are far apart can leave every vertex out of reach. Click
  **Re-snap**, or remove the pair.
- **A mesh changed since the snap**: an object's vertex count differs from
  when the pair was snapped. Click **Re-snap**.
- **The stored stitch is damaged**: it cannot be read, it is incomplete or
  inconsistent, or it names an object that is in no active dynamics group.
  Click **Re-snap**.
- **A Solid side has no recorded stitch points**: the pair was snapped
  before a **Solid** side recorded its stitch points, so that side cannot
  be placed. Click **Re-snap**.
- **An object has no UUID**, the unique ID the add-on stores on each
  object: remove the pair and snap again.
- **An object is in no active dynamics group**: one of the pair's objects
  is not assigned to any group. Add it to a group again or remove the
  pair.

## Blender Python API and MCP

The same workflow is available from Python. Both objects must already be
in groups that form a supported pair. The import below works for an
add-on in the **User Default** repository. For an add-on installed from a
remote repository, use the lookup under
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

solver.create_group("Shirt", type="SHELL").add("Shirt")
solver.create_group("Mannequin", type="STATIC").add("Mannequin")

# Move the shirt next to the mannequin, leave the applied gap and record
# the stitch points. Also registers the merge pair.
solver.snap("Shirt", "Mannequin")

# Register a pair without moving anything. It forms no stitch until a snap
# records stitch points, and Transfer refuses it until then.
solver.add_merge_pair("SleeveLeft", "BodyLeft")

# Iterate existing pairs.
for a, b in solver.get_merge_pairs():
    print(f"{a} <-> {b}")

# Remove a specific pair, or all of them.
solver.remove_merge_pair("Shirt", "Mannequin")
solver.clear_merge_pairs()
```

Calling `solver.snap` again on an existing pair records its stitch points
again. An object name that does not resolve raises `ValueError`, and a snap
the add-on refuses (an unsupported pair, for example) raises
`RuntimeError` carrying the message the button shows.

Over MCP, `snap_to_vertices(object_a, object_b)` is **Snap A to B**: it
moves object A (object B when A is in a **Static** group), records the
stitch points and registers the pair, with the same gap as the button.
`add_merge_pair` and `remove_merge_pair` register and remove a pair
without moving anything, `clear_merge_pairs` removes every pair, and
`resnap_merge_pair`, `list_merge_pairs` and `set_merge_pair_properties`
(per-pair **Stitch Stiffness** and **Show Stitch**) cover the rest of the
**Merge Pairs** box. See the
[MCP Tool Reference](../../integrations/mcp_reference.rst).

**Post Snap Exactly** is `solver.param.post_snap_exactly` in Python and the
`post_snap_exactly` argument of the MCP tool `set_scene_parameters`.
Changing it takes effect on the next fetch and needs no new run.

:::{admonition} Under the hood
:class: toggle

**Closest pair and move**

Snap finds the source vertex whose closest point on the target's nearest
triangle (or rod segment) is globally minimal. The search is feature-based:
it measures to the closest point on a triangle or segment, not to the
nearest vertex. The moving object is translated along the approach
direction by changing its location, so the closest pair ends the applied
gap apart. When the moving object is the target of the stitch (Object A is
a Solid stitched to a Shell, for example), the translation is inverted so
the target is pulled toward the source.

**Stored stitch**

Each merge pair carries the captured payload:

- Source and target object identifiers.
- Per source vertex, a six-wide row: indices `[si, si, si, t0, t1, t2]`
  (a degenerate source barycentric plus the target triangle) and weights
  `[1.0, 0.0, 0.0, α, β, γ]`.
- Source and target positions at snap time, one per row. A **Solid** side
  needs them to be placed on its tetrahedral surface.
- Vertex counts at snap time for mesh objects, so a changed mesh can be
  detected.

The full barycentric anchor is what reaches the solver: each stitch row
stores its three target vertices and their weights. Only a rod *target*
degenerates to a single vertex with weight 1, which is the rod-to-rod case;
a rod paired with a shell or a solid is the source side and still anchors to
a target triangle.

**Merge-pair encoding**

Merge pairs are tracked by object identifier, so renaming either object
preserves the link. The **Transfer** check, the parameter encoder and the
panel line all call one function that returns the reason a pair cannot
stitch, so they accept and reject the same pairs, and none of them writes
to the pair. A pair with a **Solid** side needs the six-wide rows and the
snap-time positions, so a pair stored with four-wide rows is upgraded to
six-wide rows at encode time only when neither side is a **Solid**.
:::
