# 💇 Hair Curves

A **Hair Curves** object joins a **Rod** group and is simulated as strands.
This is the object that **Add > Curve > Empty Hair** and **Fur** create, and
that **Object > Quick Effects > Quick Fur** grows on a mesh. Every point of
every curve is one vertex of a simulated strand, so the groom you see is what
the solver receives, and the solve plays back on those same points.

Hair that Blender builds on top of the curves is not simulated. In a **Fur**
setup the curves you edit are the *guides*, and the **Interpolate Hair
Curves** modifier generates the thousands of children from them. The solver
moves the guides, and Blender rebuilds the children from the result on every
frame.

This page covers everything specific to hair:

- [Adding hair to a group](#adding-hair-to-a-group) and what joining sets
- [What a strand has to be](#what-a-strand-has-to-be)
- [Pins on hair](#pins-on-hair): the **Roots** pin, pins from a selection, and
  what a pin does and does not hold
- [Weighted pins](#weighted-pins): a hold that fades along the strand
- [Stiffness along a strand](#stiffness-along-a-strand): a material that
  varies from root to tip
- [Hair on a moving body](#hair-on-a-moving-body)
- [Playback](#playback)
- [What hair does not do](#what-hair-does-not-do)
- [Python and MCP](#python-and-mcp)

```{figure} ../../images/hair/hair_weights_shake.jpg
:alt: Two furry spheres side by side in mid shake. The fur of the left one, labeled Roots only, lies flat against the sphere and is swept to one side. The fur of the right one, labeled Weighted hold plus stiffness map, still stands out from the sphere all around, with only its tips trailing.
:width: 640px

Two heads with the same **Quick Fur**, shaken from side to side. On the left
the fur has only its **Roots** pin and lies down as soon as the head moves. On
the right a [weighted pin](#weighted-pins) holds the lower half of every
strand, firmly at the root and more loosely toward the middle, and a
[stiffness map](#stiffness-along-a-strand) makes each strand stiff at its base
and limp at its tip.
```

## Adding Hair to a Group

Create a group, set its **Type** to **Rod**, select the hair object and click
**Add Selected Objects** (see [Object Groups](object_groups.md)). Only a
**Rod** group takes a Hair Curves object. In a selection added to a group of
any other type, the hair is skipped.

Joining a group sets three things that the hair itself already implies. Each
one is reported in Blender's status bar as it is made:

| Report                                                                                                  | What was set                                                                                              | When                                                                    |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| *'Fur' is attached to 'Head', so its roots are pinned*                                                  | A **Roots** pin is added to the group's pin list (see [Pins on hair](#pins-on-hair)).                     | The hair has a **Surface** object, as Quick Fur hair does.              |
| *Rest Angle set to From Initial Geometry so 'Fur' keeps its groom*                                      | **Rest Angle** in the **Bend** box changes from **Flat / Straight** to **From Initial Geometry**.         | The hair is the first object of the group.                              |
| *Contact Offset set to zero: a strand of 'Fur' is held off other surfaces by the Contact Gap alone*     | **Contact Offset** changes from the `0.005` a Rod group starts with to `0`.                               | The hair is the first object of the group.                              |

A group that already holds objects keeps the settings that were chosen for
them, so only the **Roots** pin is added when hair joins a group that is not
empty. All three are ordinary settings that you can change afterward.

The two material changes matter for hair in particular:

- With **Rest Angle** at **Flat / Straight**, every strand would try to
  straighten. **From Initial Geometry** makes the groomed shape the shape the
  strand rests in.
- A **Contact Offset** of `0.005` is the radius of a rope. Hair combed closer
  to the scalp than that would start inside its own offset, and **Transfer**
  refuses a scene that starts that way. With the offset at `0`, the **Contact
  Gap** alone keeps a strand off the scalp and off its neighbors.

## What a Strand Has to Be

**Transfer** stops, naming the object and the group, on hair it cannot
simulate as strands:

- **An object with no curves.** Add curves to it, or remove it from the
  group.
- **A curve with a single point.** A strand segment needs two points. Delete
  those curves or give them a second point.
- **A Bezier or NURBS curve.** Their control points do not lie on the strand.
  In Edit Mode, select everything and use **Curves > Set Curve Type** to make
  the curves **Catmull Rom** or **Poly**. Catmull Rom is what Blender's hair
  tools create.

The number of points on a curve is the resolution the strand is simulated at.

## Pins on Hair

A Hair Curves object has no vertex groups. A pin on hair is a named
**attribute** on its points, which Blender carries through edits to the groom
the way it carries a vertex group through edits to a mesh:

- a **Boolean** attribute holds the points it marks;
- a **Float** attribute holds every point whose value is above `0`, and is a
  [weighted pin](#weighted-pins).

The selector row of the **Pins** section (see
[The Pins Section in a Group Box](../constraints/pins.md#the-pins-section-in-a-group-box))
lists one `[Object][Attribute]` entry for each such attribute of each hair
object in the group. Blender's own point attributes `radius`, `tilt` and
`nurbs_weight` are not listed, because they mean something else.

### The Roots Pin

The selector always offers one more entry per hair object, `[Object][Roots]`.
**Roots** is not stored anywhere. It is the first two points of every curve,
worked out from the curves each time it is read, so it follows the groom
through any edit: add a strand and that strand is rooted too. Hair that has a
**Surface** object gets its **Roots** pin when it joins the group.

**Roots** cannot be renamed, because there is nothing stored to rename. To
hold a different set of points under that name, create an attribute called
`Roots`; it replaces the derived pin.

### A Pin From a Selection

Enter Edit Mode on the hair, select the points to hold, and click **Create**
in the **Pins** section. Name the pin in the dialog (it defaults to `pin`)
and confirm. The add-on stores the selection as a Boolean point attribute of
that name and adds the pin. A name that already belongs to an attribute that
cannot be a pin is refused, so the attribute is not overwritten.

**Rename**, **Select** and **Deselect** work on a hair pin as they do on a
mesh pin. **Make Keyframe** and **Delete All Keyframes** are grayed out: hair
points take no keyframes, so a hair pin moves only through
[Capture Deformation](#hair-on-a-moving-body).

### What an Exact Pin Holds on a Strand

A pin with **Pull** off holds its points exactly where they are, or exactly
where a capture puts them. On a strand that has one consequence worth knowing
before you tune a groom:

**A strand turns freely about the last point an exact pin holds, whatever its
Bend Stiffness.** The **Roots** pin holds the first two points of a strand,
so the first segment stays on the scalp, and the rest of the strand swings
from the second point like a hinge. Raising **Bend Stiffness** keeps the
strand straight. It does not keep the strand pointing the way it was combed.

To keep hair standing in its groomed direction, hold the points after the
roots with a [weighted pin](#weighted-pins) that pulls. The figure at the top
of this page shows the difference: the fur on the left has only its **Roots**
pin.

## Weighted Pins

A weighted pin gives every point its own weight between `0` and `1`. With
**Pull** on, each point is pulled toward its place with the pin's
**Strength** times its weight, so a hold can be firm at the root and fade
along the strand. A point with a weight of `0` is not held at all.

### Making One

A weighted pin is a Float point attribute on the groom. There are three ways
to make one:

1. **From a soft selection.** In Sculpt Mode a selection is a value per
   point, not on or off: the **Select** brush paints it, and **Select >
   Random** with **Partial** on gives partial values too. Switch to Edit
   Mode, which keeps those values, and click **Create**. When any
   point is partly selected, the pin is stored with the selection values as
   its weights. A selection in which every point is fully in or out makes an
   ordinary Boolean pin.
2. **From Geometry Nodes.** A **Store Named Attribute** node on the point
   domain writes one. A **Spline Parameter** node's **Factor** runs from `0`
   at the root to `1` at the tip, so `1 - Factor` is a weight that fades from
   the root. **Apply** the modifier afterward: a pin is read from the groom
   itself, so an attribute that only a live modifier produces is not offered
   in the selector.
3. **From a script.** `create_pin` and the MCP tool `add_pin_vertex_group`
   take a `weights` list beside `indices`. See
   [Python and MCP](#python-and-mcp).

Pick the attribute in the selector and click **Add**, unless **Create** has
already added it.

### Pull, Strength and the Weights

The weights count only while the pin pulls. Below the **Pull** row, the panel
says which case applies:

| Line under the Pull row                                  | Meaning                                                                                              |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| *Each point is pulled with Strength x its weight*        | **Pull** is on. A point at weight `0.25` is pulled a quarter as hard as a point at weight `1`.       |
| *Weighted pin: turn on Pull for the weights to count*    | **Pull** is off. The pin holds every point with a weight above `0` exactly, and all of them alike.   |

```{figure} ../../images/hair/hair_weighted_pin_panel.png
:alt: The Pins section of a Rod group named Fur. The pin list holds two rows, Fur Hold, which is selected, above Fur Roots. Below the list, Pull is ticked with Strength at 300, and the line under it reads Each point is pulled with Strength x its weight.
:width: 500px

The **Pins** section of a hair group with a weighted pin, `[Fur][Hold]`,
selected. It is listed above `[Fur][Roots]`, **Pull** is on, and the line
under the **Pull** row says how the weights are used.
```

A weight above `1` is read as `1`, since **Strength** is the pull at a weight
of `1`. A weight that is not a finite number stops **Transfer** with a message
naming the pin, the object and the point.

### Let the Weight Reach Zero Where Hair Should Move

A point held by a pulling pin does not fall and does not swing on its own.
The pull replaces the point's own weight and momentum: the point stays at its
place unless the rest of the strand, or a contact, drags it away, and then it
gives way by an amount that depends on **Strength** and its weight.

**A pulling pin whose weight is above zero along the whole strand therefore
holds the whole strand still**, at any **Strength**. For hair that falls and
swings, let the weight reach `0` partway along the strand. The free part then
moves with gravity and the head's motion, and the held part bends after it as
far as its weights allow. In the figure at the top of this page, the weights
are `1` on the first segment and fall in equal steps to `0` at the middle of
each strand.

This differs from the goal weights of Blender's own hair dynamics, where a
weighted point keeps its own motion and is drawn toward its goal.

### Put Roots Below the Weighted Pin

Where two pins hold the same point, the pin that is lower in the list
supplies the settings. A weighted pin that starts at the root covers the
first segment, which **Roots** also holds. Listed below **Roots**, the
weighted pin wins those points and the roots are held softly, so the hair
can slide on the scalp. Hair with a **Surface** gets its **Roots** pin first,
so a pin added afterward lands below it.

Select **Roots** and move it down with the **▼** button until it is below the
weighted pin. The panel reminds you while the order stands the other way: a
pulling hair pin that has an exact pin of the same object above it shows the
line *Pins above lose shared points to this one: move Roots below*.

## Stiffness Along a Strand

A **Rod** group has a **Spatial Material Maps** box, between the collision
windows and the **Bend** box. A map varies one material parameter along the
strands instead of holding it constant over the group: the value at a point
is a blend between the group's own slider, at weight `0`, and the map's
**Target**, at weight `1`. [Spatial Material Maps](../params/material.md#spatial-material-maps)
describes the box and the full list of parameters. On a Rod group a map can
drive **Young's Modulus**, **Bending Stiffness**, **Friction**, **Deformation
Damping**, **Bending Damping**, **Strain Limit** and **Bend Plasticity
Rate**.

On hair the weights come from a point attribute:

1. Give the hair a point attribute that holds the weights. A Float, an
   Integer or a Boolean attribute is accepted, so the points a Boolean pin
   marks can also be the region a map covers.
2. Click **+** in the **Spatial Material Maps** box. On a group that holds
   only hair, the new row starts with **Source** set to **Attribute**.
3. Pick the parameter, type the attribute's name in **Name** and set
   **Target**.

Unlike a pin, a map reads its attribute from the hair as the solver receives
it, so a live Geometry Nodes modifier is enough and does not have to be
applied. Place the modifier above the first modifier that generates curves
(above **Interpolate Hair Curves** in a Fur stack), so that it runs on the
guides. A **Source** of **Vertex Group** on hair stops **Transfer** with a
message that says to switch it to **Attribute**.

Each segment of a strand uses the average weight of its two ends. A map on a
Rod group takes a single source: it cannot change over the solve.

Stiffness and the [hinge at the last held point](#what-an-exact-pin-holds-on-a-strand)
are separate things. A strand with a high **Bending Stiffness** stays
straight, and still turns as a whole about the last point an exact pin holds.

```{figure} ../../images/hair/hair_weights_rest.jpg
:alt: The same two furry spheres at rest after the shake. The fur on the left hangs limp and flat down one side of the sphere. The fur on the right keeps its round shape, with the tips of the lower strands drooping.
:width: 640px

The same two heads once the shaking has stopped. The fur with only its
**Roots** pin hangs from its roots. The fur with the weighted pin and the
stiffness map keeps the shape it was groomed in near the scalp, and its free
tips droop under gravity.
```

## Hair on a Moving Body

Hair points take no keyframes, so the points a pin holds follow a moving head
through **Capture Deformation** (see
[Capture Deformation](../constraints/pins.md#capture-deformation)). The status
line under the capture buttons reads differently for a hair pin:

| Status line                                    | State                                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------- |
| *Nothing moves this pin's hair object*         | **Capture Deformation** is gray.                                              |
| *This hair moves; capture so its pin follows*  | Something moves the hair object, and **Capture Deformation** is available.    |
| *Pin cache: N frame(s)*                        | A capture exists.                                                             |

The capture is offered whenever anything can move the hair object, not only a
deforming modifier on it: Blender parents hair to its surface, so a head that
merely turns carries the roots with it. Capture every pin of the hair that
should travel with the head, a weighted pin included, since the places it
pulls its points toward have to travel too. Capture again when the head's
animation changes.

If a captured hair pin loses its capture file, **Transfer** stops and names
the pin instead of holding its points where they started. Press **Capture
Deformation** again, or **Clear Deformation Cache** to hold the points still.

Two more things come up with Blender's own hair setups:

- **Quick Fur puts a Geometry Nodes modifier on the head.** A **Static**
  object with a modifier that can deform it needs a capture of its own, and
  **Transfer** asks for it: *STATIC object 'Head' ... has a deforming modifier
  stack (or shape-key animation) but no deformation cache.* Capture the head
  as described under
  [Armature-driven Static objects](static_objects.md#armature-driven-static-objects).
- **Blender pins its Hair Dynamics modifier to the end of the stack.** In
  Blender 5.2, Empty Hair and Fur both come with that modifier, and the
  add-on's playback modifier cannot be placed after it. While the body only moves as a whole
  (it is keyframed, or parented to something that moves) that changes
  nothing. When the body changes *shape*, through an armature or shape keys
  for example, that modifier is what carries each root along with the skin,
  and it would do so after the solve. **Transfer** refuses that case and
  names the hair, the body and the modifier. Turn off **Pin to Last** on the
  modifier and move it above the first modifier that generates curves, or
  disable it.

## Playback

After **Fetch All Animation** the hair carries a **ContactSolverCache**
modifier, as a simulated mesh does. On hair it is a Geometry Nodes modifier
that moves the points to the solved positions of the current frame. The groom
itself is never changed: disable the modifier and the hair is back in its
groomed shape, and **Clear Local Animation** removes the modifier and the
solve with it.

The add-on places the modifier after the modifiers that keep the points as
they are and before the first one that changes their number, so in a Fur
stack it sits just above **Interpolate Hair Curves** and the children are
built from the solved guides.

## What Hair Does Not Do

- **Bake.** Blender keeps no keyframes for hair points, so a solve cannot be
  turned into Blender animation on a Hair Curves object. **Bake Animation**
  and **Bake Single Frame** on one hair object report *'Fur' is a Hair Curves
  object and cannot be baked: Blender keeps no keyframes for hair points. Its
  animation stays on the solver cache.* A bake of everything bakes the other
  objects and lists the hair it left on its cache. See
  [Baking Animation](../sim/baking.md).
- **Export.** The Alembic and USD cache exports do not carry hair, as they do
  not carry curve rods, and they name the objects left out. See
  [Exporting](../sim/exporting.md#what-gets-exported).
- **Snap and Merge.** The pickers list meshes and curve rods, not hair.
- **Hold a curl.** A strand keeps the *size* of each bend of its groom, not
  the plane it bends in, so a curl or a spiral does not keep its shape.
- **Start with strands lying on each other.** Two strands that touch at the
  start stop the first step with *contact starts overlapping*. Turn on
  **Allow Self-Intersections** on the group to take strand against strand
  contact out while keeping hair against everything else, since every strand
  of one object is the same object. See
  [Allow Intersections](../params/material.md#allow-intersections).

## Python and MCP

Hair uses the same calls as any other Rod object. `Roots` needs no indices,
and `weights` makes a weighted pin:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

fur = solver.create_group("Fur", "ROD")
fur.add("Fur")            # reports what joining set, as the panel does

# Hair with a Surface already has its Roots pin. On hair without one:
roots = fur.create_pin("Fur", "Roots")

# A pull that fades over points 2 to 5 of one strand. Point indices count
# through every curve in order.
hold = fur.create_pin(
    "Fur", "Hold",
    indices=[2, 3, 4, 5],
    weights=[0.8, 0.6, 0.4, 0.2],
)
hold.pull(strength=300.0)
```

`indices` writes a point attribute of that name and pins it in the same call.
With `weights`, one per index and each above `0` and at most `1`, the
attribute is a Float one. Without `indices`, the name has to be an attribute
the hair already carries, or `Roots`. A pin made this way is added at the end
of the list, so a weighted pin that covers the roots still has to be moved
above **Roots** (see
[Put Roots Below the Weighted Pin](#put-roots-below-the-weighted-pin)).

From MCP, `add_pin_vertex_group(group_uuid, vertex_group_identifier, indices,
weights)` does the same, `list_pins` reports `weighted` for each pin and lists
the pins in the order that decides a shared point, and
`move_pin_vertex_group` reorders them. `add_material_map` with `source_type`
`ATTRIBUTE` adds a stiffness map to the Rod group. See
[Pins and Operations](../constraints/pins.md#mcp-tools) and
[Maps From MCP](../params/material.md#maps-from-mcp).
