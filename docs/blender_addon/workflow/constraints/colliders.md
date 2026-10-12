# 👻 Invisible Colliders

Invisible colliders are walls and spheres that the solver treats as collision
boundaries. Use them for ground planes, containment volumes, bowls, cages and
quick test rigs that you do not want in the viewport or the render.

An invisible collider has no mesh and no modifier, and the scene's render and
viewport visibility settings do not affect it. Colliders are saved with the
`.blend` file.

## Adding a Collider

1. Open the sidebar (<kbd>N</kbd>) in the 3D viewport and switch to the
   add-on tab.
2. In the **Scene Configuration** panel, expand the **Invisible Colliders**
   box with its disclosure triangle.
3. Click **Add** (the button with the plus icon) and choose **Wall** or
   **Sphere**.
4. The new collider appears in the list above the **Add** / **Remove** row,
   named `Wall 1`, `Sphere 1`, and so on, using the lowest unused number for
   its type.
5. Every new collider gets an **Initial** keyframe automatically. It holds
   the collider's state at the solve's **Starting Frame**, reads its values
   from the properties box, and cannot be deleted.

Selecting a collider in the list opens a **properties box** under the **Add**
/ **Remove** row. The fields depend on the type:

- **Wall**: **Name**, **Position**, **Normal**, **Contact Gap**,
  **Friction**, **Thickness** and **Active Duration**.
- **Sphere**: **Name**, **Position**, **Radius**, **Invert** and
  **Hemisphere** side by side, then **Contact Gap**, **Friction**,
  **Thickness** and **Active Duration**.

Turning **Active Duration** on adds an **Active Until (frame)** field.
[Types and Options](#types-and-options) explains every field.

A sphere's row adds `(Inv)`, `(Hemi)` or `(Inv, Hemi)` after its name while
**Invert**, **Hemisphere** or both are on, for example `Sphere 1 (Inv, Hemi)`,
so you can tell the variants apart in the list.

To remove a collider, select it and click **Remove**, to the right of
**Add**. **Remove** is grayed out when nothing is selected.

After you add, remove or edit a collider, click **Update Params on Remote**.
No new **Transfer** is needed (see [Running the Simulation](../sim/simulating.md)).
The checks described on this page run at that point as well as at
**Transfer**.

```{figure} ../../images/invisible_colliders/sub_panel.png
:alt: Scene Configuration panel with the Invisible Colliders box expanded. The list holds Wall 1 (selected) and Sphere 1, each with a closed preview eye, the properties box shows the fields of Wall 1, and the keyframe list below holds one entry, Frame 1 (Initial).
:width: 500px

The **Invisible Colliders** box with two entries (`Wall 1` and `Sphere 1`)
and the properties box open for the selected wall (**Name**, **Position**,
**Normal**, **Contact Gap**, **Friction**, **Thickness**, **Active
Duration**). Below, the keyframe list shows the automatic **Frame 1
(Initial)** keyframe. Both preview eyes are closed in this capture, so
neither collider draws in the viewport.
```

## Preview Overlay

An invisible collider has no mesh, so the **Preview** overlay is the way to
see where it sits and which side it pushes against. Each row in the collider
list has an eye icon on the right. A collider added from the **Add** menu
starts with its preview on (open eye) and draws in the 3D viewport, updating
as you edit **Position**, **Normal** or **Radius** and as the timeline moves
through keyframes. Click the eye to hide the preview and click it again to
show it. A collider created through the Python API or MCP, or loaded from a
scene profile, starts hidden, so click its eye to see it.

Previews are per collider, so you can show one and hide the others. A preview
also follows the **Active Duration** cutoff: once the current frame reaches
the **Active Until (frame)** value, the overlay disappears, as the collider
does in the solve.

### Interpreting the Arrows

The overlay draws normal arrows that point toward the free side, where
dynamic geometry is allowed to be. Read them as "dynamic vertices stay on the
arrow side":

```{figure} ../../images/invisible_colliders/collider_variants.svg
:alt: Five side-by-side schematics. Wall: a flat plane patch with dashed outlines going to infinity and one blue arrow along its Normal; dynamic geometry stays on the arrow side. Sphere (default): a circle with outward-pointing arrows; dynamic geometry stays outside. Sphere with Invert: the same circle with inward-pointing arrows converging at the center; dynamic geometry stays inside. Sphere with Hemisphere: a lower half circle plus a cylinder rising to infinity from the equator, with outward arrows on both parts; dynamic geometry stays outside that shape. Sphere with Invert and Hemisphere: the same shape with inward arrows; dynamic geometry stays inside, so it catches objects that fall in from above.
:width: 880px

The five variants as schematics. In every panel the blue (outward) or purple
(inward) arrow points toward the free side. The shaded fill marks the
solid side (anything that lands there is projected back across the surface),
and dashed outlines mean the surface continues to infinity. The viewport draws
arrows only at the cardinal points (see the sphere description below), while
the schematic draws more along the hemisphere's cylinder and labels that
cylinder a capsule.
```

- **Wall.** One arrow sticks out of the plane along its **Normal**. Points on
  the arrow side are free, and points on the opposite side are projected back
  across the plane. The grid shows a small patch of the plane at **Position**,
  and the faint dashed outlines are a reminder that the plane is infinite.

  ```{figure} ../../images/invisible_colliders/preview_wall.png
  :alt: A wall collider previewed in the viewport, a flat grid at the origin with a single upward normal arrow
  :width: 520px

  A floor wall (**Position** `(0, 0, 0)`, **Normal** `(0, 0, 1)`). The arrow
  shows that dynamic geometry stays on the +Z side.
  ```

- **Sphere.** A full sphere has six arrows at the cardinal surface points
  (±X, ±Y, ±Z), and a hemisphere has five, with none at +Z. With **Invert**
  off the arrows point outward. With **Invert** on they point inward, toward
  the center, and are mostly hidden inside the wireframe. With **Hemisphere**
  on, the wireframe is the lower half of the sphere plus a cylinder that
  rises from the equator.

  ```{figure} ../../images/invisible_colliders/preview_spheres.png
  :alt: Three sphere colliders side by side in the viewport: a green sphere with outward arrows, a pink inverted sphere, and a blue hemisphere drawn as a lower half with a cylinder rising from it
  :width: 720px

  Left to right: a default sphere, an inverted sphere, and a hemisphere
  (lower half plus a cylinder that rises from the equator).
  ```

Visible previews are colored in turn: the first wall is blue, the first
sphere is green, and later ones rotate through the hue wheel, so several
overlays stay readable side by side. The color of a preview can change when
another preview is hidden.

## Keyframe Animation

Colliders animate through their own keyframe list below the properties box,
not through Blender's Graph Editor or Action. Each keyframe stores a frame
number plus the values that change at that frame (**Position** for both
types, **Radius** for spheres). Between two keyframes the values change
linearly.

To animate the selected collider:

1. Select the collider and scrub the timeline to the frame you want.
2. Click **Add Keyframe**. A keyframe is added at the current frame, in frame
   order, with the collider's base **Position** (and **Radius** for a sphere).
   A playhead before frame 1 counts as frame 1. A frame that already has a
   keyframe is refused with a warning. The **Initial** keyframe occupies frame
   1 whatever the **Starting Frame** is, so **Add Keyframe** at frame 1 is
   always refused. A frame at or before the **Starting Frame** is accepted
   here and refused later, at **Transfer** or **Update Params on Remote**.
3. The new keyframe is selected, and a details box shows its **Frame**,
   **Hold** and, while **Hold** is off, the value rows (**Position**, plus
   **Radius** for a sphere).
4. Adjust the values in place. To delete a keyframe, select it and click
   **Remove**.

The first keyframe in the list is marked **Initial**. It holds the collider's
state at the solve's **Starting Frame**, so the list labels it with that frame
(**Frame 1 (Initial)** at the default **Starting Frame** of 1) and the
viewport preview places it there. It shows *"Uses base properties above"* instead of
value rows, reads whatever is in the properties box, and cannot be removed;
its **Remove** button stays disabled.

Every later keyframe must come after the **Starting Frame** and after the
keyframe listed before it. **Transfer** and **Update Params on Remote** refuse
a keyframe at or before the **Starting Frame**, or one out of order with its
predecessor, naming the collider and the frame.

Turning **Hold** on for a later keyframe makes that frame keep the previous
keyframe's values, which produces a step. For "stay put until frame 60, then
move", add a **Hold** keyframe at frame 60 and a value keyframe at frame 61.

### Example: Shrink a Sphere Starting at Frame 60

With the default **Starting Frame** of 1, a sphere keeps a radius of 1.0 until
frame 60, then shrinks to 0.5 over the next 60 frames (reaching 0.5 at frame
120):

1. Add a **Sphere** collider and set **Position** to `(0, 0, 0)` and
   **Radius** to `1.0` in the properties box.
2. Scrub to frame 60, click **Add Keyframe**, and turn **Hold** on. The
   keyframe keeps the previous radius (1.0), so it needs no value rows.
3. Scrub to frame 120, click **Add Keyframe**, leave **Hold** off, and set
   **Radius** to `0.5`.

```{figure} ../../images/invisible_colliders/sphere_shrink_example.svg
:alt: Line graph of sphere radius versus frame number for the three-keyframe example. The x axis is Frame (ticks at 1, 30, 60, 90, 120) and the y axis is Radius (ticks at 0.5, 0.75, 1.0). Three keyframe markers are plotted: Frame 1 Initial at Radius 1.0, Frame 60 Hold at Radius 1.0, Frame 120 at Radius 0.5. The actual animation curve is a solid blue line that holds flat at 1.0 from frame 1 through frame 60 (the Hold plateau) and then ramps linearly down to 0.5 at frame 120. A dashed gray line overlay shows the comparison without the Hold: a single straight line from 1.0 at frame 1 down to 0.5 at frame 120, so the sphere would start shrinking at frame 1 instead of holding its initial size. Annotations call out the plateau, the linear ramp, and the comparison line.
:width: 760px

The three keyframes, plotted. The solid blue line is the radius: flat until
frame 60 (the **Hold** keyframe keeps the previous value of `1.0`), then a
linear ramp down to `0.5` at frame 120. The dashed gray line is what the
sphere would do without the Hold: a straight line from `1.0` to `0.5` over
all 119 frames, which starts the shrink at frame 1.
```

```{figure} ../../images/invisible_colliders/sphere_shrink_3d.svg
:alt: Five 3D sphere snapshots on a shared ground line, left to right. At Frame 1, Frame 30, and Frame 60 the sphere is the same size (radius 1.0); a yellow-bordered "Hold plateau" band highlights these three together. At Frame 90 the sphere is smaller (radius 0.75). At Frame 120 the sphere is smallest (radius 0.5). A faint dashed outline at the Frame 120 position shows what the original radius 1.0 would have looked like, for scale. Each sphere casts an elliptical shadow and carries a frame number and radius value underneath.
:width: 820px

The sphere at each of those frames. The first three snapshots are the same
size, and the last two show the linear ramp between **Frame 60** and **Frame
120**. The faint dashed ring at the Frame 120 position marks the original
radius of `1.0`, so you can compare the sizes.
```

```{figure} ../../images/invisible_colliders/keyframe_animation.png
:alt: Invisible Colliders box with a sphere named Shrinking Ball that has three keyframes (Frame 1 (Initial), Frame 60, Frame 120). Frame 120 is selected and its details box shows Frame 120, Hold off, Position (0, 0, 0) and Radius 0.500.
:width: 500px

The resulting box for a sphere named `Shrinking Ball`. The keyframes are
**Frame 1 (Initial)**, **Frame 60** (Hold) and **Frame 120** (**Radius**
0.500, selected). The radius stays at 1.0 through frame 60 and then changes
linearly to 0.5 across frames 60 to 120.
```

## Types and Options

| Type       | Shape                                                                 | Extra options              |
| ---------- | --------------------------------------------------------------------- | -------------------------- |
| **Wall**   | An infinite plane through **Position**. Dynamic geometry stays on the **Normal** side. | None |
| **Sphere** | A sphere at **Position** with **Radius**.                             | **Invert**, **Hemisphere** |

A sphere collides from the outside by default: objects stay outside the ball.
**Invert** and **Hemisphere** change the shape. The combinations are:

| **Invert** | **Hemisphere** | Objects stay | Shape                                                                            |
| ---------- | -------------- | ------------ | -------------------------------------------------------------------------------- |
| off        | off            | Outside      | A sphere.                                                                        |
| on         | off            | Inside       | A containment volume: a hollow sphere that keeps objects in.                     |
| off        | on             | Outside      | The lower half of a sphere with a cylinder of the same radius rising from its equator without end. |
| on         | on             | Inside       | The same shape with objects kept inside: a bowl that catches objects falling in from above. |

The table describes a sphere with no keyframe after **Initial**. A
**Hemisphere** sphere that has a later keyframe, a **Hold** keyframe included,
keeps objects inside its shape whether **Invert** is on or off.

The fields of the properties box:

| UI label                   | Type   | Default                          | Meaning                                                                 |
| -------------------------- | ------ | -------------------------------- | ----------------------------------------------------------------------- |
| **Name**                   | Both   | `Wall 1`, `Sphere 1`, ...        | The label in the list.                                                  |
| **Position**               | Both   | `(0, 0, 0)`                      | A point on the wall's plane, or the sphere's center, in world space.    |
| **Normal**                 | Wall   | `(0, 0, 1)`                      | The direction the wall faces. Dynamic geometry stays on this side.      |
| **Radius**                 | Sphere | 1.0 (minimum 0.001)              | The sphere radius.                                                      |
| **Contact Gap**            | Both   | 0.001 (minimum 0)                | The barrier gap kept between the collider and dynamic geometry.         |
| **Friction**               | Both   | 0.0 (0 to 1)                     | The tangential friction coefficient.                                    |
| **Thickness**              | Both   | 1 m (minimum 0.00001)            | The maximum penetration depth. Geometry that sinks deeper than this inside the collider passes through it: contact and continuous collision detection are both ignored. Lower it to make the collider a thin shell. |
| **Active Duration**        | Both   | off                              | When on, the collider stops acting at **Active Until (frame)**.         |
| **Active Until (frame)**   | Both   | 60 (minimum 1)                   | The first Blender frame at which the collider stops acting. It acts on frames before this number, so the value is a frame, not a frame count. |

An **Active Until (frame)** value at or before the solve's **Starting Frame**
would never act, and **Transfer** and **Update Params on Remote** refuse it, naming the
collider.

**Contact Gap** and **Thickness** are distances in scene units. Like the
collider's position and radius, they scale with
[World Scaling](../params/scene.md#world-scaling), so a gap that holds cloth
just off a wall at a **World Scaling** of `1.0` holds it at the same place
relative to the geometry at any other setting.

## Start Positions

Every unpinned vertex of a simulated object must start on the free side of
every collider: on the **Normal** side of a wall, outside a sphere, or inside
an inverted sphere. For a collider that has no keyframe after **Initial**, the
scene is refused when it is built, before the first frame is solved, if a
vertex starts on the wrong side. The message reads *"N vertices violate wall
constraints. First: vertex V, X units wrong side of wall W."* or, for a
sphere, *"N vertices violate sphere constraints. First: vertex V, sphere S."*
V is a zero-based index counted over all simulated vertices of the scene, not
a Blender vertex index. W and S are zero-based positions in the list of
colliders that are checked this way, in list order, with walls and spheres
counted separately. Move the collider or the object so that the object starts
on the free side. Pinned vertices are exempt.

A collider that has a keyframe after **Initial** is not checked at this point.
Place the simulated objects on its free side at the **Starting Frame**
yourself.

## Saving with a Scene Profile

Colliders are saved with the `.blend` file. To reuse them in another scene,
save the scene profile with the **Save Scene Profile** icon on the profile row
of **Scene Configuration**. Opening a profile that holds colliders replaces the
scene's colliders, and each one starts with its preview eye closed, so click
the eye to draw it. See
[Scene Parameters](../params/scene.md#scene-profiles).

A profile stores each collider's type, name, position, **Normal** or
**Radius**, **Invert**, **Hemisphere**, **Contact Gap**, **Friction** and
keyframes. It does not store **Thickness**, **Active Duration** or **Active
Until (frame)**; they return to their defaults (1 m, off, 60) when a profile
is opened, so set them again afterwards. Colliders are not stored per material
profile or per group.

## Blender Python API

The same workflow is available from Python. The import in the first line
assumes the add-on is in the **User Default** repository. For an add-on
installed from a remote repository, see
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# A floor (wall facing up) with 0.5 friction.
floor = solver.add_wall([0, 0, 0], [0, 0, 1])
floor.param.friction = 0.5

# A bowl: an inverted hemisphere keeps objects inside it.
bowl = solver.add_sphere([0, 0, 0], 0.98).invert().hemisphere()
bowl.param.friction = 0.3

# A wall that stops acting at frame 90.
wall = solver.add_wall([0, 0, 0], [0, 0, 1])
wall.param.enable_active_duration = True
wall.param.active_duration = 90

# Animated sphere: hold radius=1.0 until frame 60, then shrink to 0.5 at frame 61.
solver.add_sphere([0, 0, 0], 1.0).time(60).hold().time(61).radius(0.5)

# Animated wall: slide 1 m along +Y between frames 30 and 90.
(solver.add_wall([0, 0, 0], [0, 1, 0])
       .time(30).hold()
       .time(90).move_to([0, 1, 0]))

# Start over.
solver.clear_invisible_colliders()
```

Both builders are chainable. `time(frame)` advances the keyframe cursor, which
starts at frame 1 (frames are Blender frames and must strictly increase), and
`hold()` adds a hold keyframe. The value-keyframe methods differ by collider
type:

- **Wall** has `move_to(position)` (an absolute world-space position) and
  `move_by(delta)` (an offset from the position of the previous keyframe). A
  `hold()` keyframe stores no position of its own, so use `move_to` right
  after a `hold()`. Walls have no radius, and `transform_to` is for spheres
  only.
- **Sphere** has `move_to(position)`, `radius(r)` (keyframe a new radius) and
  `transform_to(position, radius)` (keyframe both together). A keyframe made
  with `move_to` keeps the sphere's base radius, and one made with `radius`
  keeps its base position, so use `transform_to` to set both. Spheres have no
  `move_by`.

The `param` proxy of a collider has five names: `contact_gap`, `friction`,
`thickness`, `enable_active_duration` and `active_duration`. They set the
collider's own values without keyframing them. `active_duration` is the first
Blender frame at which the collider stops acting.

The same operations are available as MCP tools: `add_invisible_wall`,
`add_invisible_sphere`, `list_invisible_colliders`,
`remove_invisible_collider`, `clear_invisible_colliders`,
`set_collider_properties`, `add_collider_keyframe`,
`remove_collider_keyframe` and `list_collider_keyframes`. The tools that
change or remove an existing collider address it by its zero-based position
in the list that `list_invisible_colliders` reports. See the
[MCP Tool Reference](../../integrations/mcp_reference.rst) for the signatures.
