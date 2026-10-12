# 🌀 Force Fields

Blender's force field objects can act on the objects the solver simulates,
the way they act on particles and cloth in Blender's own simulators:
**Force**, **Wind**, **Vortex**, **Turbulence**, **Magnetic**, **Harmonic**,
**Drag**, image **Texture** fields and **Fluid Flow**. The solver evaluates each
field exactly at every vertex, so nothing is sampled on a grid and there is no
resolution to set. For motion no field object can describe, a short Python
function gives the push exactly, everywhere and at every moment.

Everything here is in the **Force Fields** box of the **Scene
Configuration** panel, plus one slider per object group.

```{figure} ../../images/force_fields/panel.png
:alt: Scene Configuration panel with the Force Fields box expanded. An empty Collection field is at the top, followed by two field rows, Turbulence and Wind, each with a check mark and an Apply to All Groups checkbox. A Script box shows a text named force_field.py, a Built-in Functions button, Apply to All Groups, a grayed-out Compile and Check button and the line Connect to a server to check the script. A Visualize box below has Visualize on and a Preview Resolution of 8, 8, 8
:width: 500px

The **Force Fields** box with two field objects, a script, and **Visualize**
on. **Compile and Check** is grayed out until the add-on is connected to a
running server.
```

## Adding a Field

1. Add a field the usual way: **Add > Force Field** in the 3D viewport, or
   **Physics > Force Field** on an existing object. Move, rotate and
   keyframe it like any other object.
2. Choose its type and set its settings in Blender's **Physics** tab.
3. Open the **Force Fields** box in **Scene Configuration**.

The box lists every field it will send, one row per field object, written
`<object name>: <field type>`. A row has a check mark, or a warning triangle
and the reason when the field is not supported. A field with a warning
triangle stops **Transfer**, so fix it or remove it. With no field in the
scene the box reads **No force field objects**.

**Collection** at the top of the box limits the list to one collection and
its children. Leave it empty to take every field in the scene.

## What Each Field Does

| Field          | What it does to the simulated objects |
| -------------- | ------------------------------------- |
| **Force**      | Pushes away from the field's center (from its plane for a Plane shape), or pulls with a negative **Strength**. **Strength** is an acceleration in m/s². |
| **Wind**       | Blows air along the field's Z axis at **Strength** m/s. It acts through air drag, so a surface facing the flow receives more of it than a surface edge-on, and it needs an **Air Density** above 0. |
| **Vortex**     | Swirls around the field's Z axis. **Inflow** damps the velocity. |
| **Turbulence** | Pushes in a smooth random pattern whose swirls are **Size** across. **Size** must be above 0. **Strength** is in m/s². |
| **Magnetic**   | Pushes at right angles to the motion of the surface, so nothing acts on a surface at rest. |
| **Harmonic**   | A spring toward the field's center (toward its plane for a Plane shape) with a **Rest Length**, critically damped by **Harmonic Damping**. A negative **Strength** pushes away. |
| **Drag**       | Slows the surface with **Linear Drag** and **Quadratic Drag** against its velocity. **Strength** is not read. |
| **Texture**    | An image texture pushes the surface. See [Image Texture Fields](#image-texture-fields). |
| **Fluid Flow** | Carries the surface along with a fluid. See [Fluid Flow](#fluid-flow). |

**Strength** is per unit mass, so the same push moves a light and a heavy
fabric alike. Blender's own **Strength** is a force, so the two agree when the mass is 1.
Lengths are in scene units and time is in seconds. A **Wind** field acts
through air drag and only when **Air Density** is above zero.

**Turbulence looks different from Blender's own.** Blender's noise cannot be
reused here, so the solver's Turbulence, and the **Noise Amount** of any
field, use their own random pattern. **Size**, **Strength**, **Noise Amount**
and **Seed** mean the same thing, and the pattern is just as random, but it is
not the pattern Blender's particles would show.

The solver reads these settings of a field:

- In **Settings**: **Shape** set to Point or Plane, **Strength**, **Noise Amount**,
  **Seed**, **Global**, and for Turbulence **Size**. **Flow** is read on a
  Vortex (its **Inflow**) and on a Fluid Flow; a Wind already is an air flow,
  so its **Flow** adds nothing.
- In **Falloff**: **Shape** set to Sphere, Tube or Cone, **Z Direction**
  (which limits the field to one side), **Power**, **Min Distance** and **Max
  Distance**, and for a Tube or a Cone the **Radial** settings (a Cone's are
  angles in degrees). **Gravitation** multiplies the push by 1/d², without a
  bound, so a vertex exactly at the field's center (or on a Plane field's
  plane) stops the run with an error naming the field.
- The type's own settings in the table above.

:::{note}
**Transfer** stops with a message naming the object when a field uses
something the solver does not support, rather than ignoring it:

- A **Charge**, **Boid**, **Lennard-Jones** or **Curve Guide** field. These
  have no effect on cloth in Blender either.
- A shape other than Point or Plane, or a falloff other than Sphere, Tube or
  Cone.
- **Flow** above 0 on any type except Wind, Vortex (its **Inflow**) and Fluid
  Flow.
- **Multiple Springs** (Harmonic), **2D**, or **Absorption**.
- **Affect Location** turned off, since the field would move nothing.
- **Apply Density** on any type except Fluid Flow.
- A **Turbulence** with a **Size** of 0.
- A zero or mirrored (negative) scale.
- A Wind field in a scene whose **Air Density** is 0.
:::

A field that something can move or animate (its own keyframes, a driver, a
parent or a constraint) is sampled at every solve frame, so moving a field,
or keyframing its **Strength**, works as it does in Blender. **Transfer** then
steps through the timeline once. A field that nothing can animate is read
once, and its other settings must not change during the solve.

### Image Texture Fields

A **Texture** field uses an image texture. In RGB mode it pushes by 0.5
minus the pixel's color, scaled by **Strength** (m/s²); Gradient and Curl take
differences **Nabla** apart. The image spans x and y from -1 to
1 in scene units, in world coordinates, or in the field's own coordinates with
**Use Coordinates**. The extensions Extend, Repeat, Clip and Clip Cube work.
Pixels are converted to linear values by the image's color space: sRGB,
Non-Color and Linear Rec.709 are supported.

**Transfer** refuses, with the reason, a texture field that has no texture or
no image, a procedural or node-driven texture, the Checker extension, **Root
Texture Coordinates**, an image sequence, movie, tiled (UDIM) or Render Result
image, any other color space, and a texture setting the solver does not read
when it is not at its default: **Flip Axis**, **Crop**, **Repeat** and
**Mirror** under the Repeat extension, **Brightness**, **Contrast**,
**Saturation**, the color factors, a color ramp, **Filter Size** and
**Calculate Alpha**.

### Fluid Flow

A **Fluid Flow** field carries the simulated surface with a fluid. It pushes
each vertex toward the fluid's velocity at that point, at a rate of
**Strength** per second, and the solver handles it implicitly. With **Flow**
at 1 the surface relaxes toward the fluid's velocity, and **Apply Density**
weighs the push by the smoke density. Outside the fluid's box nothing acts, damping included.
The scene's **World Scaling** and **Time Scale** apply, as they do to every
length and rate.

Set the field's **Domain Object** to one of two sources:

- **A Mantaflow gas domain**, baked with Cache Type **All** or **Modular** over
  every frame the solve reads. The surface then moves with the smoke as it was
  simulated, and the domain's own **Time Scale** applies. **Transfer** refuses
  a Replay cache, a domain that is not baked or is not baked over the solve
  frames, a liquid domain, **Adaptive Domain**, a modifier placed after the
  domain's Fluid modifier, a domain that the view layer does not evaluate
  (disabled in viewports, in an excluded collection, or with the Fluid
  modifier's **Show in Viewport** off), and **Apply Density** on a domain with
  **Noise**.
- **A Volume object** whose Velocity settings name a vector grid, or three
  component grids. The velocity is read in the volume's **Velocity Unit** times
  **Velocity Scale**, and a grid named `density` serves **Apply Density**.
  **Transfer** refuses a modifier on the Volume, a transform that shears,
  mirrors or flattens the voxels, and a file inside a fluid domain's cache
  directory: point the field at the domain itself.

**Noise Amount**, **Wind Factor** and **Absorption** on a Fluid Flow are
refused, so set them to 0 and off.

The field's row in the box shows the source, the current frame, the grid
resolution and an upper bound on the memory the fluid adds to what **Transfer**
sends, and it shows the reason when the field is refused. **Transfer** reads the
source at every solve frame, so a problem first reached at a later frame, or in
a voxel, appears there.

## Script

For a field no object describes, write it as Python.

1. In the **Script** box, choose a text from the dropdown, or click **+**
   to create a text named `force_field.py` that holds a working template.
2. Edit the text in Blender's Text Editor.
3. Click **Compile and Check** (see below).

A script has this shape:

```python
import math


def eval(x, y, z, t):
    r = math.hypot(x, y) + 1e-6
    swirl = 2.0 * math.sin(2.0 * t)
    return (-y / r * swirl, x / r * swirl, 0.0)
```

`x, y, z` is a point in Blender's world space, in scene units, and `t`
is seconds since the simulation started. The function returns the push
at that point as `(ax, ay, az)` in m/s² along Blender's axes. The solver
evaluates it exactly at every simulated vertex, at every step, on top of
any field objects.

The script is plain Python, limited to what the solver can run:

- `import math` at the top, the only import allowed, gives `math.sqrt`,
  `math.sin`, `math.cos`, `math.tan`, `math.asin`, `math.acos`,
  `math.atan`, `math.atan2`, `math.sinh`, `math.cosh`, `math.tanh`,
  `math.exp`, `math.log`, `math.pow`, `math.hypot`, `math.fabs`,
  `math.floor` and `math.ceil`, and the constants `math.pi`, `math.e` and
  `math.tau`. Angles are in radians.
- Arithmetic (`+ - * / // % **`), comparisons, `and`, `or`, `not`, `if` /
  `elif` / `else`, `x if c else y`, and local variables.
- `for i in range(n)` loops, with `n` an integer written in the script.
  Loops are unrolled (at most 4096 iterations) and cannot contain
  `return`.
- `abs`, `min`, `max` and `float`.
- The built-in noise below.

Every path must end with `return (ax, ay, az)`. Click **Built-in
Functions** under the script for the complete list with what each one
takes. **Compile and Check** refuses a call outside the list and names the
functions that are available.

### Built-in Noise

Two noise functions are built in for turbulent motion:

- `noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)`:
  a smooth random value between about -1 and 1.
- `curl_noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)`:
  a smooth random vector field with no sources or sinks: it does not
  gather points in one place or push them apart. It gives three numbers:
  `return curl_noise(...)` directly, or `cx, cy, cz = curl_noise(...)`.

Both take the same settings:

| Setting     | What it does                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------- |
| `octaves`   | Layers of finer detail, a whole number from 1 to 8 written in the script.                     |
| `seed`      | Picks a different random pattern.                                                             |
| `time`      | Pass `t` to let the pattern change over time. Without it the pattern stays still.             |
| `frequency` | How many times per second the pattern changes, in place rather than sliding along.            |
| `decay`     | How fast it fades: the result is multiplied by `exp(-decay * time)`. 0 never fades.           |

Multiply the result to set the strength, and scale the position to
change the size of the swirls:

```python
def eval(x, y, z, t):
    # Swirls about half a meter across, changing about once a second and
    # fading to a third of their strength after two seconds.
    cx, cy, cz = curl_noise(2.0 * x, 2.0 * y, 2.0 * z, octaves=3, seed=7,
                            time=t, frequency=1.0, decay=0.55)
    return (5.0 * cx, 5.0 * cy, 5.0 * cz)
```

The **Turbulence** field uses the same gradient noise as `noise`, as three
independent copies with one octave and no time input. A Turbulence field
therefore does not change over time on its own: animate its **Strength** or
location to change it. **Visualize** draws the script with a copy of the
same algorithm, so the arrows match the solve.

### Compile and Check

**Compile and Check** has the connected solver compile the script without
transferring anything. It is available when all of these hold:

- A script is chosen in the **Script** box.
- The add-on is connected to a server.
- The server is running.

Otherwise the button is grayed out, and the line under it says why:
**Choose or create a script first**, **Connect to a server to check the
script** or **Start the server to check the script**.

The status line then reads **Checking...** while the solver compiles,
**Not checked since the last edit** before any answer, **OK** followed by a
summary when the script compiles, or the problem and its line, for example
`Line 3: 'Import' statements are not supported; see the force field script subset`.
A top-level import other than `import math` reads `only 'import math' is allowed`.
The answer stays shown until the text is edited.

## Choosing Which Groups a Field Pushes

Every field object, and the script, pushes every simulated group by
default. To limit one:

1. Click its **Apply to All Groups** checkbox off.
2. Click **+** and pick a group. Repeat for each group it should push.
3. To drop a group, select its row in the list and click **-**.

The list follows your groups: renaming a group renames it here, and
deleting a group removes it from every list. Static groups cannot be
added, since colliders follow their own animation.

The panel and **Transfer** report these cases:

- **Apply to All Groups** is off and the list is empty. The box shows
  **Choose at least one group** with a warning triangle, and **Transfer**
  stops with a message naming the field.
- A listed group has become Static. Its row shows a warning triangle and
  the reason, for example `group 'Floor' is Static, and colliders ignore
  force fields`, and **Transfer** refuses the field until you remove the
  row.
- The groups a field reaches hold no object. The row shows *Pushes no
  object: assign objects to the groups it reaches* with a warning triangle,
  and **Transfer** stops with `Force field 'Wind' pushes no object: assign
  objects to the groups it reaches`.

The script has its own **Apply to All Groups** checkbox and list in the
**Script** box, drawn once a script is chosen.

## Visualize

1. Tick **Visualize** to draw the fields as arrows in the viewport. Arrows
   appear once at least one simulated group holds objects.
2. Set **Preview Resolution**, the number of arrows along X, Y and Z. The
   default is 8 8 8 and each value runs from 2 to 48. The line under it
   reads, for example, `512 arrows at the current frame, drawing only`.

The arrows are orange for fields that act as an acceleration, blue for Wind
fields (an air velocity) and purple for the script. **Preview Resolution**
changes only the drawing: the solver evaluates each field exactly at every
vertex. A field the drawing cannot read is left out and named under the
settings as *Not drawn:* followed by the reason; the other fields still draw.
The arrows show the fields at the timeline's current frame, so scrub the
timeline to watch animated fields change.

Each color is scaled to its own strongest arrow, so arrow lengths show the
shape of that field, not its absolute strength, and lengths are not
comparable between colors. **Hide all directional arrows** in the
**Visualization** panel hides these arrows, along with the gravity, wind
and velocity arrows. See
[Visualization](../../getting_started/tour.md#visualization).

## Per-Group Weight

Each object group has a **Force Field Weight** in its **Material Params**
box, below **Allow Intersections**. **1** applies the fields as authored,
**0** leaves the group untouched, values in between scale them, and values
above 1 amplify them (the slider stops at 2). It cannot be keyframed.

Vertices held by a fixed pin follow the pin and ignore the fields.
Vertices held by a **Pull** pin ignore the pushes of Force, Vortex and
Turbulence fields and of the script, as they ignore gravity, because the
pull replaces their inertia. A Wind field still acts on them through air
drag. **Static** groups have no weight: colliders follow their own
animation.

## Python / MCP API

The Python API sets the same settings, targets and script, lists the fields
with their status, and reads the list of built-in functions. See the
[Force Fields section of the Python API page](../../integrations/python_api.md#force-fields).
The [MCP](../../integrations/mcp.md) tools are `get_force_field_settings`,
`set_force_field_settings`, `set_force_field_targets` and
`check_force_field_script`. `get_force_field_settings` returns the settings,
every field object with its status (`type`, `refused`, `groups`, `pushes`,
`tracked` and, for a Fluid Flow, `fluid`) and the last script check.
`set_force_field_settings` takes these arguments:

| Argument             | Panel                                                                               |
| -------------------- | ----------------------------------------------------------------------------------- |
| `collection`         | **Collection**                                                                      |
| `script_text`        | The name of the Text chosen in the **Script** box                                   |
| `script_source`      | The source of that Text, or of a new Text named `force_field.py` when none is chosen |
| `visualize`          | **Visualize**                                                                       |
| `preview_resolution` | **Preview Resolution**                                                              |

An empty `collection` or `script_text` clears it. There is no sampling box,
spacing, time sampling or size limit to set.

Over MCP, `check_force_field_script` returns at once, where the Python call
waits for the answer. Poll `get_force_field_settings` until `check_running`
is false, then read `check`: `ok`, and either a `summary` or an `error`
with its `line`. `check` is null when the script text changed since the
last check, when the script was never checked, or when no script is
chosen. `get_force_field_settings` also lists every function a script
may call, under `script_builtins`. See
[Scene Setup via MCP](../../integrations/mcp_scene_setup.md).

:::{admonition} Under the hood
:class: toggle

A field object reaches the solver as its settings plus a track of per-frame
samples of its position, orientation and every animatable value, taken at each
solve frame when anything can move or animate it and once otherwise.
Nothing is sent per vertex, and the solver evaluates every field exactly at
each vertex. The fields enter as an acceleration added to the inertial target
of every vertex that no pin holds, beside gravity, once per solver step, at the
position the vertex starts that step from. Wind instead becomes an air velocity
inside the air drag, which also acts on vertices held by a **Pull** pin. During
the **Inactive Momentum Frames** the fields do not act. **Visualize**
evaluates a script's noise in the add-on with a copy of the algorithm the
solver runs, and the two agree to float32 rounding. A Fluid Flow's fluid is
the one volume sent: it travels as sparse blocks of velocity and density per
frame, and only blocks that hold a nonzero value are kept. World Scaling does
not change how hard a field pushes: fields are authored in scene units and
their strengths are physical, like gravity's.
:::
