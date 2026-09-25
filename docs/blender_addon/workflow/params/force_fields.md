# 🌀 Force Fields

Blender's **Force**, **Wind**, **Vortex** and **Turbulence** field
objects push, swirl and blow the objects the solver simulates, the way
they move particles and cloth in Blender's own simulators. For motion no
field object can describe, a short Python function can give the push
exactly, everywhere and at every moment.

Everything here lives in the **Force Fields** box of the **Scene
Configuration** panel, plus one slider per object group.

```{figure} ../../images/force_fields/panel.png
:alt: Scene Configuration panel with the Force Fields box expanded, directly below Wind. It lists two field objects: Turbulence, with Apply to All Groups checked, and Wind, with Apply to All Groups unchecked and a list holding the group Cloth beside + and - buttons. Below them, a Sampling box with Padding 0.5 m, Spacing 0.1 m, Time Samples 8, the line "[Info] Force field 31x31x11x8, 31x31x11x8: 2.0 MB estimated" and Size Limit 2000 MB; a Script box holding the text force_field.py, a Built-in Functions button, Apply to All Groups checked, a grayed Compile and Check button highlighted in red and the note "Connect to a server to check the script"; and a Visualize box with Preview Resolution 8 8 4 and "256 arrows at the current frame, drawing only"
:width: 500px

The **Force Fields** box with two field objects (the Wind field limited
to the Cloth group), the sampling settings and their memory estimate, a
script, and **Visualize** on.
**Compile and Check** (red) waits for a connection to a running server.
```

## Adding a Field

Add a field the usual way: **Add > Force Field** in the 3D viewport, or
**Physics > Force Field** on an existing object. Move, rotate and
keyframe it like any other object. Its settings (**Strength**,
**Falloff**, **Min / Max Distance**, **Size** and **Seed** for
Turbulence) are read from Blender's own **Physics** tab.

The **Force Fields** box lists every field it will send, each with a
check mark, or with a warning icon and the reason when a setting is not
supported. A field outside the list does not reach the solver.

| Field          | What it does to the simulated objects                                                |
| -------------- | ------------------------------------------------------------------------------------ |
| **Force**      | Pushes away from the field's center (or from its plane), or pulls with a negative Strength. |
| **Wind**       | Blows air along the field's Z axis. Surfaces facing the flow catch it, edge-on ones do not. |
| **Vortex**     | Swirls around the field's Z axis.                                                    |
| **Turbulence** | Pushes in a random, swirling pattern set by **Size** and **Seed**.                   |

**Strength** is an acceleration in m/s² for Force, Vortex and
Turbulence, the same unit as **Gravity**, so a Strength of 9.8 pushes as
hard as gravity pulls, however heavy the fabric. For Wind it is the air
speed in m/s, like the scene **Wind**, and it only acts when **Air
Density** is above zero.

Fields work with **Shape** set to **Point** or **Plane**, and **Falloff**
set to **Sphere** or **Tube**, with the minimum and maximum distances and
the falloff power. **Z Direction** keeps one side of the field only.

:::{note}
**Transfer** stops with a message naming the object when a field uses a
setting the solver cannot reproduce: another field type (Harmonic,
Magnet, Drag, Charge, Texture and so on), another shape, the **Cone**
falloff, **Flow** on anything but Wind, a nonzero **Noise Amount**, or a
Wind field in a scene whose **Air Density** is 0. A field that would be
ignored is always reported rather than skipped.
:::

:::{note}
**Turbulence looks different from Blender's own.** Blender's noise
cannot be reused here, so the solver's Turbulence is its own random
pattern. **Size**, **Strength** and **Seed** mean the same thing, and
the swirls are just as random, but they are not the swirls Blender's
particles would show.
:::

## Where and How Finely Fields Are Sampled

Field objects are measured on a grid of points before the solve and
interpolated in between, so the settings below trade accuracy against
memory.

- **Padding**: each field is measured over the box around the objects it
  pushes (every simulated object, or the groups it is limited to, see
  below) at the starting frame, grown by **Padding** on every side. Make
  it large enough to cover where the objects will move: outside the box,
  the sampled fields do nothing. **Transfer** stops with a message if a
  box would be thinner than one **Spacing** along any axis, as a flat
  sheet with **Padding** 0 would be, since the sheet would leave it as
  soon as it moved.
- **Spacing**: the distance between neighboring points, the same along
  X, Y and Z. The number of points along each side follows from the
  size of the box; a smaller spacing follows the field more closely and
  takes more memory.
- **Time Samples**: how many moments over the simulation the fields are
  measured at, spread evenly from the first frame to the last. The solver
  blends between them, so an animated field (a moving empty, a keyframed
  Strength) needs enough samples to follow its motion. **1** measures the
  starting frame only, which is all a field that never changes needs.

Under the settings the box shows the points along X, Y and Z, the time
samples, and the memory the sampled fields take, for example
`[Info] Force field 32x24x16x8: 2.4 MB estimated`, one size per grid.
Fields limited to different groups are measured on grids of their own.
**Transfer** refuses grids larger than **Size Limit (MB)** in total.

## Script

For a field no object describes, write it as Python. Choose a text in
**Script**, or click **+** to create one from a template:

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
arithmetic, comparisons, `if` / `elif` / `else`, local variables,
`for i in range(10)` with a fixed count, `abs`, `min`, `max`, `float`,
the `math` functions and constants, and the built-in noise below. Every
path must end with `return (ax, ay, az)`. Click **Built-in Functions**
under the script for the complete list with what each one takes; a call
outside it is refused by **Compile and Check**, which names the
functions that are available.

### Built-in Noise

Two noise functions are built in for turbulent motion:

- `noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)`:
  a smooth random value between about -1 and 1.
- `curl_noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)`:
  a smooth random swirl. It stirs without gathering things in one place
  or pushing them apart, which reads as natural gusts and eddies. It
  gives three numbers: `return curl_noise(...)` directly, or
  `cx, cy, cz = curl_noise(...)`.

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

This is the same noise the **Turbulence** field uses (still, since a
field object has no time settings of its own; animate its Strength or
location to change it over time), computed by the solver itself, so the
solve and the **Visualize** arrows agree.

### Compile and Check

Click **Compile and Check** to have the connected solver compile the
script without transferring anything. The box then shows **OK** or the
problem and its line, for example `Line 3: 'Import' statements are not
supported`. The button is available once the add-on is connected to a
running server, and the answer stays shown until the text is edited.

## Choosing Which Groups a Field Pushes

Every field object, and the script, pushes every simulated group by
default. To limit one, click its **Apply to All Groups** checkbox off,
then add the groups it should push with **+** and remove them with
**-**. The list follows your groups: renaming a group renames it here,
and deleting a group removes it from every list. Static groups cannot be
added, since colliders follow their own animation. **Transfer** stops
with a message if a field is limited to no groups at all.

## Visualize

Check **Visualize** to draw the field as arrows in the viewport:
orange for pushes, blue for Wind, purple for the script. **Preview
Resolution** sets how many arrows are drawn along each axis, independent
of the **Spacing** the solver receives, so a light preview does not cost
a coarse simulation. The arrows show the field at the timeline's current
frame, so scrub the timeline to watch animated fields change.

Arrow lengths show the shape of the field, scaled to the strongest
arrow drawn, not its absolute strength.

## Per-Group Weight

Each object group has a **Force Field Weight** in its material settings.
**1** applies the fields as they are, **0** leaves the group untouched,
and values in between scale them. Pinned vertices follow their pins and
ignore the fields. Static groups have no weight: colliders follow their
own animation.

:::{admonition} Under the hood
:class: dropdown

The fields reach the solver as an acceleration added to every free
vertex once per solver step, at the position the vertex starts that step
from, alongside gravity. Wind instead becomes an air velocity inside the
air drag. The sampled grids travel with the scene parameters, so editing
a field only needs **Update Params**, not a full **Transfer**. World
Scaling does not change how hard a field pushes: fields are authored in
scene units and their strengths are physical, like gravity's.
:::
