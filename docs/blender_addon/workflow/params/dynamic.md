# ⏱️ Dynamic Parameters

Most scene parameters are single scalars. A handful of them can also be
*keyframed*: gravity flipping at frame 60, wind turning on at frame 30,
air density changing mid-simulation. These are **dynamic parameters**, and
you author them by keyframing the slider itself in **Scene
Configuration** — the same gesture as any other Blender property, so the
curves land on the timeline with everything else. There is no separate
list to fill in.

## UI Walkthrough

There is no **Dynamic Parameters** sub-panel. Keyframe a scene setting
the way you keyframe anything else in Blender: hover the field in
**Scene Configuration** and press `I`, or right-click it and choose
**Insert Keyframe**. The curve then lives on the timeline and in the
Graph Editor alongside the rest of the scene's animation, and you retime,
reshape, or delete it there.

Per-group material sliders are keyframed the same way, except that a
value row in **Material Params** carries Blender's own keyframe control
on its right, so you can click instead of hovering. Only the properties
the encoder samples accept one. See
[Material Parameters](material.md).

### Step Functions

Set a key's interpolation to **Constant** when you want a step change
rather than a ramp. The encoder samples whatever shape the curve has, so
a key left on the default eased interpolation produces a slow drift
between the two values instead of the instantaneous change you meant.

Example: "flip gravity at frame 60."

- **Frame 60**: key **Gravity** at its standing value, e.g.
  `(0, 0, -9.8)`, and set that key's interpolation to **Constant**.
- **Frame 61**: key **Gravity** at `(0, 0, 9.8)`. Gravity inverts
  across those two adjacent frames. **Constant** on the first key is
  what keeps the change a step when the second key sits further away.

:::{note}
A `.blend` saved by an older build still carries the add-on's own
scene-parameter keyframe list. It is converted into ordinary F-curves the
first time the file is opened, and a **Hold** keyframe becomes a
**Constant** interpolation segment, which is the shape it always
described. The list's first keyframe stands for the setting at the
solve's **Starting Frame**, so its key lands there, and the viewport
preview reads it there too. An entry with a later keyframe at or before
the **Starting Frame** cannot become a faithful curve; it stays in the
list, and **Transfer** refuses it by name.
:::

## Supported Parameters

| UI label             | Python / TOML key   | Enum               | Value shape           |
| -------------------- | ------------------- | ------------------ | --------------------- |
| **Gravity**          | `gravity`           | `GRAVITY`          | XYZ vector            |
| **Wind**             | `wind`              | `WIND`             | XYZ dir + strength    |
| **Air Density**      | `air_density`       | `AIR_DENSITY`      | scalar                |
| **Air Friction**     | `air_friction`      | `AIR_FRICTION`     | scalar                |
| **Vertex Air Damping** | `vertex_air_damp` | `VERTEX_AIR_DAMP`  | scalar                |
| **Step Size**        | `step_size`         | —                  | scalar                |

Six scene settings can be keyframed. The **Enum** column is the
`param_type` value of the retired keyframe list, which only ever covered
the first five; those five are also the only keys the legacy
`solver.param.dyn(...)` builder accepts. **Step Size** is reachable as
an F-curve only.

These six, and a group's animatable material sliders (see
[Material Parameters](material.md)), are the only add-on settings that
offer a keyframe. Every other setting is read once, at the solve's
**Starting Frame**, and offers no keyframe control. A keyframe that a
saved file still carries on one of them stops **Transfer** with a
message naming its data path, rather than being read once in silence;
delete it. **Inactive Momentum Frames** is one of them: it counts frames
from the start of the solve, so it has no value that could change over
it.

Drivers and NLA strips are refused on every add-on setting, the six
above included. The solve samples only the keyframes in the scene's
active action, so a driven setting, or one keyed inside an NLA strip,
would reach the solve at its **Starting Frame** value while the viewport
shows it changing. **Transfer** names the data path, and the strip and
track for an NLA strip: remove the driver and keyframe the setting
instead, or move the strip's keys into the scene's active action.

What **Transfer** sends does not depend on where the playhead sits:
every setting it reads once is read at the **Starting Frame**, and every
keyframed one is sampled across the solve's frame range.

A wind keyframe follows the same rule as the static **Wind** fields: a
**Direction** of `(0, 0, 0)` with a nonzero **Strength** is refused at
**Transfer**, naming the frame.

Everything the solver reads once at build time (CG tolerance, Max
Contact, and so on) stays constant for the whole simulation. Per-group
material sliders are keyframeable on the same principle as these; see
[Material Parameters](material.md).

## Blender Python API

A legacy path is available from Python. The `solver.param.dyn(...)`
builder writes the add-on's own keyframe list, which is converted into
ordinary F-curves the next time the `.blend` is loaded; it is kept for
existing scripts, and new ones should keyframe the property directly with
Blender's `keyframe_insert`. Every builder method returns `self`, so you
chain them freely. `time(f)` moves the cursor to frame `f` (must be
strictly increasing), and the next `hold()` or `change(...)` attaches a
keyframe at that frame. For wind, `change(direction, strength=...)`
encodes both. The list's first keyframe holds the scene value at the
solve's **Starting Frame**; until the list has been converted to
F-curves, **Transfer** refuses a later keyframe at or before the
**Starting Frame**, or one not after the keyframe before it, naming the
parameter and the frame.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# Flip gravity at frame 60.
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Turn wind on at frame 30.
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Ramp air density to 0.005 by frame 100 (no hold -> linear).
solver.param.dyn("air_density").time(100).change(0.005)

# Remove a dynamic parameter entirely (including its initial keyframe).
solver.param.dyn("gravity").clear()
```

:::{warning}
The Python API for dynamic parameters uses **frames**, not seconds.

```python
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
#                          ^^^^^^^ frame number
```

This is intentional: the Blender UI thinks in frames, and so does this
API. The frontend solver API `session.param.dyn()` (the one called from
inside the decoder on the solver side) uses **seconds**. You normally
never see the seconds form unless you're driving the solver directly
from a Python notebook.
:::

:::{admonition} Under the hood
:class: toggle

**Solver keys**

Each dynamic-parameter key maps to a solver-side key:

| Key                 | Solver key                |
| ------------------- | ------------------------- |
| `gravity`           | `gravity`                 |
| `wind`              | `wind`                    |
| `air_density`       | `air-density`             |
| `air_friction`      | `air-friction`            |
| `vertex_air_damp`   | `isotropic-air-friction`  |
| `step_size`         | `dt`                      |

**Encoding rules**

At transfer time each dynamic parameter becomes a list of
`(time_seconds, value, is_hold)` entries under the matching solver key.
The wire format is unchanged from the retired keyframe list; only the
authoring moved. The observable rules:

- Every keyframed setting is sampled once per frame across the solve's
  frame range, so the curve's own shape is what reaches the solver.
  Samples that lie on the straight line between their neighbors are
  dropped, because the solver interpolates linearly between the ones it
  keeps.
- Frames are converted to seconds as `(frame − starting frame) / fps`,
  so the scene's **Starting Frame** is simulated time zero.
- Gravity and wind are coordinate-converted from Z-up to Y-up.
- Wind is sent as `direction × strength` with the direction normalized.
  A zero direction vector is accepted only with a zero strength, and
  sends a zero wind vector.
- A curve on **Inactive Momentum Frames** is refused rather than
  sampled. The solver reads that setting's key as an on/off flag that
  the frontend schedules off at the end of the count, so a per-frame
  value has no meaning there.
- `is_hold` is always `False` for a sampled curve: the shape is already
  in the samples, with nothing left to hold. It stays in the format for
  the legacy list, whose entries are dropped when they carry fewer than
  two keyframes so the solver falls back to the global scalar.
- A setting keyframed in both places takes its F-curve, because that is
  the one visible on the timeline.
:::
