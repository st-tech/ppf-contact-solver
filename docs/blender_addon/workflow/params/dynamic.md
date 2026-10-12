# ⏱️ Dynamic Parameters

A handful of scene settings can change during the simulation: gravity
flipping at frame 60, wind turning on at frame 30, air density changing
mid-run. These are **dynamic parameters**. You author one by keyframing its
slider in **Scene Configuration**, the same way you keyframe any Blender
property, so the curve appears on the timeline with the rest of the scene's
animation.

## UI Walkthrough

1. Move the playhead to a frame.
2. Set the value, hover over the field in **Scene Configuration** and press
   `I`, or right-click the field and choose **Insert Keyframe**.
3. Move the playhead to another frame, change the value and key it again.

The curve appears on the timeline and in the Graph Editor alongside the rest
of the scene's animation. Retime, reshape or delete it there. With
**Preview Direction** on, the gravity and wind arrows show the keyed value
at the playhead.

A group's animatable material sliders are keyframed the same way, and each
animatable value row in **Material Params** carries Blender's keyframe
button on its right. See [Material Parameters](material.md).

### Step Functions

Use a **Constant** interpolation when you want a step rather than a ramp.
**Constant** holds a key's value until the next key. A key left on a
smooth interpolation changes gradually toward the next key instead.

Example: "gravity points down until frame 90, then flips."

1. Move the playhead to frame 60 and key **Gravity** at its standing value,
   for example `(0, 0, -9.8)`.
2. In the Graph Editor or the Dope Sheet, select that key, press `T` and
   choose **Constant**.
3. Move the playhead to frame 90, set **Gravity** to `(0, 0, 9.8)` and key
   it.

Gravity stays at `(0, 0, -9.8)` through frame 89 and is `(0, 0, 9.8)` from
frame 90. On a smooth interpolation it would change gradually between
frame 60 and frame 90 instead. The solver reads the curve once per frame,
so the flip takes one frame. When the two keys are one frame apart, as in
the Python example below, the interpolation makes no difference.

## Supported Parameters

| Slider in Scene Configuration                               | Python property                   | Value                                                  |
| ----------------------------------------------------------- | --------------------------------- | ------------------------------------------------------ |
| **Gravity (m/s²)**                                          | `gravity_3d`                      | XYZ vector                                             |
| **Direction** and **Strength (m/s)** in the **Wind** box    | `wind_direction`, `wind_strength` | XYZ vector and scalar; keying either one animates the wind |
| **Air Density (kg/m³)**                                     | `air_density`                     | scalar                                                 |
| **Air Friction**                                            | `air_friction`                    | scalar                                                 |
| **Vertex Air Damping** in **Advanced Params**               | `vertex_air_damp`                 | scalar                                                 |
| **Step Size**                                               | `step_size`                       | scalar                                                 |

These six settings, and a group's animatable material sliders, are the
only add-on settings that offer a keyframe. Every other setting is read
once, at the solve's **Starting Frame**, and stays constant for the whole
simulation. This includes **PCG Tolerance**, **Line Search Max T** and
**Inactive Momentum Frames**. What **Transfer** sends does not depend on
where the playhead is: a setting read once is read at the **Starting
Frame**, and a keyframed one is sampled across the solve's frame range.

**Transfer** stops and names the problem in these cases:

- **A keyframe on a setting that offers none.** The message names its data
  path. Delete the keyframe. **Inactive Momentum Frames** counts frames
  from the start of the solve, so it has no value that could change over
  it.
- **A driver on an add-on setting**, the six above included. The message
  names the data path. Remove the driver and keyframe the setting instead.
- **A keyframe inside an NLA strip.** The message names the data path, the
  strip and the track. Move the strip's keys into the scene's active
  action.
- **A wind keyframe with a `(0, 0, 0)` Direction and a nonzero Strength.**
  The message names the frame, as for the static **Wind** fields.

The solve samples only the keyframes in the scene's active action. A driven
setting, or one keyed inside an NLA strip, would otherwise reach the solve
at its **Starting Frame** value while the viewport shows it changing.

## Blender Python API

Keyframe a setting on the add-on state with Blender's `keyframe_insert`,
using the property name from the table above:

```python
import bpy

state = bpy.context.scene.zozo_contact_solver.state

# Flip gravity at frame 60.
state.gravity_3d = (0, 0, -9.8)
state.keyframe_insert(data_path="gravity_3d", frame=60)
state.gravity_3d = (0, 0, 9.8)
state.keyframe_insert(data_path="gravity_3d", frame=61)

# Ramp the wind up between frames 30 and 60.
state.wind_direction = (0, 1, 0)
state.wind_strength = 0.0
state.keyframe_insert(data_path="wind_strength", frame=30)
state.wind_strength = 5.0
state.keyframe_insert(data_path="wind_strength", frame=60)
```

`keyframe_insert` leaves the keys on Blender's default interpolation. To
hold a value until the next key, set the key points of that curve to
`CONSTANT`:

```python
anim = bpy.context.scene.animation_data
for layer in anim.action.layers:
    for strip in layer.strips:
        for fc in strip.channelbag(anim.action_slot).fcurves:
            if fc.data_path == "zozo_contact_solver.state.gravity_3d":
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
```

### The `dyn()` Builder

`solver.param.dyn(...)` is a script helper. It records keyframes in a list
that the panel does not show, instead of animation curves on the sliders.
**Transfer** reads that list, and a setting that has both a list entry and
an animation curve takes the curve. Opening a `.blend` that holds list
entries turns each entry that has two or more keyframes into animation
curves on the sliders and removes it from the list. One case stays in the
list (see **Keyframe list conversion** under **Under the hood** below).
The builder accepts five keys, `gravity`, `wind`, `air_density`,
`air_friction` and `vertex_air_damp`, and not `step_size`.

Every builder method returns `self`, so you chain them.

- `time(f)` moves the cursor to frame `f`. The cursor starts at frame 1,
  and each call must move it forward.
- `hold()` attaches a keyframe at the cursor that repeats the previous
  value, which makes a step.
- `change(value)` attaches a keyframe at the cursor with a new value. For
  wind, `change(direction, strength=...)` sets both.
- The list's first keyframe stands for the setting's value at the
  **Starting Frame**. **Transfer** refuses a later keyframe at or before
  the **Starting Frame**, or one not after the keyframe before it, naming
  the parameter and the frame.

The import in the first line below works for an add-on in the **User
Default** repository. If you installed the add-on from a remote repository,
import `solver` as shown in
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# Flip gravity at frame 60.
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Turn wind on at frame 30.
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Ramp air density to 0.005 by frame 100 (no hold -> linear).
solver.param.dyn("air_density").time(100).change(0.005)

# Remove the list entry for gravity, initial keyframe included.
solver.param.dyn("gravity").clear()
```

`clear()` empties only the list. Once a file has been reloaded and its list
converted, it removes nothing; delete the animation curve in the Graph
Editor instead.

:::{warning}
The builder takes **frames**, not seconds, like the rest of the add-on's
interface. In `.time(60)` the `60` is a frame number. The notebook
frontend's `session.param.dyn()` takes seconds.
:::

The [Python API page](../../integrations/python_api.md#dynamic-parameters)
shows the same builder.

### From MCP

Five [MCP](../../integrations/mcp.md) tools edit the same keyframe list as
`solver.param.dyn()`:
`add_dynamic_param`, `add_dynamic_param_keyframe`,
`remove_dynamic_param_keyframe`, `remove_dynamic_param` and
`list_dynamic_params`. **Transfer** applies the same checks to the list. A
tool names the setting by `param_type`: `GRAVITY`, `WIND`, `AIR_DENSITY`,
`AIR_FRICTION` or `VERTEX_AIR_DAMP`. **Step Size** has no entry, so key it
on its slider.

- `add_dynamic_param` starts an entry with its initial keyframe at frame 1,
  the list's first keyframe, and refuses a second entry for the same
  setting. An entry with only that keyframe changes nothing.
- `add_dynamic_param_keyframe` adds a keyframe at `frame` (1 or later) and
  refuses a frame that already has one. It takes `gravity` for `GRAVITY`,
  `wind_direction` with `wind_strength` (m/s) for `WIND`, and `value` for
  the other settings, plus an optional `use_hold` that repeats the previous
  value.
- `remove_dynamic_param_keyframe` removes the keyframe at a frame, except
  the initial one. `remove_dynamic_param` removes the whole entry.
- `list_dynamic_params` reports every entry with its keyframes.

:::{admonition} Under the hood
:class: toggle

**Solver keys**

Each keyframed setting maps to a solver-side key:

| Setting                | Python property                   | `dyn()` key       | Solver key               |
| ---------------------- | --------------------------------- | ----------------- | ------------------------ |
| **Gravity**            | `gravity_3d`                      | `gravity`         | `gravity`                |
| **Wind**               | `wind_direction`, `wind_strength` | `wind`            | `wind`                   |
| **Air Density**        | `air_density`                     | `air_density`     | `air-density`            |
| **Air Friction**       | `air_friction`                    | `air_friction`    | `air-friction`           |
| **Vertex Air Damping** | `vertex_air_damp`                 | `vertex_air_damp` | `isotropic-air-friction` |
| **Step Size**          | `step_size`                       | (none)            | `dt`                     |

**Keyframe list conversion**

When a `.blend` holds keyframe-list entries, each entry with two or more
keyframes becomes animation curves on its sliders. The entry's first
keyframe becomes a key at the **Starting Frame**, and a hold becomes a
**Constant** segment. An entry with a later keyframe at or before the
**Starting Frame** cannot become a curve; it stays in the list and
**Transfer** refuses it by name.

**Encoding rules**

At transfer time each keyframed setting becomes a list of
`(time_seconds, value, is_hold)` entries under its solver key. The
observable rules:

- Every keyframed setting is sampled once per frame across the solve's
  frame range, so the curve's own shape is what reaches the solver.
  Samples that lie on the straight line between their neighbors are
  dropped, because the solver interpolates linearly between the ones it
  keeps.
- Frames are converted to seconds as `(frame - starting frame) /
  (FPS x Time Scale)`, so the scene's **Starting Frame** is simulated time
  zero.
- Gravity and wind are converted from Blender's Z-up axes to the solver's
  Y-up axes.
- Wind is sent as `direction × strength` with the direction normalized.
  A zero direction vector is accepted only with a zero strength, and
  sends a zero wind vector.
- A curve on **Inactive Momentum Frames** is refused rather than sampled.
  The solver reads that setting's key as an on/off flag that the frontend
  schedules off at the end of the count, so a per-frame value has no
  meaning there.
- `is_hold` is always `False` for a sampled curve: the shape is already in
  the samples, with nothing left to hold. Keyframe-list entries with fewer
  than two keyframes are dropped, so the solver uses the setting's global
  value.
:::
