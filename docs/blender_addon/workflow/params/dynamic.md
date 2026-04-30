# ⏱️ Dynamic Parameters

Most scene parameters are single scalars. A handful of them can also be
*keyframed*: gravity flipping at frame 60, wind turning on at frame 30,
air density changing mid-simulation. These are **dynamic parameters** and
live in the **Dynamic Parameters** sub-panel under Scene Configuration.

## UI Walkthrough

The **Dynamic Parameters** sub-panel is a collapsible section inside
**Scene Configuration**. Expanding it reveals:

1. A **UIList** of currently-active dynamic parameters along the left. Each
   row shows the parameter name (**Gravity**, **Wind**, …). The selected
   row drives the editor below.
2. An **Add** dropdown above (or beside) the list, listing the five
   supported keys. Picking an entry adds a new dynamic parameter of that
   type and automatically creates its frame-1 initial keyframe. Duplicates
   are rejected.
3. A **Remove** button next to Add. It deletes the selected dynamic
   parameter entry and every one of its keyframes.

Once a dynamic parameter is selected in the list, a **per-type editor**
appears below:

- **Gravity**: an XYZ vector field for the gravity vector at the selected
  keyframe.

```{figure} ../../images/dynamic_params/gravity_editor.png
:alt: Dynamic Parameters sub-panel with Gravity selected and a non-hold keyframe showing its XYZ vector editor
:width: 500px

With **Gravity** selected in the top UIList and **Frame 30** selected in
the keyframe list (Hold off), the editor exposes an XYZ vector field for
the gravity value at that keyframe.
```

- **Wind**: an XYZ **Direction** field plus a single **Strength (m/s)**
  scalar. The direction is normalized internally; the strength is the
  magnitude.

```{figure} ../../images/dynamic_params/wind_editor.png
:alt: Dynamic Parameters sub-panel with Wind selected and a non-hold keyframe showing its direction + strength editor
:width: 500px

With **Wind** selected, the editor shows a **Direction** XYZ field and a
separate **Strength (m/s)** scalar. The direction is normalized at encode
time, so only its orientation matters; magnitude comes from strength.
```

- **Air density / Air friction / Vertex air damp**: a single scalar
  field.

```{figure} ../../images/dynamic_params/scalar_editor.png
:alt: Dynamic Parameters sub-panel with Air Density selected and a non-hold keyframe showing its single scalar value editor
:width: 500px

The three scalar-valued keys (**Air Density** shown) collapse the editor
to a single **Value** field. `Air Friction` and `Vertex Air Damp` render
the same way.
```

Below the editor is a **second UIList** showing every keyframe for the
selected parameter, with a frame number per row. The row at index 0 (the
frame-1 initial keyframe) is tagged with an `(Initial)` badge and cannot
be removed.

Two buttons sit below this keyframe list:

- **Add Keyframe**: inserts a new keyframe at the *current scene frame*.
  The new keyframe is initialized from the matching global scalar (so
  adding a gravity keyframe copies the scene's **Gravity** value into
  it); the list is then re-sorted by frame. Inserting at a frame that
  already has a keyframe is rejected.
- **Remove Keyframe**: deletes the selected keyframe. Attempting to
  remove the initial keyframe is rejected with a warning.

### The Initial Keyframe

Every dynamic parameter starts with an **initial keyframe at frame 1**.
The initial keyframe does not store its own value; when encoded it reads
from the matching global scene parameter (**Gravity**; **Direction** +
**Strength (m/s)** for wind; **Air Density**, **Air Friction**, **Vertex
Air Damping**). That way the timeline always begins from whatever the
scene-wide value is.

A dynamic parameter entry is only sent to the solver if it has **more
than one** keyframe. A single initial keyframe is treated the same as
"not keyframed": the global scalar is used.

### Hold: Step-Function Semantics

Keyframes other than the initial one expose a **Hold** checkbox in the
per-keyframe editor. When Hold is on, the previous keyframe's value is
re-emitted at this frame instead of a new value. Combined with an
adjacent non-hold keyframe one frame later, this is how you build step
functions: a hard instantaneous change with no linear ramp.

Example: "flip gravity at frame 60."

- **Frame 1** *(initial)*: value read from the scene's **Gravity**,
  e.g. `(0, 0, -9.8)`.
- **Frame 60**: **Hold** on. The encoder emits `(0, 0, -9.8)` here
  as well, holding the initial value until exactly frame 60.
- **Frame 61**: **Hold** off, value `(0, 0, 9.8)`. Gravity inverts.

Without the hold at frame 60, the solver would have linearly ramped
gravity between frame 1 and frame 61, producing a slow drift instead of a
flip.

```{figure} ../../images/dynamic_params/gravity_flip_example.png
:alt: Dynamic Parameters sub-panel showing a gravity flip at frame 30 with three keyframes (Initial, 30, 31)
:width: 500px

A concrete gravity-flip setup as it appears in the sub-panel. The
**Gravity (3 kf)** entry is selected in the top UIList; its keyframe
list below shows **Frame 1 (Initial)** (reads the scene's gravity),
**Frame 30** (Hold, which re-emits that same value), and **Frame 31** (steps
to the new `(0, 0, 9.8)` value). The `(Initial)` badge on frame 1 marks
the one keyframe that cannot be removed. The Hold checkbox on frame 30
is what makes this a step function instead of a linear ramp from frame
1 to frame 31.
```

## Supported Parameters

| UI label             | Python / TOML key   | Enum               | Value shape           |
| -------------------- | ------------------- | ------------------ | --------------------- |
| **Gravity**          | `gravity`           | `GRAVITY`          | XYZ vector            |
| **Wind**             | `wind`              | `WIND`             | XYZ dir + strength    |
| **Air Density**      | `air_density`       | `AIR_DENSITY`      | scalar                |
| **Air Friction**     | `air_friction`      | `AIR_FRICTION`     | scalar                |
| **Vertex Air Damp**  | `vertex_air_damp`   | `VERTEX_AIR_DAMP`  | scalar                |

Only these five parameters are keyframeable. Others (step size, CG
tolerance, and so on) have to stay constant across the simulation.

## Blender Python API

The same workflow is available from Python. The
`solver.param.dyn(...)` builder drives the same dynamic-parameter list
that the sub-panel shows. Every builder method returns `self`, so you
chain them freely -- `time(f)` moves the cursor to frame `f` (must be
strictly increasing), and the next `hold()` or `change(...)` attaches a
keyframe at that frame. For wind, `change(direction, strength=...)`
encodes both.

```python
from zozo_contact_solver import solver

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

**Encoding rules**

At transfer time each dynamic parameter becomes a list of
`(time_seconds, value, is_hold)` entries under the matching solver key.
The observable rules:

- Frames are converted to seconds as `(frame − 1) / fps`.
- Gravity and wind are coordinate-converted from Z-up to Y-up.
- Wind is sent as `direction × strength` with the direction normalized.
  A zero direction vector produces a zero wind vector regardless of the
  strength.
- A hold keyframe re-emits the previous keyframe's value; it does not
  sample the live scene parameter.
- Parameters with fewer than two keyframes are dropped and the solver
  falls back to the global scalar.
:::
