# 🌐 Scene Parameters

Scene-wide simulation settings (gravity, time step, wind, air density,
frame count, CG tolerances) all live in one place: the **Scene
Configuration** panel in the 3D viewport's sidebar.

## Scene Configuration Panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on's tab.
The **Scene Configuration** panel is the first panel; everything below
is scoped to the whole scene rather than a single object group.

```{figure} ../../images/scene_params/panel_overview.png
:alt: Scene Configuration panel in its default state. Open Profile button at the top, basic parameters, and four collapsed sub-section headers (Wind, Advanced Params, Dynamic Parameters, Invisible Colliders)
:width: 500px

Default layout of the **Scene Configuration** panel. Top-down: the
**Open Profile** row (no profile loaded yet), the basic parameters
block, and the four collapsible sub-section headers. Grayed rows
(like `Inactive Momentum Frames` here) are gated on scene contents.
```

The top row is the **scene profile** row. If no profile is loaded you
see an **Open Profile** button plus a small save icon; once a profile
is open the row collapses into a dropdown (the profile selector) with
four icon buttons on the right: **Open**, **Clear**, **Reload**, and
**Save** (identical layout to the per-group material profile row).
Opening a profile reads its TOML payload into every scene parameter,
replaces dynamic parameters, and replaces invisible colliders. Clear
drops the association without touching current values. Reload re-reads
the file from disk. Save writes the current scene parameters,
dynamic-parameter block, and invisible-collider block back out to the
file.

:::{important}
**The scene-profile TOML file is authored through the Save icon, not by
hand.** Configure the Scene Configuration panel (basic parameters,
Wind, Advanced, Dynamic Parameters, Invisible Colliders) the way you
want it, then click the **Save** icon on the profile row: the add-on
creates a new `.toml` on the first save and overwrites the currently
selected entry on later saves. The TOML layout shown further down is
for inspection only; the intended workflow is always UI → Save.

```{figure} ../../images/scene_params/save_icon.png
:alt: Scene Configuration panel with the floppy-disk Save icon at the top-right of the Open Profile row highlighted in red
:width: 500px

The **Save** icon (floppy disk, highlighted in red) at the top-right of
the Scene Configuration profile row. Clicking it writes the entire
scene block (basic parameters, wind, advanced, dynamic parameters,
invisible colliders) out to the `.toml` file.
```
:::

```{figure} ../../images/scene_params/profile_row_loaded.png
:alt: Scene Configuration panel with a scene profile loaded. The top row is now a Profile dropdown followed by Open / Clear / Reload / Save icons
:width: 500px

With a scene profile loaded, the top row collapses into a **Profile**
dropdown with four icons on the right: **Open**, **Clear**, **Reload**,
**Save**. Switching entries in the dropdown applies that preset to the
scene (overwriting scene parameters, dynamic parameters, and invisible
colliders).
```

Below the profile row, the panel lays out the basic parameters one per
line: **FPS** (inside a small box, with a checkbox to drive it from
Blender's render FPS instead), **Frame Count**, **Step Size**,
**Min Newton Steps**, **Air Density**, **Air Friction**, **Gravity**
(a 3-component vector), a **Preview Direction** toggle (viewport arrow),
and an **Inactive Momentum Frames** row that is grayed-out unless the
scene contains at least one **Shell** group.

Below the basics, three collapsible sub-sections hang off the panel.
Each has a disclosure triangle on its header row and toggles
open/closed independently:

- **Wind**: wind direction vector, preview toggle, wind strength.

```{figure} ../../images/scene_params/wind_expanded.png
:alt: Scene Configuration panel with the Wind sub-section expanded. Direction XYZ, Preview Direction toggle, and Strength (m/s) rows
:width: 500px

With the **Wind** disclosure triangle open, the sub-section reveals a
**Direction** XYZ field, a **Preview Direction** viewport toggle, and a
**Strength (m/s)** scalar. Encoding combines direction × strength, so a
zero-direction vector disables wind regardless of the strength value.
```

- **Advanced Params**: contact NNZ, vertex air damp, auto-save + its
  interval, CCD line-search max t, constraint ghat, CG max iter, CG
  tol, include-face-mass, friction-mode, and disable-contact toggles.

```{figure} ../../images/scene_params/advanced_expanded.png
:alt: Scene Configuration panel with the Advanced Params sub-section expanded. Max Contact, Vertex Air Damping, Auto Save, Line Search Max T, Constraint Gap, PCG Max Iterations, PCG Tolerance, Include Face Mass, Disable Contact
:width: 500px

**Advanced Params** exposes the tuning knobs most users never need to
touch: contact-matrix capacity (**Max Contact**), per-vertex air drag,
checkpointing (**Auto Save**), CCD line-search bounds, PCG iteration
cap and tolerance, and two debugging toggles. Raise PCG limits for
stiff systems; raise Max Contact only when the solver reports overflow.
```

- **Dynamic Parameters**: keyframed gravity / wind / air density /
  air friction / vertex air damp. Covered separately in
  [Dynamic Parameters](dynamic.md).
- **Invisible Colliders**: walls and spheres with their own keyframe
  lists. Covered separately in [Invisible Colliders](../constraints/colliders.md).

Only the sub-section headers are visible when collapsed; click the
triangle on any of them to expand.

## Basic

| UI label                     | Python / TOML key          | Default       | Description                                                         |
| ---------------------------- | -------------------------- | ------------- | ------------------------------------------------------------------- |
| **Frame Count**              | `frame_count`              | 180           | Simulation length in frames. Minimum 10.                            |
| **FPS**                      | `frame_rate`               | 60            | FPS used to convert frames to seconds at encode time. Minimum 24.   |
| **Step Size**                | `step_size`                | 0.01          | Solver sub-step Δt, in seconds. Range 0.001 – 0.01.                 |
| **Min Newton Steps**         | `min_newton_steps`         | 1             | Minimum Newton iterations per step. 1 – 64.                         |
| **Air Density (kg/m³)**      | `air_density`              | 0.001         | Air density, kg/m³. Range 0 – 0.01.                                 |
| **Air Friction**             | `air_friction`             | 0.2           | Tangential-to-normal air drag ratio, 0 – 1 (see below).             |
| **Gravity (m/s²)**           | `gravity_3d`               | (0, 0, -9.8)  | Gravity vector, m/s² (Z-up Blender frame).                          |
| **Preview Direction** (gravity) | `preview_gravity_direction` | `False`    | Draw the gravity-direction arrow overlay in the viewport. Overlay-only. |
| **Inactive Momentum Frames** | `inactive_momentum_frames` | 0             | Frames over which shell momentum is ignored at startup (0 – 600).   |

**Inactive Momentum Frames** is only honored when the scene contains at
least one **Shell** group; otherwise the UI disables the row.

### Air Friction

At every shell vertex, the solver accumulates an air-damping force
that resists motion relative to the wind. The move `(x¹ - x⁰) - Δt·w`
(current minus previous position, minus the wind displacement over
one step) is split by the vertex normal into a **normal** component
and a **tangential** component, and the two are weighted differently
in the damping energy:

$$
E_\text{air} = \tfrac{1}{2}\,\bigl(v_n^{\,2} + \texttt{air\_friction}\,\lVert v_t \rVert^{2}\bigr)
$$

- **Normal** drag is always on with coefficient `1`. Pushing a shell
  face into or out of the air is always resisted.
- **Tangential** drag is scaled by **Air Friction**. At `0`, air
  offers no sideways resistance, so a shell can slide edgewise through
  still air unopposed. At `1`, tangential and normal drag are equal
  in magnitude. The default `0.2` gives air a little "grip" on the
  fabric without over-damping swings.

The per-vertex air-damping force is then scaled by the vertex's
associated area and by **Air Density** before being added to the
Newton solve. A side-effect of that last multiplication: at
`air_density = 0` there is no shell air damping at all, regardless of
**Air Friction**.

**Vertex Air Damping** (under **Advanced**) is a separate, isotropic
per-vertex damper applied to every vertex in the scene, with no area or
air-density weighting and no directional split. Use it when you need to
calm a rod or a particle-heavy scene that **Air Friction** does not
reach.

Both **Air Friction** and **Vertex Air Damping** are keyframeable;
see [Dynamic Parameters](dynamic.md).

## Wind

| UI label               | Python / TOML key       | Default   | Description                                     |
| ---------------------- | ----------------------- | --------- | ----------------------------------------------- |
| **Wind** (disclosure)  | `show_wind`             | `False`   | Whether the Wind sub-section is expanded.       |
| **Direction**          | `wind_direction`        | (0, 0, 0) | Wind direction (normalized at encode time).     |
| **Preview Direction**  | `preview_wind_direction`| `False`   | Draw the wind-direction arrow overlay in the viewport. Overlay-only. |
| **Strength (m/s)**     | `wind_strength`         | 0.0       | Wind speed, m/s. Range 0 – 1000.                |

The encoder combines the two into a single `direction × strength` vector
before sending. If **Direction** is `(0, 0, 0)`, no wind is applied
regardless of **Strength (m/s)**.

```{figure} ../../images/scene_params/wind_preview.png
:alt: Scene Configuration with Wind expanded. Direction (0, 1, 0), Strength 5.0 m/s, Preview Direction on, and a green wind arrow plus sphere overlay drawn in the 3D viewport
:width: 500px

A live example of the Wind fields. With **Direction = (0, 1, 0)** and
**Strength = 5.0 m/s**, the encoder ships `direction × strength = (0, 5,
0)` to the solver. Turning **Preview Direction** on draws the green
wind-arrow overlay (and a translucent sphere whose radius scales with
strength) so you can judge the wind field relative to scene geometry
without running a simulation. The viewport label echoes the magnitude
and normalized direction live.
```

## Advanced

| UI label                  | Python / TOML key    | Default     | Description                                                      |
| ------------------------- | -------------------- | ----------- | ---------------------------------------------------------------- |
| **Max Contact**           | `contact_nnz`        | 100 000 000 | Capacity of the contact sparse matrix (non-zero entries).        |
| **Vertex Air Damping**    | `vertex_air_damp`    | 0.0         | Per-vertex isotropic air damping, 0 – 1.                         |
| **Auto Save**             | `auto_save`          | `False`     | Enable periodic solver-state snapshots on the remote.            |
| **Auto Save Interval**    | `auto_save_interval` | 10          | Auto-save interval, in frames. Minimum 1.                        |
| **Line Search Max T**     | `line_search_max_t`  | 1.25        | CCD line-search maximum step factor. Range 0.1 – 10.             |
| **Constraint Gap**        | `constraint_ghat`    | 0.001       | Barrier gap distance. Range 0.0001 – 0.1.                        |
| **PCG Max Iterations**    | `cg_max_iter`        | 10 000      | Max iterations for the PCG linear solver. 100 – 100 000.         |
| **PCG Tolerance**         | `cg_tol`             | 0.001       | PCG relative tolerance. 0.0001 – 0.1.                            |
| **Include Face Mass**     | `include_face_mass`  | `False`     | Fold shell face mass into attached solids.                       |
| **Friction Mode**         | `friction_mode`      | `MIN`       | How to combine the two contacting objects' friction coefficients. Choices: `MIN`, `MAX`, `MEAN`. See [Friction at a contact](material.md#shared-parameters). |
| **Disable Contact**       | `disable_contact`    | `False`     | Turn off contact detection entirely (debugging / ablation).      |

Raise **PCG Max Iterations** (and lower **PCG Tolerance**) for stiff
systems that fail to converge. Raise **Max Contact** only if the solver
reports a contact-matrix overflow; it's an upper bound, not a target.

## Previewing Direction

Both **Gravity** and **Wind** have a **Preview Direction** toggle next to
their vector fields. Turning it on draws an arrow (and a translucent
sphere for scale) in the 3D viewport showing the current direction (and
for wind, the magnitude) so you can judge the setup before running a
simulation.

```{figure} ../../images/scene_params/gravity_preview.png
:alt: Scene Configuration with gravity Preview Direction enabled and the resulting arrow visible in the 3D viewport
:width: 500px

With **Preview Direction** enabled next to the gravity vector, a blue
arrow is drawn in the 3D viewport pointing in the gravity direction.
The text label echoes the magnitude and normalized direction. Toggle it
off to declutter the viewport.
```

```{figure} ../../images/scene_params/wind_preview.png
:alt: Scene Configuration with wind Preview Direction enabled and the resulting green arrow visible in the 3D viewport
:width: 500px

Wind preview with direction `(0, 1, 0)` and strength `5.0 m/s`. The
green arrow's length scales with the wind strength, so you can see at a
glance how strong the wind field is relative to the scene geometry.
```

The **Preview Direction** toggles (one for gravity, one for wind) are
also exposed to Python, so you can flip them from a script. They are
purely viewport overlays; they never affect the simulation.

## What `Update Params` Does

On the Solver panel, **Update Params** re-encodes the parameters (scene
block, per-group block, pin config, merge pairs, dynamic parameters,
invisible colliders) and ships them to the solver, then triggers a
rebuild. Geometry is **not** resent; the mesh buffers on the server are
preserved. Use this when you want to tweak material parameters, gravity,
or dynamic keyframes without paying the full transfer cost.

Topology changes (adding or removing vertices, edges, or faces) still
require a full **Transfer**, because the mesh hash changes.

## Scene Profiles

A **scene profile** is a named set of scene parameters saved to a TOML
file. The shipped example contains:

| Preset        | Highlights                                                                  |
| ------------- | --------------------------------------------------------------------------- |
| `Default`     | step 0.001 s, 180 frames at 60 FPS, gravity (0, 0, -9.8).                   |
| `Windy`       | Same as `Default`, plus `wind_direction = (0, 1, 0)`, `wind_strength = 5.0`. |
| `HighRes`     | step 0.001 s (the minimum), 360 frames, `cg_max_iter = 20000`.              |
| `SlowMotion`  | 600 frames at 120 FPS. Step size stays at 0.001 s.                          |
| `ZeroGravity` | `gravity = (0, 0, 0)`, 300 frames.                                          |

```{note}
The `Default` preset above is a **profile name** in the shipped TOML, not
the addon's new-scene default. The presets deliberately tighten `step_size`
to 0.001 s for the more demanding example setups; a fresh Blender scene
still starts at the addon default of 0.01 s (see the [Basic](#basic) table).
```

Unlike material profiles, scene profiles also capture **dynamic
parameters** and **invisible colliders**. Applying the profile clears
existing dynamic parameters and invisible colliders first, then rebuilds
them from the TOML entries.

The example below is shown so you can read or diff an existing file;
it is **produced by clicking Save**, not written by hand. Tweak the
scene in the panel and click Save again rather than editing TOML
directly.

Example excerpt:

```toml
[Default]
step_size = 0.001
min_newton_steps = 1
air_density = 0.001
air_friction = 0.2
gravity = [0.0, 0.0, -9.8]
frame_count = 180
frame_rate = 60
wind_direction = [0.0, 1.0, 0.0]
wind_strength = 0.0
contact_nnz = 100000000
cg_max_iter = 10000
cg_tol = 0.001

[[Default.dyn_params]]
param_type = "WIND"

[[Default.dyn_params.keyframes]]
frame = 1

[[Default.dyn_params.keyframes]]
frame = 40
use_hold = true

[[Default.dyn_params.keyframes]]
frame = 60
wind_direction_value = [0.0, 1.0, 0.0]
wind_strength_value = 5.0

[[Default.colliders]]
collider_type = "SPHERE"
name = "Floor Bowl"
position = [0.0, 0.0, 0.0]
radius = 0.98
hemisphere = true
invert = true
contact_gap = 0.001
friction = 0.0

[[Default.colliders.keyframes]]
frame = 1
```

:::{note}
Scene profiles do **not** touch object groups, pins, or merge pairs.
Those live on per-group material profiles and on the scene file itself.
:::

## Blender Python API

The same workflow is available from Python. The settings are reachable
through `solver.param`. Assignments are immediate; they don't require a
transfer, but do require `update_params()` before the solver actually
sees the new values on the next run.

```python
from zozo_contact_solver import solver

solver.param.gravity = (0, 0, -9.8)
solver.param.frame_count = 180
solver.param.step_size = 0.001

# Ship the new values to the remote without re-sending geometry.
solver.update_params()
```

:::{admonition} Under the hood
:class: toggle

**Hidden `use_frame_rate_in_output`**

`use_frame_rate_in_output` is a hidden boolean. When true, Blender's
render FPS is used in place of `frame_rate` when converting frames to
seconds at encode time. Leave it off unless you want the output time
base tied to the render FPS.

**Scene profile shape**

A scene profile TOML entry has a flat block of the scene-parameter keys
at the top, followed by `[[<Profile>.dyn_params]]` arrays describing
keyframed parameters and `[[<Profile>.colliders]]` arrays describing
walls and spheres.

**Solver payload keys**

At transfer time the scene parameters are sent under solver-side names:

```
dt, min-newton-steps, air-density, air-friction, gravity[3], wind[3],
frames, fps, csrmat-max-nnz, constraint-ghat, cg-max-iter, cg-tol,
include-face-mass, disable-contact, inactive-momentum,
line-search-max-t, auto-save, stitch-stiffness, ...
```

**Update Params** reships this payload without re-encoding geometry.
:::
