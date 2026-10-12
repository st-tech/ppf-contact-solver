# 🌐 Scene Parameters

The **Scene Configuration** panel holds the settings that apply to the
whole simulation: gravity, time step, wind, air density, frame count,
force fields, invisible colliders and the PCG linear-solver settings. Save
them as a scene profile to reuse them in another scene.

## Scene Configuration Panel

1. Press `N` in the 3D viewport to open the sidebar.
2. Click the **ZOZO's Contact Solver** tab.
3. Find **Scene Configuration**, the third panel, below **Backend
   Communicator** and **Solver**.

Everything in this panel applies to the whole scene, not to one object
group.

```{figure} ../../images/scene_params/panel_overview.png
:alt: Scene Configuration panel in its default state, top to bottom. An Open Profile row with a Save icon; a Take FPS from Scene box holding FPS and Time Scale; a Take Starting Frame from Scene box holding Starting Frame; Frame Count, Step Size, Min Newton Steps, Air Density, Air Friction, World Scaling, Gravity, Preview Direction and a grayed Inactive Momentum Frames; then the collapsed box headers Save and Checkpoints, Wind, Invisible Colliders, Linear System Solver and Advanced Params
:width: 500px

Default layout of the **Scene Configuration** panel. Top-down: the
profile row, the basic settings, and the collapsible box headers. The
**Force Fields** header is between **Wind** and **Invisible Colliders**
and is not in this capture. A grayed row (**Inactive Momentum Frames**
here) is gated on the scene's contents.
```

From top to bottom the panel holds:

- The **profile row**, for saving and loading the settings. See
  [Scene Profiles](#scene-profiles).
- A **Take FPS from Scene** box with **FPS** and **Time Scale**, and a
  **Take Starting Frame from Scene** box with **Starting Frame**.
- The basic settings, one per row: **Frame Count**, **Step Size**, **Min
  Newton Steps**, **Air Density (kg/m³)**, **Air Friction**, **World
  Scaling**, **Gravity (m/s²)**, **Preview Direction** and **Inactive
  Momentum Frames**. See [Basic](#basic).
- Six collapsible boxes, in this order. Each has a disclosure triangle on
  its header row, opens and closes on its own, and starts closed.
  - **Save and Checkpoints**: resumable states. See
    [Save and Checkpoints](#save-and-checkpoints).
  - **Wind**: direction and strength. See [Wind](#wind).
  - **Force Fields**: field objects, sampling, script and **Visualize**.
    See [Force Fields](force_fields.md).
  - **Invisible Colliders**: walls and spheres with their own keyframes.
    See [Invisible Colliders](../constraints/colliders.md).
  - **Linear System Solver**: **PCG Max Iterations**, **PCG Tolerance**,
    **Preconditioner** and, only while **Schwarz** is selected, **Schwarz
    Levels**. See [Linear System Solver](#linear-system-solver).
  - **Advanced Params**: **Max Contact**, **Vertex Air Damping**, **Line
    Search Max T**, **Constraint Gap**, **Include Face Mass**, **Friction
    Mode** and **Disable Contact**. See [Advanced Params](#advanced-params).

**Step Size**, **Air Density**, **Air Friction**, **Gravity**, the wind
and **Vertex Air Damping** can be keyframed on their own sliders. See
[Dynamic Parameters](dynamic.md).

## Scene Profiles

A **scene profile** saves the scene settings under a name in a `.toml`
file, so you can load them in another scene or share them with a team.
One file holds any number of entries. The row at the top of the panel
manages them. It works like the profile row of the **Backend
Communicator** panel (see [Connection Profiles](../../connections/profiles.md)).

| State        | What the row shows                                          |
| ------------ | ----------------------------------------------------------- |
| No file open | An **Open Profile** button and the **Save Scene Profile** icon. |
| File open    | A **Profile** dropdown, followed by four icon buttons.      |

The add-on remembers the file's location in the `.blend`. If the file is
moved or deleted, the row shows **Open Profile** again the next time
Blender is started.

The four buttons show icons only. Hover over one to read its name.

| Icon            | Button                    | Effect                                                                                                         |
| --------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Folder          | **Open Scene Profile**    | Pick a `.toml` file. The dropdown lists its entries sorted by name, and the first one is selected and applied. |
| X               | **Clear Scene Profile**   | Stop using the file. The settings keep their values.                                                           |
| Circular arrows | **Reload Scene Profile**  | Re-read the file and apply the selected entry again.                                                           |
| Floppy disk     | **Save Scene Profile**    | Overwrite the selected entry. With no file open, ask for a file and an **Entry Name**.                         |

Picking another entry in the dropdown applies it at once.

### Saving a Profile

Create profiles with the **Save Scene Profile** icon, the floppy disk at
the right end of the profile row. You do not write the `.toml` file
yourself.

1. Set the scene up in the panel.
2. If the row shows a **Profile** dropdown, click **Clear Scene Profile**
   so that the row shows **Open Profile**.
3. Click the **Save Scene Profile** icon.
4. In the file browser, choose or type the `.toml` file name (the
   suggested name is `scene_profile.toml`). To add an entry to an existing
   file, choose that file. Set **Entry Name** in the file browser's
   options (the default is `Default`; a name the file already holds
   replaces that entry), then confirm.
5. The status bar reports `Saved to '<Entry Name>'`. The dropdown shows the
   new entry.

:::{warning}
Use an ASCII **Entry Name**: letters, digits, `_` and `-`. A name such as
`Büro`, with a non-ASCII letter and no space or punctuation, is written
without quotes, which TOML does not allow. The add-on then cannot read the
file, and the next save replaces the file with the one new entry. A name
with a space or punctuation is quoted and is safe.
:::

With a file open and an entry selected, the **Save Scene Profile** icon
overwrites that entry at once, with no file browser. The other entries of
the file stay in it. Saving rewrites the whole file, so it removes
comments and formatting. If the file cannot be parsed, the add-on reads it
as empty and saving replaces it with the one entry, so keep a backup of a
file you edit by hand.

### What a Profile Holds

A profile stores most of the settings in the [Basic](#basic),
[Wind](#wind), [Linear System Solver](#linear-system-solver) and
[Advanced Params](#advanced-params) tables, under the property name shown
there (**Gravity** is stored as `gravity`). It also stores **Auto Save**,
**Auto Save Interval** and the scene's invisible colliders. It does not
store:

- **World Scaling**, **Friction Mode**, `fix_xz`, **Save State on Finish**,
  **Keep Saved States** and the **Save Checkpoints** list.
- The **Preview Direction** toggles and whether a box is open.
- The **Force Fields** settings.
- Keyframes drawn on a slider. They stay in the `.blend`.

Opening or selecting an entry sets every setting the entry names. A
setting the entry does not name keeps its current value. If the entry
holds invisible colliders, they replace the scene's colliders; an entry
saved from a scene with no colliders leaves the current ones in place.
[Invisible Colliders](../constraints/colliders.md#saving-with-a-scene-profile)
lists which collider properties are stored.

A scene profile does not hold object groups, pins or merge pairs. Group
settings have [material profiles](material.md#material-profiles), pin
operations have their own profiles, and merge pairs are stored in the
`.blend` only.

An entry looks like this. The file is for reading and comparing; change
the scene in the panel and click **Save Scene Profile** instead of editing
it. Decimal values are written at the float32 precision Blender stores, so
a real file shows longer numbers than this excerpt, for example
`-9.800000190734863` for `-9.8`:

```toml
[Default]
step_size = 0.01
min_newton_steps = 1
air_density = 0.001
air_friction = 0.2
gravity = [0.0, 0.0, -9.8]
frame_count = 180
frame_start = 1
use_scene_frame_start = false
frame_rate = 60
use_scene_fps = false
time_scale = 1.0
inactive_momentum_frames = 0
wind_direction = [0.0, 0.0, 0.0]
wind_strength = 0.0
contact_nnz = 100000000
vertex_air_damp = 0.0
auto_save = false
auto_save_interval = 10
line_search_max_t = 1.25
constraint_ghat = 0.001
cg_max_iter = 10000
cg_tol = 0.001
precond = "BLOCK_JACOBI"
schwarz_levels = "LEVEL_2"
include_face_mass = false
disable_contact = false

[[Default.colliders]]
collider_type = "SPHERE"
name = "Floor Bowl"
position = [0.0, 0.0, 0.0]
contact_gap = 0.001
friction = 0.0
radius = 0.98
hemisphere = true
invert = true

[[Default.colliders.keyframes]]
frame = 1
```

A file might hold entries such as these (an illustration, not a file the
add-on ships):

| Entry         | Highlights                                                                   |
| ------------- | ---------------------------------------------------------------------------- |
| `Default`     | The add-on defaults: step 0.01 s, 180 frames at 60 FPS, gravity (0, 0, -9.8). |
| `Windy`       | Same as `Default`, plus `wind_direction = [0, 1, 0]`, `wind_strength = 5.0`. |
| `HighRes`     | step 0.001 s (the minimum), 360 frames, `cg_max_iter = 20000`.               |
| `SlowMotion`  | 600 frames at 120 FPS.                                                       |
| `ZeroGravity` | `gravity = [0, 0, 0]`, 300 frames.                                           |

## Basic

| UI label                           | Python property             | Default       | Description                                                                                                                     |
| ---------------------------------- | --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Take FPS from Scene**            | `use_scene_fps`             | `False`       | Run at Blender's own frame rate instead of the **FPS** field. An **FPS (from Scene)** readout replaces the field while it is on. |
| **FPS**                            | `frame_rate`                | 60            | Frame rate the simulation runs at: how much simulated time one frame covers. Minimum 1, soft max 240.                           |
| **Time Scale**                     | `time_scale`                | 1.0           | Playback speed of the Blender animation in simulated time. `0.5` plays it at half speed. Range 0.01 – 10. Solve cost grows as 1 / **Time Scale**. |
| **Take Starting Frame from Scene** | `use_scene_frame_start`     | `False`       | Start at the scene's start frame instead of the **Starting Frame** field. A **Starting Frame (from Scene)** readout replaces the field while it is on. |
| **Starting Frame**                 | `frame_start`               | 1             | Blender frame that holds the first output frame; simulated time zero. Minimum 0.                                                |
| **Frame Count**                    | `frame_count`               | 180           | Simulation length in frames. Minimum 10.                                                                                        |
| **Step Size**                      | `step_size`                 | 0.01          | Solver sub-step Δt, in seconds. Range 0.001 – 0.01.                                                                             |
| **Min Newton Steps**               | `min_newton_steps`          | 1             | Minimum Newton iterations per step. Range 1 – 64.                                                                               |
| **Air Density (kg/m³)**            | `air_density`               | 0.001         | Air density. Range 0 – 0.01.                                                                                                    |
| **Air Friction**                   | `air_friction`              | 0.2           | Ratio of sideways to head-on air drag. Range 0 – 1. See [Air Friction](#air-friction).                                          |
| **World Scaling**                  | `world_scaling`             | 1.0           | Physical size the scene is simulated at, as a uniform scale; the result is scaled back, so the scene keeps its authored size. Range 0.001 – 1000. See [World Scaling](#world-scaling). |
| **Gravity (m/s²)**                 | `gravity_3d` (or `gravity`) | (0, 0, -9.8)  | Gravity vector in Blender's Z-up frame.                                                                                         |
| **Preview Direction** (gravity)    | `preview_gravity_direction` | `False`       | Draw the gravity arrow in the viewport. Viewport only. See [Previewing Direction](#previewing-direction).                       |
| **Inactive Momentum Frames**       | `inactive_momentum_frames`  | 0             | Number of frames at the start of the solve without velocity, gravity, wind or force fields. Range 0 – 600.                      |

**Take FPS from Scene** reads the whole-number **FPS** of the scene's
**Frame Rate** (Output properties, Format) and ignores its **Base** factor.
A 23.98 fps scene, which is 24 with a **Base** of 1.001, runs at 24.

When **Time Scale** is not `1.0`, the panel adds an **Animation at N%
speed** label under it.

Changing **Frame Count**, **Starting Frame** or **Take Starting Frame from
Scene** raises the scene's **End** frame to the last frame of the solve (the
starting frame plus **Frame Count** minus 1) when the **End** frame is
earlier, so the timeline covers every simulated frame. It never lowers the
**End** frame and never moves the scene's **Start** frame, so shorten the
timeline yourself when you shorten the run. Setting `frame_count`,
`frame_start` or `use_scene_frame_start` from Python or over
[MCP](../../integrations/mcp.md) does the same.

**Inactive Momentum Frames** is available only while the scene contains at
least one **Shell** group; otherwise the row is grayed out and the value
is not sent. During those frames velocity, gravity, wind and force fields
do not move the simulated objects, which hold their positions, and pins
act as usual. The count starts at the solve's first frame, so the setting
takes no keyframe. A keyframe already on this setting stops **Transfer**
with a message naming it; delete that keyframe.

### Air Friction

**Air Friction** sets how much air resists a surface sliding sideways
compared with a surface pushing straight into the air.

- At `0`, air resists only motion into or out of the surface, so a
  surface slides edgewise through still air unopposed.
- At `1`, sideways resistance equals the resistance to pushing into the
  air.
- The default is `0.2`.

The drag is scaled by each vertex's area and by **Air Density**. At
**Air Density** `0` there is no air drag at all, whatever **Air Friction**
is. The scene **Wind** also acts through this drag, so wind has no effect
at **Air Density** `0`.

**Vertex Air Damping** (in **Advanced Params**) is a separate damper on
every vertex, with no area or **Air Density** weighting and no sideways
split. **Air Friction** does not act on a vertex that belongs to no face,
such as a rod's; **Vertex Air Damping** does.

### World Scaling

**World Scaling** changes the physical size the scene is simulated at.
The solver multiplies the scene by this factor before the solve and
divides the simulated result by it after the solve, so the scene keeps the
size you authored it at. Use it when a scene is modeled at a size it
does not represent: at `0.1` a 15 m mesh is simulated as a 1.5 m one
under the same gravity, and the result comes back at 15 m.

Every length you author in scene units scales with the geometry:
positions (the meshes, the invisible walls and spheres with their radii
and thicknesses, and pin operations' moves and centers), velocities
(initial ones and velocity keyframes), each group's **Contact Gap** and
**Contact Offset** whether absolute or a ratio of the bounding box, the
**Contact Gap** of every invisible wall and sphere, and the scene-level
**Constraint Gap**. Keyframed contact distances scale the same way as
their static values. So a contact that just touches at `1.0` just touches
at any other setting.

Physical constants are left alone, because they describe the world the
resized scene is simulated in: **Gravity**, **Wind**, the material
parameters (**Air Density** included), and a pin's **Torque**. Settings
with no length in them, such as an angular velocity, **Friction**, or a
time, do not change either. Holding gravity fixed while the geometry
changes size is what makes the scene behave as the size it represents.

A **Rod** group's **Bend Stiffness** is measured against a 1 cm reference
segment in solver units, that is, after **World Scaling** has been
applied. At the default `1.0`, solver units equal scene units, so the
reference is a 1 cm segment as drawn in Blender. See
[Material Parameters](material.md).

## Save and Checkpoints

A long simulation can save its state along the way, so you can resume it
instead of starting over. These settings control when those saves happen
and how many are kept.

1. Click the triangle on the **Save and Checkpoints** header.
2. Tick **Save State on Finish** to save the state on the final frame.
3. For saves at a regular interval, tick **Auto Save** in the **Auto
   Save** row. Click that row's own triangle to set **Auto Save Interval**
   and **Keep Saved States**. Both fields are grayed out until **Auto
   Save** is on.
4. For saves at chosen frames, click the triangle of the **Save
   Checkpoints** row. Move the timeline's playhead to a frame and click
   **Add Frame N**, where `N` is the playhead's frame. To delete a frame,
   select its row and click **Remove**.

| UI label                 | Python property          | Default | Description                                                                                                            |
| ------------------------ | ------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Save State on Finish** | `save_state_on_finish`   | `False` | Save the state on the final frame before the solver exits, so the finished result stays resumable even when **Auto Save** is off. |
| **Auto Save**            | `auto_save`              | `False` | Save the state periodically while the simulation runs.                                                                 |
| **Auto Save Interval**   | `auto_save_interval`     | 10      | How often to auto-save, in frames. Minimum 1.                                                                          |
| **Keep Saved States**    | `keep_states`            | 0       | How many auto-saved states to keep. `0` keeps all of them.                                                             |
| **Save Checkpoints**     | `save_checkpoint_frames` | (empty) | A list of frames at which to save a resumable state.                                                                   |

**Auto Save** writes a state every **Auto Save Interval** frames. At the
default **Keep Saved States** of `0` every state is kept, so you can resume
from any of them. A positive number keeps only that many of the most
recent states and discards the oldest as new ones are written.

**Save Checkpoints** saves specific frames, in addition to or instead of
the **Auto Save** schedule. Its frames are Blender frames. **Add Frame**
refuses a frame that is already in the list, and stores a frame below 1 as
1. Each frame has to come after the **Starting Frame**, where nothing has
been simulated yet: **Transfer** refuses a checkpoint at or before it,
naming the frame.

**Resume** lists the saved states the server holds, so auto-saved frames,
the on-finish save and your checkpoints all appear there. See
[Resuming a Run](../sim/simulating.md#resuming-a-run).

## Wind

1. Click the triangle on the **Wind** header.
2. Set **Direction** (an XYZ vector) and **Strength (m/s)**.
3. Optional: tick **Preview Direction** to draw the wind arrow in the
   viewport. See [Previewing Direction](#previewing-direction).

| UI label              | Python property          | Default   | Description                                                          |
| --------------------- | ------------------------ | --------- | -------------------------------------------------------------------- |
| **Direction**         | `wind_direction`         | (0, 0, 0) | Wind direction. Only its direction matters; the add-on normalizes it. |
| **Preview Direction** | `preview_wind_direction` | `False`   | Draw the wind arrow in the viewport. Viewport only.                  |
| **Strength (m/s)**    | `wind_strength`          | 0.0       | Wind speed in m/s. Range 0 – 1000.                                   |

The wind is **Direction** normalized and multiplied by **Strength**. A
**Strength** of `0` means no wind whatever the direction. A **Direction**
of `(0, 0, 0)` with a nonzero **Strength** names no direction to blow in,
and **Transfer** refuses it; the same holds for every keyframe of the
wind.

The solver varies the scene wind between 75% and 100% of **Strength**, in
gusts with a period of about 0.21 s, so **Strength** is the peak speed.
The wind does not blow during the **Inactive Momentum Frames**. It acts
through air drag on faces (see [Air Friction](#air-friction)), so it needs
**Air Density** above zero. Below an **Air Friction** of `1`, a surface
facing the flow receives more of it than a surface edge-on. A Wind force
field adds its own air flow to the
scene wind without the gust variation; see [Force Fields](force_fields.md).

## Previewing Direction

Gravity and wind each have a **Preview Direction** checkbox next to their
fields. Tick it to draw an arrow, and a translucent sphere for scale, in
the 3D viewport. Use it to check a direction before you run a simulation.

The gravity arrow is blue and the wind arrow is green. A label above the
sphere reads the strength and the normalized direction, for example
`Gravity: 9.8 m/s² (0.0, 0.0, -1.0)` or `Wind: 5.0 m/s (0.0, 1.0, 0.0)`.
When both toggles are on, each arrow gets its own sphere, placed side by
side.

- An arrow's length follows the strength, up to a strength of 20.
- The sphere is sized to a fixed fraction of the current view distance.
- The previews follow the timeline's current frame, so a keyframed gravity
  or wind shows its value at the playhead.
- Nothing is drawn for a zero gravity vector, or for a wind with a zero
  **Direction** or a zero **Strength**.
- **Hide all directional arrows** in the **Visualization** panel hides
  these arrows, along with the force-field and velocity arrows. See
  [Visualization](../../getting_started/tour.md#visualization).

The toggles are viewport overlays only. They do not affect the
simulation. They are also available from Python as
`solver.param.preview_gravity_direction` and
`solver.param.preview_wind_direction` (see
[Blender Python API](#blender-python-api)).

## Linear System Solver

The **Linear System Solver** box holds the settings of the PCG linear
solve.

| UI label               | Python property  | Default        | Description                                                                                                    |
| ---------------------- | ---------------- | -------------- | -------------------------------------------------------------------------------------------------------------- |
| **PCG Max Iterations** | `cg_max_iter`    | 10 000         | Maximum number of PCG iterations per solve. Range 100 – 100 000.                                               |
| **PCG Tolerance**      | `cg_tol`         | 0.001          | Relative tolerance at which PCG stops. Range 0.00001 – 0.1.                                                    |
| **Preconditioner**     | `precond`        | `BLOCK_JACOBI` | Preconditioner of the PCG solve. Choices: **Block Jacobi** (`BLOCK_JACOBI`, default), **Schwarz** (`SCHWARZ`). See [Preconditioner](#preconditioner). |
| **Schwarz Levels**     | `schwarz_levels` | `LEVEL_2`      | Levels of the Schwarz preconditioner. Drawn only while **Preconditioner** is **Schwarz**. Choices: **Level 1** (`LEVEL_1`), **Level 2** (`LEVEL_2`, default). See [Schwarz Levels](#schwarz-levels). |

### Preconditioner

The **Preconditioner** selects how the PCG solve is accelerated. Both
choices reach the same **PCG Tolerance**. They differ in the number of
iterations per solve, the time per iteration and the memory used.

- **Block Jacobi** (the default) treats each vertex on its own. Each
  iteration is cheap, so it is fast in practice. It stores one 3 x 3
  block per vertex, so scenes with very heavy contact are less likely to
  run out of memory with it.
- **Schwarz** groups nearby vertices and solves each group together. On
  scenes that mix stiff and soft materials it can converge in fewer
  iterations, but each iteration costs more and it uses more memory, which
  on scenes with very heavy contact can be enough to run out of memory.

If you are unsure, leave it on **Block Jacobi**. Switch to **Schwarz**
only if a scene that mixes stiff and soft materials is slow to converge
and you have memory to spare.

### Schwarz Levels

**Schwarz Levels** picks how many additive levels the **Schwarz**
preconditioner uses. It has no effect on **Block Jacobi**, and the row is
drawn only while **Schwarz** is selected.

- **Level 1** uses a single level.
- **Level 2** (the default) adds a second, coarser level, which reduces
  the worst-case PCG iteration count on stiff multibody contact.

## Advanced Params

| UI label               | Python property     | Default     | Description                                                                                             |
| ---------------------- | ------------------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| **Max Contact**        | `contact_nnz`       | 100 000 000 | Stored with the scene and sent to the solver, which sizes the contact matrix itself and does not read it, so changing it has no effect. Minimum 10 000 000. |
| **Vertex Air Damping** | `vertex_air_damp`   | 0.0         | Per-vertex isotropic air damping. Range 0 – 1.                                                          |
| **Line Search Max T**  | `line_search_max_t` | 1.25        | Factor that extends the time of impact found by continuous collision detection, to avoid possible solver divergence. Range 0.1 – 10. |
| **Constraint Gap**     | `constraint_ghat`   | 0.001       | Gap distance at which boundary-condition barriers activate, in scene units; scales with **World Scaling**. Range 0.0001 – 0.1. |
| **Include Face Mass**  | `include_face_mass` | `False`     | Include the mass of the surface (shell) elements of **Solid** groups.                                   |
| **Friction Mode**      | `friction_mode`     | `MIN`       | How to combine the friction coefficients of two contacting objects. Choices: **Minimum** (`MIN`, default), **Maximum** (`MAX`), **Mean** (`MEAN`). See [Friction at a Contact](material.md#friction-at-a-contact). |
| **Disable Contact**    | `disable_contact`   | `False`     | Turn off all contact detection in the simulation.                                                       |

```{figure} ../../images/scene_params/advanced_expanded.png
:alt: Scene Configuration panel with the Advanced Params box expanded, showing Max Contact, Vertex Air Damping, Line Search Max T, Constraint Gap, Include Face Mass, Friction Mode and Disable Contact
:width: 500px

The **Advanced Params** box. **Linear System Solver** and **Save and
Checkpoints** are separate boxes.
```

## Fix XZ Above Height

**Fix XZ Above Height** (`fix_xz`) has no row in the panel, and it is not
in the **Advanced Params** box. Set it from Python with
`solver.param.fix_xz` or over MCP with `set_scene_parameters`. It is a
height along Blender's Z axis, in scene units. Above that height the solver
holds a vertex still horizontally (in X and Y), and the hold ramps in over
the first meter above the height, measured after **World Scaling**.
Vertices held by a fixed pin follow the pin and are not affected. Use it
for cloth or rods that hang from above without a pin.

The default `0` turns it off. A negative value is refused over MCP and
clamps to `0` from Python. The height scales with **World Scaling**, the
setting offers no keyframe, and `get_scene_parameters` reports it.

## Update Params on Remote

**Update Params on Remote** on the **Solver** panel re-sends the parameters
without the geometry, then rebuilds. The mesh stays on the solver host, so
it is faster than a full **Transfer**. Use it after you change scene
settings, material parameters, keyframed settings, force fields or
invisible colliders. A topology change needs a full **Transfer**. See
[Update Params vs Transfer](../sim/simulating.md#update-params-vs-transfer)
for every other kind of edit and the button it needs.

The button is available while the add-on is connected to a running server
and no simulation is running. It needs scene data already on the server,
so run **Transfer** once first.

## Blender Python API

The settings are reachable through `solver.param`. An assignment takes
effect at once in the panel, but the solver uses the new values only after
`update_params()` (or a new **Transfer**).

The import below works for an add-on in the **User Default** repository. If
you installed the add-on from a remote repository, import `solver` as shown
in [Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

solver.param.gravity = (0, 0, -9.8)
solver.param.frame_count = 180
solver.param.step_size = 0.001

# Ship the new values to the remote without re-sending geometry.
solver.update_params()
```

Over [MCP](../../integrations/mcp.md), `set_scene_parameters` and
`get_scene_parameters` read and write the settings above and `fix_xz` (see
[Fix XZ Above Height](#fix-xz-above-height)), except the **Preview
Direction** toggles and the **Save Checkpoints** list. They also carry
**Post Snap Exactly** (see
[Post Snap Exactly](../constraints/snap_merge.md#post-snap-exactly)) and the
**Project Name** of the **Backend Communicator**. The **Save Checkpoints**
list has its own tools, `set_save_checkpoint_frames`,
`list_save_checkpoint_frames` and `clear_save_checkpoint_frames`;
`solver.param` cannot set it. See [Dynamic Parameters](dynamic.md) for
keyframing the other settings from Python.

Over MCP, `schwarz_levels` is the number `1` or `2`, where
`solver.param.schwarz_levels` and a scene profile use `LEVEL_1` and
`LEVEL_2`. `get_scene_parameters` also reports `effective_fps`, the frame
rate before **Time Scale** is applied, and `effective_start_frame`, the
starting frame, each after **Take FPS from Scene** and **Take Starting Frame
from Scene** have been applied. See **Under the hood** below, **FPS and
Starting Frame from the Blender scene**.

:::{admonition} Under the hood
:class: toggle

**FPS and Starting Frame from the Blender scene**

The **Take FPS from Scene** checkbox (`use_scene_fps`) runs the simulation
at Blender's own frame rate instead of the **FPS** field. While it is on,
the panel replaces the field with an **FPS (from Scene)** readout, because
`frame_rate` retains the value it last held: the frame rate in use is the
one on that readout, not the one in the field, and **Time Scale** is
applied on top of it. Over MCP the same number comes back as
`effective_fps`. **Take Starting Frame from Scene**
(`use_scene_frame_start`) behaves the same way for **Starting Frame**, which
comes back over MCP as `effective_start_frame`.

**Air drag energy**

Let x⁰ be a vertex's position at the start of the step, x¹ its position
being solved for, Δt the step size and w the air velocity (the scene wind
plus any Wind force fields). The vertex velocity relative to the air is
v = (x¹ − x⁰) / Δt − w. Split v along the vertex normal into a normal part
v_n and a sideways part v_t. The drag energy of the vertex is

$$
E_\text{air} = \tfrac{1}{2}\,\bigl(v_n^{\,2} + \texttt{air\_friction}\,\lVert v_t \rVert^{2}\bigr)
$$

scaled by the vertex's area and by **Air Density**.

**Scene profile shape**

A scene profile entry has a flat block of the scene-setting keys at the
top, followed by one `[[<Entry>.colliders]]` table per invisible collider.
An entry can also carry `[[<Entry>.dyn_params]]` tables describing
keyframed parameters. Saving writes them only for a scene that holds
entries in the add-on's keyframe-list format (see
[Dynamic Parameters](dynamic.md)), and applying an entry that contains
them replaces the scene's entries. Keyframes drawn on a slider are never
written to a profile.

**Solver payload keys**

At transfer time the scene settings are sent under solver-side names:

```
dt, min-newton-steps, air-density, air-friction, gravity[3], wind[3],
frames, fps, csrmat-max-nnz, constraint-ghat, cg-max-iter, cg-tol,
include-face-mass, disable-contact, inactive-momentum,
line-search-max-t, auto-save, ...
```

`frames` is **Frame Count** minus one, and `fps` is the **FPS** the solver
runs at multiplied by **Time Scale**. **Update Params on Remote** re-sends
this payload without re-encoding the geometry.
:::
