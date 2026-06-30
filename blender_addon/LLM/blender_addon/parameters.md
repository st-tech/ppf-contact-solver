# Parameters

Everything that tunes how the simulation behaves, from per-group material properties to scene-wide solver settings and keyframed dynamic overrides.

## Overview

Three chapters cover the parameter surface: Material (per-group material properties), Scene (scene-wide solver settings), and Dynamic (keyframed overrides of a handful of scene parameters).

## Scene parameters

Scene-wide simulation settings (gravity, time step, wind, air density, frame count, CG tolerances) all live in one place: the **Scene Configuration** panel in the 3D viewport's sidebar.

### Scene Configuration panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on's tab. The **Scene Configuration** panel is the first panel; everything below is scoped to the whole scene rather than a single object group.

Default layout of the Scene Configuration panel. Top-down: the Open Profile row (no profile loaded yet), the basic parameters block, and the four collapsible sub-section headers. Grayed rows (like `Inactive Momentum Frames`) are gated on scene contents.

The top row is the **scene profile** row. If no profile is loaded you see an **Open Profile** button plus a small save icon; once a profile is open the row collapses into a dropdown (the profile selector) with four icon buttons on the right: **Open**, **Clear**, **Reload**, and **Save** (identical layout to the per-group material profile row). Opening a profile reads its TOML payload into every scene parameter, replaces dynamic parameters, and replaces invisible colliders. Clear drops the association without touching current values. Reload re-reads the file from disk. Save writes the current scene parameters, dynamic-parameter block, and invisible-collider block back out to the file.

IMPORTANT: The scene-profile TOML file is authored through the Save icon, not by hand. Configure the Scene Configuration panel (basic parameters, Wind, Advanced, Dynamic Parameters, Invisible Colliders) the way you want it, then click the **Save** icon on the profile row: the add-on creates a new `.toml` on the first save and overwrites the currently selected entry on later saves. The TOML layout shown further down is for inspection only; the intended workflow is always UI → Save.

The Save icon (floppy disk) at the top-right of the Scene Configuration profile row. Clicking it writes the entire scene block (basic parameters, wind, advanced, dynamic parameters, invisible colliders) out to the `.toml` file.

With a scene profile loaded, the top row collapses into a Profile dropdown with four icons on the right: Open, Clear, Reload, Save. Switching entries in the dropdown applies that preset to the scene (overwriting scene parameters, dynamic parameters, and invisible colliders).

Below the profile row, the panel lays out the basic parameters one per line: **FPS** (inside a small box, with a checkbox to drive it from Blender's render FPS instead), **Frame Count**, **Step Size**, **Min Newton Steps**, **Air Density**, **Air Friction**, **World Scaling**, **Gravity** (a 3-component vector), a **Preview Direction** toggle (viewport arrow), and an **Inactive Momentum Frames** row that is grayed-out unless the scene contains at least one **Shell** group.

Below the basics, four collapsible sub-sections hang off the panel. Each has a disclosure triangle on its header row and toggles open/closed independently:

- **Save and Checkpoints**: Save State on Finish, the Auto Save sub-box (interval + retention), and the per-frame Save Checkpoints list.

With the Save and Checkpoints disclosure triangle open, the sub-section gathers everything that controls when the solver writes a resumable state. See Save and Checkpoints below.

- **Wind**: wind direction vector, preview toggle, wind strength.

With the Wind disclosure triangle open, the sub-section reveals a Direction XYZ field, a Preview Direction viewport toggle, and a Strength (m/s) scalar. Encoding combines direction x strength, so a zero-direction vector disables wind regardless of the strength value.

- **Linear System Solver**: CG max iter, CG tol, preconditioner, and Schwarz levels.
- **Advanced Params**: contact NNZ, vertex air damp, CCD line-search max t, constraint ghat, friction mode, include-face-mass and disable-contact toggles.

Advanced Params exposes the tuning knobs most users never need to touch: contact-matrix capacity (Max Contact), per-vertex air drag, CCD line-search bounds, the friction-combination mode, and two debugging toggles. Raise Max Contact only when the solver reports overflow.

- **Dynamic Parameters**: keyframed gravity / wind / air density / air friction / vertex air damp. Covered separately in Dynamic Parameters.
- **Invisible Colliders**: walls and spheres with their own keyframe lists. Covered separately in Invisible Colliders.

Only the sub-section headers are visible when collapsed; click the triangle on any of them to expand.

### Basic

| UI label                     | Python / TOML key          | Default       | Description                                                         |
| ---------------------------- | -------------------------- | ------------- | ------------------------------------------------------------------- |
| **Frame Count**              | `frame_count`              | 180           | Simulation length in frames. Minimum 10.                            |
| **FPS**                      | `frame_rate`               | 60            | FPS used to convert frames to seconds at encode time. Minimum 24.   |
| **Step Size**                | `step_size`                | 0.01          | Solver sub-step Δt, in seconds. Range 0.001 – 0.01.                 |
| **Min Newton Steps**         | `min_newton_steps`         | 1             | Minimum Newton iterations per step. 1 – 64.                         |
| **Air Density (kg/m³)**      | `air_density`              | 0.001         | Air density, kg/m³. Range 0 – 0.01.                                 |
| **Air Friction**             | `air_friction`             | 0.2           | Tangential-to-normal air drag ratio, 0 – 1 (see below).             |
| **World Scaling**            | `world_scaling`            | 1.0           | Uniform scale applied to all geometry before simulating; results are scaled back. Use it to simulate an over/under-sized scene at a sensible scale (see below). > 0.    |
| **Gravity (m/s²)**           | `gravity_3d`               | (0, 0, -9.8)  | Gravity vector, m/s² (Z-up Blender frame).                          |
| **Preview Direction** (gravity) | `preview_gravity_direction` | `False`    | Draw the gravity-direction arrow overlay in the viewport. Overlay-only. |
| **Inactive Momentum Frames** | `inactive_momentum_frames` | 0             | Frames over which shell momentum is ignored at startup (0 – 600).   |

**Inactive Momentum Frames** is only honored when the scene contains at least one **Shell** group; otherwise the UI disables the row.

#### Air Friction

At every shell vertex, the solver accumulates an air-damping force that resists motion relative to the wind. The move `(x¹ - x⁰) - Δt·w` (current minus previous position, minus the wind displacement over one step) is split by the vertex normal into a **normal** component and a **tangential** component, and the two are weighted differently in the damping energy:

$$
E_\text{air} = \tfrac{1}{2}\,\bigl(v_n^{\,2} + \texttt{air\_friction}\,\lVert v_t \rVert^{2}\bigr)
$$

- **Normal** drag is always on with coefficient `1`. Pushing a shell face into or out of the air is always resisted.
- **Tangential** drag is scaled by **Air Friction**. At `0`, air offers no sideways resistance, so a shell can slide edgewise through still air unopposed. At `1`, tangential and normal drag are equal in magnitude. The default `0.2` gives air a little "grip" on the fabric without over-damping swings.

#### World Scaling

**World Scaling** multiplies all input geometry by a single factor before the simulation runs, then divides the results back by the same factor so your scene keeps its authored size. It lets you simulate a scene that was modeled at the wrong physical scale without re-scaling it in Blender: a 15 m mesh set to `0.1` is simulated at 1.5 m (a sensible physical size for contact and gravity) and the playback is written back at 15 m. The default `1.0` disables it.

What scales: every vertex position (deformable meshes, their rest shapes, pinned and animated targets), collider positions and sizes (sphere radius, wall/sphere thickness), initial and keyframed velocities, both **relative** contact gaps (a fraction of the mesh size) and **absolute** contact gaps/offsets (a fixed distance, scaled so it stays the same size relative to the geometry), and the `fix_xz` threshold. What does **not** scale: gravity and material stiffness. The factor must be greater than 0.

**PDRD** rigid bodies scale correctly as well: their rest centroid, volume, inertia, mass, and joint pivot are each rescaled by the matching power of the factor, so a rigid body keeps its shape and dynamics at any World Scaling.

The per-vertex air-damping force is then scaled by the vertex's associated area and by **Air Density** before being added to the Newton solve. A side-effect of that last multiplication: at `air_density = 0` there is no shell air damping at all, regardless of **Air Friction**.

**Vertex Air Damping** (under **Advanced**) is a separate, isotropic per-vertex damper applied to every vertex in the scene, with no area or air-density weighting and no directional split. Use it when you need to calm a rod or a particle-heavy scene that **Air Friction** does not reach.

Both **Air Friction** and **Vertex Air Damping** are keyframeable; see Dynamic Parameters.

### Save and Checkpoints

The first collapsible sub-section (above Wind) collects every control over when the solver writes a resumable state (`state_<N>.bin.gz`). Resuming reads these states, so the Resume-From dialog offers exactly the frames saved here.

| UI label                       | Python / TOML key            | Default | Description                                                                 |
| ------------------------------ | ---------------------------- | ------- | -------------------------------------------------------------------------- |
| **Save and Checkpoints** (disclosure) | `show_save_and_checkpoints` | `False` | Whether the sub-section is expanded.                                       |
| **Save State on Finish**       | `save_state_on_finish`       | `False` | Save the final-frame state before the solver exits, even when auto-save is off. |
| **Auto Save** (disclosure + toggle) | `auto_save`             | `False` | Enable periodic solver-state snapshots on the remote.                       |
| **Auto Save Interval**         | `auto_save_interval`         | 10      | Auto-save interval, in frames. Minimum 1. Disabled until Auto Save is on.   |
| **Keep Saved States**          | `keep_states`                | 0       | How many saved states to retain. 0 keeps all (required to resume from older frames). |
| **Save Checkpoints**           | `save_checkpoint_frames`     | (empty) | Explicit Blender frames at which to save a state, independent of the auto-save cadence. |

The **Save Checkpoints** list is a UIList: scrub the timeline to the frame you want a checkpoint at and click **Add Frame N** (the button reflects the current timeline frame; duplicates are rejected and the list stays sorted); **Remove** drops the selected entry. The encoder converts each Blender 1-based frame to the solver's 0-based index and ships them as the comma-separated `checkpoints` scene key; the solver's `SimArgs.checkpoints` parses it and calls the same `save_state` path the auto-save cadence uses, so a checkpoint frame and an auto-save frame that coincide save only once. **Keep Saved States** retention applies to all saved states, so leave it at 0 (the default) when you need every listed checkpoint to stay resumable.

### Wind

| UI label               | Python / TOML key       | Default   | Description                                     |
| ---------------------- | ----------------------- | --------- | ----------------------------------------------- |
| **Wind** (disclosure)  | `show_wind`             | `False`   | Whether the Wind sub-section is expanded.       |
| **Direction**          | `wind_direction`        | (0, 0, 0) | Wind direction (normalized at encode time).     |
| **Preview Direction**  | `preview_wind_direction`| `False`   | Draw the wind-direction arrow overlay in the viewport. Overlay-only. |
| **Strength (m/s)**     | `wind_strength`         | 0.0       | Wind speed, m/s. Range 0 – 1000.                |

The encoder combines the two into a single `direction × strength` vector before sending. If **Direction** is `(0, 0, 0)`, no wind is applied regardless of **Strength (m/s)**.

A live example of the Wind fields. With Direction = (0, 1, 0) and Strength = 5.0 m/s, the encoder ships direction × strength = (0, 5, 0) to the solver. Turning Preview Direction on draws the green wind-arrow overlay (and a translucent sphere whose radius scales with strength) so you can judge the wind field relative to scene geometry without running a simulation. The viewport label echoes the magnitude and normalized direction live.

### Linear System Solver

| UI label                       | Python / TOML key            | Default     | Description                                                      |
| ------------------------------ | ---------------------------- | ----------- | ---------------------------------------------------------------- |
| **Linear System Solver** (disclosure) | `show_linear_system_solver` | `False` | Whether the Linear System Solver sub-section is expanded.        |
| **PCG Max Iterations**         | `cg_max_iter`                | 10 000      | Max iterations for the PCG linear solver. 100 – 100 000.         |
| **PCG Tolerance**              | `cg_tol`                     | 0.001       | PCG relative tolerance. 0.00001 – 0.1.                           |
| **Preconditioner**             | `precond`                    | `BLOCK_JACOBI` | PCG preconditioner. `BLOCK_JACOBI` (default) is the 3x3 per-vertex diagonal preconditioner; `SCHWARZ` is the opt-in additive aggregate-Schwarz, more robust on systems mixing stiff and soft elements. |
| **Schwarz Levels**             | `schwarz_levels`             | `LEVEL_2`   | Number of additive levels for the Schwarz preconditioner (shown only when the preconditioner is `SCHWARZ`). `LEVEL_1` is the single-level smoother; `LEVEL_2` (default) adds a two-level coarse correction over the connectivity partition, reducing the worst-case PCG iteration count on stiff multibody contact. |

Raise **PCG Max Iterations** (and lower **PCG Tolerance**) for stiff systems that fail to converge.

### Advanced

| UI label                  | Python / TOML key    | Default     | Description                                                      |
| ------------------------- | -------------------- | ----------- | ---------------------------------------------------------------- |
| **Max Contact**           | `contact_nnz`        | 100 000 000 | Capacity of the contact sparse matrix (non-zero entries).        |
| **Vertex Air Damping**    | `vertex_air_damp`    | 0.0         | Per-vertex isotropic air damping, 0 – 1.                         |
| **Line Search Max T**     | `line_search_max_t`  | 1.25        | CCD line-search maximum step factor. Range 0.1 – 10.             |
| **Constraint Gap**        | `constraint_ghat`    | 0.001       | Barrier gap distance. Range 0.0001 – 0.1.                        |
| **Include Face Mass**     | `include_face_mass`  | `False`     | Fold shell face mass into attached solids.                       |
| **Friction Mode**         | `friction_mode`      | `MIN`       | How to combine the two contacting elements' friction coefficients: `MIN` (minimum, default), `MAX` (maximum), or `MEAN` (average). |
| **Disable Contact**       | `disable_contact`    | `False`     | Turn off contact detection entirely (debugging / ablation).      |

Raise **Max Contact** only if the solver reports a contact-matrix overflow; it's an upper bound, not a target.

### Previewing direction

Both **Gravity** and **Wind** have a **Preview Direction** toggle next to their vector fields. Turning it on draws an arrow (and a translucent sphere for scale) in the 3D viewport showing the current direction (and for wind, the magnitude) so you can judge the setup before running a simulation.

With Preview Direction enabled next to the gravity vector, a blue arrow is drawn in the 3D viewport pointing in the gravity direction. The text label echoes the magnitude and normalized direction. Toggle it off to declutter the viewport.

Wind preview with direction `(0, 1, 0)` and strength `5.0 m/s`. The green arrow's length scales with the wind strength, so you can see at a glance how strong the wind field is relative to the scene geometry.

The **Preview Direction** toggles (one for gravity, one for wind) are also exposed to Python, so you can flip them from a script. They are purely viewport overlays; they never affect the simulation.

### What `Update Params` does

On the Solver panel, **Update Params** re-encodes the parameters (scene block, per-group block, pin config, merge pairs, dynamic parameters, invisible colliders) and ships them to the solver, then triggers a rebuild. Geometry is **not** resent; the mesh buffers on the server are preserved. Use this when you want to tweak material parameters, gravity, or dynamic keyframes without paying the full transfer cost.

Topology changes (adding or removing vertices, edges, or faces) still require a full **Transfer**, because the mesh hash changes.

### Scene profiles

A **scene profile** is a named set of scene parameters saved to a TOML file with the **Save** icon. A file can hold any number of presets; for example:

| Preset        | Highlights                                                                  |
| ------------- | --------------------------------------------------------------------------- |
| `Default`     | step 0.001 s, 180 frames at 60 FPS, gravity (0, 0, -9.8).                   |
| `Windy`       | Same as `Default`, plus `wind_direction = (0, 1, 0)`, `wind_strength = 5.0`. |
| `HighRes`     | step 0.001 s (the minimum), 360 frames, `cg_max_iter = 20000`.              |
| `SlowMotion`  | 600 frames at 120 FPS. Step size stays at 0.001 s.                          |
| `ZeroGravity` | `gravity = (0, 0, 0)`, 300 frames.                                          |

NOTE: The `Default` preset above is just a profile name in this example, not the addon's new-scene default. These presets deliberately tighten `step_size` to 0.001 s for the more demanding example setups; a fresh Blender scene still starts at the addon default of 0.01 s (see the Basic table).

Unlike material profiles, scene profiles also capture **dynamic parameters** and **invisible colliders**. Applying the profile clears existing dynamic parameters and invisible colliders first, then rebuilds them from the TOML entries.

The example below is shown so you can read or diff an existing file; it is **produced by clicking Save**, not written by hand. Tweak the scene in the panel and click Save again rather than editing TOML directly.

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

NOTE: Scene profiles do not touch object groups, pins, or merge pairs. Those live on per-group material profiles and on the scene file itself.

### Blender Python API

The same workflow is available from Python. The settings are reachable through `solver.param`. Assignments are immediate; they don't require a transfer, but do require `update_params()` before the solver actually sees the new values on the next run.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

solver.param.gravity = (0, 0, -9.8)
solver.param.frame_count = 180
solver.param.step_size = 0.001

# Ship the new values to the remote without re-sending geometry.
solver.update_params()
```

UNDER THE HOOD:

**Hidden `use_frame_rate_in_output`**

`use_frame_rate_in_output` is a hidden boolean. When true, Blender's render FPS is used in place of `frame_rate` when converting frames to seconds at encode time. Leave it off unless you want the output time base tied to the render FPS.

**Scene profile shape**

A scene profile TOML entry has a flat block of the scene-parameter keys at the top, followed by `[[<Profile>.dyn_params]]` arrays describing keyframed parameters and `[[<Profile>.colliders]]` arrays describing walls and spheres.

**Solver payload keys**

At transfer time the scene parameters are sent under solver-side names:

```
dt, min-newton-steps, air-density, air-friction, gravity[3], wind[3],
frames, fps, csrmat-max-nnz, constraint-ghat, cg-max-iter, cg-tol,
include-face-mass, disable-contact, inactive-momentum,
line-search-max-t, auto-save, stitch-stiffness, ...
```

**Update Params** reships this payload without re-encoding geometry.

## Material parameters

Every object group carries its own copy of the full material-parameter set, but which fields are relevant depends on the group's type:

- **Shell**: density, stiffness (Young's modulus, Poisson ratio, bend), shrink, strain limit, inflate, stitch, Rayleigh damping (deformation + bending), and contact settings.
- **Solid**: density, stiffness, a single shrink factor, deformation Rayleigh damping, and contact settings.
- **Rod**: density, stiffness, bend, strain limit, Rayleigh damping (deformation + bending), and contact settings.
- **PDRD**: an exactly-rigid body type. Only density, friction, contact settings, and an optional hinge joint (no Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, or Rayleigh damping). The surface mesh is not tetrahedralized.
- **SAND**: a granular body of loose grains. Only grain radius, particle mass, friction, and contact settings (no Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, or Rayleigh damping).
- **Static**: only friction and contact settings (static objects have no deformation to tune). See Static Objects for the full treatment of Static groups, including how to animate them.

Rows that don't apply to the current type are hidden in the UI.

The six options in the group-type dropdown on each group's header row. Picking one changes the Material Params box to match: Solid shows density, stiffness, and a single shrink factor; Shell shows the full cloth stack including anisotropic shrink, strain limit, inflate, and stitch; Rod shows density, stiffness, bend, and strain limit; PDRD shows its rigid-body density, an optional hinge joint, plus shared contact rows; SAND shows its grain radius, particle mass, and friction, plus shared contact rows; Static collapses to just Friction and the contact rows.

### The Material Params box

At the bottom of each group card in the **Dynamics Groups** panel is a collapsible **Material Params** box. When you expand it you see a type-specific set of parameter rows: switching the group's type (for example from **Solid** to **Shell**) immediately changes which rows are visible, so the box always reflects the parameters that actually affect the selected type. A **Static** group shows only the **Friction** and **Contact** rows; a **Shell** group shows the full stack of density, stiffness, bending, shrink, strain limit, inflation, and stitch fields; and so on.

The rows you see inside **Material Params**, top to bottom:

1. **Model** (when applicable): dropdown to pick the material model. **Shell** groups can choose Baraff-Witkin, Stable NeoHookean, or ARAP; **Solid** groups pick between Stable NeoHookean and ARAP; **Rod** groups are locked to ARAP; **Static** groups have no model row.
2. **Density**: the material's density in type-appropriate units (kg/m² for **Shell**, kg/m³ for **Solid**, kg/m for **Rod**).
3. **Young's Modulus**: stiffness. See the note below for how the solver interprets it.
4. **Poisson Ratio**: for **Shell** and **Solid** only.
5. **Friction**: Coulomb friction coefficient at contacts.
6. **Bend stiffness** and **Shrink**. **Shell** shows Bend, Shrink X/Y, a **Strain Limit** toggle, an **Inflate** toggle, and a **Stitch Stiffness** field. **Solid** collapses down to a single Shrink slider. **Rod** reuses the **Shell** Bend row.
7. **Deformation Damping** and **Bending Damping**: stiffness-proportional Rayleigh damping (in seconds). **Deformation Damping** shows on **Solid**, **Shell**, and **Rod**; **Bending Damping** shows on **Shell** and **Rod** only (Solid has no bending term). Both default to `0.0` (off). **PDRD** groups are not Rayleigh-damped. Covered in Rayleigh damping below.
8. **Contact Gap**: a toggle picks between an absolute distance and a fraction of the group's bounding-box diagonal; the relevant pair of fields shows up below the toggle. Both modes are multiplied by **World Scaling** before use, so the absolute fields are unitless numbers (not Blender meters).
9. **Collision Active Duration Windows**: optional per-object frame ranges that restrict when contact is active. Off by default for **Solid**, **Shell**, and **Rod** groups; unavailable for **Static**. Covered in Active collision windows.
10. **Plasticity**: optional non-linear permanent deformation. Covered in its own subsection below.
11. **Velocity Overwrite**: optional keyframed velocity targets for one of the assigned objects. Covered separately below.

The Material Params box expanded on a Shell group. The exact row set changes with the group's type: Solid collapses Shrink X/Y into a single Shrink, Rod drops Poisson ratio, and Static hides everything except Friction and the contact rows.

#### Material presets

At the very top of the **Material Params** box (above the profile row) is a **Preset** dropdown that applies a bundled, physically-grounded material in one click. It is shown for **Shell** and **Solid** groups.

- **Filtered by Type.** The dropdown lists only presets that match the group's current Type: a **Shell** group sees fabrics (Silk, Flag, Cotton, Wool, Denim, Leather) and a **Solid** group sees soft deformables (Rubber, Silicone, Foam, Sponge, Jelly). Set the Type first, then pick a matching preset. **PDRD** (near-rigid) groups have no preset library, since a rigid body differs only by density and friction; set those directly on the group.
- **Applying never changes the Type.** A preset only writes the material parameters it is about: density, friction, the model, Young's modulus and its normalization flag, Poisson ratio, and (fabrics only) bend and strain limit. It does not touch the group Type, contact gaps, pins, velocities, or plasticity. After it applies, the dropdown resets to **Select Preset...** so you can pick again.
- **Bundled vs. profile.** A *preset* is read from the add-on's bundled library (`blender_addon/presets/materials.toml`) and ships with the add-on; a *profile* (the row just below) is a TOML file you save and load yourself. Use a preset for a sensible starting point, then tune in the panel and optionally Save a profile.

The shipped numbers follow a per-type normalization convention:

- **Solids** store a **density-normalized Young's modulus** (E / density, with **Density-Normalized on**) and ship at a uniform **density of 100.0 kg/m³** (the fabrics follow the same density-normalized convention but ship at 1.0 kg/m²). The stored modulus is the real Young's modulus divided by the real volumetric density (for example Rubber's ~3 MPa becomes ~2727), so the static deformation is identical to using the real material, but the uniform density avoids the ill-conditioning that a wide real-density spread caused in mixed-material scenes. Density only affects inertia and contact response, not the static shape, so it is a free knob you can raise. Each preset records its real E (Pa) in a comment for reference.
- **Shells (fabrics)** have no clean bulk Young's modulus, so each preset stores a **calibrated, density-normalized membrane stiffness** (with **Density-Normalized on**) plus Poisson ratio, bend, and strain limit. The `bend` is validated against published Cusick drape coefficients by the reproducible harness in `calibration/cusick_drape/`, so the drape behavior is tied to a standard textile experiment rather than chosen by eye. The membrane stiffness (Young's modulus) and Poisson ratio are likewise calibrated against measured fabric tensile data by `calibration/tensile/`, so the fabrics stretch in the right relative order: Silk and Wool stretch more easily, Denim and Leather barely stretch. (Raising a fabric's Young's modulus also stiffens its drape, so `bend` and Young's modulus are calibrated together to keep the drape coefficient on target.) Shell `bend` is **density-normalized** in the solver (like Young's modulus), so the draped shape does not depend on areal density; the fabric presets therefore ship a uniform `shell_density` of **1.0 kg/m²**, a well-conditioned default you can freely change (it affects only inertia and contact response, not the drape). See the note under Bend Stiffness.

Adding or editing a material is a TOML edit in `blender_addon/presets/materials.toml`; no code change is needed.

#### Profile buttons: Open / Clear / Reload / Save

Along the header of the **Material Params** box are four small buttons that operate on a **material profile** (a TOML file listing named parameter presets):

- **Open**: pops a file picker and loads the selected TOML into the profile dropdown for this group. The dropdown then lists every entry in the file; picking one pushes its parameters into the group.
- **Clear**: forgets the loaded file. The dropdown disappears until you open another TOML.
- **Reload**: re-reads the currently loaded TOML from disk and re-applies the active preset.
- **Save**: writes the group's current parameters back into the loaded TOML under a chosen entry name, replacing the existing entry if the name already exists.

Before a profile is loaded, the row collapses to a single **Open Profile** button plus the **Save** icon (save can write a brand-new TOML without an existing one). Once a profile is loaded, the **Profile** dropdown appears and all four icons line up to the right of it.

IMPORTANT: Material-profile TOML files are created by the Save icon, not by hand. Tune the group's material parameters in the panel, click the Save icon, name the entry, and the add-on writes (or overwrites) it in the `.toml` for you. The TOML structure documented below is shown for inspection and sharing only; the supported edit path is always UI → Save.

The per-group Save icon (floppy disk) at the top-right of the Material Params profile row. Click it to write the group's current material-parameter values to a `.toml` file, creating the file on first save and overwriting the currently selected entry afterwards.

Before loading a profile: the row shows a full-width Open Profile button on the left and the save icon on the right. The Copy / Paste clipboard icons sit at the top-right of the Material Params header for moving parameters between groups in the same session.

After loading a profile: the Profile dropdown (e.g. set to `Cotton`) now lists every entry in the loaded TOML; the four icons to its right are Open, Clear, Reload, Save, left to right.

#### Copy / Paste

Next to the profile buttons is a pair of **Copy** and **Paste** buttons. **Copy** snapshots every field in the current group's material parameters to an internal clipboard; **Paste** applies that clipboard to another group. This is the fastest way to reuse a tuned material without writing a TOML file, but the clipboard lives only for the current Blender session.

### Shared parameters

These apply regardless of type.

| UI label                             | Python / TOML key                 | Default | Description                                                              |
| ------------------------------------ | --------------------------------- | ------- | ------------------------------------------------------------------------ |
| **Friction**                         | `friction`                        | 0.5     | Coulomb friction coefficient at contacts (0 – 1).                        |
| **Contact Gap**                      | `contact_gap`                     | 0.001   | Absolute contact gap distance, in Blender units.                         |
| **Contact Offset**                   | `contact_offset`                  | 0.0     | Absolute contact offset, in Blender units.                               |
| **Use Group Bounding Box Diagonal**  | `use_group_bounding_box_diagonal` | `True`  | When true, contact distances are ratios of the group's bbox diagonal.    |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | 0.001   | Contact gap as a fraction of the group's bounding-box diagonal.          |
| **Contact Offset Ratio**             | `contact_offset_rat`              | 0.0     | Contact offset as a fraction of the group's bounding-box diagonal.       |

**Friction at a contact** is asymmetric in the material parameters but symmetric in the solve: each object carries its own **Friction** coefficient, and when two objects come into contact the solver combines the two values using the scene's **Friction Mode** (Advanced Params), which defaults to the **minimum** of the two but can be set to maximum or mean instead. Under the default minimum rule, the lower-friction surface wins: a slippery cloth sliding over a grippy body behaves as if the whole contact were slippery. If you want a particular contact to feel grippy, both sides need to be set high.

See Contact gap: absolute vs ratio below for which pair you should be editing.

### Shell-specific

| UI label                 | Python / TOML key      | Default          | Description                                                    |
| ------------------------ | ---------------------- | ---------------- | -------------------------------------------------------------- |
| **Model**                | `shell_model`          | `BARAFF_WITKIN`  | Material model. One of `BARAFF_WITKIN`, `STABLE_NEOHOOKEAN`, `ARAP`. |
| **Density (kg/m²)**      | `shell_density`        | 1.0              | Areal density, kg/m².                                          |
| **Young's Modulus (Pa/ρ)** | `shell_young_modulus`  | 1000.0         | Young's modulus (see note below). Accepted range 0.01 – 1 G (soft cap 10 M). |
| **Poisson's Ratio**      | `shell_poisson_ratio`  | 0.35             | Poisson ratio, 0 – 0.4999.                                     |
| **Bend Stiffness**       | `bend`                 | 10.0             | Bending stiffness, 0 – 100. Density-normalized (see note below). |
| **Shrink X**             | `shrink_x`             | 1.0              | Anisotropic warp scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Shrink Y**             | `shrink_y`             | 1.0              | Anisotropic weft scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Enable Strain Limit**  | `enable_strain_limit`  | `False`          | Turns on non-physical strain clamp (good for stiff cloth).     |
| **Strain Limit**         | `strain_limit_percent` | 5.0              | Max stretch in percent (5.0 = 5% stretch) when **Enable Strain Limit** is on. |
| **Inflate**              | `enable_inflate`       | `False`          | Turns on per-face pressure along face normals.                 |
| **Pressure (Pa)**        | `inflate_pressure`     | 0.0              | Inflation pressure, Pa. Active only when **Inflate** is on.    |
| **Stitch Stiffness**     | `stitch_stiffness`     | 1.0              | Loose-edge stitch stiffness; a direct factor on the stitch force (no normalization). |

Loose edges (edges not belonging to any face) are automatically treated as stitch constraints, with stiffness set by **Stitch Stiffness**.

NOTE: Shell **Bend Stiffness** is density-normalized, the same way Young's modulus is. The solver scales the shell bending stiffness by the fabric's areal density, so the bent / draped shape is invariant to **Density**: changing a shell's density alone leaves its drape unchanged (it only changes inertia and contact response). This matches the membrane and the rod bend, and it lets a very light fabric be mixed with much denser bodies without the mass-ratio conditioning trouble a tiny areal density would otherwise cause, so density is a free knob you can raise for stability. The drape calibration in `calibration/cusick_drape/` confirms the drape coefficient is constant across a wide density range at fixed `bend`.

Shell **Bend Stiffness** is also **resolution-independent**: the solver uses the convergent Discrete Shells per-hinge stiffness (which scales as edge-length squared over triangle area), so the same `bend` value produces the same drape and bending whether the mesh is coarse or fine. You can re-mesh a cloth at a different density without re-tuning `bend`, and the calibrated preset values hold across resolutions. (An earlier formulation scaled only with edge length, so finer meshes drooped more; this is fixed.)

#### Shrink X / Shrink Y

What it does: anisotropic rest-shape scale. **Shrink X** scales the warp direction and **Shrink Y** the weft; values below 1 shrink the cloth along that axis, values above 1 extend it. They act on the rest shape, so the solver sees the stretched/shrunk target as the relaxed configuration and drives the mesh toward it under the usual stiffness.

When to enable: use to bake in pre-tension (shrink to pull seams taut), to inflate panels slightly, or to recover the target shape after mesh sewing. Leave both at `1.0` when you want the mesh drawn in Blender to be the rest shape.

Example values:
- `(1.0, 1.0)`: default; no anisotropic rescale.
- `(0.95, 0.95)`: ~5% uniform shrink (mild curl / gathers).
- `(0.9, 1.1)`: shrink warp, extend weft (asymmetric tension).

Note: enabling shrink/extend disables **Strain Limit** for the same group. The two systems fight, so the UI warns when both are active.

Shell groups expose Shrink X and Shrink Y on the same row. Each is a scale factor relative to the rest shape; 1.0 leaves the axis alone.

#### Strain Limit

Available on **Shell** and **Rod** groups (not **Solid**).

What it does: non-physical clamp that prevents mesh edges from stretching beyond the strain limit. Helpful for stiff cloth or ropes that look rubbery in a plain spring formulation.

When to enable: cloth that should keep its silhouette (denim, tablecloths, airbags) or ropes that must not visibly stretch. Disable when you want the mesh to deform freely under force, or when **Shrink X** / **Shrink Y** are non-unity on a **Shell** group (the two systems conflict).

Example values:
- **Strain Limit** = 2.5: very stiff (~2.5% stretch).
- **Strain Limit** = 5.0: default; tight but drapes visibly.
- **Strain Limit** = 15.0: loose; bigger ripples.

With Enable Strain Limit on, the Strain Limit field activates. The value is a percentage of stretch beyond rest length (5.0 = 5% stretch), not a force.

#### Inflate

What it does: applies a per-face pressure along each face normal, pushing the mesh outward (or inward with negative values, once you dip below zero via the Python API). Acts uniformly over the surface like a balloon or airbag.

When to enable: inflatables (pillows, airbags, balloons), soft garments that need a puffy silhouette, or any shell that should resist collapse into a flat sheet. Leave off for ordinary cloth; gravity and bending already do the right thing.

Example values:
- **Pressure (Pa)** = 0.0: default; feature is inert even when toggled on.
- **Pressure (Pa)** = 1.0: gentle puff; subtle volume for a pillow.
- **Pressure (Pa)** = 10.0: strong; airbag-style rapid fill.

Enable Inflate exposes the Pressure (Pa) slider. The unit label is Pa but the solver applies it relative to density, like Young's modulus (see the note below), so tune by eye rather than against SI values.

#### Plasticity

What it does: adds permanent deformation on top of the elastic response. When the local stretch exceeds the **Threshold** (a dead zone around zero strain), the rest shape drifts toward the current shape at a rate controlled by **Theta**. A matching **Bend Plasticity** section does the same for the bending energy, with its own theta and angular threshold.

When to enable: materials that remember their deformation, such as crushed foil, wrinkled paper, dented metal sheets, or sagging fabric. Keep off for perfectly elastic cloth.

Example values:
- **Theta** = 0.0: disabled even if the checkbox is on.
- **Theta** = 0.5: default; ~40%/s creep once over threshold.
- **Theta** = 5.0: fast creep (~99%/s); nearly immediate set.
- **Threshold** = 0.02: ignore strains below 2%.

Shell groups expose two plasticity sections: Plasticity (stretch) and Bend Plasticity (hinge/rod-joint rest angle). Each has its own theta rate and threshold; bend plasticity also lets you pick the rest-angle source (flat, initial geometry, or current frame).

#### Reference rest angle (per object)

This applies to both **Shell** and **Rod** groups (rods reuse the same control; see Rod-specific). The **Rest Angle** dropdown (`bend_rest_angle_source`) sets how every object in the group seeds its bending rest angle: `FLAT` (shell hinge flat / rod straight) or `FROM_GEOMETRY` (use the initial pose). Below it, the **From Reference Geometry** checkbox (`bend_rest_from_reference`, group-level, default off) is a master toggle that turns on **per-object** reference rest angles. When it is on, a box appears with a pulldown of the group's objects; the selected object exposes an **Enable Reference Rest Angle** checkbox (`bend_ref_enable`, per object) and an eyedropper to **pick a reference object** for it (plus an X to clear).

A reference object is a topological copy whose vertices were moved (typically by a modifier or geometry nodes). When you pick it, the add-on checks that it is a positions-only copy and rejects anything else with an error (the same check runs again at Transfer):

- **Shell** and **mesh Rod**: the reference is evaluated through its full modifier / geometry-nodes stack; the result must have the same vertex count and identical connectivity (faces for shells, edges for rods) as the source base mesh.
- **Curve Rod**: the reference must be a curve with the same spline structure; it is sampled at the control-point level the same way the source curve rod is, so move the reference's control points (curve modifiers / geometry nodes that do not bake into control points are not sampled, matching how the source curve is shipped).

When a valid reference is enabled, that object's bending rest angles (shell hinge dihedral, or rod interior-vertex bend angle) are computed from the reference's shape, **overriding** the group's Rest Angle source for that one object (so it takes effect even when the group is set to Flat). The viewport line "Reference geometry overrides the group Rest Angle source for this object." confirms the override is active. Objects without a reference keep following the group `bend_rest_angle_source`.

This is a per-object surface like the PDRD hinge and Velocity Overwrite: the master toggle round-trips through material profiles, but the per-object reference (which object, and its picked reference) lives on the assigned object and is set from the UI picker, not from the group material API or MCP.

#### Rayleigh damping

Two per-element, stiffness-proportional (Rayleigh) damping knobs, both expressed in **seconds** and both defaulting to `0.0` (off):

| UI label                 | Python / TOML key       | Default | Applies to              | Description                                                          |
| ------------------------ | ----------------------- | ------- | ----------------------- | -------------------------------------------------------------------- |
| **Deformation Damping**  | `deformation_damping`   | 0.0     | **Solid**, **Shell**, **Rod** | Stiffness-proportional damping for stretch / membrane / solid deformation. |
| **Bending Damping**      | `bending_damping`       | 0.0     | **Shell**, **Rod** only | Stiffness-proportional damping for shell and rod bending; usually smaller than deformation damping. |

What it does: each coefficient `β` (seconds) adds a damping term `(β/Δt)·K` built from the element stiffness `K`, so it dissipates high-frequency motion of that energy. **Deformation Damping** damps the stretch / membrane / solid response and applies to **Solid**, **Shell**, and **Rod**. **Bending Damping** damps the bending response and applies to **Shell** and **Rod** only: a **Solid** (tet) group has no bending term, so the row is hidden there. **PDRD** groups are not Rayleigh-damped (a rigid body carries no per-element stiffness to scale).

When to enable: calming jitter or ringing on stiff cloth, rods, or solids without globally raising air damping. Leave at `0.0` for fully elastic motion.

Bending-CFL caveat: bending damping adds a stiffness term that tightens the explicit step-size stability bound on fine meshes. On a heavily subdivided shell or rod, a large **Bending Damping** value can force a smaller **Step Size** to stay stable; raise it gradually and watch for the solver needing a smaller step.

#### Velocity Overwrite

What it does: the bottom box in the Material Params stack. It stores a per-object list of keyframed velocity vectors. Each entry pins the whole group to a given `(direction, speed)` at a chosen frame, overriding the velocity produced by the simulation. The dropdown on the header row picks which assigned object receives the keyframes; the eye icon toggles a viewport preview arrow; the copy/paste icons move the keyframe list between groups.

When to enable: scripted cloth launches (flag unfurling, parachute drops), matching reference motion on hero shots, or giving the solver a strong initial push that no constant velocity could time. Leave empty for fully passive simulations.

The Velocity Overwrite section with four keyframes populated (frames 1, 30, 60, 90). Each row is `frame (speed m/s [direction])`. The selected row expands into per-keyframe editor rows (Frame, Direction (XYZ), and Speed) so you can tweak one entry without opening an animation editor. The Cloth dropdown at the top picks which assigned object the keyframes belong to, and the `+` / `-` buttons on the right add or remove entries.

Two checkboxes gate which part of the velocity a keyframe overwrites: **Enable Translational Velocity Overwrite** (the Direction / Speed rows) and **Enable Angular Velocity Overwrite** (the Spin rows). Each component is overwritten **only when its box is checked**, so you can do either or both: a pure-spin keyframe (angular only) leaves the translation untouched, and a pure-translation keyframe leaves the spin alone. Translational defaults on, angular defaults off.

Spin (angular) component: on **Solid**, **Shell**, and **PDRD** groups (not **Rod**) the selected keyframe exposes a **Spin** box with a **Spin Axis** dropdown and a signed **Angular Speed** in degrees per second. The dropdown offers three kinds of axis:

- **Principal Axis 1 / 2 / 3** (PC1 = largest extent ... PC3 = thinnest, the PDRD-hinge convention). These are **resolved dynamically from the simulated geometry at the keyframe time**, not frozen at frame 1: a PDRD gear that has already rotated, or a deformed solid/shell, spins about its current principal axis.
- **World X / World Y / World Z**: fixed world-space axes.
- **Custom Axis**: a user-entered world-space vector (a **Custom Axis** field appears below the dropdown; it is normalized before use).

World and Custom axes are fixed directions and do not track the body. When **Enable Angular** is checked, a rigid spin `ω × (x − c)` about the chosen axis is applied; when **Enable Translational** is also checked, the body's velocity is overwritten to `direction · speed` plus that spin. For a PDRD body the field is exactly rigid; on a hinged PDRD body only the component admissible by the hinge axle is kept, so picking a different axis than the hinge spins it only partially. The viewport preview (eye icon) draws a rotation arc along the chosen axis when the playhead is on the keyframe's frame.

### Solid-specific

| UI label                   | Python / TOML key     | Default              | Description                                               |
| -------------------------- | --------------------- | -------------------- | --------------------------------------------------------- |
| **Model**                  | `solid_model`         | `ARAP`               | Material model. Either `STABLE_NEOHOOKEAN` or `ARAP`.     |
| **Density (kg/m³)**        | `solid_density`       | 100.0                | Volumetric density, kg/m³.                                |
| **Young's Modulus (Pa/ρ)** | `solid_young_modulus` | 500.0                | Young's modulus (see note below). Range 0.01 – 1 G (soft cap 10 M). |
| **Poisson's Ratio**        | `solid_poisson_ratio` | 0.35                 | Poisson ratio, 0 – 0.4999.                                |
| **Shrink**                 | `shrink`              | 1.0                  | Uniform rest-shape scale (min 0.1).                       |

#### Shrink

What it does: uniform (isotropic) rest-shape scale for the whole solid. The solver treats the shrunk / expanded shape as the relaxed target and drives the mesh toward it under the usual stiffness, so values below 1 visually contract the body and values above 1 swell it.

When to enable: pre-stressed solids (e.g. a rubber band that should self-tension once the simulation starts), volumetric shrink after tetrahedralization, or recovering a target volume after scale tweaks in Blender. Leave at `1.0` for bodies that should rest exactly at their modeled size.

Example values:
- **Shrink** = 1.0: default; no rescale.
- **Shrink** = 0.9: 10% shrink; body contracts and pulls on its neighbors.
- **Shrink** = 1.05: 5% expansion; useful for "puffy" solids.

Solid groups expose a single Shrink row in the Material Params box (Shell groups instead get anisotropic Shrink X / Shrink Y).

#### Tetrahedralizer (per object)

**Solid** groups only. The bottom of the Material Params box on a **Solid** group has a **Tetrahedralizer** box. Like Velocity Overwrite, it is per object: an **Object** dropdown on the header picks which assigned mesh to edit, then a backend dropdown selects the tetrahedralizer for that mesh, followed by that backend's per-field overrides. The settings live on the assigned object, so each mesh in the group can choose its own backend and overrides.

Two backends are available:

- **fTetWild** (default): a tolerant remesher. It handles open, cracked, or near-duplicate input, but it resamples the surface, so the original Blender vertices are reconstructed through a surface map rather than preserved.
- **TetGen**: preserves the input surface exactly. Every input vertex maps 1-1 to a tet-surface vertex (the add-on asserts this), so the simulated surface has the same vertices and triangles as the Blender mesh. It requires a clean, closed, manifold input; open, coplanar, or non-manifold meshes are rejected (assign those to a SHELL group or use fTetWild).

Each override row has an **Override** checkbox on the left and the value on the right; the value is only forwarded when its checkbox is on. With all overrides off, the chosen backend runs at its own defaults.

fTetWild overrides:

| UI label               | Python / TOML key         | Default   | Description                                                          |
| ---------------------- | ------------------------- | --------- | -------------------------------------------------------------------- |
| **Edge Length Factor** | `ftetwild_edge_length_fac`| 0.05      | Ideal tet edge length as a fraction of the bbox diagonal (`-l`).     |
| **Epsilon**            | `ftetwild_epsilon`        | 0.001     | Envelope size as a fraction of the bbox diagonal (`-e`).             |
| **Stop Energy**        | `ftetwild_stop_energy`    | 10.0      | AMIPS energy threshold; larger = faster, lower quality.              |
| **Max Opt Iterations** | `ftetwild_num_opt_iter`   | 80        | Maximum fTetWild optimization passes.                                |
| **Optimize**           | `ftetwild_optimize`       | `True`    | Improve cell quality (slower).                                       |
| **Simplify Input**     | `ftetwild_simplify`       | `True`    | Simplify the input surface before tetrahedralization.                |
| **Coarsen Output**     | `ftetwild_coarsen`        | `False`   | Coarsen output while preserving quality.                             |

TetGen overrides (the surface is always preserved; these only tune the interior):

| UI label                  | Python / TOML key     | Default | Description                                                                |
| ------------------------- | --------------------- | ------- | -------------------------------------------------------------------------- |
| **Min Radius-Edge Ratio** | `tetgen_min_ratio`    | 2.0     | TetGen quality bound (`-q`); smaller forces rounder cells via more interior Steiner points. |
| **Max Tet Volume**        | `tetgen_max_volume`   | 0.0     | TetGen max tetrahedron volume (`-a`) in object units; caps interior cell size. 0 leaves it uncapped. |

Each value has a matching `<backend>_override_<field>` boolean that gates whether the override is sent. Leave the overrides untouched to get the backend's out-of-box behavior; reach for them only when a solid is meshing too coarsely, missing features, or taking too long to tetrahedralize.

### PDRD-specific

**PDRD** (Painless Differentiable Rotation Dynamics) is an exactly-rigid group type: each body moves as a single best-fit rigid transform (translation + rotation) over its surface mesh, with no per-element elasticity and no tetrahedralization. It exposes only density (plus the shared friction and contact rows); Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, and Rayleigh damping do not apply.

| UI label                   | Python / TOML key  | Default  | Description                                                          |
| -------------------------- | ------------------ | -------- | ------------------------------------------------------------------- |
| **Density (kg/m³)**        | `pdrd_density`      | 100.0    | Volumetric density, kg/m³. Mass is density times the enclosed volume of the surface mesh. Range 0.01 – 10 000. |

The **Hinge** joint is **per object**, not a group material: one PDRD group can hold several bodies (a gear train), each pinned on its own axle. In the PDRD group panel, the "Hinge" box has an object pulldown (like Velocity Overwrite) to focus one assigned body, then edits that body's hinge:

| Per-object control | Python key (on the assigned object) | Default | Description                                                          |
| ------------------ | ----------------------------------- | ------- | ------------------------------------------------------------------- |
| **Hinge**          | `pdrd_hinge_enable`                  | False   | Pin this body and lock its rotation to a single principal axis (a hinge / pin joint). Build gears by hinging each gear to its axle and letting tooth contact transmit the torque. |
| **Axle**           | `pdrd_hinge_axis`                    | `"2"`   | Which principal (PCA) axis of the rest shape is the free hinge axle, like the torque-axis dropdown: `"0"` = largest extent, `"1"` = middle, `"2"` = thinnest (the usual axle for a flat gear or disk). |

From scripting / MCP, set the per-object hinge with `Group.set_hinge(object_name, pca_axis=2)` (the `set_pdrd_hinge` MCP tool), not with the group material API. When a body's **Hinge** is enabled the viewport draws a cyan ring around the chosen axle through that body's centroid (the same gizmo idiom as the static-object torque axis). At simulation time the body's reduced rigid degrees of freedom are filtered so translation is locked and rotation is restricted to that axle; contact still transmits torque between hinged bodies, so meshing gears counter-rotate without any explicit gear-ratio constraint.

### Rod-specific

| UI label                   | Python / TOML key   | Default   | Description                                       |
| -------------------------- | ------------------- | --------- | ------------------------------------------------- |
| **Model**                  | `rod_model`         | `ARAP`    | Material model. `ARAP` is the only option.        |
| **Density (kg/m)**         | `rod_density`       | 1.0       | Line density, kg/m.                               |
| **Young's Modulus (Pa/ρ)** | `rod_young_modulus` | 10000.0   | Young's modulus (see note below).                 |

**Rod** groups expose the same **Bend Stiffness** field as **Shell**; it writes into the single `bend` property on the group, so both types read and serialize it identically. The **Rest Angle** dropdown and the **From Reference Geometry** per-object reference rest angle (see Reference rest angle (per object) under Shell-specific) also apply to rods: a reference sets the rod's interior-vertex rest bend angles. For a mesh rod the reference is a modifier-evaluated mesh copy (matching vertex count + edges); for a curve rod it is a curve with the same spline structure, sampled at the control-point level.

NOTE: Young's modulus behaves non-conventionally. The solver divides the entered Young's modulus by density internally. The practical effect is that animated behavior is invariant to density alone: doubling density without touching Young's modulus produces the same motion (the mass doubles, but the effective stiffness scales with it). This decouples "how heavy the material is" from "how stiff it looks", so you can tune stiffness and mass independently. The bundled material presets (`Cotton`, `Silk`, `Rubber`, ...) are set to physically meaningful values with that normalization in mind (see Material presets above).

#### Density-normalized vs true pascals

| UI label                          | Python / TOML key               | Default | Description                                                              |
| --------------------------------- | ------------------------------- | ------- | ------------------------------------------------------------------------ |
| **Density-Normalized (Pa/ρ)**     | `young_mod_density_normalized`  | `True`  | What the Young's Modulus field above means: density-normalized or true pascals. |

`young_mod_density_normalized` controls how the Young's Modulus field is interpreted, and it relabels that field to match (**Solid**, **Shell**, **Rod**; not relevant to **PDRD** or **Static**).

- `True` (default): the entered value is a **density-normalized** modulus in `Pa/ρ`, the solver's native convention. It is sent unchanged, so changing a body's density alone leaves its motion unchanged. Keep it on to match existing scenes. The field reads `Young's Modulus (Pa/ρ)`.
- `False`: the entered value is a **true Young's modulus in pascals** (for example a value from a material reference table). The add-on divides it by this group's density before sending, so a denser body of the same material is correspondingly stiffer to move. The field label flips to plain `Young's Modulus (Pa)`.

### SAND-specific

**SAND** is a granular group type: a faceless mesh of loose vertices simulated as individual grains. It exposes only the three grain parameters below plus the shared friction and contact rows; Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, and Rayleigh damping do not apply.

| UI label              | Python / TOML key     | Default | Description                                                                 |
| --------------------- | --------------------- | ------- | --------------------------------------------------------------------------- |
| **Grain Radius (m)**  | `sand_grain_radius`   | 0.02    | Per-grain radius, in Blender units. Locked in at convert time, so the field is read-only once the group has been converted. |
| **Particle Mass (g)** | `sand_particle_mass`  | 1.0     | Mass of a single grain, in grams (sent to the solver in kilograms).         |
| **Friction**          | `sand_friction`       | 0.0     | Inter-grain friction coefficient of the granular body.                      |

The grain radius doubles as the contact offset (the grain's physical skin); the contact gap is the extra barrier-activation distance on top.

### Contact gap and contact offset

**Contact Gap** and **Contact Offset** are two distances that together shape the invisible contact layer around each group's geometry. They serve different roles and both are configurable.

- **Contact Gap** is the barrier's reach: the distance at which the solver starts applying a push-back force between two surfaces. A larger gap gives a softer, earlier-engaging barrier and costs more contact pairs; a smaller gap lets surfaces sit closer before the barrier kicks in. This is the setting most scenes need to tune.
- **Contact Offset** is per-group padding added on top of the gap. At each contact check the solver sums the two participants' offsets with the (averaged) gap and treats that total as the effective separation threshold. You can think of it as the group's "skin thickness": it guarantees a minimum clearance regardless of what the other side chose. The default is `0.0` (no extra clearance), which is what most scenes want.

Reach for **Contact Offset** when one group needs a specific thickness for visual or collision reasons independent of what its neighbors do, for example a garment that should never touch the body by less than a millimeter no matter which body group it comes near. For day-to-day tuning of how tightly surfaces sit, leave **Contact Offset** at zero and adjust **Contact Gap** instead.

### Contact gap: absolute vs ratio

Both **Contact Gap** and **Contact Offset** can be specified in either of two ways:

- **Absolute** (the **Contact Gap** and **Contact Offset** fields): a literal distance in Blender units. Good when you want a hard, known thickness, e.g. a 1 mm skin for a body.
- **Ratio** (the **Contact Gap Ratio** and **Contact Offset Ratio** fields): a fraction of the group's bounding-box diagonal, computed at transfer time. Good because it scales with the scene: rescaling a character by 10x doesn't make the cloth penetrate.

Absolute vs ratio contact-gap comparison. In absolute mode the halo is the same thickness on small and large objects; in ratio mode it scales with the object. The dashed red ring shows the contact-gap layer. Absolute mode keeps the layer thickness constant in world units, so it looks huge around a small object and thin around a large one. Ratio mode scales the layer with the object's bounding box, so both look proportionally wrapped regardless of scale.

The **Use Group Bounding Box Diagonal** toggle picks between them. The **default is ratio-of-bbox-diagonal** because that's what most users want; you only need to flip to absolute when the group contains unusually elongated objects (where the diagonal overestimates characteristic size) or when you need an exact contact thickness for matching against another group.

Both pairs (**Contact Gap** / **Contact Gap Ratio** and **Contact Offset** / **Contact Offset Ratio**) are independently controlled by the same toggle.

### Material profiles

A **material profile** is a named set of material parameters saved to a TOML file with the **Save** icon. A single file can hold any number of presets; profiles like these are easy to build:

| Preset   | Type       | Notes                                                               |
| -------- | ---------- | ------------------------------------------------------------------- |
| `Flag`   | **Shell**  | Light, stiff. Young = 100, density = 0.1 kg/m², strain limited.     |
| `Cotton` | **Shell**  | Young = 50, density = 0.5 kg/m², bend = 0.5.                        |
| `Silk`   | **Shell**  | Soft, low-density, bend = 0.2, friction = 0.15.                     |
| `Denim`  | **Shell**  | Heavier, stiffer; full block of shell/solid/rod fields for hybrids. |
| `Rubber` | **Solid**  | Stable NeoHookean, density = 1100 kg/m³, friction = 0.8.            |
| `Steel`  | **Solid**  | Stable NeoHookean, Young = 200 000, density = 7800 kg/m³.           |
| `Rope`   | **Rod**    | Young = 10 000, density = 1.0 kg/m, bend = 1.0.                     |
| `Static` | **Static** | Just a friction value, used for colliders.                          |

NOTE: Material profiles do not carry any object assignments, pin vertex groups, or per-object velocity overrides. They describe a material, not a scene.

#### Example TOML stanza

The block below shows what the add-on writes out when you click Save. It is **not** a template to fill in by hand. Adjust a group's Material Params in the panel and click the **Save** icon to produce (or update) a file like this.

The per-group Save icon (floppy disk) on the Material Params row. Clicking it writes the group's current material-parameter values to a `.toml` file, creating the file on the first save and overwriting the currently selected entry afterwards.

```toml
[Cotton]
object_type = "SHELL"
shell_model = "BARAFF_WITKIN"
shell_density = 0.5
shell_young_modulus = 50.0
shell_poisson_ratio = 0.35
bend = 0.5
friction = 0.3

[Denim]
object_type = "SHELL"
solid_model = "ARAP"
shell_model = "BARAFF_WITKIN"
rod_model = "ARAP"
solid_density = 100.0
shell_density = 0.8
rod_density = 1.0
solid_young_modulus = 500.0
shell_young_modulus = 200.0
rod_young_modulus = 10000.0
solid_poisson_ratio = 0.35
shell_poisson_ratio = 0.35
friction = 0.5
contact_gap = 0.001
contact_offset = 0.0
use_group_bounding_box_diagonal = true
contact_gap_rat = 0.001
contact_offset_rat = 0.0
bend = 2.0
shrink = 1.0
enable_strain_limit = true
strain_limit_percent = 5.0
stitch_stiffness = 1.0
```

Only the keys you include are applied; missing keys keep their current value on the group. You don't have to list every field for a preset to be valid — a `Static` collider preset, for instance, can carry just a `friction` value.

### Blender Python API

The same workflow is available from Python. Every field in the **Material Params** box is reachable through each group's `.param` attribute. Changes from Python appear in the panel immediately and vice versa.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

cloth = solver.create_group("Cloth", "SHELL")
cloth.param.shell_density       = 0.5
cloth.param.shell_young_modulus = 50.0
cloth.param.friction            = 0.3
cloth.param.bend                = 0.5

# Solid body with Stable NeoHookean and a tighter contact skin.
body = solver.create_group("Body", "SOLID")
body.param.solid_density       = 1100.0
body.param.solid_young_modulus = 5000.0
body.param.use_group_bounding_box_diagonal = False
body.param.contact_gap         = 0.001

# Static collider: only friction and contact settings matter.
floor = solver.create_group("Floor", "STATIC")
floor.param.friction = 0.8
```

UNDER THE HOOD:

**Loose-edge stitch encoding**

At transfer time, edges on **Shell** meshes that are not adjacent to any face are automatically emitted as stitch constraints with stiffness set by `stitch_stiffness`. There is no UI surface for this; it happens on every transfer.

Two subdivided square Shell patches joined by vertical loose edges (rendered as red tubes). The patches are separate face regions; the connecting edges belong to no face, so the transfer step emits each one as a stitch constraint with stiffness `stitch_stiffness`.

**Copy / Paste clipboard**

The **Copy** / **Paste** buttons move parameters between groups within a single Blender session. The clipboard is not persisted to the `.blend` file, so restarting Blender clears it.

## Dynamic parameters

Most scene parameters are single scalars. A handful of them can also be *keyframed*: gravity flipping at frame 60, wind turning on at frame 30, air density changing mid-simulation. These are **dynamic parameters** and live in the **Dynamic Parameters** sub-panel under Scene Configuration.

### UI walkthrough

The **Dynamic Parameters** sub-panel is a collapsible section inside **Scene Configuration**. Expanding it reveals:

1. A **UIList** of currently-active dynamic parameters along the left. Each row shows the parameter name (**Gravity**, **Wind**, ...). The selected row drives the editor below.
2. An **Add** dropdown above (or beside) the list, listing the five supported keys. Picking an entry adds a new dynamic parameter of that type and automatically creates its frame-1 initial keyframe. Duplicates are rejected.
3. A **Remove** button next to Add. It deletes the selected dynamic parameter entry and every one of its keyframes.

Once a dynamic parameter is selected in the list, a **per-type editor** appears below:

- **Gravity**: an XYZ vector field for the gravity vector at the selected keyframe.

With Gravity selected in the top UIList and Frame 30 selected in the keyframe list (Hold off), the editor exposes an XYZ vector field for the gravity value at that keyframe.

- **Wind**: an XYZ **Direction** field plus a single **Strength (m/s)** scalar. The direction is normalized internally; the strength is the magnitude.

With Wind selected, the editor shows a Direction XYZ field and a separate Strength (m/s) scalar. The direction is normalized at encode time, so only its orientation matters; magnitude comes from strength.

- **Air density / Air friction / Vertex air damp**: a single scalar field.

The three scalar-valued keys (Air Density shown) collapse the editor to a single Value field. Air Friction and Vertex Air Damp render the same way.

Below the editor is a **second UIList** showing every keyframe for the selected parameter, with a frame number per row. The row at index 0 (the frame-1 initial keyframe) is tagged with an `(Initial)` badge and cannot be removed.

Two buttons sit below this keyframe list:

- **Add Keyframe**: inserts a new keyframe at the *current scene frame*. The new keyframe is initialized from the matching global scalar (so adding a gravity keyframe copies the scene's **Gravity** value into it); the list is then re-sorted by frame. Inserting at a frame that already has a keyframe is rejected.
- **Remove Keyframe**: deletes the selected keyframe. Attempting to remove the initial keyframe is rejected with a warning.

#### The initial keyframe

Every dynamic parameter starts with an **initial keyframe at frame 1**. The initial keyframe does not store its own value; when encoded it reads from the matching global scene parameter (**Gravity**; **Direction** + **Strength (m/s)** for wind; **Air Density**, **Air Friction**, **Vertex Air Damping**). That way the timeline always begins from whatever the scene-wide value is.

A dynamic parameter entry is only sent to the solver if it has **more than one** keyframe. A single initial keyframe is treated the same as "not keyframed": the global scalar is used.

#### Hold: step-function semantics

Keyframes other than the initial one expose a **Hold** checkbox in the per-keyframe editor. When Hold is on, the previous keyframe's value is re-emitted at this frame instead of a new value. Combined with an adjacent non-hold keyframe one frame later, this is how you build step functions: a hard instantaneous change with no linear ramp.

Example: "flip gravity at frame 60."

- **Frame 1** *(initial)*: value read from the scene's **Gravity**, e.g. `(0, 0, -9.8)`.
- **Frame 60**: **Hold** on. The encoder emits `(0, 0, -9.8)` here as well, holding the initial value until exactly frame 60.
- **Frame 61**: **Hold** off, value `(0, 0, 9.8)`. Gravity inverts.

Without the hold at frame 60, the solver would have linearly ramped gravity between frame 1 and frame 61, producing a slow drift instead of a flip.

A concrete gravity-flip setup as it appears in the sub-panel. The Gravity (3 kf) entry is selected in the top UIList; its keyframe list below shows Frame 1 (Initial) (reads the scene's gravity), Frame 30 (Hold, which re-emits that same value), and Frame 31 (steps to the new `(0, 0, 9.8)` value). The `(Initial)` badge on frame 1 marks the one keyframe that cannot be removed. The Hold checkbox on frame 30 is what makes this a step function instead of a linear ramp from frame 1 to frame 31.

### Supported parameters

| UI label             | Python / TOML key   | Enum               | Value shape           |
| -------------------- | ------------------- | ------------------ | --------------------- |
| **Gravity**          | `gravity`           | `GRAVITY`          | XYZ vector            |
| **Wind**             | `wind`              | `WIND`             | XYZ dir + strength    |
| **Air Density**      | `air_density`       | `AIR_DENSITY`      | scalar                |
| **Air Friction**     | `air_friction`     | `AIR_FRICTION`     | scalar                |
| **Vertex Air Damp**  | `vertex_air_damp`   | `VERTEX_AIR_DAMP`  | scalar                |

Only these five parameters are keyframeable. Others (step size, CG tolerance, and so on) have to stay constant across the simulation.

### Blender Python API

The same workflow is available from Python. The `solver.param.dyn(...)` builder drives the same dynamic-parameter list that the sub-panel shows. Every builder method returns `self`, so you chain them freely: `time(f)` moves the cursor to frame `f` (must be strictly increasing), and the next `hold()` or `change(...)` attaches a keyframe at that frame. For wind, `change(direction, strength=...)` encodes both.

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

WARNING: The Python API for dynamic parameters uses frames, not seconds.

```python
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
#                          ^^^^^^^ frame number
```

This is intentional: the Blender UI thinks in frames, and so does this API. The frontend solver API `session.param.dyn()` (the one called from inside the decoder on the solver side) uses **seconds**. You normally never see the seconds form unless you're driving the solver directly from a Python notebook.

UNDER THE HOOD:

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

At transfer time each dynamic parameter becomes a list of `(time_seconds, value, is_hold)` entries under the matching solver key. The observable rules:

- Frames are converted to seconds as `(frame − 1) / fps`.
- Gravity and wind are coordinate-converted from Z-up to Y-up.
- Wind is sent as `direction × strength` with the direction normalized. A zero direction vector produces a zero wind vector regardless of the strength.
- A hold keyframe re-emits the previous keyframe's value; it does not sample the live scene parameter.
- Parameters with fewer than two keyframes are dropped and the solver falls back to the global scalar.
