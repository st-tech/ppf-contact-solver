# 🧪 Material Parameters

Every object group carries its own copy of the full material-parameter set,
but which fields are relevant depends on the group's type:

- **Shell**: density, stiffness (Young's modulus, Poisson ratio, bend),
  shrink, strain limit, inflate, stitch, and contact settings.
- **Solid**: density, stiffness, a single shrink factor, and contact
  settings.
- **Rod**: density, stiffness, bend, shrink, strain limit, and contact
  settings.
- **PDRD**: density, friction, and contact settings only. PDRD is an
  exactly-rigid body type with no Young's modulus, Poisson ratio, bend,
  shrink, strain limit, or inflate.
- **Static**: friction, contact settings, and **Apply Soft Constraints**
  (static objects have no deformation to tune). See
  [Static Objects](../scene/static_objects.md) for the full treatment of Static
  groups, including how to animate them.
- **Sand**: a granular body whose relevant fields are grain radius,
  particle mass, friction, and contact settings.

Rows that don't apply to the current type are hidden in the UI.

```{figure} ../../images/material_params/group_type_popdown.png
:alt: The group-type dropdown menu, popped open. A Type label with a disclosure arrow sits above a vertical list of six buttons (Solid, highlighted as the current selection; Shell; Rod; Static; PDRD; Sand) each spanning the full width of the popup.
:width: 360px

The six options in the group-type dropdown on each group's header row.
Picking one changes the **Material Params** box to match: **Solid**
shows density, stiffness, and a single shrink factor; **Shell** shows
the full cloth stack including anisotropic shrink, strain limit,
inflate, and stitch; **Rod** shows density, stiffness, shrink, bend,
and strain limit; **Static** collapses to **Friction**, **Apply Soft
Constraints**, and the contact rows; and **Sand** shows grain radius,
particle mass, friction,
and the contact rows.
```

## The Material Params Box

At the bottom of each group card in the **Dynamics Groups** panel is a
collapsible **Material Params** box. When you expand it you see a
type-specific set of parameter rows: switching the group's type (for
example from **Solid** to **Shell**) immediately changes which rows are
visible, so the box always reflects the parameters that actually affect
the selected type. A **Static** group shows only the **Friction** and
**Contact** rows; a **Shell** group shows the full stack of density,
stiffness, bending, shrink, strain limit, inflation, and stitch fields;
and so on.

The rows you see inside **Material Params**, top to bottom:

1. **Model** (when applicable): dropdown to pick the material model.
   **Shell** groups can choose Baraff-Witkin or ARAP; **Solid** groups
   pick between Stable NeoHookean and ARAP; **Rod** groups are locked
   to ARAP; **Static** groups have no model row. Older `.blend` files
   that stored Stable NeoHookean on a **Shell** group still load: the
   `shell_model` enum keeps the slot for `.blend` index stability and
   the transfer step coerces that selection to ARAP at encode time.
2. **Density**: the material's density in type-appropriate units (kg/m²
   for **Shell**, kg/m³ for **Solid**, kg/m for **Rod**).
3. **Young's Modulus**: stiffness. See the note below for how the solver
   interprets it.
4. **Poisson Ratio**: for **Shell** and **Solid** only.
5. **Friction**: Coulomb friction coefficient at contacts.
6. **Bend stiffness** and **Shrink**. **Shell** shows Bend, Shrink X/Y,
   a **Strain Limit** toggle, an **Inflate** toggle, and a **Stitch
   Stiffness** field. **Solid** collapses down to a single Shrink
   slider. **Rod** draws its **Shrink** row just under **Friction** and
   its **Bend Stiffness** field in a separate **Bend** box below the
   contact rows.
7. **Contact Gap**: a toggle picks between absolute distance (in Blender
   units) and a fraction of the group's bounding-box diagonal; the
   relevant pair of fields shows up below the toggle.
8. **Collision Active Duration Windows**: optional per-object frame
   ranges that restrict when contact is active. Off by default for
   **Solid**, **Shell**, and **Rod** groups; unavailable for **Static**.
   Covered in
   [Active collision windows](../scene/object_groups.md#active-collision-windows).
9. **Plasticity**: optional non-linear permanent deformation. Covered in
   its own subsection below.
10. **Velocity Overwrite**: optional keyframed velocity targets for one
    of the assigned objects. Covered separately below.

```{figure} ../../images/material_params/box_shell.png
:alt: Material Params box expanded on a Shell group, showing Model, Density, Young's Modulus, Poisson's Ratio, Friction, contact gap rows, Bend Stiffness, Shrink X/Y, and the Strain Limit, Inflate, Plasticity, Bend Plasticity, and Velocity Overwrite toggles
:width: 500px

The **Material Params** box expanded on a **Shell** group. The exact
row set changes with the group's type: **Solid** collapses Shrink X/Y
into a single Shrink, **Rod** drops Poisson ratio, and **Static** hides
everything except **Friction** and the contact rows.
```

### Profile Buttons: Open / Clear / Reload / Save

Along the header of the **Material Params** box are four small buttons that
operate on a **material profile** (a TOML file listing named parameter
presets):

- **Open**: pops a file picker and loads the selected TOML into the
  profile dropdown for this group. The dropdown then lists every entry in
  the file; picking one pushes its parameters into the group.
- **Clear**: forgets the loaded file. The dropdown disappears until you
  open another TOML.
- **Reload**: re-reads the currently loaded TOML from disk and re-applies
  the active preset.
- **Save**: writes the group's current parameters back into the loaded
  TOML under a chosen entry name, replacing the existing entry if the name
  already exists.

Before a profile is loaded, the row collapses to a single **Open Profile**
button plus the **Save** icon (save can write a brand-new TOML without
an existing one). Once a profile is loaded, the **Profile** dropdown
appears and all four icons line up to the right of it.

:::{important}
**Material-profile TOML files are created by the Save icon, not by
hand.** Tune the group's material parameters in the panel, click the
**Save** icon, name the entry, and the add-on writes (or overwrites) it
in the `.toml` for you. The TOML structure documented below is shown
for inspection and sharing only; the supported edit path is always UI →
Save.

```{figure} ../../images/material_params/save_icon.png
:alt: Dynamics Groups panel with the floppy-disk Save icon next to the Open Profile button on a group's Material Params row highlighted in red
:width: 500px

The per-group **Save** icon (floppy disk, highlighted in red) at the
top-right of the **Material Params** profile row. Click it to write the
group's current material-parameter values to a `.toml` file, creating
the file on first save and overwriting the currently selected entry
afterwards.
```
:::

```{figure} ../../images/material_params/profile_copy_paste_row.png
:alt: Material Params header row with the Open Profile button and the save icon, the state when no material profile has been loaded yet
:width: 500px

**Before loading a profile.** The row shows a full-width **Open Profile**
button on the left and the save icon on the right. The **Copy** / **Paste**
clipboard icons sit at the top-right of the **Material Params** header
for moving parameters between groups in the same session.
```

```{figure} ../../images/material_params/profile_row_loaded.png
:alt: Material Params profile row after loading a TOML. The Profile dropdown is set to Cotton and four icons follow it: folder (Open), X (Clear), refresh (Reload), disk (Save)
:width: 500px

**After loading a profile.** The **Profile** dropdown (here set to
`Cotton`) now lists every entry in the loaded TOML; the four icons to
its right are **Open**, **Clear**, **Reload**, **Save**, left to
right.
```

### Copy / Paste

Next to the profile buttons is a pair of **Copy** and **Paste** buttons.
**Copy** snapshots every field in the current group's material parameters
to an internal clipboard; **Paste** applies that clipboard to another
group. This is the fastest way to reuse a tuned material without writing
a TOML file, but the clipboard lives only for the current Blender session.

## Shared Parameters

These apply regardless of type.

| UI label                             | Python / TOML key                 | Default | Description                                                              |
| ------------------------------------ | --------------------------------- | ------- | ------------------------------------------------------------------------ |
| **Friction**                         | `friction`                        | 0.5     | Coulomb friction coefficient at contacts (0 – 1).                        |
| **Contact Gap**                      | `contact_gap`                     | 0.001   | Absolute contact gap distance, in Blender units.                         |
| **Contact Offset**                   | `contact_offset`                  | 0.0     | Absolute contact offset, in Blender units.                               |
| **Use Group Bounding Box Diagonal**  | `use_group_bounding_box_diagonal` | `True`  | When true, contact distances are ratios of the group's bbox diagonal.    |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | 0.001   | Contact gap as a fraction of the group's bounding-box diagonal.          |
| **Contact Offset Ratio**             | `contact_offset_rat`              | 0.0     | Contact offset as a fraction of the group's bounding-box diagonal.       |

**Friction at a contact** is asymmetric in the material parameters
but symmetric in the solve: each object carries its own **Friction**
coefficient, and when two objects come into contact the solver
combines the two values into a single contact friction. The
combination rule is selected scene-wide by the **Friction Mode**
setting (Python / TOML key `friction_mode`) under the **Scene
Configuration** panel's **Advanced Params** sub-section:

- **Minimum** (`min`, default): take `min(friction_A, friction_B)`.
  The lower-friction surface wins, so a slippery cloth sliding over a
  grippy body behaves as if the whole contact were slippery. To make
  a contact feel grippy, both sides need to be set high.
- **Maximum** (`max`): take `max(friction_A, friction_B)`. The
  grippier surface wins, so a single high-friction object acts as a
  brake against everything it touches.
- **Mean** (`mean`): take `0.5 * (friction_A + friction_B)`. Each
  object contributes equally regardless of which side is grippier.

The default `min` reproduces the behavior of earlier releases and is
the safest choice when you have not set per-object friction values
deliberately.

See [Contact gap: absolute vs ratio](#contact-gap-absolute-vs-ratio) below
for which pair you should be editing.

## Rayleigh Damping

**Solid**, **Shell**, and **Rod** groups expose stiffness-proportional
Rayleigh damping in a **Rayleigh Damping** box. Both coefficients default
to `0.0` (no damping), must be non-negative, and are measured in seconds.

| UI label                | Python / TOML key      | Default | Applies to          | Description                                                       |
| ----------------------- | ---------------------- | ------- | ------------------- | ---------------------------------------------------------------- |
| **Deformation Damping** | `deformation_damping`  | 0.0     | Solid, Shell, Rod   | Damps stretch / membrane / solid deformation (seconds).          |
| **Bending Damping**     | `bending_damping`      | 0.0     | Shell, Rod only     | Damps shell and rod bending (seconds). Solid has no bending term. |

Start near zero and raise these only to calm jitter. Small values
(roughly 0.001 – 0.01 s) already reduce visible jitter noticeably;
bending damping is usually smaller than deformation damping.

:::{note}
**PDRD** groups are not Rayleigh-damped. The damping coefficients apply to
the FEM element types (Solid, Shell, Rod) only.
:::

:::{admonition} Under the hood
:class: toggle

Stiffness-proportional damping adds a `(beta/dt) * K` term to the system,
where `K` is the element tangent stiffness and `beta` is the coefficient
in seconds. The deformation term reuses the SPD-projected deformation
Hessian; the bending term uses a lagged-dihedral form so it stays
dissipative. Tetrahedral (Solid) elements use only the deformation term,
since a tet has no bending energy.
:::

## Shell-Specific

| UI label                 | Python / TOML key      | Default          | Description                                                    |
| ------------------------ | ---------------------- | ---------------- | -------------------------------------------------------------- |
| **Model**                | `shell_model`          | `BARAFF_WITKIN`  | Material model. One of `BARAFF_WITKIN`, `ARAP`.                |
| **Density (kg/m²)**      | `shell_density`        | 1.0              | Areal density, kg/m².                                          |
| **Young's Modulus (Pa/ρ)** | `shell_young_modulus`  | 1000.0         | Young's modulus (see note below). Accepted range 0 – 10 M.     |
| **Poisson's Ratio**      | `shell_poisson_ratio`  | 0.35             | Poisson ratio, 0 – 0.4999.                                     |
| **Bend Stiffness**       | `bend`                 | 10.0             | Hinge bending stiffness between neighboring faces. Min 0, soft max 100. **Rod** groups write the same property on their own scale; see [Bend Stiffness on a Rod](#bend-stiffness-on-a-rod). |
| **Shrink X**             | `shrink_x`             | 1.0              | Anisotropic warp scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Shrink Y**             | `shrink_y`             | 1.0              | Anisotropic weft scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Enable Strain Limit**  | `enable_strain_limit`  | `False`          | Turns on non-physical strain clamp (good for stiff cloth).     |
| **Strain Limit**         | `strain_limit_percent` | 5.0              | Max stretch beyond rest length, as a percentage (5.0 = 5%). Active only when **Enable Strain Limit** is on. |
| **Inflate**              | `enable_inflate`       | `False`          | Turns on per-face pressure along face normals.                 |
| **Pressure (Pa)**        | `inflate_pressure`     | 0.0              | Inflation pressure, Pa. Active only when **Inflate** is on.    |
| **Stitch Stiffness**     | `stitch_stiffness`     | 1.0              | Stiffness of loose-edge stitches detected in the mesh.         |

Loose edges (edges not belonging to any face) are automatically treated as
stitch constraints, with stiffness set by **Stitch Stiffness**.

### Shrink X / Shrink Y

What it does: anisotropic rest-shape scale. **Shrink X** scales the warp
direction and **Shrink Y** the weft; values below 1 shrink the cloth
along that axis, values above 1 extend it. They act on the rest shape,
so the solver sees the stretched/shrunk target as the relaxed
configuration and drives the mesh toward it under the usual stiffness.

When to enable: use to bake in pre-tension (shrink to pull seams taut),
to inflate panels slightly, or to recover the target shape after mesh
sewing. Leave both at `1.0` when you want the mesh drawn in Blender to
be the rest shape.

Example values:
- `(1.0, 1.0)`: default; no anisotropic rescale.
- `(0.95, 0.95)`: ~5% uniform shrink (mild curl / gathers).
- `(0.9, 1.1)`: shrink warp, extend weft (asymmetric tension).

Note: enabling shrink/extend disables **Strain Limit** for the same
group. The two systems fight, so the UI warns when both are active.

```{figure} ../../images/material_params/shrink_shell.png
:alt: Shrink X and Shrink Y rows highlighted in the Material Params box
:width: 500px

**Shell** groups expose **Shrink X** and **Shrink Y** on the same row.
Each is a scale factor relative to the rest shape; 1.0 leaves the axis
alone.
```

### Strain Limit

Available on **Shell** and **Rod** groups (not **Solid**).

What it does: non-physical clamp that prevents mesh edges from stretching
beyond the strain limit. Helpful for stiff cloth or ropes that look
rubbery in a plain spring formulation.

When to enable: cloth that should keep its silhouette (denim, tablecloths,
airbags) or ropes that must not visibly stretch. Disable when you want the
mesh to deform freely under force, or when **Shrink X** / **Shrink Y** are
non-unity on a **Shell** group (the two systems conflict).

Example values:
- **Strain Limit** = 2.5%: very stiff (~2.5% stretch).
- **Strain Limit** = 5%: default; tight but drapes visibly.
- **Strain Limit** = 15%: loose; bigger ripples.

```{figure} ../../images/material_params/strain_limit.png
:alt: Strain Limit toggle and value field highlighted in Material Params box
:width: 500px

With **Enable Strain Limit** on, the **Strain Limit** field activates.
The value is a stretch percentage (5% means edges may stretch 5% beyond
rest length), not a force.
```

### Inflate

What it does: applies a per-face pressure along each face normal, pushing
the mesh outward (or inward with negative values, once you dip below zero
via the Python API). Acts uniformly over the surface like a balloon or
airbag.

When to enable: inflatables (pillows, airbags, balloons), soft garments
that need a puffy silhouette, or any shell that should resist collapse
into a flat sheet. Leave off for ordinary cloth; gravity and bending
already do the right thing.

Example values:
- **Pressure (Pa)** = 0.0: default; feature is inert even when toggled on.
- **Pressure (Pa)** = 1.0: gentle puff; subtle volume for a pillow.
- **Pressure (Pa)** = 10.0: strong; airbag-style rapid fill.

```{figure} ../../images/material_params/inflate.png
:alt: Inflate toggle and Pressure (Pa) field highlighted in Material Params box
:width: 500px

**Enable Inflate** exposes the **Pressure (Pa)** slider. The unit label
is Pa but the solver applies it relative to density, like Young's modulus
(see the note below), so tune by eye rather than against SI values.
```

### Plasticity

What it does: adds permanent deformation on top of the elastic response.
When the local stretch exceeds the **Threshold** (a dead zone around
zero strain), the rest shape drifts toward the current shape at a rate
controlled by **Theta**. A matching **Bend Plasticity** section does the
same for the bending energy, with its own theta and angular threshold.

When to enable: materials that remember their deformation, such as crushed foil,
wrinkled paper, dented metal sheets, or sagging fabric. Keep off for
perfectly elastic cloth.

Example values:
- **Theta** = 0.0: disabled even if the checkbox is on.
- **Theta** = 0.5: default; ~40%/s creep once over threshold.
- **Theta** = 5.0: fast creep (~99%/s); nearly immediate set.
- **Threshold** = 0.02: ignore strains below 2%.

```{figure} ../../images/material_params/plasticity.png
:alt: Plasticity and Bend Plasticity sections highlighted in Material Params box
:width: 500px

**Shell** groups expose two plasticity sections: **Plasticity** (stretch)
and **Bend Plasticity** (hinge/rod-joint rest angle). Each has its own
theta rate and threshold; bend plasticity also lets you pick the
rest-angle source (Flat / Straight, or From Initial Geometry).
```

### Velocity Overwrite

What it does: the bottom box in the Material Params stack. It stores a
per-object list of keyframed velocity vectors. Each entry pins the
whole group to a given `(direction, speed)` at a chosen frame, overriding
the velocity produced by the simulation. The dropdown on the header row
picks which assigned object receives the keyframes; the eye icon toggles
a viewport preview arrow; the copy/paste icons move the keyframe list
between groups.

When to enable: scripted cloth launches (flag unfurling, parachute
drops), matching reference motion on hero shots, or giving the solver a
strong initial push that no constant velocity could time. Leave empty
for fully passive simulations.

```{figure} ../../images/material_params/velocity_overwrite.png
:alt: Velocity Overwrite section with four keyframes listed and one selected
:width: 500px

The **Velocity Overwrite** section with four keyframes populated
(frames 1, 30, 60, 90). Each row is `frame (speed m/s [direction])`.
The selected row expands into per-keyframe editor rows (**Frame**,
**Direction** (XYZ), and **Speed**) so you can tweak one entry
without opening an animation editor. The **Cloth** dropdown at the
top picks which assigned object the keyframes belong to, and the
`+` / `-` buttons on the right add or remove entries.
```

## Solid-Specific

| UI label                   | Python / TOML key     | Default              | Description                                               |
| -------------------------- | --------------------- | -------------------- | --------------------------------------------------------- |
| **Model**                  | `solid_model`         | `ARAP`               | Material model. Either `STABLE_NEOHOOKEAN` or `ARAP`.     |
| **Density (kg/m³)**        | `solid_density`       | 100.0                | Volumetric density, kg/m³.                                |
| **Young's Modulus (Pa/ρ)** | `solid_young_modulus` | 500.0                | Young's modulus (see note below). Range 0 – 10 M.         |
| **Poisson's Ratio**        | `solid_poisson_ratio` | 0.35                 | Poisson ratio, 0 – 0.4999.                                |
| **Shrink**                 | `shrink`              | 1.0                  | Uniform rest-shape scale (min 0.1).                       |

### Shrink

What it does: uniform (isotropic) rest-shape scale for the whole solid.
The solver treats the shrunk / expanded shape as the relaxed target and
drives the mesh toward it under the usual stiffness, so values below 1
visually contract the body and values above 1 swell it.

When to enable: pre-stressed solids (e.g. a rubber band that should
self-tension once the simulation starts), volumetric shrink after
tetrahedralization, or recovering a target volume after scale tweaks in
Blender. Leave at `1.0` for bodies that should rest exactly at their
modeled size.

Example values:
- **Shrink** = 1.0: default; no rescale.
- **Shrink** = 0.9: 10% shrink; body contracts and pulls on its neighbors.
- **Shrink** = 1.05: 5% expansion; useful for "puffy" solids.

```{figure} ../../images/material_params/shrink_solid.png
:alt: Shrink value row highlighted in Material Params box for a Solid group
:width: 500px

**Solid** groups expose a single **Shrink** row in the Material Params
box (**Shell** groups instead get anisotropic **Shrink X** / **Shrink
Y**).
```

### Tetrahedralizer (per object)

**Solid** groups only. The bottom of the Material Params box on a
**Solid** group has a **Tetrahedralizer** box. A solid's surface is
tetrahedralized before it is sent to the solver, and the box lets you
pick how, per assigned object:

- An **Object** dropdown on the header row picks which assigned mesh in
  the group you are configuring, so each solid in the group can use its
  own backend and overrides.
- A backend dropdown below it chooses the tetrahedralizer:
  - **fTetWild** (the default): a tolerant remesher. It accepts open,
    cracked, or non-manifold input, but it resamples the surface, so
    your input vertices are reconstructed through a surface map rather
    than preserved exactly.
  - **TetGen**: surface-exact (a one-to-one vertex map). It requires a
    clean, closed, manifold mesh and rejects open, coplanar, or
    non-manifold input. If TetGen refuses a mesh, repair it, route the
    object to a **Shell** group, or switch it back to **fTetWild**.

The override rows below the backend dropdown change to match the selected
backend.

#### fTetWild Overrides

When **fTetWild** is selected, six per-object overrides appear. Each row
has an **Override** checkbox on the left and the value on the right; the
value is only forwarded to fTetWild when its checkbox is on. With all
overrides off, fTetWild runs at its own defaults.

| UI label               | Python / TOML key         | Default   | Description                                                          |
| ---------------------- | ------------------------- | --------- | -------------------------------------------------------------------- |
| **Edge Length Factor** | `ftetwild_edge_length_fac`| 0.05      | Ideal tet edge length as a fraction of the bbox diagonal (`-l`).     |
| **Epsilon**            | `ftetwild_epsilon`        | 0.001     | Envelope size as a fraction of the bbox diagonal (`-e`).             |
| **Stop Energy**        | `ftetwild_stop_energy`    | 10.0      | AMIPS energy threshold; larger = faster, lower quality.              |
| **Max Opt Iterations** | `ftetwild_num_opt_iter`   | 80        | Maximum fTetWild optimization passes.                                |
| **Optimize**           | `ftetwild_optimize`       | `True`    | Improve cell quality (slower).                                       |
| **Simplify Input**     | `ftetwild_simplify`       | `True`    | Simplify the input surface before tetrahedralization.                |
| **Coarsen Output**     | `ftetwild_coarsen`        | `False`   | Coarsen output while preserving quality.                             |

Each value has a matching `ftetwild_override_<field>` boolean that gates
whether the override is sent. Leave the box collapsed and untouched to
get the tetrahedralizer's out-of-box behavior; reach for these only when
a solid is meshing too coarsely, missing features, or taking too long to
tetrahedralize.

```{figure} ../../images/material_params/ftetwild_expanded.png
:alt: Material Params box on a Solid group with the fTetWild disclosure expanded. Edge Length Factor has its Override checkbox on and value 0.05. Epsilon, Stop Energy, and Max Opt Iterations rows are grayed out because their Override checkboxes are off. Optimize has its Override checkbox on with the value toggled on.
:width: 500px

The **fTetWild** box expanded at the bottom of a **Solid** group's
**Material Params**. The left column is the per-field **Override**
checkbox; with it off, the row is grayed and the tetrahedralizer's
own default is used. In this example **Edge Length Factor** and
**Optimize** are overridden; the rest stay at defaults.
```

#### TetGen Overrides

When **TetGen** is selected, the override rows switch to TetGen's
interior controls. TetGen always preserves the input surface exactly, so
these tune only the interior refinement. Each row uses the same
**Override** checkbox pattern: the value is forwarded only when its box
is on, and the rest of the time TetGen runs at its own defaults.

| UI label                   | Python / TOML key    | Default | Description                                                          |
| -------------------------- | -------------------- | ------- | -------------------------------------------------------------------- |
| **Min Radius-Edge Ratio**  | `tetgen_min_ratio`   | 2.0     | Quality bound; smaller forces rounder interior cells (`-q`).         |
| **Max Tet Volume**         | `tetgen_max_volume`  | 0.0     | Caps interior cell size in object units (`-a`); 0 leaves it uncapped. |

## Rod-Specific

| UI label                   | Python / TOML key   | Default   | Description                                       |
| -------------------------- | ------------------- | --------- | ------------------------------------------------- |
| **Model**                  | `rod_model`         | `ARAP`    | Material model. `ARAP` is the only option.        |
| **Density (kg/m)**         | `rod_density`       | 1.0       | Line density, kg/m.                               |
| **Young's Modulus (Pa/ρ)** | `rod_young_modulus` | 10000.0   | Young's modulus (see note below).                 |
| **Shrink**                 | `length_factor`     | 1.0       | Rest-length scale for every segment of the strand (min 0.1). Below 1 pulls the strand taut, above 1 leaves it slack. |
| **Bend Stiffness**         | `bend`              | 1.0       | How strongly the strand resists being curved. Min 0, soft max 100. |

**Bend Stiffness** writes the same `bend` property a **Shell** group
writes, so a material profile carries one value for both types. Each
type scales that value on its own terms, so a number tuned on cloth is
not a number tuned on a strand: switching a group's type to **Rod** sets
`bend` to `1.0`, the rod-tuned default, while a **Shell** group keeps the
global default of `10.0`.

**Rest Angle** sits in that same **Bend** box on a rod. A **Shell** group
draws it instead as the first row of the unlabeled box that also carries
**Bend Plasticity**, so look above the **Bend Plasticity** checkbox, not
below it. Either way, pick **Flat / Straight** to keep the analytic rest
angle (rod θ₀ = π, shell hinge θ₀ = 0), or **From Initial Geometry** to
take the rest angle from the input pose.

### Bend Stiffness on a Rod

What it does: sets how strongly the strand resists being curved. At `0`
the rod is a limp thread that resists only stretching; raise it and the
strand holds a curve the way a wire, a cable, or a bristle does.

**A rod bends the same way no matter how many segments it is drawn
with.** The stiffness is measured against a reference segment of one
centimeter and rescaled to whatever segment length the strand actually
has, so adding control points for a smoother silhouette or for finer
contact resolution leaves the drape unchanged, and the same strand drawn
coarse and drawn fine is the same rod. Tune the value on a draft mesh and
keep it when you refine the geometry.

The reference segment also fixes what the numbers mean, which is why a
rod starts at `1.0` where a shell starts at `10.0`: the same figure on the
two types is not the same stiffness.

When to change it: raise it for wire, cable, tubing, or hair that should
keep the shape it was drawn with; lower it toward `0` for thread, string,
and yarn that should fall limp and take its shape from gravity and
contact alone.

:::{important}
**Opening a saved rod scene.** A **Bend Stiffness** value picked against a
particular segment length carries that length with it, so a saved rod whose
segments are not about one centimeter long opens looking different from the
shape it was authored with: at 1 mm segments it reads much stiffer, at 10 cm
segments much softer. To restore the look, multiply the
group's **Bend Stiffness** by the square of its segment length in
centimeters.

| Segment length | Multiply **Bend Stiffness** by |
| -------------- | ------------------------------ |
| 5 cm           | 25                             |
| 2 cm           | 4                              |
| 1 cm           | 1 (unchanged)                  |
| 5 mm           | 0.25                           |
| 1 mm           | 0.01                           |

Measure the segment length on the rest pose: it is the spacing between
two neighboring points of the strand, times **Shrink** when that is not
`1.0`. **Shell** groups are unaffected; the reference segment applies to
rods only.
:::

### Shrink on a Rod

What it does: scales the rest length of every segment in the strand.
Below `1.0` the rest length is shorter than the drawn geometry, so a
strand pinned at both ends pulls itself taut; above `1.0` the rest length
is longer, so the strand sags or buckles between its pins. It is the rod
counterpart of **Shell**'s **Shrink X** / **Shrink Y** and **Solid**'s
**Shrink**, and like them it moves the rest shape rather than the drawn
geometry, so nothing changes until the simulation starts.

When to enable: stringing a warp or a guy line that should be under
tension from frame one, taking the slack out of a rope that was modeled
loose, or deliberately adding slack so a cable drapes.

Mass is unchanged: a strand's mass comes from its density and its drawn
length, so tensioning a rod with **Shrink** does not make it lighter.

Bending changes with it. **Bend Stiffness** is measured against the rest
length, which is what **Shrink** scales, so **Shrink** also moves how
stiff the strand is in bending by the inverse square: **Shrink** = `0.5`
leaves the rod about four times stiffer in bending and **Shrink** = `2.0`
about four times floppier. When you want the tension without the change
in stiffness, multiply **Bend Stiffness** by the square of the **Shrink**
value.

Example values:
- **Shrink** = 1.0: default; the drawn geometry is the rest shape.
- **Shrink** = 0.97: mild pre-tension; a strand pulled straight between
  its pins.
- **Shrink** = 1.05: slack; the strand bows out between its pins.

:::{admonition} Under the hood
:class: toggle

Each interior point of a strand carries the bending energy
`0.5 * k * (θ - θ₀)²`, where `θ` is the angle between the two segments
meeting at that point and `θ₀` the rest angle chosen by **Rest Angle**
(`π` for a straight rod). The coefficient is

```text
k = bend * m * (L_ref / l)²        L_ref = 1 cm
```

with `m` the lumped mass at that point and `l` its Voronoi rest length
(half the sum of the two segment rest lengths meeting there, so it carries
the **Shrink** factor).

The two are not the same length. `m` is half of each incident segment's
mass, and a segment's mass is its line density times its **drawn** length,
taken from the geometry before **Shrink** scales the rest length. So with
`d` the drawn spacing at that point, `m = density × d` while
`l = d ×` **Shrink**, and the coefficient expands to

```text
k = bend × density × L_ref² / (d × Shrink²)
```

which reads two ways. Hold **Shrink** fixed and `k` falls as `1/d`: that
is `k = B / l` with `B` the flexural rigidity `bend × linear density ×
L_ref²`, which is what the continuum bending energy `0.5 * B * κ²`
integrated along the strand discretizes to at an interior point. Linear
density is `m / l`, so `B` works out to
`bend × density × L_ref² /` **Shrink** and carries no `d` at all, which
is why the bent shape does not depend on how finely the strand is drawn.
Hold the geometry fixed instead and `k` goes as `1 / Shrink²`, the
inverse square described above. Taking `B` per unit density is a separate
normalization, and it is what keeps the shape independent of the density
you set, the same one the shell hinge applies through areal density.
`L_ref` only places the numeric range of the **Bend Stiffness** field.

Rest lengths are measured in solver space, after **World Scaling** has
been applied to the geometry, so at the default **World Scaling** of `1.0`
a segment's rest length is its length in Blender units.
:::

## Sand-Specific

A **Sand** group is a granular body: a cloud of grain centers, each grain
a sphere of one shared radius. Its geometry is that cloud rather than a
surface, so a mesh has to be converted first (see
[Creating a Sand Body](#creating-a-sand-body) below). The **Material
Params** box for a Sand group is short because most of the cloth and
solid stack has nothing to act on.

| UI label                | Python / TOML key     | Default | Description                                                                      |
| ----------------------- | --------------------- | ------- | ---------------------------------------------------------------------------------- |
| **Grain Radius (m)**    | `sand_grain_radius`   | 0.02    | Radius of one grain, in meters. Chosen when the mesh is converted and drawn read-only after that. Min 0.0001. |
| **Particle Mass (g)**   | `sand_particle_mass`  | 1.0     | Mass of a single grain, in grams (the solver receives it in kilograms). Range 0.000001 – 1000000. |
| **Friction**            | `sand_friction`       | 0.0     | Coulomb friction coefficient between grains. Min 0. Raise it to make a pile hold a steeper slope. |
| **Contact Gap**         | `contact_gap`         | 0.001   | Barrier activation distance on top of the grain radius, in Blender units.        |

**Friction** on a Sand group is its own field (`sand_friction`), not the
shared **Friction** row, which the box does not draw for this type.

**Grain Radius** is fixed at conversion and drawn grayed out afterward,
with a *Grain radius is locked at convert* note under it. The
non-overlapping spacing of the grains is derived from the radius when the
cloud is seeded, so the two have to agree: a larger radius on the same
cloud would put grains inside each other's contact skin, and the solver
refuses an overlapping cloud at startup. To work at a different radius,
convert the source mesh again.

The grain radius is also the **contact offset**: the sphere it describes
is the grain's physical skin, so there is no separate **Contact Offset**
row and **Contact Gap** is the only extra barrier distance you set. The
box says so with a *Grain radius is the contact offset* note.

### Creating a Sand Body

**Convert To Solid Particle Mesh** sits on the group's box just above
**Delete Group**, so you can reach it without expanding **Material
Params**. It is enabled when the active object is a selected mesh that has
faces and is not already a particle mesh; otherwise the button is grayed
out and a line underneath says which of those three is missing.

The dialog has three fields:

- **Grain Radius**: the physical radius of one grain, and the value that
  gets locked onto the object.
- **Extra Spacing**: how much room to leave between grains beyond
  touching. `0` packs them as densely as the non-overlap rule allows;
  larger values give a looser, sparser cloud.
- **Random Seed**: picks a different arrangement at the same radius and
  spacing.

The grain count is not something you set. Grains fill the volume at the
radius and spacing you chose, and the count is whatever that comes to; the
report line after the conversion tells you the number.

:::{warning}
**The conversion is destructive.** The object's faces are discarded and
replaced by a cloud of loose vertices plus a render-only **Particle Mesh**
modifier that draws each vertex as a sphere. Keep a copy of the source
mesh if you may want to re-convert at a different radius.
:::

:::{note}
Leave **Preconditioner** on **Block Jacobi** for a scene with a Sand
group. A grain cloud has no faces and no edges, so it carries none of the
connectivity the **Schwarz** preconditioner builds its aggregates from.
See [Preconditioner](scene.md#preconditioner).
:::

## PDRD-Specific

**PDRD** (Painless Differentiable Rotation Dynamics) is an exactly-rigid
body type. It exposes only density, friction, and contact settings. There
is no Young's modulus, Poisson ratio, bend, shrink, strain limit, or
inflate, no rigidity or stiffness control (the body is exactly rigid, not
a stiff penalty), and PDRD bodies are not tetrahedralized.

| UI label                  | Python / TOML key | Default | Description                                                                  |
| ------------------------- | ----------------- | ------- | ---------------------------------------------------------------------------- |
| **Density (kg/m³)**       | `pdrd_density`    | 100.0   | Volumetric density, kg/m³. Mass is the density times the enclosed mesh volume. |

Density is the only material number a PDRD body exposes: it sets the mass
(and, through the rest shape, the rotational inertia), which is what
determines how the body responds to gravity, contact, and pins. The motion
is exactly rigid at any mesh resolution, so there is no stiffness or
rigidity value to tune.

:::{admonition} Under the hood
:class: toggle

Each PDRD body is solved in reduced 6-DOF coordinates (translation plus
rotation): every Newton iteration fits the single best-fit rigid transform
to the body and reconstructs its surface from that transform, so the body
stays exactly rigid by construction rather than through a stiff penalty.
:::

### Hinge Joints

A PDRD body can be turned into a **hinge**: its position is pinned and its
rotation is locked to a single principal (PCA) axis of its rest shape, so
the body spins on that axis like a wheel on an axle. This is a per-object
setting, so each body in one PDRD group can hinge on its own axle (for
example, a train of gears that each turn on their own pin while tooth
contact passes torque from one to the next).

The free axle is chosen by **principal axis** of the rest shape: `0` is
the largest extent, `1` the middle, and `2` the thinnest extent (the
usual axle for a flat gear or disk, and the default). From the Python API
a hinge is set per object with `Group.set_hinge`; from the MCP layer use
the `set_pdrd_hinge` tool. Pass `enable=False` to clear the hinge and let
the body move freely again.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

gears = solver.create_group("Gears", type="PDRD")
gears.add("GearA")
gears.set_hinge("GearA", pca_axis=2)   # spin on the thinnest axis
```

The hinge is a per-object property, not a group material attribute, so it
does not appear in the **Material Params** table above.

:::{note}
**Young's modulus behaves non-conventionally.** The solver divides the
entered Young's modulus by density internally. The practical effect is that
animated behavior is invariant to density alone: doubling density without
touching Young's modulus produces the same motion (the mass doubles, but
the effective stiffness scales with it). This decouples "how heavy the
material is" from "how stiff it looks", so you can tune stiffness and mass
independently. The example material presets in this guide (`Cotton`, `Silk`, `Steel`, …)
use physically meaningful values with that normalization in mind.
:::

### Density-Normalized (Pa/ρ)

Below the **Young's Modulus** field on **Shell**, **Solid**, and **Rod**
groups is a **Density-Normalized (Pa/ρ)** checkbox that sets what the
number you type means.

| UI label                       | Python / TOML key              | Default | Description                                                                |
| ------------------------------ | ------------------------------ | ------- | -------------------------------------------------------------------------- |
| **Density-Normalized (Pa/ρ)**  | `young_mod_density_normalized` | `True`  | On: Young's modulus is the solver's native Pa/ρ. Off: a true value in Pa.  |

When it is **on** (the default), the Young's modulus is a
density-normalized value in Pa/ρ, the solver's native convention:
changing a body's density alone leaves its motion unchanged, and the
field label reads **(Pa/ρ)**. Keep it on to match existing scenes and the
example presets in this guide.

When you turn it **off**, you enter a true Young's modulus in pascals (for
example a value from a material reference table); the add-on divides it by
this group's density before sending it to the solver, so a denser body of
the same material is correspondingly stiffer to move. The field label
flips to **(Pa)** to show which convention is active.

## Static-Specific

A **Static** group is a collider: you animate its shape, and the solver
pushes everything else out of its way. Besides **Friction** and the contact
rows it carries one option.

### Apply Soft Constraints

By default a Static collider follows its animation exactly. Nothing can move
it, however hard the cloth presses. That is what you want when the cloth can
always get out of the way.

It is the wrong answer when the collider's own shape closes onto the cloth.
If a character's arm comes down against the torso, or a hand presses into a
thigh, the garment in between is caught between two surfaces that will not
budge, and the simulation stops with an error saying it cannot advance.

Turn on **Apply Soft Constraints** and the collider is held toward its
animated shape by springs instead of being locked to it. Where the cloth
pushes harder than the springs, the collider gives way; once the pinch
passes, it settles back onto its animation.

### Stiffness

How firmly the springs hold, shown only while **Apply Soft Constraints** is
on. Lower values let the collider yield more; higher values behave more like
the exact default.

There is no universal number, because the springs are competing against
however hard your scene's contacts push. Start at the default of **10** and
adjust from what you see. On a clothed character rig, **0.1** was so soft
the collider folded into itself, **1** worked but let the body dent by
around 3 cm at the tightest contact, **10** kept it within a fraction of a
millimetre on average and 8 mm at worst, and **1000** was close enough to
exact that the original pinch came back.

:::{note}
A Static group with **Apply Soft Constraints** on is simulated, not just
collided against, even if it never moves. There has to be something for the
springs to act on. Expect it to cost more time per frame than the same
collider left exact, so turn it on for the colliders that need it rather
than for all of them.
:::

:::{admonition} Under the hood
:class: toggle

An exact collider vertex is a boundary condition: the solver removes its
degrees of freedom, so the contact force has nothing to act on. A soft one
keeps its degrees of freedom and gains a spring term, `k * (target - x)`,
which is what the contact force can work against.

Two exclusions come with it. Collider geometry no longer collides with
itself or with another collider, which is what allows a rigged mesh that
ships self-intersecting (layered eyelashes, an arm resting inside a torso)
to be used at all. Collider faces also carry no stretch or bending energy;
their shape comes from the springs, not from a material.
:::

## Contact Gap and Contact Offset

**Contact Gap** and **Contact Offset** are two distances that together
shape the invisible contact layer around each group's geometry. They
serve different roles and both are configurable.

- **Contact Gap** is the barrier's reach: the distance at which the
  solver starts applying a push-back force between two surfaces. A
  larger gap gives a softer, earlier-engaging barrier and costs more
  contact pairs; a smaller gap lets surfaces sit closer before the
  barrier kicks in. This is the setting most scenes need to tune.
- **Contact Offset** is per-group padding added on top of the gap. At
  each contact check the solver sums the two participants' offsets
  with the (averaged) gap and treats that total as the effective
  separation threshold. You can think of it as the group's "skin
  thickness": it guarantees a minimum clearance regardless of what the
  other side chose. The default is `0.0` (no extra clearance), which
  is what most scenes want.

Reach for **Contact Offset** when one group needs a specific thickness
for visual or collision reasons independent of what its neighbors do,
for example a garment that should never touch the body by less than a
millimeter no matter which body group it comes near. For day-to-day
tuning of how tightly surfaces sit, leave **Contact Offset** at zero
and adjust **Contact Gap** instead.

## Contact Gap: Absolute vs Ratio

Both **Contact Gap** and **Contact Offset** can be specified in either
of two ways:

- **Absolute** (the **Contact Gap** and **Contact Offset** fields): a
  literal distance in Blender units. Good when you want a hard, known
  thickness, e.g. a 1 mm skin for a body.
- **Ratio** (the **Contact Gap Ratio** and **Contact Offset Ratio**
  fields): a fraction of the group's bounding-box diagonal, computed at
  transfer time. Good because it scales with the scene: rescaling a
  character by 10× doesn't make the cloth penetrate.

```{figure} ../../images/material_params/contact_gap_modes.svg
:alt: Absolute vs ratio contact-gap comparison. In absolute mode the halo is the same thickness on small and large objects; in ratio mode it scales with the object
:width: 500px

The dashed red ring shows the contact-gap layer. **Absolute** mode keeps
the layer thickness constant in world units, so it looks huge around a
small object and thin around a large one. **Ratio** mode scales the
layer with the object's bounding box, so both look proportionally
wrapped regardless of scale.
```

The **Use Group Bounding Box Diagonal** toggle picks between them. The
**default is ratio-of-bbox-diagonal** because that's what most users
want; you only need to flip to absolute when the group contains
unusually elongated objects (where the diagonal overestimates
characteristic size) or when you need an exact contact thickness for
matching against another group.

Both pairs (**Contact Gap** / **Contact Gap Ratio** and **Contact
Offset** / **Contact Offset Ratio**) are independently controlled by
the same toggle.

## Material Profiles

A **material profile** is a named set of material parameters saved to a
TOML file with the **Save** icon. A single file can hold any number of
presets; profiles like these are easy to build:

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

:::{note}
Material profiles do **not** carry any object assignments, pin vertex
groups, or per-object velocity overrides. They describe a material, not
a scene.
:::

### Example TOML Stanza

The block below shows what the add-on writes out when you click Save.
It is **not** a template to fill in by hand. Adjust a group's Material
Params in the panel and click the **Save** icon to produce (or update)
a file like this.

```{figure} ../../images/material_params/save_icon.png
:alt: Dynamics Groups panel with the floppy-disk Save icon on the Material Params row highlighted in red
:width: 500px

The per-group **Save** icon (floppy disk, highlighted in red) on the
**Material Params** row. Clicking it writes the group's current
material-parameter values to a `.toml` file, creating the file on the
first save and overwriting the currently selected entry afterwards.
```

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
solid_density = 1000.0
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

Only the keys you include are applied; missing keys keep their current
value on the group. You don't have to list every field for a preset to be
valid — a `Static` collider preset, for instance, can carry just a
`friction` value.

## Blender Python API

The same workflow is available from Python. Every field in the
**Material Params** box is reachable through each group's `.param`
attribute. Changes from Python appear in the panel immediately and vice
versa.

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

# Static collider: friction, contact settings, and optionally soft constraints.
floor = solver.create_group("Floor", "STATIC")
floor.param.friction = 0.8

# A collider whose own shape closes onto the cloth can trap it. Holding the
# collider with springs lets it give way where the cloth pushes back.
body = solver.create_group("Body", "STATIC")
body.param.enable_soft_constraint = True
body.param.soft_constraint_stiffness = 10.0
```

:::{admonition} Under the hood
:class: toggle

**Loose-edge stitch encoding**

At transfer time, edges on **Shell** meshes that are not adjacent to
any face are automatically emitted as stitch constraints with stiffness
set by `stitch_stiffness`. There is no UI surface for this; it happens
on every transfer.

```{figure} ../../images/material_params/loose_edge_stitch.png
:alt: Two subdivided square Shell patches stacked with a gap, connected by vertical red edges. Each edge has no adjacent face and is automatically emitted as a stitch constraint
:width: 500px

Two subdivided square Shell patches joined by vertical **loose edges**
(rendered here as red tubes). The patches are separate face regions;
the connecting edges belong to no face, so the transfer step emits each
one as a stitch constraint with stiffness `stitch_stiffness`.
```

**Copy / Paste clipboard**

The **Copy** / **Paste** buttons move parameters between groups within a
single Blender session. The clipboard is not persisted to the `.blend`
file, so restarting Blender clears it.
:::
