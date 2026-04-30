# 🧪 Material Parameters

Every object group carries its own copy of the full material-parameter set,
but which fields are relevant depends on the group's type:

- **Shell**: density, stiffness (Young's modulus, Poisson ratio, bend),
  shrink, strain limit, inflate, stitch, and contact settings.
- **Solid**: density, stiffness, a single shrink factor, and contact
  settings.
- **Rod**: density, stiffness, bend, strain limit, and contact settings.
- **Static**: only friction and contact settings (static objects have no
  deformation to tune). See [Static Objects](../scene/static_objects.md) for the
  full treatment of Static groups, including how to animate them.

Rows that don't apply to the current type are hidden in the UI.

```{figure} ../../images/material_params/group_type_popdown.png
:alt: The group-type dropdown menu, popped open. A Type label with a disclosure arrow sits above a vertical list of four buttons (Solid, highlighted as the current selection; Shell; Rod; Static) each spanning the full width of the popup.
:width: 360px

The four options in the group-type dropdown on each group's header row.
Picking one changes the **Material Params** box to match: **Solid**
shows density, stiffness, and a single shrink factor; **Shell** shows
the full cloth stack including anisotropic shrink, strain limit,
inflate, and stitch; **Rod** shows density, stiffness, bend, and
strain limit; **Static** collapses to just **Friction** and the
contact rows.
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
   **Shell** groups can choose Baraff-Witkin, Stable NeoHookean, or
   ARAP; **Solid** groups pick between Stable NeoHookean and ARAP;
   **Rod** groups are locked to ARAP; **Static** groups have no model
   row.
2. **Density**: the material's density in type-appropriate units (kg/m²
   for **Shell**, kg/m³ for **Solid**, kg/m for **Rod**).
3. **Young's Modulus**: stiffness. See the note below for how the solver
   interprets it.
4. **Poisson Ratio**: for **Shell** and **Solid** only.
5. **Friction**: Coulomb friction coefficient at contacts.
6. **Bend stiffness** and **Shrink**. **Shell** shows Bend, Shrink X/Y,
   a **Strain Limit** toggle, an **Inflate** toggle, and a **Stitch
   Stiffness** field. **Solid** collapses down to a single Shrink
   slider. **Rod** reuses the **Shell** Bend row.
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
| **Friction**                         | `friction`                        | 0.0     | Coulomb friction coefficient at contacts (0 – 1).                        |
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

## Shell-Specific

| UI label                 | Python / TOML key      | Default          | Description                                                    |
| ------------------------ | ---------------------- | ---------------- | -------------------------------------------------------------- |
| **Model**                | `shell_model`          | `BARAFF_WITKIN`  | Material model. One of `BARAFF_WITKIN`, `STABLE_NEOHOOKEAN`, `ARAP`. |
| **Density (kg/m²)**      | `shell_density`        | 1.0              | Areal density, kg/m².                                          |
| **Young's Modulus (Pa/ρ)** | `shell_young_modulus`  | 1000.0         | Young's modulus (see note below). Accepted range 0 – 10 M.     |
| **Poisson's Ratio**      | `shell_poisson_ratio`  | 0.35             | Poisson ratio, 0 – 0.4999.                                     |
| **Bend Stiffness**       | `bend`                 | 100.0            | Bending stiffness, 0 – 100.                                    |
| **Shrink X**             | `shrink_x`             | 1.0              | Anisotropic warp scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Shrink Y**             | `shrink_y`             | 1.0              | Anisotropic weft scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Enable Strain Limit**  | `enable_strain_limit`  | `False`          | Turns on non-physical strain clamp (good for stiff cloth).     |
| **Strain Limit**         | `strain_limit`         | 0.05             | Max strain permitted when **Enable Strain Limit** is on.       |
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
- **Strain Limit** = 0.025: very stiff (~2.5% stretch).
- **Strain Limit** = 0.05: default; tight but drapes visibly.
- **Strain Limit** = 0.15: loose; bigger ripples.

```{figure} ../../images/material_params/strain_limit.png
:alt: Strain Limit toggle and value field highlighted in Material Params box
:width: 500px

With **Enable Strain Limit** on, the **Strain Limit** field activates.
The value is a strain ratio (0.05 = 5%), not a force.
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
rest-angle source (flat, initial geometry, or current frame).
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
| **Density (kg/m³)**        | `solid_density`       | 1000.0               | Volumetric density, kg/m³.                                |
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

### fTetWild Overrides

**Solid** groups only. The bottom of the Material Params box on a
**Solid** group has an **fTetWild** disclosure row; expanding it reveals
six per-group overrides for the tetrahedralizer the add-on runs on the
input surface before sending the mesh to the solver. Each row has an
**Override** checkbox on the left and the value on the right; the value
is only forwarded to fTetWild when its checkbox is on. With all
overrides off, the tetrahedralizer runs at its own defaults.

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

## Rod-Specific

| UI label                   | Python / TOML key   | Default   | Description                                       |
| -------------------------- | ------------------- | --------- | ------------------------------------------------- |
| **Model**                  | `rod_model`         | `ARAP`    | Material model. `ARAP` is the only option.        |
| **Density (kg/m)**         | `rod_density`       | 1.0       | Line density, kg/m.                               |
| **Young's Modulus (Pa/ρ)** | `rod_young_modulus` | 10000.0   | Young's modulus (see note below).                 |

**Rod** groups expose the same **Bend Stiffness** field as **Shell**;
it writes into the single `bend` property on the group, so both types
read and serialize it identically.

:::{note}
**Young's modulus behaves non-conventionally.** The solver divides the
entered Young's modulus by density internally. The practical effect is that
animated behavior is invariant to density alone: doubling density without
touching Young's modulus produces the same motion (the mass doubles, but
the effective stiffness scales with it). This decouples "how heavy the
material is" from "how stiff it looks", so you can tune stiffness and mass
independently. The shipped material profiles (`Cotton`, `Silk`, `Steel`, …)
are tuned to physically meaningful values with that normalization in mind.
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
TOML file. The add-on ships an example material profile with the following
presets:

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
strain_limit = 0.05
stitch_stiffness = 1.0
```

Only the keys you include are applied; missing keys keep their current
value on the group. You don't have to list every field for a preset to be
valid. See `Silk` or `Static` in the shipped file for minimal examples.

## Blender Python API

The same workflow is available from Python. Every field in the
**Material Params** box is reachable through each group's `.param`
attribute. Changes from Python appear in the panel immediately and vice
versa.

```python
from zozo_contact_solver import solver

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
