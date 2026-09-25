# 🧪 Material Parameters

Every object group carries its own copy of the full material-parameter set,
but which fields are relevant depends on the group's type:

- **Shell**: density, stiffness (Young's modulus, Poisson ratio, bend),
  shrink, strain limit, inflate, stitch, and contact settings.
- **Solid**: density, stiffness, a single shrink factor, stitch, and
  contact settings.
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
particle mass, friction, and the contact rows.
```

## The Material Params Box

At the bottom of each group card in the **Dynamics Groups** panel is a
collapsible **Material Params** box. When you expand it you see a
type-specific set of parameter rows: switching the group's type (for
example from **Solid** to **Shell**) immediately changes which rows are
visible, so the box always reflects the parameters that actually affect
the selected type. A **Static** group shows only **Friction**, the
**Apply Soft Constraints** box, the **Contact** rows and **Allow
Intersections**; a **Shell** group shows the full stack of density,
stiffness, bending, shrink, strain limit, inflation, and stitch fields;
and so on.

The **Material Params** header row carries the **Copy** / **Paste**
icons. Inside the box, **Solid** and **Shell** groups get a **Preset**
dropdown of the bundled materials above the profile row, filtered by the
group's Type: a **Shell** group lists the six fabrics — Silk, Flag,
Cotton, Wool, Denim and Leather (see
[Fabric Presets](fabric_presets.md)) — and a **Solid** group lists
Rubber, Silicone, Foam, Sponge and Jelly. The other types have no
bundled presets and omit that row. Each value row below carries a
padlock, which holds its value against **Preset** and **Paste**, plus
Blender's own keyframe control; the toggles and dropdowns between them
(**Model**, **Enable Strain Limit**, and so on) carry neither. A
keyframe is offered only on the properties the encoder samples, so the
densities, the shrink factors, **Particle Mass**, **Sand Friction** and
**Stitch Stiffness** are lockable but refuse an F-curve. Every group
setting outside those sampled sliders is read once, at the solve's
**Starting Frame**. A keyframe that a saved file still carries on one of
them, and a driver or an NLA strip on any group setting, sampled sliders
included, stops **Transfer** with a message naming its data path rather
than being read once in silence; see
[Dynamic Parameters](dynamic.md#supported-parameters).

On a **Solid** group a keyframe reaches the solve only on **Friction**,
**Contact Gap** and **Contact Offset** (or their ratio forms), which live
on the solid's surface. Its elastic values, such as **Young's Modulus**,
**Poisson's Ratio**, **Plasticity** and **Deformation Damping**, live on
its tetrahedra, which hold one value for the whole solve, so **Transfer**
refuses a keyframe on any of them and names the property. Remove the
keyframes, or split the change into separate solves.

The parameter rows, in roughly the order they are drawn (the exact
sequence varies by type — a **Shell** group draws its contact box before
Bend and Shrink, for instance):

1. **Model** (when applicable): dropdown to pick the material model.
   **Shell** groups can choose Baraff-Witkin or ARAP; **Solid** groups
   pick between Stable NeoHookean and ARAP; **Rod** groups are locked
   to ARAP; **Static** groups have no model row. The **Shell** picker
   does not offer Stable NeoHookean, but a `.blend` or material profile
   saved with it on a **Shell** group still loads it. The **Model** row
   then draws blank, with *Stable NeoHookean is not offered for shells;
   Transfer will refuse* under it, and **Transfer** refuses the group
   until you choose ARAP or Baraff-Witkin. A **Solid** group still
   offers Stable NeoHookean.
2. **Density**: the material's density in type-appropriate units (kg/m²
   for **Shell**, kg/m³ for **Solid**, kg/m for **Rod**).
3. **Young's Modulus**: stiffness. See the note below for how the solver
   interprets it.
4. **Poisson's Ratio**: for **Shell** and **Solid** only.
5. **Friction**: Coulomb friction coefficient at contacts.
6. **Bend stiffness** and **Shrink**. **Shell** shows Bend, the two
   directional rows **Bending Stiffness (Warp)** and **Bending Stiffness
   (Weft)** right below it, Shrink X/Y, a **Strain Limit** toggle, an
   **Inflate** toggle, and a **Stitch Stiffness** field. **Solid**
   collapses down to a single Shrink slider and keeps a **Stitch
   Stiffness** field near the bottom of the box. **Rod** draws its
   **Shrink** row just under **Friction** and its **Bend Stiffness**
   field in a separate **Bend** box below the contact rows.
7. **Contact Gap**: on **Solid**, **Shell**, **PDRD** and **Static**
   groups a toggle picks between absolute distance (in Blender units)
   and a fraction of the group's bounding-box diagonal, and the relevant
   pair of fields shows up below the toggle. A **Rod** group has no
   toggle and always uses the absolute pair; a **Sand** group shows
   **Contact Gap** alone, because its grain radius is the contact
   offset.
8. **Collision Active Duration Windows**: optional per-object frame
   ranges that restrict when contact is active. Off by default for
   **Solid**, **Shell**, **Rod**, and **PDRD** groups; unavailable for
   **Static** and **Sand**.
   Covered in
   [Active collision windows](../scene/object_groups.md#active-collision-windows).
9. **Spatial Material Maps**: optional per-group maps that vary one
   material parameter across the surface from painted per-vertex weights.
   **Shell** and **Solid** only; covered in its own section below.
10. **Plasticity**: optional non-linear permanent deformation. Covered in
    its own subsection below.
11. **Velocity Overwrite**: optional keyframed velocity targets for one
    of the assigned objects. Covered separately below.

```{figure} ../../images/material_params/box_shell.png
:alt: Material Params box expanded on a Shell group, showing Model, Density, Young's Modulus, Poisson's Ratio, Friction, contact gap rows, Bend Stiffness, Shrink X/Y, and the Strain Limit, Inflate, Plasticity, Bend Plasticity, and Velocity Overwrite toggles
:width: 500px

The **Material Params** box expanded on a **Shell** group. The exact
row set changes with the group's type: **Solid** collapses Shrink X/Y
into a single Shrink, **Rod** drops Poisson ratio, and **Static** hides
everything except **Friction**, **Apply Soft Constraints**, the contact
rows and **Allow Intersections**.
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

A profile carries the core material block only. The Rayleigh damping
coefficients, the two directional bending rows, the **Allow
Intersections** flags, a **Rod**'s **Shrink**, and the PDRD and Sand
fields are outside it, so set those on the group after applying a
profile rather than expecting Save to preserve them.

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
| **Allow Self-Intersections**         | `allow_self_intersection`         | `False` | Let an object pass through itself, with no contact between its parts.    |
| **Allow Inter-Object Intersections** | `allow_inter_object_intersection` | `False` | Let an object pass through every other object, with no contact.          |
| **Allow Inter-Group Intersections**  | `allow_inter_group_intersection`  | `False` | Let an object pass through objects of other groups, with no contact.     |

**Friction at a contact** is asymmetric in the material parameters
but symmetric in the solve: each object carries its own **Friction**
coefficient, and when two objects come into contact the solver
combines the two values into a single contact friction. The
combination rule is selected scene-wide by the **Friction Mode**
setting (Python / TOML key `friction_mode`) under the **Scene
Configuration** panel's **Advanced Params** sub-section:

- **Minimum** (`MIN`, default): take `min(friction_A, friction_B)`.
  The lower-friction surface wins, so a slippery cloth sliding over a
  grippy body behaves as if the whole contact were slippery. To make
  a contact feel grippy, both sides need to be set high.
- **Maximum** (`MAX`): take `max(friction_A, friction_B)`. The
  grippier surface wins, so a single high-friction object acts as a
  brake against everything it touches.
- **Mean** (`MEAN`): take `0.5 * (friction_A + friction_B)`. Each
  object contributes equally regardless of which side is grippier.

The default `MIN` reproduces the behavior of earlier releases and is
the safest choice when you have not set per-object friction values
deliberately.

See [Contact gap: absolute vs ratio](#contact-gap-absolute-vs-ratio) below
for which pair you should be editing, and
[Allow Intersections](#allow-intersections) for the last three rows.

## Spatial Material Maps

What it does: a **Spatial Material Maps** box, drawn on **Shell** and
**Solid** groups just above the **Plasticity** box, holding a list of
maps. Each row varies one material parameter *across the surface* instead
of holding it constant over the whole group. The value at a vertex is a
blend between two numbers you already have in front of you: the group's
own slider, which is what a weight of `0` gives, and the row's **Target**,
which is what a weight of `1` gives. The weights are per vertex, and come
from a vertex group you paint or from a float attribute.

The blend runs slider → target, rather than between a minimum and a
maximum, on purpose: a weight of `0` reproduces the unmapped result
exactly, so unpainted geometry keeps whatever the group was tuned to and a
map can be added to a finished material without re-tuning it.

When to use it: a collar or a waistband that should be stiffer than the
panel it is sewn to, a crease that should take a set while the rest of the
sheet stays elastic, a patch that should grip while the rest of the
surface slides. Leave the box empty — the default on every group — when
one value over the whole group is what you want.

A map belongs to the group, not to an object, so each row is read against
every object assigned to the group and the source is looked up by name on
each of them in turn.

### Which Parameters Can Be Mapped

A map is reduced to one coefficient per element, so the parameter has to
be one this group's elements read. That is what confines the feature to
**Shell** and **Solid**: a **Rod**, a **PDRD** body and a **Sand** cloud
carry no element table to reduce over, and a map on one of those types
is refused at transfer, in a message naming both the parameter and the
type.

| Parameter (row dropdown) | Blends away from                       | Applies to   |
| ------------------------ | -------------------------------------- | ------------ |
| **Young's Modulus**      | **Young's Modulus**                    | Shell, Solid |
| **Friction**             | **Friction**                           | Shell, Solid |
| **Deformation Damping**  | **Deformation Damping**                | Shell, Solid |
| **Plasticity Rate**      | **Theta**, in the **Plasticity** box   | Shell, Solid |
| **Bending Stiffness**    | **Bend Stiffness**                     | Shell        |
| **Bending Damping**      | **Bending Damping**                    | Shell        |
| **Strain Limit**         | **Strain Limit** (a percentage)        | Shell        |
| **Bend Plasticity Rate** | **Bend Theta**, in **Bend Plasticity** | Shell        |
| **Bending (Warp)**       | **Bending Stiffness (Warp)**           | Shell        |
| **Bending (Weft)**       | **Bending Stiffness (Weft)**           | Shell        |
| **Inflation Pressure**   | — (refused; see the note below)        | none         |

The dropdown offers the same eleven entries whatever the group's type, so
the list row is what tells you a parameter is wrong for this group: a row
naming one this type does not read is drawn in alert color, so a
**Bending Stiffness** map on a **Solid** shows red in the list instead of
waiting to fail at **Transfer**.

**Target** is in the same units as the slider it blends away from — a
**Strain Limit** target is a percentage exactly as the field is, and a
**Young's Modulus** target follows the group's **Density-Normalized
(Pa/ρ)** checkbox — and it is held to the same minimum. `0` is a legal
target everywhere except **Young's Modulus**, whose slider stops at
`0.01`, and a target below the floor is refused rather than quietly
clamped.

A **Bending (Warp)** or **Bending (Weft)** map with a positive target
counts as asking for directional bending even when both sliders are left
at `0`, so a group whose mesh carries no UV map raises the same *has no UV
map* warning the sliders raise.

Changing a group's type does not delete its rows. A retyped group keeps
drawing the box even when its new type can use nothing in it, so the rows
stay visible and removable instead of becoming state you cannot reach.

:::{note}
**Inflation Pressure cannot be mapped.** The dropdown keeps the entry so
that saved files naming it still resolve, but a map on it is refused at
transfer. Its per-face potential is translation-invariant only while the
pressure is uniform: measured on a 5 cm sphere with pressure painted from
20 to 100, the per-vertex forces are identical under translation with a
uniform pressure and change by 275% of their peak once the object is moved
1 m, 2202% at 8 m. The same painted map would mean something different
depending on where the object sits in the scene, so it is refused instead
of shipped.
:::

### Adding a Map

Expand **Material Params** and find the **Spatial Material Maps** box. The
`+` beside the list adds a row and selects it; `-` removes the selected
row and leaves the selection on a row that still exists.

Each list row carries, left to right: an **Enable** checkbox, the
parameter dropdown, the source name, a `+N` badge once the row carries
time samples, and the **Target**. Below the list, the selected row
expands into **Source**, **Name** and **Target**. A row with no source
name is drawn in alert color, because an unnamed source cannot be
resolved and would stop the transfer.

**Source** says where the weights are read from:

- **Vertex Group**: what weight paint writes. A vertex the group does not
  contain reads `0` rather than failing, so painting a region is enough
  and everything you left unpainted keeps the group's own value.
- **Attribute**: a scalar attribute on the point domain, read off the
  *evaluated* mesh — which is where a Store Named Attribute node in a
  Geometry Nodes modifier puts one. The evaluated mesh has to have the
  same vertex count as the base mesh, since the weights ship against the
  base mesh's vertex order. An object excluded from evaluation ran no
  modifiers at all, so its attribute cannot be read: **Disable in
  Viewports** (the monitor icon) and excluding the collection from the
  view layer both do this, while **Hide in Viewport** (the eye icon) does
  not.

Weights outside `0`–`1` are clamped into the interval. A weight that is
not a finite number is refused instead, naming the object, the source and
the vertex, because clamping a NaN would land on `0` and read as "use the
group's value" while hiding where it came from.

Nothing in a map row takes a keyframe. The fields are deliberately not
animatable, and an F-curve found on one stops the transfer with a message
pointing at time samples instead. Neither **Copy** / **Paste** nor a
material profile carries map rows either: both move plain scalar fields
only.

### Time Samples

A map can also change over the course of the solve. The row's own source
is the map **at the start frame**; each *time sample* names a different
source, reached at its own frame, and between two consecutive samples the
weights are their linear interpolation. Before the first sample and after
the last, the nearest one holds. A constant hold is therefore two samples
naming one source, which is why there is no hold flag to look for.

The samples list sits under the selected row's detail column, below the
line reminding you that the source above is the map at the start frame.
Its `+` adds a sample at the playhead and copies the row's source type,
then re-sorts the list by frame; `-` removes the selected one. Two samples
cannot share a frame — the second is refused with a warning rather than
raising into the UI — and a sample at or before the start frame is moved
to the frame after it, with a message saying why: the row's own source is
already the weights there.

Time samples are a **Shell** feature. A **Solid** takes a static map only,
because its map lands on the tetrahedra, which carry no per-element
material schedule; a keyed map on one is refused at transfer.

A parameter can be keyframed and mapped at once. The keyframed slider
moves the base the map blends away from, so the map spreads each frame's
value toward the target, and the map's own authored times are voted onto
the scene-wide material keyframe axis so they survive its decimation.

:::{important}
**A sample names a different source; it does not re-read one source at a
different frame.** Every source — the row's own and each sample's — is
read once, while the scene sits at the solve's starting frame. An
attribute that a Geometry Nodes setup varies over time therefore
contributes its start-frame values and nothing else. To move the weights,
author several vertex groups (or several attributes) and name them from
consecutive samples.
:::

### What Happens at Transfer

The weights are read with the scene held at the solve's starting frame,
along with everything else geometry-derived, and shipped per object as one
value per vertex; the reduction to one coefficient per element is an
average of that element's own vertices.

On a **Solid** the weights are painted on the mesh you see, but the
simulated vertices are the tetrahedralized ones, which is what the box's
*Interior values are extended from the painted surface* note is about:
each tet surface vertex takes the weights of the closest Blender triangle,
and each interior vertex the Laplace extension of those surface values.
Both stages are convex combinations of painted values, so the carried
weights stay inside `0`–`1` and every tet ends up between the group's
slider and the map's target.

A row whose **Enable** checkbox is off is skipped entirely. Everything
else is checked, and a map that cannot be resolved stops the transfer with
a message rather than simulating something else. The refusals are:

- a row with no source name, or a name some assigned object does not
  carry;
- a parameter this group's type does not read;
- two rows driving the same parameter, since the blend is base-to-target
  and a second target is a different answer for the same value rather
  than a refinement of it;
- a target that is not a finite number, or one below the mapped slider's
  own minimum;
- a parameter the group has switched off, such as a **Strain Limit** map
  on a group whose **Enable Strain Limit** is unticked. A map cannot
  reintroduce a value that is zero for the whole solve, and the message
  names the condition, which is not always a checkbox: a **Shell** with a
  shrink factor other than `1` closes the **Strain Limit** gate of its
  own;
- a time sample at or before the start frame, two samples on one frame, a
  sample with no source name, or any sample at all on a group that is not
  a **Shell**;
- an assigned object that has gone missing or is not a mesh.

:::{admonition} Under the hood
:class: toggle

The value on an element is `base + (target - base) * w`, with `w` the mean
of that element's own vertices' weights and `base` the group's parameter
for it — that frame's animated value when the slider is keyframed. The
reduction to one coefficient per element happens on the solver side, where
the per-vertex weights the add-on ships are averaged over each element's
own vertices as the parameter tables are assembled: a coefficient varying
*inside* an element would stop the force being the gradient of any energy,
and stop its Hessian being the one the SPD projection was derived for.

The per-element values then replace the replicated scalar in the
per-element parameter tables the solver already reads — triangles for a
shell, tetrahedra for a solid — so a mapped parameter costs no machinery
beyond the tables a uniform one uses.
:::

## Lock Translation and Lock Rotation

What it does: a **Lock Translation** box, drawn below the type-specific
parameters on every dynamic group (**Solid**, **Shell**, **Rod**,
**PDRD**, **Sand**), holding two independent locks. **Lock Translation**
constrains an object's mass-weighted center of mass to a fixed
world-space line through its initial position; **Lock Rotation**
restricts its mass-weighted best-fit rigid rotation to a fixed
world-space axis. Deformation stays free under either, and the two are
separate checkboxes, so either, both, or neither can be on. Both are set
per object rather than per group, since one group can hold several
bodies each on its own axis: the header row carries an object pulldown
that picks which assigned object you are editing, and an eye icon that
previews every enabled lock in the group.

Each lock adds an all-axes escalation and an axis:

- **Lock All Translations** pins the center of mass to its initial point
  instead of letting it slide, and **Lock All Rotations** forbids
  rotation about every axis. Either one makes the axis below it
  meaningless, so that axis is grayed out — not hidden — under a line
  saying it is ignored, and the value you typed comes back when you
  untick the box.
- **Translation Axis** and **Rotation Axis** are world-space directions.
  The encoder normalizes them, so only the direction matters and not the
  magnitude, but a zero axis is refused: the panel warns that the scene
  build will fail until it is non-zero.
- **Prohibit Rotation on Axis** flips what the rotation axis means.
  Unchecked, the axis is the body's only rotational freedom. Checked,
  rotation about that axis is forbidden instead and the perpendicular
  rotation plane stays free.

When to enable: a body that should slide along a rail, a wheel or gear
that should turn on one axle without drifting off it, or a piece that
should go on deforming while its bulk motion stays where you put it.
Leave both off for a fully free body.

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
| **Young's Modulus (Pa/ρ)** | `shell_young_modulus`  | 1000.0         | Young's modulus (see note below). Min 0.01, soft max 10 M (hard max 1e9). |
| **Poisson's Ratio**      | `shell_poisson_ratio`  | 0.35             | Poisson ratio, 0 – 0.4999.                                     |
| **Bend Stiffness**       | `bend`                 | 10.0             | Hinge bending stiffness between neighboring faces. Min 0, soft max 100. **Rod** groups write the same property on their own scale; see [Bend Stiffness on a Rod](#bend-stiffness-on-a-rod). |
| **Bending Stiffness (Warp)** | `bend_warp`        | 0.0              | Extra bending stiffness for the warp (UV X) fibers, added on top of **Bend Stiffness**. `0` keeps bending the same in every direction. Min 0, soft max 1e7. Needs a UV map. |
| **Bending Stiffness (Weft)** | `bend_weft`        | 0.0              | Extra bending stiffness for the weft (UV Y) fibers, added on top of **Bend Stiffness**. `0` keeps bending the same in every direction. Min 0, soft max 1e7. Needs a UV map. |
| **Shrink X**             | `shrink_x`             | 1.0              | Anisotropic warp scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Shrink Y**             | `shrink_y`             | 1.0              | Anisotropic weft scale (min 0.1). < 1 shrinks, > 1 extends.    |
| **Enable Strain Limit**  | `enable_strain_limit`  | `False`          | Turns on non-physical strain clamp (good for stiff cloth).     |
| **Strain Limit**         | `strain_limit_percent` | 5.0              | Max stretch beyond rest length, as a percentage (5.0 = 5%). Active only when **Enable Strain Limit** is on. |
| **Inflate**              | `enable_inflate`       | `False`          | Turns on per-face pressure along face normals.                 |
| **Pressure (Pa)**        | `inflate_pressure`     | 0.0              | Inflation pressure, Pa. Active only when **Inflate** is on.    |
| **Stitch Stiffness**     | `stitch_stiffness`     | 1.0              | Stiffness of loose-edge stitches detected in the mesh.         |

Loose edges (edges not belonging to any face) are automatically treated as
stitch constraints, with stiffness set by **Stitch Stiffness**. The field
is drawn on **Shell** and **Solid** groups, and grayed out with *No
loose-edge stitches in this group* when no assigned mesh has a loose edge.
A **Rod** group has no **Stitch Stiffness** field, because its edges are
the rod itself. A seam between two objects is a merge pair, which carries
its own stiffness; see [Snap and Merge](../constraints/snap_merge.md).

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
the mesh outward. The property's minimum is a hard `0.0`, so there is no
inward (suction) pressure from the panel, from Python, or over MCP. Acts
uniformly over the surface like a balloon or airbag.

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

On a **Solid** group, **Plasticity** cannot be on while one of the
group's pins drives the rest shape from its captured deformation through
[Track Rest-Pose Deformation](../constraints/pins.md#pin-properties-reference),
because both rewrite the rest shape. The pin's panel shows
*Plasticity and rest-pose tracking cannot both be on; Transfer will
refuse*, and **Transfer** stops naming the group and the pin until one
of the two is turned off.

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

What it does: a box near the bottom of the Material Params stack, above
**Lock Translation**, **Rayleigh Damping**, **Stitch Stiffness** and
**Allow Intersections**. It stores a
per-object list of keyframed velocity vectors. Each entry pins that
object to a given `(direction, speed)` at a chosen frame, overriding
the velocity produced by the simulation. The dropdown on the header row
picks which assigned object receives the keyframes; the eye icon toggles
a viewport preview arrow; the copy/paste icons move the keyframe list
between groups.

When to enable: scripted cloth launches (flag unfurling, parachute
drops), matching reference motion on hero shots, or giving the solver a
strong initial push that no constant velocity could time. Leave empty
for fully passive simulations.

The **Direction** and a **Custom Axis** are normalized, so only their
direction matters. With its box ticked, a keyframe whose **Direction**
is `(0, 0, 0)` while its **Speed** is not zero, or whose **Custom Axis**
is `(0, 0, 0)` while its **Angular Speed** is not zero, names no
direction to move or turn in, and **Transfer** refuses it, naming the
object and the frame.

**Speed** and **Angular Speed** are measured against the Blender
animation's own time, so **Time Scale** (see
[Scene Parameters](scene.md)) applies to them like any other keyed
motion: at `0.5` every keyframe's speed is halved, the keyframe that
sets the object's velocity at the **Starting Frame** included. A
**Speed** is a length per second, so it also scales with
[World Scaling](scene.md#world-scaling); an **Angular Speed** does not.

```{figure} ../../images/material_params/velocity_overwrite.png
:alt: Velocity Overwrite section with four keyframes listed and one selected
:width: 500px

The **Velocity Overwrite** section with four keyframes populated
(frames 1, 30, 60, 90). Each row is `frame (speed m/s [direction])`.
The selected row expands into per-keyframe editor rows: **Frame**, then
a translational box gated by **Enable Translational Velocity Overwrite**
(**Direction** XYZ and **Speed**), and — on **Solid**, **Shell** and
**PDRD** groups — an angular box gated by **Enable Angular Velocity
Overwrite** (**Spin Axis**, a **Custom Axis** field when that is chosen,
and **Angular Speed (°/s)**). A gated field is hidden, not grayed, while
its checkbox is off, and **Rod** groups omit the angular box entirely.
The **Cloth** dropdown at the top picks which assigned object the
keyframes belong to, and the `+` / `-` buttons on the right add an
entry at the current timeline frame (a second press on the same frame
is refused with *Frame N already has a keyframe*) or remove the selected
one.
```

## Solid-Specific

| UI label                   | Python / TOML key     | Default              | Description                                               |
| -------------------------- | --------------------- | -------------------- | --------------------------------------------------------- |
| **Model**                  | `solid_model`         | `ARAP`               | Material model. Either `STABLE_NEOHOOKEAN` or `ARAP`.     |
| **Density (kg/m³)**        | `solid_density`       | 100.0                | Volumetric density, kg/m³.                                |
| **Young's Modulus (Pa/ρ)** | `solid_young_modulus` | 500.0                | Young's modulus (see note below). Min 0.01, soft max 10 M (hard max 1e9). |
| **Poisson's Ratio**        | `solid_poisson_ratio` | 0.35                 | Poisson ratio, 0 – 0.4999.                                |
| **Shrink**                 | `shrink`              | 1.0                  | Uniform rest-shape scale (min 0.1).                       |
| **Stitch Stiffness**       | `stitch_stiffness`    | 1.0                  | Stiffness of loose-edge stitches detected in the mesh.    |

A **Solid** mesh's loose edges become stitches the same way a **Shell**'s
do. The solid is rebuilt as tetrahedra from its faces, so each end of such
an edge is placed on the tetrahedralized surface at the point it marks,
and the stitch joins those two surface points. Each end therefore has to
be a vertex of a face: a vertex that sits partway along a loose edge and
belongs to no face is refused at **Transfer**, pinned or not.

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

When **fTetWild** is selected, seven per-object overrides appear. Each row
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
| *(no Model row)*           | `rod_model`         | `ARAP`    | Rods are always ARAP, so the panel draws no Model dropdown. The property still exists and is still written to a material profile. |
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

Directly under that dropdown is a third route, the **From Reference
Geometry** checkbox. Tick it and a **Reference Rest Angle (per object)**
box opens: pick one of the group's objects in the pulldown, tick
**Enable Reference Rest Angle**, then press the eyedropper to take the
active object as its reference. The reference has to be a topological
copy of that object whose vertices were moved — by a modifier, by
geometry nodes, or by hand — and it is checked against the source's
topology both when you pick it and again at **Transfer**, so a mesh
that has drifted is refused with a message instead of silently
encoded. Every object with an enabled reference that still resolves
takes its bending rest angle from that reference, overriding the
group's **Rest Angle** for that object alone; the rest of the group
keeps it.

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

A Sand group is solved at one grain radius, so every object in it has to
have been converted at the same **Grain Radius**. When two were converted
at different radii, the group box shows *Grains were converted at
different radii; Transfer will refuse*, and **Transfer** stops, naming
one object at each radius. Convert them at one radius, or put them in
separate Sand groups.

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
report line after the conversion tells you the number. If none fit at
all, the conversion stops with *No grains fit; reduce the grain radius or
the extra spacing* — but the destructive half has already run: the faces
are gone, the object is left as an empty particle mesh, and the button
then greys out with *Active object is already a particle mesh*, so
retrying at a smaller radius means starting from a copy of the source
mesh.

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
usual axle for a flat gear or disk, and the default).

In the panel, a PDRD group's **Material Params** box carries a
collapsible **Hinge** box. Expand it and you get:

- **Visualize** draws the hinge axle gizmo in the viewport for the
  hinged bodies of this group. On by default.
- An unlabeled object pulldown lists the bodies assigned to the group
  and picks whose hinge the two controls below it edit. A PDRD group
  can hold several bodies (a gear train), so the pulldown is how you
  move between them.
- **Hinge** is the per-object enable. Off by default; ticking it pins
  that body and locks its rotation to one principal axis.
- **Axle** offers **Principal Axis 1**, **2**, and **3**, grayed out
  until **Hinge** is ticked. These are the shown names for `pca_axis`
  `0`, `1` and `2` respectively, so the default **Principal Axis 3** is
  the thinnest-extent axle.

From the Python API a hinge is set per object with `Group.set_hinge`;
from the MCP layer use the `set_pdrd_hinge` tool. Pass `enable=False`
to clear the hinge and let the body move freely again.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

gears = solver.create_group("Gears", type="PDRD")
gears.add("GearA")
gears.set_hinge("GearA", pca_axis=2)   # spin on the thinnest axis
```

The hinge is a per-object property, not a group material attribute, so it
gets its own box inside **Material Params** rather than a row in the
table above.

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

On **Solid**, **Shell**, **PDRD** and **Static** groups the **Use Group
Bounding Box Diagonal** toggle picks between them, and the
**default is ratio-of-bbox-diagonal** because that's what most users
want; you only need to flip to absolute when the group contains
unusually elongated objects (where the diagonal overestimates
characteristic size) or when you need an exact contact thickness for
matching against another group.

Both pairs (**Contact Gap** / **Contact Gap Ratio** and **Contact
Offset** / **Contact Offset Ratio**) are independently controlled by
the same toggle.

A **Rod** group is not offered the toggle: setting a group's type to
**Rod** switches it to absolute mode and seeds **Contact Gap** at
`0.001` and **Contact Offset** at `0.005`, and its panel draws that
absolute pair only. A **Sand** group draws **Contact Gap** alone,
because its grain radius already is the contact offset.

(allow-intersections-settings)=

## Allow Intersections

By default the solver keeps every pair of surfaces apart: nothing passes
through anything, and a scene whose geometry already overlaps is refused
before the first frame is solved. Every group carries four settings that let
chosen objects pass through each other instead. They sit in an **Allow
Intersections** box at the bottom of the group's **Material Params**, and all
four are off by default.

- **Allow Self-Intersections**: an object in this group may pass through
  itself. A falling cloth crosses its own folds instead of piling up on them.
- **Allow Inter-Object Intersections**: an object in this group may pass
  through every other object, including the other objects of this same group
  and **Static** colliders.
- **Allow Inter-Group Intersections**: an object in this group may pass
  through every object assigned to a different group, including **Static**
  colliders, which are always in a group of their own because a group holds
  one type. Objects of this same group still collide with each other.
- **Allow Existing Intersections**: the places where an object in this group
  starts the simulation already overlapping itself, another object, or a
  **Static** collider pass through each other for the whole simulation,
  instead of stopping it before the first frame. Everywhere else keeps full
  contact. See [Allow Existing Intersections](#allow-existing-intersections)
  below.

Each is drawn for every group type, **Static** included, except that a
**Sand** group has no **Allow Existing Intersections** box. On a **Static**
group whether its own boxes take effect depends on how the collider is
driven (see below). With any of the first three on, the panel adds the line
"Allowed pairs pass through, with no contact" beneath the checkboxes; with
**Allow Existing Intersections** on, it adds "Only pairs tangled at the start
pass through".

:::{important}
An allowed pair has no contact at all. The solver applies no contact force
between the two, does not stop them from crossing, and never reports their
overlap as an error. Every pair you did not allow keeps full contact, and the
solver never lets it intersect.

None of these settings affects the
[invisible walls and spheres](../constraints/colliders.md): those always
collide, so a cloth that passes through itself still lands on an invisible
floor.
:::

### Apply to All Objects

Turning a checkbox on opens a box under it with an **Apply to All Objects**
switch and a list of objects. The switch is on by default, and the setting
then reaches every object assigned to the group; the list is grayed out.

Turn the switch off to name the objects yourself. Select them in the
viewport and click the **+** button beside the list (**Add Selected
Objects**); only objects assigned to this group are accepted. The **-**
button (**Remove**) takes out the highlighted entry, and the trash button
(**Remove All**) empties the list. While the switch is off, only the listed
objects get the setting. An empty list reaches no object, and the panel says
"No objects listed; this allowance covers none". When some entries do not
reach the solver, for example an object whose **Include** checkbox is off,
the panel shows how many do, as "2 of 3 entries reach the solver".

The four lists are independent, so one garment can pass through itself
while another only passes through the body.

### Only One Side Has to Allow It

For **Allow Inter-Object Intersections** and **Allow Inter-Group
Intersections**, a pair of objects passes through when **either** of them
has the setting. Turn **Allow Inter-Object Intersections** on for a garment
and it passes through every other object, including the character body it is
fitted to, even though the body's group has the box off. If the garment should
rest on the body, leave that box off on the garment.

**Allow Self-Intersections** is asked of one object only: the object that
passes through itself is the one that needs it.

A **Static** collider's own boxes reach the solver only when the collider is
part of the solved scene: when it is animated, when it uses **Apply Soft
Constraints**, or when it is one end of a cross-stitch. A collider that is
none of those never moves and is a collision surface only, so the boxes on
its group have no effect. A moving object still passes through it when the
moving object's group has **Allow Inter-Object Intersections** or **Allow
Inter-Group Intersections** on. Setting one of those on the moving object's
group works in every case, since it does not depend on how the collider is
driven.

### Which Pairs Each Setting Covers

| Pair                                                           | Self-Intersections | Inter-Object Intersections | Inter-Group Intersections |
| -------------------------------------------------------------- | ------------------ | -------------------------- | ------------------------- |
| One object with itself                                         | Yes                | No                         | No                        |
| Two objects in the same group                                  | No                 | Yes                        | No                        |
| Two objects in different groups, **Static** colliders included | No                 | Yes                        | Yes                       |
| An object and an invisible wall or sphere                      | No                 | No                         | No                        |

**Allow Self-Intersections** says nothing about other objects, and the other
two say nothing about an object folding through itself. A group with only the
first still collides with every other object; a group with only one of the
other two still collides with itself. Turn on each one you need.

**Allow Existing Intersections** has no column here because it names no kind
of pair. It covers whichever pairs start the simulation overlapping, of any
kind in the first three rows, and never the invisible walls and spheres.

"Self" here means one mesh object, not one group. A group holding two meshes
holds two objects, so the pair those two form is an inter-object pair, which
only **Allow Inter-Object Intersections** covers. When a group holds more
than one object and has **Allow Self-Intersections** or **Allow Inter-Group
Intersections** on without **Allow Inter-Object Intersections**, the panel
adds the line "Two objects in one group are an inter-object pair".

That difference is what **Allow Inter-Group Intersections** is for. Put
several garments layered on one another in one group and turn it on there:
the layers keep colliding with each other, while all of them pass through a
body or prop assigned to another group.

### When to Use Them

Use them for geometry that should not collide:

- A mesh that arrives tangled in its pose and should stay that way, such as a
  sleeve folded through its own cuff, or a collar that passes into the
  shoulder. Without an allowance the start pose is refused. When only a few
  places are tangled and the rest of the garment should keep colliding,
  **Allow Existing Intersections** is the narrower choice.
- A cloth whose self-collision is not wanted, where folds may cross each
  other.
- Layers that should pass through a body or a prop rather than rest on it.

An allowance does not separate an overlap. The allowed pair has no contact,
so nothing pushes the two surfaces apart: they stay crossed until the
object's own motion carries them out, if it does. Where the geometry should
end up apart, separate it before the run instead: see
[Mesh Cleaning](../scene/mesh_cleaning.md). With all four settings off, the
solver keeps every pair apart and never lets anything intersect.

Where only a region you have pinned should pass through, the per-pin
[Allow Intersections Here](../constraints/pins.md#allow-intersections-here)
option is narrower than any group setting.

:::{admonition} Under the hood
:class: toggle

The check at scene build and the checks the solver runs while it steps all
consult the allowance, so a scene that builds is not then stopped by the
first step. It is evaluated per pair of elements, whether that is two
triangles, a rod segment against a triangle or against another rod segment,
or two grains of a Sand cloud, and it decides whether that pair takes part in
contact at all. An allowed pair is left out of the contact force, out of the
continuous-collision test that limits how far each step may move, and out of
both intersection checks. Every other pair keeps all of them, so the guarantee
that nothing intersects holds for every pair you did not allow.

**Allow Inter-Group Intersections** compares the groups the two objects are
assigned to. A **Static** collider that is not part of the solved scene counts
as a different group from every object.

The start-pose check that rejects elements closer than their contact offset
skips allowed pairs as well, since an allowed pair has no contact offset to
keep. Every other pair is still checked. The other geometry checks, such as a
rod segment shorter than its own contact offset, are unrelated to these
settings and still apply.
:::

(allow-existing-intersections-settings)=
### Allow Existing Intersections

A garment posed by a rig often arrives with a fold that passes through itself,
such as an armpit, or with a sleeve that dips into the body. The scene is then
refused before the first frame. The three settings above let it run only by
giving up contact for the whole object, everywhere. **Allow Existing
Intersections** gives up contact only where the overlap already is.

When it is on, the build finds every place where an object of this group
starts the simulation overlapping something, or sitting closer to it than the
[contact offset](#contact-gap-and-contact-offset) allows, and lets exactly those
places pass through each other for the whole simulation, together with the
faces right next to them. Everything else keeps full contact: the rest of the
garment still collides with itself, with the body, and with every other
object. As with the other settings, only one of the two sides needs it on,
and it covers an object with itself, two objects, and an object against a
**Static** collider. It does not affect the invisible walls and spheres, and it
is not available on **Sand**.

:::{important}
This setting tolerates a tangle, it does not untangle one. The overlapping
places are left alone for the whole run, so a fold that starts crossed stays
crossed unless the motion carries it out. Nothing new is allowed to pass
through: a pair that did not start overlapping never intersects, and if one
ever would, the run stops with an intersection error. Where the start pose
can be fixed, fixing it is still the better choice.
:::

After a build with this setting on, the viewport shows what was let through.
On the frame the simulation starts from, the overlapping places are tinted
light blue, with a label such as "19 Existing Intersections Allowed". The
highlight is drawn on the starting frame only, because it marks where the
geometry was at the start. When nothing overlaps, nothing is drawn.

**Apply to All Objects** narrows it to named objects in the same way as the
other settings. Leaving the list empty reaches no object, so the tangled
start pose is refused again.

:::{admonition} Under the hood
:class: toggle

The overlapping places are found once, by the same check that would
otherwise refuse the scene, on the start pose. For every pair of elements that
check finds, and where either side's object has the setting on, every vertex
of one element is linked to every vertex of the other. During the solve, two
elements whose vertices are linked are treated like two faces that share a
corner: no contact force, no limit on the step from the collision test, and no
intersection report. Because the link is on vertices, a face sharing a corner
with an overlapping face is covered too. The links are fixed when the scene is
built and last for the whole run.

The solver never adds a link of its own. The start-pose check and the solver's
own check measure positions at slightly different precision, so a pair that
barely touches can pass one and fail the other. The run then stops before the
first frame with an error that names the pair; moving the two apart, or
further into each other, makes both checks agree.
:::

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

The same workflow is available from Python. Most fields in the
**Material Params** box are reachable through each group's `.param`
attribute, which is whitelisted: the accepted names are listed in the
[Blender Python API Reference](../../integrations/python_api_reference.rst),
and anything outside that list raises `AttributeError`. Changes from
Python appear in the panel immediately and vice versa.

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

# Static collider: friction and contact settings.
floor = solver.create_group("Floor", "STATIC")
floor.param.friction        = 0.8
floor.param.contact_gap_rat = 0.001
```

**Apply Soft Constraints** and its **Stiffness** are among the fields
off the whitelist, so a collider whose own shape closes onto the cloth is
held with springs from the **Material Params** box rather than from
Python.

:::{admonition} Under the hood
:class: toggle

**Loose-edge stitch encoding**

At transfer time, edges on **Shell** and **Solid** meshes that are not
adjacent to any face are automatically emitted as stitch constraints
with stiffness set by `stitch_stiffness`. On a **Solid**, each end of
such an edge is projected onto the tetrahedralized surface as a
barycentric point on a surface triangle, and the row becomes a
point-to-point stitch; an end that finds no triangle stops the build,
naming the object. There is no UI surface for this; it happens on every
transfer.

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
