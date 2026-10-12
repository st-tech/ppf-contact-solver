# 🧪 Material Parameters

Every object group carries its own material parameters. The group's **Type**
decides which rows its **Material Params** box draws, so a **Shell** group
shows cloth rows and a **Static** group shows friction, soft-constraint,
contact and intersection rows. [Rows by Group Type](#rows-by-group-type)
lists every row, the types that draw it and the section that describes it.

This page covers, in this order: the box itself (presets, padlocks,
keyframes, copy and paste), the rows that several types share, spatial
material maps, Lock Translation and Lock Rotation, Rayleigh damping, the
density-normalized Young's modulus, strain limit, plasticity and velocity
overwrite, the rows that belong to a single type, the contact gap and offset
with the Local Gap Limit, the Allow Intersections settings, loose-edge
stitches, material profiles and the Blender Python API. The rows that belong
to a single type are under [Shell-Specific](#shell-specific),
[Solid-Specific](#solid-specific), [Rod-Specific](#rod-specific),
[Sand-Specific](#sand-specific), [PDRD-Specific](#pdrd-specific) and
[Static-Specific](#static-specific). For Static groups, including how to
animate them, see [Static Objects](../scene/static_objects.md).

```{figure} ../../images/material_params/group_type_popdown.png
:alt: The group-type dropdown, popped open. Four full-width buttons are stacked under a Type label: Solid (highlighted as the current selection), Shell, Rod and Static.
:width: 360px

The group-type dropdown, next to the group's **Name** field. It lists
**Solid**, **Shell**, **Rod**, **Static**, **PDRD** and **Sand**; the
screenshot shows the first four. Picking a **Type** changes the rows of the
**Material Params** box (see [Rows by Group Type](#rows-by-group-type)).
```

## The Material Params Box

At the bottom of each group card in the **Dynamics Groups** panel is the
collapsible **Material Params** box. The arrow at the left of its header
shows and hides the value rows. The **Copy Material Params** and
**Paste Material Params** icons sit at the right end of the header. Below the
header are the **Preset** row (**Solid** and **Shell** groups only) and the
profile row. Both stay visible while the box is collapsed.

Changing a group's **Type** changes the rows at once. Switching a group to
**Rod** also turns off **Use Object Bounding Box Diagonal**, sets **Contact
Gap** to `0.001` and **Contact Offset** to `0.005`, and sets **Bend
Stiffness** to `1.0`.

Switching a group away from **Rod** does not restore those values: **Use
Object Bounding Box Diagonal** stays off, **Contact Offset** stays at `0.005`
and **Bend Stiffness** stays at `1.0`, where a new **Shell** group starts at
`10.0`. Set them again for the new type. Any other change of **Type** only
resets the overlay color to the color of that type and changes which rows are
drawn. A row the new type does not draw keeps its value.

### Rows by Group Type

A row that a type does not read is not drawn for that type. The *contact
box* is the unlabeled box that holds the contact gap and offset rows, with
**Local Gap Limit** as its last row on the types that draw it.

| Row                                                              | Drawn on                              | Details                                                                                                                                                                                        |
| ---------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model**                                                        | Solid, Shell                          | Solid: **Stable NeoHookean** or **ARAP**. Shell: **Baraff-Witkin** or **ARAP**. A Rod always uses ARAP and has no **Model** row.                                                              |
| **Density**                                                      | Solid, Shell, Rod, PDRD               | kg/m³ on Solid and PDRD, kg/m² on Shell, kg/m on Rod.                                                                                                                                          |
| **Young's Modulus** and **Density-Normalized (Pa/ρ)**            | Solid, Shell, Rod                     | The field label reads **(Pa/ρ)** while the checkbox is on and **(Pa)** while it is off. See [Density-Normalized (Pa/ρ)](#density-normalized-paρ).                                              |
| **Poisson's Ratio**                                              | Solid, Shell                          | The slider stops at 0.4999. The solver needs a value above 0 and below 0.5, and at `0` it stops with *Poisson's ratio must be positive*.                                                                                                                    |
| **Friction**                                                     | Solid, Shell, Rod, PDRD, Static, Sand | A Sand group has its own **Friction** row. See [Shared Parameters](#shared-parameters).                                                                                                        |
| **Contact Gap** and **Contact Offset**, or their ratios          | All types                             | Solid, Shell, PDRD and Static have a **Use Object Bounding Box Diagonal** checkbox. Rod uses the absolute pair only. Sand draws **Contact Gap** alone, because its grain radius is the contact offset. See [Contact Gap and Contact Offset](#contact-gap-and-contact-offset). |
| **Local Gap Limit**                                              | Solid, Shell, Rod                     | Drawn in the contact box. See [Local Gap Limit](#local-gap-limit).                                                                                                                             |
| **Collision Active Duration Windows**                            | Solid, Shell, Rod, PDRD               | Off by default. See [Active collision windows](../scene/object_groups.md#active-collision-windows).                                                                                            |
| **Bend Stiffness**                                               | Shell, Rod                            | A Shell also draws **Bending Stiffness (Warp)** and **Bending Stiffness (Weft)** right below it. A Rod draws its field in a **Bend** box below the contact rows; see [Bend Stiffness on a Rod](#bend-stiffness-on-a-rod). |
| **Shrink**                                                       | Solid, Shell, Rod                     | Solid: one **Shrink** field. Shell: **Shrink X** and **Shrink Y**. Rod: **Shrink** just under **Friction**.                                                                                   |
| **Enable Strain Limit** and **Strain Limit**                     | Shell, Rod                            | See [Strain Limit](#strain-limit).                                                                                                                                                             |
| **Inflate** and **Pressure (Pa)**                                | Shell                                 | See [Inflate](#inflate).                                                                                                                                                                       |
| **Spatial Material Maps**                                        | Solid, Shell, Rod                     | See [Spatial Material Maps](#spatial-material-maps).                                                                                                                                           |
| **Plasticity**                                                   | Solid, Shell                          | See [Plasticity](#plasticity).                                                                                                                                                                 |
| **Rest Angle**, **From Reference Geometry** and **Bend Plasticity** | Shell, Rod                         | Shell: one box below **Plasticity**. Rod: **Rest Angle** and **From Reference Geometry** sit in the **Bend** box, and **Bend Plasticity** has its own box.                                     |
| **Velocity Overwrite**                                           | Solid, Shell, Rod, PDRD               | Titled **Velocity (free launch)** on PDRD. Rod has no spin controls. See [Velocity Overwrite](#velocity-overwrite).                                                                            |
| **Lock Translation** box                                         | Solid, Shell, Rod, PDRD, Sand         | See [Lock Translation and Lock Rotation](#lock-translation-and-lock-rotation).                                                                                                                 |
| **Tetrahedralizer**                                              | Solid                                 | See [Tetrahedralizer (per object)](#tetrahedralizer-per-object).                                                                                                                               |
| **Rayleigh Damping**                                             | Solid, Shell, Rod                     | See [Rayleigh Damping](#rayleigh-damping).                                                                                                                                                     |
| **Stitch Stiffness**                                             | Solid, Shell                          | See [Shell-Specific](#shell-specific) and [Solid-Specific](#solid-specific).                                                                                                                   |
| **Hinge**                                                        | PDRD                                  | See [Hinge Joints](#hinge-joints).                                                                                                                                                             |
| **Grain Radius (m)** and **Particle Mass (g)**                   | Sand                                  | See [Sand-Specific](#sand-specific).                                                                                                                                                           |
| **Apply Soft Constraints**                                       | Static                                | See [Apply Soft Constraints](#apply-soft-constraints).                                                                                                                                         |
| **Allow Intersections** box                                      | All types                             | A Sand group has no **Allow Existing Intersections** checkbox. See [Allow Intersections](#allow-intersections).                                                                                |
| **Force Field Weight**                                           | All types except Static               | See [Per-Group Weight](force_fields.md#per-group-weight).                                                                                                                                      |

The **Model** dropdown of a **Shell** group offers only **Baraff-Witkin** and
**ARAP**. A `.blend` or material profile can carry **Stable NeoHookean** on a
**Shell** group. The **Model** row then draws blank, with
*Stable NeoHookean is not offered for shells; Transfer will refuse* under it,
and **Transfer** refuses the group until you choose **ARAP** or
**Baraff-Witkin**. A **Solid** group offers **Stable NeoHookean**.

```{figure} ../../images/material_params/box_shell.png
:alt: The Material Params box expanded on a Shell group, as one tall column. The Preset and Open Profile rows are at the top, followed by the material rows, the contact box with a Local Gap Limit slider at 25 percent, the bending, shrink, strain limit, inflate, map, plasticity, velocity and lock boxes, Rayleigh Damping, Stitch Stiffness, the Allow Intersections box and Force Field Weight at the bottom.
:width: 500px

The **Material Params** box expanded on a **Shell** group. The contact box,
about a quarter of the way down, holds the gap and offset pair and the
**Local Gap Limit** slider. Other types draw fewer rows; see
[Rows by Group Type](#rows-by-group-type).
```

### Preset

The **Preset** dropdown applies a bundled material to the group in one
click. A **Shell** group lists six fabrics and a **Solid** group lists five
soft solids; [Fabric Presets](fabric_presets.md) lists them and the values
each one sets. The other types have no presets and no **Preset** row.

The dropdown returns to **Select Preset...** after it applies a preset, so it
never shows which preset you used last. A preset sets only the rows it names,
turns on **Density-Normalized (Pa/ρ)** and never changes the group's
**Type**. A row with a closed padlock keeps its value.

### Padlocks

Most numeric rows (**Friction**, the densities, the stiffnesses, the contact
distances, the damping values and so on) carry a padlock at their right end.
A closed padlock keeps that row's value when you apply a **Preset** or use
**Paste Material Params**. The value field is grayed out while the padlock is
closed, and the padlock and the keyframe control stay usable.

A padlock guards the value only against those two actions. It does not mute
animation: a locked row that has keyframes still follows its curve during the
solve.

A material profile ignores padlocks and overwrites every row it stores.
Toggles and dropdowns (**Model**, **Enable Strain Limit** and so on) have no
padlock. Neither do **Local Gap Limit**, **Force Field Weight**, the
soft-constraint **Stiffness**, or the fields inside the map, lock, velocity
and hinge boxes.

### Keyframes

These rows offer Blender's keyframe control (the dot or diamond at the right
end of the row), whatever the group's **Type**:

- **Young's Modulus**, **Poisson's Ratio** and **Friction**.
- **Bend Stiffness**, **Bending Stiffness (Warp)** and **Bending Stiffness
  (Weft)**.
- **Deformation Damping** and **Bending Damping**.
- **Contact Gap** and **Contact Offset**, and their ratio rows.
- **Strain Limit** and **Pressure (Pa)**.
- **Theta** and **Threshold** in the **Plasticity** box, and **Bend Theta**
  and **Bend Threshold** in the **Bend Plasticity** box.

Which keyframes **Transfer** accepts depends on the group's **Type**. It
refuses every other keyframe and names the property. The accepted ones are:

- **Shell**: every row in the list above.
- **Solid**: **Friction**, **Contact Gap** and **Contact Offset** (or their
  ratio rows), which belong to the solid's surface. The elastic values of a
  solid are fixed for the whole solve, so a keyframe on **Young's Modulus**,
  **Poisson's Ratio**, the **Plasticity** **Theta** and **Threshold**, or
  **Deformation Damping** is refused.
- **PDRD**: **Friction**, **Contact Gap** and **Contact Offset**.
- **Static**: **Contact Gap** and **Contact Offset**, only on a collider that
  is part of the solved scene: one that is animated, uses **Apply Soft
  Constraints** or is one end of a cross-stitch (see
  [Only One Side Has to Allow It](#only-one-side-has-to-allow-it)). On a
  collider that never moves, the scene build stops and names the key.
- **Rod** and **Sand**: none.

Remove a refused keyframe, or split the change into separate solves.

The densities, the shrink factors, **Particle Mass**, **Friction** on a
**Sand** group and **Stitch Stiffness** have a padlock and no keyframe
control. Every group setting outside the list above is read once, at the
solve's **Starting Frame**, and offers no keyframe. **Transfer** stops with a
message naming the data path when a saved file still carries a keyframe on
one of those settings, and when a driver or an NLA strip sits on any group
setting, keyframeable ones included. See
[Dynamic Parameters](dynamic.md#supported-parameters).

### Profile Buttons: Open / Clear / Reload / Save

A **material profile** is a TOML file of named entries, each holding the
material values of a group. Use one to reuse a tuned material across groups
and scenes. The profile row sits below the **Material Params** header (below
the **Preset** row on **Solid** and **Shell** groups) and stays visible while
the box is collapsed. It works like the profile row described in
[Connection Profiles](../../connections/profiles.md#the-profile-row).

With no profile loaded, the row shows an **Open Profile** button and the
**Save Material Profile** icon. With a profile loaded, it shows the **Profile**
dropdown followed by four icon buttons. The buttons show icons only, so hover
over one to read its name.

```{figure} ../../images/material_params/profile_copy_paste_row.png
:alt: Material Params header with the Copy Material Params and Paste Material Params icons at its right end. Below it, the profile row holds a full-width Open Profile button and a Save Material Profile icon.
:width: 500px

**No profile loaded.** The profile row holds **Open Profile** and the
**Save Material Profile** icon. The **Copy Material Params** and
**Paste Material Params** icons sit at the right end of the header.
```

```{figure} ../../images/material_params/profile_row_loaded.png
:alt: Material Params profile row after loading a TOML. The Profile dropdown is set to Cotton and four icons follow it: folder (Open Material Profile), X (Clear Material Profile), refresh (Reload Material Profile), disk (Save Material Profile).
:width: 500px

**Profile loaded.** The **Profile** dropdown (here set to `Cotton`, an entry
of the loaded file) lists every entry in the file. The four icons to its
right are the buttons in the table below, left to right.
```

| Icon            | Button                      | Effect |
| --------------- | --------------------------- | ------ |
| Folder          | **Open Material Profile**   | Pick a `.toml` file. The **Profile** dropdown lists the file's entries, sorted by name, and the first entry is applied to the group at once, replacing its current values. Choosing another entry in the dropdown applies that entry. |
| X               | **Clear Material Profile**  | Forget the loaded file and return the row to **Open Profile**. The group keeps its current values. |
| Circular arrows | **Reload Material Profile** | Read the file again from disk and apply the entry selected in the **Profile** dropdown. Use it after editing the file. |
| Floppy disk     | **Save Material Profile**   | Write the group's current values to the file. With a profile loaded, it overwrites the entry selected in the **Profile** dropdown and asks for nothing. With no profile loaded, it opens a file picker with an **Entry Name** field (default `Default`), creates the file or adds the entry to an existing file, and loads the file in the dropdown. The file's other entries stay, and an entry of the same name is replaced. |

To add a new entry to a file you have loaded, click **Clear Material
Profile**, click **Save Material Profile**, choose the file and type a new
**Entry Name**.

```{figure} ../../images/material_params/save_icon.png
:alt: Dynamics Groups panel of a Solid group with the Material Params header collapsed. Below the header, the Open Profile button has the floppy-disk Save Material Profile icon at its right end, outlined in red.
:width: 500px

The **Save Material Profile** icon (floppy disk, outlined in red) at the
right end of the profile row.
```

:::{important}
**Create profile files with the Save Material Profile icon, not by hand.**
Set the group's values in the panel and click it, and the add-on writes the
file. [Material Profiles](#material-profiles) shows the file format.

Applying a profile also restores the group's **Type** and overlay color, so
an entry saved from a **Shell** group turns the group it is applied to into a
**Shell** and swaps its rows. A profile stores group-level material values
only. It does not store **Rayleigh Damping**, the two directional bending
rows, the **Allow Intersections** settings, **Force Field Weight**, a
**Rod**'s **Shrink**, the **PDRD** and **Sand** rows, the **Spatial Material
Maps** rows, or any per-object setting (velocity keyframes, locks, hinge,
tetrahedralizer overrides, collision window ranges). Set those on the group
after applying a profile.
:::

### Copy / Paste

The **Copy Material Params** and **Paste Material Params** icons at the right
end of the **Material Params** header move material values between groups
without a file. **Copy Material Params** stores the group's material values on
an internal clipboard, and **Paste Material Params** applies the clipboard to
another group.

- **Copy Material Params** takes every material value, including the **Allow
  Intersections** checkboxes. It does not take the group's name, **Type**,
  overlay color, padlocks, profile or preset selection, the **Apply to All
  Objects** switches, or any list (map rows, object lists of the allowances,
  per-object settings).
- **Paste Material Params** skips every row with a closed padlock and reports
  how many locked values the group holds, for example *Material params pasted;
  2 locked value(s) kept*.
- **Paste Material Params** takes the type-specific rows (density, Young's
  Modulus, Poisson's Ratio and so on) only from the copied group's own
  **Type**. Pasting a **Shell** onto a **Solid** carries the Shell's density
  and moduli, which the Solid does not use, and leaves the Solid's own density
  and moduli alone.
- **Paste Material Params** is disabled until you have made a **Copy Material
  Params**.
- The clipboard lasts for the current Blender session only.

Over MCP, `copy_material_parameters(group_uuid)` does what **Copy Material
Params** does and `paste_material_parameters(group_uuid)` does what **Paste
Material Params** does. A paste is refused until a copy has been made in the
same Blender session, and its result lists the rows a closed padlock kept under
`kept_locked`.

### Per-Object Pulldowns

Several boxes edit one object at a time through a pulldown: **Velocity
Overwrite**, **Collision Active Duration Windows**, **Lock Translation**,
**Hinge**, **Reference Rest Angle** and **Tetrahedralizer**. A pulldown lists
only the objects whose **Include** checkbox is on, the checkbox at the left of
the object's row in the group's object list (see
[Object Groups](../scene/object_groups.md)). It reads **None** while no object
of the group is included. Tick **Include** on an object to edit its settings.

## Shared Parameters

These rows are drawn on several types. The **Drawn on** column lists the
types that show each row.

| UI label                             | Property key                      | Default | Drawn on                                   | Description                                                                                                                                         |
| ------------------------------------ | --------------------------------- | ------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Friction**                         | `friction`                        | 0.5     | Solid, Shell, Rod, PDRD, Static            | Coulomb friction coefficient at contacts (0 – 1). A **Sand** group has its own **Friction** row (`sand_friction`, default 0.0).                     |
| **Contact Gap**                      | `contact_gap`                     | 0.001   | All types                                  | Absolute contact gap, in Blender units (0.001 – 0.01). Drawn while **Use Object Bounding Box Diagonal** is off, and always on **Rod** and **Sand**.   |
| **Contact Offset**                   | `contact_offset`                  | 0.0     | Solid, Shell, Rod, PDRD, Static            | Absolute contact offset, in Blender units (0 – 0.03). Drawn while **Use Object Bounding Box Diagonal** is off, and always on **Rod**.                |
| **Use Object Bounding Box Diagonal** | `use_group_bounding_box_diagonal` | `True`  | Solid, Shell, PDRD, Static                 | When on, the contact distances are ratios of each object's own bounding-box diagonal.                                                               |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | 0.001   | Solid, Shell, PDRD, Static                 | Contact gap as a fraction of each object's own bounding-box diagonal (0.00001 – 1). Drawn while the checkbox above is on.                           |
| **Contact Offset Ratio**             | `contact_offset_rat`              | 0.0     | Solid, Shell, PDRD, Static                 | Contact offset as a fraction of each object's own bounding-box diagonal (0 – 1). Drawn while the checkbox above is on.                              |
| **Local Gap Limit**                  | `contact_gap_limit_percent`       | 25      | Solid, Shell, Rod                          | Largest share of the room between mesh neighbors the contact gap may take, in percent (1 – 50). It has no keyframe and no padlock.                  |
| **Allow Self-Intersections**         | `allow_self_intersection`         | `False` | All types                                  | Let an object pass through itself, with no contact between its parts.                                                                               |
| **Allow Inter-Object Intersections** | `allow_inter_object_intersection` | `False` | All types                                  | Let an object pass through every other object, with no contact.                                                                                     |
| **Allow Inter-Group Intersections**  | `allow_inter_group_intersection`  | `False` | All types                                  | Let an object pass through objects of other groups, with no contact.                                                                                |
| **Allow Existing Intersections**     | `allow_existing_intersection`     | `False` | All types except Sand                      | Let the pairs that already overlap at the start pass through, with no contact.                                                                      |
| **Force Field Weight**               | `force_field_weight`              | 1.0     | All types except Static                    | Scale on the scene's force fields for this group's objects (0 or more; the slider stops at 2). `0` leaves the group unaffected.                     |

A **Rod** group starts with **Contact Offset** at `0.005`, because switching
a group to **Rod** sets it.

The keys are the add-on's property names. The Blender Python API takes every
key in the table through a group's `.param` attribute except the four
**Allow** settings, which you set in the panel or with the MCP tool
`set_group_material_properties`. A material profile stores the friction and
contact rows, **Local Gap Limit** included, and not the **Allow** settings or
**Force Field Weight**.

See [Contact Gap: Absolute vs Ratio](#contact-gap-absolute-vs-ratio) for which
pair to edit, [Local Gap Limit](#local-gap-limit) for the cap on the gap, and
[Allow Intersections](#allow-intersections) for the last four **Allow**
rows.

### Friction at a Contact

Each object carries its own **Friction** coefficient. When two objects touch,
the solver combines the two values into one contact friction. The scene-wide
**Friction Mode** setting (key `friction_mode`), in the **Advanced Params**
section of the **Scene Configuration** panel, selects the rule:

- **Minimum** (`MIN`, the default): `min(friction_A, friction_B)`. The lower
  coefficient is used, so a low-friction cloth sliding over a high-friction
  body gets low friction at the contact. Both sides need a high value for a
  high-friction contact.
- **Maximum** (`MAX`): `max(friction_A, friction_B)`. The higher coefficient
  is used, so one high-friction object raises the friction of everything it
  touches.
- **Mean** (`MEAN`): `0.5 * (friction_A + friction_B)`.

## Spatial Material Maps

A **Spatial Material Maps** box holds a list of maps. Each map varies one
material parameter across the surface, or along a strand, instead of holding
it constant over the whole group. Use it for a collar or waistband that
should be stiffer than the panel it is sewn to, a crease that should deform
permanently while the rest of the sheet stays elastic, a patch with higher
friction than the rest of the surface, or hair that should be stiff at the
root and limp at the tip. Leave the list empty, which is the default, when one
value over the whole group is what you want.

The box is drawn on **Shell** and **Solid** groups, just above the
**Plasticity** box, and on **Rod** groups, just above the **Bend** box. The
value at a vertex is a blend between two numbers: the
group's own slider, which a weight of `0` gives, and the row's **Target**,
which a weight of `1` gives. The weights are per vertex and come from a
vertex group you paint or from a float attribute. A weight of `0`
reproduces the unmapped result, so unpainted geometry keeps the value the
group was tuned to, and you can add a map to a finished material without
re-tuning it.

A map belongs to the group, not to an object. Each row is read against every
object of the group whose **Include** checkbox is on, and each of those
objects needs a source with the row's name.

### Which Parameters Can Be Mapped

Maps work on **Shell**, **Solid** and **Rod** groups. A **PDRD**, **Static**
or **Sand** group refuses an enabled map at **Transfer**, with a message naming
the parameter and the **Type**.

| Parameter (row dropdown) | Key (MCP)             | Value at weight 0                      | Applies to        |
| ------------------------ | --------------------- | -------------------------------------- | ----------------- |
| **Young's Modulus**      | `young-mod`           | **Young's Modulus**                    | Shell, Solid, Rod |
| **Friction**             | `friction`            | **Friction**                           | Shell, Solid, Rod |
| **Deformation Damping**  | `deformation-damping` | **Deformation Damping**                | Shell, Solid, Rod |
| **Plasticity Rate**      | `plasticity`          | **Theta**, in the **Plasticity** box   | Shell, Solid      |
| **Bending Stiffness**    | `bend`                | **Bend Stiffness**                     | Shell, Rod        |
| **Bending Damping**      | `bending-damping`     | **Bending Damping**                    | Shell, Rod        |
| **Strain Limit**         | `strain-limit`        | **Strain Limit** (a percentage)        | Shell, Rod        |
| **Bend Plasticity Rate** | `bend-plasticity`     | **Bend Theta**, in **Bend Plasticity** | Shell, Rod        |
| **Bending (Warp)**       | `bend-warp`           | **Bending Stiffness (Warp)**           | Shell             |
| **Bending (Weft)**       | `bend-weft`           | **Bending Stiffness (Weft)**           | Shell             |
| **Inflation Pressure**   | `pressure`            | n/a                                    | none              |

The dropdown lists the same eleven entries whatever the group's **Type**. A
row naming a parameter that the group's **Type** does not read is drawn in
alert color, so a **Bending Stiffness** map on a **Solid** shows red in the
list before **Transfer** refuses it.

**Target** uses the same unit as the slider it blends from. A **Strain
Limit** target is a percentage, as the field is, and a **Young's Modulus**
target follows the group's **Density-Normalized (Pa/ρ)** checkbox. A target
below the slider's minimum is refused, not clamped. `0` is a valid target for
every parameter except **Young's Modulus**, whose minimum is `0.01`.

A **Bending (Warp)** or **Bending (Weft)** map with a positive target counts
as asking for directional bending even when both sliders are `0`. A group
whose mesh has no UV map then shows the same *has no UV map* warning, naming
the object, that the sliders show.

Changing a group's **Type** does not delete its map rows. A group whose
**Type** can use none of them keeps drawing the box, so the rows stay
visible and removable.

:::{note}
**Inflation Pressure cannot be mapped.** The dropdown lists it, but
**Transfer** refuses a map on it, because the effect of a pressure that
varies over the surface would depend on where the object sits in the scene.
Use the uniform **Pressure (Pa)** field.
:::

### Adding a Map

1. Paint a vertex group, or write a float attribute, on each object of the
   group (see **Source** below).
2. Expand **Material Params** and find the **Spatial Material Maps** box.
3. Click the **+** button (**Add Material Map**) beside the list. It adds a row
   and selects it. A new row starts with **Parameter** set to **Young's
   Modulus**, **Source** set to **Vertex Group**, an empty **Name**, **Target**
   `0` and **Enable** ticked. On a group that holds only Hair Curves objects,
   **Source** starts on **Attribute**, the one source hair can use.
4. Pick the parameter in the row's dropdown.
5. Type the vertex group or attribute name in **Name**. A row with no name is
   drawn in alert color, because an unnamed source stops **Transfer**.
6. Set **Target**. A **Young's Modulus** row that keeps its starting **Target**
   of `0` is refused, because `0` is below that slider's minimum.

Click the **-** button (**Remove Material Map**) to remove the selected row.
The selection moves to a row that still exists.

Each list row shows, left to right: an **Enable** checkbox, the parameter
dropdown, the source name, a `+N` badge once the row has map samples, and
the **Target**. Below the list, the selected row expands into **Source**,
**Name** and **Target**.

**Source** says where the weights are read from:

- **Vertex Group**: what weight paint writes. A vertex that the group does
  not contain reads `0`, so painting a region is enough and everything left
  unpainted keeps the group's own value.
- **Attribute**: a scalar attribute on the point domain, read from the
  *evaluated* mesh, which is where a Store Named Attribute node in a
  Geometry Nodes modifier puts one. The evaluated mesh must have the same
  vertex count as the base mesh. An object excluded from evaluation runs no
  modifiers, so its attribute cannot be read. **Disable in Viewports** (the
  monitor icon) and excluding the collection from the view layer both do
  this, while **Hide in Viewport** (the eye icon) does not.

On a **Hair Curves** object, which has no vertex groups, **Source** has to be
**Attribute**. The attribute is a Float, an Integer or a Boolean on the hair's
points, and it is read from the hair as the solver receives it: an attribute
of the groom itself, or one written by a Geometry Nodes modifier that sits
above the first modifier that generates curves. A **Vertex Group** source on
hair stops **Transfer** with a message that says to switch it. See
[Stiffness Along a Strand](../scene/hair_curves.md#stiffness-along-a-strand).

Weights outside `0`–`1` are clamped into that interval. A weight that is not
a finite number is refused, and the message names the object, the source and
the vertex.

A map row takes no keyframe. A keyframe found on one of its fields stops
**Transfer** with a message that points to map samples instead. Neither the
**Copy Material Params** and **Paste Material Params** icons nor a material
profile carries map rows.

### Map Samples

A map can change over the solve. The row's own source is the map at the
start frame. Each *map sample* names a different source that applies at its
own frame, and the weights between two consecutive entries are their linear
interpolation. After the last sample, its source holds. To hold one source
over an interval, name the same source in the two consecutive entries that
bound the interval. The row's own source counts as the entry at the start
frame.

The sample list sits below the selected row's fields, under the line *The
source above is the map at the start frame*. Each list row reads **Frame N**
beside a source-name field, and the selected sample shows **Frame**,
**Source** and **Name** fields.

- The **+** button (**Add Map Sample**) adds a sample at the playhead, copies
  the row's **Source** type and sorts the list by frame. A sample added at or
  before the start frame is placed on the frame after it, with a message saying
  why.
- The **-** button (**Remove Map Sample**) removes the selected sample.
- Two samples cannot share a frame: the second is refused with a warning.

Map samples are a **Shell** feature. A **Solid** or a **Rod** takes a static
map only, and **Transfer** refuses a keyed map on one.

You can keyframe a parameter and map it at once. The keyframed slider sets
the value the map blends from at each frame, and the map's sample frames are
honored alongside the keyframes.

:::{important}
**A sample names a different source. It does not re-read one source at a
different frame.** Every source, the row's own and each sample's, is read
once, while the scene sits at the solve's starting frame. An attribute that a
Geometry Nodes setup varies over time therefore contributes its start-frame
values and nothing else. To move the weights, author several vertex groups
(or several attributes) and name them from consecutive samples.
:::

### What Happens at Transfer

**Transfer** reads every source with the scene held at the solve's
**Starting Frame**, along with the rest of the geometry data, and ships one
weight per vertex. Each triangle (**Shell**), tetrahedron (**Solid**) or
strand segment (**Rod**) uses the average weight of its own vertices.

On a **Solid** the weights are painted on the mesh you see, and the interior
of the solid takes values extended from them. That is what the box's note
*Interior values are extended from the painted surface* refers to. Every
weight stays inside `0`–`1`, so every tetrahedron ends up between the group's
slider and the map's target.

A row whose **Enable** checkbox is off is skipped. **Transfer** checks
everything else, and a map that cannot be resolved stops it with a message.
The refusals are:

- a row with no source name, or a name that an included object of the group
  does not carry;
- a source that cannot be read: an attribute that is not a scalar on the
  point domain, or an evaluated mesh whose vertex count differs from the
  base mesh;
- a parameter that this group's **Type** does not read;
- two rows driving the same parameter (each parameter takes at most one
  map);
- a target that is not a finite number, or one below the mapped slider's
  minimum;
- a parameter the group has switched off, such as a **Strain Limit** map on a
  group whose **Enable Strain Limit** is unticked, or a **Plasticity Rate**
  map while **Plasticity** is off. A map cannot bring back a value that is
  zero for the whole solve, and the message names the condition. It is not
  always a checkbox: a **Shell** with a shrink factor other than `1` closes
  the **Strain Limit** gate on its own;
- a map sample at or before the start frame, two samples on one frame, a
  sample with no source name, or any sample on a group that is not a
  **Shell**;
- a **Vertex Group** source on a Hair Curves object;
- an included object that is missing, or that cannot carry weights: a curve
  object in a **Rod** group is refused by name, because its rod vertices are
  sampled along the curve and none of them is a point a weight was written
  on. Convert it to a mesh or to Hair Curves, or remove it from the group.

:::{admonition} Under the hood
:class: toggle

The value on an element is `base + (target - base) * w`, with `w` the
element's average weight and `base` the group's value for the parameter (that
frame's animated value when the slider is keyframed). The add-on ships one
weight per vertex, and the averaging happens when the scene is built, before
the solve starts. A coefficient that varied inside one element would stop the
force from being the gradient of any energy.

The per-element values replace the uniform scalar in the per-element
parameter tables the solver already reads: triangles for a shell,
tetrahedra for a solid, segments for a rod. A mapped parameter therefore
needs no machinery beyond the tables a uniform one uses. Bending on a rod
acts at the vertex between two segments, which takes the mean of the two.

Inflation Pressure is refused because only the sum of the per-face pressure
forces over a closed surface is translation-invariant, and only while the
pressure is uniform. A painted pressure would push differently depending on
where the object sits in the scene.

On a Solid, each surface vertex of the tetrahedral mesh takes the weights of
the closest Blender triangle, and each interior vertex takes the Laplace
extension of those surface values. Both steps are convex combinations of
painted values, which is why the weights stay inside `0`–`1`.
:::

### Maps From MCP

Maps have no Python method and no `group.param` attribute. Six MCP tools edit
them. `add_material_map(group_uuid, parameter, source_type, source_name,
target_value, enabled)` adds a row, and `set_material_map(group_uuid, index,
...)` changes any of its fields. An omitted field keeps its value, and the
whole row is checked, so change `parameter` and `target_value` in one call.
`remove_material_map(group_uuid, index)` removes a row with its samples, and
`list_material_maps(group_uuid)` reports the rows, the parameters the group's
**Type** can map, the slider each row blends from (`base_property`) and a
`gate_closed_reason` for a row whose parameter the group has switched off.
`index` counts from zero and renumbers after a removal.

The arguments follow the panel. `parameter` is the row dropdown, written as the
key in the table under [Which Parameters Can Be Mapped](#which-parameters-can-be-mapped).
`source_type` is **Source**, either `VERTEX_GROUP` (**Vertex Group**) or
`ATTRIBUTE` (**Attribute**), `source_name` is **Name**, `target_value` is
**Target** and `enabled` is **Enable**.
`add_material_map_sample(group_uuid, index, frame, source_name, source_type)`
and `remove_material_map_sample(group_uuid, index, frame)` edit the map
samples of a **Shell** map, and an omitted `source_type` takes the row's own.

The tools check a row when you add or edit it, without waiting for
**Transfer**. They refuse **Inflation Pressure**, a parameter the group's
**Type** does not read, a **Target** outside the range of its slider, a second
enabled row on one parameter, a sample on a group that is not a **Shell**, a
sample at or before the starting frame and two samples on one frame. They do
not look for the vertex group or attribute until **Transfer**, so it can be
created after the row. See the
[MCP Tool Reference](../../integrations/mcp_reference.rst).

## Lock Translation and Lock Rotation

A **Lock Translation** box, drawn below the **Velocity Overwrite** box (below
the contact box on a **Sand** group) on every dynamic group (**Solid**,
**Shell**, **Rod**, **PDRD**, **Sand**), holds two independent locks:

- The **Lock Translation** checkbox constrains an object's mass-weighted
  center of mass to a fixed world-space line through its initial position.
  Rotation and deformation stay free.
- The **Lock Rotation** checkbox restricts an object's mass-weighted
  best-fit rigid rotation to a fixed world-space axis. Translation and
  deformation stay free.

Either checkbox, both or neither can be on. Both are set per object, because
one group can hold several bodies that each need their own axis. The box
header carries an eye icon (**Preview Axis**) that shows every enabled lock in
the group in the viewport, and an object pulldown that picks which assigned
object you are editing.

Ticking **Lock Translation** reveals **Lock All Translations** and
**Translation Axis**. Ticking **Lock Rotation** reveals **Lock All
Rotations**, **Prohibit Rotation on Axis** and **Rotation Axis**. Both axes
start along X, `(1, 0, 0)`.

- **Lock All Translations** pins the center of mass to its initial point
  instead of letting it slide along the axis. **Lock All Rotations** forbids
  rotation about every axis. While either is on, the axis row under it is
  grayed out, not hidden, under a line saying it is ignored (for the rotation
  lock, **Prohibit Rotation on Axis** is grayed out too). The value you typed
  comes back when you untick the box.
- **Translation Axis** and **Rotation Axis** are world-space directions. Only
  the direction matters, not the length. A zero axis is refused: the panel
  warns *Axis is zero; scene build will fail until it is non-zero*.
- **Prohibit Rotation on Axis** flips what the rotation axis means. Unchecked,
  the axis is the body's only rotational freedom. Checked, rotation about that
  axis is forbidden and the perpendicular rotation plane stays free.

Enable a lock for a body that should slide along a rail, a wheel or gear that
should turn on one axle without drifting off it, or a piece that should keep
deforming while its overall motion stays where you put it. Leave both off for
a fully free body.

The locks belong to each assigned object, so `group.param` has no attribute
for them. Over MCP, `set_object_locks(group_uuid, object_name, ...)` sets them,
and `get_group_objects` reports them under `locks`. Its arguments are
`lock_translation_enable` (**Lock Translation**), `lock_translation_all`
(**Lock All Translations**), `lock_translation_axis` (**Translation Axis**),
`lock_rotation_enable` (**Lock Rotation**), `lock_rotation_all` (**Lock All
Rotations**), `lock_rotation_axis` (**Rotation Axis**) and
`lock_rotation_prohibit_axis` (**Prohibit Rotation on Axis**). An axis is an
`[x, y, z]` direction. An omitted argument keeps its value. The tool refuses a
**Static** group and a call that leaves an enabled per-axis lock with a zero
axis. A lock reaches the solver only for an object whose **Include** checkbox
is on.

## Rayleigh Damping

**Solid**, **Shell** and **Rod** groups draw a **Rayleigh Damping** box with
stiffness-proportional damping. Both coefficients default to `0.0` (no
damping), must not be negative, and are measured in seconds. The sliders
stop at 0.1, and you can type a larger value.

| UI label                | Property key          | Default | Applies to        | Description                                                         |
| ----------------------- | --------------------- | ------- | ----------------- | ------------------------------------------------------------------- |
| **Deformation Damping** | `deformation_damping` | 0.0     | Solid, Shell, Rod | Damps stretch, membrane and solid deformation (seconds).            |
| **Bending Damping**     | `bending_damping`     | 0.0     | Shell, Rod        | Damps shell and rod bending (seconds). A Solid has no bending term. |

Bending damping is usually smaller than deformation damping.

:::{note}
**PDRD**, **Sand** and **Static** groups draw no **Rayleigh Damping** box.
:::

:::{admonition} Under the hood
:class: toggle

Stiffness-proportional damping adds a `(beta/dt) * K` term to the system,
where `K` is the element tangent stiffness and `beta` is the coefficient in
seconds. The deformation term reuses the SPD-projected deformation Hessian.
The bending term uses a lagged-dihedral form so that it always dissipates
energy. Tetrahedral (Solid) elements use only the deformation term, since a
tet has no bending energy.
:::

## Density-Normalized (Pa/ρ)

**Shell**, **Solid** and **Rod** groups draw a **Young's Modulus** field with a
**Density-Normalized (Pa/ρ)** checkbox in the same box. The checkbox decides
what the number in the field means. Its key is `young_mod_density_normalized`
and it defaults to `True`. The field label reads **Young's Modulus (Pa/ρ)**
while the checkbox is on and **Young's Modulus (Pa)** while it is off.

- **On** (the default): the field holds the density-normalized modulus, in
  Pa/ρ, which is the unit the solver reads. Changing **Density** alone leaves
  the motion unchanged, so you can tune stiffness and mass independently.
- **Off**: the field holds a true Young's modulus in pascals, for example a
  value from a material table. The add-on divides it by the group's density
  before it reaches the solver, so changing **Density** then changes the
  motion, as it does for a physical material.

The bundled [presets](fabric_presets.md) turn the checkbox on and store their
moduli in Pa/ρ.

The checkbox is not a `group.param` attribute. Set it in the panel, apply a
[material profile](#material-profiles) that stores it, or set
`young_mod_density_normalized` with the MCP tool
`set_group_material_properties`. A script that sets a Young's Modulus through
`group.param` writes a value in Pa/ρ while the checkbox is on.

## Strain, Plasticity and Velocity

These rows are drawn on more than one group type. Each subsection names the
types that draw it.

### Strain Limit

Drawn on **Shell** and **Rod** groups. **Strain Limit** caps how far an
element can stretch beyond its rest length. Use it for cloth or a rope that
must not visibly stretch, and leave it off when the material's own stiffness
should decide.

| UI label                | Property key           | Default | Description                                                                                                                                         |
| ----------------------- | ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Enable Strain Limit** | `enable_strain_limit`  | `False` | Turns the limit on.                                                                                                                                 |
| **Strain Limit**        | `strain_limit_percent` | 5.0     | Largest stretch beyond rest length, as a percentage (`5.0` is 5%). Range 0 – 100. `0` applies no limit. Shown only while **Enable Strain Limit** is on. |

While **Shrink X** or **Shrink Y** is not `1` on a **Shell** group, the
strain limit has no effect. While **Enable Strain Limit** is on, the panel
shows *Shrink/extend disables strain limiting* under it. A **Rod** group has
no such condition.

```{figure} ../../images/material_params/strain_limit.png
:alt: Part of a Shell group's Material Params box. The Enable Strain Limit checkbox is ticked and the Strain Limit field is below it, both outlined in red. A Bend Stiffness row and a Shrink X and Shrink Y row are above, and the Inflate and Plasticity checkboxes are below.
:width: 500px

**Enable Strain Limit** on, with the **Strain Limit** field below it.
```

### Plasticity

Drawn on **Solid** and **Shell** groups. **Plasticity** adds permanent
deformation on top of the elastic response. Where the local stretch differs
from the rest size by more than **Threshold**, the rest shape drifts toward the
current shape at the rate **Theta**. Enable it for a material that keeps a
deformation, such as crushed foil, wrinkled paper or a dented metal sheet.
Leave it off for a perfectly elastic material.

| UI label       | Property key           | Default | Description                                                                                                                                                  |
| -------------- | ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Plasticity** | `enable_plasticity`    | `False` | Turns plasticity on.                                                                                                                                         |
| **Theta**      | `plasticity`           | 0.5     | Creep rate per second. Range 0 and up (the slider stops at 10). Shown only while **Plasticity** is on.                                                       |
| **Threshold**  | `plasticity_threshold` | 0.0     | Dead zone around zero strain: creep acts only where the stretch differs from the rest size by more than this fraction. Range 0 and up (the slider stops at 1). Shown only while **Plasticity** is on. |

**Theta** is a rate per second, with `0` purely elastic. At `1` the rest shape
absorbs 63% of its difference from the current shape each second, and at `5`
it absorbs 99%. The default `0.5` absorbs about 39%. A **Threshold** of `0.02`
ignores stretches within 2% of the rest size.

On a **Solid** group, **Plasticity** cannot be on while one of the group's pins
drives the rest shape from its captured deformation through
[Track Rest-Pose Deformation](../constraints/pins.md#track-rest-pose-deformation),
because both rewrite the rest shape. The pin's panel shows *Plasticity and
rest-pose tracking cannot both be on; Transfer will refuse*, and **Transfer**
stops naming the group and the pin until one of the two is turned off.

```{figure} ../../images/material_params/plasticity.png
:alt: Part of a Shell group's Material Params box, below an Inflate checkbox. The Plasticity checkbox is ticked with Theta 0.50 and Threshold 0.000. Below it the Bend Plasticity checkbox is ticked with Bend Theta 0.50 and Bend Threshold 0 degrees. The checkboxes and fields are outlined in red.
:width: 500px

The **Plasticity** checkbox with **Theta** and **Threshold**, and the **Bend
Plasticity** checkbox with **Bend Theta** and **Bend Threshold**, on a
**Shell** group. See [Rest Angle and Bend Plasticity](#rest-angle-and-bend-plasticity)
for the rows that sit with the second pair.
```

### Rest Angle and Bend Plasticity

Drawn on **Shell** and **Rod** groups. **Rest Angle** and **From Reference
Geometry** set the angle at which each shell hinge or rod joint is relaxed.
**Bend Plasticity** lets that angle drift.

| UI label                     | Property key                | Default | Description                                                                                                                                                                  |
| ---------------------------- | --------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Rest Angle**               | `bend_rest_angle_source`    | `FLAT`  | **Flat / Straight** (`FLAT`) relaxes a shell hinge flat and a rod joint straight. **From Initial Geometry** (`FROM_GEOMETRY`) takes the angle from the input pose. |
| **From Reference Geometry**  | `bend_rest_from_reference`  | `False` | Takes the rest angle of chosen objects from a reference object. See below.                                                                                                   |
| **Bend Plasticity**          | `enable_bend_plasticity`    | `False` | Turns on the drift of the rest angle.                                                                                                                                        |
| **Bend Theta**               | `bend_plasticity`           | 0.5     | Creep rate per second. Range 0 and up (the slider stops at 10). Shown only while **Bend Plasticity** is on.                                                                  |
| **Bend Threshold**           | `bend_plasticity_threshold` | 0       | An angle: the dead zone around the rest angle. Range 0 and up (the slider stops at 180°). Shown only while **Bend Plasticity** is on.                                        |

With **Bend Plasticity** on, the rest angle drifts toward the current angle at
the rate **Bend Theta** once the angle differs from the rest angle by more than
**Bend Threshold**. **Rest Angle** and **From Reference Geometry** apply
whether or not **Bend Plasticity** is on.

The rows are laid out per type:

- **Shell**: one unlabeled box, below **Plasticity**, holds **Rest Angle**
  as its first row, then **From Reference Geometry** and **Bend Plasticity**.
- **Rod**: **Rest Angle** and **From Reference Geometry** sit in the **Bend**
  box, under **Bend Stiffness**. **Bend Plasticity** has its own box, below
  **Strain Limit**.

#### Reference Rest Angle

**From Reference Geometry** takes a bending rest angle from a second mesh. A
reference has to be a topological copy of the object whose vertices were moved
by a modifier, by geometry nodes or by hand.

1. Tick **From Reference Geometry**. A **Reference Rest Angle (per object)**
   box opens under it.
2. Pick one of the group's objects in the box's pulldown.
3. Tick **Enable Reference Rest Angle**.
4. Make the reference mesh the active object, then click the eyedropper. The
   row shows the reference's name, or *No reference set*, or *Missing:* and
   the name when the reference has gone. The **X** button clears it.

The reference is checked against the object's topology when you pick it and
again at **Transfer**, so a mesh that has drifted, or a reference that has gone
missing, is refused with a message. An object with an enabled reference takes
its bending rest angle from it, which overrides the group's **Rest Angle** for
that object alone. The rest of the group keeps its **Rest Angle**.

Over MCP, turn on **From Reference Geometry** first with
`set_group_material_properties` (`bend_rest_from_reference` set to `true`).
Then call `set_bend_reference(group_uuid, object_name, reference_object_name,
enable)` for each object. The reference must be a topological copy: the same
vertex count and connectivity (faces for a **Shell**, edges for a **Rod**),
with its modifiers and geometry nodes evaluated first. A curve rod is compared
at its control points. A mismatch is refused and named. An empty
`reference_object_name` clears the reference, and that is always accepted.
`enable` is **Enable Reference Rest Angle**. It defaults to true when a
reference is given and to false when one is cleared, and `enable` set to true
with no reference is refused. `get_group_objects` reports each object's
reference under `bend_reference`.

### Velocity Overwrite

**Velocity Overwrite** sets an object's velocity at frames you choose instead
of leaving it to the simulation. Use it to give an object a starting velocity
or to push it at a later frame. Leave the list empty for a passive
simulation.

The box is drawn on **Solid**, **Shell**, **Rod** and **PDRD** groups, above
the **Lock Translation** box. On a **PDRD** group it is titled **Velocity
(free launch)**. **Sand** and **Static** groups draw no velocity box. Each
object of the group has its own list of velocity keyframes.

1. In the box header, pick the object in the pulldown at the right.
2. Move the timeline to the frame where the velocity should apply.
3. Click the **+** button (**Add**) beside the list. The keyframe is added at
   the current frame, or at frame 1 when the timeline is earlier. A second keyframe on the same frame is
   refused with *Frame N already has a keyframe*.
4. Select the keyframe in the list and set its fields.
5. To delete a keyframe, select it and click the **-** button (**Remove**).

The header row also holds the **Copy Velocity Keyframes** and **Paste
Velocity Keyframes** icons and an eye icon (**Preview Direction**) that shows
the velocity directions of the group's objects in the viewport. **Copy
Velocity Keyframes** stores the selected object's keyframe list and is grayed
out while that list is empty. **Paste Velocity Keyframes** replaces the
selected object's list with the copied one and is grayed out until you have
made a copy. The clipboard lasts for the current Blender session.

A list row reads `Frame N (summary)`. The summary shows `speed m/s [x, y, z]`
for a translational overwrite and `ω=…°/s axis` for a spin, such as
`ω=90°/s PC3`, where the axis is its identifier (`PC1`, `PC2`, `PC3`, `X`,
`Y`, `Z` or `CUSTOM`). A keyframe with both overwrites on shows the two
summaries joined by a semicolon, and a keyframe with neither shows *No Op*. A
selected keyframe has these fields:

- **Frame**: the frame, 1 or later.
- **Enable Translational Velocity Overwrite**: on by default. It reveals
  **Direction** (X, Y and Z) and **Speed (m/s)**.
- **Enable Angular Velocity Overwrite**: off by default, and not drawn on
  **Rod** groups. It reveals **Spin Axis**, a **Custom Axis** field when that
  axis is chosen, and **Angular Speed (°/s)**, which is signed. **Spin Axis**
  offers **Principal Axis 1**, **Principal Axis 2**, **Principal Axis 3** (the
  default), **World X**, **World Y**, **World Z** and **Custom Axis**.

A field that depends on a checkbox is hidden, not grayed out, while the
checkbox is off.

A **Direction** and a **Custom Axis** are normalized, so only their direction
matters. With its checkbox ticked, a keyframe whose **Direction** is
`(0, 0, 0)` while its **Speed** is not zero, or whose **Custom Axis** is
`(0, 0, 0)` while its **Angular Speed** is not zero, names no direction to move
or turn in. **Transfer** refuses it, naming the object and the frame.

The translational keyframe at or before the **Starting Frame**, the last one if
there are several, sets the velocity the object starts with, and later
keyframes set the velocity at their own frame. A spin keyframe at or before the
**Starting Frame** applies from the start of the solve.

**Speed** and **Angular Speed** are measured against the Blender animation's
own time, so **Time Scale** (see [Scene Parameters](scene.md)) applies to them
like any other keyed motion: at `0.5` every keyframe's speed is halved, the
keyframe that sets the object's velocity at the **Starting Frame** included. A
**Speed** is a length per second, so it also scales with
[World Scaling](scene.md#world-scaling). An **Angular Speed** does not.

```{figure} ../../images/material_params/velocity_overwrite.png
:alt: Velocity Overwrite box with four keyframes listed. The rows read Frame 1 (0.0 m/s [0.0, 0.0, 1.0]), Frame 30 (0.5 m/s [1.0, 0.0, 0.0]) selected, Frame 60 (1.0 m/s [0.0, 1.0, 0.0]) and Frame 90 (0.3 m/s [-1.0, 0.0, 0.0]). Above the list are the box title, three icons and an object pulldown reading Cloth, and a plus and a minus button sit at the right of the list. Below it, the Frame field reads 30, Direction reads X 1.00, Y 0.00 and Z 0.00, and Speed (m/s) reads 0.50.
:width: 500px

The **Velocity Overwrite** box with four keyframes (frames 1, 30, 60 and 90).
The object pulldown at the top picks the object the list belongs to, here an
object named `Cloth`. The **+** and **-** buttons add and remove a keyframe. The
selected keyframe's **Frame**, **Direction** and **Speed (m/s)** fields are
below the list.
```

From Python, `Group.set_velocity(object_name, direction, speed, frame=1, ...)`
adds a keyframe to one object's list; see
[Group Surface](../../integrations/python_api.md#group-surface). Over MCP,
`add_velocity_keyframe(group_uuid, object_name, frame, direction, speed, ...)`
adds one too, and `set_velocity_keyframe(group_uuid, object_name, index, ...)`
changes fields of one by its position in the frame-ordered list. A new `frame`
retimes the keyframe, and the result reports `new_index`.
`remove_velocity_keyframe` removes the keyframe at a `frame`,
`list_velocity_keyframes` lists each keyframe's frame, direction and speed, and
`clear_velocity_keyframes` empties the list. The arguments follow the panel:

| Argument               | Panel                                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `frame`                | **Frame**, 1 or later. A second keyframe on one frame is refused.                                                                  |
| `direction`, `speed`   | **Direction** and **Speed (m/s)**. `speed` is 0 or more.                                                                           |
| `enable_translational` | **Enable Translational Velocity Overwrite**.                                                                                       |
| `enable_angular`       | **Enable Angular Velocity Overwrite**. On by default when `angular_speed` is not `0`.                                              |
| `angular_axis`         | **Spin Axis**: `PC1`, `PC2`, `PC3`, `X`, `Y`, `Z` or `CUSTOM`. The integers `0`, `1` and `2` also name the three principal axes.  |
| `angular_axis_custom`  | **Custom Axis**.                                                                                                                   |
| `angular_speed`        | **Angular Speed (°/s)**, signed, in degrees per second.                                                                            |

## Shell-Specific

A **Shell** group (cloth) draws the rows in this table. **Friction**, the
contact rows and the other rows that several types draw are listed with their
types in [Rows by Group Type](#rows-by-group-type).

| UI label                     | Property key          | Default         | Description                                                                                                                                                                      |
| ---------------------------- | --------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model**                    | `shell_model`         | `BARAFF_WITKIN` | **Baraff-Witkin** (`BARAFF_WITKIN`) or **ARAP** (`ARAP`).                                                                                                                        |
| **Density (kg/m²)**          | `shell_density`       | 1.0             | Areal density, in kg/m². Range 0.01 – 10000.                                                                                                                                     |
| **Young's Modulus (Pa/ρ)**   | `shell_young_modulus` | 1000.0          | Membrane stiffness. See [Density-Normalized (Pa/ρ)](#density-normalized-paρ). Range 0.01 – 1e9 (the slider stops at 1e7).                                                        |
| **Poisson's Ratio**          | `shell_poisson_ratio` | 0.35            | The slider covers 0 – 0.4999. The solver needs a value above 0 and below 0.5, and at `0` it stops with *Poisson's ratio must be positive*.                                                                                                    |
| **Bend Stiffness**           | `bend`                | 10.0            | Bending stiffness of the hinge between neighboring faces. Range 0 and up (the slider stops at 100). A **Rod** group writes the same key on its own scale; see [Bend Stiffness on a Rod](#bend-stiffness-on-a-rod). |
| **Bending Stiffness (Warp)** | `bend_warp`           | 0.0             | Extra bending stiffness for the warp (UV X) fibers, added on top of **Bend Stiffness**. `0` bends the same in every direction. Range 0 and up (the slider stops at 1e7). Needs a UV map. |
| **Bending Stiffness (Weft)** | `bend_weft`           | 0.0             | Extra bending stiffness for the weft (UV Y) fibers, added on top of **Bend Stiffness**. `0` bends the same in every direction. Range 0 and up (the slider stops at 1e7). Needs a UV map. |
| **Shrink X**                 | `shrink_x`            | 1.0             | Rest-shape scale along UV X (warp). Range 0.1 and up (the slider stops at 2). Below `1` shrinks, above `1` extends.                                                              |
| **Shrink Y**                 | `shrink_y`            | 1.0             | Rest-shape scale along UV Y (weft). Range 0.1 and up (the slider stops at 2). Below `1` shrinks, above `1` extends.                                                              |
| **Inflate**                  | `enable_inflate`      | `False`         | Turns on a pressure along the face normals. See [Inflate](#inflate).                                                                                                             |
| **Pressure (Pa)**            | `inflate_pressure`    | 0.0             | Pressure on each face. Range 0 and up (the slider stops at 100). Shown only while **Inflate** is on.                                                                             |
| **Stitch Stiffness**         | `stitch_stiffness`    | 1.0             | Stiffness of the loose-edge stitches in the mesh. Range 0 and up (the slider stops at 10000).                                                                                    |

Every key in this table works in the panel, with `group.param`, in a material
profile and with the MCP tool `set_group_material_properties`, except
`bend_warp` and `bend_weft`. Set those two in the panel or with the MCP tool.
When a directional value is set on a mesh without a UV map, the panel shows a
*has no UV map* warning that names the object, and the scene build rejects that
object.

Loose edges (edges that belong to no face) become stitch constraints, with
their stiffness set by **Stitch Stiffness**. The field is drawn on **Shell**
and **Solid** groups, and it is grayed out with *No loose-edge stitches in this
group* when no assigned mesh has a loose edge. A **Rod** group has no **Stitch
Stiffness** field, because its edges are the rod itself. A seam between two
objects is a merge pair, which carries its own stiffness; see
[Snap and Merge](../constraints/snap_merge.md).

A stitch seam must run between two vertices that belong to faces. On a
**Shell**, a vertex that sits partway along a loose edge and belongs to no
face is refused at **Transfer** unless it is pinned. See
[Transfer or build rejects the mesh](../../troubleshooting.md#transfer-or-build-rejects-the-mesh).

### Shrink X / Shrink Y

**Shrink X** and **Shrink Y** scale the rest shape of the cloth along the UV X
(warp) and UV Y (weft) directions. A value below `1` makes the rest shape
smaller along that direction, and a value above `1` makes it larger. The solver
treats the scaled shape as the relaxed one and drives the mesh toward it under
the usual stiffness. At `1.0` for both, the mesh you drew is the rest shape.
`0.95`, for example, makes the rest shape 5% smaller along that direction.
On cloth pinned at its edges, a value below `1` puts the cloth under tension,
because the pins hold it larger than its rest shape.

While either value is not `1`, the group's strain limit has no effect. See
[Strain Limit](#strain-limit).

```{figure} ../../images/material_params/shrink_shell.png
:alt: Part of a Shell group's Material Params box. The Shrink X and Shrink Y fields share one row, both reading 1.00, outlined in red. The Collision Active Duration Windows checkbox and the Bend Stiffness row are above, and the Enable Strain Limit and Inflate checkboxes are below.
:width: 500px

**Shrink X** and **Shrink Y** on one row of a **Shell** group. `1.0` leaves
the rest shape alone.
```

### Inflate

**Inflate** applies a pressure along each face normal, which pushes the shell
outward like a balloon or an airbag. Use it for inflatables such as pillows,
airbags and balloons, and leave it off for ordinary cloth.

The **Pressure (Pa)** field appears while **Inflate** is on. Its minimum is
`0`, so there is no inward (suction) pressure. A pressure of `0` applies
nothing.

The pressure on a face does not depend on **Density**, unlike **Young's
Modulus**, which is density-normalized by default. The same **Pressure (Pa)**
therefore accelerates a heavier shell less. A pressure cannot be painted over
the surface with a [spatial material map](#spatial-material-maps); use the
uniform field.

```{figure} ../../images/material_params/inflate.png
:alt: Part of a Shell group's Material Params box. The Inflate checkbox is ticked and the Pressure (Pa) field below it reads 0.000, both outlined in red. The Shrink X and Shrink Y row and the Enable Strain Limit checkbox are above, and the Plasticity and Bend Plasticity checkboxes are below.
:width: 500px

**Inflate** on, with the **Pressure (Pa)** field below it.
```

## Solid-Specific

A **Solid** group draws the rows in this table, the
[Tetrahedralizer](#tetrahedralizer-per-object) box and the
[Plasticity](#plasticity) and [Velocity Overwrite](#velocity-overwrite) boxes.
**Friction**, the contact rows and the other rows that several types draw are
listed with their types in [Rows by Group Type](#rows-by-group-type).

| UI label                   | Property key          | Default | Description                                                                                                               |
| -------------------------- | --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Model**                  | `solid_model`         | `ARAP`  | **Stable NeoHookean** (`STABLE_NEOHOOKEAN`) or **ARAP** (`ARAP`).                                                         |
| **Density (kg/m³)**        | `solid_density`       | 100.0   | Volumetric density, in kg/m³. Range 0.01 – 10000.                                                                         |
| **Young's Modulus (Pa/ρ)** | `solid_young_modulus` | 500.0   | Stiffness. See [Density-Normalized (Pa/ρ)](#density-normalized-paρ). Range 0.01 – 1e9 (the slider stops at 1e7).          |
| **Poisson's Ratio**        | `solid_poisson_ratio` | 0.35    | The slider covers 0 – 0.4999. The solver needs a value above 0 and below 0.5, and at `0` it stops with *Poisson's ratio must be positive*.                                             |
| **Shrink**                 | `shrink`              | 1.0     | Uniform rest-shape scale. Range 0.1 and up (the slider stops at 2). Below `1` shrinks, above `1` extends.                 |
| **Stitch Stiffness**       | `stitch_stiffness`    | 1.0     | Stiffness of the loose-edge stitches in the mesh. Range 0 and up (the slider stops at 10000).                             |

Every key in this table works in the panel, with `group.param`, in a material
profile and with the MCP tool `set_group_material_properties`.

A **Solid** mesh's loose edges become stitches the same way a **Shell**'s do.
Each end of such an edge has to be a vertex of a face: a vertex that sits
partway along a loose edge and belongs to no face is refused at **Transfer**,
pinned or not.

### Shrink

**Shrink** scales the rest shape of the whole solid uniformly. A value below `1`
makes the rest shape smaller, and a value above `1` makes it larger. The
solver treats the scaled shape as the relaxed one and drives the mesh toward
it under the usual stiffness. At `1.0` the solid rests at its modeled size.

```{figure} ../../images/material_params/shrink_solid.png
:alt: Part of a Solid group's Material Params box. The Density, Young's Modulus (Pa/ρ) and Poisson's Ratio rows are above a Shrink row reading 1.00, which is outlined in red. The Friction row and the contact box follow.
:width: 500px

**Solid** groups have one **Shrink** row. **Shell** groups have **Shrink X** and
**Shrink Y** instead; see [Shrink X / Shrink Y](#shrink-x--shrink-y).
```

### Tetrahedralizer (per object)

A solid's surface is tetrahedralized before it goes to the solver. The
**Tetrahedralizer** box picks how, for each assigned object. It is drawn on
**Solid** groups only, below the **Lock Translation** box and above
**Rayleigh Damping**.

- The pulldown at the right of the box title picks which assigned mesh you are
  configuring, so each solid in the group can use its own backend and
  overrides.
- The backend dropdown below it picks the tetrahedralizer:
  - **fTetWild** (the default) is a tolerant remesher. It accepts open,
    cracked or non-manifold input, but it resamples the surface, so your input
    vertices are reconstructed through a surface map rather than preserved
    exactly.
  - **TetGen** preserves the surface exactly (a one-to-one vertex map). It
    requires a clean, closed, manifold mesh and rejects open, coplanar or
    non-manifold input. If TetGen refuses a mesh, repair it, move the object to
    a **Shell** group, or switch it to **fTetWild**.

The override rows below the backend dropdown change to match the backend.

The distribution for Windows on ARM ships no fTetWild. On that platform choose
**TetGen** for each solid. With **fTetWild** selected, the build stops with a
message that fTetWild is not available on Windows on ARM. The scene is not
switched to TetGen, because the two backends produce different meshes.

The override values are properties of each assigned object. They are not in a
material profile and not reachable through `group.param`. The MCP tool
`set_object_tet_settings` sets them.

#### fTetWild Overrides

With **fTetWild** selected, seven overrides appear. Each row has a checkbox at
its left (its tooltip reads **Override**) and the value at its right. The
value is grayed out until its checkbox is on, and it goes to fTetWild only
then. With every checkbox off, fTetWild runs at its own defaults. Use the
overrides when a solid meshes too coarsely, misses features or takes too long
to tetrahedralize.

| UI label               | Property key               | Default | Range        | Description                                                              |
| ---------------------- | -------------------------- | ------- | ------------ | ------------------------------------------------------------------------ |
| **Edge Length Factor** | `ftetwild_edge_length_fac` | 0.05    | 0.0001 – 1   | Ideal tetrahedron edge length as a fraction of the bounding-box diagonal (`-l`). |
| **Epsilon**            | `ftetwild_epsilon`         | 0.001   | 0.000001 – 1 | Envelope size as a fraction of the bounding-box diagonal (`-e`).         |
| **Stop Energy**        | `ftetwild_stop_energy`     | 10.0    | 3 – 1000     | AMIPS energy threshold. A larger value runs faster and gives lower quality. |
| **Max Opt Iterations** | `ftetwild_num_opt_iter`    | 80      | 1 – 1000     | Maximum number of optimization passes.                                   |
| **Optimize**           | `ftetwild_optimize`        | `True`  |              | Improves cell quality, and is slower.                                    |
| **Simplify Input**     | `ftetwild_simplify`        | `True`  |              | Simplifies the input surface before tetrahedralization.                  |
| **Coarsen Output**     | `ftetwild_coarsen`         | `False` |              | Coarsens the output while preserving quality.                            |

Each value has a checkbox property named `ftetwild_override_<field>`, where
`<field>` is the key without the `ftetwild_` prefix, for example
`ftetwild_override_epsilon`. It defaults to off.

```{figure} ../../images/material_params/ftetwild_expanded.png
:alt: The override rows at the bottom of a Solid group's Material Params box. Edge Length Factor has its checkbox ticked and reads 0.0500. Epsilon, Stop Energy and Max Opt Iterations have unticked checkboxes and grayed values. Optimize has its checkbox ticked and its value ticked. Simplify Input and Coarsen Output have unticked checkboxes and grayed values.
:width: 500px

The override rows of a **Solid** group. The checkbox at the left of a row turns
its override on. With it off the value is grayed out and fTetWild's own default
is used. Here **Edge Length Factor** and **Optimize** are overridden and the
other five rows stay at their defaults.
```

#### TetGen Overrides

With **TetGen** selected, the override rows switch to TetGen's interior
controls. TetGen always preserves the input surface exactly, so these tune only
the interior refinement. Each row uses the same checkbox pattern: the value goes
to TetGen only while its checkbox is on, and otherwise TetGen runs at its own
defaults.

| UI label                  | Property key        | Default | Range                       | Description                                                              |
| ------------------------- | ------------------- | ------- | --------------------------- | ------------------------------------------------------------------------ |
| **Min Radius-Edge Ratio** | `tetgen_min_ratio`  | 2.0     | 1 and up (slider stops at 5) | Quality bound (`-q`). A smaller value gives rounder interior cells.      |
| **Max Tet Volume**        | `tetgen_max_volume` | 0.0     | 0 and up                    | Caps the interior cell volume in object units (`-a`). `0` leaves it uncapped. |

The checkbox properties are `tetgen_override_min_ratio` and
`tetgen_override_max_volume`.

## Rod-Specific

A **Rod** group (a strand) draws the rows in this table and the **Bend** box.
**Friction**, the contact rows and the other rows that several types draw are
listed with their types in [Rows by Group Type](#rows-by-group-type).
[The Material Params Box](#the-material-params-box) lists what switching a
group to **Rod** sets.

| UI label                   | Property key        | Default | Description                                                                                                           |
| -------------------------- | ------------------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| *(no Model row)*           | `rod_model`         | `ARAP`  | A rod always uses ARAP, so the panel draws no **Model** dropdown. `ARAP` is the only accepted value.                  |
| **Density (kg/m)**         | `rod_density`       | 1.0     | Line density, in kg/m. Range 0.01 – 10000.                                                                            |
| **Young's Modulus (Pa/ρ)** | `rod_young_modulus` | 10000.0 | Stiffness. See [Density-Normalized (Pa/ρ)](#density-normalized-paρ). Range 0.01 – 1e9 (the slider stops at 1e7).      |
| **Shrink**                 | `length_factor`     | 1.0     | Rest-length scale for every segment of the strand. Range 0.1 and up (the slider stops at 2). Below `1` pulls the strand taut, above `1` leaves it slack. |
| **Bend Stiffness**         | `bend`              | 1.0     | How strongly the strand resists being curved. Range 0 and up (the slider stops at 100). The default is the value a group takes when its **Type** becomes **Rod**. |

Every key in this table works in the panel, with `group.param` and with the MCP
tool `set_group_material_properties`. A material profile stores `rod_model`,
`rod_density`, `rod_young_modulus` and `bend`, and not `length_factor`.

A rod's **Contact Offset** is the strand's radius, and the room its **Local Gap
Limit** works with is what a segment has left once two of those radii are
taken out of it. Keep every segment comfortably longer than twice the offset;
see [Thick Strands Need Room](#thick-strands-need-room).

**Bend Stiffness** writes the same `bend` property a **Shell** group writes,
so a material profile carries one value for both types. Each type scales that
value on its own terms, so the same number is not the same stiffness on a
**Shell** and on a **Rod**. Switching a group's **Type** to **Rod** sets `bend`
to `1.0`, while a new **Shell** group starts at `10.0`.

A **Rod** group also draws [Strain Limit](#strain-limit),
[Rest Angle and Bend Plasticity](#rest-angle-and-bend-plasticity) (**Rest
Angle** and **From Reference Geometry** are in the **Bend** box),
[Spatial Material Maps](#spatial-material-maps), which vary a parameter along
the strands, and [Velocity Overwrite](#velocity-overwrite) without spin
controls.

A Hair Curves object joining an empty **Rod** group sets **Rest Angle** to
**From Initial Geometry** and **Contact Offset** to `0`; see
[Hair Curves](../scene/hair_curves.md#adding-hair-to-a-group).

### Bend Stiffness on a Rod

**Bend Stiffness** sets how strongly a strand resists being curved. At `0` the
rod is a limp thread that resists only stretching. Raise it for wire, cable,
tubing or hair that should keep the shape it was drawn with. Lower it toward
`0` for thread, string or yarn that should take its shape from gravity and
contact.

**A rod bends the same way however many segments it is drawn with.** The
stiffness is measured against a reference segment of 1 cm and rescaled to the
segment length the strand actually has. Adding control points for a smoother
silhouette or finer contact therefore leaves the drape unchanged, and a strand
drawn coarse and the same strand drawn fine are the same rod. Tune the value
on a draft mesh and keep it when you refine the geometry.

The value is also independent of **Density**, so changing **Density** does not
change the bent shape.

### Shrink on a Rod

**Shrink** (`length_factor`) scales the rest length of every segment in the
strand. Below `1.0` the rest length is shorter than the drawn geometry, so a
strand pinned at both ends is pulled taut. Above `1.0` the rest length is
longer, so the strand sags or buckles between its pins. It is the rod
counterpart of **Shell**'s **Shrink X** / **Shrink Y** and **Solid**'s
**Shrink**. Like them it changes the rest shape and not the drawn geometry, so
nothing changes until the simulation starts.

Mass is unchanged: a strand's mass comes from its density and its drawn
length, so tensioning a rod with **Shrink** does not make it lighter.

**Bend Stiffness** is measured against the rest length, which **Shrink**
scales, so **Shrink** also changes the stiffness of each bending joint, by the
inverse square. **Shrink** = `0.5` makes each joint about four times stiffer,
and **Shrink** = `2.0` about four times less stiff. Multiplying **Bend
Stiffness** by the square of the **Shrink** value cancels that change at each
joint. The stiffness per unit length of strand (the flexural rigidity)
changes by the inverse of **Shrink** alone, so multiply **Bend Stiffness** by
**Shrink** instead to hold that one fixed.

:::{admonition} Under the hood
:class: toggle

Each interior point of a strand carries the bending energy
`0.5 * k * (θ - θ₀)²`, where `θ` is the angle between the two segments meeting
at that point and `θ₀` the rest angle chosen by **Rest Angle** (`π` for a
straight rod). The coefficient is

```text
k = bend * m * (L_ref / l)²        L_ref = 1 cm
```

with `m` the lumped mass at that point and `l` its Voronoi rest length (half
the sum of the two segment rest lengths meeting there, so it carries the
**Shrink** factor).

The two are not the same length. `m` is half of each incident segment's mass,
and a segment's mass is its line density times its **drawn** length, taken from
the geometry before **Shrink** scales the rest length. So with `d` the drawn
spacing at that point, `m = density × d` while `l = d ×` **Shrink**, and the
coefficient expands to

```text
k = bend × density × L_ref² / (d × Shrink²)
```

Hold **Shrink** fixed and `k` falls as `1/d`. That is `k = B / l`, with `B` the
flexural rigidity of the continuum bending energy `0.5 * B * κ²` integrated
along the strand. `B` works out to `bend × density × L_ref² /` **Shrink** and
carries no `d`, which is why the bent shape does not depend on how finely the
strand is drawn. Hold the geometry fixed instead and `k` goes as
`1 / Shrink²`. `B` is proportional to density, so the bent shape does not
depend on the density you set. `L_ref` only places the numeric range of the
**Bend Stiffness** field.

Rest lengths are measured in solver space, after **World Scaling** has been
applied to the geometry, so at the default **World Scaling** of `1.0` a
segment's rest length is its length in Blender units.
:::

## Sand-Specific

A **Sand** group is a granular body: a cloud of grain centers, each grain a
sphere of one shared radius. Its geometry is that cloud, not a surface, so a
mesh has to be converted first (see [Creating a Sand Body](#creating-a-sand-body)).
The **Material Params** box of a **Sand** group draws the rows in this table,
the **Lock Translation** box, the **Allow Intersections** box (without **Allow
Existing Intersections**) and **Force Field Weight**. It draws no **Density**,
stiffness or **Velocity Overwrite** rows.

| UI label              | Property key         | Default            | Description                                                                                                   |
| --------------------- | -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Grain Radius (m)**  | `sand_grain_radius`  | 0.02 until a mesh is converted, then the converted radius | Radius of one grain, in meters. Read-only: once a mesh is converted, it shows the radius chosen in the **Convert To Solid Particle Mesh** dialog. |
| **Particle Mass (g)** | `sand_particle_mass` | 1.0                | Mass of one grain, in grams (the solver receives kilograms). Range 0.000001 – 1000000.                        |
| **Friction**          | `sand_friction`      | 0.0                | Coulomb friction coefficient between grains. Range 0 and up.                                                  |
| **Contact Gap**       | `contact_gap`        | 0.001              | Barrier distance on top of the grain radius, in Blender units. Range 0.001 – 0.01.                            |

**Friction** on a **Sand** group is its own field (`sand_friction`), not the
shared **Friction** row, which the box does not draw for this type.

The keys in this table work in the panel, with `group.param` and with the MCP
tool `set_group_material_properties`. A material profile stores `contact_gap`
and not the three `sand_` keys.

The **Grain Radius (m)** field is always grayed out, with a *Grain radius is
locked at convert* note under it. Before a mesh is converted it shows `0.02`,
the group's starting value. The **Convert To Solid Particle Mesh** dialog sets
the radius, and once an object is converted the field shows that object's
radius. The spacing of the grains is derived from the radius when the cloud is
seeded, and a different radius on the same cloud could put grains inside each
other's contact skin. To work at a different radius, convert the source mesh
again.

A **Sand** group is solved at one grain radius, so every object in it has to
have been converted at the same **Grain Radius**. When two were converted at
different radii, the group box shows *Grains were converted at different radii;
Transfer will refuse*, and **Transfer** stops, naming one object at each
radius. Convert them at one radius, or put them in separate **Sand** groups.

The grain radius is also the contact offset: the sphere it describes is the
grain's physical skin. There is therefore no separate **Contact Offset** row,
and **Contact Gap** is the only extra distance you set. The box says so with a
*Grain radius is the contact offset* note.

### Creating a Sand Body

1. Set a group's **Type** to **Sand**. The **Convert To Solid Particle Mesh**
   button belongs to **Sand** groups.
2. Select the mesh to convert and make it the active object. The mesh must
   have faces and must not already be a particle mesh.
3. Click **Convert To Solid Particle Mesh**. It sits in the group's box just
   above **Delete Group**, so you can reach it without expanding **Material
   Params**. The button is grayed out while the active object does not qualify,
   and a line under it names the missing condition: *Select a solid mesh to
   convert* (the active object is not a selected mesh), *Active object is
   already a particle mesh* or *Active mesh has no faces to convert*. Clicking
   the button opens a dialog.
4. Set the three dialog fields and confirm:
   - **Grain Radius** (default `0.05`): the physical radius of one grain. It is
     locked onto the object.
   - **Extra Spacing** (default `0`): how much room to leave between grains
     beyond touching. `0` packs them as densely as the non-overlap rule
     allows, and larger values give a looser, sparser cloud.
   - **Random Seed** (default `0`): picks a different arrangement at the same
     radius and spacing.
5. Add the converted object to the **Sand** group with **Add Selected
   Objects**.

The grain count is not a setting. Grains fill the volume at the radius and
spacing you chose, and the report line after the conversion gives the number.
If none fit, the conversion stops with *No grains fit; reduce the grain radius
or the extra spacing*. The destructive part has already run by then: the faces
are gone, the object is left as an empty particle mesh, and the button grays
out with *Active object is already a particle mesh*. To retry at a smaller
radius, start from a copy of the source mesh.

The grains are drawn with a tan material named **Sand**. The add-on creates it
at the first conversion and reuses it for later ones, so editing it in the
**Material** tab changes every sand body. It does not affect the simulation.

:::{warning}
**The conversion is destructive.** The object's faces are discarded and replaced
by a cloud of loose vertices plus a render-only **Particle Mesh** modifier that
draws each vertex as a sphere. Keep a copy of the source mesh if you may want
to re-convert at a different radius.
:::

:::{note}
Leave **Preconditioner** on **Block Jacobi** for a scene with a **Sand**
group. See [Preconditioner](scene.md#preconditioner).
:::

## PDRD-Specific

**PDRD** (Painless Differentiable Rotation Dynamics) is an exactly rigid body
type. Its material block is **Density** and **Friction**. It has no Young's
modulus, Poisson's ratio, bend, shrink, strain limit or inflate, because the
body is exactly rigid and has no stiffness to tune, and PDRD bodies are not
tetrahedralized.

A **PDRD** group's **Material Params** box draws these rows and boxes, in this
order:

1. **Density** and **Friction**.
2. The [Hinge](#hinge-joints) box.
3. The contact box, then **Collision Active Duration Windows**. See
   [Shared Parameters](#shared-parameters).
4. **Velocity (free launch)**, which is [Velocity Overwrite](#velocity-overwrite)
   under another title.
5. The **Lock Translation** box. See
   [Lock Translation and Lock Rotation](#lock-translation-and-lock-rotation).
6. The **Allow Intersections** box and **Force Field Weight**.

| UI label            | Property key   | Default | Description                                                                                                  |
| ------------------- | -------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| **Density (kg/m³)** | `pdrd_density` | 100.0   | Volumetric density, in kg/m³. Mass is the density times the enclosed mesh volume. Range 0.01 – 10000.        |

**Density** sets the body's mass and, through its shape, its rotational
inertia. The motion is exactly rigid at any mesh resolution, so there is no
stiffness or rigidity value to tune. The key works in the panel, with
`group.param` and with the MCP tool `set_group_material_properties`. A material
profile does not store it.

:::{admonition} Under the hood
:class: toggle

Each PDRD body is solved in reduced 6-DOF coordinates (translation plus
rotation): every Newton iteration fits the single best-fit rigid transform to
the body and reconstructs its surface from that transform, so the body stays
exactly rigid by construction rather than through a stiff penalty.
:::

### Hinge Joints

A **PDRD** body can be a **hinge**: its position is pinned and its rotation is
locked to one principal axis of its rest shape, so the body spins on that axis
like a wheel on an axle. The setting is per object, so each body in one
**PDRD** group can hinge on its own axle. For example, each gear in a train can
turn on its own pin while tooth contact passes torque from one gear to the
next.

The free axle is one of the three principal axes of the body's rest shape:
**Principal Axis 1** is the largest extent, **Principal Axis 2** the middle
extent and **Principal Axis 3** the thinnest extent. **Principal Axis 3**, the
default, is the usual axle for a flat gear or disk.

1. Expand the **Hinge** box in the group's **Material Params**. It starts
   collapsed.
2. In the unlabeled pulldown, pick the body whose hinge you are editing. A
   **PDRD** group can hold several bodies, such as a gear train, and the
   pulldown moves between them.
3. Tick **Hinge**. It is off by default. Ticking it pins that body and locks its
   rotation to one principal axis.
4. Pick the axle in **Axle**. It is grayed out until **Hinge** is ticked.

**Visualize**, on by default, draws the hinge axle gizmo in the viewport for the
hinged bodies of the group.

The hinge is a property of each object, not a group material row, so it has its
own box instead of a row in the table above.

:::{admonition} Under the hood
:class: toggle

A script sets a hinge per object with
`Group.set_hinge(object_name, pca_axis=2, enable=True)`. `pca_axis` `0`, `1`
and `2` select **Principal Axis 1**, **2** and **3**, and `enable=False` clears
the hinge so the body moves freely again. The MCP tool `set_pdrd_hinge` does
the same.

The import below works when the add-on is in the **User Default** repository.
For an add-on installed from a remote repository, import `solver` with the
lookup in [Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

gears = solver.create_group("Gears", type="PDRD")
gears.add("GearA")
gears.set_hinge("GearA", pca_axis=2)   # spin on the thinnest axis
```
:::

## Static-Specific

A **Static** group is a collider: you animate its shape, and the solver pushes
everything else out of its way. Besides **Friction**, the contact rows and the
**Allow Intersections** box, it has one option. See
[Static Objects](../scene/static_objects.md) for the full treatment of Static
groups.

| UI label                   | Property key                | Default | Description                                                                                                                                     |
| -------------------------- | --------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Apply Soft Constraints** | `enable_soft_constraint`    | `False` | Holds the collider toward its animation with springs instead of locking it to it.                                                              |
| **Stiffness**              | `soft_constraint_stiffness` | 10.0    | How firmly the springs hold. Shown only while **Apply Soft Constraints** is on. Must be above 0 (the minimum is 0.000001), and the slider covers 0.001 – 1000. |

Both keys work in the panel, in a material profile and with the MCP tool
`set_group_material_properties`. They are not available through `group.param`.
**Transfer** refuses a **Stiffness** of zero or less.

### Apply Soft Constraints

By default a **Static** collider follows its animation exactly. Nothing the
cloth does can move it. That is right when the cloth can always get out of the
way.

It fails when the collider's own shape closes onto the cloth. If a character's
arm comes down against the torso, or a hand presses into a thigh, the garment in
between is caught between two surfaces that do not move, and the solve stops
with a solver failure.

1. Select the **Static** group.
2. Expand its **Material Params** box.
3. Tick **Apply Soft Constraints**.
4. Adjust **Stiffness** from what you see in the result.

With **Apply Soft Constraints** on, springs hold the collider toward its
animated shape instead of locking it to that shape. Where the cloth pushes
harder than the springs, the collider yields. Once the pinch passes, it returns
to its animation.

### Stiffness

**Stiffness** sets how firmly the springs hold. It is shown only while **Apply
Soft Constraints** is on. Lower values let the collider yield more. Higher
values hold it closer to its animation, which is what a collider with **Apply
Soft Constraints** unticked follows exactly. Very high values slow the solve.

There is no universal number, because the springs compete against however hard
your scene's contacts push. Start at the default `10` and adjust from the
result.

:::{note}
A **Static** group with **Apply Soft Constraints** on is part of the solved
scene even if it never moves, because the springs need something to act on.
Turn it on for the colliders that need it and not for every collider.
:::

:::{admonition} Under the hood
:class: toggle

An exact collider vertex is a boundary condition: the solver removes its
degrees of freedom, so the contact force has nothing to act on. A soft one
keeps its degrees of freedom and gains a spring term, `k * (target - x)`, which
is what the contact force can work against.

Two properties hold for every **Static** collider that is part of the solved
scene, whether it is animated, soft-constrained or both. Contact between two
collider elements is excluded, whether they belong to one collider or to two,
which allows a rigged mesh that ships self-intersecting (layered eyelashes, an
arm resting inside a torso) to be used. Collider faces also carry no stretch or
bending energy, so a collider's shape comes from its animation, or from the
springs, and not from a material.
:::

## Contact Gap and Contact Offset

**Contact Gap** and **Contact Offset** set how close two surfaces may come
before the solver pushes them apart. Both are rows of a group's contact box in
**Material Params** (see [Rows by Group Type](#rows-by-group-type)).

- **Contact Gap** is the distance at which the solver starts to push two
  surfaces apart. For a pair of surfaces the solver uses the mean of their two
  gaps, and a larger gap starts the push earlier.
- **Contact Offset** is padding added on top of the gap. A pair is in contact
  once its distance is below the mean of the two gaps plus the sum of the two
  offsets, so a group's offset adds clearance whatever the other side chose.
  The default is `0.0`, or `0.005` on a **Rod** group (see
  [Absolute vs Ratio](#contact-gap-absolute-vs-ratio)).

Use **Contact Offset** when one group needs a fixed thickness whatever it
touches, for example a garment that must keep a minimum clearance from every
body group it comes near. To tune how tightly surfaces sit, leave **Contact
Offset** at zero and change **Contact Gap**.

**A gap that is too small for the step size is refused.** The contact gap
cannot be smaller than the **Step Size** divided by 4096, and **Transfer**
stops when the smallest gap in the scene is. The message names the objects
involved and both numbers, for example *The contact gap between 'A' and 'B' is
... below the ... that a step of 0.01 s can solve*, and gives the two remedies:
raise **Contact Gap** to at least the number shown, or lower **Step Size**.
Where **Local Gap Limit** holds an object's gap to a share of the room between
its mesh neighbors, a larger **Contact Gap** does not lift it, so the message
names the limit and suggests lowering **Step Size**, coarsening the mesh there,
or reducing **Contact Offset**. The check is skipped when contact is turned
off for the scene.

## Contact Gap: Absolute vs Ratio

Each of the two distances can be set in either of two ways:

- **Absolute** (the **Contact Gap** and **Contact Offset** fields): a distance
  in Blender units, the same for every object of the group. Use it when you
  need an exact, known thickness.
- **Ratio** (the **Contact Gap Ratio** and **Contact Offset Ratio** fields): a
  fraction of each object's own bounding-box diagonal, measured at
  **Transfer**. Objects of different sizes in one group get different
  distances, and spreading the objects of a group apart changes none of them.
  The same ratio suits objects of any size.

The absolute **Contact Gap** cannot go below `0.001` Blender units, and the
absolute **Contact Offset** stops at `0.03`. For a finer gap, use **Contact Gap
Ratio**, which goes down to `0.00001` of the diagonal. Defaults and ranges of
all four fields are in [Shared Parameters](#shared-parameters).

```{figure} ../../images/material_params/contact_gap_modes.svg
:alt: Absolute vs ratio contact-gap comparison. In absolute mode the halo is the same thickness on small and large objects; in ratio mode it scales with the object.
:width: 500px

The dashed red ring shows the contact-gap layer. In **Absolute** mode the
layer has the same thickness in Blender units around every object, so it is
wide relative to a small object and narrow relative to a large one. In
**Ratio** mode the layer scales with the object's bounding box, so it keeps the
same proportion to objects of any size.
```

On **Solid**, **Shell**, **PDRD** and **Static** groups, the **Use Object
Bounding Box Diagonal** checkbox picks between the two. It is on by default,
so the ratio rows are drawn. Turn it off to type exact distances in Blender
units, for example to match the distances of another group. One checkbox
switches both pairs: **Contact Gap** with **Contact Offset**, and their ratio
rows.

A **Rod** group has no checkbox. Setting a group's **Type** to **Rod** turns
the checkbox off, sets **Contact Gap** to `0.001` and **Contact Offset** to
`0.005`, and the panel draws the absolute pair only. A **Sand** group draws
**Contact Gap** alone, because its grain radius is the contact offset.

Both forms scale with the scene under **World Scaling**; see
[World Scaling](scene.md#world-scaling).

On **Solid**, **Shell** and **Rod** groups the gap you set is an upper value:
the [Local Gap Limit](#local-gap-limit) lowers it where mesh neighbors are
closer together than the gap allows.

(local-gap-limit)=

## Local Gap Limit

A contact gap that suits a coarse mesh can be too wide for a fine one. The
solver keeps every point of a mesh apart from the other points and edges of
that same mesh, its direct neighbors included: the corner across a triangle,
the next point along a strand. When the gap is wider than the distance to
those neighbors, they are inside each other's contact layer before anything
has moved. The contact force pushes them apart and the elasticity pulls them
back. A sheet lying still swells and resists bending, a garment looks
inflated and resists folding, and the effect grows as the mesh gets finer.
Gomez-Nogales et al. name this *contact locking* in
[Unlocking Thickness Modeling for Codimensional Contact Simulation](https://arxiv.org/abs/2510.10256)
(arXiv:2510.10256, 2025), and **Local Gap Limit** follows that paper; the
full reference is [below](#local-gap-limit-reference).

**Local Gap Limit** prevents that. It is a percentage, `25` by default, and
it caps the contact gap of each part of a mesh at that share of the room its
own neighbors leave. Where the cap is above the gap you set, nothing changes.
Where the mesh is fine, the gap is lowered there, and only there.

```{figure} ../../images/material_params/contact_gap_local_limit.svg
:alt: Local Gap Limit comparison. Each mesh point carries a dashed ring of half the contact gap. With the gap as authored, the rings on a fine mesh overlap their neighbors while the rings on a coarse mesh stay clear. With the limit at 25 percent, the rings on the fine mesh shrink until they do not touch and the rings on the coarse mesh are unchanged.
:width: 720px

Each dashed red ring is drawn at half the contact gap around a mesh point, so
two rings that overlap mean the two points are already in contact. **Left:**
one gap on every mesh. It is fine on the coarse mesh and too wide on the fine
one, where every point is in contact with its neighbors. **Right:** with
**Local Gap Limit** at 25, the gap on the fine mesh is lowered to a quarter of
the room between neighbors, and the coarse mesh keeps the gap it was given.
```

What the setting does:

- **It only lowers the gap.** The gap you set, absolute or ratio, stays the
  upper value. Each part of the mesh is solved at the smaller of that gap and
  its cap. A keyframed **Contact Gap** is held to the cap at every frame.
- **It works per triangle and per strand segment**, not per object. A garment
  that is dense around a collar and coarse across the back keeps its full gap
  across the back.
- **The room it measures** is the smallest triangle height around a point, or
  the shortest strand segment next to it, less twice the object's **Contact
  Offset** (its largest value, if you keyframe it), because the offsets of two
  neighbors take up the same distance.
- **Solid, Shell and Rod groups have it.** A **Static** or **PDRD** group
  draws no such row: a collider or a rigid body forms no contact pairs among
  its own faces, so nothing bounds its gap and it is used as set. A **Sand**
  group has no mesh to measure.
- **An object that Allow Self-Intersections reaches keeps the gap you set.**
  None of its parts is in contact with another part of itself, so the limit
  has nothing to bound. The row is still drawn for the group. See
  [Allow Intersections](#allow-intersections).
- **The range is 1 to 50.** A lower value caps the gap at a smaller share of
  the room, so a fine mesh gets a smaller gap. Lower it when a fine mesh still
  resists folding. A higher value lets a fine mesh keep more of the gap you
  set, for instance to keep stacked layers apart, and brings the gap closer to
  the distance at which neighbors start to touch.
- **It has no keyframe and no padlock.** It is read once, at the solve's
  **Starting Frame** (see [Keyframes](#keyframes)). **Paste Material
  Params** and a material profile overwrite it, and a profile stores it as
  `contact_gap_limit_percent`.

:::{note}
A **Solid**, **Shell** or **Rod** group starts with **Local Gap Limit** at
`25`. Any part of a fine mesh whose gap exceeds a quarter of its room is
therefore solved at that smaller gap, unless you raise the value.
:::

:::{admonition} Under the hood
:class: toggle

**How the cap is computed**

The limit is applied when the scene is built, to the mesh the solver
receives. For a **Solid**, that mesh is the surface of its tetrahedral mesh,
not the Blender mesh. For each triangle or strand segment:

```text
clearance = smallest size among the element and its neighbors
room      = clearance - 2 x contact offset
gap       = min(gap as set, limit x room)
```

A triangle's size is its smallest height and a segment's is its length, and an
element's neighbors are the elements that share a point with it. The neighbors
are included because the solver gives each point the average gap of the
elements around it, so a large triangle beside a small one would otherwise
carry its wider gap across the small one. An element whose clearance, counting
its neighbors, is no more than twice the offset has no room and keeps the gap
as set: no gap separates its neighbors, and they are either refused as
overlapping or exempted by an allowance such as
[Allow Existing Intersections](#allow-existing-intersections).

The add-on sends the percentage as the solver parameter `contact-gap-limit`,
a fraction between 0.01 and 0.5. The value `0` means no limit in the solver,
and the add-on's range starts at 1 percent. A notebook sets the same parameter
itself; see the [JupyterLab Python API](../../../jupyterlab_api/index.rst).
:::

### Thick Strands Need Room

On a **Rod** group the **Contact Offset** acts as the strand's radius: two
neighboring points of a strand are in contact once they are closer than the
gap plus twice the offset. The room between them is the segment length less
twice the offset, and that is the room the limit works with. A strand drawn with
segments only slightly longer than twice its offset leaves almost none: its
gap becomes a small fraction of the one you set, and the solve slows down.

Draw strands so that every segment is comfortably longer than twice the
offset. At the default of `25`, a strand keeps the gap you set once

```text
segment length >= 2 x Contact Offset + 4 x Contact Gap
```

At the **Rod** defaults (**Contact Gap** `0.001`, **Contact Offset** `0.005`)
that is `0.014` Blender units. A curve sampled unevenly has its shortest
segments where it turns, and those set the room for the segments next to them,
so give a curved strand an even spacing between its points before you assign
it.

(local-gap-limit-reference)=

### Reference

**Local Gap Limit** follows this paper, which names the effect *contact
locking*:

> Gonzalo Gomez-Nogales, Zhen Chen, Rosalie Martin, Elena Garces and
> Danny M. Kaufman. "Unlocking Thickness Modeling for Codimensional
> Contact Simulation." arXiv:2510.10256, 2025.
> <https://arxiv.org/abs/2510.10256>

The paper finds the pairs of mesh neighbors that start closer than the
distance at which contact engages, and lowers that distance for those
pairs. **Local Gap Limit** applies the same idea as a cap on the gap of
each triangle and strand segment.

(allow-intersections-settings)=

## Allow Intersections

By default the solver keeps every pair of surfaces apart: nothing passes
through anything, and a scene whose geometry already overlaps is refused
before the first frame is solved. Four settings let chosen objects pass
through each other instead. They are in the **Allow Intersections** box, near
the bottom of the group's **Material Params** box, above **Force Field Weight**
where that row is drawn. All four are off by default.

- **Allow Self-Intersections**: an object in this group may pass through
  itself. A falling cloth crosses its own folds instead of piling up on them.
  The [Local Gap Limit](#local-gap-limit) then leaves that object's gap as
  set.
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
**Sand** group has no **Allow Existing Intersections** checkbox. A group that
still has the checkbox on after its **Type** was changed to **Sand** is
refused at **Transfer** with a message naming the group. Change the **Type**
back, turn the checkbox off, and change it to **Sand** again.

With any of the first three checkboxes on, the box shows the line *Allowed
pairs pass through, with no contact*. With **Allow Existing Intersections** on,
it shows *Only pairs tangled at the start pass through*. On a **Static** group,
the checkboxes take effect only while the collider is part of the solved scene;
see [Only One Side Has to Allow It](#only-one-side-has-to-allow-it).

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

Turn the switch off to name the objects yourself:

1. Select the objects in the viewport.
2. Click the **+** button beside the list (**Add Selected Objects**). Only
   objects assigned to this group are accepted. The add-on reports the names
   it skipped, for example *Not in group 'Cloth': Body*, and reports *Already
   listed* when every selected object is already in the list.
3. To change the list, highlight an entry and click the **-** button
   (**Remove**), or click the trash button (**Remove All**) to empty it.

While the switch is off, only the listed objects get the setting. An empty
list reaches no object, and the panel says *No objects listed; this allowance
covers none*. When some entries do not reach the solver, for example an object
whose **Include** checkbox is off, the panel shows how many do, as *2 of 3
entries reach the solver*.

A list row reads *name (Missing)* when the object cannot be found, and *name
(Not in Group)* when the object exists but is not assigned to this group.
Neither reaches the solver. Remove such a row with the **-** button.

The four lists are independent, so one garment can pass through itself while
another only passes through the body. **Copy Material Params** and **Paste
Material Params** carry the four checkboxes and not the switches or the lists;
see [Copy / Paste](#copy--paste).

### Only One Side Has to Allow It

For **Allow Inter-Object Intersections** and **Allow Inter-Group
Intersections**, a pair of objects passes through when **either** of them has
the setting. Turn **Allow Inter-Object Intersections** on for a garment and it
passes through every other object, including the character body it is fitted
to, even though the body's group has the box off. If the garment should rest on
the body, leave that box off on the garment.

**Allow Self-Intersections** is asked of one object only: the object that
passes through itself is the one that needs it.

A **Static** collider's own checkboxes reach the solver only when the collider
is part of the solved scene: when it is animated, when it uses **Apply Soft
Constraints**, or when it is one end of a cross-stitch. A collider that is none
of those never moves and is a collision surface only, so the checkboxes on its
group have no effect. A moving object still passes through it when the moving
object's group has **Allow Inter-Object Intersections** or **Allow Inter-Group
Intersections** on, and through the places where the two start overlapping
when its group has **Allow Existing Intersections** on. Setting one of those on
the moving object's group works in every case, since it does not depend on how
the collider is driven.

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

**Allow Existing Intersections** has no column here because it names no kind of
pair. It covers whichever pairs start the simulation overlapping, of any kind
in the first three rows, and never the invisible walls and spheres.

"Self" here means one mesh object, not one group. A group holding two meshes
holds two objects, so the pair those two form is an inter-object pair, which
only **Allow Inter-Object Intersections** covers. When a group holds more than
one object and has **Allow Self-Intersections** or **Allow Inter-Group
Intersections** on without **Allow Inter-Object Intersections**, the panel adds
the line *Two objects in one group are an inter-object pair*.

That difference is what **Allow Inter-Group Intersections** is for. Put several
garments layered on one another in one group and turn it on there: the layers
keep colliding with each other, while all of them pass through a body or prop
assigned to another group.

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

An allowance does not separate an overlap. The allowed pair has no contact, so
nothing pushes the two surfaces apart: they stay crossed until the object's own
motion carries them out, if it does. Where the geometry should end up apart,
separate it before the run instead: see
[Mesh Cleaning](../scene/mesh_cleaning.md). With all four settings off, the
solver keeps every pair apart and never lets anything intersect.

Where only a region you have pinned should pass through, the per-pin
[Allow Intersections Here](../constraints/pins.md#allow-intersections-here)
option is narrower than any group setting.

:::{admonition} Under the hood
:class: toggle

The check at scene build and the checks the solver runs while it steps all
consult the allowance, so a scene that builds is not then stopped by the first
step. It is evaluated per pair of elements, whether that is two triangles, a
rod segment against a triangle or against another rod segment, or two grains of
a Sand cloud, and it decides whether that pair takes part in contact at all. An
allowed pair is left out of the contact force, out of the continuous-collision
test that limits how far each step may move, and out of both intersection
checks. Every other pair keeps all of them, so the guarantee that nothing
intersects holds for every pair you did not allow.

**Allow Inter-Group Intersections** compares the groups the two objects are
assigned to. A **Static** collider that is not part of the solved scene counts
as a different group from every object.

The start-pose check that rejects elements closer than their contact offset
skips allowed pairs as well, since an allowed pair has no contact offset to
keep. Every other pair is still checked. The other geometry checks, such as a
rod segment shorter than its own contact offset, are unrelated to these
settings and still apply.
:::

### Allow Existing Intersections

A garment posed by a rig often arrives with a fold that passes through itself,
such as an armpit, or with a sleeve that dips into the body. The scene is then
refused before the first frame. The three settings above let it run only by
giving up contact for the whole object, everywhere. **Allow Existing
Intersections** gives up contact only where the overlap already is.

When it is on, the build finds every place where an object of this group starts
the simulation overlapping something, or sitting closer to it than the
[contact offset](#contact-gap-and-contact-offset) allows, and lets exactly
those places pass through each other for the whole simulation, together with
the faces right next to them. Everything else keeps full contact: the rest of
the garment still collides with itself, with the body, and with every other
object. As with the other settings, only one of the two sides needs it on, and
it covers an object with itself, two objects, and an object against a
**Static** collider. It does not affect the invisible walls and spheres, and it
is not available on **Sand**.

:::{important}
This setting leaves a tangle in place; it does not resolve one. The overlapping
places are left alone for the whole run, so a fold that starts crossed stays
crossed unless the motion carries it out. Nothing new is allowed to pass
through: a pair that did not start overlapping never intersects, and if one
ever would, the run stops with an intersection error. Where the start pose can
be fixed, fixing it is still the better choice.
:::

After a build with this setting on, the viewport shows what was let through. On
the frame the simulation starts from, the overlapping places are tinted light
blue, with a label such as *19 Existing Intersections Allowed*. The highlight
is drawn on the starting frame only, because it marks where the geometry was at
the start. When nothing overlaps, nothing is drawn.

**Apply to All Objects** narrows it to named objects in the same way as the
other settings. Leaving the list empty reaches no object, so the tangled start
pose is refused again.

:::{admonition} Under the hood
:class: toggle

The overlapping places are found once, by the same check that would otherwise
refuse the scene, on the start pose. For every pair of elements that check
finds, and where either side's object has the setting on, every vertex of one
element is linked to every vertex of the other. During the solve, two elements
whose vertices are linked are treated like two faces that share a corner: no
contact force, no limit on the step from the collision test, and no
intersection report. Because the link is on vertices, a face sharing a corner
with an overlapping face is covered too. The links are fixed when the scene is
built and last for the whole run.

The solver never adds a link of its own. The start-pose check and the solver's
own check measure positions at slightly different precision, so a pair that
barely touches can pass one and fail the other. The run then stops before the
first frame with an error that names the pair; moving the two apart, or
further into each other, makes both checks agree.
:::

## Loose-Edge Stitches

A loose edge is an edge that belongs to no face. When you click **Transfer**,
each loose edge of a **Shell** or **Solid** mesh becomes a stitch constraint
with the stiffness of the group's **Stitch Stiffness** row. That row is the
only control: no button creates or removes these stitches. See
[Shell-Specific](#shell-specific) and [Solid-Specific](#solid-specific) for the
row and for which vertices a stitch may end on.

```{figure} ../../images/material_params/loose_edge_stitch.png
:alt: Two subdivided square Shell patches stacked with a gap, connected by vertical red tubes. Each tube is an edge that belongs to no face and becomes a stitch constraint.
:width: 500px

Two subdivided square **Shell** patches joined by vertical **loose edges**
(rendered here as red tubes). The patches are separate face regions. The
connecting edges belong to no face, so **Transfer** turns each one into a
stitch constraint with the stiffness set by `stitch_stiffness`.
```

:::{admonition} Under the hood
:class: toggle

On a **Solid**, the mesh the solver receives is a tetrahedral mesh, so a
Blender vertex index does not name the same point there. Each end of a loose
edge is placed in world space and projected onto the tetrahedral surface as a
barycentric point on a surface triangle, and the row becomes a point-to-point
stitch. An end that finds no triangle stops the build and names the object.
:::

## Material Profiles

A **material profile** is a TOML file of named entries, each holding the
material values of one group. **Save Material Profile** in the profile row
writes the file and **Open Material Profile** loads it. The profile row, and
what applying an entry changes, are described under
[Profile Buttons](#profile-buttons-open--clear--reload--save).

A single file can hold any number of entries, for any mix of group types. For
example:

| Entry    | **Type**   | Holds                                         |
| -------- | ---------- | --------------------------------------------- |
| `Tent`   | **Shell**  | A stiff cloth with a strain limit.            |
| `Gown`   | **Shell**  | A soft, light cloth.                          |
| `Block`  | **Solid**  | A soft solid.                                 |
| `Rope`   | **Rod**    | A strand.                                     |
| `Static` | **Static** | A collider's friction.                        |

These names are examples. They are unrelated to the bundled
[presets](fabric_presets.md) in the **Preset** dropdown.

Choosing an entry sets every key it holds. A saved entry holds the group's
**Type** and overlay color, so the group changes **Type** when the entry's
differs. Padlocks do not protect rows from a profile.

:::{note}
A profile stores group-level material values only. It does not carry object
assignments, pin vertex groups or per-object velocity overrides: it describes
a material, not a scene.
:::

### Example TOML Stanza

A saved entry lists every material key of the group, whatever its **Type**, and
the keys are the property keys used in the tables on this page. The excerpt
below is shortened. Create profile files with the **Save Material Profile**
icon, not by hand; see
[Profile Buttons](#profile-buttons-open--clear--reload--save) for how it
behaves.

```toml
[MyCloth]
object_type = "SHELL"
shell_model = "BARAFF_WITKIN"
shell_density = 1.0
shell_young_modulus = 1000.0
young_mod_density_normalized = true
shell_poisson_ratio = 0.35
friction = 0.5
use_group_bounding_box_diagonal = true
contact_gap_rat = 0.001
contact_offset_rat = 0.0
contact_gap_limit_percent = 25.0
bend = 10.0
shrink_x = 1.0
shrink_y = 1.0
enable_strain_limit = true
strain_limit_percent = 5.0
stitch_stiffness = 1.0
# ... the remaining material keys of the group
```

Only the keys present in an entry are applied, and the other values keep their
current value on the group. A trimmed entry is therefore valid: a `Static`
entry, for instance, can hold just a type and a friction value.

```toml
[Static]
object_type = "STATIC"
friction = 0.8
```

## Blender Python API

Most rows of the **Material Params** box can be set from Python through each
group's `.param` attribute. The attribute is whitelisted: the accepted names
are listed in the
[Blender Python API Reference](../../integrations/python_api_reference.rst),
and any other name raises `AttributeError`. A change from Python appears in the
panel at once, and a change in the panel is visible from Python.

The examples import `solver` directly, which works when the add-on is in the
**User Default** repository. For an add-on installed from a remote repository,
use the lookup in
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# Young's Modulus values are in Pa/ρ while Density-Normalized (Pa/ρ) is on,
# which is the default.
cloth = solver.create_group("Cloth", "SHELL")
cloth.param.shell_density       = 0.5
cloth.param.shell_young_modulus = 50.0
cloth.param.friction            = 0.3
cloth.param.bend                = 0.5

# Solid body with Stable NeoHookean, an absolute contact gap of 0.001
# Blender units, and a Local Gap Limit of 10 percent.
body = solver.create_group("Body", "SOLID")
body.param.solid_model         = "STABLE_NEOHOOKEAN"
body.param.solid_density       = 1100.0
body.param.solid_young_modulus = 5000.0
body.param.use_group_bounding_box_diagonal = False
body.param.contact_gap         = 0.001
body.param.contact_gap_limit_percent = 10.0

# Static collider: friction and contact settings.
floor = solver.create_group("Floor", "STATIC")
floor.param.friction        = 0.8
floor.param.contact_gap_rat = 0.001
```

Rows outside the whitelist are set in the panel or with the MCP tool
`set_group_material_properties`. They include **Apply Soft Constraints** with
its **Stiffness**, **Bending Stiffness (Warp)** and **(Weft)**, **Rayleigh
Damping**, **Density-Normalized (Pa/ρ)** (`young_mod_density_normalized`) and
the four **Allow** settings (see
[Allow Intersections](#allow-intersections)).

The locks, the reference rest angle and the spatial material maps are not
reachable through `.param` or a method of the group either. The MCP tools that
edit them are named under
[Lock Translation and Lock Rotation](#lock-translation-and-lock-rotation),
[Rest Angle and Bend Plasticity](#rest-angle-and-bend-plasticity) and
[Maps From MCP](#maps-from-mcp). A script calls an MCP tool as
`solver.<tool>(...)`; see
[Fallback: Raw Operator Dispatch](../../integrations/python_api.md#fallback-raw-operator-dispatch)
for what such a call returns and how it takes list arguments.
