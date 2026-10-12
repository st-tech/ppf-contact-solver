# 🧵 Fabric Presets

The **Preset** dropdown applies a bundled material to a group in one click.
**Shell** groups list six fabrics, calibrated against published fabric drape
measurements. **Solid** groups list five soft solids. **Rod**, **PDRD**,
**Static** and **Sand** groups have no presets and no **Preset** row.

## Applying a Preset

1. Set the group's **Type** to **Shell** or **Solid**.
2. Find the **Preset** dropdown in the group's **Material Params** box, above
   the profile row. It reads **Select Preset...** and stays visible while the
   box is collapsed.
3. Pick a preset. The values apply at once, and the dropdown returns to
   **Select Preset...**.

A preset sets only the rows named in the tables below, and it turns on
**Density-Normalized (Pa/ρ)**. It never changes the group's **Type**. A row
with a closed padlock keeps its value; see
[Padlocks](material.md#padlocks).

From a script, the MCP tools `list_material_presets` and
`apply_material_preset` list and apply the same presets; see
[MCP Scene Setup](../../integrations/mcp_scene_setup.md). A preset made for
another **Type** than the group's is refused. The result of
`apply_material_preset` lists the rows that hold the preset's values under
`written` and the rows a closed padlock left alone under `kept_locked`.

## Shell Presets

Every fabric preset sets **Model** to **Baraff-Witkin**, **Density** to
1.0 kg/m², **Enable Strain Limit** on, and the values in this table:

| Preset      | Young's Modulus (Pa/ρ) | Poisson's Ratio | Bend Stiffness | Friction | Strain Limit (%) |
| ----------- | ---------------------- | --------------- | -------------- | -------- | ---------------- |
| **Silk**    | 500                    | 0.4             | 1.42           | 0.25     | 6                |
| **Flag**    | 1000                   | 0.4             | 0.83           | 0.30     | 4                |
| **Cotton**  | 5500                   | 0.35            | 4.3            | 0.35     | 5                |
| **Wool**    | 2000                   | 0.4             | 3.67           | 0.40     | 8                |
| **Denim**   | 10000                  | 0.25            | 10.0           | 0.50     | 3                |
| **Leather** | 13000                  | 0.4             | 1.8            | 0.50     | 2                |

- **Silk** and **Flag**: very drapey, with low friction. **Flag** models a
  lightweight synthetic banner cloth.
- **Cotton** and **Wool**: medium stiffness and drape.
- **Denim**: stiff and the least drapey of the set, so it holds its shape.
- **Leather**: barely stretchy. It droops more than Cotton or Wool in the
  cantilever test.

The six fabrics share one **Density**, so they differ in stiffness, Poisson's
ratio, bending, strain limit and friction, not in mass. **Density** can be
changed freely: while **Density-Normalized (Pa/ρ)** is on, it changes inertia
and contact and does not change the draped shape.

A fabric's **Young's Modulus** is a calibrated membrane stiffness in the
density-normalized convention (Pa/ρ), not a textbook modulus in Pa. On a
**Shell** group whose **Shrink X** or **Shrink Y** is not `1`, the strain limit
that a fabric preset turns on has no effect. The panel shows *Shrink/extend
disables strain limiting*.

## Solid Presets

Every solid preset sets **Model** to **ARAP**, **Density** to 100 kg/m³,
**Density-Normalized (Pa/ρ)** on, and the values in this table:

| Preset                                  | Young's Modulus (Pa/ρ) | Real Young's modulus | Poisson's Ratio | Friction |
| --------------------------------------- | ---------------------- | -------------------- | --------------- | -------- |
| **Rubber** (vulcanized rubber)          | 2727.3                 | about 3 MPa          | 0.49            | 0.85     |
| **Silicone** (silicone elastomer, PDMS) | 1941.7                 | about 2 MPa          | 0.49            | 0.45     |
| **Foam** (flexible polyurethane foam)   | 1111.1                 | about 50 kPa         | 0.30            | 0.50     |
| **Sponge** (dry cellulose sponge)       | 3000.0                 | about 0.3 MPa        | 0.20            | 0.50     |
| **Jelly** (gelatin or ballistic gel)    | 144.2                  | about 150 kPa        | 0.49            | 0.10     |

Unlike the fabrics, the solid presets are not calibrated by simulation. Each
**Young's Modulus** is the real material's modulus divided by its real
density, which is the density-normalized form. The five presets share one
**Density** instead of each material's real density, because a wide spread
of masses among materials mixed in one scene makes the system
ill-conditioned. **Density** can be changed freely: while
**Density-Normalized (Pa/ρ)** is on, it changes inertia and contact and does
not change the static deformed shape.

## Calibration Report

The report linked below shows, for every fabric, how it behaves in standard
fabric tests, rendered from simulations:

- a **drape** over a disc (how far it collapses into folds),
- a **cantilever bend** (how far a held strip droops under its own weight), and
- its **stretch** parameters (how easily it pulls in its plane: Silk and Wool
  stretch more easily than Denim and Leather).

:::{raw} html
<p><a class="reference external" href="../../../fabric-report/index.html"><strong>Open the fabric calibration report &raquo;</strong></a></p>
:::

:::{note}
The report opens as a standalone page. Each fabric is shown with a top-down
and an angled view of its drape, a side view of its cantilever bend, and a
summary table of its parameters. The drape values follow the Cusick drape
method, and the cantilever bend follows ASTM D1388.
:::
