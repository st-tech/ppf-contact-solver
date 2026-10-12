# 🧭 Tour of the UI

All panels are in the **3D Viewport** sidebar (<kbd>N</kbd>), on the **ZOZO's
Contact Solver** tab. This page names every panel and what it holds, in the
order the panels appear. Most sections link to the page that explains them in
full.

## Backend Communicator

The main panel, where you connect to a solver and watch its status.

```{figure} ../images/tour/backend_communicator.png
:alt: Backend Communicator panel with the Open Profile row, the Connection box set to SSH and its fields, an Advanced box, the Connect and Disconnect buttons, and the Connect button outlined in red
:width: 500px

The Backend Communicator panel, with **Connect** (the button that opens the
connection to the solver) outlined.
```

From top to bottom:

1. **Profile row.** **Open Profile** loads a connection profile file. Once a
   file is loaded, the row shows a profile dropdown with icon buttons to open,
   clear, reload, and save. See
   [Connection Profiles](../connections/profiles.md).
2. **Connection box** (collapsible), holding:
   - **Type**: **SSH**, **SSH Command**, **Docker**, **Docker over SSH**,
     **Docker over SSH Command**, **Windows Native**, **macOS Native**, or
     **Linux Native**.
   - The fields the chosen type needs. The native types ask for **Solver
     Path** and show *Solver path valid* once it resolves to a solver. See
     [Connections](../connections/index.md).
   - The device controls described below the list.
   - **Project Name**, followed on the Docker types by a **Docker Port**
     field.
   - **Connect** and **Disconnect**, then **Start Server on Remote** and
     **Stop Server on Remote**, followed by a one-line hint naming the next
     step.
3. **Status line**, which reads `Status:` followed by the current state. While
   a connect or a server start is in progress, a **Cancel** button under it
   cancels the attempt.
4. **Error lines.** When a native type fails to connect, or any type reports
   a port-in-use or already-running error or a protocol version mismatch, a
   **Force Terminate Process** button appears beside the error, outside the
   Connection box so that collapsing the box does not hide it (an SSH or
   Docker type draws it only while connected). On a native type it can end any
   program listening on the server port, not only the solver, so read the line
   under the button first. See
   [Port Already in Use](../connections/index.md#port-already-in-use).
5. **Update Stat**, **Show Console**, and the **Debug Options** checkbox on
   one row.
   - **Update Stat** asks the server for its current status and refreshes the
     panel. It is grayed out unless you are connected, the server is running,
     and the add-on is not in the middle of another operation such as a
     transfer or a fetch.
   - **Show Console** opens the add-on console in a new Blender window: a
     **Text Editor** showing a text named *ZOZO's Contact Solver*, with line
     numbers, scrolled to the newest lines. If a window already shows it, that
     window scrolls to the newest lines instead. The console holds the add-on's
     own log lines, such as connection and server messages, install output,
     and the full report behind a failure. Each line starts with a
     `[YYYY-MM-DD HH:MM:SS]` time stamp. The console keeps the last **Max
     Console Lines** lines; see
     [Console](../troubleshooting.md#console-max-console-lines-and-console-log-export).
   - **Debug Options** unlocks the developer tools described under
     [Debug Tools](#debug-tools).
6. As the connection and the run supply them: **Remote Hardware**, **Realtime
   Statistics** during a run (**Average Statistics** once it ends), and
   **Scene Info**. During a run the panel also shows a progress bar with
   **Save and Quit** and **Terminate** buttons.

The Connection box also holds the device controls: the **GPU** / **CPU**
toggle (the **Compute Device** setting), **GPU Backend** and the **GPU** picker.
Which of them is drawn, and what each one picks, depends on the connection type
and the builds the solver folder or host holds.

See {ref}`Choosing the Build <choosing-the-build>` and
{ref}`Picking a GPU <gpu-picker>`.

## Solver

The buttons that drive a simulation, two to a row: **Transfer** and **Update
Params on Remote**; **Run** and **Resume**; **Fetch All Animation** and
**Delete Remote Data**; **Bake Animation** and **Bake Single Frame**; and
**Clear Local Animation**. A line under the buttons names the next step, for
example *Click "Transfer" to upload data*. See
[Running the Simulation](../workflow/sim/simulating.md) and
[Baking Animation](../workflow/sim/baking.md).

Below the buttons are two always-visible boxes:

- **Deformations** holds **Re-capture All Deformations**, which re-records
  every deforming Static collider and every animated pin in the scene in one
  pass, and **Clear All Deformations**, which deletes every one of those
  recordings, including any left behind by an object that was deleted or
  taken out of its group.
- **Export** holds **Export USD** and **Export Alembic (ABC)**, which write
  the simulated result as a point cache another application can read, once
  every frame has been fetched. See
  [Exporting USD and Alembic Caches](../workflow/sim/exporting.md).

The **JupyterLab** and **MCP Server** collapsible sections are also in this
panel; see [JupyterLab](../workflow/sim/jupyterlab.md) and
[MCP](../integrations/mcp.md).

```{figure} ../images/tour/solver.png
:alt: Solver panel, Transfer button highlighted
:width: 500px

The Solver panel, with **Transfer** (the button that uploads geometry
and parameters to the solver) highlighted.
```

## Scene Configuration

Global solver inputs. The fields at the top of the panel are:

- A profile row, like the one on the Backend Communicator panel, for scene
  presets. See [Scene Parameters](../workflow/params/scene.md).
- **Take FPS from Scene**, which replaces the **FPS** field with the scene's
  frame rate, and **Time Scale**.
- **Take Starting Frame from Scene**, which replaces the **Starting Frame**
  field with the scene's start frame.
- **Frame Count**, **Step Size**, **Min Newton Steps**, **Air Density
  (kg/m³)**, **Air Friction**, **World Scaling**, and **Gravity (m/s²)**.
- **Preview Direction**, and **Inactive Momentum Frames** (enabled only while
  a Shell group exists).

Below them are six collapsible boxes: **Save and Checkpoints**, **Wind**,
**Force Fields**, **Invisible Colliders** (walls and spheres), **Linear
System Solver**, and **Advanced Params**. See
[Force Fields](../workflow/params/force_fields.md) for the force-field box.

Six settings can be keyframed: **Gravity**, the **Wind** fields, **Air
Density**, **Air Friction**, **Vertex Air Damping** (in **Advanced Params**)
and **Step Size**. To keyframe one, hover over its field and press <kbd>I</kbd>,
or right-click it and choose **Insert Keyframe**. The animatable rows of
**Material Params** also show Blender's keyframe button. See
[Dynamic Parameters](../workflow/params/dynamic.md).

```{figure} ../images/tour/scene_configuration.png
:alt: Scene Configuration panel with the scene fields above collapsed boxes named Save and Checkpoints, Wind, Invisible Colliders, Linear System Solver, and Advanced Params
:width: 500px

The Scene Configuration panel. Every field here applies to the whole
scene; the collapsible boxes below the flat fields expand to reveal
more inputs. The capture shows five of the six boxes and omits **Force
Fields**, which sits between **Wind** and **Invisible Colliders**.
```

## Dynamics Groups

Up to 32 groups. **Create Group** adds a group and **Delete All Groups**
removes them all. Each group is its own box, whose header has a
duplicate-group button. A group box holds:

- **Name** and the type dropdown (**Solid**, **Shell**, **Rod**, **Static**,
  **PDRD**, or **Sand**), then **Overlay Color**.
- The assigned-objects list, with **Add Selected Objects** and **Remove
  Object**.
- **Bake Animation** and **Bake Single Frame** for the selected object, and
  **Delete Group**.
- Collapsible **Pins** (**Transform** on a Static group, **Pins & Motion** on
  a PDRD group), **Stats**, and **Material Params** boxes. Pin operations are
  **Move By**, **Spin**, **Scale**, and **Torque**.

See [Object Groups](../workflow/scene/object_groups.md) for what each type
means, [Pins and Operations](../workflow/constraints/pins.md), and
[Material Parameters](../workflow/params/material.md).

```{figure} ../images/tour/dynamics_groups.png
:alt: Dynamics Groups panel, Create Group button highlighted
:width: 500px

The Dynamics Groups panel in its empty state, with **Create Group**
(the button that allocates a new group slot) highlighted. Each created
group appears as its own box below.
```

A **Sand** group simulates a cloud of grain centers rather than a
surface, so its box has an extra **Convert To Solid Particle Mesh**
button above **Delete Group**. It fills the active solid mesh with grains
at the **Grain Radius** you type into the dialog and discards the
original faces, which is what a Sand group takes as input. The radius is
fixed at that moment (the grain spacing is derived from it) and reads
back as a grayed-out field on the group afterward. See
[Sand-Specific](../workflow/params/material.md#sand-specific).

## Snap and Merge

Moves one object next to another and records the pair, so that the solver
stitches the two objects together at **Transfer**. The panel is collapsed by
default; click the header to expand.

Choose the two objects in the **Object A (moves)** and **Object B (target)**
fields of the **Snap To Nearby Vertices** box (each field has an eyedropper
button), then press **Snap A to B**.

- **Snap A to B** moves Object A (or Object B, when A is a Static collider)
  as a whole, without deforming it, until the two closest points are a small
  contact gap apart. The pair then appears under **Merge Pairs**.
- **Merge Pairs** is drawn once the first pair exists. It lists the recorded
  pairs, with **Re-snap** and **Remove** buttons and, where it applies, a
  stitch stiffness field.
- **Post Snap Exactly** is a single checkbox for all pairs.

See [Snap and Merge](../workflow/constraints/snap_merge.md).

```{figure} ../images/tour/snap_and_merge.png
:alt: Snap and Merge panel, Snap A to B button highlighted
:width: 500px

The Snap and Merge panel, with **Snap A to B** (the button that moves
Object A up to a small gap from Object B) highlighted.
```

## Mesh Repair

Mesh preparation that belongs to no single group, collapsed by default.
**Mesh Cleaning** scans the selected mesh objects for geometry the solver
rejects and offers a repair button for each finding that has a safe repair.
The six repairs are **Merge by Distance**, **Remove Loose Vertices**,
**Dissolve Degenerate**, **Delete Duplicate Faces**, **Triangulate**, and
**Recalculate Outside**. The first three change the vertex count, so they ask
for confirmation first and offer to clear the caches the change invalidates.
**Symmetric Triangulate**, in its own box below, pokes each face into a center
fan so a symmetric mesh folds symmetrically; that adds one vertex per face and
it does not ask first. **Intersection Correction**, in a third box, moves
overlapping **Solid** and **Pdrd** objects apart so that **Transfer** accepts
the scene, and **Resolve All Intersections**, in a fourth, also reshapes
**Shell** cloth that passes through itself or through another surface. See
[Mesh Cleaning](../workflow/scene/mesh_cleaning.md).

## Visualization

Five checkboxes, each hiding one overlay category in the viewport: **Hide all
pins**, **Hide all directional arrows**, **Hide all overlaid colors**, **Hide
all snaps**, and **Hide all pin operations**. They change only what the
viewport draws. The panel is collapsed by default.

```{figure} ../images/tour/visualization.png
:alt: Visualization panel, Hide all pins toggle highlighted
:width: 500px

The Visualization panel, with **Hide all pins** highlighted.
```

## Object Statistics

The last panel in the tab, collapsed by default: a per-object readout of what
the solver measured on the frame you are looking at. Pick an object from the
**Object** menu, which lists every object included in an active group (Static
colliders among them). The rows follow the playhead and appear in this order:

| Rows | Contents |
| ---- | -------- |
| **Frame**, **Time (s)** | The frame under the playhead and its simulated time. |
| **Location (m)**, **Velocity (m/s)**, **Acceleration (m/s²)**, **Angular Velocity (rad/s)**, **Angular Axis** | The vector rows every object has. |
| **Volume (m³)**, **Surface Area (m²)**, **Rod Length (m)** | The measures the object's geometry supports: volume for a solid or a closed shell, surface area for anything with faces, rod length for a rod. Each is followed by its percentage of the same measure on the first frame of the run. |
| **Speed (m/s)**, **Acceleration Magnitude (m/s²)**, **Angular Speed (rad/s)** | The scalar rows. |
| **Contact Count** | Last in the list. |

**Export CSV** picks one of those values from a menu and writes it for every
fetched frame to a `.csv` file. An object with no recorded statistics reads
*Statistics unavailable; rerun the simulation*. See
[Reading Per-Frame Statistics](../workflow/sim/simulating.md#reading-per-frame-statistics)
for where the numbers come from and when they go away.

## Debug Tools

The **Debug** tools (Shell Calls, Data Transfer Tests, Options, Console
Log Export, GitHub Repo on Remote / Local, UUID Migration, Render, and
the Add-on Local Debug Server) are hidden behind the **Debug Options**
checkbox on the Backend Communicator panel. What each tool does, what it
needs before it will run, and the problem it exists to solve are described
under [Debug Options](../troubleshooting.md#debug-options).

```{figure} ../images/tour/debug_options.png
:alt: Backend Communicator panel with Debug Options toggled on, Debug Options checkbox highlighted and the debug section expanded below
:width: 500px

With **Debug Options** toggled on (the checkbox on the **Update Stat**
/ **Show Console** row of the Backend Communicator panel), a debug
section appears below exposing Shell Calls, Data Transfer Tests,
Options, Console Log Export, GitHub Repo on Remote / Local, UUID
Migration, Render, and the Add-on Local Debug Server controls.
```
