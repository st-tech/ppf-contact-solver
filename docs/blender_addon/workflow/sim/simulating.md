# 🚀 Running the Simulation

Once the scene has its groups, parameters, pins and colliders, you run it
from the **Solver** panel: **Transfer** → **Run** → **Fetch All
Animation**. **Resume**, **Update Params on Remote** and **Clear Local
Animation** support iteration. The status line, the transfer, solve and
fetch progress bars and the **Terminate**, **Save and Quit** and **Abort**
buttons are on **Backend Communicator**, directly above the **Solver**
panel. The progress bar for a bake or a capture is on the **Solver** panel,
with an **Abort** button of its own.

## The Solver Panel

The **Solver** panel is in the add-on tab of the 3D viewport sidebar
(`N`), directly below **Backend Communicator**. Connection settings and
the **Connect** and **Disconnect** buttons are on **Backend
Communicator**, not here.

```{figure} ../../images/simulating/solver_connected.png
:alt: Solver panel right after the server starts, with only Transfer enabled and the notice Click "Transfer" to upload data
:width: 500px

The **Solver** panel right after the server comes up. Only **Transfer**
is enabled. The buttons after it stay grayed out until the solver host
holds the scene.
```

From top to bottom, the panel holds:

1. **Buttons.** Five rows: **Transfer** and **Update Params on Remote**;
   **Run** and **Resume**; **Fetch All Animation** and **Delete Remote
   Data**; **Bake Animation** and **Bake Single Frame**; **Clear Local
   Animation**. [The Buttons](#the-buttons) lists what each does and when
   it is enabled.
2. **Notices.** Lines drawn only while they apply, listed in the table
   below. Most sit under the buttons; the table says where the others are.
3. **A progress bar with an Abort button**, drawn while a bake or a
   capture runs.
4. **A Deformations box** with **Re-capture All Deformations** and
   **Clear All Deformations**.
5. **An Export box** with **Export USD** and **Export Alembic (ABC)**.
6. **The JupyterLab and MCP Server boxes.** **JupyterLab** holds the
   notebook export, open and delete buttons and the Jupyter port; see
   [JupyterLab](jupyterlab.md). **MCP Server** starts and stops the
   local MCP server and sets its port; see
   [MCP Server](../../integrations/mcp.md).

The notices are:

| Notice | What it means |
| ------ | ------------- |
| *cbor2 is missing; Transfer and Run cannot encode* | The `cbor2` package the add-on needs to encode the scene is not installed. Install it from **Backend Communicator**. Drawn at the top of the button box. |
| *Click "Transfer" to upload data* | The solver host holds no scene yet. |
| *Clear local animation before running* | The scene still holds fetched animation, which is the only thing keeping **Run** grayed out. Press **Clear Local Animation**. |
| *N frames unfetched. Press "Fetch All Animation".* | The solver host has more frames than the scene. |
| *Data path: ... does not exist.* | The folder with the fetched animation is missing, usually after the `.blend` was renamed or copied. A button named after the two folders, such as **Migrate data/project/ → data/project_baked/**, appears when the old folder can be copied; see [Baking Animation](baking.md). |
| *Stale cache on 'Cube' (not in any group)* | Duplicating a simulated object copies its `ContactSolverCache` modifier onto a copy that belongs to no group. **Remove Stale Cache** deletes those modifiers and leaves the cache files in place. |
| *No dynamics to simulate* | No active group of a dynamic type holds an object, so **Transfer** is grayed out. Drawn below the button box. |

## The Buttons

The **Solver** panel buttons:

| Button | What it does | Enabled when |
| ------ | ------------ | ------------ |
| **Transfer** | Uploads the scene geometry and parameters together and builds the remote project. The status line ends at *Ready to Run*. It starts the remote project over: simulated frames and saved states already on the solver host are removed, so **Resume** has nothing to offer afterward. Cached results such as tetrahedralization are kept when they still apply to the new scene; **Delete Remote Data** removes them. | The server is running, nothing else is in progress, and an active group of a dynamic type holds an object. |
| **Update Params on Remote** | Uploads only the parameters, then rebuilds. Geometry is not re-sent. Simulated frames and saved states stay on the solver host. | After a **Transfer**, while no solve is running. |
| **Run** | Compares the scene with the last upload, refuses if geometry or parameters differ, then starts the solve. | The status line reads *Ready to Run*, *Resumable* or *Simulation Failed*, no solve or bake is running, and the scene holds no fetched animation. Press **Clear Local Animation** first when it does. |
| **Resume** | Opens a picker of the saved states on the solver host and continues the run from the one you choose. No re-upload, no rebuild. See [Resuming a Run](#resuming-a-run). | The status line reads *Resumable* or *Simulation Failed*, and at least one saved state exists. |
| **Fetch All Animation** | Downloads the frames the scene does not have yet and applies them to the simulated objects. See [How Animation Plays Back](#how-animation-plays-back). | The solver host has simulated frames, and no solve or bake is running. |
| **Clear Local Animation** | Removes the fetched animation from the simulated objects (the `ContactSolverCache` modifiers and the cache files behind them), forgets which frames were fetched, and sets the playhead to the **Starting Frame**. Captured pin animation is kept. The solver host is not touched. | The scene holds fetched animation and nothing is running. |
| **Delete Remote Data** | Wipes the project data on the solver host, cached results included. The next step is **Transfer**. | The solver host holds a project and nothing is running. |
| **Bake Animation** | Converts the fetched animation into shape keys (meshes) or control-point keyframes (rods). See [Baking Animation](baking.md). | The scene holds fetched animation, and no solve or bake is running. |
| **Bake Single Frame** | Writes the current pose into the object's geometry and sets the playhead to the **Starting Frame**. It creates no keyframes. See [Baking Animation](baking.md). | The scene holds fetched animation, and no solve or bake is running. |
| **Re-capture All Deformations** | Re-records every deforming **Static** collider and every animated pin in the scene, colliders first, with a progress bar and an **Abort** button while it runs. See [Static Objects](../scene/static_objects.md). | Something can be captured, and no capture or bake is running. |
| **Clear All Deformations** | Deletes every captured deformation in the scene, including recordings left behind by an object that was deleted or taken out of its group. | A recording exists, and no capture or bake is running. |
| **Export USD**, **Export Alembic (ABC)** | Write the fetched animation to a point cache for another application. See [Exporting USD and Alembic Caches](exporting.md). | A simulated mesh is in the view layer, and no solve, bake or capture is running. |

The buttons on **Backend Communicator** that control a transfer, a fetch
or a running solve:

| Button | What it does | Drawn when |
| ------ | ------------ | ---------- |
| **Terminate** | Stops the running solve at once. No state is written. The frames simulated so far stay on the solver host and can be fetched. The status line then reads *Resumable* if a saved state exists, otherwise *Ready to Run*. | A solve is running. |
| **Save and Quit** | Asks the running solve to write a state and stop. The status line then reads *Resumable*, and **Resume** lists the new state. | A solve is running. |
| **Abort** | Interrupts the transfer or fetch in progress. It does not stop a running solve. | Data is being sent, received or fetched, under the progress bar. |
| **Cancel** | Interrupts the transfer in progress, like **Abort**. | Data is being sent or the scene is building, under the status line. |

Neither **Terminate** nor **Save and Quit** disconnects from the solver
host or stops the server. **Stop Server on Remote** stops the server.

```{figure} ../../images/simulating/solver_state_machine.svg
:alt: State diagram of the Solver panel showing which buttons are enabled in each state from Waiting for Data to fetched animation
:width: 820px

Which buttons are enabled along one run. Blue arrows are the forward
flow; purple dashed arrows are loops and recovery. Read the **Complete**
column as a run that finished with nothing fetched into the scene yet,
for example after reconnecting. When frames streamed into the scene
during the run, **Run** is grayed out and the bake buttons and **Clear
Local Animation** are enabled instead; see [Run](#run). The figure does
not draw the *Resumable* state: after **Save and Quit**, the status line
reads *Resumable* and **Resume** is enabled. The **Deformations** buttons
follow what the scene holds, and the **Export** buttons are also grayed out
while a solve runs.
```

### Status Bar Reports

Each of these actions ends with a line in Blender's status bar:

| Action | Report |
| ------ | ------ |
| **Fetch All Animation** | *Animation data fetch finished.* |
| **Delete Remote Data** | *Remote data deleted successfully.* |
| **Save and Quit** | *Save and Quit executed.* |
| **Remove Stale Cache** | *Removed stale cache from Cube.*, naming each object, or *No stale caches to remove.* |
| **Re-capture All Deformations** | *Re-captured deformation for 2 object(s), 1 pin(s)*, naming only the parts that were captured. *Re-capture aborted* after **Abort**, *No deformations to capture* or *Nothing to capture* when there is nothing to record, and *Re-capture failed:* with the reason when it fails. |
| **Clear All Deformations** | *Cleared deformation cache for 2 object(s), 1 pin(s), 1 orphaned cache(s)*, naming only what was cleared. |
| **Migrate data/old/ → data/new/** | *Migrated N modifier path(s); data/old/ copied to data/new/*, or *moved* when **Keep Copy** is off. *Target already exists and is non-empty, resolve manually:* with the path when `data/new/` already holds files, and *Migration failed:* with the reason when the folder cannot be copied or moved. |

## Visual Feedback During Each Stage

The status line on **Backend Communicator** shows the solver's current
state. It reads *Status:* followed by the state, for example *Status:
Ready to Run*; this page names the state alone. A progress-bar label has no
*Status:* prefix. This is what you see during each major operation.

### Status Line States

The status line reads one of these states. Three kinds of text replace the
state name while they apply:

- A button's own text while it is working, for example *Uploading scene...*
  during **Transfer**, *Updating parameters...* during **Update Params on
  Remote**, *Deleting Remote Data...* during **Delete Remote Data**, and
  *Uploading Test Data...* and *Downloading Test Data...* during the
  **Data Send** and **Data Receive** tests under **Debug Options**.
- While a build runs, the progress text of the solver host, which changes as
  the build advances, for example *Preparing build...*, *Loading scene
  data...* and *Building scene: assembling tetrahedra...*. The progress bar
  carries the same text.
- While the solver runs or writes a saved state, *Simulation Running, frame
  N/M*, and the progress bar fills as N approaches M. N and M are the
  solver's own frame indices, which start at 0, so they read one lower than
  the `frame` row of **Realtime Statistics** and **Total Frames**.

The state names in the table apply when no such text is present.

| State | What it means | What to do |
| ----- | ------------- | ---------- |
| *Disconnected* | No connection is open. | Press **Connect**. |
| *Connecting...* | **Connect** is opening the connection. | Wait, or press **Cancel**. |
| *Waiting for Server Start...* | Connected, but the server is not running. | Press **Start Server on Remote**. |
| *Server Launching...* | **Start Server on Remote** is waiting up to 16 seconds for the server to answer. | Wait, or press **Cancel**. |
| *Stopping Server...* | **Stop Server on Remote** or **Force Terminate Process** is ending the server. | Wait. |
| *Protocol Version Mismatch* | The add-on and the server speak different protocol versions. | See [Troubleshooting](../../troubleshooting.md#status-protocol-version-mismatch). |
| *Waiting for Data* | The server is running and holds no scene. | Press **Transfer**. |
| *Waiting for Build* | The solver host holds an uploaded scene with no successful build: after **Transfer without Build** in the **JupyterLab** box, or after a build the solver host refused (see [Transfer](#transfer)). **Run** and **Resume** are grayed out. | Press **Build** in the **JupyterLab** box, or press **Transfer** again. |
| *Building Scene...* | The solver host is building the scene. The line shows the build's progress text. | Wait, or press **Cancel**. |
| *Ready to Run* | The scene is built. | Press **Run**. |
| *Resumable* | The scene is built and the solver host holds saved states. | Press **Resume**, or **Run** to start over. |
| *Initializing...* | **Run** or **Resume** was pressed and the solver is starting. | Wait. |
| *Simulation Running...* | The solver is stepping. The line reads *Simulation Running, frame N/M*. | See [Run](#run). |
| *Saving In Progress...* | **Save and Quit** was pressed and the solver is writing its state before it stops. | Wait. The line then reads *Resumable*. |
| *Simulation Failed* | The last run failed, or the last build did. The line reads *Simulation Failed (Resumable)* while the solver host holds a saved state. | See [Recovery Scenarios](#recovery-scenarios), or [Transfer](#transfer) for a failed build. |
| *Fetching Animation...*, *Applying Downloaded Animation...* | **Fetch All Animation** is downloading frames and then applying them. | See [Fetch](#fetch). |
| *Aborting...* | **Abort** or **Cancel** was pressed and the add-on is stopping the transfer, build or fetch, or **Terminate** was pressed and the solve is stopping. | Wait. |
| *Executing Command...* | **Execute Shell Command on Remote** under **Debug Options** is running. **Exec Command via Server** in the same block shows its own text, *Communicating with Server...*, until the reply arrives. | Wait. |

### Transfer

A progress bar on **Backend Communicator** changes its label through
these stages:

1. *Encoding scene geometry...*: the add-on reads the vertex positions,
   edges and faces of every active group.
2. *Encoding parameters...*: material parameters, scene parameters, pins,
   operations, merge pairs and invisible colliders are gathered for
   upload.
3. *Uploading scene...*: geometry and parameters go up together, so the
   solver host never holds a mismatched pair. The status line shows the
   transfer rate in parentheses.
4. *Building Scene...*: the solver host builds the scene. While it builds,
   the status line and the progress bar show the build's progress text, such
   as *Preparing build...* and *Building scene: assembling tetrahedra...*.
   This can take a few seconds for complex scenes.
5. *Ready to Run*: the transfer is complete and **Run** is enabled, unless
   the scene still holds fetched animation from an earlier run. In that
   case **Run** stays grayed out until you press **Clear Local Animation**;
   see **Run** in [The Buttons](#the-buttons).

```{figure} ../../images/simulating/solver_ready.png
:alt: Solver panel after a successful Transfer, with Transfer, Update Params on Remote, Run and Delete Remote Data enabled
:width: 500px

After **Transfer** completes, the status line reads *Ready to Run* and
**Run** is enabled. **Delete Remote Data** is enabled because the solver
host holds a project after the transfer. This is the state you press
**Run** from.
```

If a stage fails, an error line with an error icon appears under the
status line and the panel stays in the pre-transfer state, so you can fix
the problem and press **Transfer** again. For example:

> Objects missing UUID: Cloth. Run UUID Migration first.

For the repair buttons that some errors offer, see
[Transfer, run, fetch](../../troubleshooting.md#transfer-run-fetch).

**When the build fails on the solver host.** The *Building Scene...* stage
can refuse the scene,
for example because two objects overlap in the first frame. The status line
then reads *Simulation Failed*, Blender's status bar reports *Build failed:*
followed by the first line of the error, and the same first line appears
under the status line as *Remote: ...*. The rest of the report, including
the traceback, is in the add-on console (**Show Console**), and `server.log`
on the solver host has the server's side. Reading the status again with
**Update Stat**, or connecting again, can turn the line into *Waiting for
Build*, because the solver host holds the uploaded scene without a
successful build. Fix the scene and press **Transfer** again, or press
**Update Params on Remote** after a parameter edit.

The status bar reports every build the same way, for **Transfer**,
**Update Params on Remote** and **Build**: *Build completed successfully.*
when the scene is ready, *Build failed: ...* as above, and *Build ended
without a result: ...* followed by the status line's text when the build
stopped for another reason, such as a lost connection or a stopped server.
In that case there is no built scene.

### Run

1. Before the solver starts, a progress bar reads *Checking scene
   geometry...*, *Checking parameters...* and *Starting solver...*. The
   add-on compares the scene with the last upload. If they differ, the run
   stops with *Geometry has changed since the last transfer. Click
   "Transfer" to re-upload before running.* or *Parameters have changed
   since the last transfer. Click "Update Params" before running.*
   *Update Params* in that message is the **Update Params on Remote**
   button.
2. While the solver steps, the status line and a progress bar below it read
   *Simulation Running, frame N/M*, and the bar fills as N approaches M.
   **Save and Quit** and **Terminate** appear under the bar.
3. The **Realtime Statistics** box shows the newest values and updates as
   the solve proceeds; [Realtime and Average Statistics](#realtime-and-average-statistics)
   explains each row. The **Scene Info** box counts *Simulated Frames*
   against *Total Frames*.
4. Each new frame is shown in the viewport when it arrives: the add-on
   downloads the newest frame on every status update, and once more when
   the run ends. Only these frames are in the scene; the rest stay on the
   solver host.
5. When the solve ends, the status line returns to *Ready to Run*, or to
   *Resumable* when a saved state exists. **Realtime Statistics** becomes
   **Average Statistics**, and the **Solver** panel shows *N frames
   unfetched. Press "Fetch All Animation".* **Fetch All Animation**, the
   bake buttons and **Clear Local Animation** are enabled. Because frames
   already streamed into the scene, **Run** stays grayed out until you press
   **Clear Local Animation**.

```{figure} ../../images/simulating/solver_running.png
:alt: Solver panel with every button in the main block grayed out while the simulation runs
:width: 500px

While the solve runs, every button in the main block of the **Solver**
panel is grayed out. The **JupyterLab** and **MCP Server** boxes stay
available. The progress and the statistics are on **Backend
Communicator**.
```

**Export USD** and **Export Alembic (ABC)** are grayed out during a solve
too, while the two **Deformations** buttons follow what the scene holds.

### Fetch

1. Press **Fetch All Animation** once the solve has ended. The status line
   reads *Fetching Animation...* with the download rate in parentheses
   (for example `12.30 MB/s`), and a progress bar fills as frames arrive.
   Frames already in the scene are skipped.
2. The status line then reads *Applying Downloaded Animation...*, and a
   second progress bar fills while the meshes are updated.
3. When it finishes, the status line returns to *Ready to Run* (or
   *Resumable*), the scene's frame range is set to the fetched range, and
   the timeline starts playing.

```{figure} ../../images/simulating/solver_fetched.png
:alt: Solver panel after a fetch, with Transfer, Update Params on Remote, Fetch All Animation, Delete Remote Data, the bake buttons and Clear Local Animation enabled, Run grayed out, and the notice Clear local animation before running
:width: 500px

After the fetch, **Run** is grayed out because the scene holds fetched
animation, and the notice *Clear local animation before running* names
that reason. **Resume** is grayed out because the status line does not
read *Resumable*. **Fetch All Animation** stays enabled to re-fetch, and
the bake buttons and **Clear Local Animation** are enabled because the
scene holds animation.
```

### Aborting a Transfer or Fetch

**Cancel** and **Abort** interrupt only the transfer or fetch in progress.
They do not stop a running solve; use **Terminate** for that. An aborted
fetch discards the frames that were downloaded but not yet applied. Press
**Fetch All Animation** again to continue from the first missing frame.

## Realtime and Average Statistics

**Realtime Statistics** (while a solve runs) and **Average Statistics**
(afterward) are collapsible boxes on **Backend Communicator**. Both list
the same rows in the order of the table. Times read as `ms`, `s` or `m`.
Counts of 1000 or more read with a `k`, `M` or `B` suffix. Ratios and
`stretch` read as percentages.

| Row | Meaning | Realtime | Average |
| --- | ------- | -------- | ------- |
| `frame` | The newest simulated frame, counted from 1. | Shown. | Not shown. |
| `time-per-frame` | Wall-clock time to compute one frame. | Latest frame. | Average over all frames. |
| `time-per-step` | Wall-clock time of one solver step. | Latest step. | Average over all steps. |
| `matrix-assembly` | Wall-clock time to assemble the system matrix, per Newton iteration. | Average over the latest step. | Average over all iterations. |
| `num-contact` | Number of contacts. | Average over the latest step's Newton iterations. | Peak, shown as `num-contact (max)`. |
| `dyn-consumed` | Share of the memory reserved for contact data that is in use. | Latest value. | Peak, shown as `dyn-consumed (max)`. |
| `newton-steps` | Newton iterations in a step, the rounds of the solver's nonlinear solve. | Latest step. | Average per step, with two decimals. |
| `pcg-iter` | Iterations of the linear solver (PCG, the preconditioned conjugate gradient method), per Newton iteration. | Average over the latest step. | Average over all iterations. |
| `pcg-linsolve` | Wall-clock time of the linear solve, per Newton iteration. | Average over the latest step. | Average over all iterations. |
| `line-search` | Wall-clock time of the line search, per Newton iteration. The line search finds how far the vertices can move along the search direction without a collision or a strain-limit violation. | Average over the latest step. | Average over all iterations. |
| `toi` | Time of impact found by the line search: the fraction of the search direction it allowed, from 0 to 1, shown as a percentage. 100% means the full step was accepted. | Average over the latest step. | Average over all iterations. |
| `toi-advanced` | The fraction of the target step size actually advanced, shown as a percentage. | Latest step. | Average over all steps. |
| `stretch` | Largest stretch of shells and rods: 100 × (largest stretch ratio − 1), as a percentage. Drawn only when the solver reports a positive ratio. | Latest step. | Computed from the average of the per-step values. |
| `GPU Util`, `VRAM Usage`, `CPU Usage`, `RAM Usage` | Usage of the solver host. | Shown when the solver host reports them. | Not shown. |

A **Realtime Statistics** row reads `N/A` until the solver has logged its
value. **Average Statistics** leaves out rows with no logged data, and
stays available as long as the log files of the run are on the solver
host.

**Scene Info** lists these rows, in this order:

| Row | What it shows |
| --- | ------------- |
| **Vertices**, **Triangles**, **Tetrahedra**, **Rod Edges** | Element counts of the scene. **Rod Edges** appears only when the scene has rods. |
| **Dynamic Objects**, **Static Objects** | Objects in the dynamic groups (every group type except **Static**), and objects in **Static** groups. |
| **Shared Shells**, **Shared Solids**, and so on | One row per group type, drawn when several objects have an identical mesh, which the add-on sends once. Each number in parentheses, such as `(3,2)`, counts the objects that use one shared mesh. |
| **Total Frames** | The number of frames in the run, which is the **Frame Count** setting. |
| **FPS** | The rate the solver runs at, which includes **Time Scale**. |
| **Simulated Frames** | The number of frames on the solver host. |
| **Last Saved** | The frame of the latest saved state, or `None`. |

**Last Saved**, the `frame` row of the statistics and the rows of the
**Resume** picker number frames from 1. The first simulated frame is Frame 1
whatever the **Starting Frame** is, and Frame N plays on the timeline at the
**Starting Frame** plus N minus 1: with a **Starting Frame** of 101, Frame 50
plays at Blender frame 150.

## Update Params vs Transfer

When you iterate on a scene, press the button that sends what changed:

- **Transfer** re-sends geometry and parameters, rebuilds, and starts the
  remote project over: the simulated frames and saved states on the solver
  host are removed.
- **Update Params on Remote** re-sends parameters only, then rebuilds. The
  geometry stays on the solver host, and so do the simulated frames and
  saved states, so **Resume** remains available.

| Edit | What to press |
| ---- | ------------- |
| Mesh topology change (add or remove vertices, edges, faces) | **Transfer** |
| Add an object to a group or remove it | **Transfer** |
| Change a group's type (for example Shell → Solid) | **Transfer** |
| Move, rotate or scale an assigned object (it becomes the new rest pose) | **Transfer** |
| Add a pin vertex group, or change which vertices a pin holds | **Transfer** |
| Edit a **Static** collider's operations, or re-capture its deformation | **Transfer** |
| Edit an existing pin's operations (Move By / Spin / Scale / Torque), **Duration** or **Pull** | **Update Params on Remote** |
| Change material parameters (density, stiffness, friction, ...) | **Update Params on Remote** |
| Change scene parameters (gravity, wind, air, step size, ...) | **Update Params on Remote** |
| Edit dynamic-parameter keyframes | **Update Params on Remote** |
| Add, remove or keyframe an invisible collider | **Update Params on Remote** |
| Add, remove or edit a merge pair (a snap also moves an object, which needs **Transfer**) | **Update Params on Remote** |
| Load a scene profile or material profile | **Update Params on Remote** |
| Toggle an overlay (a pin's eye, preview arrows) | Nothing; viewport only |

If you press the wrong button, **Run** and **Resume** refuse and point to
the right one. In their messages, *Update Params* is the **Update Params on
Remote** button.

If the linear solve takes many iterations (`pcg-iter` in the statistics),
the **Preconditioner** scene parameter (**Block Jacobi** by default, or
**Schwarz**) selects another preconditioner; see
[Preconditioner](../params/scene.md#preconditioner).

## Mesh Topology Changed Warning

When you press **Run**, **Resume** or **Fetch All Animation** after editing
the scene since the last **Transfer**, Blender's status bar can report:

```text
Mesh topology changed since last transfer (groups differing: SHELL_<group ID>).
Re-transfer to sync.
```

The names in parentheses are group identifiers (the group's type followed
by an ID), not the names shown in **Dynamics Groups**. The same line goes to
the Console. The report does not block the action. It appears when a
group's vertex, polygon, edge or object count changed, when a pin vertex
group was added or changed in size, or when a **Static** collider's
captured deformation changed. Press **Transfer** to upload the current
scene before running or fetching.

**Run** and **Resume** also compare the scene with the last upload and
refuse when it differs:

| Change | **Run** says | **Resume** says |
| ------ | ------------ | --------------- |
| Geometry | *Geometry has changed since the last transfer. Click "Transfer" to re-upload before running.* | *Geometry has changed; resume is not possible. Click "Transfer" and "Run" for a fresh simulation.* |
| Parameters | *Parameters have changed since the last transfer. Click "Update Params" before running.* | *Parameters have changed since the last transfer. Click "Update Params" before resuming.* |

A material-parameter edit leaves the topology report silent and still
stops **Run** until you press **Update Params on Remote**.

## How Animation Plays Back

After **Fetch All Animation**, each simulated mesh plays back the solver's
output as you scrub the timeline. The per-frame vertex data is stored in a
`data/<blend_basename>/` folder next to your `.blend`, and the add-on
attaches it to the mesh through a `ContactSolverCache` modifier in
**Modifier Properties**. Rods have no modifier: the add-on updates their
control points on every frame change.

```{figure} ../../images/simulating/pc2_modifier.png
:alt: Blender Properties editor on the Modifier tab showing the ContactSolverCache modifier with Format PC2 and a File Path inside the data folder next to the .blend
:width: 500px

The **ContactSolverCache** entry in **Modifier Properties** after a
fetch. Its **File Path** points at the object's cache file under
`<blend_dir>/data/<blend_basename>/`, and that file plays the simulation
back while you scrub. Leave the modifier in place: deleting it or its file
removes the animation. To turn the fetched animation into regular
Blender animation (shape keys and animation curves), use **Bake
Animation**; see [Baking Animation](baking.md).
```

:::{tip}
Save the `.blend` after fetching. Until the file is first saved, the cache
files are in the system temporary folder. Saving moves them into the
`data` folder next to the `.blend`, so the animation survives closing and
reopening the file.
:::

## Reading Per-Frame Statistics

**Fetch All Animation** also downloads a small record of numbers for every
simulated frame and object. The **Object Statistics** panel, last in the
sidebar and collapsed by default, shows the record for the frame under the
playhead, so the numbers change as you scrub the timeline.

1. Open **Object Statistics**.
2. Choose an object in the **Object** menu at the top. It lists every
   included object of every active group.
3. Scrub the timeline. The rows follow the playhead.

The panel starts with **Frame** and **Time (s)**, the solver time in
seconds at that frame. The other rows depend on the object:

| Row | Shown for |
| --- | --------- |
| **Location (m)**, **Velocity (m/s)**, **Speed (m/s)**, **Acceleration (m/s²)**, **Acceleration Magnitude (m/s²)**, **Angular Velocity (rad/s)**, **Angular Speed (rad/s)**, **Angular Axis** | Every object, **Static** colliders included. Vector rows read `[x, y, z]`. |
| **Surface Area (m²)** | An object with faces, and a **Sand** group. |
| **Volume (m³)** | An object built from tets or closed by its own surface, and a **Sand** group. |
| **Rod Length (m)** | A rod. |
| **Contact Count** | Every object when the solver records it, otherwise `N/A`. Counts past a thousand read as `12K` or `1.5M`. |

**Surface Area**, **Volume** and **Rod Length** carry a percentage in
parentheses: the value relative to the same measure on the run's first
frame. `102%` beside a surface area means the sheet has stretched two
percent since the run started.

A frame that was not fetched has no record: every row reads `N/A` and
**Export CSV** is grayed out. **Clear Local Animation** deletes the
statistics along with the fetched animation. The panel then reads
*Statistics unavailable; rerun the simulation* for the object, without an
**Export CSV** row. The data is gone, not corrupt, and a fresh **Run** and
**Fetch All Animation** brings it back.

**Export CSV** opens a menu of the values the selected object supports.
Pick one, and it writes that value for every fetched frame to a file named
`<object>_<value>.csv` by default (for example `Cloth_speed.csv`), with
three columns:

| Column | What is in it |
| ------ | ------------- |
| `frame` | Timeline frame number: the solver's frame index with the **Starting Frame** added back, so it lines up with the timeline. The panel's numbering from 1 matches it only when the **Starting Frame** is 1. |
| `time_s` | Solver time in seconds at that frame. |
| `value` | The value. A vector is written as one `[x,y,z]` cell. A frame the solver did not measure the value on is written empty. |

Only fetched frames are written, so a partial fetch produces a shorter
file rather than one padded with blanks.

**Export CSV** reports *Exported statistics to* followed by the file path
when it succeeds, and *CSV export failed:* followed by the reason when the
file cannot be written or the statistics cannot be read. The panel reads
*No dynamics objects* when no object is included in an active group, and
shows the reason with an error icon when the statistics file cannot be read.

### From MCP

An MCP client reads the same statistics after **Fetch All Animation**, with
three tools:

- `list_statistics_objects` lists the objects that have statistics, the
  channels measured for each, a catalog of every channel with its label and
  unit, and the `start_frame` that every frame number in these tools counts
  against.
- `get_object_statistics(object_name, frame)` returns every channel the
  object supports at one Blender frame. A frame the run never wrote is
  refused.
- `get_object_statistics_series(object_name, channel, frame_start,
  frame_end)` returns one channel as samples of `frame`, `time_s` and
  `value`. Both bounds are Blender frames, inclusive, and optional.
  A channel the object does not measure is refused, and the refusal names
  the channels it does.

`object_name` is the object's name, or its UUID from the list when the
object has left the scene. Without fetched statistics, each tool refuses
with *Statistics unavailable; rerun the simulation*.

A vector is read one component at a time (`LOCATION_X`, `LOCATION_Y`,
`LOCATION_Z`), and the stretch ratios `AREA_STRETCH`, `LENGTH_STRETCH` and
`VOLUME_STRETCH` are channels of their own. A stretch channel is the ratio
to the first frame, so `1.02` is the `102%` the panel shows. The full set of
channel ids is under *Statistics channel ids* in the *Under the hood* block
at the end of this page.

## Disconnecting While a Simulation Runs

After **Run**, the solve runs as a process on the solver host, and Blender
only displays its status. You do not have to keep Blender open for the run
to continue:

- **Disconnect** on **Backend Communicator** drops the live connection.
  The solve keeps going, whatever the connection type.
- You can quit Blender.
- Later (minutes, hours, or after a reboot of your workstation), open the
  same `.blend` and press **Connect**. **Fetch All Animation** stays
  grayed out while the solve is still running. When the status line reads
  *Ready to Run* or *Resumable*, press it to download the frames.

Reconnecting attaches to the server that is still running. If another
program holds its port, see
[Port Already in Use](../../connections/index.md#port-already-in-use).

Disconnecting is separate from driving the run from JupyterLab. A run
started with **Run** in Blender continues while Blender is closed.
[JupyterLab](jupyterlab.md) is another way to reach the same project on
the solver host.

For the line the Console logs when you reconnect after saving the `.blend`,
see [Sessions and Recovery](#sessions-and-recovery). For keeping the solver's
own state across a crash between sessions, see
[Auto-Save and Graceful Shutdown](#auto-save-and-graceful-shutdown).

## Sessions and Recovery

Every successful **Connect** starts a new session with its own ID, and
saving the `.blend` while connected stores the ID of that session. On the
next **Connect**, the add-on compares the stored ID with the new one. They
differ after a reconnect, and then the Console (**Show Console**) logs a
line starting with `Reconcile: previous session`. The line is
informational: it does not by itself mean that the animation fetched into
the `.blend` belongs to a different run, nothing is blocked, and no panel
warning appears. If the project on the solver host is not the one you
expect (for example, **Scene Info** shows other frame counts), connect to
the solver host where the original run lives, or press **Transfer** to
replace the project with your current scene.

For what to do after a stop or a crash, see
[Recovery Scenarios](#recovery-scenarios).

## Auto-Save and Graceful Shutdown

A run can be resumed only from a saved state, which the **Resume** picker
calls a saved checkpoint. These write one:

- **Auto Save**, **Save Checkpoints** and **Save State on Finish**, in the
  **Save and Checkpoints** box of **Scene Configuration**. Defaults and
  intervals are in [Save and Checkpoints](../params/scene.md#save-and-checkpoints).
  A **Keep Saved States** value above 0 keeps only that many auto-saved
  states, so **Resume** offers fewer, later frames.
- **Save and Quit** on **Backend Communicator**, which writes a state and
  stops the running solve.

**Terminate** writes no state.

## Resuming a Run

**Resume** continues a run the solver host already holds, without
re-uploading geometry or rebuilding.

1. Press **Resume**. It is enabled when the status line reads
   *Resumable* or *Simulation Failed* and the solver host holds at least
   one saved state.
2. The add-on first compares the scene with the last upload, before any
   picker opens. A geometry or parameter change is refused at this point,
   with the messages in
   [Mesh Topology Changed Warning](#mesh-topology-changed-warning).
3. The picker, headed *Resume from saved checkpoint:*, lists one *Frame N*
   row per saved state, numbered from 1 the way *Last Saved* in **Scene
   Info** is (see [Realtime and Average Statistics](#realtime-and-average-statistics)).
   The latest state is preselected. The list reads *No saved
   checkpoints available.* when there is none, and confirming with no row
   selected reports *Select a checkpoint frame to resume from.*
4. Confirm. The solve continues from that frame. Frames before it are
   kept, and frames after it are overwritten.

Because **Resume** reuses the state already on the solver host, the
comparison in step 2 refuses when the scene differs from the last upload:

- If the **geometry** changed, **Resume** asks you to press **Transfer**
  and then **Run** for a fresh simulation. The saved state does not
  match the mesh.
- If only the **parameters** changed, **Resume** asks you to press
  **Update Params on Remote** first. That re-sends the parameters and
  keeps the saved states, so you can **Resume** right afterward.

## Recovery Scenarios

What to do depends on what stopped:

| What happened | What the solver host has | What to do next |
| ------------- | ------------------------ | --------------- |
| You closed Blender or lost the network mid-run. The solve kept running. | Everything simulated so far. | Reopen the `.blend` and press **Connect**. While the solve is still running, leave it. When the status line reads *Ready to Run* or *Resumable*, press **Fetch All Animation**. |
| You pressed **Terminate**. | The frames simulated so far, and any saved states. | The status line reads *Resumable* when a saved state exists, otherwise *Ready to Run*. Press **Resume** and pick a saved state, or press **Clear Local Animation** (when it is enabled) and then **Run** to start over. |
| The run failed (a frame did not converge, or the solver stopped on an error). | The frames up to the failure, and any saved states. | The status line reads *Simulation Failed*, or *Simulation Failed (Resumable)* while a saved state exists. Press **Resume** and pick an earlier state to continue past the trouble spot, or fix the scene and press **Run** again. See [Status: "Simulation Failed"](../../troubleshooting.md#status-simulation-failed) for the crash report, **Show Console** and **Open Session Folder**. |
| The solver process crashed (segfault, out of memory) or the solver host rebooted. | Whatever was saved (when **Auto Save** was on), plus the frames already written. | Reconnect, and press **Start Server on Remote** if the status line asks for it. With a saved state, **Resume** lists it. Without one, press **Run** to simulate again. |

## Blender Python API

The same workflow is available from Python. The import below is for an
add-on in the **User Default** repository; for an add-on installed from
another repository, see
[Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# The core loop.
solver.transfer_data()
solver.run_simulation()
solver.fetch_animation()

# Iterate on parameters without resending geometry.
solver.param.step_size = 0.002
solver.update_params()

# Continue from the latest saved state, or from a chosen frame.
solver.resume_simulation()
solver.resume_simulation_from(frame=40)

# Stop a running solve.
solver.save_and_quit_simulation()   # write a state, then stop
solver.terminate_simulation()       # stop at once, no state written

# Clean up.
solver.clear_local_animation()
solver.delete_remote_data()
```

MCP clients also have `list_checkpoint_frames`, which returns the saved
states that `resume_simulation_from` accepts; see
[MCP Server](../../integrations/mcp.md). For the full call list, see
the [Blender Python API Reference](../../integrations/python_api_reference.rst).

:::{admonition} Under the hood
:class: toggle

**Operator names**

| Button | `bl_idname` |
| ------ | ----------- |
| Transfer | `solver.transfer` |
| Run | `solver.run` |
| Resume | `solver.resume_from` |
| Update Params on Remote | `solver.update_params` |
| Fetch All Animation | `solver.fetch_remote_data` |
| Clear Local Animation | `solver.clear_animation` |
| Delete Remote Data | `solver.delete_remote_data` |
| Remove Stale Cache | `solver.remove_stale_caches` |
| Re-capture All Deformations | `solver.recapture_all_deformations` |
| Clear All Deformations | `solver.clear_all_deformations` |
| Export USD | `solver.export_usd` |
| Export Alembic (ABC) | `solver.export_alembic` |
| Terminate | `solver.terminate` |
| Save and Quit | `solver.save_quit` |
| Abort | `ssh.abort` |

Any `solver.*` method not defined on the solver proxy is forwarded to
`bpy.ops.zozo_contact_solver.<name>`, an operator generated from the MCP
handler of the same name. So the names in the Python block above are
handler names (`transfer_data`, `terminate_simulation`, and so on), not
the `bl_idname`s. `resume_simulation` runs `solver.resume`, which
continues from the latest saved state, while the **Resume** button runs
`solver.resume_from`, which asks for a frame.

**Statistics channel ids**

The channel ids that the MCP statistics tools accept are `LOCATION_X`,
`LOCATION_Y`, `LOCATION_Z`, `VOLUME`, `SURFACE_AREA`, `AREA_STRETCH`,
`ROD_LENGTH`, `LENGTH_STRETCH`, `VELOCITY_X`, `VELOCITY_Y`, `VELOCITY_Z`,
`SPEED`, `ACCELERATION_X`, `ACCELERATION_Y`, `ACCELERATION_Z`,
`ACCELERATION_MAGNITUDE`, `ANGULAR_VELOCITY_X`, `ANGULAR_VELOCITY_Y`,
`ANGULAR_VELOCITY_Z`, `ANGULAR_SPEED`, `ANGULAR_AXIS_X`, `ANGULAR_AXIS_Y`,
`ANGULAR_AXIS_Z`, `VOLUME_STRETCH` and `CONTACT_COUNT`.

**Mesh hash**

The topology hash is computed per active group over vertex count, polygon
count, edge count, object count, the pin vertex groups and their sizes,
and a digest of each captured static-deformation cache. Pure transforms
and material-parameter edits do not affect it. Adding a pin vertex group
or re-capturing a **Static** collider does.

**Files on disk**

Per-object cache files are in `<blend_dir>/data/<blend_basename>/`, one
`<object-uuid>.pc2` per simulated object. Fetched statistics are in the
same folder, as one `statistics_manifest.cbor` plus one `.stats` file per
object, and saving the `.blend` moves them there with the `.pc2` files.
The solver's own per-frame `vert_N.bin` files and its object-to-vertex
maps stay on the solver host under the project's `session/` directory. A
fetch reads them over the wire and writes the local `.pc2` files from
them.

**`ContactSolverCache` entry**

Mesh playback is done by a `ContactSolverCache` entry in the **Modifier
Properties** tab, pointing at the object's `.pc2` file. It sits in the
**first modifier slot** with `frame_start` set to the solve's **Starting
Frame** (1 by default), so it deforms the rest mesh before any other
deformer runs. The exception is an object whose own deformers fed the
solver: a **Static** collider, or a dynamic object carrying a deforming
modifier stack such as an Armature or Lattice. There the entry is placed
after those position-preserving deformers and before the first
topology-changing one, so the solver's output wins over the input that
produced it. Curves are updated directly on every frame change and need
no entry. A Hair Curves object gets a **ContactSolverCache** entry of its
own kind, a Geometry Nodes modifier placed before the first modifier that
generates curves; see [Playback](../scene/hair_curves.md#playback).

**Session ID**

The session ID is a 12-character hexadecimal string. Saving the `.blend`
stores the current session ID on the scene, and each simulated object is
stamped with the ID of the run that bound its cache. The `Reconcile` line
compares the saved ID with the one created on **Connect**.
:::
