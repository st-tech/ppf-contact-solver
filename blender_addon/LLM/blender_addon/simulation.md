# Simulation workflow

Running the solve and getting the result back onto your Blender meshes, whether interactively from the sidebar, from JupyterLab, or baked down to plain keyframes.

## Workflow overview

Once you have a running solver and a live connection (see Connections if you haven't set that up yet), the day-to-day loop is:

1. **Organize your scene** into object groups: **Solid**, **Shell**, **Rod**, or **Static**.
2. **Assign material parameters** per group (density, Young's modulus, Poisson ratio, friction, bend, shrink, strain limit, and so on).
3. **Set scene parameters**: gravity, wind, time step, frame count, air density, air friction.
4. **Add pins** and attach operations: **Move By**, **Spin**, **Scale**, **Torque**, or keyframed **Embedded Move**.
5. *(Optional)* **Add invisible colliders**: infinite walls or parametric spheres (including bowl / inverted sphere variants).
6. *(Optional)* **Snap and merge** overlapping meshes to stitch them across group boundaries.
7. **Transfer** geometry and parameters to the solver, then **Build**.
8. **Run** the simulation and **Fetch** frames back as PC2 animation on your Blender objects.
9. *(Optional)* **Bake** the fetched animation onto the objects as standard Blender keyframes, dropping the add-on's cache modifier.

Nine numbered step boxes across two rows. Row one covers scene-setup steps 1 through 5: object groups (Solid/Shell/Rod/Static), material parameters (density, Young's, Poisson, bend, shrink, strain limit), scene parameters (gravity, wind, time step, frame count, air density), pins and operations (Move By/Spin/Scale/Torque/Embedded Move), and the optional invisible colliders (walls, spheres, bowls). Row two covers steps 6 through 9: the optional Snap and Merge that stitches overlapping meshes across group boundaries, Transfer and Build which encodes the scene for the solver, Run and Fetch which solves on the GPU and downloads per-frame PC2 vertex data onto each Blender mesh, and the optional Bake step that converts the PC2 cache into standard Blender shape keys and fcurves and removes the ContactSolverCache modifier. Scene-setup boxes are blue, solver boxes are orange, and the bake box is green. Optional steps have dashed borders. Steps 1 through 6 are scene setup in Blender, 7 and 8 are the solve on the GPU, and 9 hands the result off as self-contained Blender animation. In practice you bounce between Run and the material / scene parameters many times before baking.

Step 8 can also happen entirely from JupyterLab, including with Blender closed. See JupyterLab for the full export -> simulate -> relaunch Blender -> fetch loop.

### Coordinate system

Blender is Z-up; the solver is Y-up. The add-on converts in both directions automatically. A Blender vector `(x, y, z)` is sent to the solver as `[x, z, -y]`, and results are flipped back on fetch. You only need to think about this when you read raw solver output. Everything visible in Blender (pin centers, gravity arrows, overlay previews, baked animation) is already in Z-up.

Side-by-side diagram of Blender's Z-up right-handed coordinate frame and the solver's Y-up right-handed frame, linked by arrows showing the encode conversion (x, y, z) -> [x, z, -y] and the decode conversion [X, Y, Z] -> (X, -Z, Y). Both frames are right-handed. On the encode side, Blender's `Y` (into the scene) becomes the solver's `-Z`, and Blender's `Z` (up) becomes the solver's `Y`. On fetch the conversion runs in reverse, so everything read back on Blender meshes, pins, and overlays stays in Z-up.

NOTE: The solver stores vertices in untranslated object space (rotation + scale only) with object translation tracked separately. Any absolute coordinate you supply through the Python API (pin centers, operation pivots, collider positions) is transformed into that space at encode time. You do not need to pre-subtract the origin.

### Where parameters live

Each sidebar panel maps to one chapter below:

| Panel                                  | What's there                                                                                 | Docs                                           |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Scene Configuration**                | Global sim params: gravity, time step, frame count, CG tolerances, air density / friction, auto-save. | Scene Parameters                               |
| **Scene Configuration → Dynamic Parameters** | Keyframed gravity / wind / air density / air friction / vertex air damping.          | Dynamic Parameters                             |
| **Dynamics Groups**                    | Per-group type, material model, densities, moduli, contact gap, overlay color.               | Object Groups, Material Parameters             |
| **Dynamics Groups → Pin Vertex Groups** | Pins and their list of operations (**Move By** / **Spin** / **Scale** / **Torque** / **Embedded Move**). | Pins and Operations                      |
| **Dynamics Groups → Transform** *(Static only)* | Per-object **Move By** / **Spin** / **Scale** ops, or Blender transform keyframes.  | Static Objects                                 |
| **Scene Configuration → Invisible Colliders** | Walls and spheres with keyframed position / radius.                                   | Invisible Colliders                            |
| **Snap and Merge**                     | Snap/merge pairs with optional stitch stiffness.                                             | Snap and Merge                                 |

NOTE: Everything reachable from the Scene Configuration panel is also reachable from a fluent Blender Python API. See the Blender Python API reference for the full surface; each chapter below also ends with a short `## Blender Python API` section covering the calls relevant to that chapter.

### Blender Python API

The same workflow is available from Python. Import `solver` and read or write through `solver.param`, `solver.create_group(...)`, `solver.add_wall(...)`, and friends:

```python
from zozo_contact_solver import solver

solver.param.gravity = (0, 0, -9.8)
cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt")
```

UNDER THE HOOD:

All add-on data hangs off `scene.zozo_contact_solver`:

| Lives on                                              | What's there                                                                                 |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `scene.zozo_contact_solver.state`                     | Global sim params: gravity, dt, frame count, CG tolerances, air density/friction, auto-save. |
| `scene.zozo_contact_solver.state.dyn_params`          | Keyframed gravity / wind / air density / air friction / vertex air damp.                     |
| `scene.zozo_contact_solver.object_group_0…31`         | Per-group type, material model, densities, moduli, contact gap, overlay color.              |
| `object_group_N.pin_vertex_groups`                    | Pins and their list of operations (MOVE_BY / SPIN / SCALE / TORQUE / EMBEDDED_MOVE).         |
| `scene.zozo_contact_solver.state.invisible_colliders` | Walls and spheres with keyframed position / radius.                                          |
| `scene.zozo_contact_solver.state.merge_pairs`         | Snap/merge pairs with optional stitch stiffness.                                             |

## Running the simulation

Once the scene is organized into groups, parameters are set, and pins / colliders are in place, the day-to-day loop is: **Transfer -> Run -> Fetch**, with **Resume**, **UpdateParams**, and **ClearAnimation** for iteration.

### The Solver panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab. The **Solver** panel is the second panel in the tab, directly below **Backend Communicator** and above **Scene Configuration**, **Dynamics Groups**, **Snap and Merge**, and **Visualization**. It is always visible (never collapsed by default) because it is the primary control surface during simulation work.

The Solver panel right after the server comes up. Only Transfer is enabled; everything downstream of it is grayed out until the remote has the scene. The info line at the bottom (*Click "Transfer" to upload data*) reinforces which step is next.

The panel is laid out as a single vertical column of buttons, status indicators, and controls:

1. **Connection row.** At the very top of the panel is a row showing the current connection target (host and port) and a **Connect** / **Disconnect** button. When connected, the label changes to show the active session.

2. **Primary action buttons.** Below the connection row, the main buttons are stacked vertically in this order:
   - **Transfer**. Uploads geometry and parameters to the solver.
   - **Run**. Starts the simulation.
   - **Resume**. Continues a paused or partially completed run.
   - **Update Params**. Re-uploads parameters without resending geometry.
   - **Fetch All Animation**. Downloads simulation results.
   - **Bake Animation** / **Bake Single Frame**. Convert the fetched cache into standard Blender keyframes (see Baking Animation).

3. **Status line.** Between the primary action buttons and the secondary controls, a status line displays the current solver state. This line updates in real time during long operations (see "Visual feedback during each stage" below).

4. **Secondary controls.** Below the status line:
   - **Clear Local Animation**. Removes fetched animation data.
   - **Delete Remote Data**. Wipes the solver's project data.
   - **Terminate**. Hard-stops the running simulation.
   - **Save & Quit**. Gracefully shuts down the server.
   - **Abort**. Interrupts the current transfer or fetch operation.

5. **Progress bar.** During **Fetch All Animation**, a horizontal progress bar appears inline within the panel, showing download progress as a percentage alongside bandwidth statistics.

Buttons that are not applicable to the current state are grayed out. For example, **Run** is grayed out until a successful **Transfer** has completed, and **Resume** is grayed out unless a simulation has been paused or partially run.

### The buttons

| Button                    | What it does                                                         |
| ------------------------- | -------------------------------------------------------------------- |
| **Transfer**              | Multi-stage pipeline: delete existing remote data → send mesh → send params → build. |
| **Run**                   | Warns on stale mesh hash, clears existing animation, starts the solve. |
| **Resume**                | Resumes a paused / partially-run simulation.                         |
| **Update Params**         | Re-encodes and uploads parameters, then rebuilds. No geometry resend. |
| **Fetch All Animation**   | Downloads per-frame vertex data and applies it as PC2 animation.     |
| **Clear Local Animation** | Removes simulation keyframes and cache modifiers. Preserves pin keyframes. |
| **Delete Remote Data**    | Asks the server to wipe its current project data.                    |
| **Terminate**             | Hard-stops the current simulation on the server.                     |
| **Save & Quit**           | Graceful shutdown: flushes state to disk, then exits the server.     |
| **Abort**                 | Interrupts the *current* transfer or fetch. Does not touch running sim. |

`Terminate` and `Save & Quit` target the server itself; **Terminate** is the hard equivalent of pulling the plug, while **Save & Quit** lets the solver flush its state first so a later reconnect can pick up from there.

Five state boxes arranged left to right (Connected, Ready, Running, Complete, Fetched) with the status-line text and enabled-button list inside each box. Blue solid arrows mark the canonical forward flow (Transfer → Run → sim ends → Fetch); purple dashed arrows mark loops and recovery transitions (Update Params self-loop on Ready, Terminate and Resume between Running and Complete, Save & Quit from Running back to Connected, Clear Local Animation from Fetched back to Ready). A footer explains that Delete Remote Data is reachable from any post-Transfer state, that Abort only interrupts a Transfer or Fetch, and that the panel is otherwise fully inert while Running. Which buttons light up in each state. The canonical forward flow runs left-to-right along the blue arrows; the purple dashed arrows cover the loops (Update Params), recovery transitions (Clear Local Animation, Resume), and early exits (Terminate, Save & Quit). If a button is grayed out, find the current state on the diagram; the enabled list inside that box is the full answer.

After Transfer completes, the panel settles into the Ready to Run state. Run lights up (the play icon turns solid), and Delete Remote Data is also armed because the remote now holds a project. This is the state you press Run from.

### Visual feedback during each stage

The **status line** on the Solver panel keeps you informed of what the solver is doing at every moment. Here is what you see during each major operation:

#### Transfer

The status line cycles through several sub-stages as the transfer proceeds:

1. **"Uploading Mesh Data"**. The add-on serializes and sends vertex positions, edges, and faces for every active group.
2. **"Uploading Parameters"**. Material parameters, scene parameters, pins, operations, merge pairs, and invisible colliders are encoded and sent.
3. **"Building"**. The solver is constructing its internal data structures (BVH trees, constraint graphs, contact maps). This step can take a few seconds for complex scenes.
4. **"Ready"**. The transfer is complete. The **Run** button is now enabled.

If any sub-stage fails, the status line shows an error message in red text describing the failure (e.g. "Transfer failed: mesh has zero faces"). The panel remains in the pre-transfer state so you can fix the issue and retry.

#### Run

Once the simulation starts:

1. The Solver-panel status line reads **"Simulation Running..."** and a blue progress bar appears on the **Backend Communicator** panel above, labeled with the same status.
2. Live counters appear on **Backend Communicator** in two blocks. **Realtime Statistics** shows the current `frame`, `time-per-frame` and `time-per-step` wall-clock, `num-contact`, `newton-steps`, `pcg-iter`, and `stretch`. **Scene Info** tracks `Simulated Frames` against `Total Frames` so you can see how far through the run you are.
3. When the simulation completes, the status line returns to **"Ready to Run"**, `Simulated Frames` reaches `Total Frames`, and a warning row (*N frames unfetched. Press "Fetch All Animation"*) appears above the Solver panel, directing you at the next step.

Backend Communicator mid-solve: the status line reads *Simulation Running...*, the blue progress bar carries the same label, and the Realtime Statistics block updates in place as each frame lands. The Scene Info block below shows `Simulated Frames` ticking toward `Total Frames`.

After the run finishes, the status line returns to *Ready to Run*, the live counters collapse into an Average Statistics block, and `Simulated Frames` matches `Total Frames`. The warning row above the Solver panel tells you exactly how many frames are still on the remote and which button to press next.

#### Fetch

During a fetch:

1. A **progress bar** fills from left to right as frame data is downloaded. The bar shows a percentage and the number of frames fetched so far (e.g. `120 / 240 frames`).
2. **Bandwidth statistics** appear alongside the progress bar (e.g. `12.3 MB/s`).
3. On completion, the status line reads **"Animation Ready"** and the fetched frames are immediately available for timeline scrubbing.

While the simulation is live, the Solver panel goes fully inert; every button is grayed out so you cannot double-fire an action or tear down state mid-solve. The progress counter and per-frame stats live on the Backend Communicator panel above; this panel simply stays out of the way until the solve ends.

Once Fetch All Animation finishes, the panel returns to its post-sim resting state: Fetch All Animation and Delete Remote Data stay enabled so you can re-fetch or wipe the remote project, and the info line *Clear local animation before running* reminds you that a second Run needs a cleared cache first.

### Update Params vs Transfer

When iterating on a scene, the question comes up: does this change need a full **Transfer**, or is **Update Params** enough? The rule is:

- **Transfer** re-sends geometry and parameters, then rebuilds.
- **Update Params** re-sends parameters only, then rebuilds. Mesh buffers on the server are preserved, which is why it completes much faster on large scenes.

So **Transfer** is required whenever mesh topology or group membership changes; **Update Params** is enough for everything else that lives in the parameter payload (scene settings, material params, pins, colliders, dynamic parameters). The table below enumerates the common edits:

| Edit                                                          | What to press                |
| ------------------------------------------------------------- | ---------------------------- |
| Mesh topology change (add/remove verts, edges, faces)         | **Transfer**                 |
| Add or remove an object from a group                          | **Transfer**                 |
| Change a group's type (e.g. Shell → Solid)                    | **Transfer**                 |
| Pure transform of an assigned object (move/rotate/scale)      | **Transfer** *(new rest)*    |
| Add a new pin vertex group on existing geometry               | **Update Params**            |
| Edit an existing pin's operations (Move By / Spin / Scale / Torque) | **Update Params**      |
| Change material parameters (density, stiffness, friction, …)  | **Update Params**            |
| Change scene parameters (gravity, wind, air, step size, …)    | **Update Params**            |
| Edit dynamic-parameter keyframes                              | **Update Params**            |
| Add, remove, or keyframe an invisible collider                | **Update Params**            |
| Add or remove a snap/merge pair                               | **Update Params**            |
| Load a scene profile or material profile                      | **Update Params**            |
| Toggle an overlay (Show Pins, preview arrows)                 | *(nothing; viewport only)*   |

The **Mesh hash mismatch** warning in the next section is the add-on's safety net: if you skip **Transfer** after a topology change, it shows up before **Run** or **Fetch** and tells you to re-transfer.

### "Groups have changed" warning

If you edit your meshes (add or remove vertices, change group membership, reassign an object's type) after **Transfer** but before **Run** or **Fetch**, the panel shows a warning:

> Mesh hash mismatch: groups have changed since last transfer.

The warning does not block you, but it means the solver's data no longer matches what is in Blender. Click **Transfer** again to re-upload before running or fetching.

Pure transforms and material-parameter edits do not trigger this warning; only topology changes do.

### How animation plays back

After **Fetch**, the add-on attaches a cache modifier to each simulated mesh and writes per-frame vertex data next to your `.blend` file. Scrub the timeline and the mesh deforms to the solver's output. Curves (rods) update directly on every frame change without a modifier.

Concretely, the cache modifier is a standard Blender **Mesh Cache** (`MESH_CACHE`) modifier named **ContactSolverCache**, with **Format** set to **PC2** and **File Path** pointing at a `.pc2` file that the add-on writes to `<blend_dir>/data/<blend_basename>/`. You can inspect it in the **Modifier Properties** tab:

The ContactSolverCache modifier as it appears in Modifier Properties after a successful fetch. The Format is PC2 and File Path points at the per-object cache file under `<blend_dir>/data/<blend_basename>/`. This modifier (not a shape-key stack or baked fcurves) is what plays the simulation back while you scrub. Leave the modifier in place; deleting it (or its `.pc2` file) removes the animation. To convert the cache into regular Blender animation (shape keys + fcurves), run Bake Animation instead; see Baking.

TIP: Save the `.blend` after fetching. The add-on migrates temporary cache files into a permanent location on save, so the animation survives closing and reopening the file.

### Disconnecting while a simulation runs

Once **Run** is pressed on a remote backend (SSH, Docker, or Windows Native), the solver is doing its work on the remote host; Blender is just watching. You do **not** have to keep Blender open for the run to continue:

- Press **Disconnect** on the Backend Communicator to drop the live connection. The remote solver keeps going.
- You can even quit Blender entirely. The remote process is owned by the solver host, not by the add-on.
- Later (minutes, hours, or after a reboot of your workstation), launch Blender, reopen the same `.blend`, press **Connect**, and **Fetch All Animation** pulls in whatever frames have landed on disk so far.

The session ID baked into the `.blend` is what the add-on uses to recognize "this is the same run I started before". See Sessions and recovery below for how the session check works, and Auto-save and graceful shutdown for making sure the solver's own state survives a crash between sessions.

This is not the same as running the sim *from* JupyterLab. In that scenario the sim was launched from the Blender add-on; JupyterLab just happens to be another way to drive a project that lives on the same solver host. See JupyterLab if you also want to poke at the run from a notebook while Blender is closed.

### Sessions and recovery

Every successful connect mints a fresh **session ID** and the add-on stamps simulation artifacts (PC2 files, the cache modifier, the remote project directory) with it. Saving the `.blend` stores the active session on the scene; on reopen, reconnecting compares the new session against the saved one and warns if they differ, meaning the cached frames on disk may no longer correspond to anything the remote knows about. This is how the add-on distinguishes "this is the sim I was running before I closed Blender" from "this is a fresh run on a different server".

### Auto-save and graceful shutdown

Two related features:

- **Auto-save** (the **Auto Save** and **Auto Save Interval** fields on the Scene Configuration → Advanced Params sub-panel): when enabled, the solver periodically dumps its state so a crash or disconnect does not cost all the progress. This runs inside the server process.
- **Save & Quit**: a one-shot operator that asks the server to flush state and exit cleanly. After **Save & Quit**, the next reconnect can pick up the run where it left off.

**Terminate** does not flush state. Use it when the simulation is misbehaving and you want it gone.

#### Recovery scenarios

The behavior depends on *who* failed. Three cases cover the common ones; the table shows the first move in each.

| What happened                                                         | What the solver has on disk | What to do next                                                                                       |
| --------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| You closed Blender or lost the network mid-run. Solver kept running.  | Everything up to "now".     | Reopen the .blend, **Connect**, **Fetch All Animation**. If still running, leave it. If it finished while you were gone, fetch picks up the rest. |
| You clicked **Terminate** or killed the run.                          | Frames up to terminate.     | The solver transitions to **Resumable**. Click **Resume** to continue from the last completed frame, or **Run** to clear the animation and restart. |
| Solver process crashed (segfault, OOM, server reboot).                | Whatever was auto-saved (if **Auto Save** was on) or just the frames already written. | Reconnect and **Start Server**. If auto-save snapshots exist, the solver comes up **Resumable** and **Resume** picks up from the last snapshot. If not, press **Run** to re-simulate. |

**Auto Save** is what distinguishes "lose a few seconds of solve" from "redo the last hour" in the third case. It's on the Scene Configuration → Advanced Params panel; enable it before long runs and leave the default interval for most scenes.

If **Fetch** finds a **session ID mismatch** at reconnect, the remote project on disk is not the one your .blend remembers (a fresh server start, a different cloud host, or a colleague's run). Either **Transfer** to replace the remote with your current scene, or connect to the host where the original run lives.

### Aborting a transfer or fetch

**Abort** interrupts the *current transfer or fetch*. It does not cancel a running simulation; for that, use **Terminate**. An aborted fetch clears the in-flight animation buffer; rerun **Fetch** to restart from the first missing frame. The Solver panel surfaces a warning line when the remote has more frames than have been fetched locally:

> N frames unfetched. Press "Fetch All Animation".

### Blender Python API

The same workflow is available from Python:

```python
from zozo_contact_solver import solver

# The core loop.
solver.transfer_data()
solver.run_simulation()
solver.fetch_animation()

# Iterate on parameters without resending geometry.
solver.param.step_size = 0.002
solver.update_params()

# Clean up.
solver.clear_local_animation()
solver.delete_remote_data()

# Recovery flow: graceful vs hard stop.
solver.save_and_quit_simulation()   # flush, exit server
solver.terminate_simulation()       # no flush, immediate
```

UNDER THE HOOD:

**Operator names**

| Button                | `bl_idname`                  |
| --------------------- | ---------------------------- |
| Transfer              | `solver.transfer`            |
| Run                   | `solver.run`                 |
| Resume                | `solver.resume`              |
| Update Params         | `solver.update_params`       |
| Fetch All Animation   | `solver.fetch_remote_data`   |
| Clear Local Animation | `solver.clear_animation`     |
| Delete Remote Data    | `solver.delete_remote_data`  |
| Terminate             | `solver.terminate`           |
| Save & Quit           | `solver.save_quit`           |
| Abort                 | `ssh.abort`                  |

Any `solver.*` method in Python that is not explicitly defined on the solver proxy is forwarded to the matching Blender operator, which is why the Python API above maps one-to-one onto this table.

**Mesh hash**

The topology hash is computed over each active group's vertex count, edge topology, and face topology. Pure transforms and material-parameter edits do not affect it.

**PC2 files on disk**

Per-object PC2 lives under `<blend_dir>/data/<blend_basename>/`. Each object has a `vert_N.bin` per simulated frame. A fetch also writes `map.pickle` and `surface_map.pickle` for the object-to-vertex mappings.

**`ContactSolverCache` modifier**

Mesh playback goes through a Blender `MESH_CACHE` modifier named `ContactSolverCache`, pointing at the object's PC2 file. The modifier sits in the **first modifier slot** with `frame_start = 1.0`, so it deforms the rest mesh before any other deformer runs. Curves are updated directly on every frame change and do not use a cache modifier.

**Session ID format**

The session ID is a 12-hex-character string. It is embedded in the PC2 header, stamped onto each object carrying the cache modifier, and attached to the remote project directory. That is what the mismatch warning on reopen is comparing against.

## Baking animation

After a successful **Fetch All Animation** the solver's output drives each simulated object through a `ContactSolverCache` cache modifier (meshes) or a per-CV curve cache (rods). That is perfect while you iterate on a shot, but it keeps every object tied to the add-on: the animation depends on the `.pc2` files on disk, the modifier on the mesh, and the group membership in the sidebar.

**Baking** converts that live cache into standard Blender animation: per-frame shape keys for meshes, per-control-point keyframes for curves. After a bake, the cache modifier is dropped, the object leaves its dynamics group, and the animation is self-contained Blender data that plays back in any scene, with or without the add-on installed.

There are two flavors:

- **Bake Animation**. Keyframes **every** fetched frame onto the object. Use this when you are done iterating and want to hand the scene off.
- **Bake Single Frame**. Captures the **current** frame's pose only and freezes the object there as Frame 1. Use this to turn a cloth-draping step into the rest pose of the next simulation, or to freeze a pin operation's final shape.

Two-panel before-and-after diagram. Before (left): the mesh carries a ContactSolverCache MESH_CACHE modifier with Format PC2 pointing at a per-object PC2 file under the blend's data folder, has pin vertex groups (ShoulderPins, CollarPins), and is assigned to a Dynamics Group (Cloth, Shell); it has no shape keys and no value fcurves. A central red arrow labeled Bake Animation (destructive, irreversible) points from the before panel to the after panel. After (right): the same mesh with the ContactSolverCache modifier removed (slot freed), the PC2 file on disk deleted, the pin vertex groups removed, the Dynamics Group assignment cleared, and the animation now stored as one shape key per fetched frame (ContactSolverBake_00001, _00002, and so on) with value fcurves using CONSTANT interpolation that drive each shape key 0 to 1 to 0 across its active frame. A footer explains that rods bake to per-control-point keyframes on co / handle_left / handle_right instead of shape keys, and that Abort during a modal bake restores the full pre-bake state. Everything on the left that is orange (modifier), mauve (pin vertex groups), or yellow (group membership) is the add-on's; baking removes those and the PC2 file on disk that backs the modifier. What remains on the right (green) is the object's own Blender data: per-frame shape keys plus the fcurves that drive them. The red arrow only goes one way.

NOTE: Baking requires simulation data to exist for the object. Run the Transfer -> Run -> Fetch All Animation loop (see Running the Simulation) first; the bake buttons are grayed out until there is a fetched cache to bake from.

### Baking is destructive: duplicate the .blend first

Once a bake finalizes, it cannot be undone from the UI. For every baked object the add-on:

- removes the `ContactSolverCache` modifier from the mesh (or drops the live curve cache for rods),
- **deletes the `.pc2` file** from `<blend_dir>/data/<basename>/` on disk, and
- removes the object from its dynamics group, together with any pin vertex groups attached to it.

**Recommended workflow**: always bake in a *copy* of your `.blend`, so the pre-bake scene (with its PC2 cache, modifiers, and group setup) stays intact as a fallback.

1. **Save** the current `.blend` (`File → Save`).
2. **Duplicate the `.blend` on disk**, outside Blender. Run `cp project.blend project_baked.blend` in a terminal, or copy the file in Finder / Explorer. Do *not* use `File → Save As` from the currently-open session; the PC2 files need to stay paired with the original `.blend`.
3. **Open the duplicate**. The Solver panel detects that the `ContactSolverCache` modifiers still point at `<blend_dir>/data/<old_basename>/` while the new `.blend`'s canonical data folder is `<blend_dir>/data/<new_basename>/`, and surfaces a migration prompt:

   The Solver panel in a freshly-opened duplicate. The error line (*Data path: …/data/project_baked/…pc2 does not exist.*) flags the missing canonical cache, the Migrate data/project/ -> data/project_baked/ button migrates in one click, and the Missing files (N) list underneath enumerates every PC2 that would need to be moved.

4. **Click Migrate**. A dialog appears with *From* / *To* folders and a **Keep Copy** checkbox (checked by default). Leave **Keep Copy** on; the operator then `shutil.copytree`s `data/<old>/` → `data/<new>/` and rewrites every `ContactSolverCache` modifier filepath to point at the new folder. The original `data/<old>/` stays untouched, so the source `.blend` keeps working.
5. **Save** the duplicate once more so the rewritten modifier filepaths are persisted.
6. **Bake** in the duplicate. The PC2 files that get deleted belong to `data/<new_basename>/`; the original `.blend`'s `data/<old_basename>/` is preserved.

TIP: Unchecking Keep Copy switches the migration from `shutil.copytree` to `os.rename`, saving disk space at the cost of leaving the original `.blend` pointing at a folder that no longer exists. Only turn it off when you are certain you want to abandon the source `.blend`.

WARNING: The migration button only appears when the mismatch is unambiguous: every `ContactSolverCache` modifier points into a single `data/<old>/` folder, and that folder is still on disk. Mixed prefixes or a missing source folder return no migration target, and the Solver panel shows only the missing-file warning. In that case the cleanest recovery is to manually copy `data/<old>/` -> `data/<new>/` with the file manager / `cp -r`, then reopen the `.blend`.

### The Solver panel (bake all objects)

The **Solver** panel exposes a row that bakes every dynamic object in the scene in one go. The buttons sit directly under **Fetch All Animation** and **Delete Remote Data**:

The scene-wide bake row on the Solver panel. Bake Animation (left) writes every fetched frame onto every simulated object as shape-key or curve keyframes; Bake Single Frame (right) freezes the current frame onto every simulated object as their new Frame 1.

Bake Animation runs as a modal job with a progress bar. The panel shows a live `Baking <object> [N/total]` status line and an Abort button while frames are keyframed. Abort restores every modifier / shape key / fcurve to its pre-bake state.

Bake Single Frame is synchronous and finishes in one click. Every dynamic object is frozen at the viewport's current evaluated pose, applied to frame 1, and removed from its group.

Both scene-wide buttons are enabled only when at least one active group has an assigned object carrying simulation animation (that is, only after a successful **Fetch**). While a bake job is running every other button on the Solver panel is inert; wait for the progress bar to finish or press **Abort**.

### The Dynamics Groups panel (bake one object)

Each group box has its own **Bake Animation** / **Bake Single Frame** row, sitting below the **Add Selected Objects** / **Remove Object** row. These bake only the object currently selected in the group's list, not the whole group or the whole scene:

The per-group bake row on each Dynamics Groups box. The two buttons are enabled when the row highlighted in the assigned-object list has a fetched cache. The baked object leaves the group (so it won't be re-simulated on the next Transfer) but keeps its keyframes.

Per-object **Bake Animation** uses the same modal progress + Abort flow as the scene-wide version, so the panel stays interactive for the other groups while one object bakes.

### What ends up on the object

The output format depends on the object type:

| Object type | What gets keyed                                              | Modifier removed       |
| ----------- | ------------------------------------------------------------ | ---------------------- |
| **Mesh**    | One shape key per fetched frame, named `ContactSolverBake_NNNNN`, each driven 0 to 1 to 0 with `CONSTANT` interpolation so exactly one key is active per frame. | `ContactSolverCache`   |
| **Curve**   | Per-control-point keyframes on `co` / `handle_left` / `handle_right` (bezier) or on `co` (poly / NURBS). | `ContactSolverCache` (curves keep the baked rest pose) |

After baking, the object:

- No longer appears in its **Dynamics Groups** slot; the assignment is cleared, together with any pin vertex groups attached to it.
- Still plays back its simulated motion in Blender's timeline, because the animation now lives in shape keys / fcurves on the object itself.
- Can be moved, duplicated, appended to other `.blend` files, or rendered on a render farm without shipping the `.pc2` cache.

WARNING: For meshes, Bake Animation adds one shape key per frame. The add-on refuses to bake a mesh that already carries non-Basis shape keys, because mixing baked keys with user-authored ones would double-blend. Remove extras manually (shape-keys panel, then trash icon) before baking.

### Aborting a bake

**Bake Animation** runs modally. While the bake is in progress the Solver panel shows:

- A status line (`Baking <object> [N/total]`) updating as frames are keyed.
- A progress bar (`% (done/total)`).
- An **Abort** button (red "X" icon).

Pressing **Abort** reverts every partial change: shape keys inserted so far are removed, their value fcurves are stripped, bezier handle types are restored, and the `ContactSolverCache` modifier + PC2 file stay intact, so you can immediately press **Bake Animation** again or go back to iterating.

**Bake Single Frame** does not run modally and cannot be aborted; it finishes in one operator call.

### Bake order and Static groups

**Bake Animation** walks through active groups in slot order (`object_group_0` → `object_group_31`) and bakes every assigned object. **Static** groups are included so their `ContactSolverCache` modifier and PC2 file get cleaned up too, leaving every bakeable group empty once the scene-wide bake finishes. See Static Objects -> Baking behavior for the full story on what bake does to a Static object depending on whether it was driven by Blender keyframes or by Static ops.

Groups that hold curves alongside meshes bake both: meshes emit shape keys, curves emit per-CV keyframes, and both end up on standard Blender timelines.

### Blender Python API

Baking is exposed through the add-on's registered operators:

```python
import bpy

# Scene-wide: bake every dynamic object.
bpy.ops.solver.bake_all_animation()          # modal; returns immediately
bpy.ops.solver.bake_all_single_frame()       # synchronous

# Per-object, per-group (group_index is the slot 0..31).
bpy.ops.object.bake_animation(group_index=0)
bpy.ops.object.bake_single_frame(group_index=0)

# Abort the running bake (no-op if none is running).
bpy.ops.solver.bake_abort()
```

For LLM / MCP-driven flows, the same actions are exposed as the `scene.bake_all_animation`, `scene.bake_all_single_frame`, `group.bake_group_animation`, and `group.bake_group_single_frame` handlers (see MCP Integration).

UNDER THE HOOD:

**Operator names**

| Button                                       | `bl_idname`                       |
| -------------------------------------------- | --------------------------------- |
| Solver panel → **Bake Animation**            | `solver.bake_all_animation`       |
| Solver panel → **Bake Single Frame**         | `solver.bake_all_single_frame`    |
| Dynamics Groups → **Bake Animation**         | `object.bake_animation`           |
| Dynamics Groups → **Bake Single Frame**      | `object.bake_single_frame`        |
| In-progress bake → **Abort**                 | `solver.bake_abort`               |

**Shape key naming**

Baked mesh shape keys are named `ContactSolverBake_<frame:05d>`. The value channel for each key is driven with three keyframes (`0` at `frame-1`, `1` at `frame`, `0` at `frame+1`), and every keyframe is forced to `CONSTANT` interpolation so frame `N` plays exactly one shape key, with no blending.

**Abort safety**

The modal operator snapshots enough state before it starts that an abort is non-destructive:

- For meshes: the pre-existing set of shape-key names and whether the shape-key block existed at all. Abort removes only the keys the bake added, and deletes the shape-key block if the bake created it.
- For curves: the pre-existing set of `(data_path, array_index)` fcurve keys and the per-spline `(handle_left_type, handle_right_type)` snapshot. Abort removes fcurves that didn't exist before and restores bezier handle types that were forced to `FREE` during the bake.

The `ContactSolverCache` modifier and the `.pc2` file on disk are never touched by an abort, so the pre-bake state is fully recoverable.

## JupyterLab integration

JupyterLab is a first-class way to drive the solver, not just a place to export a notebook. Once a scene has been transferred, the same project can be simulated, previewed, inspected, and iterated on entirely from a notebook on the solver host. Blender itself does not have to stay open: you can quit it, run the full simulation from JupyterLab, relaunch Blender later, and fetch the finished animation back onto the original meshes.

This page covers the whole loop in one place.

### When to reach for JupyterLab

- **Headless simulation.** You have a laptop that you need to close, but the solver host can keep running. Export, quit Blender, and drive `app.session.run()` from a notebook that stays alive on the server.
- **Parameter sweeps and variant generation.** Scripting `solver.param` / `solver.session.param` edits in a notebook is faster than clicking through the sidebar, and the results stream back as plots instead of viewport redraws.
- **Inspecting the codebase.** The notebook attaches to the same `frontend` package the add-on talks to, so you can poke at `app.scene`, `app.session`, `app.session.param`, the fixed-scene report, and the live solver state interactively. It is a much faster way to learn the API than reading source, and a convenient surface for summarizing what a project contains.
- **Long runs you want to leave unattended.** Kick off `app.session.run()` in a notebook cell, close the browser tab, and reconnect later to `app.session.stream()` the tail of stdout.

### The end-to-end loop

Sequence diagram with three lanes. Blender (1) on the left transferring and exporting to JupyterLab, then quitting; JupyterLab in the middle running app.session.run / preview / stream while frames land on disk; Blender (2) on the right launching later and pulling the animation back via Connect and Fetch. The first Blender session transfers the scene, exports the notebook, and optionally starts the solver, then quits. JupyterLab drives `run() / preview() / stream()` on the solver host while frames land on disk. A later Blender session launches, reconnects, and fetches the animation back onto the original meshes.

The thing that makes this work is the solver's on-disk project state: mesh / param pickles from **Transfer**, plus whatever frames the solver has produced. As long as those live on the solver host, it does not matter whether Blender, JupyterLab, or neither is currently attached.

NOTE: You do not have to switch to JupyterLab just to leave a run unattended. A simulation launched from the Blender add-on Run button also survives Disconnect and even quitting Blender; reconnect later and Fetch All Animation to pull the frames. See Disconnecting while a simulation runs. Reach for JupyterLab when you want to *drive* the run (parameter sweeps, live previews, interactive inspection), not just step away.

### Prerequisites

- JupyterLab running and reachable from the add-on, by convention on the solver host, on the **JupyterLab Port** (default `8080`).
- A live connection: **Connect** plus **Start Server** on the main panel.
- At least one **Transfer** into the current session so the notebook has something to attach to. Without it, `BlenderApp.open(...)` has no pickles to recover from.

### The Jupyter row

The main panel's **JupyterLab** section has three buttons:

| Button     | What it does                                                                  |
| ---------- | ----------------------------------------------------------------------------- |
| **Export** | Write the template notebook onto the solver host                              |
| **Open**   | Open `http://localhost:<port>/lab/tree/<path>` in your browser                |
| **Delete** | Remove the notebook file from the solver host                                 |

The JupyterLab section, expanded. The Export button writes the notebook onto the solver host. Open URL points the browser at it once an export exists, and Delete removes the server-side file.

**Export** pops a dialog for the target path. The default is `blender-export/<project_name>.ipynb` (or whatever you exported last). The filename must end with `.ipynb`. Writes go through the solver connection, not through Jupyter's REST API.

**Open** is enabled once an export exists. It does not check that the file still exists on the server; if you deleted it manually, you will get a 404 in the browser.

**Delete** removes the server-side file and clears the add-on's record of the last export, which disables **Open** again.

### The generated notebook

Three cells, all Python:

1. **Banner comment** naming the project, so the notebook is self-identifying when shared.

2. **Attach**:

   ```python
   from frontend import BlenderApp

   app = BlenderApp.open("<project>")
   app.scene.report()
   app.scene.preview()
   ```

   `BlenderApp.open(...)` recovers a built scene if one is available; otherwise it populates and builds from the uploaded mesh and parameters. It is a no-op if the scene is already built, so running this cell repeatedly is safe.

3. **Run**:

   ```python
   app.session.run()
   app.session.preview()
   app.session.stream()
   ```

   `run()` starts the solver (or attaches to an existing run, which is useful if you kicked it off from the add-on already). `preview()` gives a live frame playback widget, `stream()` tails stdout in realtime. To pick up a saved state instead, call `app.session.resume()` explicitly.

NOTE: The notebook's `BlenderApp` comes from the solver repo's `frontend` package, not the add-on. The add-on only writes the file.

### Simulating entirely from JupyterLab

Once the notebook is open, Blender is no longer required. A typical "quit Blender and simulate in the notebook" session looks like this:

1. In Blender, finish scene setup, **Transfer**, then **Export** notebook. Optionally press **Start Server** so a solver is already warm; if not, the notebook will spawn one when `run()` is called.
2. **Open** the notebook in your browser. Confirm the attach cell brings up the expected scene via `app.scene.report()` and `app.scene.preview()`.
3. **Quit Blender.** The solver host keeps the pickles and any running server process; closing Blender only drops the add-on's live connection.
4. From the notebook, run the simulation:

   ```python
   app.session.run()        # or app.session.resume() to continue
   app.session.preview()    # live frame widget in the notebook
   app.session.stream()     # tail solver stdout inline
   ```

5. Iterate in place. You can tweak parameters through `app.session.param.*` (see the Frontend Python API) and re-run without re-exporting. Because the notebook attaches to the pickled scene on disk, you can close the browser tab and reopen it later. The cell outputs may be gone, but rerunning the attach cell puts you right back where you were.

### Returning to Blender and fetching

When the run is done (or whenever you want the animation back on your Blender meshes):

1. **Launch Blender** and reopen the `.blend` you transferred from.
2. **Connect** to the same solver host. The session ID baked into the scene is how the add-on recognizes it is the same run.
3. Press **Fetch All Animation**. The add-on downloads the frames the JupyterLab run produced and wires up the PC2 cache modifier on each simulated mesh, exactly as if Blender had driven the run itself.
4. Scrub the timeline to confirm, then optionally **Bake** (see Baking Animation) to drop the cache dependency.

TIP: If the reopened `.blend` warns about a session mismatch, it means the solver host has a different session stamped than your `.blend` remembers. Either reconnect to the run you actually want (matching session), or accept the mismatch if you are deliberately attaching to a new one; see Simulating -> Sessions and recovery.

### Inspecting the codebase from a notebook

The same notebook is a very natural place to explore what the solver understands about a project. Useful idioms:

```python
app.scene.report()                       # group summary, counts, flags
app.scene.preview()                      # 3D preview of the fixed scene
app.session.param                         # live param object; tab-complete it
help(app.session)                         # frontend Session API surface
app.session.param.dyn("gravity")          # dynamic-parameter builder
```

Because `frontend` is just a Python package, `inspect.getsource(...)`, `??` magics, and regular `dir()` all work. This makes a notebook a good place to summarize a project (what groups exist, what pins are bound, what parameters are in play) without clicking through every sidebar section in Blender.

### Tips

- **Re-export after major scene changes.** The notebook's `BlenderApp.open(...)` picks up whatever pickles are on disk at the time the cell runs, so you normally do not need to re-export just to iterate on parameters. Re-export when you add or remove objects, rebuild groups, or change the project name.
- **Repeated exports overwrite.** The filename is derived from the project name, so hitting **Export** twice is safe.
- **Delete to tidy up.** Useful when flipping between experiments so stale `.ipynb` files do not accumulate under `blender-export/`.
- **If Open goes to the wrong port**, the add-on uses the **JupyterLab Port** from scene state. Adjust it in the Jupyter preferences row, not via the URL.
- **Headless-friendly.** `run()`, `preview()`, and `stream()` all work with Blender closed; the notebook does not depend on Blender's Python at all.

### See also

- Connecting to a solver host: required before Export is enabled.
- Simulating: covers **Transfer**, **Start Server**, and the **Fetch** button you press when you come back to Blender.
- Baking Animation: converting a fetched cache into standard Blender keyframes once a JupyterLab-driven run is done.
- Frontend Python API: the full surface of `app.scene`, `app.session`, and `app.session.param` that the notebook exposes.

UNDER THE HOOD:

**How it fits together**

Sequence diagram with three lanes. Blender plus add-on on the left, solver host in the middle, browser on the right. The add-on transfers mesh and parameters and exports the notebook to the solver host, opens the JupyterLab URL in the browser, and the browser attaches back to the solver host via BlenderApp.open. Transport topology: the add-on pushes mesh, params, and the notebook file through the solver connection, then opens the JupyterLab URL in the browser. Once loaded, the notebook attaches back to the solver host through `BlenderApp.open(...)`.

The running JupyterLab process lives on the solver host (or wherever the configured port can reach). The add-on never touches JupyterLab's REST API directly.

**Transport**

Writes go through the solver's control channel as JSON requests; the server resolves them under `<src>/examples/` and replies OK or an error. This indirection exists because writing through JupyterLab's own contents API was making the exported file (and its parent `blender-export/` directory) vanish after a while, a suspected interaction with JupyterLab's cloned-workspace state. Routing through the solver sidesteps that entirely.
