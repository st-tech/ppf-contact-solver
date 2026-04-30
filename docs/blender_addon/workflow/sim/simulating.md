# 🚀 Running the Simulation

Once the scene is organized into groups, parameters are set, and pins /
colliders are in place, the day-to-day loop is: **Transfer -> Run -> Fetch**,
with **Resume**, **UpdateParams**, and **ClearAnimation** for iteration.

## The Solver Panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab.
The **Solver** panel is the second panel in the tab, directly below
**Backend Communicator** and above **Scene Configuration**, **Dynamics
Groups**, **Snap and Merge**, and **Visualization**. It is always visible
(never collapsed by default) because it is the primary control surface
during simulation work.

```{figure} ../../images/simulating/solver_connected.png
:alt: Solver panel immediately after connecting and starting the server
:width: 500px

The Solver panel right after the server comes up. Only **Transfer** is
enabled; everything downstream of it is grayed out until the remote has
the scene. The info line at the bottom (*Click "Transfer" to upload
data*) reinforces which step is next.
```

The panel is laid out as a single vertical column of buttons, status
indicators, and controls:

1. **Connection row.** At the very top of the panel is a row showing the
   current connection target (host and port) and a **Connect** /
   **Disconnect** button. When connected, the label changes to show the
   active session.

2. **Primary action buttons.** Below the connection row, the main
   buttons are stacked vertically in this order:
   - **Transfer**. Uploads geometry and parameters to the solver.
   - **Run**. Starts the simulation.
   - **Resume**. Continues a paused or partially completed run.
   - **Update Params**. Re-uploads parameters without resending geometry.
   - **Fetch All Animation**. Downloads simulation results.
   - **Bake Animation** / **Bake Single Frame**. Convert the fetched
     animation into standard Blender keyframes (see
     [Baking Animation](baking.md)).

3. **Status line.** Between the primary action buttons and the secondary
   controls, a status line displays the current solver state. This line
   updates in real time during long operations (see "Visual feedback
   during each stage" below).

4. **Secondary controls.** Below the status line:
   - **Clear Local Animation**. Removes fetched animation data.
   - **Delete Remote Data**. Wipes the solver's project data.
   - **Terminate**. Hard-stops the running simulation.
   - **Save & Quit**. Gracefully shuts down the server.
   - **Abort**. Interrupts the current transfer or fetch operation.

5. **Progress bar.** During **Fetch All Animation**, a horizontal
   progress bar appears inline within the panel, showing download
   progress as a percentage alongside bandwidth statistics.

Buttons that are not applicable to the current state are grayed out. For
example, **Run** is grayed out until a successful **Transfer** has
completed, and **Resume** is grayed out unless a simulation has been
paused or partially run.

## The Buttons

| Button                    | What it does                                                         |
| ------------------------- | -------------------------------------------------------------------- |
| **Transfer**              | Multi-stage pipeline: delete existing remote data → send mesh → send params → build. |
| **Run**                   | Warns on stale mesh hash, clears existing animation, starts the solve. |
| **Resume**                | Resumes a paused / partially-run simulation.                         |
| **Update Params**         | Re-encodes and uploads parameters, then rebuilds. No geometry resend. |
| **Fetch All Animation**   | Downloads per-frame vertex data and applies it as PC2 animation.     |
| **Clear Local Animation** | Removes simulation keyframes and `ContactSolverCache` modifiers. Preserves pin keyframes. |
| **Delete Remote Data**    | Asks the server to wipe its current project data.                    |
| **Terminate**             | Hard-stops the current simulation on the server.                     |
| **Save & Quit**           | Graceful shutdown: flushes state to disk, then exits the server.     |
| **Abort**                 | Interrupts the *current* transfer or fetch. Does not touch running sim. |

`Terminate` and `Save & Quit` target the server itself; **Terminate** is
the hard equivalent of pulling the plug, while **Save & Quit** lets the
solver flush its state first so a later reconnect can pick up from there.

```{figure} ../../images/simulating/solver_state_machine.svg
:alt: Five state boxes arranged left to right (Connected, Ready, Running, Complete, Fetched) with the status-line text and enabled-button list inside each box. Blue solid arrows mark the canonical forward flow (Transfer → Run → sim ends → Fetch); purple dashed arrows mark loops and recovery transitions (Update Params self-loop on Ready, Terminate and Resume between Running and Complete, Save & Quit from Running back to Connected, Clear Local Animation from Fetched back to Ready). A footer explains that Delete Remote Data is reachable from any post-Transfer state, that Abort only interrupts a Transfer or Fetch, and that the panel is otherwise fully inert while Running.
:width: 820px

Which buttons light up in each state. The canonical forward flow runs
left-to-right along the blue arrows; the purple dashed arrows cover the
loops (**Update Params**), recovery transitions (**Clear Local
Animation**, **Resume**), and early exits (**Terminate**, **Save &
Quit**). If a button is grayed out, find the current state on the
diagram; the enabled list inside that box is the full answer.
```

```{figure} ../../images/simulating/solver_ready.png
:alt: Solver panel after a successful Transfer, with Run enabled
:width: 500px

After **Transfer** completes, the panel settles into the *Ready to Run*
state. **Run** lights up (the play icon turns solid), and **Delete
Remote Data** is also armed because the remote now holds a project.
This is the state you press **Run** from.
```

## Visual Feedback During Each Stage

The **status line** on the Solver panel keeps you informed of what the
solver is doing at every moment. Here is what you see during each major
operation:

### Transfer

The status line cycles through several sub-stages as the transfer
proceeds:

1. **"Uploading Mesh Data"**. The add-on serializes and sends vertex
   positions, edges, and faces for every active group.
2. **"Uploading Parameters"**. Material parameters, scene parameters,
   pins, operations, merge pairs, and invisible colliders are encoded
   and sent.
3. **"Building"**. The solver is constructing its internal data
   structures (BVH trees, constraint graphs, contact maps). This step
   can take a few seconds for complex scenes.
4. **"Ready"**. The transfer is complete. The **Run** button is now
   enabled.

If any sub-stage fails, the status line shows an error message in red
text describing the failure (e.g. "Transfer failed: mesh has zero
faces"). The panel remains in the pre-transfer state so you can fix the
issue and retry.

### Run

Once the simulation starts:

1. The Solver-panel status line reads **"Simulation Running..."** and a
   blue progress bar appears on the **Backend Communicator** panel
   above, labeled with the same status.
2. Live counters appear on **Backend Communicator** in two blocks.
   **Realtime Statistics** shows the current `frame`, `time-per-frame`
   and `time-per-step` wall-clock, `num-contact`, `newton-steps`,
   `pcg-iter`, and `stretch`. **Scene Info** tracks `Simulated Frames`
   against `Total Frames` so you can see how far through the run you
   are.
3. When the simulation completes, the status line returns to **"Ready
   to Run"**, `Simulated Frames` reaches `Total Frames`, and a warning
   row (*N frames unfetched. Press "Fetch All Animation"*) appears
   above the Solver panel, directing you at the next step.

```{figure} ../../images/simulating/sim_in_progress.png
:alt: Backend Communicator panel mid-solve. Status "Simulation Running...", blue progress bar, Realtime Statistics block showing frame, time-per-frame, time-per-step, num-contact, newton-steps, pcg-iter, stretch, and Scene Info with Simulated Frames 96 of 240
:width: 500px

Backend Communicator mid-solve: the status line reads *Simulation
Running...*, the blue progress bar carries the same label, and the
**Realtime Statistics** block updates in place as each frame lands. The
**Scene Info** block below shows `Simulated Frames` ticking toward
`Total Frames`.
```

```{figure} ../../images/simulating/sim_complete.png
:alt: Backend Communicator panel after a completed run. Status "Ready to Run", Average Statistics block, Scene Info with Simulated Frames 240 of 240, and a warning row reading "178 frames unfetched. Press Fetch All Animation" above the Solver panel
:width: 500px

After the run finishes, the status line returns to *Ready to Run*, the
live counters collapse into an **Average Statistics** block, and
`Simulated Frames` matches `Total Frames`. The warning row above the
Solver panel tells you exactly how many frames are still on the remote
and which button to press next.
```

### Fetch

During a fetch:

1. A **progress bar** fills from left to right as frame data is
   downloaded. The bar shows a percentage and the number of frames
   fetched so far (e.g. `120 / 240 frames`).
2. **Bandwidth statistics** appear alongside the progress bar (e.g.
   `12.3 MB/s`).
3. On completion, the status line reads **"Animation Ready"** and the
   fetched frames are immediately available for timeline scrubbing.

```{figure} ../../images/simulating/solver_running.png
:alt: Solver panel with every action disabled while the simulation is running
:width: 500px

While the simulation is live, the Solver panel goes fully inert; every
button is grayed out so you cannot double-fire an action or tear down
state mid-solve. The progress counter and per-frame stats live on the
**Backend Communicator** panel above; this panel simply stays out of the
way until the solve ends.
```

```{figure} ../../images/simulating/solver_fetched.png
:alt: Solver panel after Fetch, with fetched animation available locally
:width: 500px

Once **Fetch All Animation** finishes, the panel returns to its
post-sim resting state: **Fetch All Animation** and **Delete Remote
Data** stay enabled so you can re-fetch or wipe the remote project, and
the info line *Clear local animation before running* reminds you that a
second **Run** needs the previous animation cleared first.
```

## Update Params vs Transfer

When iterating on a scene, the question comes up: does this change
need a full **Transfer**, or is **Update Params** enough? The rule is:

- **Transfer** re-sends geometry and parameters, then rebuilds.
- **Update Params** re-sends parameters only, then rebuilds. Mesh
  buffers on the server are preserved, which is why it completes much
  faster on large scenes.

So **Transfer** is required whenever mesh topology or group membership
changes; **Update Params** is enough for everything else that lives in
the parameter payload (scene settings, material params, pins,
colliders, dynamic parameters). The table below enumerates the common
edits:

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

The **Mesh hash mismatch** warning in the next section is the add-on's
safety net: if you skip **Transfer** after a topology change, it shows
up before **Run** or **Fetch** and tells you to re-transfer.

## "Groups Have Changed" Warning

If you edit your meshes (add or remove vertices, change group membership,
reassign an object's type) after **Transfer** but before **Run** or
**Fetch**, the panel shows a warning:

> Mesh hash mismatch: groups have changed since last transfer.

The warning does not block you, but it means the solver's data no longer
matches what is in Blender. Click **Transfer** again to re-upload before
running or fetching.

Pure transforms and material-parameter edits do not trigger this warning;
only topology changes do.

## How Animation Plays Back

After **Fetch**, the add-on downloads per-frame vertex data next to
your `.blend` file and wires it up to each simulated mesh. Scrub the
timeline and the mesh deforms to the solver's output. Curves (rods)
update directly on every frame change.

Internally the hookup lives in the **Modifier Properties** tab, as a
`ContactSolverCache` entry whose **File Path** points at a `.pc2`
file under `<blend_dir>/data/<blend_basename>/`:

```{figure} ../../images/simulating/pc2_modifier.png
:alt: Blender Properties editor on the Modifier tab showing the ContactSolverCache MESH_CACHE modifier with Format PC2 and File Path //data/project/Cube.pc2
:width: 500px

What the **ContactSolverCache** entry looks like in **Modifier
Properties** after a successful fetch. The **File Path** points at
the per-object `.pc2` file under
`<blend_dir>/data/<blend_basename>/`, and that is what plays the
simulation back while you scrub. Leave it in place; deleting it (or
its `.pc2` file) removes the animation. To convert the fetched
animation into regular Blender animation (shape keys + fcurves), run
**Bake Animation** instead; see [Baking](baking.md).
```

:::{tip}
Save the `.blend` after fetching. The add-on migrates the fetched PC2
files into a permanent location on save, so the animation survives
closing and reopening the file.
:::

## Disconnecting While a Simulation Runs

Once **Run** is pressed on a remote backend (SSH, Docker, or Windows
Native), the solver is doing its work on the remote host; Blender is
just watching. You do **not** have to keep Blender open for the run to
continue:

- Press **Disconnect** on the Backend Communicator to drop the live
  connection. The remote solver keeps going.
- You can even quit Blender entirely. The remote process is owned by
  the solver host, not by the add-on.
- Later (minutes, hours, or after a reboot of your workstation),
  launch Blender, reopen the same `.blend`, press **Connect**, and
  **Fetch All Animation** pulls in whatever frames have landed on disk
  so far.

The session ID baked into the `.blend` is what the add-on uses to
recognize "this is the same run I started before". See
[Sessions and recovery](#sessions-and-recovery) below for how the
session check works, and
[Auto-save and graceful shutdown](#auto-save-and-graceful-shutdown) for
making sure the solver's own state survives a crash between sessions.

This is not the same as running the sim *from* JupyterLab. In that
scenario the sim was launched from the Blender add-on; JupyterLab just happens to be another
way to drive a project that lives on the same solver host. See
[JupyterLab](jupyterlab.md) if you also want to poke at the run from a
notebook while Blender is closed.

## Sessions and Recovery

Every successful connect mints a fresh **session ID** and the add-on
stamps simulation artifacts (the fetched PC2 files and the remote
project directory) with it. Saving the `.blend` stores the active
session on the scene; on reopen, reconnecting compares the new session
against the saved one and warns if they differ, meaning the fetched
frames on disk may no longer correspond to anything the remote knows
about. This
is how the add-on distinguishes "this is the sim I was running before I
closed Blender" from "this is a fresh run on a different server".

## Auto-Save and Graceful Shutdown

Two related features:

- **Auto-save** (the **Auto Save** and **Auto Save Interval** fields on
  the Scene Configuration → Advanced Params sub-panel): when enabled,
  the solver periodically dumps its state so a crash or disconnect does
  not cost all the progress. This runs inside the server process.
- **Save & Quit**: a one-shot operator that asks the server to flush
  state and exit cleanly. After **Save & Quit**, the next reconnect can
  pick up the run where it left off.

**Terminate** does not flush state. Use it when the simulation is
misbehaving and you want it gone.

### Recovery Scenarios

The behavior depends on *who* failed. Three cases cover the common
ones; the table shows the first move in each.

| What happened                                                         | What the solver has on disk | What to do next                                                                                       |
| --------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| You closed Blender or lost the network mid-run. Solver kept running.  | Everything up to "now".     | Reopen the .blend, **Connect**, **Fetch All Animation**. If still running, leave it. If it finished while you were gone, fetch picks up the rest. |
| You clicked **Terminate** or killed the run.                          | Frames up to terminate.     | The solver transitions to **Resumable**. Click **Resume** to continue from the last completed frame, or **Run** to clear the animation and restart. |
| Solver process crashed (segfault, OOM, server reboot).                | Whatever was auto-saved (if **Auto Save** was on) or just the frames already written. | Reconnect and **Start Server**. If auto-save snapshots exist, the solver comes up **Resumable** and **Resume** picks up from the last snapshot. If not, press **Run** to re-simulate. |

**Auto Save** is what distinguishes "lose a few seconds of solve" from
"redo the last hour" in the third case. It's on the Scene Configuration
→ Advanced Params panel; enable it before long runs and leave the
default interval for most scenes.

If **Fetch** finds a **session ID mismatch** at reconnect, the remote
project on disk is not the one your .blend remembers (a fresh server
start, a different cloud host, or a colleague's run). Either
**Transfer** to replace the remote with your current scene, or connect
to the host where the original run lives.

## Aborting a Transfer or Fetch

**Abort** interrupts the *current transfer or fetch*. It does not cancel
a running simulation; for that, use **Terminate**. An aborted fetch
clears the in-flight animation buffer; rerun **Fetch** to restart from
the first missing frame. The Solver panel surfaces a warning line when
the remote has more frames than have been fetched locally:

> N frames unfetched. Press "Fetch All Animation".

## Blender Python API

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

:::{admonition} Under the hood
:class: toggle

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

Any `solver.*` method in Python that is not explicitly defined on the
solver proxy is forwarded to the matching Blender operator, which is why
the Python API above maps one-to-one onto this table.

**Mesh hash**

The topology hash is computed over each active group's vertex count,
edge topology, and face topology. Pure transforms and material-parameter
edits do not affect it.

**PC2 files on disk**

Per-object PC2 lives under `<blend_dir>/data/<blend_basename>/`. Each
object has a `vert_N.bin` per simulated frame. A fetch also writes
`map.pickle` and `surface_map.pickle` for the object-to-vertex mappings.

**`ContactSolverCache` entry**

Mesh playback is driven by a `ContactSolverCache` entry in the
**Modifier Properties** tab, pointing at the object's PC2 file. It
sits in the **first modifier slot** with `frame_start = 1.0`, so it
deforms the rest mesh before any other deformer runs. Curves are
updated directly on every frame change and do not need an entry in
that tab.

**Session ID format**

The session ID is a 12-hex-character string. It is embedded in the PC2
header, stamped onto each simulated object, and attached to the remote
project directory. That is what the mismatch warning on reopen is
comparing against.
:::
