# 🎬 Your First Simulation

This page takes a cloth-over-sphere scene from an empty Blender file to a
drape playing back in the timeline. You build two objects, connect to a
solver, transfer the scene, run it, and fetch the result. Tune parameters
later; the goal here is one complete run.

The default scene settings simulate 180 frames at 60 FPS, which is 3
seconds. See [Scene Parameters](../workflow/params/scene.md) to change them.

## Before You Start

- The add-on is installed ([Install](install.md)) and a solver is unzipped or
  built where Blender can reach it ([Connections](../connections/index.md)).
- Optional: save the `.blend` (**File > Save**) before step 5, so that the
  copy icon beside **Project Name** can fill in the file name.
- If the file is still unsaved when you fetch the animation, the animation is
  kept in a temporary folder and moved into the `data` folder next to the
  `.blend` when you first save.

## Build the Scene

Lay out two objects before you touch the add-on: a subdivided **plane** that
acts as the cloth, and a **sphere** underneath it that acts as a static
collider for the cloth to drape over.

1. **Delete the default cube.** A fresh Blender start-up has one. Select it
   and press <kbd>X</kbd> (or use **Object > Delete**).

2. **Add the static sphere.** Choose **Add > Mesh > Ico Sphere**, then in the
   **Adjust Last Operation** panel set **Subdivisions** to **4** (1,280
   triangles) and leave **Radius** at 1 m. The triangles of an ico sphere are
   close to uniform in size, unlike those of a UV sphere. Optionally choose
   **Object > Shade Smooth**, which changes how the sphere is drawn and not
   its geometry.

   ```{figure} ../images/getting_started/step2_sphere.png
   :alt: Ico sphere at the origin, drawn with its wireframe
   :width: 280px

   After step 2: the ico sphere at the origin, with its wireframe shown.
   ```

3. **Add the cloth plane, clear of the sphere.** Choose **Add > Mesh >
   Plane**, then in the **Adjust Last Operation** panel set **Size** to
   **2.4 m** and **Location Z** to **1.05 m**. The sphere's top is at Z = 1 m
   (with the 3D cursor at the origin), so this leaves a gap of 5 cm. By
   default the solver rejects a scene that starts with overlapping objects, so
   the plane must not touch the sphere in frame 1.

   ```{figure} ../images/getting_started/step3_plane.png
   :alt: Square plane above the ico sphere
   :width: 380px

   After step 3: the plane above the sphere. It is a single quad, with no
   internal geometry to deform.
   ```

4. **Subdivide the plane so it can drape.** A single-quad plane has no
   flexibility and would fall onto the sphere as a rigid slab. Choose **Add
   Modifier > Generate > Subdivision Surface**, switch **Type** from
   *Catmull-Clark* to **Simple** (Simple leaves the plane's outline unchanged,
   while Catmull-Clark rounds the corners), set **Levels Viewport** to **5**, set
   **Render** to the same value, and click **Apply** so the solver receives
   the subdivided mesh. The result is a grid of 32 × 32 quads (33 × 33
   vertices).

   ```{figure} ../images/getting_started/step4_subdiv.png
   :alt: Plane subdivided into a 32 by 32 grid of quads above the ico sphere
   :width: 380px

   After step 4: the applied Simple subdivision, a 32 × 32 grid of quads.
   This is the mesh the solver receives.
   ```

   :::{tip}
   Aim for an **average edge length of 1–3 % of the cloth's bounding-box
   diagonal**. Here the edges are 0.075 m against a diagonal of 3.39 m, which
   is 2.2 %. Finer meshes cost more to simulate, and coarser ones fold in
   blocky wrinkles. See [Scene Setup via MCP](../integrations/mcp_scene_setup.md)
   for the full explanation.
   :::

## Connect to the Solver

5. **Pick a connection type.** In the **Backend Communicator** panel, set
   **Type** to your own platform's native type: **Windows Native**,
   **macOS Native**, or **Linux Native**. A native type runs the solver on
   this machine and has the fewest moving parts. Then fill in the fields:

   - **Solver Path**: the unzipped solver distribution, or your repository
     checkout. A subfolder is accepted and resolved upward, and a *Solver
     path valid* line under the field confirms the folder. The per-platform
     page lists the exact layout ([Windows](../connections/windows.md),
     [macOS](../connections/macos.md), [Linux](../connections/linux.md)).
   - The **GPU** / **CPU** toggle (the **Compute Device** setting): leave it
     on **GPU** unless this machine has no GPU.
   - **Project Name**: a short name using only letters, digits, `.`, `-`, and
     `_` (no spaces). When the `.blend` is saved and the field still reads
     `unnamed`, the copy icon beside it fills in the file name.

   For SSH and Docker connections, see [Connections](../connections/index.md)
   and the pages for [SSH](../connections/ssh.md) and
   [Docker](../connections/docker.md).

6. **Connect, then start the server.** Click **Connect**. For a native type
   this checks that **Solver Path** holds a build for the device you picked,
   and the status line reads `Status: Waiting for Server Start...`. Click
   **Start Server on Remote**; the label is the same for every type, and on a
   native type the server starts on this machine. The button reads **Server
   Starting...** while the server starts, and the status line changes to
   `Status: Waiting for Data` once the server answers.

   If the start fails with a port-in-use error, see
   [Port N is in use](../troubleshooting.md#port-n-is-in-use), which also
   explains the **Force Terminate Process** button that appears beside the
   error.

## Register the Objects with the Add-on

7. **Create the Cloth group (Shell).** In the **Dynamics Groups** panel,
   click **Create Group**, set the unlabeled type dropdown beside **Name** to
   **Shell**, and rename the group *Cloth*. Select the plane in the 3D
   Viewport, then click **Add Selected Objects** in the group. The plane is
   drawn in green, the Shell overlay color.

   ```{figure} ../images/getting_started/step3_group_created.png
   :alt: Dynamics Groups panel after Create Group, Add Selected Objects highlighted
   :width: 500px

   The **Dynamics Groups** panel after **Create Group**. The Assigned
   Objects list is still empty; **Add Selected Objects** (red outline)
   is the button you press next to attach the selected plane to this
   group.
   ```

8. **Create the Sphere group (Static).** Click **Create Group** again, set
   the type dropdown to **Static**, and rename the group *Sphere*. Select the
   sphere, then click **Add Selected Objects**. The sphere is drawn in blue,
   the Static overlay color. The scene should look like the image below: a
   green cloth sheet above a blue sphere.

   ```{figure} ../images/getting_started/first_sim_start.png
   :alt: 3D viewport showing a green subdivided plane above a blue ico sphere
   :width: 640px

   The starting scene. Green is Shell (cloth) and blue is Static (sphere).
   The wireframe on the plane shows the Simple subdivision the solver will
   deform.
   ```

   The default material (Baraff-Witkin cloth) is fine for a first run.

## Transfer, Run, and Play Back

9. **Transfer.** In the **Solver** panel, click **Transfer**. The panel
   encodes the mesh and the parameters, uploads them, and builds the scene on
   the solver. A progress bar reads *Encoding scene geometry...*, *Encoding
   parameters...*, *Uploading scene...*, and *Building Scene...* in turn,
   and the status line ends at `Status: Ready to Run`.

10. **Run.** Click **Run**. The status reads *Initializing...* while the
    solver starts, then *Simulation Running...* once it reports that it has
    finished initializing. **Realtime Statistics** appear, and the **Save and
    Quit** and **Terminate** buttons end the run early. The newest frame
    appears in the viewport as the run proceeds, and the frames in between
    stay on the solver host until you fetch them in step 11. See
    [Running the Simulation](../workflow/sim/simulating.md).

    :::{note}
    If **Run** is grayed out and the panel shows *Clear local animation before
    running*, click **Clear Local Animation** first. The previous run's
    animation stays attached to the objects until you do.
    :::

11. **Fetch and play back.** When the run finishes, click **Fetch All
    Animation**. If frames are still unfetched, the **Solver** panel says how
    many, and this downloads them. When the fetch finishes, Blender's frame
    range is set to the fetched frames and playback starts, and the cloth
    settles onto the sphere frame by frame. The animation is stored in
    `data/<blend name>/` next to the `.blend` and plays through a **Mesh
    Cache** modifier named `ContactSolverCache` on the plane.

    ```{figure} ../images/getting_started/first_sim_end.png
    :alt: 3D viewport showing green cloth draped over the blue sphere
    :width: 640px

    The draped end state. The Shell overlay is still green and the Static
    overlay still blue; only the vertex positions that Blender reads from the
    fetched animation change from frame to frame.
    ```

    ```{figure} ../images/simulating/pc2_modifier.png
    :alt: Blender Properties editor on the Modifier tab showing the ContactSolverCache Mesh Cache modifier with Format PC2 and a File Path inside the data folder
    :width: 500px

    What the fetched result looks like in **Properties > Modifier
    Properties**. The **File Path** points into the `data/<blend name>/`
    folder next to the `.blend`. Blender reads that file as you move the
    playhead.
    ```

:::{admonition} Under the hood
:class: toggle

Each simulated mesh gets one point-cache file in PC2 format. Until the
`.blend` is first saved, the files are written to a temporary folder, and the
add-on moves them into `data/<blend name>/` when you save. See
[How Animation Plays Back](../workflow/sim/simulating.md#how-animation-plays-back).
:::

:::{tip}
Keep the `data/<blend name>/` folder next to the `.blend` when you move or
share the file. The modifier finds the animation through a path relative to the
`.blend`, so the scene reloads it from there.
:::

:::{tip}
To hand off the scene without the add-on, convert the fetched animation into
standard Blender keyframes. See [Bake Animation](../workflow/sim/baking.md).
:::

## Where to Go Next

- **Change how the fabric behaves.** [Material Parameters](../workflow/params/material.md)
  holds density, stiffness, bending and friction for each group, and
  [Fabric Presets](../workflow/params/fabric_presets.md) starts from a known
  fabric.
- **Hold or move parts of an object.** [Pins and Operations](../workflow/constraints/pins.md)
  anchors vertices and drives them, and
  [Invisible Colliders](../workflow/constraints/colliders.md) adds walls and
  spheres without a mesh.
- **Set the scene.** [Scene Parameters](../workflow/params/scene.md) holds
  gravity, wind, the time step and the frame count.
- **Organize groups.** [Object Groups](../workflow/scene/object_groups.md)
  covers the group types and how objects are assigned.
- **Keep the result.** [Baking](../workflow/sim/baking.md) and
  [Exporting](../workflow/sim/exporting.md) turn the fetched animation into
  standard Blender keyframes, or into files other applications read.
- **Fix a problem.** [Troubleshooting](../troubleshooting.md) lists the
  messages the add-on shows and what to do about each.
