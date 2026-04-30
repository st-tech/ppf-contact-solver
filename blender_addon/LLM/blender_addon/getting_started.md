# Getting Started

This file condenses `docs/blender_addon/getting_started/index.md`, `install.md`, `tour.md`, and `first_simulation.md` into one self-contained onboarding reference covering prerequisites, installation, UI layout, and a complete first simulation walkthrough.

ZOZO's Contact Solver (https://github.com/st-tech/ppf-contact-solver) is a GPU-accelerated contact simulation engine; the Blender 5.0+ add-on is one front-end that ships with it, turning Blender into an interactive editor for the solver. You model in Blender, assign material groups, pins, and colliders, and the add-on streams geometry and parameters to the solver backend (local, SSH, Docker, or Windows native), runs the simulation, and pulls the animation back as a **Mesh Cache** modifier you can scrub on the timeline. By the end of this chapter you will have the add-on installed, a connection open, a single cloth sheet cached, and a simulated animation playing in the viewport.

## Where to go next

- **Connections**: set up the backend that matches your environment (local, SSH, Docker, Windows native), and learn how connection profiles let you switch between them in one click.
- **Workflow**: material parameters, pin operations, keyframed scene parameters, invisible colliders, snap-and-merge, and the full lifecycle from **Transfer** through **Fetch**.
- **Blender Python API**: drive every operator on this page from a script or a Jupyter notebook instead of the sidebar.

## Install

### Prerequisites

- **Blender 5.0 or newer.** The extension manifest pins `blender_version_min = "5.0.0"`; older builds will refuse to enable it.
- **A solver backend.** Any one of: a solver checkout on the same machine (simplest), an SSH-reachable Linux host, a Docker container, or a Windows workstation. The solver itself requires an NVIDIA GPU with CUDA 12.x. See Connections for the full matrix and GPU requirements. The add-on is just a client and runs fine on any machine Blender runs on (including macOS).
- **(Optional) paramiko / docker-py.** Needed only for SSH and Docker connections. You do not need to install them yourself. When you pick an SSH or Docker server type without the module present, the main panel surfaces an **Install Paramiko** / **Install Docker-Py** button that installs into the add-on's vendored `lib/` directory.

NOTE: The solver binary itself is not shipped with the add-on. You build or deploy it separately at the path you point the connection at. The add-on only looks for a `server.py` entry point there.

### Install the add-on

The add-on is packaged as a Blender 5.x extension (`blender_manifest.toml`, schema `1.0.0`, id `ppf_contact_solver`).

1. In Blender, open **Edit → Preferences → Get Extensions** (or **Install from Disk…** in the drop-down).
2. Point the dialog at the `blender_addon/` directory (or a zip of it) and confirm. Blender copies it into your user extensions folder and enables it.
3. Alternatively, drag-and-drop the `blender_addon/` folder onto an open Blender window. Blender 5.x recognises the extension manifest and offers the same install dialog.
4. Verify by opening the 3D viewport sidebar (`N`). A new tab labeled **ZOZO's Contact Solver** should appear with the panels described below.

TIP: If the sidebar tab is missing after install, the add-on probably crashed while enabling. Open **Window → Toggle System Console** and re-enable the extension from Preferences to see the traceback.

## Tour of the UI

All panels live in **View3D → Sidebar (`N`) → ZOZO's Contact Solver**.

### Backend Communicator

The main panel. Profile row (Open / Clear / Reload / Save), server-type selector, Project Name, **Connect** / **Disconnect**, **Start Server on Remote** / **Stop**, live status line, remote hardware readout, and realtime statistics. Enable **Debug Options** at the bottom to unlock shell, data-transfer, and reload-server tools.

Figure: the Backend Communicator panel with **Connect** (the button that opens the transport to the solver) highlighted.

### Solver

The buttons that drive a simulation: **Transfer**, **Update Params on Remote**, **Run**, **Resume**, **Fetch All Animation**, **Delete Remote Data**, **Clear Local Animation**, plus Bake buttons. The **JupyterLab** and **MCP Server** collapsible sections live inside this panel.

Figure: the Solver panel with **Transfer** (the button that uploads geometry and parameters to the solver) highlighted.

### Scene Configuration

Global solver inputs: FPS, frame count, step size, gravity, air density, air friction. Four collapsible sub-sections: **Wind**, **Advanced Params**, **Dynamic Parameters** (keyframed gravity / wind / air), and **Invisible Colliders** (walls and spheres).

Figure: the Scene Configuration panel. Every field here applies to the whole scene; the four collapsible sections at the bottom (Wind, Advanced Params, Dynamic Parameters, and Invisible Colliders) expand to reveal more inputs.

### Dynamics Groups

Up to 32 groups. Create a group, pick its type (**Solid** / **Shell** / **Rod** / **Static**), assign meshes, set per-group material parameters, manage pin vertex groups, and attach pin operations (**Move By**, **Spin**, **Scale**, **Torque**, **Embedded Move**).

Figure: the Dynamics Groups panel in its empty state with **Create Group** (the button that allocates a new group slot) highlighted. Each created group appears as its own box below.

### Snap and Merge

Snap vertex positions between two objects and register merge pairs so the solver stitches them at build time.

Figure: the Snap and Merge panel with **Snap A to B** (the KDTree-based vertex snap that pulls Object A's vertices onto Object B's closest vertices) highlighted. The panel is collapsed by default; click the header to expand.

### Visualization

Toggle overlay draws for group colors, pins, spin/scale centers, and other editing guides.

Figure: the Visualization panel with **Hide all pins** highlighted. Each checkbox hides one overlay category (pins, directional arrows, group color tints, snap indicators, pin operations) so you can declutter the viewport without actually disabling the underlying data.

### Other sections

The **Debug** tools (shell calls, data-transfer benchmarks, git pull on remote, add-on reload server) are hidden behind the **Debug Options** toggle on the Backend Communicator panel.

Figure: with **Debug Options** toggled on at the top of the Backend Communicator panel, a debug section unfolds below exposing Shell Calls, Data Transfer Tests, GitHub Repo on Remote / Local, API export, UUID migration, and the Add-on Local Debug Server controls.

## Your first simulation

This walks a cloth-over-sphere scene through a complete sim end-to-end. Adjust parameters later; the goal here is to see a cached drape play back in the timeline.

### Build the scene

Before touching the add-on, lay out the two objects the sim needs: a **subdivided plane** to act as the cloth, and a **sphere** underneath as a static collider for it to drape over.

1. **Delete the default cube.** A fresh Blender start-up ships one; select it and press X (or **Object → Delete**).

2. **Add the static sphere.** **Add → Mesh → Ico Sphere**, then bump **Subdivisions** in the operator redo panel to around **4** (~1200 faces). Prefer an ico-sphere over a UV-sphere: its near-uniform triangulation avoids direction-dependent stiffness at the poles when the cloth starts interacting with it. Shade-smooth the sphere (**Object → Shade Smooth**) so the silhouette stays clean as the cloth wraps it.

   Figure: after step 2, the ico-sphere at the origin. The wireframe shows the near-uniform triangulation that keeps contact response direction-independent.

3. **Add the cloth plane, well clear of the sphere.** **Add → Mesh → Plane**, scale it up (or pass `size=2.4` in the redo panel), and move it up along **Z** so it sits **above** the sphere with a clear gap; a few centimetres is plenty. The solver rejects self-intersecting rest geometry, so the plane must not touch the sphere in frame 1.

   Figure: after step 3, the plane hovers above the sphere with a visible gap. A single quad for now, with no internal topology to deform.

4. **Subdivide the plane enough to drape.** A 2×2 plane has no flexibility; it will land on the sphere as a rigid slab. Add a **Subdivision Surface** modifier (**Add Modifier → Subdivision Surface**), switch **Type** from the default *Catmull-Clark* to **Simple** (Catmull-Clark rounds corners and can introduce self-intersections), raise **Levels Viewport** to **5**, match **Render**, then **Apply** it so the solver sees the dense topology directly. That gives you a 33×33 grid, which is enough resolution to fold naturally over the sphere.

   Figure: after step 4, the Simple subdivision bakes into the mesh as a 33×33 grid of quads. That is the resolution the solver will see.

   TIP: As a rule of thumb, aim for an **average edge length of 1-3 % of the cloth's bounding-box diagonal**. Below ~1 % you pay a large simulation cost without much visual gain; above ~3 % folds and wrinkles get blocky. See Scene Setup via MCP for the full rationale.

### Register the objects with the add-on

1. **Pick a connection type.** In the **Backend Communicator** panel, choose `Local` if the solver lives on this machine. It has the fewest moving parts. Fill **Local Path** with the solver checkout (the folder containing `server.py`) and set **Project Name** to something short. For other backends see Connections and the per-backend pages (local, ssh, docker, windows).

   Figure: step 1, pick **Local** from the **Type** dropdown, fill the **Path** and **Project Name** fields, then click the highlighted **Connect** button.

2. **Connect, then start the server.** Click **Connect**. The status line flips to *Connected* when the handshake completes. Click **Start Server on Remote**. The add-on launches `server.py` on the remote in the background and waits up to 16 seconds for it to come up. Status advances to *Waiting for data* once the server answers.

3. **Create the Cloth group (Shell).** In the **Dynamics Groups** panel, click **Create Group**, set **Object Type** to **Shell**, and rename it *Cloth*. Select the plane in the 3D viewport, then click **Add Selected Objects** in the group. The add-on tints the plane green (the Shell overlay color) as confirmation.

   Figure: the **Dynamics Groups** panel after **Create Group**. The Assigned Objects list is still empty; **Add Selected Objects** (red outline) is the button you press next to attach the currently selected plane to this group.

4. **Create the Sphere group (Static).** Click **Create Group** again, set **Object Type** to **Static**, and rename it *Sphere*. Select the sphere, then **Add Selected Objects**. The sphere picks up the blue Static overlay color. At this point the scene should look like the image below: a green cloth sheet floating cleanly above a blue sphere.

   Figure: the starting scene. Green = Shell (cloth), blue = Static (sphere). The wireframe on the plane shows the Simple subdivision the solver will deform; the icosphere's uniform triangles are the contact surface the cloth will drape against.

   Material defaults (Baraff-Witkin cloth) are fine for a first run.

### Transfer, run, and play back

5. **Transfer.** In the **Solver** panel, click **Transfer**. This encodes the mesh and parameters, uploads both, and triggers the remote **build** stage. The status line walks through *Uploading Mesh Data, Uploading Parameters, Building, Ready*.

6. **Run.** Click **Run**. The status flips to *Simulation in Progress*. Realtime statistics appear; an **Abort** button lets you stop it early. Frames are fetched incrementally as they complete.

   NOTE: If **Run** is grayed out and you see *Clear local animation before running*, click **Clear Local Animation** first. Previous simulation output lingers on the objects until you do.

7. **Fetch and play back.** When the sim finishes, click **Fetch All Animation** to pull any frames the live-fetch missed. The add-on writes per-object PC2 caches into `data/<blend_basename>/` next to the `.blend` file and attaches a **Mesh Cache** modifier (named **ContactSolverCache**) to each simulated mesh, pinned as the first modifier. Scrub the timeline or press play; Blender reads vertex positions straight from the PC2 files, and the cloth settles onto the sphere frame-by-frame.

   Figure: the draped end state. The Shell overlay is still green and the Static overlay still blue; the only thing that has changed frame-to-frame is the vertex positions the Mesh Cache modifier reads out of the PC2 file.

   Figure: the **ContactSolverCache** modifier as it appears in **Properties → Modifier Properties** after the first fetch. That is where the simulation lives: a standard Blender **Mesh Cache** modifier in **PC2** format, with its **File Path** pointing at a `.pc2` file under `<blend_dir>/data/<blend_basename>/`. Scrubbing the timeline just reads vertex positions from that file through this modifier.

TIP: Save the `.blend` at this point. The add-on migrates PC2 temp files into `data/<basename>/` on save and writes a manifest so the scene reopens with its caches intact.

TIP: To hand off the scene without the add-on, convert the live cache into standard Blender keyframes. See Bake Animation.

NOTE: Both viewport screenshots on this page were produced end-to-end via MCP, no human clicking. See Scene Setup via MCP.
