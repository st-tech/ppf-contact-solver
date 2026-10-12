# 📓 JupyterLab

Once a scene has been transferred, a notebook on the solver host can run
the simulation, preview it, and inspect it. Blender does not have to stay
open: quit it, let the notebook run the solve, start Blender again later,
and fetch the finished animation onto the original meshes.

This page covers the whole loop: exporting the notebook, running the solve
from it, and fetching the result.

## When to Reach for JupyterLab

- **Simulate with Blender closed.** Export the notebook, start the run with
  `app.session.run()`, then quit Blender or close the laptop while the
  solver host keeps working.
- **Leave a long run unattended.** Start `app.session.run()` in a cell and
  close the browser tab. The solver keeps running on the host. Reopen the
  notebook and call `app.session.stream()` from the same kernel to see the
  tail of the solver's output. The kernel is the Python session behind the
  notebook, which holds the `app` variable; **Kernel → Restart Kernel**
  starts a new one and discards it.
- **Inspect a project.** `app.scene` and `app.session` come from the
  solver's `frontend` package, so you can read them interactively. See
  [Inspecting a Project from a Notebook](#inspecting-a-project-from-a-notebook).
- **Set a parameter from Python.** Override it on the session and build
  again; the *Overriding a Session Parameter* section of the
  [JupyterLab Python API](../../../jupyterlab_api/index.rst) shows how and
  why a built session ignores a later change. The rebuild clears the session
  folder, like the attach cell does (see the warning under
  [The Generated Notebook](#the-generated-notebook)).

## The End-to-End Loop

```{figure} ../../images/jupyterlab/end_to_end_loop.svg
:alt: Sequence diagram with three lanes. Blender (1) on the left transferring the scene and exporting the notebook to JupyterLab, then quitting; JupyterLab in the middle running app.session.run / preview / stream while frames land on disk; Blender (2) on the right launching later and pulling the animation back via Connect and Fetch
:width: 760px

The three-lane loop. The first Blender session transfers the scene and
exports the notebook, then quits. JupyterLab drives
`run() / preview() / stream()` on the solver host while frames land on
disk, and the notebook starts the solver itself. A later Blender session
connects and fetches the animation onto the original meshes. The figure
draws **Start Server** after the export, but the server has to be running
before you press **Transfer** and **Export**, so start it first unless it
is already running (see [Prerequisites](#prerequisites)).
```

The loop works because the solver host holds the scene that **Transfer**
uploaded and the frames the solver writes. While they stay on the host, it
does not matter whether Blender, JupyterLab, or neither is attached.

:::{note}
You do not have to switch to JupyterLab to leave a run unattended. A
simulation started with the add-on's **Run** button also survives
**Disconnect** and quitting Blender; reconnect later and press **Fetch All
Animation**. See
[Disconnecting while a simulation runs](simulating.md#disconnecting-while-a-simulation-runs).
Use JupyterLab when you want to drive the run from Python, preview it live,
or inspect it, not only to step away.
:::

## Prerequisites

- **JupyterLab on the solver host.** The solver's launcher or install
  script starts it and prints its URL; see the
  [solver README](https://github.com/st-tech/ppf-contact-solver).
  **Open URL** opens `http://localhost:<Port>/lab/tree/<path>` in the
  browser on the machine running Blender, and the add-on does not forward
  ports. When the solver runs on another machine, forward JupyterLab's port
  yourself, for example `ssh -L 8080:localhost:8080 your-host`, and set
  **Port** to the local port you forwarded.
- **A connection and a running server.** Press **Connect** and **Start
  Server on Remote** on the **Backend Communicator** panel. **Transfer** is
  grayed out until the server is running. **Export** is enabled once you
  are connected, and **Delete** once you are connected and have exported a
  notebook; both send their request to the server, so start it first.
  **Start Server on Remote** starts the server the add-on talks to; the
  notebook starts the solver itself when it calls `run()`. See
  [Connections](../../connections/index.md).
- **A Transfer of this project** to the solver host, from this connection
  or an earlier one. Without it, `BlenderApp.open(...)` raises
  `FileNotFoundError`.

## The JupyterLab Box

The **JupyterLab** box sits in the **Solver** panel below the **Export** box.
It starts collapsed; click the triangle in its header to open it. It holds
a row of notebook buttons, a row of build buttons, and a **Port** field.
**Port** is the port **Open URL** uses. It defaults to `8080` and accepts
1024 to 65535.

| Notebook button | What it does |
| --------------- | ------------ |
| **Export**      | Writes the template notebook to the solver host. A dialog asks for the path. |
| **Open URL**    | Opens `http://localhost:<Port>/lab/tree/<path>` in your default browser. |
| **Delete**      | Removes the last exported notebook from the solver host. |

| Build button               | What it does | Enabled when |
| -------------------------- | ------------ | ------------ |
| **Transfer without Build** | Uploads the scene and parameters and stops there. While the status line reads *Ready to Run* or *Resumable*, it first deletes the project's data on the host, cached results and simulated frames included, and reports *Remote data already exists. Deleting first.* In other states it uploads over what is there. Afterward the status line reads *Waiting for Build* and **Run** is grayed out. | The server is running, an active group of a dynamic type holds an object, and nothing is in progress. |
| **Build**                  | Builds the uploaded scene on the host without uploading again. | The server is running, the status line reads *Waiting for Build*, *Ready to Run* or *Resumable*, and nothing is in progress. |

The two build buttons are the halves of what **Transfer** does in one click.
**Transfer** keeps the cached results that still apply; **Transfer without
Build** deletes all of them when it deletes at all. Simulated frames and
saved states on the host are removed when **Transfer without Build**
deletes the data, or otherwise when **Build** runs, so press **Fetch All
Animation** first for any you want to keep.

```{figure} ../../images/jupyterlab/jupyterlab_row.png
:alt: JupyterLab box in the Solver panel showing the notice Connect to server to export/delete, the Export, Open URL and Delete buttons, the Transfer without Build and Build buttons, and the Port field set to 8080, with every button grayed out
:width: 500px

The JupyterLab box before a connection exists: the buttons are grayed out
and a notice reads *Connect to server to export/delete*. The red box marks
**Export**.
```

**Export** needs a connection. Its dialog asks for a notebook path relative
to JupyterLab's root folder, which is the solver's `examples` folder. The
dialog starts from the last exported path, or from
`blender-export/<Project Name>.ipynb` on the first export. The path must
end with `.ipynb`, and missing subfolders are created. A path that leaves
the root folder is refused with *Server rejected notebook: ...*. **Export**
overwrites the file at the path you confirm.

The notebook embeds the **Project Name**, not the file name, so renaming
the file is harmless. Export again after the **Project Name** changes.

**Open URL** is enabled whenever the add-on holds a last-export path, which
is saved with the `.blend`. It does not check that the file still exists on
the host; a file removed outside the add-on cannot be opened.

**Delete** needs a connection and a recorded export. It removes the last
exported notebook from the solver host and clears the add-on's record of it,
which grays out **Open URL** and **Delete** until the next **Export**.

Each button reports its result in Blender's status bar:

| Button | Report |
| ------ | ------ |
| **Export** | Success: *Exported blender-export/shirt.ipynb via solver server* (with the path you confirmed). Errors: *Filepath is empty*, *Filename must end with .ipynb*, *Not connected to solver server. Connect first.*, *Server rejected notebook:* followed by the server's reason (the server answered with an error, for example because the path leaves the root folder), and *Export failed:* followed by the reason (the request did not complete). |
| **Open URL** | *Opened* followed by the URL, for example `http://localhost:8080/lab/tree/blender-export/shirt.ipynb`. |
| **Delete** | Success: *Deleted blender-export/shirt.ipynb* (with the recorded path). Errors: *Not connected to solver server. Connect first.*, *Server rejected delete:* followed by the server's reason, and *Delete failed:* followed by the reason. |

## The Generated Notebook

Three code cells:

1. **Banner.** A comment naming the project.

2. **Attach.**

   ```python
   from frontend import BlenderApp

   app = BlenderApp.open("<project>")
   app.scene.report()
   app.scene.preview()
   ```

   `BlenderApp.open(...)` builds the scene and session from the files that
   **Transfer** uploaded, and clears the project's session folder on the
   solver host first (see the warning below). `app.scene.report()` prints
   element counts, and `app.scene.preview()` shows the scene in 3D.

3. **Run.**

   ```python
   app.session.run()
   app.session.preview()
   app.session.stream()
   ```

   `run()` starts the solver from frame 0, or leaves a running solver
   alone, for example when the cell is run a second time. `preview()` shows
   a live frame player. `stream()` shows the tail of the solver's output and
   draws **Stop Live Stream**, **Terminate Solver**, and **Save and Quit**
   buttons. To continue from a saved state instead of starting over, call
   `app.session.resume()`.

:::{warning}
**Run the attach cell once per kernel, before the Run cell.** It rebuilds
the project's session folder on the solver host from the uploaded scene,
and it clears that folder first. The frames and saved states of an earlier
run live in that folder, so re-running the attach cell discards them.

**Do not attach a notebook to a solve that Blender started with Run.**
Running the attach cell while a solver is running removes the folder the
solver is writing to. Wait for the solve to end, then press **Fetch All
Animation** in Blender to collect that run instead.

To return to a finished or running notebook session, reuse the existing
`app` variable in the same kernel. Before you re-run the attach cell (for
example in a restarted kernel), press **Fetch All Animation** in Blender for
any frames you want to keep. The saved states are cleared as well, so
`app.session.resume()` finds nothing afterward.
:::

:::{note}
The notebook's `BlenderApp` comes from the solver's `frontend` package, not
from the add-on. The add-on only writes the notebook file.
:::

## Simulating Entirely from JupyterLab

Once the notebook is open, Blender is not required. A session that quits
Blender and simulates in the notebook looks like this:

1. In Blender, finish the scene setup and press **Transfer**. Then press
   **Export** in the **JupyterLab** box. The server must be running for
   both buttons (see [Prerequisites](#prerequisites)).
2. Press **Open URL**. In the notebook, run the attach cell once and check
   the scene with `app.scene.report()` and `app.scene.preview()`.
3. Quit Blender. The solver host retains the uploaded scene and any running
   solver; quitting only ends the add-on's connection.
4. In the notebook, run the Run cell from
   [The Generated Notebook](#the-generated-notebook). After a run that saved
   a state and then stopped, call `app.session.resume()` in the same kernel
   instead of `run()` to continue from the latest saved state.
5. To run again from frame 0, call `app.session.run()` once the solver has
   stopped. To change the scene or its parameters, start Blender, connect
   again, update them there, and press **Transfer** or **Update Params on
   Remote** (see [Update Params vs Transfer](simulating.md#update-params-vs-transfer)).
   Then re-run the attach cell. **Transfer** removes the simulated frames
   and saved states on the host, and the attach cell removes them after
   **Update Params on Remote**, so fetch any you want to keep first, as the
   warning above describes.

## Returning to Blender and Fetching

When the run is done, or whenever you want the animation on your Blender
meshes:

1. Start Blender and open the `.blend` you transferred from.
2. Press **Connect** with the same solver host and **Project Name**
   settings. If the server is not running, press **Start Server on Remote**.
   The console may print a `Reconcile: previous session ... differs` line;
   it is informational and does not block **Fetch All Animation**. See
   [Sessions and Recovery](simulating.md#sessions-and-recovery).
3. Press **Fetch All Animation**. It stays grayed out while the solver is
   running. The add-on downloads the frames and attaches them to each
   simulated mesh, as if Blender had driven the run itself.
4. Scrub the timeline to confirm. Then optionally press **Bake Animation**
   (see [Baking Animation](baking.md)) to turn the fetched animation into
   plain Blender data, or write it out with
   [Exporting USD and Alembic Caches](exporting.md).

## Inspecting a Project from a Notebook

The notebook can read the transferred scene and its session while Blender
is closed:

```python
app.scene.report()                       # element counts: vertices, rods, triangles, tets, pins, stitches
app.scene.preview()                      # 3D preview of the scene
app.session.session.param.get("gravity") # read one parameter
help(app.session)                        # the FixedSession API
```

Because `frontend` is a Python package, `inspect.getsource(...)`, the `??`
magic, and `dir()` work on it. The
[JupyterLab Python API](../../../jupyterlab_api/index.rst) section lists
the classes and methods the `frontend` package exports, including
`FixedScene` and `FixedSession`, which `app.scene` and `app.session`
return. For a run held at a frame,
`app.session.update_params()` re-exports the `dyn()` schedules; the same
page describes it under stepping a run frame by frame.

## Tips

- **Transfer again after scene changes.** The attach cell reads whatever
  **Transfer** last uploaded. Press **Transfer** after you add or remove
  objects or rebuild groups, and **Update Params on Remote** after a
  parameter-only change. Re-export the notebook only when the **Project
  Name** changes.
- **Delete to tidy up.** It removes the last exported notebook from the
  host, so `blender-export/` does not fill with files from old experiments.
- **If Open URL goes to the wrong port**, change the **Port** field at the
  bottom of the **JupyterLab** box. **Open URL** always uses that value.

## See Also

- [Connections](../../connections/index.md): connecting, and **Start Server
  on Remote**, which **Export** and **Delete** need.
- [Running the Simulation](simulating.md): covers **Transfer**, **Run**, and
  **Fetch All Animation**.
- [Baking Animation](baking.md): converting a fetched animation into
  standard Blender keyframes.
- [Exporting USD and Alembic Caches](exporting.md): writing the fetched
  animation to a point cache file.
- [JupyterLab Python API](../../../jupyterlab_api/index.rst): the classes
  and methods the `frontend` package exports, including those behind
  `app.scene`, `app.session`, and `app.session.session.param`.

:::{admonition} Under the hood
:class: toggle

**How it fits together**

```{figure} ../../images/jupyterlab/how_it_fits.svg
:alt: Sequence diagram with three lanes. Blender plus add-on on the left, solver host in the middle, browser on the right. The add-on transfers mesh and parameters and exports the notebook to the solver host, opens the JupyterLab URL in the browser, and the browser attaches back to the solver host via BlenderApp.open
:width: 760px

Transport topology. The add-on sends the scene, parameters, and notebook
file through the solver connection, then opens the JupyterLab URL in the
browser. Once loaded, the notebook attaches back to the solver host through
`BlenderApp.open(...)`. **Open URL** launches the browser on the machine
running Blender at `http://localhost:<Port>/lab/tree/<path>` (the host and
port drawn in the figure are placeholders), so a remote host needs the
port forward described under Prerequisites.
```

The add-on never calls JupyterLab's REST API. The JupyterLab process lives
on the solver host.

**What the notebook reads**

`BlenderApp.open(...)` reads `data.pickle`, `param.pickle`, and
`upload_id.txt` under the project root on the solver host, which
**Transfer** writes. It raises `FileNotFoundError` when any of them is
missing. It then builds the scene and session, which clears the project's
`session` folder first, including its `output` subfolder with the frames and
saved states.

**Transport**

The add-on writes and deletes the notebook through the solver's control
channel as JSON requests. The server resolves the path under the solver's
`examples` folder, which is JupyterLab's root, creates missing subfolders,
and refuses a path that escapes that folder.
:::
