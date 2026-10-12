# 📤 Exporting USD and Alembic Caches

**Export USD** and **Export Alembic (ABC)** write the simulated deformation
to a point cache: one file that holds where every vertex sits on every
frame, readable by Blender and by other applications that open USD or
Alembic. The `.blend` is left as it is, so you can export at any point in a
shot and again whenever the result changes.

This page covers:

- [Where it lives](#where-it-lives)
- [Exporting or baking](#exporting-or-baking): which of the two you want
- [Fetch every frame first](#fetch-every-frame-first): the preconditions,
  and the messages that stop an export
- [What gets exported](#what-gets-exported): including what is left out
- [The frame range](#the-frame-range)
- [File types](#file-types)
- [What the export touches in your scene](#what-the-export-touches-in-your-scene)
- [Python / MCP API](#python--mcp-api)

## Where It Lives

Open the sidebar (<kbd>N</kbd>) in the 3D viewport, switch to the add-on
tab, and find the **Export** box in the **Solver** panel, below the
**Deformations** box and above **JupyterLab**. It holds two buttons side by
side: **Export USD** and **Export Alembic (ABC)**.

Both are grayed out until the view layer holds at least one mesh carrying a
`ContactSolverCache` modifier. They also gray out while a bake, a **Capture
Deformation** on a collider or a pin, a solve, or a server build is running.

Pressing one opens Blender's file browser. The preconditions below are
checked *before* the browser opens, so the export reports a refusal before
you pick a filename.

## Exporting or Baking

Both turn a fetched simulation into something that plays back without the
solver. They differ in where the result goes and in what happens to the
scene:

|                          | **Export USD / Alembic** | **[Bake Animation](baking.md)** |
| ------------------------ | ------------------------ | ------------------------------- |
| **What you get**         | One point-cache file at the path you pick, readable by other applications. | Blender data on the object: one shape key per frame for a mesh, per-control-point keyframes for a curve. |
| **The scene after**      | Untouched. The `ContactSolverCache` modifier, the simulation cache file, the group membership, the pins and the mesh's vertex groups all stay. | The modifier is removed, the simulation cache file is deleted, and the object leaves its group together with the pins registered on it. The vertex groups on the mesh stay. Deforming modifiers above the cache (Armature, Lattice), as on an animated collider, are deleted too. |
| **Repeatable**           | Yes. Run it again after the next **Fetch All Animation**. | No. A finalized bake cannot be undone from the UI. |
| **Rods (curves)**        | Not carried. See the warning under [What gets exported](#what-gets-exported). | Baked to per-control-point keyframes. |
| **Hair Curves**          | Not carried, as rods are not.                                                 | Not baked: Blender keeps no keyframes for hair points. The hair stays on its solver cache. |
| **What it adds to the `.blend`** | Nothing. | One shape key per frame on every baked mesh, each with keyframes on its **Value**. |

Bake when the result has to live inside a self-contained `.blend`: handing
the file to someone who does not have the add-on, or rendering on a farm
without shipping the simulation cache files. Export when the result is going
to another application, or when you want to keep iterating on the same scene
and hand off a file each time.

## Fetch Every Frame First

:::{important}
**The export reads the same cache the viewport plays back**, so every frame
has to be on disk before it can be written. Run **Fetch All Animation** to
completion first (see [Running the Simulation](simulating.md)).
:::

When an export cannot go ahead, it stops and puts the reason in the status
bar. The first four messages are preconditions, checked before the file
browser opens. The buttons are already grayed out while another solver
activity runs, so *Another solver activity is in progress* appears only when
the export is called from [Python or MCP](#python--mcp-api). The last two
are reported by the add-on after it has asked Blender's exporter to write the
file, and the reason after *Export failed:* is Blender's own.

| Message | What it means, and what to do |
| ------- | ----------------------------- |
| *Unfetched animation frames exist. Fetch all animation frames first.* | The run produced frames that are still on the server. Press **Fetch All Animation** and let it finish. |
| *Another solver activity is in progress* | A bake, a **Capture Deformation** on a collider or a pin, a solve, or a server build is still running. Wait for it to finish, press **Abort** on a bake or capture, or stop the solve with **Terminate** or **Save and Quit** on the **Backend Communicator** panel. |
| *Exit Edit/Sculpt mode before exporting* | Return to Object Mode. |
| *No simulated mesh sequence to export* | No mesh in the view layer carries a `ContactSolverCache` modifier whose cache file exists and holds at least one frame. Run the **Transfer → Run → Fetch All Animation** loop first. |
| *Export failed: ...* | Blender's USD or Alembic exporter reported the error shown after the colon. The scene is restored either way. |
| *Export wrote nothing* | Blender's exporter ended without finishing the export. |

## What Gets Exported

Every **mesh** in the view layer that carries a live `ContactSolverCache`
modifier and a cache file with at least one frame. Your selection does not
decide this: the add-on selects those objects itself for the duration of
the export and puts your selection back afterward.

The export samples each mesh as the viewport shows it, so any other
modifier that is shown in the viewport affects what is written. Turn off a
modifier's viewport display if you do not want it in the file.

:::{warning}
**Rods are not carried.** A **Rod** group's curves are animated by the
add-on rather than by a modifier on the object, and Blender's USD and
Alembic exporters do not reliably pick that up, so every simulated curve is
skipped. The add-on names them in a warning once the export finishes: *"N
rod/curve object(s) not exported (unsupported by this cache export):
names"*. To keep a rod's motion, use [**Bake Animation**](baking.md), which
keys the curve's control points inside the `.blend`.

A Hair Curves object in a **Rod** group is skipped and named the same way.
It cannot be baked either, so its motion stays on the add-on's cache (see
[Hair Curves](../scene/hair_curves.md#what-hair-does-not-do)).
:::

Two other things are not carried:

- **Objects hidden in the viewport.** The exporter samples the viewport
  deformation, so an object that is not visible produces nothing. Hiding on
  the object itself (its eye icon or its **Disable in Viewports** toggle) is
  overridden for you: the add-on shows its targets for the export and
  restores their flags afterward. Hiding that comes from the collection (the
  collection's **Disable in Viewports** toggle or its eye icon) cannot be
  overridden that way, so those meshes are skipped and named in a warning:
  *"N object(s) hidden in the viewport were not exported: names"*. A
  collection excluded from the view layer holds no view-layer objects, so
  its meshes are left out without a warning. The success line counts only
  the meshes that were written, so compare it against what you expected.
- **Materials.** These files carry deforming geometry. The USD export is
  written with material export off, and shading stays in the `.blend`.

## The Frame Range

The export writes one continuous range. It starts on the frame the solve
starts on (**Starting Frame** on the **Scene Configuration** panel, or the
scene's start frame while **Take Starting Frame from Scene** is on) and runs
for as many frames as the longest cache in the scene. The success line names
the range it wrote: *"Exported 3 mesh(es), frames 1-180 to
/path/to/sim.usdc"*.

When the caches are not all the same length, the shorter ones hold their
final pose for the rest of the range, and the success line ends with
*(shorter caches hold their final pose past their length)*. To give every
mesh the same length, **Run** and **Fetch All Animation** again.

## File Types

| Button | Suffix | What is written |
| ------ | ------ | --------------- |
| **Export USD** | `.usdc` (default), `.usda`, `.usd`, `.usdz` | The suffix picks the USD flavor: `.usdc` is the binary crate, `.usda` is ASCII text, `.usdz` is a zipped package, and `.usd` is the generic USD suffix. In the file browser, a name with no suffix gets `.usdc` and any other suffix is replaced with `.usdc`. A path passed from a script is written exactly as given, so pass one of the four. |
| **Export Alembic (ABC)** | `.abc` | An Alembic cache. |

## What the Export Touches in Your Scene

For the duration of the export the add-on takes over the selection, the
active object, the exported meshes' visibility flags and their cache
modifier's viewport toggle, and the scene's frame range, frame step, and
current frame. All of them are restored when the export finishes, whether
it succeeded or failed.

## Python / MCP API

From Python inside Blender, the add-on's `solver` API (see
[Blender Python API](../../integrations/python_api.md) and the
[Blender Python API Reference](../../integrations/python_api_reference.rst))
has both exporters and a way to list the rods that will be left out:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

missing = solver.get_unexportable_curves()   # names of the rods the export skips
solver.export_usd("/path/to/sim.usdc")
solver.export_alembic("/path/to/sim.abc")
```

The import on the first line is the direct form for an add-on in the **User
Default** repository, where **Install from Disk** and the install script put
it. For an add-on installed from a remote repository, import `solver` as
shown under [Import](../../integrations/python_api.md#import).

`export_usd` and `export_alembic` skip the file browser, write straight to
the path (a leading `//` is resolved relative to the `.blend`), and return
`solver` so calls chain. They raise `ValueError` when a precondition from
the table above fails, and also when the destination folder does not exist.

For LLM / MCP-driven flows the same two actions are the `export_usd` and
`export_alembic` handlers (see [MCP Server](../../integrations/mcp.md)).
They answer a failed precondition with an error that carries its own
message instead of a bare cancel, require the destination folder to exist,
and list the skipped curve objects in an `excluded_curves` field, so a
skipped rod is visible to the caller rather than only to whoever reads the
status bar. The `get_fetch_status` handler reports `export_ready`, whether
the preconditions above hold, and `export_blocked_reason`, the message of the
one that does not.

:::{admonition} Under the hood
:class: toggle

**Operator names**

| Button                                  | `bl_idname`             |
| --------------------------------------- | ----------------------- |
| Export box → **Export USD**             | `solver.export_usd`     |
| Export box → **Export Alembic (ABC)**   | `solver.export_alembic` |

Both operators also run directly from a script:

```python
import bpy

bpy.ops.solver.export_usd(filepath="/path/to/sim.usdc")
bpy.ops.solver.export_alembic(filepath="/path/to/sim.abc")
```

Called this way they skip the file browser and write straight to
`filepath`. A failed precondition is reported as an error, which `bpy.ops`
raises as `RuntimeError` carrying the message from the table above. While a
solver activity is running, or when no mesh carries the `ContactSolverCache`
modifier, the operator's poll fails first and the `RuntimeError` says only
that the context is incorrect. Wrap the call in `try` / `except`.

**How the frames are sampled**

Both drive Blender's own exporter (`wm.usd_export`, `wm.alembic_export`)
across the frame range with `evaluation_mode='VIEWPORT'` and
selected-objects-only, so what lands in the file is the `ContactSolverCache`
deformation exactly as the viewport shows it while you scrub. That is also
why the cache modifier's viewport display toggle is forced on for the
targets: with it off the exporter samples the undeformed mesh.

The cache files behind the modifiers are the `.pc2` files that **Fetch All
Animation** writes. A Rod group's curves play back through a frame-change
handler instead of a modifier, which is why the exporters skip them.

The Alembic export runs in the foreground rather than as a background job,
so the state restore above happens after the file is complete rather than
alongside a job still writing to it.

USD stores sparse time samples, so a stretch of identical poses (a collider
that stops moving, or a shorter cache holding its final frame) is stored
once and covers the rest of the range through the stage's start and end
time codes.
:::
