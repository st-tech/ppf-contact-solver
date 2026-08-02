# 📤 Exporting USD and Alembic Caches

**Export USD** and **Export Alembic (ABC)** write the simulated deformation
to a point cache: one file holding where every vertex sits on every frame,
in a format Blender, other DCC applications, and renderers read directly.
The `.blend` is left exactly as it is, so an export can be run at any point
in a shot and repeated as the result changes.

This page covers:

- [Where it lives](#where-it-lives)
- [Exporting or baking](#exporting-or-baking): which of the two you want
- [Fetch every frame first](#fetch-every-frame-first): the precondition,
  and every message the export refuses with
- [What gets exported](#what-gets-exported): including what is left out
- [The frame range](#the-frame-range)
- [File types](#file-types)
- [What the export touches in your scene](#what-the-export-touches-in-your-scene)
- [Python / MCP API](#python--mcp-api)

## Where It Lives

Open the sidebar (<kbd>N</kbd>) in the 3D viewport, switch to the add-on
tab, and scroll to the bottom of the **Solver** panel. Below the
**Deformations** box and above **JupyterLab** is an **Export** box holding
two buttons side by side: **Export USD** and **Export Alembic (ABC)**.

Both are grayed out until the view layer holds at least one mesh carrying a
`ContactSolverCache` modifier, and they gray out again while a bake, a
**Capture Deformation**, a pin capture, or a solve is in flight.

Pressing one opens Blender's file browser. The preconditions below are
checked *before* the browser opens, so a refusal reaches you before you
pick a filename rather than after.

## Exporting or Baking

Both turn a fetched simulation into something that plays back without the
solver. They differ in where the result goes and in what happens to the
scene:

|                        | **Export USD / Alembic**                                                                                    | **[Bake Animation](baking.md)**                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **What you get**       | One point-cache file at the path you pick, readable by other applications.                                      | Blender data on the object: one shape key per frame for a mesh, per-control-point keyframes for a curve.     |
| **The scene after**    | Untouched. The modifier, the `.pc2` file, group membership, and pin vertex groups all stay.                     | The modifier is removed, the `.pc2` file is deleted, the object leaves its group, and its pin groups go too. |
| **Repeatable**         | Yes. Run it again after the next **Fetch All Animation**.                                                       | No. A finalized bake cannot be undone from the UI.                                                           |
| **Rods (curves)**      | Not carried. See the warning [below](#what-gets-exported).                                                      | Baked to per-control-point keyframes.                                                                        |
| **Timeline weight**    | One file, played back through a cache modifier.                                                                 | One shape key plus its value fcurve per frame, on every baked mesh.                                          |

Bake when the result has to live inside a self-contained `.blend`: handing
the file to someone who does not have the add-on, or rendering on a farm
without shipping the `.pc2` files. Export when the result is going to
another application, or when you want to keep iterating on the same scene
and hand off a file each time.

## Fetch Every Frame First

:::{important}
**The export reads the same cache the viewport plays back**, so every frame
has to be on disk before it can be written. Run **Fetch All Animation** to
completion first (see [Running the Simulation](simulating.md)).
:::

When a precondition is not met, the export stops and puts the reason in the
status bar:

| Message                                                              | What it means, and what to do                                                                                                              |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| *Unfetched animation frames exist. Fetch all animation frames first.* | The run produced frames that are still on the server. Press **Fetch All Animation** and let it finish.                                        |
| *Another solver activity is in progress*                             | A bake, a **Capture Deformation**, a pin capture, or the solve itself is still running. Wait for it to finish, or press its **Abort**.        |
| *Exit Edit/Sculpt mode before exporting*                             | Return to Object Mode.                                                                                                                       |
| *No simulated mesh sequence to export*                               | No mesh in the view layer has a readable cache with at least one frame in it. Run the **Transfer → Run → Fetch** loop first.                  |

## What Gets Exported

Every **mesh** in the view layer that carries a live `ContactSolverCache`
modifier and a cache file with at least one frame. Your selection does not
decide this: the add-on selects those objects itself for the duration of
the export and puts your selection back afterward.

Three things are not carried:

:::{warning}
**Rods are not carried.** A **Rod** group's curve objects are moved by a
frame-change handler rather than by a modifier, and Blender's own USD and
Alembic exporters do not fire it, so every simulated curve is skipped. The
add-on names them in a warning once the export finishes: *"N rod/curve
object(s) not exported (unsupported by this cache export)"*. To get a rod
out, use [**Bake Animation**](baking.md), which writes per-control-point
keyframes onto the curve itself.
:::

**Objects hidden in the viewport.** The exporter samples the viewport
deformation, so an object that is not visible produces nothing. Object-level
visibility is handled for you: the add-on unhides its targets for the
export and restores their flags afterward. Visibility that comes from the
collection (**Disable in Viewports** on the collection, or an excluded
collection) cannot be overridden that way, so those meshes are skipped and
named in a warning: *"N object(s) hidden in the viewport were not
exported"*. The success line counts only the meshes that were written, so
compare it against what you expected.

**Materials.** These files carry deforming geometry. The USD export is
written with material export off; shading stays in the `.blend`.

## The Frame Range

The export writes one continuous range. It starts on the frame the solve
starts on (**Starting Frame** on the **Scene Configuration** panel, or the
scene's start frame while **Take Starting Frame from Scene** is on) and
runs for as many frames as the longest cache in the scene. The success line
names the range it wrote: *"Exported 3 mesh(es), frames 1-180 to
/path/to/sim.usdc"*.

When the caches are not all the same length, the shorter ones hold their
final pose for the rest of the range, and the report line says so. That
clause is worth reading: it means at least one mesh stops moving partway
through the exported file while the others carry on, which is a mismatch to
resolve with a fresh **Run** and **Fetch All Animation** rather than in the
receiving application.

## File Types

| Button                       | Suffix                              | What is written                                                            |
| ---------------------------- | ----------------------------------- | -------------------------------------------------------------------------- |
| **Export USD**               | `.usdc` (default), `.usda`, `.usd`, `.usdz` | Kept as typed; the suffix is what tells Blender's USD exporter which flavor to write. The file browser fills in `.usdc` when you leave the suffix off, and a path passed from a script is written exactly as given, so pass one of the four. |
| **Export Alembic (ABC)**     | `.abc`                              | Alembic.                                                                    |

## What the Export Touches in Your Scene

For the duration of the export the add-on takes over the selection, the
active object, the exported meshes' visibility flags and their cache
modifier's viewport toggle, and the scene's frame range, frame step, and
current frame. All of them are restored when the export finishes, whether
it succeeded or failed.

Nothing on the objects themselves changes: the `ContactSolverCache`
modifier stays, the `.pc2` file on disk stays, and group membership and pin
vertex groups stay. The scene is ready for another **Transfer → Run →
Fetch** round, and for another export after it.

## Python / MCP API

Both exporters are registered operators:

```python
import bpy

bpy.ops.solver.export_usd(filepath="/path/to/sim.usdc")
bpy.ops.solver.export_alembic(filepath="/path/to/sim.abc")
```

Called this way they skip the file browser and write straight to
`filepath`. A failed precondition returns `{'CANCELLED'}` with the message
from the table above in the report.

For LLM / MCP-driven flows the same two actions are the `export_usd` and
`export_alembic` handlers (see [MCP Integration](../../integrations/mcp.md)).
They raise with the precondition's own message instead of returning a bare
cancel, require the destination directory to exist already, and list the
skipped curve objects in an `excluded_curves` field, so a skipped rod is
visible to the caller rather than only to whoever reads the status bar.

:::{admonition} Under the hood
:class: toggle

**Operator names**

| Button                                  | `bl_idname`             |
| --------------------------------------- | ----------------------- |
| Export box → **Export USD**             | `solver.export_usd`     |
| Export box → **Export Alembic (ABC)**   | `solver.export_alembic` |

**How the frames are sampled**

Both drive Blender's own exporter (`wm.usd_export`, `wm.alembic_export`)
across the frame range with `evaluation_mode='VIEWPORT'` and
selected-objects-only, so what lands in the file is the
`ContactSolverCache` deformation exactly as the viewport shows it while you
scrub. That is also why the cache modifier's viewport display toggle is
forced on for the targets: with it off the exporter samples the undeformed
mesh.

The Alembic export runs in the foreground rather than as a background job,
so the state restore above happens after the file is complete rather than
alongside a job still writing to it.

USD stores sparse time samples, so a stretch of identical poses (a collider
that stops moving, or a shorter cache holding its final frame) is stored
once and covers the rest of the range through the stage's start and end
time codes.
:::
