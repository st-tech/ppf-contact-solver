# 🍰 Baking Animation

After a successful **Fetch All Animation** the solver's output drives each
simulated object through a `ContactSolverCache` modifier (meshes) or
per-CV keyframes written directly on the curve (rods). That is perfect
while you iterate on a shot, but it keeps every object tied to the
add-on: the animation depends on the `.pc2` files on disk, the modifier
on the mesh, and the group membership in the sidebar.

**Baking** converts that fetched animation into standard Blender data:
per-frame shape keys for meshes, per-control-point keyframes for curves.
After a bake, the `ContactSolverCache` modifier is dropped, the object leaves its
dynamics group, and the animation is self-contained Blender data that
plays back in any scene, with or without the add-on installed.

There are two flavors:

- **Bake Animation**. Keyframes **every** fetched frame onto the object.
  Use this when you are done iterating and want to hand the scene off.
- **Bake Single Frame**. Captures the **current** frame's pose only and
  freezes the object there as Frame 1. Use this to turn a cloth-draping
  step into the rest pose of the next simulation, or to freeze a pin
  operation's final shape.

```{figure} ../../images/baking/before_after_bake.svg
:alt: Two-panel before-and-after diagram. Before (left): the mesh carries a ContactSolverCache MESH_CACHE modifier with Format PC2 pointing at a per-object PC2 file under the blend's data folder, has pin vertex groups (ShoulderPins, CollarPins), and is assigned to a Dynamics Group (Cloth, Shell); it has no shape keys and no value fcurves. A central red arrow labeled Bake Animation (destructive, irreversible) points from the before panel to the after panel. After (right): the same mesh with the ContactSolverCache modifier removed (slot freed), the PC2 file on disk deleted, the pin vertex groups removed, the Dynamics Group assignment cleared, and the animation now stored as one shape key per fetched frame (ContactSolverBake_00001, _00002, and so on) with value fcurves using CONSTANT interpolation that drive each shape key 0 to 1 to 0 across its active frame. A footer explains that rods bake to per-control-point keyframes on co / handle_left / handle_right instead of shape keys, and that Abort during a modal bake restores the full pre-bake state.
:width: 900px

What the object looks like before and after **Bake Animation**.
Everything on the left that is orange (modifier), mauve (pin vertex
groups), or yellow (group membership) is the add-on's; baking
removes those and the PC2 file on disk that backs the modifier. What
remains on the right (green) is the object's own Blender data:
per-frame shape keys plus the fcurves that drive them. The red arrow
only goes one way.
```

:::{note}
Baking requires simulation data to exist for the object. Run the
**Transfer → Run → Fetch All Animation** loop (see [Running the
Simulation](simulating.md)) first; the bake buttons are grayed out until
there is fetched animation to bake from.
:::

## Baking Is Destructive: Duplicate the .blend First

Once a bake finalizes, it cannot be undone from the UI. For every baked
object the add-on:

- removes the `ContactSolverCache` modifier from the mesh (or drops the
  per-CV animation the solver wrote onto the curve for rods),
- **deletes the `.pc2` file** from `<blend_dir>/data/<basename>/` on
  disk, and
- removes the object from its dynamics group, together with any pin
  vertex groups attached to it.

**Recommended workflow**: always bake in a *copy* of your `.blend`, so
the pre-bake scene (with its PC2 files, modifiers, and group setup)
stays intact as a fallback.

1. **Save** the current `.blend` (`File → Save`).
2. **Duplicate the `.blend` on disk**, outside Blender. Run `cp
   project.blend project_baked.blend` in a terminal, or copy the file
   in Finder / Explorer. Do *not* use `File → Save As` from the
   currently-open session; the PC2 files need to stay paired with the
   original `.blend`.
3. **Open the duplicate**. The Solver panel detects that the
   `ContactSolverCache` modifiers still point at
   `<blend_dir>/data/<old_basename>/` while the new `.blend`'s canonical
   data folder is `<blend_dir>/data/<new_basename>/`, and surfaces a
   migration prompt:

   ```{figure} ../../images/baking/migrate_pc2_prompt.png
   :alt: Solver panel showing the PC2 migration prompt after opening a duplicated .blend
   :width: 500px

   The Solver panel in a freshly-opened duplicate. The error line
   (*Data path: …/data/project_baked/…pc2 does not exist.*) flags the
   missing canonical PC2 data, the **Migrate data/project/ →
   data/project_baked/** button migrates in one click, and the
   **Missing files (N)** list underneath enumerates every PC2 that
   would need to be moved.
   ```

4. **Click Migrate**. A dialog appears with *From* / *To* folders and a
   **Keep Copy** checkbox (checked by default). Leave **Keep Copy** on;
   the operator then `shutil.copytree`s `data/<old>/` →
   `data/<new>/` and rewrites every `ContactSolverCache` modifier
   filepath to point at the new folder. The original `data/<old>/`
   stays untouched, so the source `.blend` keeps working.
5. **Save** the duplicate once more so the rewritten modifier filepaths
   are persisted.
6. **Bake** in the duplicate. The PC2 files that get deleted belong to
   `data/<new_basename>/`; the original `.blend`'s `data/<old_basename>/`
   is preserved.

:::{tip}
Unchecking **Keep Copy** switches the migration from `shutil.copytree`
to `os.rename`, saving disk space at the cost of leaving the original
`.blend` pointing at a folder that no longer exists. Only turn it off
when you are certain you want to abandon the source `.blend`.
:::

:::{warning}
The migration button only appears when the mismatch is unambiguous:
every `ContactSolverCache` modifier points into a single
`data/<old>/` folder, and that folder is still on disk. Mixed prefixes
or a missing source folder return *no* migration target, and the
Solver panel shows only the missing-file warning. In that case the
cleanest recovery is to manually copy `data/<old>/` →
`data/<new>/` with the file manager / `cp -r`, then reopen the
`.blend`.
:::

## The Solver Panel (Bake All Objects)

The **Solver** panel exposes a row that bakes every dynamic object in the
scene in one go. The buttons sit directly under **Fetch All Animation**
and **Delete Remote Data**:

```{figure} ../../images/baking/solver_bake_row.png
:alt: Solver panel showing the Bake Animation and Bake Single Frame row
:width: 500px

The scene-wide bake row on the **Solver** panel. **Bake Animation** (left)
writes every fetched frame onto every simulated object as shape-key or
curve keyframes; **Bake Single Frame** (right) freezes the current frame
onto every simulated object as their new Frame 1.
```

```{figure} ../../images/baking/solver_bake_animation.png
:alt: Bake Animation button highlighted on the Solver panel
:width: 500px

**Bake Animation** runs as a modal job with a progress bar. The panel
shows a live `Baking <object> [N/total]` status line and an **Abort**
button while frames are keyframed. Abort restores every modifier / shape
key / fcurve to its pre-bake state.
```

```{figure} ../../images/baking/solver_bake_single_frame.png
:alt: Bake Single Frame button highlighted on the Solver panel
:width: 500px

**Bake Single Frame** is synchronous and finishes in one click. Every
dynamic object is frozen at the viewport's current evaluated pose,
applied to frame 1, and removed from its group.
```

Both scene-wide buttons are enabled only when at least one active group
has an assigned object carrying simulation animation (that is, only
after a successful **Fetch**). While a bake job is running every other button on
the Solver panel is inert; wait for the progress bar to finish or press
**Abort**.

## The Dynamics Groups Panel (Bake One Object)

Each group box has its own **Bake Animation** / **Bake Single Frame**
row, sitting below the **Add Selected Objects** / **Remove Object** row.
These bake only the object currently selected in the group's list, not
the whole group or the whole scene:

```{figure} ../../images/baking/dynamics_bake_row.png
:alt: Dynamics Groups panel showing per-group Bake Animation and Bake Single Frame buttons
:width: 500px

The per-group bake row on each **Dynamics Groups** box. The two buttons
are enabled when the row highlighted in the assigned-object list has
fetched animation. The baked object leaves the group (so it won't be
re-simulated on the next **Transfer**) but keeps its keyframes.
```

Per-object **Bake Animation** uses the same modal progress + Abort flow
as the scene-wide version, so the panel stays interactive for the other
groups while one object bakes.

## What Ends Up on the Object

The output format depends on the object type:

| Object type | What gets keyed                                              | Modifier removed       |
| ----------- | ------------------------------------------------------------ | ---------------------- |
| **Mesh**    | One shape key per fetched frame, named `ContactSolverBake_NNNNN`, each driven 0 to 1 to 0 with `CONSTANT` interpolation so exactly one key is active per frame. | `ContactSolverCache`   |
| **Curve**   | Per-control-point keyframes on `co` / `handle_left` / `handle_right` (bezier) or on `co` (poly / NURBS). | `ContactSolverCache` (curves keep the baked rest pose) |

After baking, the object:

- No longer appears in its **Dynamics Groups** slot; the assignment is
  cleared, together with any pin vertex groups attached to it.
- Still plays back its simulated motion in Blender's timeline, because
  the animation now lives in shape keys / fcurves on the object itself.
- Can be moved, duplicated, appended to other `.blend` files, or
  rendered on a render farm without shipping the `.pc2` file.

:::{warning}
For meshes, **Bake Animation** adds one shape key per frame. The add-on
refuses to bake a mesh that already carries non-Basis shape keys, because
mixing baked keys with user-authored ones would double-blend. Remove
extras manually (shape-keys panel, then trash icon) before baking.
:::

## Aborting a Bake

**Bake Animation** runs modally. While the bake is in progress the
Solver panel shows:

- A status line (`Baking <object> [N/total]`) updating as frames are
  keyed.
- A progress bar (`% (done/total)`).
- An **Abort** button (red "X" icon).

Pressing **Abort** reverts every partial change: shape keys inserted so
far are removed, their value fcurves are stripped, bezier handle types
are restored, and the `ContactSolverCache` modifier + PC2 file stay
intact, so you can immediately press **Bake Animation** again or go
back to iterating.

**Bake Single Frame** does not run modally and cannot be aborted; it
finishes in one operator call.

## Bake Order and Static Groups

**Bake Animation** walks through active groups in slot order
(`object_group_0` → `object_group_31`) and bakes every assigned object.
**Static** groups are included so their `ContactSolverCache` modifier
and PC2 file get cleaned up too, leaving every bakeable group empty
once the scene-wide bake finishes. See
[Static Objects → Baking behavior](../scene/static_objects.md#baking-behavior)
for the full story on what bake does to a Static object depending on
whether it was driven by Blender keyframes or by Static ops.

Groups that hold curves alongside meshes bake both: meshes emit shape
keys, curves emit per-CV keyframes, and both end up on standard Blender
timelines.

## Blender Python API

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

For LLM / MCP-driven flows, the same actions are exposed as the
`scene.bake_all_animation`, `scene.bake_all_single_frame`,
`group.bake_group_animation`, and `group.bake_group_single_frame`
handlers (see [MCP Integration](../../integrations/mcp.md)).

:::{admonition} Under the hood
:class: toggle

**Operator names**

| Button                                       | `bl_idname`                       |
| -------------------------------------------- | --------------------------------- |
| Solver panel → **Bake Animation**            | `solver.bake_all_animation`       |
| Solver panel → **Bake Single Frame**         | `solver.bake_all_single_frame`    |
| Dynamics Groups → **Bake Animation**         | `object.bake_animation`           |
| Dynamics Groups → **Bake Single Frame**      | `object.bake_single_frame`        |
| In-progress bake → **Abort**                 | `solver.bake_abort`               |

**Shape key naming**

Baked mesh shape keys are named `ContactSolverBake_<frame:05d>`. The
value channel for each key is driven with three keyframes (`0` at
`frame-1`, `1` at `frame`, `0` at `frame+1`), and every keyframe is
forced to `CONSTANT` interpolation so frame `N` plays exactly one shape
key, with no blending.

**Abort safety**

The modal operator snapshots enough state before it starts that an
abort is non-destructive:

- For meshes: the pre-existing set of shape-key names and whether the
  shape-key block existed at all. Abort removes only the keys the bake
  added, and deletes the shape-key block if the bake created it.
- For curves: the pre-existing set of `(data_path, array_index)`
  fcurve keys and the per-spline `(handle_left_type, handle_right_type)`
  snapshot. Abort removes fcurves that didn't exist before and restores
  bezier handle types that were forced to `FREE` during the bake.

The `ContactSolverCache` modifier and the `.pc2` file on disk are never
touched by an abort, so the pre-bake state is fully recoverable.
:::
