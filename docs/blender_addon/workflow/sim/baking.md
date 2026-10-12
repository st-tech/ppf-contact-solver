# 🍰 Baking Animation

Baking turns the animation fetched from the solver into ordinary Blender
animation, so the scene does not depend on the add-on.

After **Fetch All Animation**, the solver's output animates each
simulated mesh through a `ContactSolverCache` modifier, and each rod
through a playback cache the add-on applies to the curve's control
points. That suits iterating on a shot, but every object then depends on
the add-on: the animation needs its cache files on disk, the modifier on
the mesh and the group membership in the sidebar.

**Baking** converts that animation into standard Blender data: one shape
key per frame for meshes, keyframes on every control point for curves.
After a bake, the playback modifier or cache is removed, the object leaves
its dynamics group, and the animation plays back in any scene, with or
without the add-on installed.

There are two flavors:

- **Bake Animation**. Keys **every** fetched frame onto the object, placed
  at the solve's **Starting Frame** plus the frame's index (frame 1 for
  the first frame by default). Use it when you are done iterating and want
  to hand the scene off.
- **Bake Single Frame**. Writes the pose the object has at the current
  frame into its geometry as the object's new shape, and sets the playhead
  to the **Starting Frame**. The dialog and the report call this *Frame 1*,
  which is the Starting Frame's default. Use it to turn a cloth-draping
  step into the rest pose of the next simulation, or to freeze a pin
  operation's final shape.

Baking is not the only way to get a result out of the add-on. When the
destination is another application, or when you want to keep iterating on
this scene, [export a USD or Alembic cache](exporting.md) instead: it
writes the animation to a file and leaves every object, modifier and group
exactly where it is.

:::{note}
Baking needs fetched animation. Run the **Transfer** → **Run** → **Fetch
All Animation** loop first (see [Running the Simulation](simulating.md));
the bake buttons are grayed out until the scene holds fetched animation.
:::

## Baking Is Destructive: Duplicate the .blend First

A finished bake cannot be reversed: **Edit → Undo** does not bring back the
deleted cache file. For every baked object the add-on:

- removes the `ContactSolverCache` modifier from a mesh, or the playback
  cache from a rod;
- **deletes every modifier above `ContactSolverCache`** in a mesh's stack
  (Armature, Lattice, MeshDeform and so on), because those fed the
  captured-deformation cache and would re-deform the baked shape keys at
  playback. Nothing is deleted for the usual dynamic object, whose cache
  sits in the first slot with nothing above it;
- **deletes the object's cache file** from `<blend_dir>/data/<basename>/`;
- removes the object from its dynamics group, together with the pins
  registered on it, its entries in the group's **Allow Intersections**
  object lists and every merge pair that names it. The vertex groups on
  the mesh stay. The object's viewport color is reset to white.

Every bake button opens a confirmation dialog before any of that runs. Its
button reads **Bake** for one object and **Bake All** on the **Solver**
panel. When modifiers will be deleted, the dialog lists them by name under
*These deforming modifiers will also be removed:*.

:::{note}
**A captured deformation is not part of that cleanup.** If the object
carried a **Capture Deformation** recording, as a deforming Static collider
or an animated pin does, that recording stays on disk, and the per-object
**Clear Deformation Cache** button cannot reach it, because the bake has
taken the object out of its group. **Clear All Deformations**, in the
**Deformations** box on the **Solver** panel, deletes every captured
recording in the scene, including the ones left behind this way. The
**Re-capture All Deformations** button in the same box re-records every
deforming Static collider and every animated pin in one pass, which brings
a scene up to date before the next **Transfer**. See
[Static Objects](../scene/static_objects.md).
:::

Always bake in a *copy* of your `.blend`, so the pre-bake scene (with its
cache files, modifiers and group setup) stays intact as a fallback:

1. **Save** the current `.blend` (`File → Save`).
2. **Duplicate the `.blend` on disk**, outside Blender. Run `cp
   project.blend project_baked.blend` in a terminal, or copy the file in
   Finder or Explorer. Do *not* use `File → Save As` from the open
   session; the cache files need to stay paired with the original
   `.blend`.
3. **Open the duplicate**. Its `ContactSolverCache` modifiers still point
   at `<blend_dir>/data/<old_basename>/`, while the duplicate's own data
   folder is `<blend_dir>/data/<new_basename>/`. The **Solver** panel
   reports the missing files and shows a migration button:

   ```{figure} ../../images/baking/migrate_pc2_prompt.png
   :alt: Solver panel showing the missing-data notice, the Migrate data/project/ to data/project_baked/ button and the Missing files list after opening a duplicated .blend
   :width: 500px

   The **Solver** panel in a freshly opened duplicate. The error line
   (*Data path: ... does not exist.*) flags the missing data folder. The
   **Migrate data/project/ → data/project_baked/** button migrates it in
   one click, and the **Missing files (N)** list underneath names every
   file that would be moved.
   ```

4. **Press the Migrate button**. A **Migrate PC2 Cache** dialog shows the
   *From* and *To* folders and a **Keep Copy** checkbox, on by default.
   Leave **Keep Copy** on: the add-on copies `data/<old>/` to
   `data/<new>/` and rewrites the **File Path** of every
   `ContactSolverCache` modifier to the new folder. The original
   `data/<old>/` stays untouched, so the source `.blend` keeps working.
5. **Save** the duplicate again so the rewritten paths are stored.
6. **Bake** in the duplicate. The cache files that get deleted belong to
   `data/<new_basename>/`. The original `.blend`'s `data/<old_basename>/`
   is preserved.

:::{tip}
Turning **Keep Copy** off moves the folder instead of copying it. That
saves disk space but leaves the original `.blend` pointing at a folder
that has been moved. Turn it off only when you are sure you will
abandon the source `.blend`.
:::

:::{warning}
The migration button appears only when the mismatch is unambiguous: every
`ContactSolverCache` modifier points into one `data/<old>/` folder inside
the `.blend`'s `data` folder, and that folder is still on disk. With mixed
folders or a missing source folder, the **Solver** panel shows only the
missing-file notice. Then copy `data/<old>/` to `data/<new>/` with the
file manager (or `cp -r`) and reopen the `.blend`.
:::

## The Solver Panel (Bake All Objects)

The **Solver** panel has a row that bakes every simulated object in one
go. The buttons sit directly under **Fetch All Animation** and **Delete
Remote Data**:

```{figure} ../../images/baking/solver_bake_row.png
:alt: Solver panel showing the Bake Animation and Bake Single Frame row
:width: 500px

The scene-wide bake row on the **Solver** panel. **Bake Animation** (left)
writes every fetched frame onto every simulated object. **Bake Single
Frame** (right) writes the current pose into every simulated object.
```

```{figure} ../../images/baking/solver_bake_animation.png
:alt: Bake Animation button highlighted on the Solver panel
:width: 500px

**Bake Animation** on the **Solver** panel. It opens the confirmation
dialog, and **Bake All** starts the bake.
```

While the bake runs, Blender stays responsive and the **Solver** panel
shows a status line (`Baking <object> [N/total]`), a progress bar
(`% (done/total)`) and an **Abort** button. See
[Aborting a Bake](#aborting-a-bake).

```{figure} ../../images/baking/solver_bake_single_frame.png
:alt: Bake Single Frame button highlighted on the Solver panel
:width: 500px

**Bake Single Frame** on the **Solver** panel. It opens the same
confirmation dialog, then applies the pose to every simulated object in
one step and removes each from its group.
```

Both scene-wide buttons are enabled only when the scene holds fetched
animation. **Bake Animation** also refuses to start while frames are still
unfetched (*Unfetched animation frames exist. Fetch all animation frames
first.*): press **Fetch All Animation** first. **Bake Single Frame** has no
such check.

While a bake runs, **Run**, **Resume**, **Fetch All Animation**, **Clear
Local Animation**, the bake buttons themselves and everything in the
**Deformations** and **Export** boxes are grayed out. **Transfer**,
**Update Params on Remote** and **Delete Remote Data** are not grayed out
during a bake, but do not press any of the three until the progress bar
finishes, or press **Abort** first. The **JupyterLab** and **MCP Server**
boxes stay available.

## The Dynamics Groups Panel (Bake One Object)

Each group box in **Dynamics Groups** has its own **Bake Animation** and
**Bake Single Frame** row, below the **Add Selected Objects** and **Remove
Object** row. These bake only the object selected in the group's list, not
the whole group or the whole scene:

```{figure} ../../images/baking/dynamics_bake_row.png
:alt: Dynamics Groups panel showing per-group Bake Animation and Bake Single Frame buttons
:width: 500px

The per-group bake row on a **Dynamics Groups** box. The two buttons are
enabled when the selected object in the assigned-object list has fetched
animation and no bake is running. The baked object leaves the group, so
the next **Transfer** does not simulate it again, but it keeps its
keyframes.
```

Per-object **Bake Animation** shows the same status line, progress bar and
**Abort** button on the **Solver** panel as the scene-wide bake, and it
refuses to start while frames are unfetched.

## What Ends Up on the Object

The result depends on the object type:

| Object type | What gets keyed | Removed |
| ----------- | --------------- | ------- |
| **Mesh** | One shape key per fetched frame, named `ContactSolverBake_NNNNN` after its Blender frame number. Each key's **Value** is keyed 0 → 1 → 0 with constant interpolation, so exactly one key is active on each frame. | The `ContactSolverCache` modifier and the object's cache file. |
| **Curve** (rod) | Keyframes on every control point: the position and both handles for Bezier points, the position for poly and NURBS points. Bezier handles are set to Free. | The playback cache and the object's cache file. The curve keeps the baked pose. |
| **Hair Curves** | Nothing. Blender keeps no keyframes for hair points, so hair cannot be baked. | Nothing. The hair keeps its `ContactSolverCache` modifier and its cache file, and stays in its group. |

After baking, the object:

- Is not in its **Dynamics Groups** list.
- Plays back its simulated motion in the timeline, because the animation
  is stored in shape keys and animation curves on the object itself.
- Can be moved, duplicated, appended to other `.blend` files, or rendered
  on a render farm without shipping the cache file.

:::{warning}
For meshes, **Bake Animation** adds one shape key per frame. It refuses to
bake a mesh that already has shape keys other than **Basis** (*Remove all
shape keys except Basis before baking*), because mixing baked keys with
authored ones would double-blend. In **Object Data Properties**, remove
every shape key except **Basis**, then bake.
:::

## Aborting a Bake

**Bake Animation** runs in the background with a progress bar. While it
runs, the **Solver** panel shows:

- A status line (`Baking <object> [N/total]`) that updates as frames are
  keyed.
- A progress bar (`% (done/total)`).
- An **Abort** button with an X icon.

**Abort** reverts everything the bake has written so far. The shape keys
it added and their value curves are removed (the shape-key block too, if
the bake created it), the keyframes it added to curves are removed, and
Bezier handle types are restored. The modifier (or playback cache), the
cache file and the group membership are removed only after every frame
is keyed, so after an abort they are intact. You can press **Bake
Animation** again or go back to iterating.

**Bake Single Frame** finishes in one step and cannot be aborted. Like
every bake button, it opens a confirmation dialog first.

## Status Bar Reports

Each bake button ends with a line in Blender's status bar:

| Action | Report |
| ------ | ------ |
| **Bake Animation** (one object or all) | *Baked N frame(s) for M object(s)*. *Bake aborted* after **Abort**. |
| **Bake Single Frame** (one object) | *Baked frame F as Frame 1 for 'Cloth'*. F is the **Starting Frame**, where the playhead ends up; the report says *Frame 1* whatever the **Starting Frame** is. *No animation data on 'Cloth'* when the object has no fetched animation, and *No object selected* or *Object 'Cloth' not found* when the group's list has no usable selection. |
| **Bake Single Frame** (all) | *Baked single frame for N object(s)*. *UUID resolve failed for:* followed by the object names when an object cannot be matched to the add-on's record of it. |
| Either **Bake Animation** | *No bakeable objects* when the bake finds no mesh or curve to queue: the selected object is gone or is neither a mesh nor a curve, or no active group holds one. *Another bake job is in progress* when a bake is already running. |

The per-group buttons are grayed out unless the selected object holds fetched
animation, and every bake button is grayed out while a bake runs. So
*No bakeable objects*, *No animation data on 'Cloth'*, *No object selected*
and *Object 'Cloth' not found* come only from a script that calls a per-group
operator directly, and *Another bake job is in progress* is a safeguard behind
the grayed-out buttons.

## Bake Order and Static Groups

**Bake Animation** on the **Solver** panel bakes the active groups one
after another, and every assigned object in each. **Static** groups are
included, so their `ContactSolverCache` modifier and cache file are
cleaned up too, and every bakeable group is empty once the scene-wide bake
finishes. See
[Static Objects → Baking behavior](../scene/static_objects.md#baking-behavior)
for what a bake does to a Static object, depending on whether Blender
keyframes or Static operations drove it.

Groups that hold curves alongside meshes bake both: meshes get shape keys,
curves get control-point keyframes, and both end up on standard Blender
timelines.

A Hair Curves object is left out of every bake. A bake of everything bakes
the other objects and then warns *N Hair Curves object(s) were not baked and
keep their solver cache:* followed by their names. **Bake Animation** or
**Bake Single Frame** asked of one hair object stops with *'Fur' is a Hair
Curves object and cannot be baked: Blender keeps no keyframes for hair
points. Its animation stays on the solver cache.* See
[Hair Curves](../scene/hair_curves.md#what-hair-does-not-do).

## Blender Python API

Baking is exposed through the add-on's registered operators:

```python
import bpy

# Scene-wide: bake every dynamic object.
bpy.ops.solver.bake_all_animation()          # runs in the background; returns at once
bpy.ops.solver.bake_all_single_frame()       # synchronous

# One object: the one selected in the group in slot group_index (0 to 31).
bpy.ops.object.bake_animation(group_index=0)
bpy.ops.object.bake_single_frame(group_index=0)

# Abort the running bake (no-op if none is running).
bpy.ops.solver.bake_abort()
```

For LLM and MCP-driven flows, the same actions are the
`bake_all_animation`, `bake_all_single_frame`, `bake_group_animation` and
`bake_group_single_frame` handlers, and `abort_bake` stops a running bake
(see [MCP Server](../../integrations/mcp.md)). An MCP tool carries the
bare handler-function name. The `scene.` and `group.` prefixes are only
the handler module files and are not part of the name a client calls.

The confirmation dialog and the unfetched-frames check belong to the
buttons. A script or an MCP call runs neither and bakes whatever has been
fetched, so fetch all frames before baking from a script. The check for
shape keys other than **Basis** still applies.

:::{admonition} Under the hood
:class: toggle

**Operator names**

| Button | `bl_idname` |
| ------ | ----------- |
| Solver panel → **Bake Animation** | `solver.bake_all_animation` |
| Solver panel → **Bake Single Frame** | `solver.bake_all_single_frame` |
| Dynamics Groups → **Bake Animation** | `object.bake_animation` |
| Dynamics Groups → **Bake Single Frame** | `object.bake_single_frame` |
| In-progress bake → **Abort** | `solver.bake_abort` |

**Bake order**

The scene-wide bake walks the group slots `object_group_0` to
`object_group_31` in order and queues every assigned mesh and curve of each
active group.

**Shape key naming**

Baked mesh shape keys are named `ContactSolverBake_<frame:05d>`, where
`frame` is the Blender frame the key is active on. The value channel of
each key has three keyframes (`0` at `frame-1`, `1` at `frame`, `0` at
`frame+1`), and every keyframe uses `CONSTANT` interpolation, so frame `N`
plays exactly one shape key with no blending.

**Abort safety**

The bake job records enough state before it starts that an abort is
non-destructive:

- For meshes: the pre-existing set of shape-key names and whether the
  shape-key block existed at all. Abort removes only the keys the bake
  added, and deletes the shape-key block if the bake created it.
- For curves: the pre-existing set of `(data_path, array_index)` curve
  channels and the per-spline `(handle_left_type, handle_right_type)`
  snapshot. Abort removes channels that did not exist before and restores
  Bezier handle types that were set to `FREE` during the bake.

The cleanup of the modifier, the `.pc2` file and the group membership runs
only when keyframing completes, so an abort never reaches it.
:::
