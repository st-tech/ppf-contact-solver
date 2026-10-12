# 🐍 Blender Python API

A script in Blender's scripting editor can create groups, pin vertex
groups, keyframe spins, add invisible colliders, snap meshes and more. Use
it for procedural scene setup, batch variants and regression tests, and for
anything you would otherwise click through many times.

:::{tip}
This page is a walkthrough. The
[Blender Python API Reference](python_api_reference.rst) lists every method
and is generated from the source.
:::

## Import

The add-on must be installed and enabled. This import finds it under
whichever extension repository it was installed from:

```python
import addon_utils, importlib

name = next(m.__name__ for m in addon_utils.modules()
            if m.__name__.endswith(".ppf_contact_solver"))
solver = importlib.import_module(f"{name}.ops.api").solver
```

An add-on installed from a remote repository (Option A of
[Install the Add-on](../getting_started/install.md#install-the-add-on)) loads
under a different module name, which is why the code above looks the name
up. An add-on in the **User Default** repository, where **Install from
Disk** and the install script put it, has a fixed name, and a direct import
works:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver
```

Every example below assumes `solver` is imported. The snippets continue from
the ones before them: `cloth` is the group created under [Groups](#groups),
and `pin` is a pin created from it.

## Scene Parameters

`solver.param` reads and writes the add-on's settings by their Python names.
It reaches the **Scene Configuration** panel, the **Force Fields** box, the
**Project Name** of the **Backend Communicator**, the **Port** fields of the
**MCP Server** and **JupyterLab** boxes, and the connection settings of the
**Backend Communicator** (see [Connection Settings](#connection-settings)).
Scene settings look like this:

```python
solver.param.project_name = "shirt_drape"
solver.param.frame_count  = 180
solver.param.frame_rate   = 60             # the FPS field
solver.param.step_size    = 0.001
solver.param.gravity      = (0, 0, -9.8)   # alias for gravity_3d
solver.param.air_density  = 0.001225
```

- `gravity` is an alias for `gravity_3d`. Reads and writes work with either
  name.
- `frame_rate` keeps its value while **Take FPS from Scene**
  (`use_scene_fps`) is on, and the simulation then runs at the scene's
  frame rate.
- A name that no scene or connection property has raises `AttributeError`
  on a read and `RuntimeError` on a write.

See [Scene Parameters](../workflow/params/scene.md) for every setting with
its default and range.

### Dynamic Parameters

Gravity, wind, air density, air friction, vertex air damping and step size
can change during the run. Keyframe the property on the scene state to put
the animation on the timeline and in the Graph Editor:

```python
import bpy

state = bpy.context.scene.zozo_contact_solver.state

# Flip gravity at frame 60: key the old value, then the new one a frame later.
state.gravity_3d = (0, 0, -9.8)
state.keyframe_insert(data_path="gravity_3d", frame=60)
state.gravity_3d = (0, 0, 9.8)
state.keyframe_insert(data_path="gravity_3d", frame=61)
```

The data path is the property name `gravity_3d`, not the `solver.param`
alias `gravity`. See [Dynamic Parameters](../workflow/params/dynamic.md) for
the supported properties, **Constant** interpolation and what **Transfer**
refuses.

`solver.param.dyn()` is a builder for the same kind of schedule. It records
its keyframes in a list that **Transfer** reads, and it takes **frames**,
where the [JupyterLab Python API](../../jupyterlab_api/index.rst)'s
`session.param.dyn()` takes seconds.

```python
# Flip gravity at frame 60: hold the initial value through 60, then switch at 61.
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Start a wind gust at frame 30.
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Scalars such as air_density take a float.
solver.param.dyn("air_density").time(100).change(0.005)

# Remove the list entry for gravity.
solver.param.dyn("gravity").clear()
```

The valid keys are `"gravity"`, `"wind"`, `"air_density"`, `"air_friction"`
and `"vertex_air_damp"`. The cursor starts at the initial keyframe at frame
1, so each `time()` must be later than the one before it: `time(1)` and
`time(30).time(30)` raise. `clear()` empties only that list. Once a file has
been reloaded and its list converted to animation curves, it removes
nothing; delete the animation curve in the Graph Editor instead. See
[Dynamic Parameters](../workflow/params/dynamic.md) for how `hold()` and
`change()` differ.

## Connection Settings

`solver.param` also reaches the properties of the **Backend Communicator**,
so a script can set up a connection and open it without touching the panel.
`solver.connect()` starts the connection with the values these hold:

```python
import os

solver.param.server_type        = "LINUX_NATIVE"
solver.param.linux_native_path  = os.path.expanduser("~/path/to/solver")
solver.param.native_device      = "CPU"            # or "GPU"
solver.param.native_gpu_backend = "AUTO"           # or "CUDA", "ROCM"
solver.param.docker_port        = 9090
solver.connect()
```

`server_type` takes the identifiers in the table, not the labels the panel
shows:

| Value                | Panel label                  |
| -------------------- | ---------------------------- |
| `CUSTOM`             | **SSH**                      |
| `COMMAND`            | **SSH Command**              |
| `DOCKER`             | **Docker**                   |
| `DOCKER_SSH`         | **Docker over SSH**          |
| `DOCKER_SSH_COMMAND` | **Docker over SSH Command**  |
| `WIN_NATIVE`         | **Windows Native**           |
| `MAC_NATIVE`         | **macOS Native**             |
| `LINUX_NATIVE`       | **Linux Native**             |

- The native path properties are `win_native_path`, `mac_native_path` and
  `linux_native_path`. The add-on uses the path as written, apart from
  resolving a leading `//` against the `.blend`, and does not expand a
  `~`, so give an absolute path, as `os.path.expanduser` does above.
- `native_device` and `native_gpu_backend` name which **build** of the
  solver runs. Every connection type reads `native_device` when the server
  starts, and every type except **macOS Native** also reads
  `native_gpu_backend`. A native connection also checks at **Connect** that
  the folder holds that build. See {ref}`Choosing the build <choosing-the-build>`.
- `docker_port` is the server port for every connection type despite its
  name. The panel draws **Docker Port** for the Docker types only.
- `solver.connect()` starts the connection and returns at once. The
  connection completes on Blender's event loop, so a script that keeps
  running delays it. A refusal raises `RuntimeError` naming the reason: a
  connection that is already open or still connecting, a required field left
  empty (**Solver Path** on a native type), or a project name that holds a
  character other than letters, digits, `.`, `-` and `_`. An empty project
  name is replaced by `default`, not refused.
- Connecting does not start the server. That is a separate step,
  `solver.start_remote_server()`, the call behind **Start Server on
  Remote**, and it needs an open connection. A script that transfers right
  after `connect()` fails because the connection is not open yet. The Python
  API has no call that reports whether the connection or the server is up,
  so a script cannot wait for either. Run the next step in a separate run
  once the panel shows the connection, or poll `get_connection_info` (its
  `connection_status.connected`) and `get_remote_status` through
  [MCP](./mcp.md).

MCP has a connect tool for SSH (`connect_ssh`, which takes an optional
container for **Docker over SSH**), one for Docker (`connect_docker`) and one
for each native type (`connect_win_native`, `connect_mac_native` and
`connect_linux_native`). Each sets these properties and then connects. The
**SSH Command** and **Docker over SSH Command** types are set up in the panel
or with `solver.param`.

## Groups

```python
# type is one of: SOLID | SHELL | ROD | STATIC | PDRD | SAND
cloth = solver.create_group("Cloth", type="SHELL")
cloth.add("Shirt", "Pants")
cloth.param.shell_density       = 0.5
cloth.param.shell_young_modulus = 50.0
cloth.param.friction            = 0.3
cloth.param.bend                = 0.5
cloth.set_overlay_color(0.0, 0.75, 0.0, 0.75)         # rgba in [0, 1]

body = solver.create_group("Body", type="STATIC")
body.add("Mannequin")
```

`create_group` returns a group proxy. Look one up later by UUID with
`solver.get_group(uuid)`, which raises `KeyError` when no group has that
UUID, or walk every active group:

```python
for g in solver.get_groups():
    print(g.uuid, g.param.friction)
```

### Group Surface

| Member                                                       | Purpose                                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `add(*object_names)`                                         | Add mesh objects by name, and curve and Hair Curves objects to a **Rod** group |
| `remove(object_name)`                                        | Remove one object                                                        |
| `set_overlay_color(r, g, b, a=1.0)`                          | Set and enable the viewport overlay color                                |
| `set_velocity(object_name, direction, speed, frame=1, ...)`  | Add a velocity keyframe to an assigned object                            |
| `set_hinge(object_name, pca_axis=2, enable=True)`            | Hinge a **PDRD** body on one principal axis                              |
| `create_pin(object_name, vertex_group_name, indices=None, weights=None)` | Pin a vertex group, the control points `indices` of a curve, or the points `indices` of a Hair Curves object, with `weights` for a weighted hair pin; returns a pin proxy |
| `get_pins()`                                                 | List every pin in this group as pin proxies                              |
| `delete()`                                                   | Delete this group and its pins; returns `None`                           |
| `.param.<name>`                                              | Material and contact parameters                                          |
| `.uuid`                                                      | UUID string, stable across renames                                       |
| `.name`, `.type`, `.slot`                                    | The group's name, its type, and its slot number in the scene, which can differ from its position in the panel |

`add` skips a name that does not exist, an object of an unsupported type and
an object that already belongs to a group. It raises nothing, so check the
group in the panel when a script depends on the assignment.

Material parameters on `.param` are validated: assigning a name outside the
whitelist raises `AttributeError`. See
[Material Parameters](../workflow/params/material.md) for the full list.

`set_velocity` keys the **Velocity Overwrite** list of an object in the
group. A keyframe at frame 1 is the initial velocity, and later frames build
a schedule. `direction` is normalized and `speed` is in m/s. `angular_speed`
is in degrees per second, with `angular_axis` as `"PC1"`, `"PC2"`, `"PC3"`,
`"X"`, `"Y"`, `"Z"` or `"CUSTOM"`. It raises `ValueError` when the object is
not in the group or a keyframe already exists at that frame. See
[Velocity Overwrite](../workflow/params/material.md#velocity-overwrite).

The full signature is `set_velocity(object_name, direction, speed, frame=1,
angular_axis="PC3", angular_speed=0.0, angular_axis_custom=(0, 0, 1),
enable_translational=True, enable_angular=None)`. `angular_axis_custom` is the
world axis used when `angular_axis` is `"CUSTOM"`. `angular_axis` also takes
`0`, `1` and `2` for `PC1`, `PC2` and `PC3`. `enable_translational=False`
leaves the translation alone, for a pure spin keyframe. `enable_angular`
defaults to true when `angular_speed` is not zero and to false otherwise.

```python
ball = solver.create_group("Ball", type="SOLID")
ball.add("Sphere")
ball.set_velocity("Sphere", direction=(1, 0, 0), speed=2.3)                # initial velocity
ball.set_velocity("Sphere", direction=(0, 0, 1), speed=1.0, frame=30)      # a push at frame 30
```

`set_hinge` needs a **PDRD** group. See
[Hinge Joints](../workflow/params/material.md#hinge-joints).

### Rods

A **Rod** group simulates curve objects. `solver.create_curve` builds a Bezier
curve, and `create_pin` takes control-point indices for it:

```python
points = [(0.1 * i, 0.0, 0.0) for i in range(8)]   # eight control points

curve = solver.create_curve("Strands", bevel_depth=3e-3)
curve.add_spline(points)                      # (n, 3) coordinates, n >= 2
obj = curve.finalize()                        # links the curve object into the scene

rod = solver.create_group("Rod", type="ROD")
rod.add(obj.name)
rod.create_pin(obj.name, "left", indices=[0, 7])   # pin control points 0 and 7
```

`create_curve` removes an existing object of the same name first, unless you
pass `clear_existing=False`. It also takes `bevel_resolution` (the
cross-section subdivisions of the viewport tube, default 2), `resolution_u`
(the spline interpolation resolution, default 4) and `dimensions` (`"3D"`, the
default, or `"2D"`). `curve.name` is the object name that `finalize()`
creates. `add_spline(points, *, closed=False)` takes
`closed` as a keyword argument and returns the spline's index, which
`set_material(index, material)` takes to color that spline. A second `finalize()` on the same builder raises `RuntimeError`.
`indices` is for curves and Hair Curves only: a mesh pin uses an existing
vertex group, and passing `indices` for a mesh raises `ValueError`. See
[Pins on a Curve Rod](../workflow/constraints/pins.md#pins-on-a-curve-rod).

A **Rod** group also takes a Hair Curves object, the object Blender's hair
tools create. On hair, `create_pin(name, "Roots")` pins the first two points
of every curve with no indices, `indices` writes a point attribute and pins
it, and `weights`, one per index and each above 0 and at most 1, makes the
pin a weighted one whose pull fades along the strand:

```python
fur = solver.create_group("Fur", type="ROD")
fur.add("Fur")
hold = fur.create_pin("Fur", "Hold", indices=[2, 3, 4, 5],
                      weights=[0.8, 0.6, 0.4, 0.2])
hold.pull(strength=300.0)
```

`weights` raises `ValueError` for any object that is not Hair Curves, without
`indices`, for a weight of zero and for a list whose length differs from that
of `indices`. See [Hair Curves](../workflow/scene/hair_curves.md).

## Reset

```python
solver.delete_all_groups()   # every active group and its pins
solver.clear()               # the add-on's state in this file
```

`solver.clear()` resets the file's add-on state, so call it at the top of any
script that must start from default settings. It:

- deletes every group and its pins, and empties the merge pairs, the
  invisible colliders, the dynamic-parameter list and the fetched frames;
- resets every scene setting to its default, the force field settings
  included, and also the **Project Name** (to `unnamed`), the **Port** fields
  of the **MCP Server** and **JupyterLab** boxes and the **Console Log
  Export** path. The connection settings (type, host, user, key, paths,
  container, **Compute Device** and **GPU Backend**) are kept. If the MCP
  server runs on a substituted port when you call `solver.clear()` or the
  `clear_solver` tool, the **MCP Server (Running :port)** header then shows
  the default port, not the port the server listens on;
- removes the **ContactSolverCache** Mesh Cache modifier and its cache file
  from every mesh object;
- deletes the animation data stored on the mesh data of every mesh object in
  the file.

:::{warning}
The last step also affects meshes that are in no group. A keyframe stored on
mesh data, such as a vertex keyframe, is removed from every mesh object in
the file, whether or not it belongs to a group.
:::

`solver.clear()` leaves the animation curves on the scene settings, such as a
keyframed **Gravity**, in place. Delete them in the Graph Editor.

## Pins and Operations

`group.create_pin(object_name, vertex_group_name)` returns a pin proxy.
Every method except `delete()` returns `self`, so calls chain:

```python
pin = cloth.create_pin("Shirt", "ShoulderPins")
pin.spin(axis=(0, 0, 1), angular_velocity=360, frame_start=1, frame_end=60)
pin.unpin(frame=90)

# A soft pin instead of a hard constraint.
cloth.create_pin("Shirt", "HemPins").pull(strength=2.0)

# Chain a scale and a spin on one pin.
(cloth.create_pin("Shirt", "CuffPins")
      .scale(factor=0.5, center_direction=(0, 0, -1), frame_start=1, frame_end=60)
      .spin(axis=(0, 1, 0), angular_velocity=180, center_vertex=42))
```

`create_pin` on a vertex group that already has a pin adds nothing and
returns a proxy for the existing pin.

Each call to `move_by`, `spin`, `scale` or `torque` puts its operation at the
top of the pin's **Operations** list, as the panel's **Add** does, and the
list order is the order in which the solver applies the operations. The last
call in a chain is therefore applied first: in the chain above the spin acts
before the scale. Call the methods in the order opposite to the one you want,
or reorder the list in the panel. See
[Pins and Operations](../workflow/constraints/pins.md).

### Pin Surface

| Method                                                                  | Purpose                                                                   |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `pull(strength=1.0)`                                                    | Switch to a soft pull force                                               |
| `move_by(delta, frame_start=1, frame_end=60, transition="LINEAR")`      | Ramp a translation over a frame range                                     |
| `spin(axis, angular_velocity, flip, center, center_mode, center_direction, center_vertex, frame_start=1, frame_end=60, transition)` | Rotate about a pivot, as in [Center-Mode Inference](#center-mode-inference-for-spin-and-scale) |
| `scale(factor, center, center_mode, center_direction, center_vertex, frame_start=1, frame_end=60, transition)` | Scale from a pivot, as in [Center-Mode Inference](#center-mode-inference-for-spin-and-scale) |
| `torque(magnitude, axis_component="PC3", flip, frame_start=1, frame_end=60)` | Torque about a principal axis of the pinned vertices                 |
| `set_animation(positions)`                                              | Drive the pin through per-frame vertex positions, as **Capture Deformation** does |
| `unpin(frame)`                                                          | Release the pin after `frame` frames, counted from the solve's **Starting Frame** (the argument is a frame count, not a frame number) |
| `delete()`                                                              | Remove this pin from its group; returns `None`                            |

`transition` is `"LINEAR"` or `"SMOOTH"` and shapes `move_by` and `scale`.
`spin` accepts it and ignores it, because a spin turns at a constant angular
velocity. `torque`'s `axis_component` is `"PC1"`, `"PC2"` or `"PC3"`, the
panel's 1st, 2nd and 3rd component.

A pin proxy also has two read-only properties, `object_name` and
`vertex_group_name` (for a curve pin, the pin's name), which identify the
proxies that `group.get_pins()` returns:
`[(p.object_name, p.vertex_group_name) for p in cloth.get_pins()]`.

`frame_start` and `frame_end` are Blender frames, kept as given. They default
to 1 and 60 whatever the **Starting Frame** is, so pass both when the
**Starting Frame** is not 1. **Transfer** refuses an operation whose
`frame_end` is not after its `frame_start`, or whose `frame_start` is before
the **Starting Frame**.

`torque` cannot share a pin with `move_by`, `spin` or `scale`, and it needs at
least three pinned vertices. The panel refuses the combination when you add
the operation. A script is refused only at **Transfer**, which stops with
*torque cannot be mixed with Move/Spin/Scale operations*.

A proxy keeps working only while its group and pin exist. After
`solver.clear()`, deleting the group, or removing the pin, every pin method
in the table raises `ValueError` saying whether the group or the pin is
gone, rather than returning `self` as if the change had been applied. A
group proxy does the same when you read or write `.param.<name>`, `.name`,
`.type` or `.slot`.

### `move_by` over a Frame Range

`move_by` ramps a translation of the pinned vertices by `delta` from
`frame_start` to `frame_end`, with a `"LINEAR"` or `"SMOOTH"` transition.
Pair it with `unpin` to release the pin once the move completes.

```python
pin = cloth.create_pin("Shirt", "SleevePins")
pin.move_by(delta=(0, 0, 0.5), frame_start=10, frame_end=30, transition="SMOOTH")
pin.unpin(frame=40)   # release after the move finishes
```

### Center-Mode Inference for `spin` and `scale`

Pass whichever argument names your pivot, and the API picks the matching
mode:

| Argument you pass    | Inferred `center_mode`        | Panel label of **Center** |
| -------------------- | ----------------------------- | ------------------------- |
| `center=(x, y, z)`   | `ABSOLUTE` (a world position) | **Fixed**                 |
| `center_direction=v` | `MAX_TOWARDS`                 | **Max Towards**           |
| `center_vertex=idx`  | `VERTEX`                      | **Vertex**                |
| none of the above    | `CENTROID`                    | **Centroid**              |

When you pass more than one, `center_vertex` wins over `center_direction`,
and `center_direction` wins over `center`. Passing `center_mode="..."`
explicitly overrides the inference. See
[Pins and Operations](../workflow/constraints/pins.md) for what each mode
computes.

```python
pin.spin(axis=(0, 0, 1), angular_velocity=360)                        # CENTROID
pin.spin(axis=(0, 0, 1), angular_velocity=360, center=(0, 0, 1))      # ABSOLUTE
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_direction=(0, 0, -1))  # MAX_TOWARDS
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_vertex=42)      # VERTEX
```

### `set_animation` from Per-Frame Positions

`set_animation(positions)` writes the same per-pin cache **Capture
Deformation** writes, straight from an array, so no deformer is needed and
it works in a `--background` Blender. `positions` is
`(n_frames, n_verts, 3)`: every vertex of the pin's mesh on each frame, in
the mesh's own coordinates, with at least two frames. Row `k` is the pose `k`
frames after the solve's **Starting Frame**, so row 0 is the mesh in its
current pose. The pinned vertices follow the sequence and the rest of the mesh is
simulated.

```python
import numpy as np

pin = cloth.create_pin("Shirt", "BandPins")
pin.set_animation(np.stack(frames))   # frames: list of (n_verts, 3) arrays
```

It raises `ValueError` when the pin's object is not a mesh, its group is
**Static**, the pin holds no vertex, the pin carries a torque or manual
vertex keyframes, or `positions` has the wrong shape, too few frames, or a
non-finite value.

## Snap and Merge

```python
solver.snap("Shirt", "Mannequin")                 # move Shirt next to Mannequin, register the pair

solver.add_merge_pair("SleeveLeft", "BodyLeft")   # register a pair without moving anything
solver.remove_merge_pair("Shirt", "Mannequin")
solver.get_merge_pairs()                          # returns [("Shirt", "Mannequin"), ...]
solver.clear_merge_pairs()
```

`solver.snap(a, b)` moves `a` next to `b` and leaves a small contact gap at
the closest pair of points. A pair whose closest points already coincide is
not moved and gets no gap. When `a` is in a **Static** group, `b` moves
instead. The call also records a stitch for every
vertex within reach and registers the merge pair, so it is the one call that
prepares a pair for **Transfer**. Both objects must already be in groups that
form a supported pair. Snap reads each object's shape at the current frame,
with its modifiers applied, while **Transfer** reads it at the **Starting
Frame**, so put the timeline on the **Starting Frame** first, for example
with `bpy.context.scene.frame_set(...)`. See
[Gap Rules](../workflow/constraints/snap_merge.md#gap-rules) for the gap.

`add_merge_pair` registers the pair and nothing else. A pair with no stitch
points is refused at **Transfer**, so call `solver.snap(a, b)` instead, or
after it. See
[Merge Pairs Without Snapping](../workflow/constraints/snap_merge.md#merge-pairs-without-snapping).

All of these share the validation layer of the MCP interface and the UI, so
the errors are the same. A name that does not resolve raises `ValueError`.

## Invisible Colliders

Walls and spheres return a chainable builder. Parameters on `.param` cover
`friction`, `contact_gap`, `thickness`, `enable_active_duration` and
`active_duration`.

```python
# A ground plane with extra friction.
solver.add_wall(position=(0, 0, 0), normal=(0, 0, 1)).param.friction = 0.5

# An inverted hemispherical container (keeps the cloth inside a bowl).
(solver.add_sphere(position=(0, 0, 0), radius=0.98)
       .invert()
       .hemisphere())

# A sphere that shrinks at frame 61.
(solver.add_sphere(position=(0, 0, 0), radius=1.0)
       .time(60).hold()
       .time(61).radius(0.5))

# A wall that slides to a new position.
(solver.add_wall(position=(0, 0, 0), normal=(0, 1, 0))
       .time(60).hold()
       .time(61).move_to((0, 1, 0)))

solver.get_invisible_colliders()   # returns [("WALL", "Wall 1"), ("SPHERE", "Sphere 1"), ...]
solver.clear_invisible_colliders()
```

A collider added from Python starts with its **Preview** eye off, where the
panel's **Add** turns it on.

### Builder Surface

| Method             | Wall | Sphere | Purpose                                              |
| ------------------ | :--: | :----: | ---------------------------------------------------- |
| `.time(frame)`     |  yes |  yes   | Advance the keyframe cursor (must be increasing)     |
| `.hold()`          |  yes |  yes   | Hold the previous value at the cursor                |
| `.move_to(pos)`    |  yes |  yes   | Keyframe a new position                              |
| `.move_by(delta)`  |  yes |        | Keyframe a position offset from the previous         |
| `.radius(r)`       |      |  yes   | Keyframe a new radius                                |
| `.transform_to(p, r)` |   |  yes   | Keyframe position and radius together                |
| `.invert()`        |      |  yes   | Flip inside-out (contact on the inside)              |
| `.hemisphere()`    |      |  yes   | Treat as a hemisphere                                |
| `.param.*`         |  yes |  yes   | The five names below                                 |
| `.delete()`        |  yes |  yes   | Remove this collider; returns `None`                 |

The keyframe cursor starts at the initial keyframe at frame 1, so `time(1)`
raises. Every later keyframe must come after the **Starting Frame**, or
**Transfer** refuses it.

| `.param` name            | Panel label              | Meaning                                                                         |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------------- |
| `friction`               | **Friction**             | Friction coefficient, 0 to 1                                                    |
| `contact_gap`            | **Contact Gap**          | The barrier gap kept between the collider and dynamic geometry                  |
| `thickness`              | **Thickness**            | Maximum penetration depth. Geometry that sinks deeper passes through the collider |
| `enable_active_duration` | **Active Duration**      | `True` to limit how long the collider acts                                      |
| `active_duration`        | **Active Until (frame)** | The first Blender frame at which the collider stops acting: a frame number, not a frame count |

See [Invisible Colliders](../workflow/constraints/colliders.md) for how the
keyframe timeline is evaluated.

## Force Fields

The **Force Fields** box is reached through the same `solver.param` proxy.
The **Collection** and the **Script** text can be given as datablocks or by
name, and `None` clears them.

```python
import bpy

solver.param.force_field_visualize    = True
solver.param.force_field_preview_resolution = (8, 8, 8)
# solver.param.force_field_collection = bpy.data.collections["Fields"]   # take fields from one collection only

text = bpy.data.texts.new("swirl.py")
text.from_string(
    "def eval(x, y, z, t):\n"
    "    return curl_noise(x, y, z, octaves=2, seed=1, time=t, frequency=0.5)\n"
)
solver.param.force_field_script = text

# Every function and constant a script may use, as the Built-in Functions
# button lists them: [{"section", "name", "signature", "description"}, ...]
print([row["signature"] for row in solver.get_force_field_builtins()])

# Compile the script on the running server, as Compile and Check does.
print(solver.check_force_field_script())
# {'ok': True, 'summary': '...'} or {'ok': False, 'error': ..., 'line': n}
```

`check_force_field_script()` needs a connection to a running server. It
raises `RuntimeError` when no script is chosen, when there is no such
connection, or when no answer arrives within `timeout` seconds (60 by
default). The other settings are `force_field_collection` and
`force_field_preview_resolution`. A force field is evaluated exactly at every
vertex, so there is no sampling box, spacing, time sampling or size limit:
assigning `force_field_padding`, `force_field_spacing`,
`force_field_time_samples` or `force_field_max_mb` raises `AttributeError`.

`get_force_fields()` lists every force field object that **Transfer**
considers, as the panel's rows do, without changing anything. Each entry holds
`object`, `type` (Blender's identifier, such as `"FORCE"` or `"FLUID_FLOW"`) and
`type_name` (the name the Physics tab shows), `refused` (why **Transfer** would
refuse the field, or `None`), `groups` (the group uuids it is limited to, or
`None`), `pushes` (the simulated objects it reaches), `tracked` and, for a
Fluid Flow, `fluid`:

```python
for field in solver.get_force_fields():
    if field["refused"]:
        print(field["object"], "is refused:", field["refused"])
```

A field object (by name) or the script (`"SCRIPT"`) can be limited to some
groups. `None` restores every group, and a **Static** group or an empty list
raises `ValueError`:

```python
solver.set_force_field_targets("Turbulence", [cloth])
solver.set_force_field_targets("SCRIPT", None)
print(solver.get_force_field_targets("Turbulence"))   # [cloth.uuid]
```

Each group scales the fields on its objects with `force_field_weight`:

```python
cloth.param.force_field_weight = 0.0   # this group ignores every field
```

See [Force Fields](../workflow/params/force_fields.md) for what each field
type does and what the script may contain.

## Sand

`solver.convert_to_particle_mesh` replaces a solid mesh with the grain cloud
a **Sand** group simulates, as **Convert To Solid Particle Mesh** does. It
discards the faces and returns the number of grains it seeded:

```python
n_grains = solver.convert_to_particle_mesh("Pile", grain_radius=0.01)
sand = solver.create_group("Sand", type="SAND")
sand.add("Pile")
sand.param.sand_particle_mass = 10.0   # grams per grain
```

`extra_spacing` (default `0.0`) leaves room between grains beyond touching, as
**Extra Spacing** does, and `rng_seed` (default `0`) picks a different,
repeatable arrangement, as **Random Seed** does.

The grain radius is locked once the object is converted, so choose it
first. The call raises `ValueError` when the object is missing, is not a
mesh, has no faces or is already a particle mesh, when `grain_radius` is not
positive, or when no grain fits inside the mesh. See
[Sand-Specific](../workflow/params/material.md#sand-specific).

## Deformation Captures

`solver.recapture_all_deformations()` is **Re-capture All Deformations**, and
`solver.clear_all_deformations()` is **Clear All Deformations**. Captures
advance on Blender's event loop. The call returns once the first capture has
started, and the rest finish on later ticks. A script that keeps running
holds the loop and blocks them, and a Blender started with `--background`
runs no ticks and captures nothing. Schedule whatever depends on the caches
from a timer, and gate it on `solver.is_capture_running()`:

```python
import bpy

def transfer_when_captured():
    if solver.is_capture_running():
        return 0.1          # check again in 100 ms
    solver.transfer_data()
    return None

solver.recapture_all_deformations()
bpy.app.timers.register(transfer_when_captured)
```

Both calls raise `ValueError` when there is nothing to capture or clear, or
a capture or bake is already running. See
[Static Objects](../workflow/scene/static_objects.md).

## More from Python

- [Mesh Cleaning](../workflow/scene/mesh_cleaning.md): `scan_meshes`, which
  returns its report as data, and the repairs, which take object names and
  return `solver` for chaining.
- [Exporting](../workflow/sim/exporting.md): `export_usd`, `export_alembic`
  and `get_unexportable_curves`.
- [Blender Python API Reference](python_api_reference.rst): every method.

## Fallback: Raw Operator Dispatch

Operations without a dedicated method are reachable by attribute lookup.
Unknown attributes on `solver` fall through to
`bpy.ops.zozo_contact_solver.<name>`:

```python
# Equivalent to bpy.ops.zozo_contact_solver.transfer_data()
solver.transfer_data()

# Keyword args are forwarded as the operator's properties.
solver.set(key="project_name", value="hero_shot")
```

Every MCP tool name (see [MCP Server](./mcp.md)) has a matching
operator. Three limits apply to a call made this way:

- It returns only Blender's operator status, never the tool's result. The
  `get_*` and `list_*` tools, and the tools that report status, give a script
  nothing back. Use the fluent methods above, or MCP, for data.
- List and dictionary arguments are JSON strings. Build them with
  `json.dumps`:

  ```python
  import json

  solver.add_objects_to_group(group_uuid=cloth.uuid, object_names=json.dumps(["Shirt"]))
  ```
- A tool that starts a job, such as `connect`, `transfer_data`,
  `run_simulation` or `fetch_animation`, returns as soon as the job starts.
  The job advances on Blender's event loop. A refusal raises `RuntimeError`.

## See Also

- [Pins and Operations](../workflow/constraints/pins.md): the full
  semantics of `spin`, `scale`, `torque`, and the center modes.
- [Dynamic Parameters](../workflow/params/dynamic.md): keyframing the scene
  properties, the `.dyn()` keys, `hold()` and `change()`.
- [Object Groups](../workflow/scene/object_groups.md): what each group type
  does in the solver.
- [Snap and Merge](../workflow/constraints/snap_merge.md): snap, merge pairs
  and stitches.
- [Invisible Colliders](../workflow/constraints/colliders.md): walls, spheres
  and their keyframes.
- [Force Fields](../workflow/params/force_fields.md): field types, the
  script, and group targets.
- [Mesh Cleaning](../workflow/scene/mesh_cleaning.md) and
  [Exporting](../workflow/sim/exporting.md): the scripted repairs and
  exports.
- [MCP Server](./mcp.md): the same surface over JSON-RPC for agents.

:::{admonition} Under the hood
:class: toggle

The fluent API is a set of proxy objects that call the add-on's operators
and read its scene state:

- `solver.param` forwards any name to the scene state or the connection
  state. Reading an unknown name raises `AttributeError`; assigning one goes
  through the `zozo_contact_solver.set` operator, which reports an error
  that surfaces as `RuntimeError`. `solver.param.dyn(name)` returns a
  dynamic-parameter builder.
- The builder's list is the collection `state.dyn_params` on the scene
  state. When a `.blend` is loaded, the add-on converts each entry into
  animation curves on the scene state and empties the list, so a script that
  reads `state.dyn_params` after a reload finds the converted entries gone.
  An entry with any keyframe after its first that is at or before the
  **Starting Frame** cannot become a curve. It stays in the list, and
  **Transfer** refuses it by name.
- `solver.create_group(...)` returns a group proxy. Its `.param` exposes
  that group's material and contact whitelist. The collider builders have a
  `.param` with a whitelist of their own.
- `group.create_pin(...)` returns a pin proxy; every method except
  `delete()` returns `self` so calls chain.
- `solver.add_wall(...)` and `solver.add_sphere(...)` return collider
  builders. The `.time()` cursor is tracked on the builder itself; frames
  must be strictly increasing.

For type hints, import the public names from the same `ops.api` package, for
example `from bl_ext.user_default.ppf_contact_solver.ops.api import Group, Pin`
for an add-on in the **User Default** repository. The
names are `Solver`, `Group`, `Pin`, `Curve`, `SceneParam`, `GroupParam`,
`ColliderParam`, `DynParam`, `Wall` and `Sphere`. The proxy classes behind
them are not part of the public contract. Depend on the attribute and
method names shown on this page rather than on `isinstance` checks.
:::
