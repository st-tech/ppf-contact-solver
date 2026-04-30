# 🐍 Blender Python API

Everything the add-on does from the UI (creating groups, pinning
vertex groups, keyframing spins, dropping invisible colliders, snapping
meshes) can be driven from Python inside Blender's scripting editor.
This is the right tool for procedural scene setup, batch variant
generation, regression tests, and anything you do not want to click
through three hundred times.

:::{tip}
This page is a tutorial-style walkthrough. For the full method-by-method
list, generated directly from the source, see the
[Blender Python API Reference](python_api_reference.rst).
:::

## Import

```python
from zozo_contact_solver import solver
```

The add-on publishes the `zozo_contact_solver` package at registration
time, so this import works regardless of where the add-on lives on
disk. Every example below assumes it is already imported.

## Scene Parameters

`solver.param` is a whitelisted proxy over the scene-level state. Set
any exposed property by attribute:

```python
solver.param.project_name = "shirt_drape"
solver.param.frame_count  = 180
solver.param.frame_rate   = 60
solver.param.step_size    = 0.001
solver.param.gravity      = (0, 0, -9.8)   # alias for gravity_3d
solver.param.air_density  = 0.001225
```

`gravity` is an alias for `gravity_3d`; reads and writes go through it
transparently. See [Scene Parameters](../workflow/params/scene.md) for
the full list.

### Dynamic Parameters

Keyframe-driven scene parameters use the `dyn()` builder. The API
mirrors the frontend's `session.param.dyn()` but takes **frames**, not
seconds:

```python
# Flip gravity at frame 60: hold the initial value through 60,
# then snap to the new value at 61.
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Start a wind gust at frame 30.
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Scalars like air_density just take a float.
solver.param.dyn("air_density").time(100).change(0.005)

# Nuke a dynamic param entirely.
solver.param.dyn("gravity").clear()
```

Valid keys: `"gravity"`, `"wind"`, `"air_density"`, `"air_friction"`,
`"vertex_air_damp"`. Frames must be strictly increasing within a chain;
`time(30).time(30)` raises.

See [Dynamic Parameters](../workflow/params/dynamic.md) for the
semantics of `hold()` vs. `change()`.

## Groups

```python
cloth = solver.create_group("Cloth", type="SHELL")   # SOLID | SHELL | ROD | STATIC
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
`solver.get_group(uuid)`, or walk every active group:

```python
for g in solver.get_groups():
    print(g.uuid, g.param.friction)
```

### Group Surface

| Method                                | Purpose                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| `add(*object_names)`                  | Add one or more mesh objects by name                          |
| `remove(object_name)`                 | Remove one object                                             |
| `set_overlay_color(r, g, b, a=1.0)`   | Set and enable the viewport overlay color                     |
| `create_pin(obj, vg)`                 | Pin a vertex group; returns a pin proxy (see below)           |
| `get_pins()`                          | List every pin in this group as pin proxies                   |
| `clear_keyframes()`                   | Shortcut for `pin.clear_keyframes()` on every pin             |
| `delete()`                            | Remove this group                                             |
| `.param.<name>`                       | Whitelisted material/contact parameter access                 |
| `.uuid`                               | UUID string, stable across renames                            |

Material parameters on `.param` are validated: assigning a name outside
the whitelist raises `AttributeError`. See
[Material Parameters](../workflow/params/material.md) for the full
list.

Bulk lifecycle operations live on the solver:

```python
solver.delete_all_groups()
solver.clear()   # full reset: groups, scene params, merge pairs,
                 #             colliders, dyn params, fetched animation
```

## Pins and Operations

`group.create_pin(object_name, vertex_group_name)` returns a pin proxy.
Every mutating method returns `self`, so chaining works:

```python
pin = cloth.create_pin("Shirt", "ShoulderPins")
pin.spin(axis=(0, 0, 1), angular_velocity=360, frame_start=1, frame_end=60)
pin.unpin(frame=90)

# Soft pin instead of a hard constraint.
cloth.create_pin("Shirt", "HemPins").pull(strength=2.0)

# Chain a scale and a spin on the same pin.
(cloth.create_pin("Shirt", "HemPins")
      .scale(factor=0.5, center_direction=(0, 0, -1), frame_start=1, frame_end=60)
      .spin(axis=(0, 1, 0), angular_velocity=180, center_vertex=42))
```

### Pin Surface

| Method                                                          | Purpose                                                   |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| `pull(strength=1.0)`                                            | Switch to a soft pull force                               |
| `move(delta, frame=None)`                                       | Translate the pinned verts; auto-keyframes when `frame` given |
| `move_by(delta, frame_start, frame_end, transition="LINEAR")`   | Ramp a translation over a frame range                     |
| `spin(axis, angular_velocity, flip, center*, frame_start, frame_end, transition)` | Rotate about a derived pivot          |
| `scale(factor, center*, frame_start, frame_end, transition)`    | Scale from a derived pivot                                |
| `torque(magnitude, axis_component="PC3", flip, frame_start, frame_end)` | PCA-axis torque                                   |
| `unpin(frame)`                                                  | Release the pin at `frame`; also blocks later `move(frame=N≥frame)` |
| `clear_keyframes()`                                             | Drop all positional keyframes for this pin's verts        |
| `delete()`                                                      | Remove this pin from its group                            |

`transition` is `"LINEAR"` or `"SMOOTH"`. `torque`'s `axis_component`
is `"PC1"` / `"PC2"` / `"PC3"`.

### `move` with Auto-Keyframing

The first `move(...)` call with a `frame=` argument auto-keyframes the
current vertex positions at the current scene frame before applying
`delta` and keyframing again at `frame`. Subsequent calls only keyframe
at the target frame. Calls with `frame >= unpin_frame` are silently
ignored, which is handy when you want to leave the pin released cleanly.

```python
pin = cloth.create_pin("Shirt", "SleevePins")
pin.unpin(frame=30)
pin.move(delta=(0, 0, 0.5), frame=20)   # keyframed; under the unpin frame
pin.move(delta=(0, 0, 0.5), frame=40)   # ignored; past unpin
```

### Center-Mode Inference for `spin` and `scale`

Pass whichever argument names your pivot, and the API picks the matching
mode for you:

| Argument you pass    | Inferred `center_mode` |
| -------------------- | ---------------------- |
| `center=(x, y, z)`   | `ABSOLUTE`             |
| `center_direction=v` | `MAX_TOWARDS`          |
| `center_vertex=idx`  | `VERTEX`               |
| none of the above    | `CENTROID`             |

Passing `center_mode="..."` explicitly overrides the inference. See
[Pins and Operations](../workflow/constraints/pins.md) for what each
mode actually computes.

```python
pin.spin(axis=(0, 0, 1), angular_velocity=360)                        # CENTROID
pin.spin(axis=(0, 0, 1), angular_velocity=360, center=(0, 0, 1))      # ABSOLUTE
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_direction=(0, 0, -1))  # MAX_TOWARDS
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_vertex=42)      # VERTEX
```

## Snap and Merge

```python
solver.snap("Shirt", "Mannequin")               # translate Shirt onto nearest vertex on Mannequin

solver.add_merge_pair("Shirt", "Mannequin")
solver.remove_merge_pair("Shirt", "Mannequin")
solver.get_merge_pairs()                        # → [("Shirt", "Mannequin"), ...]
solver.clear_merge_pairs()
```

All of these share the same validation layer as the MCP interface and
the UI, so you get identical errors. Bad names raise `ValueError`.

## Invisible Colliders

Walls and spheres return a chainable builder. Parameters on `.param`
cover `friction`, `contact_gap`, `thickness`, and `enable_active_duration`
/ `active_duration`.

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

solver.get_invisible_colliders()   # → [("WALL", "Wall"), ("SPHERE", "Sphere"), ...]
solver.clear_invisible_colliders()
```

### Builder Surface

| Method             | Wall | Sphere | Purpose                                              |
| ------------------ | :--: | :----: | ---------------------------------------------------- |
| `.time(frame)`     |  yes |  yes   | Advance the keyframe cursor (must be increasing)     |
| `.hold()`          |  yes |  yes   | Hold the previous value at the cursor                |
| `.move_to(pos)`    |  yes |  yes   | Keyframe a new position                              |
| `.move_by(delta)`  |  yes |        | Keyframe a position offset from the previous         |
| `.radius(r)`       |      |  yes   | Keyframe a new radius                                |
| `.transform_to(p, r)` |   |  yes   | Keyframe position + radius together                  |
| `.invert()`        |      |  yes   | Flip inside-out (contact on the inside)              |
| `.hemisphere()`    |      |  yes   | Treat as a hemisphere                                |
| `.param.*`         |  yes |  yes   | `friction`, `contact_gap`, `thickness`, `active_duration`, `enable_active_duration` |
| `.delete()`        |  yes |  yes   | Remove this collider                                 |

See [Invisible Colliders](../workflow/constraints/colliders.md) for how
the keyframe timeline is evaluated.

## Reset

```python
solver.clear()
```

Wipes every active group, resets scene parameters to their property
defaults, clears merge pairs, invisible colliders, dynamic parameters,
fetched frames, and residual `MESH_CACHE` modifiers. Run it at
the top of any script that needs a clean slate.

## Fallback: Raw Operator Dispatch

Anything not yet on the fluent API is reachable by attribute lookup.
Unknown attributes on `solver` fall through to
`bpy.ops.zozo_contact_solver.<name>`:

```python
# Equivalent to bpy.ops.zozo_contact_solver.transfer_data()
solver.transfer_data()

# Keyword args are forwarded as the operator's properties.
solver.set(key="project_name", value="hero_shot")
```

Every MCP handler name (see [MCP Server](./mcp.md)) has a matching
operator, so whatever you can call over MCP you can also call here.

## See Also

- [Pins and Operations](../workflow/constraints/pins.md): the full
  semantics of `spin`, `scale`, `torque`, and the center modes.
- [Dynamic Parameters](../workflow/params/dynamic.md): `.dyn()`
  keys, `hold()` / `change()`, and how they interpolate.
- [Object Groups](../workflow/scene/object_groups.md): what each group type
  actually does in the solver.
- [MCP Server](./mcp.md): the same surface over JSON-RPC for agents.

:::{admonition} Under the hood
:class: toggle

The fluent API is a thin layer of proxy objects over the add-on's
operators and scene state:

- `solver.param` exposes a whitelisted attribute surface over
  scene-level properties. Assigning an unknown name raises
  `AttributeError`. `solver.param.dyn(name)` returns a dynamic-parameter
  builder.
- `solver.create_group(...)` returns a group handle. Its `.param`
  exposes that group's material/contact whitelist.
- `group.create_pin(...)` returns a pin handle; every mutating method
  returns `self` so calls chain.
- `solver.add_wall(...)` and `solver.add_sphere(...)` return builder
  handles. The `.time()` cursor is tracked on the builder itself;
  frames must be strictly increasing.

The underlying proxy types are not part of the public contract. Pin
your scripts to the attribute and method names shown above, not to
`isinstance` checks.
:::
