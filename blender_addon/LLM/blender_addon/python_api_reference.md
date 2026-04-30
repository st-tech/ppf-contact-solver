# Blender Python API reference

This document mirrors the auto-generated `python_api_reference.rst` found at `docs/blender_addon/integrations/python_api_reference.rst`. It catalogs every public class, method, property, and attribute on the ZOZO Contact Solver Python surface so an LLM can script the add-on after `from zozo_contact_solver import solver`.

If you reached this file as MCP resource `llm://python_api_reference`, its sibling resources (`llm://index`, `llm://overview`, `llm://parameters`, and so on) cover the surrounding concepts. Call `resources/list` once and pick the matching URI; the full resource surface (URI scheme, list/read examples, error handling) is documented under the **Resources** section of `llm://integrations`.

## Import and basic use

```python
from zozo_contact_solver import solver

# Scene parameters
solver.param.gravity = (0, 0, -9.8)

# Create a group and assign an object
group = solver.create_group("Cloth", type="SHELL")
group.add("Plane")

# Pin a vertex group and add an operation
pin = group.create_pin("Plane", "TopEdge")
pin.spin(axis=(0, 0, 1), angular_velocity=180.0)

# Transfer, simulate, fetch results back into Blender
solver.transfer_data()
solver.run_simulation()
solver.fetch_animation()
```

Fallback: any operator registered under `bpy.ops.zozo_contact_solver.<name>()`, including every MCP handler, is reachable as `solver.<name>(...)` via the `Solver.__getattr__` passthrough.

**Classes:**

- `Solver`
- `SceneParam`
- `DynParam`
- `Group`
- `GroupParam`
- `Pin`
- `Wall`
- `Sphere`
- `ColliderParam`

## Class: Solver

Top-level entry point for the ZOZO Contact Solver.

Available as `solver` when imported via:

```python
from zozo_contact_solver import solver
```

Scene parameters are accessed via `param` (a `SceneParam` proxy). Groups, pins, and invisible colliders are created via the methods below.

Unrecognized attribute access falls through to `bpy.ops.zozo_contact_solver.<name>()`, so every operator registered under that namespace, including every MCP handler, can be called as a method on `solver`.

```python
solver.param.gravity = (0, 0, -9.8)
group = solver.create_group("Sphere", type="SOLID")
group.add("Sphere")
group.param.solid_density = 1000
```

### create_group(name: str='', type: str='SOLID') -> Group

Create a new dynamics group.

**Parameters:**

- **name**: Display name for the group. Empty string leaves the auto-generated name in place.
- **type**: One of `"SOLID"`, `"SHELL"`, `"ROD"`, `"STATIC"`.

**Returns:** A `Group` proxy for the newly created group.

### get_group(group_uuid: str) -> Group

Look up a group by UUID.

**Parameters:**

- **group_uuid**: UUID string of the group.

**Returns:** A `Group` proxy.

**Raises:** `KeyError` if the group does not exist.

### get_groups() -> list[Group]

Return `Group` proxies for every active group.

### delete_all_groups() -> Solver

Delete every active group and the pins they own.

**Returns:** `self` for chaining.

### clear() -> Solver

Reset the entire solver state to defaults.

Deletes every active group, resets scene parameters to their property defaults, clears merge pairs, invisible colliders, dynamic parameters, fetched-frame cache, saved pin keyframes, and any residual `MESH_CACHE` modifiers on mesh objects. Call this at the top of any script that needs a clean slate.

**Returns:** `self` for chaining.

### snap(object_a: str, object_b: str) -> Solver

Translate *object_a* so its nearest vertex lands on *object_b*.

**Parameters:**

- **object_a**: Name of the mesh that moves.
- **object_b**: Name of the mesh that stays in place.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if either object is missing, not a mesh, or validation in the underlying mutation service fails.

```python
solver.snap("Shirt", "Mannequin")
```

### add_merge_pair(object_a: str, object_b: str) -> Solver

Mark two objects to be merged at their shared contact.

**Parameters:**

- **object_a**: Name of the first mesh.
- **object_b**: Name of the second mesh.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if either object is missing, not a mesh, or the pair is invalid.

```python
solver.add_merge_pair("SleeveLeft", "BodyLeft")
```

### remove_merge_pair(object_a: str, object_b: str) -> Solver

Remove a previously added merge pair.

The ordering of *object_a* and *object_b* does not matter: the pair is matched by UUID in either direction.

**Parameters:**

- **object_a**: Name of the first mesh.
- **object_b**: Name of the second mesh.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if validation fails for the given pair.

### get_merge_pairs() -> list[tuple[str, str]]

Return every merge pair as a list of `(object_a, object_b)` tuples.

### clear_merge_pairs() -> Solver

Remove every merge pair.

**Returns:** `self` for chaining.

### add_wall(position, normal) -> Wall

Add an invisible infinite-plane wall collider.

**Parameters:**

- **position**: `(x, y, z)` world-space point on the plane.
- **normal**: `(nx, ny, nz)` outward-facing plane normal. Need not be unit-length.

**Returns:** A chainable `Wall` builder bound to the newly added collider.

**Raises:** `ValueError` if the position or normal fails vec3 validation.

### add_sphere(position, radius) -> Sphere

Add an invisible sphere collider.

**Parameters:**

- **position**: `(x, y, z)` world-space center.
- **radius**: Sphere radius.

**Returns:** A chainable `Sphere` builder bound to the newly added collider.

**Raises:** `ValueError` if the position or radius fails validation.

### get_invisible_colliders() -> list

Return every invisible collider as a list of `(type, name)` tuples.

*type* is one of `"WALL"` or `"SPHERE"`.

### clear_invisible_colliders() -> Solver

Remove every invisible collider.

**Returns:** `self` for chaining.

## Class: SceneParam

Attribute proxy for scene and SSH/connection parameters.

Accessed as `Solver.param`. Supports both get and set via attribute access. Writes go through the `zozo_contact_solver.set` operator (with auto type coercion), reads fall through to the scene's addon state or SSH state.

`gravity` is an alias for `gravity_3d`.

```python
solver.param.step_size = 0.004
print(solver.param.gravity)
```

Dynamic (keyframed) parameters are accessed via `dyn`:

```python
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
```

### dyn(key: str) -> DynParam

Select a parameter for dynamic keyframing.

**Parameters:**

- **key**: One of `"gravity"`, `"wind"`, `"air_density"`, `"air_friction"`, `"vertex_air_damp"`.

**Returns:** A chainable `DynParam` builder.

**Raises:** `ValueError` if *key* is not one of the valid dynamic keys.

## Class: DynParam

Fluent builder for dynamic scene parameter keyframes.

Mirrors the frontend `session.param.dyn()` API but uses **frames** instead of seconds. Obtained from `SceneParam.dyn`.

Valid parameter keys: `"gravity"`, `"wind"`, `"air_density"`, `"air_friction"`, `"vertex_air_damp"`.

Frames must be strictly increasing within a chain. Every mutating method returns `self` so operations chain.

```python
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)
```

### time(frame: int) -> DynParam

Advance the frame cursor.

**Parameters:**

- **frame**: Target frame (must be strictly greater than the current cursor position).

**Returns:** `self` for chaining.

**Raises:** `ValueError` if *frame* is not strictly increasing.

### hold() -> DynParam

Hold the previous value at the current cursor frame (step function).

**Returns:** `self` for chaining.

### change(value, strength=None) -> DynParam

Set a new value at the current cursor frame.

**Parameters:**

- **value**: For `"gravity"`, an `(x, y, z)` tuple. For `"wind"`, an `(x, y, z)` direction tuple. For scalar keys (`"air_density"`, `"air_friction"`, `"vertex_air_damp"`), a `float`.
- **strength**: Wind strength (only for `"wind"`).

**Returns:** `self` for chaining.

### clear() -> DynParam

Remove this dynamic parameter entirely.

**Returns:** `self` for chaining (though no further method on this builder will do anything meaningful after `clear()`).

## Class: Group

A dynamics group proxy.

Created via `Solver.create_group`. Material parameters are accessed via `param`:

```python
group.param.friction = 0.5
group.param.shell_density = 1.0
```

Every mutating method returns `self` so operations chain.

### uuid

Type: `str`

The UUID of this group. Stable across renames.

### param

Type: `GroupParam`

Material and simulation parameter proxy. See `GroupParam`.

### set_overlay_color(r: float, g: float, b: float, a: float=1.0) -> Group

Set the viewport overlay color for this group and enable it.

**Parameters:**

- **r**: Red channel in `[0, 1]`.
- **g**: Green channel in `[0, 1]`.
- **b**: Blue channel in `[0, 1]`.
- **a**: Alpha in `[0, 1]` (default `1.0`).

**Returns:** `self` for chaining.

```python
group.set_overlay_color(0.9, 0.2, 0.1)  # red overlay
```

### add(*object_names: str) -> Group

Add mesh objects to this group by name.

**Parameters:**

- **\*object_names**: One or more Blender object names.

**Returns:** `self` for chaining.

```python
group.add("Shirt", "Skirt", "Sleeve")
```

### remove(object_name: str) -> Group

Remove an object from this group.

**Parameters:**

- **object_name**: Name of the object to remove.

**Returns:** `self` for chaining.

### create_pin(object_name: str, vertex_group_name: str) -> Pin

Pin a vertex group so its vertices stay fixed during simulation.

**Parameters:**

- **object_name**: Name of the mesh object.
- **vertex_group_name**: Name of the vertex group on that object.

**Returns:** A `Pin` proxy for the newly created pin.

**Raises:** `ValueError` if the object is missing, not a mesh, or the vertex group does not exist on it.

```python
pin = group.create_pin("Cloth", "collar")
pin.move(delta=(0, 0, 0.2), frame=60)
```

### get_pins() -> list[Pin]

Return all pins in this group as `Pin` proxies.

### clear_keyframes() -> Group

Delete all keyframes for all pins in this group.

Convenience method that calls `Pin.clear_keyframes` on every pin returned by `get_pins`.

**Returns:** `self` for chaining.

### delete() -> None

Delete this group and every pin it owns.

## Class: GroupParam

Proxy for material and simulation parameters on a group.

Accessed via `Group.param`. Attribute access is whitelisted: reading or writing a name outside the whitelist raises `AttributeError`.

Whitelisted attributes:

- **Solver model**: `solid_model`, `shell_model`
- **Density**: `solid_density`, `shell_density`, `rod_density`
- **Young's modulus**: `solid_young_modulus`, `shell_young_modulus`, `rod_young_modulus`
- **Poisson ratio**: `solid_poisson_ratio`, `shell_poisson_ratio`
- **Contact**: `friction`, `contact_gap`, `contact_gap_rat`, `contact_offset`, `contact_offset_rat`
- **Strain limit**: `enable_strain_limit`, `strain_limit`
- **Inflation**: `enable_inflate`, `inflate_pressure`
- **Plasticity**: `enable_plasticity`, `plasticity`, `plasticity_threshold`
- **Bend plasticity**: `enable_bend_plasticity`, `bend_plasticity`, `bend_plasticity_threshold`, `bend_rest_angle_source`
- **Shell-specific**: `bend`, `shrink`, `shrink_x`, `shrink_y`, `stitch_stiffness`

```python
group.param.friction = 0.5
group.param.shell_density = 1.0
```

## Class: Pin

A pinned vertex group bound to a dynamics group.

Created via `Group.create_pin(object_name, vertex_group_name)`. Every mutating method returns `self` so operations chain.

```python
pin = group.create_pin("Cloth", "hem")
pin.move(delta=(0, 0, 1.0), frame=60)  # lift hem over 60 frames
pin.unpin(frame=120)                   # release at frame 120
```

### object_name

Type: `str`

Name of the mesh object this pin belongs to.

### vertex_group_name

Type: `str`

Name of the vertex group this pin targets.

### pull(strength: float=1.0) -> Pin

Use pull force instead of hard pin constraint.

Pull allows the vertices to move but applies a restoring force toward their target position.

**Parameters:**

- **strength**: Pull force strength (default 1.0).

**Returns:** `self` for chaining.

```python
group.create_pin("Cloth", "shoulder").pull(strength=2.5)
```

### spin(axis: tuple[float, float, float]=(1, 0, 0), angular_velocity: float=360.0, flip: bool=False, center: tuple[float, float, float] | None=None, center_mode: str | None=None, center_direction: tuple[float, float, float] | None=None, center_vertex: int | None=None, frame_start: int=1, frame_end: int=60, transition: str='LINEAR') -> Pin

Add a spin operation to this pin.

**Parameters:**

- **axis**: Rotation axis vector.
- **angular_velocity**: Degrees per second.
- **flip**: Reverse spin direction.
- **center**: Center of rotation (for ABSOLUTE mode).
- **center_mode**: `"CENTROID"`, `"ABSOLUTE"`, `"MAX_TOWARDS"`, or `"VERTEX"`. If `None`, inferred from other args (`None` center gives `"CENTROID"`).
- **center_direction**: Direction for `MAX_TOWARDS` mode.
- **center_vertex**: Vertex index for `VERTEX` mode.
- **frame_start**: Start frame.
- **frame_end**: End frame.
- **transition**: `"LINEAR"` or `"SMOOTH"`.

**Returns:** `self` for chaining.

```python
# Spin about the centroid at 180 deg/s for frames 1-60
pin.spin(axis=(0, 0, 1), angular_velocity=180.0)
# Spin about an absolute world-space pivot
pin.spin(axis=(0, 1, 0), center=(0, 0, 1),
         frame_start=30, frame_end=90)
```

### scale(factor: float=1.0, center: tuple[float, float, float] | None=None, center_mode: str | None=None, center_direction: tuple[float, float, float] | None=None, center_vertex: int | None=None, frame_start: int=1, frame_end: int=60, transition: str='LINEAR') -> Pin

Add a scale operation to this pin.

**Parameters:**

- **factor**: Scale factor.
- **center**: Center point (for `ABSOLUTE` mode).
- **center_mode**: `"CENTROID"`, `"ABSOLUTE"`, `"MAX_TOWARDS"`, or `"VERTEX"`. If `None`, inferred from other args (`None` center gives `"CENTROID"`).
- **center_direction**: Direction for `MAX_TOWARDS` mode.
- **center_vertex**: Vertex index for `VERTEX` mode.
- **frame_start**: Start frame.
- **frame_end**: End frame.
- **transition**: `"LINEAR"` or `"SMOOTH"`.

**Returns:** `self` for chaining.

```python
# Shrink to 50% over frames 1-60 about the centroid
pin.scale(factor=0.5, transition="SMOOTH")
```

### torque(magnitude: float=1.0, axis_component: str='PC3', flip: bool=False, frame_start: int=1, frame_end: int=60) -> Pin

Add a torque operation to this pin.

Applies a rotational force around a PCA-computed axis.

**Parameters:**

- **magnitude**: Torque in N·m.
- **axis_component**: `"PC1"` (major), `"PC2"` (middle), or `"PC3"` (minor).
- **flip**: Reverse torque direction.
- **frame_start**: Start frame.
- **frame_end**: End frame.

**Returns:** `self` for chaining.

```python
pin.torque(magnitude=2.0, axis_component="PC1",
           frame_start=1, frame_end=30)
```

### move_by(delta: tuple[float, float, float]=(0, 0, 0), frame_start: int=1, frame_end: int=60, transition: str='LINEAR') -> Pin

Ramp a translation of the pinned vertices over a frame range.

**Parameters:**

- **delta**: `(dx, dy, dz)` offset.
- **frame_start**: Start frame.
- **frame_end**: End frame.
- **transition**: `"LINEAR"` or `"SMOOTH"`.

**Returns:** `self` for chaining.

```python
# Lift 1.0m along +Z between frames 10 and 90
pin.move_by(delta=(0, 0, 1.0),
            frame_start=10, frame_end=90,
            transition="SMOOTH")
```

### unpin(frame: int) -> Pin

Mark this pin to be released at the given frame.

Sets the duration on the underlying UI property so the encoder and clear logic are aware. Also prevents future `move(frame=N)` calls where *N* >= *frame*.

**Parameters:**

- **frame**: Frame number at which the pin is released.

**Returns:** `self` for chaining.

```python
pin.move(delta=(0, 0, 1), frame=60).unpin(frame=120)
```

### move(delta: tuple[float, float, float]=(0, 0, 0), frame: int | None=None) -> Pin

Move pin vertices by *delta* and optionally keyframe at *frame*.

On the first call with *frame*, the current vertex positions are automatically keyframed at the current scene frame before any movement is applied. Ignored if *frame* >= the unpin frame.

**Parameters:**

- **delta**: `(dx, dy, dz)` offset to apply (default no movement).
- **frame**: Frame number to keyframe at. `None` means no keyframe.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if the target object is missing, not a mesh, or the vertex group does not exist on it.

```python
pin = group.create_pin("Cloth", "hem")
pin.move(delta=(0, 0, 1.0), frame=60)   # auto-keyframes start
pin.move(delta=(0.5, 0, 0), frame=120)  # adds another keyframe
```

### clear_keyframes() -> Pin

Delete all positional keyframes for this pin's vertices.

**Returns:** `self` for chaining.

### delete() -> None

Remove this pin from its group.

**Raises:** `ValueError` if the owning group or pin item can no longer be found (for example, after `solver.clear()`).

## Class: Wall

Chainable builder for invisible wall colliders.

Returned by `Solver.add_wall`. Keyframe frames must be strictly increasing. Every mutating method returns `self`.

```python
solver.add_wall((0, 0, 0), (0, 0, 1)).param.friction = 0.5
(solver.add_wall((0, 0, 0), (0, 1, 0))
       .time(60).hold().time(61).move_to((0, 1, 0)))
```

### param

Type: `ColliderParam`

Collider parameter proxy. See `ColliderParam`.

### time(frame: int) -> Wall

Advance the keyframe cursor.

**Parameters:**

- **frame**: Target frame (must be strictly greater than the current cursor position).

**Returns:** `self` for chaining.

**Raises:** `ValueError` if *frame* is not strictly increasing.

### hold() -> Wall

Hold the previous position at the current cursor frame.

**Returns:** `self` for chaining.

### move_to(position) -> Wall

Keyframe a new absolute position at the current cursor frame.

**Parameters:**

- **position**: `(x, y, z)` world-space position.

**Returns:** `self` for chaining.

### move_by(delta) -> Wall

Keyframe a position offset from the previous keyframe.

**Parameters:**

- **delta**: `(dx, dy, dz)` offset added to the previous keyframed position.

**Returns:** `self` for chaining.

### delete() -> None

Remove this wall collider from the scene.

## Class: Sphere

Chainable builder for invisible sphere colliders.

Returned by `Solver.add_sphere`. Keyframe frames must be strictly increasing. Every mutating method returns `self`.

```python
solver.add_sphere((0, 0, 0), 0.98).invert().hemisphere()
(solver.add_sphere((0, 0, 0), 1.0)
       .time(60).hold().time(61).radius(0.5))
```

### param

Type: `ColliderParam`

Collider parameter proxy. See `ColliderParam`.

### invert() -> Sphere

Flip the sphere inside-out so contact is on the inside surface.

**Returns:** `self` for chaining.

### hemisphere() -> Sphere

Treat this collider as a hemisphere rather than a full sphere.

**Returns:** `self` for chaining.

### time(frame: int) -> Sphere

Advance the keyframe cursor.

**Parameters:**

- **frame**: Target frame (must be strictly greater than the current cursor position).

**Returns:** `self` for chaining.

**Raises:** `ValueError` if *frame* is not strictly increasing.

### hold() -> Sphere

Hold the previous position and radius at the current cursor frame.

**Returns:** `self` for chaining.

### move_to(position) -> Sphere

Keyframe a new absolute position at the current cursor frame.

**Parameters:**

- **position**: `(x, y, z)` world-space position.

**Returns:** `self` for chaining.

### radius(r) -> Sphere

Keyframe a new radius at the current cursor frame.

**Parameters:**

- **r**: New radius.

**Returns:** `self` for chaining.

### transform_to(position, radius) -> Sphere

Keyframe both position and radius together.

**Parameters:**

- **position**: `(x, y, z)` world-space position.
- **radius**: New radius.

**Returns:** `self` for chaining.

### delete() -> None

Remove this sphere collider from the scene.

## Class: ColliderParam

Attribute proxy for invisible-collider parameters.

Accessed via `Wall.param` or `Sphere.param`. Attribute access is whitelisted: reading or writing a name outside the whitelist raises `AttributeError`.

Whitelisted attributes:

- `friction`: contact friction coefficient
- `contact_gap`: contact gap thickness
- `thickness`: wall/sphere shell thickness
- `enable_active_duration`: `True` to limit collider lifetime
- `active_duration`: number of frames the collider is active when `enable_active_duration` is set

```python
solver.add_wall((0, 0, 0), (0, 0, 1)).param.friction = 0.5
```

---

Reference mirror of `docs/blender_addon/integrations/python_api_reference.rst` regenerated from `blender_addon/ops/api.py`.
