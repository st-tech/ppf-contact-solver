# Blender Python API reference

This document tracks the bundled Python API surface exported from `blender_addon/ops/api/__init__.py` and its submodules. It catalogs every public class, method, property, and attribute on the ZOZO Contact Solver Python surface so an LLM can script the add-on after `from bl_ext.user_default.ppf_contact_solver.ops.api import solver`.

If you reached this file as MCP resource `llm://python_api_reference`, its sibling resources (`llm://index`, `llm://overview`, `llm://parameters`, and so on) cover the surrounding concepts. Call `resources/list` once and pick the matching URI; the full resource surface (URI scheme, list/read examples, error handling) is documented under the **Resources** section of `llm://integrations`.

## Import and basic use

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

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
- `Curve`
- `Wall`
- `Sphere`
- `ColliderParam`

## Class: Solver

Top-level entry point for the ZOZO Contact Solver.

Available as `solver` when imported via:

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver
```

Scene parameters are accessed via `param` (a `SceneParam` proxy). Groups, pins, and invisible colliders are created via the methods below.

Unrecognized attribute access falls through to `bpy.ops.zozo_contact_solver.<name>()`, so every operator registered under that namespace, including every MCP handler, can be called as a method on `solver`.

```python
solver.param.gravity = (0, 0, -9.8)
group = solver.create_group("Sphere", type="SOLID")
group.add("Sphere")
group.param.solid_density = 1000
```

### param

Type: `SceneParam`

Scene and connection parameter proxy. See `SceneParam`.

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

### create_curve(name: str, \*, bevel_depth: float=0.0, bevel_resolution: int=2, resolution_u: int=4, dimensions: str="3D", clear_existing: bool=True) -> Curve

Start building a multi-spline Bezier curve object.  Returns a `Curve` builder.

**Parameters:**

- **name**: Object name.  When `clear_existing` is true (default) any existing object with this name is removed first.
- **bevel_depth**: Tube radius for visualization (`Curve.bevel_depth`).  `0` leaves the curve as a wireframe.
- **bevel_resolution**: Tube cross-section subdivisions (`Curve.bevel_resolution`).
- **resolution_u**: Spline interpolation resolution (`Curve.resolution_u`).
- **dimensions**: `"3D"` (default) or `"2D"`.
- **clear_existing**: Set `False` to skip the same-name cleanup.

```python
curve = solver.create_curve("Strands", bevel_depth=3e-3)
for points, closed in strands:
    curve.add_spline(points, closed=closed)
obj = curve.finalize()
```

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

The attribute surface is intentionally proxy-based rather than a fixed method list: reads and writes are forwarded to the add-on's scene state and SSH/connection state. Stable day-to-day keys include simulation parameters such as `step_size`, `frame_count`, `frame_rate`, `gravity`, `wind_direction`, `wind_strength`, `air_density`, `air_friction`, `vertex_air_damp`, `project_name`, and connection parameters such as `host`, `port`, `username`, `key_path`, `local_path`, `docker_path`, `ssh_remote_path`, `server_type`, and `container`.

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

### name

Type: `str`

Display name of this group.

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

### set_velocity(object_name: str, direction: tuple[float, float, float], speed: float, frame: int=1) -> Group

Keyframe a velocity on an object assigned to this group.

Appends an entry to the assigned object's `velocity_keyframes` collection. Call once with `frame=1` for an initial-velocity launch; call again with higher `frame` values to build a velocity schedule.

**Parameters:**

- **object_name**: Name of an object already added to this group via `add`.
- **direction**: `(dx, dy, dz)` velocity direction; normalized by the solver before use.
- **speed**: Velocity magnitude in m/s.
- **frame**: Frame at which the keyframe takes effect. `1` (the default) is the initial-velocity slot.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if the object is not assigned to this group, or a keyframe already exists at the requested frame.

```python
ball = solver.create_group("Ball", type="SOLID")
ball.add("Sphere")
ball.set_velocity("Sphere", direction=(1, 0, 0), speed=2.3)
```

### create_pin(object_name: str, vertex_group_name: str, indices: list[int] | None=None) -> Pin

Pin a vertex group (mesh) or set of control points (curve).

**Parameters:**

- **object_name**: Name of the mesh or curve object.
- **vertex_group_name**: For meshes, the name of an existing vertex group on the object.  For curves, the logical name used for the curve's `_pin_<vertex_group_name>` custom property holding the pinned control-point indices.
- **indices**: Control-point indices for curves only.  When given, the curve's `_pin_<vertex_group_name>` property is written before the pin is registered, so the same call both defines and binds the pin.  Must be `None` for meshes (meshes use existing vertex groups).

**Returns:** A `Pin` proxy for the newly created pin.

**Raises:** `ValueError` if the object is missing, not a mesh or curve, the vertex group does not exist on a mesh, the `_pin_<name>` property is missing on a curve and no `indices` were supplied, or `indices` is passed for a mesh.

```python
# Mesh: vertex group must already exist.
pin = group.create_pin("Cloth", "collar")
pin.move_by(delta=(0, 0, 0.2), frame_start=1, frame_end=60)

# Curve: pass control-point indices to define and bind in one call.
rod_pin = rod_group.create_pin(
    "WovenCylinder", "left", indices=[0, 7, 14, 21],
)
```

### get_pins() -> list[Pin]

Return all pins in this group as `Pin` proxies.

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
- **Contact**: `friction`, `use_group_bounding_box_diagonal`, `contact_gap`, `contact_gap_rat`, `contact_offset`, `contact_offset_rat`
- **Strain limit**: `enable_strain_limit`, `strain_limit`
- **Inflation**: `enable_inflate`, `inflate_pressure`
- **Plasticity**: `enable_plasticity`, `plasticity`, `plasticity_threshold`
- **Bend plasticity**: `enable_bend_plasticity`, `bend_plasticity`, `bend_plasticity_threshold`, `bend_rest_angle_source`
- **Shell / solid / rod shape controls**: `bend`, `shrink`, `shrink_x`, `shrink_y`, `length_factor`, `stitch_stiffness`

```python
group.param.friction = 0.5
group.param.shell_density = 1.0
```

## Class: Pin

A pinned vertex group bound to a dynamics group.

Created via `Group.create_pin(object_name, vertex_group_name)`. Every mutating method returns `self` so operations chain.

```python
pin = group.create_pin("Cloth", "hem")
pin.move_by(delta=(0, 0, 1.0), frame_start=1, frame_end=60)
pin.unpin(frame=120)
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

Sets the duration on the underlying pin item so the encoder knows when to stop enforcing the pin constraint.

**Parameters:**

- **frame**: Frame number at which the pin is released.

**Returns:** `self` for chaining.

```python
pin.move_by(delta=(0, 0, 1.0), frame_start=1, frame_end=60)
pin.unpin(frame=120)
```

### delete() -> None

Remove this pin from its group.

**Raises:** `ValueError` if the owning group or pin item can no longer be found (for example, after `solver.clear()`).

## Class: Curve

Builder for a multi-spline Bezier curve object.

Created via `Solver.create_curve`.  Each `add_spline` appends one Bezier spline to the underlying curve datablock; `finalize` links the resulting object into the active scene and returns it.

Pin definition is *not* part of this builder.  Pass the control-point indices to `Group.create_pin` instead, which writes the `_pin_<name>` custom property and registers the pin in one call.

```python
curve = solver.create_curve("WovenCylinder", bevel_depth=3e-3)
for points, closed in strands:
    curve.add_spline(points, closed=closed)
obj = curve.finalize()

rod = solver.create_group("Strands", type="ROD")
rod.add(obj.name)
rod.create_pin(obj.name, "left", indices=left_indices)
```

### name

Type: `str`

Object name this builder will create on `finalize()`.

### add_spline(points, \*, closed: bool=False) -> int

Append a Bezier spline with AUTO handles.

**Parameters:**

- **points**: Iterable of `(x, y, z)` control-point coordinates (a NumPy array of shape `(n, 3)` works).
- **closed**: Set `True` to make the spline cyclic.

**Returns:** Zero-based index of the new spline within this curve.  Use it with `set_material`.

**Raises:** `ValueError` if `points` has fewer than two coordinates.

### set_material(spline_index: int, material: bpy.types.Material) -> Curve

Bind a material to a spline by index.

The material is appended to the curve's slots if it isn't already present.  Pre-existing slots are reused so repeated calls with the same material don't grow the slot list.

**Parameters:**

- **spline_index**: Index returned by `add_spline`.
- **material**: An existing `bpy.types.Material`.  Create it with `bpy.data.materials.new(...)` before calling.

**Returns:** `self` for chaining.

**Raises:** `IndexError` if `spline_index` is out of range.

### finalize() -> bpy.types.Object

Create the `bpy.types.Object`, link it to the scene, and return it.

**Raises:** `RuntimeError` if called more than once on the same builder.

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

Bundled Python API reference synced to `blender_addon/ops/api/__init__.py` and its submodules.
