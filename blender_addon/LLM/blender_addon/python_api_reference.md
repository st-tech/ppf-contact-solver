# Blender Python API reference

This document tracks the bundled Python API surface exported from `blender_addon/ops/api/__init__.py` and its submodules. It catalogs every public class, method, property, and attribute on the ZOZO Contact Solver Python surface so an LLM can script the add-on after `from bl_ext.user_default.ppf_contact_solver.ops.api import solver`.

If you reached this file as MCP resource `llm://python_api_reference`, its sibling resources (`llm://index`, `llm://overview`, `llm://parameters`, and so on) cover the surrounding concepts. Call `resources/list` once and pick the matching URI; the full resource surface (URI scheme, list/read examples, error handling) is documented under the **Resources** section of `llm://integrations`.

## Import and basic use

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

# Scene parameters
solver.param.gravity = (0, 0, -9.8)

# Mesh preparation: the scan returns its report, the repairs chain
if solver.scan_meshes("Plane")["Plane"]["n_errors"]:
    solver.merge_by_distance("Plane").delete_duplicate_faces("Plane")

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
- **type**: One of `"SOLID"`, `"SHELL"`, `"ROD"`, `"STATIC"`, `"PDRD"`, `"SAND"`.

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

### scan_meshes(*object_names: str, merge_threshold: float=1e-4, area_eps: float=0.0) -> dict[str, dict]

Report the geometry the solver rejects, modifying nothing.

**Parameters:**

- **\*object_names**: One or more mesh object names.
- **merge_threshold**: Vertices closer together than this, in local units, count as near-coincident. Matches Blender's Merge by Distance default.
- **area_eps**: Faces at or below this area, in local units squared, count as degenerate. `0.0` reports only exactly zero-area faces.

**Returns:** `{object_name: report}`. Each report carries `n_verts`, `n_polys`, the two thresholds it was scanned at, `n_errors`, `n_notes`, a `defects` dict keyed by defect name, and `dependents`, what a vertex-count change on that object would affect. Every defect entry carries a `count`. Three of them, `near_duplicates`, `degenerate_faces` and `resplittable`, short-circuit to a bare `{"count": 0}` and carry nothing else at zero; the other five carry their extra fields unconditionally, empty when the count is zero. Guarding on `count` before reading an extra field is correct either way. A note is not a defect: an open quad panel is an ordinary cloth mesh.

**Raises:** `ValueError` if a name is missing, is not a mesh, or an object produced no report, which is how an object the active view layer cannot select surfaces here.

```python
report = solver.scan_meshes("Shirt")["Shirt"]
if report["defects"]["near_duplicates"]["count"]:
    solver.merge_by_distance("Shirt")
```

The `defects` keys, what each carries beyond `count`, and whether those fields survive a zero count:

| Key                | Extra fields                                              | At count 0 |
| ------------------ | --------------------------------------------------------- | ---------- |
| `near_duplicates`  | `min_dist` (local), `min_dist_world`, `preview` (up to 8 index pairs), `verts` | absent |
| `isolated_verts`   | `preview` (up to 8 indices), `verts`                      | present, empty |
| `hanging_verts`    | `preview` (up to 8 indices), `verts`                      | present, empty |
| `degenerate_faces` | `preview` (up to 8 face indices)                          | absent |
| `duplicate_faces`  | none                                                       | n/a |
| `surface`          | `boundary`, `non_manifold`, `bad_winding`                 | present, zeroed |
| `resplittable`     | `max_fold_deg`, `past_flip`                               | absent |
| `linked_duplicate` | `siblings` (names of the objects sharing the datablock)   | present, empty |

`n_errors` sums the near-coincident pairs, isolated and hanging vertices, duplicate and degenerate faces, linked duplicates, and inconsistently wound edges: what stops a run. `n_notes` sums boundary edges, non-manifold edges, and faces with more than three corners, all of which are normal on cloth.

The seven repairs below each take the same `*object_names` and each returns `self`, so a preparation pass chains. None of them takes an `acknowledge` flag: the explicit method call is the confirmation the panel's dialog asks for, exactly as `solver.clear()` needs no prompt.

### merge_by_distance(*object_names: str, merge_threshold: float=1e-4, clear_stale_caches: bool=True) -> Solver

Weld near-coincident vertices. **Changes the vertex count.**

A surviving pair sits far inside the contact gap, where the cubic barrier's dynamic stiffness contributes Hessian entries many orders of magnitude larger than the rest of the row, so the fp32 Newton matrix loses rank and the run stops on the solver's SPD guard naming no geometry. Welding is what makes such a mesh simulable.

**Parameters:**

- **\*object_names**: One or more mesh object names.
- **merge_threshold**: Weld vertices closer together than this, in local units.
- **clear_stale_caches**: Delete the display and capture caches the count change invalidates, which is what the panel's confirmation does. Pass `False` to keep them, and expect the viewport overlay to read data sized for the old vertex count until the next Transfer rewrites it.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

```python
solver.merge_by_distance("Shirt", merge_threshold=1e-4)
```

### remove_loose_vertices(*object_names: str, clear_stale_caches: bool=True) -> Solver

Delete vertices that belong to no face. **Changes the vertex count.**

The solver averages a vertex's contact parameters over its incident faces and aborts when it has none. Loose edges go with the vertices they connect. Pinned vertices are exempt, since a pin holds them regardless, and a SAND particle mesh is skipped outright: every grain center is legitimately faceless.

**Parameters:**

- **\*object_names**: One or more mesh object names.
- **clear_stale_caches**: As for `merge_by_distance`.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

```python
solver.remove_loose_vertices("Shirt")
```

### dissolve_degenerate_faces(*object_names: str, merge_threshold: float=1e-4, clear_stale_caches: bool=True) -> Solver

Collapse zero-area faces and zero-length edges. **Changes the vertex count.**

A face with no area has no defined normal, so the contact normal and the bending hinge built on it are both undefined.

**Parameters:**

- **\*object_names**: One or more mesh object names.
- **merge_threshold**: Edges shorter than this, in local units, are collapsed.
- **clear_stale_caches**: As for `merge_by_distance`.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

```python
solver.dissolve_degenerate_faces("Shirt")
```

### delete_duplicate_faces(*object_names: str) -> Solver

Delete faces that repeat an existing face's vertex set.

Two faces on the same vertices contribute their contact and elastic terms twice. The vertex count is unchanged, so no cache is invalidated. Run Transfer again afterward.

**Parameters:**

- **\*object_names**: One or more mesh object names.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

### triangulate_for_solver(*object_names: str) -> Solver

Triangulate every face with more than three corners.

Transfer triangulates on its own at encode time, so this is for when the triangulation has to be visible and stable in the viewport: with the diagonals fixed in the mesh, Blender has none left to re-pick from the deformed shape, and the displayed surface stops drifting from the simulated one. The vertex count is unchanged, so no cache is invalidated. For a mesh whose symmetry matters under bending, use `symmetric_triangulate` instead.

**Parameters:**

- **\*object_names**: One or more mesh object names.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

### recalculate_normals_outside(*object_names: str) -> Solver

Make face winding consistent and outward.

Inconsistent winding flips the normal a face contributes, which the contact and inflate terms read. The vertex count is unchanged, so no cache is invalidated, but the encoder captures winding at Transfer time, so run Transfer again afterward.

**Parameters:**

- **\*object_names**: One or more mesh object names.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

### symmetric_triangulate(*object_names: str) -> Solver

Triangulate by poking each face, keeping the mesh symmetric. **Adds one vertex per face.**

A single-diagonal triangulation breaks a mirror-symmetric mesh's symmetry, which shows up as a lopsided drape under bending. Poking inserts a center vertex and fans the face into triangles around it instead.

The added vertices invalidate the PC2 display cache and any captured deformation on the object exactly as the count-changing repairs above do, and this method deletes neither: run Transfer to rewrite the display cache, and Capture Deformation (or `recapture_all_deformations`) to re-take the captures.

**Parameters:**

- **\*object_names**: One or more mesh object names.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if a name is missing, is not a mesh, or the object is outside the active view layer.

### convert_to_particle_mesh(object_name: str, grain_radius: float, extra_spacing: float=0.0, rng_seed: int=0) -> int

Replace a solid mesh with the grain cloud a SAND group simulates.

Destructive: the faces are discarded and the object becomes a faceless mesh of loose vertices carrying a render-only Particle Mesh modifier. The grain count is not chosen, it is whatever fills the volume at the given separation, which is why it is the return value.

`grain_radius` is stamped onto the object and is what the encoder reads, in preference to the group's `sand_grain_radius`. The non-overlapping seed spacing derives from it and it is also the contact skin, so it is locked once the object is converted and the panel shows it read-only. Pick it before converting.

**Parameters:**

- **object_name**: Name of a mesh object that has faces and is not already a particle mesh.
- **grain_radius**: Physical grain radius, which is also the contact skin.
- **extra_spacing**: Gap added between grains beyond touching. `0.0` packs them as densely as non-overlap allows.
- **rng_seed**: Seed for the Poisson-disk seeding, for a repeatable cloud.

**Returns:** The number of grains seeded.

**Raises:** `ValueError` if the object is missing, is not a mesh, has no faces, is already a particle mesh, `grain_radius` is not positive, or no grain fits inside the mesh.

```python
n_grains = solver.convert_to_particle_mesh("Pile", grain_radius=0.01)
sand = solver.create_group("Sand", type="SAND")
sand.add("Pile")
# Grams per grain, not kilograms: a 1 cm-radius grain of sand is a few grams.
sand.param.sand_particle_mass = 5.0
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

### recapture_all_deformations() -> Solver

Re-capture every deforming STATIC collider and every animated pin.

One pass over the whole scene instead of one Capture Deformation per object. The statics are captured first and the pins after, since the two share the depsgraph and cannot run at once.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if there is nothing to re-capture, or a capture or bake is already running.

NOTE: the captures advance on Blender's event loop. This returns once the first one has started, and the rest complete over later ticks. A script that keeps running on the same tick holds the loop and blocks them, so schedule whatever depends on the caches and gate it on `is_capture_running`. A Blender started with `--background` runs no ticks and captures nothing.

```python
import bpy

def transfer_when_captured():
    if solver.is_capture_running():
        return 0.1  # come back in 100 ms
    solver.transfer_data()
    return None

solver.recapture_all_deformations()
bpy.app.timers.register(transfer_when_captured)
```

### clear_all_deformations() -> Solver

Delete every captured deformation cache in the scene.

Covers the STATIC-collider deform caches and the animated-pin captures across the active groups, plus any cache orphaned by an object that was deleted or taken out of its group, which nothing else reaches. The objects keep their deformers, so `recapture_all_deformations` rebuilds what this removes.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if there is no captured cache to clear, or a capture or bake is already running.

### is_capture_running() -> bool

`True` while a deformation capture is in flight.

Covers both phases of `recapture_all_deformations`, the STATIC-collider captures and the pin captures, so a script waits on the whole pass with one predicate. Read it from a timer or a handler: a capture advances only when the script has handed control back to Blender's event loop.

### export_usd(filepath: str) -> Solver

Export the simulated mesh sequence as a USD cache.

A lighter result than baking shape keys: the deformation is sampled per frame straight from the solver cache into a file other DCC tools play back, and the scene itself is left untouched.

Every frame must be fetched before the export runs, and the export refuses while a run, bake or capture is in flight, and outside Object Mode. Rod and curve objects are not carried by this format; `get_unexportable_curves` names the ones that will be left out.

**Parameters:**

- **filepath**: Destination path, taken as written once a `//` blend-relative prefix is resolved. The suffix is what picks the USD flavor, so give one of `.usdc` (crate), `.usda` (ASCII), `.usd`, or `.usdz` (package). The parent directory must exist.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if frames are unfetched, another solver activity is in progress, the scene is in Edit or Sculpt mode, there is no simulated mesh sequence, the destination directory does not exist, or the export did not complete.

```python
solver.export_usd("/tmp/drape.usdc")
```

### export_alembic(filepath: str) -> Solver

Export the simulated mesh sequence as an Alembic (ABC) cache.

Same preconditions and same curve exclusion as `export_usd`.

**Parameters:**

- **filepath**: Destination `.abc` path, taken as written once a `//` blend-relative prefix is resolved. The parent directory must exist.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if frames are unfetched, another solver activity is in progress, the scene is in Edit or Sculpt mode, there is no simulated mesh sequence, the destination directory does not exist, or the export did not complete.

### get_unexportable_curves() -> list[str]

Names of the simulated curves a cache export leaves out.

A rod deforms through a frame-change handler rather than through the cache modifier the exporters sample, so a CURVE object carrying a solver cache cannot be written to USD or Alembic. Bake Animation is the route that carries one.

```python
missing = solver.get_unexportable_curves()
if missing:
    print("not exported:", ", ".join(missing))
solver.export_usd("/tmp/drape.usdc")
```

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

### slot

Type: `int`

The `object_group_N` slot index this group occupies. This is the index every group operator addresses, resolved through `object_group_{index}`.

It is **not** `ObjectGroup.index`, which numbers the active groups consecutively for display: the two agree only while every slot below this one is active, so a scene that has ever deleted a group can have them disagree.

**Raises:** `ValueError` if no slot holds this group's UUID.

```python
group = solver.create_group("Shirt", type="SHELL")
print(f"stored in object_group_{group.slot}")
```

### type

Type: `str`

Dynamics type of this group: one of `"SOLID"`, `"SHELL"`, `"ROD"`, `"STATIC"`, `"PDRD"`, `"SAND"`. Set at creation through `Solver.create_group`.

```python
for g in solver.get_groups():
    if g.type == "ROD":
        g.param.length_factor = 0.97
```

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

### set_velocity(object_name: str, direction: tuple[float, float, float], speed: float, frame: int=1, angular_axis: int|str="PC3", angular_speed: float=0.0, angular_axis_custom: tuple[float, float, float]=(0.0, 0.0, 1.0), enable_translational: bool=True, enable_angular: bool|None=None) -> Group

Keyframe a velocity on an object assigned to this group.

Appends an entry to the assigned object's `velocity_keyframes` collection. Call once with `frame=1` for an initial-velocity launch; call again with higher `frame` values to build a velocity schedule.

**Parameters:**

- **object_name**: Name of an object already added to this group via `add`.
- **direction**: `(dx, dy, dz)` velocity direction; normalized by the solver before use.
- **speed**: Velocity magnitude in m/s.
- **frame**: Frame at which the keyframe takes effect. `1` (the default) is the initial-velocity slot.
- **angular_axis**: Axis to spin about (SOLID, SHELL, PDRD only). `"PC1"`/`"PC2"`/`"PC3"` (principal axes resolved dynamically from the geometry), `"X"`/`"Y"`/`"Z"` (fixed world axes), or `"CUSTOM"` (the `angular_axis_custom` vector). Ints `0`/`1`/`2` map to PC1/PC2/PC3. Ignored when `angular_speed == 0`.
- **angular_speed**: Signed spin speed in degrees per second (0 = no spin).
- **angular_axis_custom**: World `(x, y, z)` axis used when `angular_axis == "CUSTOM"` (normalized before use).
- **enable_translational**: Overwrite the translational velocity at this frame (False leaves translation alone, e.g. a pure spin).
- **enable_angular**: Overwrite the angular velocity at this frame. Defaults to True when `angular_speed` is non-zero, else False.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if the object is not assigned to this group, or a keyframe already exists at the requested frame.

```python
ball = solver.create_group("Ball", type="SOLID")
ball.add("Sphere")
ball.set_velocity("Sphere", direction=(1, 0, 0), speed=2.3)
```

### set_hinge(object_name: str, pca_axis: int=2, enable: bool=True) -> Group

Pin a PDRD body assigned to this group as a hinge (per object).

Locks the body's position and restricts its rotation to a single principal (PCA) axis of its rest shape, the building block for gears: hinge each gear to its axle and let tooth contact transmit the torque, so meshing gears counter-rotate with no explicit gear-ratio constraint. The group must be of type `PDRD`, and the setting is per object, so different bodies in the same group can be hinged on different axles.

**Parameters:**

- **object_name**: Name of an object already added to this group via `add`.
- **pca_axis**: Which principal axis of the rest shape is the free axle: `0` (largest extent), `1` (middle), or `2` (thinnest, the usual axle for a flat gear or disk). Defaults to `2`.
- **enable**: `True` (the default) sets the hinge; `False` clears it and lets the body move freely.

**Returns:** `self` for chaining.

**Raises:** `ValueError` if the group is not `PDRD`, the object is not assigned to it, or `pca_axis` is not in `{0, 1, 2}`.

```python
gears = solver.create_group("Gears", type="PDRD")
gears.add("GearA")
gears.set_hinge("GearA", pca_axis=2)
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

- **Solver model**: `solid_model`, `shell_model`, `rod_model` (`rod_model` currently accepts only `"ARAP"`: the enum has a single item and ROD groups force-pin it to ARAP)
- **Density**: `solid_density`, `shell_density`, `rod_density`
- **Young's modulus**: `solid_young_modulus`, `shell_young_modulus`, `rod_young_modulus`
- **Poisson ratio**: `solid_poisson_ratio`, `shell_poisson_ratio`
- **Contact**: `friction`, `use_group_bounding_box_diagonal`, `contact_gap`, `contact_gap_rat`, `contact_offset`, `contact_offset_rat`
- **Strain limit**: `enable_strain_limit`, `strain_limit_percent`
- **Inflation**: `enable_inflate`, `inflate_pressure`
- **Plasticity**: `enable_plasticity`, `plasticity`, `plasticity_threshold`
- **Bend plasticity**: `enable_bend_plasticity`, `bend_plasticity`, `bend_plasticity_threshold`, `bend_rest_angle_source`
- **Reference rest angle (SHELL / ROD)**: `bend_rest_from_reference` (group master toggle). The per-object reference itself (which object, and its picked reference object) is **not** a group material: it lives on the assigned object (`bend_ref_enable`, `bend_ref_uuid`) and is set from the UI eyedropper (`object.pick_bend_reference` / `object.clear_bend_reference`), not via the group param API. When enabled with a valid reference, that object's bending rest angle (shell hinge dihedral, or rod interior-vertex bend angle) is computed from the reference geometry, overriding `bend_rest_angle_source` for that object. Mesh references are modifier-evaluated; curve-rod references are sampled at the control-point level.
- **Shell / solid / rod shape controls**: `bend`, `shrink`, `shrink_x`, `shrink_y`, `length_factor`, `stitch_stiffness`. `shrink_x` / `shrink_y` are the Shell pair, `shrink` is the single Solid factor, and `length_factor` is the Rod rest-length scale (the panel labels it Shrink). `bend` is read by both Shell and Rod; each type derives its own scaling from it, and a rod's is normalized against a 1 cm reference segment so it does not vary with segment count. Because `length_factor` scales the rest length that normalization divides by, halving it makes a rod about four times stiffer in bending.
- **PDRD-specific**: `pdrd_density` (kg/m^3, volumetric). The PDRD hinge joint is per-object, set via `set_hinge` (not a group material).
- **SAND-specific**: `sand_grain_radius`, `sand_particle_mass` (grams), `sand_friction`. Write `sand_grain_radius` only before the group's objects are converted: `convert_to_particle_mesh` stamps the radius it seeded onto each object, and that stamped value is what the panel shows and the encoder ships.

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

A pin is one of exactly two kinds, and there is nothing in between and no stiffness scalar to tune:

- **Exact hold** (the default, and what a freshly created pin is). The solver removes the pinned vertices' degrees of freedom from its Newton system and prescribes their positions, so they track their target to round-off and never yield to contact or elasticity. `move_by`, `spin` and `scale` script that prescribed motion; `unpin` ends it at a chosen frame.
- **Soft spring** (`pull(strength)`). The vertices carry a restoring force toward their target and are free to be pushed off it by contact and elasticity. This is the only compliant hold, so it is what to reach for when a pin should give.

`torque` is a force rather than a prescribed motion, so it cannot share a pin with `move_by` / `spin` / `scale`; transfer raises `ValueError` on a pin that carries both.

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

Sets the duration on the underlying pin item, which is what the encoder ships as the frame the pin constraint stops being enforced.

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
