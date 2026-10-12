# 🖼️ Gallery

These clips are small examples produced with the Blender add-on. They cover
a range of motion types: drape, compression, inflation, twisting, and
contact-driven shape change. To set up a scene like these, start with
[Getting Started](getting_started/index.md) and [Workflow](workflow/index.md).

## Scenes Authored in the Blender UI

These scenes were authored in Blender's UI. Each card links to a
downloadable `.blend` and a video.

<div class="gallery-grid">
  <figure class="gallery-card">
    <img src="../crumple.webp" alt="Sheets of paper crumpling into balls that drop into a wire waste basket." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Crumple</strong>
      <span>Sheets of paper crumpling into balls that drop into a waste basket.</span>
      <span><a href="https://zozo.box.com/s/69ysygaqfud3bba8v3w33eqcbfn6l76u">crumple.blend</a> &middot; <a href="https://zozo.box.com/s/ddrqqq87gpn0mekqx0yaukez87casten">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../curtain.webp" alt="Animated curtains waving in a breeze." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Curtain</strong>
      <span>Curtains waving in a breeze.</span>
      <span><a href="https://zozo.box.com/s/f8775589v2jd3nnmm7dzjrfy44xmbuhl">curtain.blend</a> &middot; <a href="https://zozo.box.com/s/e558genjdno7jz9m0svojte5eco6q7tm">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../kite.webp" alt="Animated kite blown by the wind and caught on tree branches." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Kite</strong>
      <span>A kite blown by the wind and caught on tree branches.</span>
      <span><a href="https://zozo.box.com/s/j5tg9hy7nf1fdea1yg0s6holzwwus77t">kite.blend</a> &middot; <a href="https://zozo.box.com/s/7siwyp04s1vs48znwnt5vx1vhgodyr4h">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../press.webp" alt="A press closing on a prawn that lies on a round disc, then lifting to leave the prawn flattened onto the disc." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Press</strong>
      <span>A press flattens a prawn onto a round disc, and the prawn stays flat after the press lifts.</span>
      <span><a href="https://zozo.box.com/s/n1upezi7j0eufmrsief4qh252of7g1nq">press.blend</a> &middot; <a href="https://zozo.box.com/s/nt8s46e6kke9poruvxmtxv5v56p2ysit">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../puff.webp" alt="Inflated patchwork cushions in a wooden bowl with monkey heads dropping onto them." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Puff</strong>
      <span>Inflated patchwork cushions in a wooden bowl, with monkey heads dropping onto them.</span>
      <span><a href="https://zozo.box.com/s/mfc64djyjyunuhnmn51rm4jexxenx0si">puff.blend</a> &middot; <a href="https://zozo.box.com/s/8dpuoqbg80vxvxwsga36nz1633vx4k6u">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../zebra.webp" alt="Animated striped zebra car sweeping through a grass field." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Zebra</strong>
      <span>A striped zebra car sweeping through a grass field.</span>
      <span><a href="https://zozo.box.com/s/qcos081dolarpczz8mheegvwqalxcnu2">zebra.blend</a> &middot; <a href="https://zozo.box.com/s/rvcqynftqk27fczplafm0wgt95xmhb1k">video</a></span>
    </figcaption>
  </figure>
</div>

## Scenes Built by a Python Script

In these scenes, a Python script builds the simulated objects and sets up
their groups and parameters. Cameras, lighting, and any non-simulated props
are still set up in Blender's UI. Each card links to its script and a video.

To run a script, download it from GitHub, enable the add-on, and open the
file in the Text Editor of the Scripting workspace (**Text > Open**). Run
it with **Text > Run Script**. Each script starts with `solver.clear()`,
which resets the add-on's state in the file and also deletes the animation
data stored on the mesh data of every mesh object in it, such as vertex
keyframes (see [Reset](integrations/python_api.md#reset)). Run a script in a
new file, or in one where you do not need that. A script builds the scene and
stops. Open a connection (see
[Connections](connections/index.md)), then use **Transfer**, **Run**, and
**Fetch All Animation** as for any other scene. The
[`examples/blender`](https://github.com/st-tech/ppf-contact-solver/tree/main/examples/blender)
folder on GitHub holds more scripts.

<div class="gallery-grid">
  <figure class="gallery-card">
    <img src="../cards.webp" alt="A house of cards struck by a sphere flying along the floor." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Cards</strong>
      <span>A house of cards struck by a sphere flying along the floor.</span>
      <span><a href="https://github.com/st-tech/ppf-contact-solver/blob/main/examples/blender/cards.py">cards.py</a> &middot; <a href="https://zozo.box.com/s/3xihxxigc6atsoss4vrh72qppgdvihy5">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../five-twist.webp" alt="Five cylinders arranged in a ring, spun about their own axis and revolved around the world X axis." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Five-twist</strong>
      <span>Five cylinders in a ring, spun about their own axis and then revolved around the world X axis.</span>
      <span><a href="https://github.com/st-tech/ppf-contact-solver/blob/main/examples/blender/five-twist.py">five-twist.py</a> &middot; <a href="https://zozo.box.com/s/71a22jxs5r68wk08j283p4fwj1q4543u">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../noodle.webp" alt="A grid of dangling rod strands settling into a bowl." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Noodle</strong>
      <span>A grid of dangling rod strands settling into a bowl.</span>
      <span><a href="https://github.com/st-tech/ppf-contact-solver/blob/main/examples/blender/noodle.py">noodle.py</a> &middot; <a href="https://zozo.box.com/s/orx2xs715i15l4urmnxns8qs9djhntvs">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../woven.webp" alt="A woven cylinder twisted by spinning and pulling its two open ends." width="640" height="360" loading="lazy">
    <figcaption>
      <strong>Woven</strong>
      <span>A woven cylinder twisted by spinning and pulling its two open ends.</span>
      <span><a href="https://github.com/st-tech/ppf-contact-solver/blob/main/examples/blender/woven.py">woven.py</a> &middot; <a href="https://zozo.box.com/s/16pmmgkpk6r18ae90cwjaknod6cd7y2w">video</a></span>
    </figcaption>
  </figure>
</div>

## Example Script: Drape a Sheet over a Sphere

A Python script run in the Scripting workspace can set up a whole scene,
which suits procedural scene setup and batch variants. The script below
drapes a sheet over a sphere. It creates the geometry, a **Shell** group
with its strain limit and bend parameters set, a **Static** group, a pin,
and two scene parameters, and it stops before the solve. To connect and run
from a script, see the
[Blender Python API](integrations/python_api.md#connection-settings) and
[Running the Simulation](workflow/sim/simulating.md#blender-python-api).

```python
import addon_utils
import importlib
import bpy

# Find the add-on under whichever extension repository Blender installed it
# into, and get the public solver API.
addon = next(m for m in addon_utils.modules() if m.__name__.endswith(".ppf_contact_solver"))
solver = importlib.import_module(f"{addon.__name__}.ops.api").solver

# Reset any prior state.
solver.clear()

# Create a sphere (the static collider) at the origin.
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=0.5, location=(0, 0, 0))
bpy.context.object.name = "Sphere"

# Create a 2x2 sheet just above the sphere as a 64x64 grid.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=64, size=2, location=(0, 0, 0.6))
sheet = bpy.context.object
sheet.name = "Sheet"

# Pin the two corners on the -x edge via a vertex group.
vg = sheet.vertex_groups.new(name="Corners")
corner_indices = [
    i for i, v in enumerate(sheet.data.vertices)
    if v.co.x < -0.99 and abs(abs(v.co.y) - 1.0) < 0.01
]
vg.add(corner_indices, 1.0, "REPLACE")

# Build solver groups.
cloth = solver.create_group("Cloth", type="SHELL")
cloth.add("Sheet")
cloth.param.enable_strain_limit = True
cloth.param.strain_limit_percent = 5.0
cloth.param.bend = 1

ball = solver.create_group("Ball", type="STATIC")
ball.add("Sphere")

# Pin the two sheet corners.
cloth.create_pin("Sheet", "Corners")

# Scene parameters.
solver.param.frame_count = 100
solver.param.step_size = 0.01
```

## UI Look

Two screenshots of scenes set up with the add-on, with its panels in the
3D Viewport sidebar. The full-size versions are linked under each
thumbnail. Panel labels and layout in these screenshots can differ from the
current add-on. The [Tour of the UI](getting_started/tour.md) describes the
current panels.

<table>
<tr>
<td width="50%" valign="top"><img src="../kite-ui-small.jpg" alt="The kite scene set up inside Blender using the add-on."></td>
<td width="50%" valign="top"><img src="../zebra-ui-small.jpg" alt="The zebra scene set up inside Blender using the add-on."></td>
</tr>
<tr>
<td valign="top">Kite scene set up in Blender. <a href="https://zozo.box.com/s/dbtktx71fd0gb4z2trnvew7l3t8fuwxu">(full-size)</a></td>
<td valign="top">Zebra scene set up in Blender. <a href="https://zozo.box.com/s/bkt5uviyqx825os7r854xslurqpxcj2k">(full-size)</a></td>
</tr>
</table>

## The Add-on in Action

The clip below shows the add-on running inside Blender. Geometry and
constraints are authored locally, the simulation runs on a remote
solver, and the resulting frames are fetched back and played in the
viewport.

```{figure} images/blender.webp
:alt: A screencast of the Blender add-on dispatching a scene to a remote solver and playing the fetched results in the Blender viewport.
:width: 100%

The add-on dispatching a scene to a remote solver and playing the
fetched results in the viewport.
```

## From Prompt to Simulation (via MCP)

The add-on also exposes an [MCP server](integrations/mcp.md) so an external
agent can drive Blender and the solver directly. The clip below shows a
natural-language prompt building a bowl-and-spheres scene end to end, with
no clicks in Blender's UI. The clip starts with the connection already open.
To reproduce it, start the server and add it to your MCP client (see
[Starting the Server](integrations/mcp.md#starting-the-server) and
[Adding the Server to an MCP Client](integrations/mcp.md#adding-the-server-to-an-mcp-client)),
and have a solver connection ready (see [Connections](connections/index.md)).

```{figure} images/gallery/mcp.webp
:alt: Codex terminal on the left driving Blender on the right through the MCP server, building a bowl-and-spheres scene from a natural language prompt.
:width: 100%

Codex (left) driving Blender (right) through the add-on's MCP server.
See the [exact prompt used to produce this clip](gallery_mcp_prompt.md).
```

```{figure} images/gallery/drape-over-sphere.webp
:alt: A cloth sheet draped over a sphere, produced from a single natural language prompt through the MCP server.
:width: 100%

The prompt that produced this clip: `drape a sheet over a sphere and make an animation video mp4 render 300 frames`.
```
