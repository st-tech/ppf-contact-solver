# 🖼️ Gallery

These clips are small examples produced with the Blender add-on, covering
a range of motion types: drape, compression, inflation, and contact-driven
shape change. If you want the setup steps behind them, start with
[Getting Started](getting_started/index.md) and [Workflow](workflow/index.md).

<div class="gallery-grid">
  <figure class="gallery-card">
    <img src="../crumple.webp" alt="Animated pieces of paper crumpling under compression.">
    <figcaption>
      <strong>Crumple</strong>
      <span>Compressed pieces of paper crumpling into a tight stack.</span>
      <span><a href="https://zozo.box.com/s/69ysygaqfud3bba8v3w33eqcbfn6l76u">crumple.blend</a> &middot; <a href="https://zozo.box.com/s/ddrqqq87gpn0mekqx0yaukez87casten">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../curtain.webp" alt="Animated curtains waving in a breeze.">
    <figcaption>
      <strong>Curtain</strong>
      <span>Curtains waving in a breeze.</span>
      <span><a href="https://zozo.box.com/s/f8775589v2jd3nnmm7dzjrfy44xmbuhl">curtain.blend</a> &middot; <a href="https://zozo.box.com/s/e558genjdno7jz9m0svojte5eco6q7tm">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../kite.webp" alt="Animated kite blown by the wind and caught on tree branches.">
    <figcaption>
      <strong>Kite</strong>
      <span>A kite blown by the wind and caught on tree branches.</span>
      <span><a href="https://zozo.box.com/s/j5tg9hy7nf1fdea1yg0s6holzwwus77t">kite.blend</a> &middot; <a href="https://zozo.box.com/s/7siwyp04s1vs48znwnt5vx1vhgodyr4h">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../press.webp" alt="Animated prawn and flower ball pressed together into a cracker.">
    <figcaption>
      <strong>Press</strong>
      <span>A prawn pressed permanently together with a flower ball to form a cracker.</span>
      <span><a href="https://zozo.box.com/s/n1upezi7j0eufmrsief4qh252of7g1nq">press.blend</a> &middot; <a href="https://zozo.box.com/s/nt8s46e6kke9poruvxmtxv5v56p2ysit">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../puff.webp" alt="Animated inflated cushion with multiple monkey heads resting on top.">
    <figcaption>
      <strong>Puff</strong>
      <span>An inflated cushion with multiple monkey heads resting on top.</span>
      <span><a href="https://zozo.box.com/s/mfc64djyjyunuhnmn51rm4jexxenx0si">puff.blend</a> &middot; <a href="https://zozo.box.com/s/8dpuoqbg80vxvxwsga36nz1633vx4k6u">video</a></span>
    </figcaption>
  </figure>
  <figure class="gallery-card">
    <img src="../zebra.webp" alt="Animated striped zebra car sweeping through a grass field.">
    <figcaption>
      <strong>Zebra</strong>
      <span>A striped zebra car sweeping through a grass field.</span>
      <span><a href="https://zozo.box.com/s/qcos081dolarpczz8mheegvwqalxcnu2">zebra.blend</a> &middot; <a href="https://zozo.box.com/s/rvcqynftqk27fczplafm0wgt95xmhb1k">video</a></span>
    </figcaption>
  </figure>
</div>

## UI Look

A couple of static screenshots of the add-on running inside Blender. The full-size versions are linked under each thumbnail.

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

## From a Python Script to Simulation

You can also drive the entire pipeline from a Python script inside Blender's
scripting editor. This is handy for procedural scene setup and batch variant
generation. Below is a full example that drapes a sheet over a sphere:

```python
import bpy
from zozo_contact_solver import solver

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
cloth.param.strain_limit = 0.05
cloth.param.bend = 1

ball = solver.create_group("Ball", type="STATIC")
ball.add("Sphere")

# Pin the two sheet corners.
cloth.create_pin("Sheet", "Corners")

# Scene parameters.
solver.param.frame_count = 100
solver.param.step_size = 0.01
```

```{figure} images/screenshots/python-scripting.jpg
:alt: The drape script running inside Blender's Scripting workspace, with the simulated sheet draped over the sphere in the viewport on the left and the script in the text editor on the right.
:width: 100%

The script above running inside Blender's Scripting workspace.
[(full-size)](https://zozo.box.com/s/m86w3jyprvz5dug2pr7k73bozdbgldfl)
```

## From Prompt to Simulation (via MCP)

The add-on also exposes an [MCP server](integrations/mcp.md) so an external
agent can drive Blender and the solver directly. The clip below shows a
natural language prompt building a bowl-and-spheres scene end to end, with
no UI clicks. See [MCP Server](integrations/mcp.md) for the full setup.

```{figure} images/gallery/mcp.webp
:alt: Codex terminal on the left driving Blender on the right through the MCP server, building a bowl-and-spheres scene from a natural language prompt.
:width: 100%

Codex (left) driving Blender (right) through the add-on's MCP server.
See the [exact prompt used to produce this clip](gallery_mcp_prompt.md).
```

```{figure} images/gallery/drape-over-sphere.webp
:alt: A cloth sheet draped over a sphere, produced from a single natural language prompt through the MCP server.
:width: 100%

A prompt: drape a sheet over a sphere and make an animation video mp4 render 300 frames.
```
