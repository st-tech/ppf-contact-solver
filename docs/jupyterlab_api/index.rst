📓 JupyterLab Python API
===========================

This section documents the ``frontend`` Python package, the API you
import inside a JupyterLab notebook running on the solver host to drive
the solver directly. Once a Blender scene has been transferred, the
same project can be run, previewed, parameter-swept, and inspected
entirely from a notebook, with Blender closed.

For when to reach for JupyterLab and how it fits into the end-to-end
loop, see :doc:`/blender_addon/workflow/sim/jupyterlab`. This section
is the reference material the notebook calls into.

Typical entry point:

.. code-block:: python

   from frontend import BlenderApp

   app = BlenderApp.open("my-project")  # attach to the transferred project
   app.scene.report()                   # inspect what was transferred
   app.session.run()                    # start the solver, or attach to it
   app.session.stream()                 # tail the solver's live output

``BlenderApp.open()`` is the cell the add-on writes into the notebook it
generates, and ``app.scene`` / ``app.session`` are the already-built
:class:`frontend.FixedScene` and :class:`frontend.FixedSession`. A
session snapshots its parameters when it is built, so overriding one
means going through :class:`frontend.App` and setting it before the
build: ``app = App.load("my-project")``, then
``session = app.session.create(scene)``,
``session.param.set("dt", 0.01)``, then ``session.build().run()``.

Which backend a notebook runs on
--------------------------------

A tree or distribution can hold several solver builds at once -- CUDA,
ROCm, Metal, and the portable CPU build, one per directory. The panel's
**Compute Device** and **GPU Backend** rows answer that question for a
connection; a notebook answers it with four :class:`frontend.App` static
methods:

.. code-block:: python

   from frontend import App

   App.list_backends()        # {'cuda': '.../target/cuda/release', ...}
   App.probe_backend("rocm")  # can this machine run that build?
   App.get_backend()          # what the next run will use
   App.set_backend("rocm")    # pin it; None returns to the automatic rule

The same four are also exported as module-level functions
(``frontend.set_backend`` and so on) for notebooks written against them.

The automatic rule is the same one the add-on applies: the first of
CUDA, ROCm, and Metal whose own solver reports a usable device, and the
CPU build if none does, with one line printed naming what each GPU
backend said. An **explicit** choice never falls back -- naming a GPU
backend on a machine that cannot run it raises rather than quietly
running on the CPU. ``CARGO_TARGET_DIR``, which the add-on sets when it
launches a server, counts as an explicit choice for the same reason.
See :doc:`module_reference` for the full signatures, and
:ref:`choosing-the-build` for the add-on's side of it.

Letting objects pass through each other
---------------------------------------

By default the solver keeps every pair of surfaces apart and refuses a
scene whose geometry already overlaps. Four per-object parameters,
float-encoded booleans set with ``object.param.set(key, 1.0)``, let
chosen pairs pass through each other instead: an allowed pair gets no
contact force, is not held apart during a step, and is never reported
as an intersection.

- ``allow-self-intersection``: the object passes through itself.
- ``allow-inter-object-intersection``: the object passes through every
  other object, static colliders included.
- ``allow-inter-group-intersection``: the object passes through every
  object in a different group, while objects of its own group still
  collide with it. :meth:`frontend.Object.group` names the group an
  object belongs to; objects never given one share a default group, and
  a static collider counts as another group from every object.
- ``allow-existing-intersection``: only the pairs the object STARTS
  overlapping, or closer than their contact offsets, pass through each
  other for the whole run, together with the elements sharing a vertex
  with them; every other pair keeps full contact, and a new intersection
  still stops the run. It tolerates a tangle and does not untangle it.
  Not available on a points (sand) object, which is refused by name.
  After ``scene.build()``, ``fixed.start_links`` holds the vertex links
  the build made and ``fixed.start_link_exemptions()`` describes the
  exempted pairs.

For the cross-object parameters and ``allow-existing-intersection`` one
side of a pair is enough. A pin
does the same for the elements it holds completely when it is created
with ``obj.pin(ind, allow_intersection=True)``, or later with
``holder.set_allow_intersection(True)``. None of these affects the
invisible walls and spheres.

.. code-block:: python

   # The shirt and jacket collide with each other and pass through the body.
   scene.add("shirt").group("garments").param.set(
       "allow-inter-group-intersection", 1.0)
   scene.add("jacket").group("garments")
   scene.add("body").group("character").pin()

See :doc:`material_parameters` for each parameter's full description,
and :ref:`allow-intersections-settings` for the same settings in the
Blender add-on.

Stitching two objects together
------------------------------

The add-on's snap-and-merge pairs are cross-object stitches, and a
notebook adds the same with :meth:`frontend.Scene.cross_stitch`. Each row
joins a point on one object to a point on another, each written as three
vertices of its own object and their barycentric weights; a point at one
vertex ``i`` is ``[i, i, i]`` with weights ``[1, 0, 0]``.

.. code-block:: python

   # Sleeve cuff vertices stitched to the closest points of a fixed shirt.
   scene.add("shirt").pin()
   scene.add("sleeve")
   ind = [[i, i, i, *tri] for i, tri in zip(cuff, shirt_tris)]
   w = [[1, 0, 0, *bary] for bary in shirt_barys]
   scene.cross_stitch("sleeve", "shirt", ind, w, stiffness=10.0)

A stitch is soft, not a weld: it pulls the two points to contact distance
and contact still applies. An object a stitch names is always simulated,
so a fully pinned one stays held at rest by its pins rather than becoming a
static collision mesh.

Each call is checked when it is made, and a violation raises
``ValueError`` naming it:

- ``source`` and ``target`` name two different objects, each a surface, a
  tetrahedral mesh, or a rod. A seam within one object is
  :meth:`frontend.Object.stitch`.
- ``ind`` and ``w`` are both ``(K, 6)`` with ``K > 0``. Slots 0 to 2 are
  vertices of ``source`` and slots 3 to 5 vertices of ``target``, each in
  its own object's vertex numbering (a tetrahedral object is numbered by
  its tetrahedral mesh).
- Each side's three weights are finite, non-negative, and sum to one.
- ``stiffness`` is finite and non-negative; ``0`` keeps the stitches and
  applies no force.

:meth:`frontend.Scene.build` also refuses a stitch that names a PDRD body,
which moves as one rigid transform that a stitch cannot hold.

A stitch asset for :meth:`frontend.Object.stitch`, registered with
``app.asset.add.stitch(name, (Ind, W))``, takes either of two layouts,
one row per stitch. Four columns join a vertex ``Ind[:, 0]`` to the point
of the vertices ``Ind[:, 1:4]`` weighted by ``W[:, 1:4]``. Six columns join
two barycentric points, ``Ind[:, 0:3]`` / ``W[:, 0:3]`` and
``Ind[:, 3:6]`` / ``W[:, 3:6]``, for a seam where neither end is a vertex.
``Ind`` and ``W`` must have the same shape.

Pin settings
------------

A :class:`frontend.PinHolder`, returned by ``obj.pin(...)``, carries every
setting the add-on's pin panel offers:

- ``holder.set_allow_intersection(True)`` is the add-on's **Allow
  Intersections Here** (see above).
- ``holder.torque(magnitude, axis_component, hint_vertex)`` requires
  ``hint_vertex``, a vertex of the pinned object in its own numbering. A
  principal axis has no sign of its own, so the solver points the torque
  axis from the pinned vertices' centroid toward that vertex, and that is
  what decides which way a positive ``magnitude`` turns the object. A
  missing hint, or one that is not a vertex of the object, raises
  ``ValueError``.
- ``holder.set_pin_group_id(name)`` puts holders into one named group.
  Holders that share a name and each carry a torque are one torque group:
  the torque turns about the centroid and principal axes of all their
  vertices together, with the magnitude, axis component, and hint vertex
  of the group's first holder. The name must be a non-empty string.
- ``holder.spin(center, axis, angular_velocity)`` turns the other way for
  a negative ``angular_velocity``. An ``axis`` of zero length with a
  nonzero angular velocity raises ``ValueError``, because there is no axis
  to turn about.
- ``holder.track_rest_shape()`` is the add-on's **Track Rest-Pose
  Deformation**: the object's rest shape follows the holder's operations,
  so the body settles into the prescribed deformation rather than resisting
  it. Both a fixed pin and a ``pull`` pin can track. :meth:`frontend.Scene.build`
  refuses tracking when the tracking pins of an object do not together hold
  every one of its vertices, when a tracking pin carries no operation, and
  when the object also has ``plasticity`` or ``bend-plasticity`` above zero,
  animated, or mapped, since both rewrite the rest shape.

.. code-block:: python

   # Twist a post about its long axis, oriented toward its topmost vertex.
   post = scene.add("post")
   top = int(post.vertex(False)[:, 1].argmax())
   post.pin().torque(magnitude=0.5, axis_component=0, hint_vertex=top)

Animating parameters over the run
---------------------------------

``obj.set_param_anim(key, values)`` gives a material parameter one value
per entry of the scene's ``set_param_anim_times(...)``. On a tetrahedral
object only ``friction``, ``contact-gap`` and ``contact-offset`` can be
animated, because they live on its surface triangles; its elastic
parameters live on its tetrahedra, which hold one value for the whole run,
and animating one raises ``ValueError``. The add-on applies the same rule
to a **Solid** group's keyframes.

``session.param.dyn(key)`` selects a scene parameter's schedule and places
the time cursor on the last time already scheduled for that key (``0``
for a key with none), so a second ``dyn`` on the same key continues the
schedule. ``time(t)`` refuses a ``t`` that is not later than the cursor,
so no entry lands before one already scheduled.

.. code-block:: python

   g = session.param.get("gravity")
   session.param.dyn("gravity").time(1.0).hold().time(1.5).change([-x for x in g])
   # A second dyn on the same key continues from t = 1.5.
   session.param.dyn("gravity").time(2.0).change(g)

External force fields
---------------------

``scene.force_field`` adds an acceleration (m/s², like gravity) to every
free vertex, evaluated once per step at the position the vertex starts
that step from. Two sources can be mixed.

An exact Python function, compiled to run on the solver, evaluated at
each vertex's own position with no resolution and no domain:

.. code-block:: python

   import math

   def eval(x, y, z, t):
       r = math.sqrt(x * x + z * z) + 1e-6
       return (-z / r, 0.0, x / r)     # a swirl about the Y axis

   scene.force_field.script(eval)      # or the source text as a string

The function may use arithmetic, comparisons, ``if`` / ``else``, local
variables, ``for i in range(<number>)``, ``abs``, ``min``, ``max``,
``float``, the ``math`` functions, and two built-in noises:
``noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)``,
a smooth scalar in about [-1, 1], and ``curl_noise(...)`` with the same
arguments, a swirling vector field with no sources or sinks, returned as
three values (``return curl_noise(...)`` or ``cx, cy, cz = curl_noise(...)``).
``octaves`` (1 to 8, written in the script) adds finer detail and
``seed`` picks another pattern; with ``time=t`` the pattern changes in
place ``frequency`` times per second and fades as ``exp(-decay * t)``:

.. code-block:: python

   def gusts(x, y, z, t):
       return curl_noise(2.0 * x, 2.0 * y, 2.0 * z, octaves=3, seed=7,
                         time=t, frequency=1.0, decay=0.2)

It must
return ``(ax, ay, az)`` on every path. Anything else is refused when
``script`` is called, with its line. ``scene.force_field.builtins()``
prints every function, constant and construct a script may use. The
solver runs it in single precision. ``frontend.noise`` and ``frontend.curl_noise`` are the same
functions for NumPy, for sampling grids.

Scripts add up, and every source reaches every object unless
``groups=[...]`` names the groups (``Object.group(label)``) it reaches:

.. code-block:: python

   scene.add("sheet").group("flags")
   scene.force_field.script(eval, groups=["flags"])

A sampled grid, trilinear in space and linear in time, zero outside its
box:

.. code-block:: python

   scene.force_field.grid(values, box_min=(-1, 0, -1), box_max=(1, 2, 1),
                          times=[0.0, 1.0])     # values: (T, D, H, W, 3)
   scene.force_field.sample(fn, box_min, box_max,
                            spacing=0.05, times=[0, 1, 2, 3])

``sample`` places points at most ``spacing`` apart along every axis of
the box, so the counts follow from its size. Each prints
``[Info] Force field WxHxDxT: <N> MB estimated``. A grid of
kind ``"air-velocity"`` adds to the scene wind inside the air drag
instead. Scale the field per object with
``obj.param.set("force-field-weight", w)``; ``0`` opts it out, and
fix-pinned vertices ignore it. Positions are in scene units, so World
Scaling needs no adjustment.

Stepping a run frame by frame
-----------------------------

A session can hold its solver between two frames, with the process and
its memory alive, so a notebook can inspect the result and change the
force field before the next frame:

.. code-block:: python

   session.run_until_frame(30)          # start if needed, hold at frame 30
   vert, _ = session.get.vertex(session.held_frame())
   scene.force_field.clear().script(new_source)
   session.update_force_field(scene.force_field)
   session.step_frame()                  # advance one frame, hold again
   session.release()                     # run on to the last frame

``session.update_params()`` re-exports the ``dyn()`` schedules of the
scene-wide keys for the next step. A hold the caller abandons saves and
quits on its own after its timeout (one hour by default), and
``session.save_and_quit()`` works while held. ``examples/force-field.ipynb``
puts both features together.

Reading per-object statistics
-----------------------------

The solver records each object's center of mass, velocity, acceleration,
angular velocity, volume, surface area, rod length, their stretch ratios
and its contact count at every output frame, which is what the add-on's
Object Statistics panel shows. A built session reads them through its
``get`` accessor (``app.session.get`` in a notebook the add-on wrote):

- ``get.statistics_frames()`` lists the output frames that carry
  statistics, in ascending order.
- ``get.statistics(n)`` returns every object's values at frame ``n``, or at
  the latest frame when ``n`` is omitted, as ``{"frame", "time",
  "objects"}`` with ``objects`` keyed by object name. A value the solver
  did not have at that frame is ``None`` (an object's first frame has no
  acceleration, for example), and a channel the object's type does not
  have, such as a rod's volume, is absent. With ``n`` omitted it returns
  ``None`` until the first frame is written.
- ``get.statistics_series(name, channel)`` returns one channel of one
  object over the run as two arrays, the simulated times in seconds and
  the values, leaving out the frames with no value.

.. code-block:: python

   stats = app.session.get.statistics()      # the latest frame
   print(stats["objects"]["sheet"]["values"]["speed"])
   t, speed = app.session.get.statistics_series("sheet", "speed")

.. toctree::
   :maxdepth: 1
   :caption: Reference

   module_reference
   simulation_parameters
   material_parameters
   log_channels

**What each reference page covers**

- :doc:`module_reference` lists every class, method, and property the
  package exports: :class:`frontend.App`, the scene / session / asset /
  mesh managers, and the parameter and plotting helpers.
- :doc:`simulation_parameters` lists the application-wide parameters
  set via ``session.param.set(...)`` (step size, Newton iteration
  bounds, contact gaps, frame rate, etc.).
- :doc:`material_parameters` lists the per-object material parameters
  set via ``object.param.set(...)``, with separate defaults for the
  five element types: triangles, tetrahedra, rods, PDRD rigid bodies,
  and granular point clouds.
- :doc:`log_channels` lists the named log streams the solver emits,
  retrievable with ``session.get.log.numbers(name)`` /
  ``session.get.log.stdout()`` for live plotting and post-run analysis.
