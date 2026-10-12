📓 JupyterLab Python API
===========================

This section documents the ``frontend`` Python package, the API you import
in a JupyterLab notebook running on the solver host to drive the solver
directly. Once a Blender scene has been transferred, the same project can
be run, previewed, parameter-swept, and inspected from a notebook, with
Blender closed. The package also builds a scene from scratch with no
Blender at all.

For when to reach for JupyterLab and how it fits into the end-to-end
loop, see :doc:`/blender_addon/workflow/sim/jupyterlab`. This section
is the reference material the notebook calls into.

This page is the guide: attaching to a transferred project, building a scene
from scratch, recovering a session after a restart, the example notebooks,
choosing a backend, intersection allowances, the contact gap, stitching,
simulated time and units, invisible colliders, pins, animated and mapped
parameters, the notebook counterparts of add-on settings, force fields,
starting and stepping a run, viewing and exporting it, and reading results.
The class and parameter references are :doc:`module_reference`,
:doc:`simulation_parameters`, :doc:`material_parameters` and
:doc:`log_channels`.

Attach to a Transferred Project
-------------------------------

The notebook the add-on exports attaches to the project that **Transfer**
uploaded, then runs it:

.. code-block:: python

   from frontend import BlenderApp

   app = BlenderApp.open("my-project")  # attach to the transferred project
   app.scene.report()                   # inspect what was transferred
   app.session.run()                    # start the solver, or attach to it
   app.session.stream()                 # tail the solver's live output

``app.scene`` and ``app.session`` are the already built
:class:`frontend.FixedScene` and :class:`frontend.FixedSession`.
``BlenderApp.open()`` rebuilds the project's session folder from the
uploaded scene and clears that folder first, so the frames and saved states
of an earlier run are discarded. :doc:`/blender_addon/workflow/sim/jupyterlab`
has the warning that goes with it. To attach again and keep them, build in two
steps with ``BlenderApp("my-project").populate().make(preserve_output=True)``.
It exports the scene input again but leaves the ``output`` folder, where the
frames and saved states are, in place, so ``app.session.resume()`` can
continue from a saved state.

Overriding a Session Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A built session holds a copy of the application-wide parameters taken when
it was built, so setting ``app.session.session.param`` changes nothing by
itself: ``run()`` still launches the copy. Set the value on the
:class:`frontend.Session` behind the built session, then call ``build()`` on
it. The :class:`frontend.FixedSession` that returns carries the new value:

.. code-block:: python

   session = app.session.session   # the Session behind app.session
   session.param.set("dt", 0.004)  # a key from the Simulation Parameters page
   fixed_session = session.build() # exports the scene again with the new value
   fixed_session.run()

``build()`` deletes the project's session folder before it exports, so it
discards the frames and saved states of an earlier run, and a solver that
is still running loses the folder it writes to. To keep the frames and saved
states, call ``session.build(preserve_output=True)``. It removes the scene
input and exports it again but leaves the ``output`` folder in place, so
``resume()`` can continue from a saved state with the edited input. Use
``fixed_session`` from here on: ``app.session`` still holds the session built
earlier. Only the application-wide parameters of :doc:`simulation_parameters`
change this way. Per-object parameters are applied when the scene is built,
which ``BlenderApp.open()`` has already done, so change those in Blender and
press **Transfer** again.

Building a Scene from Scratch
-----------------------------

A notebook or script can build a scene without a Blender project. This
follows ``examples/headless.py``, the shortest complete script, simplified:

.. code-block:: python

   from frontend import App

   app = App.create("headless")           # a new project

   V, F = app.mesh.square(res=64, ex=[0, 0, 1], ey=[0, 1, 0])
   app.asset.add.tri("sheet", V, F)       # register a mesh as an asset

   scene = app.scene.create()
   sheet = scene.add("sheet")             # an Object
   sheet.pin(sheet.grab([0, 1, 0]))       # pin the vertices furthest along +Y
   sheet.param.set("strain-limit", 0.05)  # a per-object material parameter
   fixed_scene = scene.build()            # a FixedScene

   session = app.session.create(fixed_scene)
   session.param.set("dt", 0.01).set("frames", 60)  # application-wide parameters
   fixed_session = session.build()        # a FixedSession; exports the solver input
   fixed_session.start(blocking=True)     # run to the last frame
   fixed_session.export.animation().zip()

``App.load("headless")`` reopens the state that ``app.save()`` wrote. The
snippets in the sections below use these names: ``scene`` is the
:class:`frontend.Scene` before ``build()``, ``fixed_scene`` and
``fixed_session`` are the built objects (``app.scene`` and ``app.session`` in
a notebook the add-on wrote), and ``session`` is the
:class:`frontend.Session` that takes parameters before ``build()``.

Meshing a Signed Distance Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frontend.sdf`` builds a solid from spheres and capsules and writes it as a
triangle mesh that ``app.mesh.load_tri`` reads back. ``examples/trapped.ipynb``
makes its spiked ball this way.

.. code-block:: python

   from frontend import sdf

   shape = sdf.sphere(1.1)                                   # radius, center=(0, 0, 0)
   shape = shape | sdf.capsule([-1, 0, 0], [1, 0, 0], 0.05)  # end points, radius
   shape.save("squishy.ply", step=0.03)                      # grid step in scene units

- ``sdf.sphere(radius, center=(0, 0, 0))`` and ``sdf.capsule(p0, p1, radius)``
  create the two primitives. A capsule is every point within ``radius`` of the
  segment from ``p0`` to ``p1``.
- ``a | b`` is the union. ``a & b`` (intersection) and ``a - b`` (difference)
  are accepted and report ``bounds()``, but ``save`` refuses them with
  ``NotImplementedError``: only a union of spheres and capsules can be meshed.
- ``shape.save(path, step=0.01)`` meshes the shape with marching cubes and
  writes it in the format of the file extension. A smaller ``step`` gives a
  finer mesh and a longer run. ``shape.bounds()`` returns the box
  ``((min_x, min_y, min_z), (max_x, max_y, max_z))`` that ``save`` samples,
  which extends 0.1 beyond the bounding box of the shapes.

The class and function signatures are listed on :doc:`module_reference`.

Recovering a Session After a Restart
------------------------------------

A session built with ``App.create`` can be reopened in a new kernel or
notebook while the solver host still holds its folder:

.. code-block:: python

   from frontend import App

   session = App.recover("large-twist")   # the built FixedSession, not rebuilt
   if not App.busy():
       session.resume()                   # continue from the latest saved state
   session.preview()
   session.stream()

``App.recover(name)`` takes the app name for the first session created
without a name, ``"<app name>-1"``, ``"<app name>-2"`` and so on for the next
ones, and the ``name`` given to ``app.session.create(scene, name=...)`` for a
named one. It returns the :class:`frontend.FixedSession` that ``build()``
produced, without rebuilding or clearing anything, so the frames and saved
states on disk stay. It raises *No session found with name: <name>* when there
is no such session, and *No recoverable fixed session found at named
location: <name>* when the saved link names a folder that does not hold the
session. ``BlenderApp.open()``, by contrast, rebuilds the session and clears
its folder. Saved states come from the ``auto-save`` and ``checkpoints``
parameters, from ``save-state-on-finish`` and from ``save_and_quit()``.

Example Notebooks
-----------------

The ``examples/`` folder of the
`repository <https://github.com/st-tech/ppf-contact-solver/tree/main/examples>`__
holds one notebook per scene, beside ``headless.py``. Find a scene by what it
shows:

.. list-table::
   :header-rows: 1

   * - Shows
     - Notebooks
   * - Pin motion with ``spin``
     - ``belt``, ``five-twist``, ``large-five-twist``, ``large-twist``, ``large-woven``, ``roller``, ``twist``, ``woven``
   * - Rods
     - ``codim``, ``large-woven``, ``noodle``, ``woven``, ``yarn``
   * - Tetrahedral solids
     - ``cards``, ``codim``, ``domino``, ``friction``, ``large-animals``, ``large-fluffy``, ``needle``, ``plastic``, ``ribbon``, ``roller``, ``stack``, ``trampoline``, ``trapped``
   * - Invisible walls
     - ``cards``, ``codim``, ``domino``, ``roller``, ``stack``, ``trampoline``
   * - Invisible spheres
     - ``noodle``, ``ribbon``, ``trapped``
   * - Stitching
     - ``fitting``
   * - Plasticity
     - ``plastic``
   * - Force fields and stepping a run frame by frame
     - ``force-field``
   * - A mesh from a signed distance function
     - ``trapped``
   * - Recovering a long run with ``App.recover``
     - ``large-animals``, ``large-five-twist``, ``large-fluffy``, ``large-twist``, ``large-woven``
   * - Meshes and plots only, no simulation
     - ``walkthrough``

The first run of some notebooks needs internet access on the solver host.
``codim``, ``friction``, ``roller``, ``trampoline`` and ``walkthrough``
download a preset mesh, ``fishingknot`` downloads its mesh from the project's
GitHub release, and ``fitting``, ``large-animals`` and ``large-fluffy`` clone
repositories from GitHub with ``git``. Later runs read the downloaded copy
from the cache folder. The notebooks in ``examples/fail-examples`` are written
to fail and exercise the error paths.

Which Backend a Notebook Runs On
--------------------------------

A tree or distribution can hold several solver builds at once (CUDA, ROCm,
Metal, and the portable CPU build, one per directory). The panel's
**Compute Device** and **GPU Backend** rows answer that question for a
connection. A notebook answers it with four :class:`frontend.App` static
methods:

.. code-block:: python

   from frontend import App

   App.list_backends()        # {'cuda': '.../target/cuda/release', ...}
   App.probe_backend("rocm")  # can this machine run that build?
   App.get_backend()          # what the next run will use
   App.set_backend("rocm")    # pin it; None returns to the automatic rule

The same four are also exported as module-level functions
(``frontend.set_backend`` and so on).

The automatic rule asks each GPU build through its own solver, takes the
first of CUDA, ROCm, and Metal that reports a usable device, and otherwise
takes the CPU build and prints one line naming what each GPU backend said.
The add-on never moves to the CPU build by itself. Its **Compute Device** is
an explicit **GPU** or **CPU** choice, and its **GPU Backend** setting
**Automatic** picks only among the GPU builds (CUDA, then ROCm), with an
error when several are present and none has a usable device.

An explicit choice never falls back: naming a GPU backend on a machine that
cannot run it raises instead of running on the CPU. ``CARGO_TARGET_DIR``,
which the add-on sets when it launches a server, counts as an explicit
choice for the same reason. See :doc:`module_reference` for the full
signatures, and :ref:`choosing-the-build` for the add-on's side of it.

Letting Objects Pass Through Each Other
---------------------------------------

By default the solver keeps every pair of surfaces apart and refuses a
scene whose geometry already overlaps. Four per-object parameters, set as
float-encoded booleans with ``object.param.set(key, 1.0)``, let chosen
pairs pass through each other instead. An allowed pair gets no contact
force, is not held apart during a step, and is never reported as an
intersection.

- ``allow-self-intersection``: the object passes through itself.
- ``allow-inter-object-intersection``: the object passes through every
  other object, static colliders included.
- ``allow-inter-group-intersection``: the object passes through every
  object in a different group, while objects of its own group still
  collide with it. :meth:`frontend.Object.group` assigns an object to a
  named group; objects never given one share a default group, and a static
  collider counts as another group from every object.
- ``allow-existing-intersection``: only the pairs the object starts
  overlapping, or closer than their contact offsets, pass through each
  other for the whole run, together with the elements sharing a vertex
  with them. Every other pair keeps full contact, and a new intersection
  still stops the run. It tolerates a tangle and does not untangle it. It
  is not available on a points (sand) object, which is refused by name. On
  the :class:`frontend.FixedScene` that ``scene.build()`` returns,
  ``start_links`` holds the vertex links the build made and
  ``start_link_exemptions()`` describes the exempted pairs.

For the cross-object parameters and ``allow-existing-intersection``, one
side of a pair is enough. A pin does the same for the elements it holds
completely when it is created with ``obj.pin(ind, allow_intersection=True)``,
or later with ``holder.set_allow_intersection(True)``. None of these
affects the invisible walls and spheres.

.. code-block:: python

   # The shirt and the jacket collide with each other and pass through the body.
   for name in ("shirt", "jacket"):
       scene.add(name).group("garments").param.set(
           "allow-inter-group-intersection", 1.0)
   scene.add("body").group("character").pin()

See :doc:`material_parameters` for each parameter's full description,
and :ref:`allow-intersections-settings` for the same settings in the
Blender add-on.

Keeping the Contact Gap Smaller Than the Mesh
---------------------------------------------

The solver forms a contact pair between a vertex and the vertices and
edges of its own mesh that it is not part of, the opposite edge of its
own triangle and the far end of its own rod segment included. A
``contact-gap`` wider than the distance between those neighbors, less
their contact offsets, therefore starts a run with the mesh pressing on
itself: a sheet at rest swells, and a fine mesh stiffens against
bending. Gomez-Nogales et al. name this *contact locking* in
`Unlocking Thickness Modeling for Codimensional Contact Simulation
<https://arxiv.org/abs/2510.10256>`__ (arXiv:2510.10256, 2025). The paper
lowers the distance at which contact engages for the pairs of mesh
neighbors that start closer than it in the undeformed mesh, and
``contact-gap-limit`` applies the same idea as a cap on each element's gap.

``contact-gap-limit`` caps each element's gap at a fraction of the room
its own neighbors leave. It is ``0.0`` by default, which sets no limit,
and otherwise lies between ``0.01`` and ``0.5``; ``scene.build()`` refuses
any other value with ``ValueError``. It is one fraction per object, so to
vary the gap over the run, animate or map ``contact-gap`` instead (see
:ref:`spatially-varying-parameters`); the limit holds every value
``contact-gap`` takes to the same bound. Animating or mapping
``contact-gap-limit`` itself is refused.

.. code-block:: python

   sheet = scene.add("sheet")
   sheet.param.set("contact-gap", 2e-3).set("contact-gap-limit", 0.25)

For each triangle or rod segment the limit works from three lengths:

.. code-block:: text

   clearance = smallest size among the element and its vertex neighbors
   room      = clearance - 2 * (the object's largest contact-offset)
   gap       = min(contact-gap, contact-gap-limit * room)

A triangle's size is its smallest altitude and a rod segment's is its
length. It is measured on the mesh as it enters the solve, which for a
tetrahedral object is its surface triangles.

- The limit only lowers a gap, element by element, so the coarse part
  of a mesh keeps the value it was given. Every value of an animated or
  mapped ``contact-gap`` is held to the same bound.
- It binds only an object that can be in contact with itself. A fully
  pinned object with no pin operation (a static collider), a PDRD body
  and an object with ``allow-self-intersection`` form no pair among
  their own elements and are left as authored.
- An element whose clearance is no more than twice its contact offset
  has no room and keeps its authored gap. Its neighbors are inside their
  offsets whatever the gap is, which the scene checks refuse unless an
  intersection allowance exempts them.
- ``scene.build()`` prints one line for each object whose gap it
  lowered, unless ``quiet=True``.
- A rod whose segments are barely longer than twice its contact offset
  leaves almost no room, so its gap becomes very small and the solve
  slows down. At a limit of ``0.25`` a rod keeps its authored gap once
  each segment is at least ``2 * contact-offset + 4 * contact-gap`` long.
  Sampling a strand at equal arc length, as the vertical strands of
  ``examples/yarn.ipynb`` are, gives every segment the same length.

The add-on sends this parameter for its **Solid**, **Shell** and **Rod**
groups, at 25 percent unless the artist changes it; see
:ref:`local-gap-limit`.

Reference:

    Gonzalo Gomez-Nogales, Zhen Chen, Rosalie Martin, Elena Garces and
    Danny M. Kaufman. "Unlocking Thickness Modeling for Codimensional
    Contact Simulation." arXiv:2510.10256, 2025.
    https://arxiv.org/abs/2510.10256

Stitching Two Objects Together
------------------------------

The add-on's snap-and-merge pairs (see
:doc:`/blender_addon/workflow/constraints/snap_merge`) are cross-object
stitches, and a notebook adds the same with
:meth:`frontend.Scene.cross_stitch`. Each row joins a point on one object
to a point on another, each written as three vertices of its own object and
their barycentric weights; a point at one vertex ``i`` is ``[i, i, i]`` with
weights ``[1, 0, 0]``.

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

.. _simulated-time-and-units:

Simulated Time and Units
------------------------

The solver writes output frame ``n`` at ``n / fps`` seconds of simulated time,
so a run of ``frames`` frames covers ``frames / fps`` seconds. The defaults of
300 frames at 60 fps cover 5 seconds. ``dt`` is the step between solver
updates, and a ``dt`` longer than ``0.9999 / fps`` is lowered to that value.
Every time argument, namely pin ``t_start``, ``t_end`` and ``unpin``,
``dyn().time()``, ``Object.move``, ``velocity(..., t=...)``, collider
keyframes, ``set_param_anim_times``, force field ``times`` and the ``t`` a
force field script receives, is on this clock in seconds. A motion scheduled
to end after the last frame does not finish within the run.

Other conventions:

- Axes are the solver's: Y is up, and the default ``gravity`` is
  ``[0, -9.8, 0]``.
- ``Object.rotate`` and ``Object.animate_rotate`` take degrees.
  ``PinHolder.spin`` takes degrees per second. ``Object.angular_velocity``,
  ``angular_velocity_world`` and ``angular_velocity_pca`` take radians per
  second. ``PinHolder.torque`` takes newton-meters.

Invisible Walls and Spheres
---------------------------

``scene.add.invisible.wall(position, normal)`` and
``scene.add.invisible.sphere(position, radius)`` add the add-on's **Invisible
Colliders** (see :doc:`/blender_addon/workflow/constraints/colliders`) to a
notebook scene. A wall is a flat boundary through ``position``, and objects
stay on the side its ``normal`` points to. Both calls return the collider, and
its keyframe methods return it again, so calls chain.

.. code-block:: python

   scene.add.invisible.wall([0, 0, 0], [0, 1, 0])   # a floor, normal pointing up
   bowl = scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
   piston = scene.add.invisible.wall([1, 0, 0], [-1, 0, 0])
   piston.move_to([0.5, 0, 0], 3.0).interp("smooth")

- A wall has ``move_by(delta, time)`` and ``move_to(pos, time)``. A sphere has
  the same two, ``radius(r, time)``, which changes its radius, and
  ``transform_to(pos, r, time)``, which changes position and radius together.
  Times are absolute seconds of simulated time, and each must be later than
  the one before it. The position given when the collider is created is the
  keyframe at time 0. ``interp(transition)`` sets the transition between
  keyframes, ``"linear"`` unless set, or ``"smooth"``.
- ``invert()`` keeps objects inside a sphere instead of outside it.
  ``hemisphere()`` empties the top half, like a bowl, so an inverted
  hemisphere catches objects that fall in from above.
- ``collider.param.set(name, value)`` takes ``contact-gap`` (default
  ``0.001``), ``friction`` (``0.0``), ``thickness`` (``1.0``, the depth
  beyond which geometry passes through the collider) and ``active-duration``
  (seconds; the collider stops acting at that time, and the default ``-1.0``
  means it never stops). Any other name raises an error.
- ``scene.build()`` raises ``ValidationError`` when a free vertex starts on
  the wrong side of a collider that has a single keyframe, with a message such
  as *N vertices violate wall constraints* or *N vertices violate sphere
  constraints*. Pinned vertices are exempt.

.. _jupyterlab-pin-settings:

Pin Settings
------------

``obj.pin(...)`` returns a ``PinHolder``. The class is not exported from
``frontend``, so :doc:`module_reference` does not list it, and its methods
are described here. Each method returns the holder, so calls chain. Time
arguments (``t_start``, ``t_end``, and the time of ``unpin``) are seconds of
simulated time, where the add-on's pin methods take frames.

- ``holder.move_by(delta_pos, t_start=0.0, t_end=1.0, transition=None,
  bezier_handles=None)`` ramps a translation from ``t_start`` to ``t_end``,
  which must be later. ``delta_pos`` is one offset for every pinned vertex or
  one row per pinned vertex. ``bezier_handles`` is the easing curve of the
  ``"bezier"`` transition and takes effect only with it; see
  ``holder.interp`` below. ``holder.move_to(target_pos, t_start=0.0,
  t_end=1.0, transition=None, bezier_handles=None)`` takes absolute positions
  instead: one per pinned vertex, or a single position, which moves the pinned
  vertices by the offset from the object's position to it.
- ``holder.scale(scale, t_start=0.0, t_end=1.0, center=None,
  center_mode="absolute")`` scales from ``1.0`` to ``scale``. ``center_mode``
  is ``"absolute"`` (the default), which uses the fixed ``center`` (the origin
  when omitted), or ``"centroid"``, which uses the centroid of the pinned
  vertices. ``holder.spin(...)`` below takes the same two arguments.
- ``holder.transform_keyframes(local_vert, times, translations, quaternions,
  scales, segments, rest_translation)`` pins the vertices to sparse
  translation, rotation, and scale keyframes, so a rotating object follows an
  arc. ``segments`` holds one entry per interval between ``times``, each
  with an ``"interpolation"`` of ``"LINEAR"`` (the default), ``"CONSTANT"``
  or ``"BEZIER"``; a ``"BEZIER"`` entry also needs ``handle_right`` and
  ``handle_left``, two values each.
- ``holder.interp(transition)`` sets the transition, ``"linear"``,
  ``"smooth"`` or ``"bezier"``, that the ``move_by``, ``move_to`` and
  ``scale`` calls made after it use, unless a ``move_by`` or ``move_to``
  names its own. ``bezier_handles`` is ``((x1, y1), (x2, y2))``, the two
  inner control points of a cubic Bezier easing curve that runs from (0, 0) to
  (1, 1), each value between 0 and 1: x is the fraction of the time window and
  y the fraction of the motion done. It takes effect only with the
  ``"bezier"`` transition. A ``"bezier"`` transition without handles runs as
  ``"linear"``, and so does a ``scale`` under ``interp("bezier")``, because
  ``scale`` takes no handles.
- ``holder.unpin(time)`` releases the pin at ``time`` seconds.
- ``holder.pull(strength=1.0)`` turns the pin into a soft pull toward the
  prescribed positions. ``holder.pull_per_vertex(weights)`` gives each pinned
  vertex its own weight; call ``pull`` with a positive strength as well.
- ``holder.set_allow_intersection(True)`` is the add-on's **Allow
  Intersections Here** (see above).
- ``holder.torque(magnitude=1.0, axis_component=2, hint_vertex=None,
  t_start=0.0, t_end=inf)`` requires ``hint_vertex``, a vertex of the pinned
  object in its own numbering. A principal axis has no sign of its own, so
  the solver points the torque axis from the pinned vertices' centroid toward
  that vertex, and that is what decides which way a positive ``magnitude``
  turns the object. A missing hint, or one that is not a vertex of the
  object, raises ``ValueError``. ``magnitude`` is in newton-meters.
  ``axis_component`` picks the principal axis of the pinned vertices: 0 the
  major axis (the longest extent), 1 the middle one, 2 the minor axis (the
  shortest, the default). The torque turns about the centroid of the pinned
  vertices from ``t_start`` to ``t_end`` seconds, with no end by default.
- ``holder.set_pin_group_id(name)`` puts holders into one named group, which
  matters only for torque. Holders that share a name and each carry a torque
  are one torque group: the torque turns about the centroid and principal
  axes of all their vertices together, with the magnitude, axis component,
  and hint vertex of the group's first holder. The name must be a non-empty
  string.
- ``holder.spin(center=None, axis=None, angular_velocity=360.0, t_start=0.0,
  t_end=inf, center_mode="absolute")`` turns about ``axis`` through
  ``center``, which default to ``[0, 1, 0]`` and the origin.
  ``angular_velocity`` is in degrees per second and turns the other way when
  negative. A spin with no ``t_end`` runs for the whole simulation. An
  ``axis`` of zero length with a nonzero angular velocity raises
  ``ValueError``, because there is no axis to turn about.
- ``holder.track_rest_shape()`` is the add-on's **Track Rest-Pose
  Deformation**: the object's rest shape follows the holder's operations,
  so the body settles into the prescribed deformation rather than resisting
  it. Both a fixed pin and a ``pull`` pin can track.
  :meth:`frontend.Scene.build` refuses tracking when the tracking pins of
  an object do not together hold every one of its vertices, when a
  tracking pin carries no operation, and when the object also has
  ``plasticity`` or ``bend-plasticity`` above zero, animated, or mapped,
  since both rewrite the rest shape.

.. code-block:: python

   # Slide a pinned sphere 8 units along +X between t = 0 and t = 5.
   scene.add("sphere").pin().move_by([8, 0, 0], t_start=0.0, t_end=5.0)

   # Twist a post about its long axis, oriented toward its topmost vertex.
   post = scene.add("post")
   top = int(post.vertex(False)[:, 1].argmax())
   post.pin().torque(magnitude=0.5, axis_component=0, hint_vertex=top)

Animating Parameters Over the Run
---------------------------------

``obj.set_param_anim(key, values)`` gives a material parameter one value
per entry of the scene's ``set_param_anim_times(...)``, which are seconds
of simulated time. On a tetrahedral object only ``friction``,
``contact-gap`` and ``contact-offset`` can be animated, because they live
on its surface triangles; its elastic parameters live on its tetrahedra,
which hold one value for the whole run, and animating one raises
``ValueError``. The add-on applies the same rule to a **Solid** group's
keyframes.

``session.param.dyn(key)`` selects an application-wide parameter's schedule
and places the time cursor on the last time already scheduled for that key
(``0`` for a key with none), so a second ``dyn`` on the same key continues
the schedule. ``time(t)`` takes seconds of simulated time, where the
add-on's ``solver.param.dyn`` takes frames. It refuses a ``t`` that is not
later than the cursor, so no entry lands before one already scheduled.

.. code-block:: python

   g = session.param.get("gravity")
   session.param.dyn("gravity").time(1.0).hold().time(1.5).change([-x for x in g])
   # A second dyn on the same key continues from t = 1.5.
   session.param.dyn("gravity").time(2.0).change(g)

.. _spatially-varying-parameters:

Spatially Varying Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``obj.set_param_spatial(key, weights, target)`` is the notebook form of the
add-on's **Spatial Material Maps** (see
:doc:`/blender_addon/workflow/params/material`). ``key`` is a material
parameter of :doc:`material_parameters`. ``weights`` holds one number from 0
to 1 per vertex. The value at a vertex is the object's ordinary value blended
toward ``target`` by its weight: 0 keeps the ordinary value and 1 gives
``target``. Each triangle or tetrahedron uses the mean of the weights of its
own vertices. Each parameter takes one map.

.. code-block:: python

   # A map that stays fixed.
   sheet.set_param_spatial("friction", weights, target=0.8)

   # A map that changes over the run, here for another parameter.
   sheet.set_param_spatial_anim("young-mod", [w0, w1], times=[0.0, 2.0],
                                target=5000.0)

``set_param_spatial_anim(key, weight_frames, times, target)`` takes one weight
array per entry of ``times``, in seconds, strictly increasing and starting at
0.0. The weights are interpolated linearly between samples, and the nearest
sample holds outside the range. An unknown key raises ``KeyError``. A weight
count that differs from the vertex count, a weight outside 0 to 1 or not
finite, a second map for the same key, or an object with neither triangles nor
tetrahedra raises ``ValueError``.

An object whose every vertex is pinned, with no pin carrying an operation, a
pull or a release time, is a static collision mesh. Its parameters are fixed
when the scene is built, so ``scene.build()`` raises ``ValueError`` when it
animates or maps one: *'<name>' is built as a static collision mesh ... but it
animates or maps '<key>'.* Remove the animation or the map, or give one of its
pins an operation, a pull or a release time so the object is simulated.

Notebook Counterparts of Add-on Settings
----------------------------------------

These :class:`frontend.Object` methods set from a notebook what the add-on
sets per group. The time arguments are in seconds, as under
:ref:`Simulated Time and Units <simulated-time-and-units>`.

.. list-table::
   :header-rows: 1

   * - Add-on setting
     - Notebook call
   * - **Lock Translation**
     - ``obj.lock_translation(x, y, z)`` keeps the center of mass on the line through its start position along the direction. ``obj.lock_all_translations()`` holds it at a point. The last call of the two wins.
   * - **Lock Rotation**
     - ``obj.lock_rotation(x, y, z)`` allows rotation about one world axis only. ``obj.lock_rotation_prohibit_axis()``, called after it, forbids that axis instead. ``obj.lock_all_rotations()`` forbids all rotation.
   * - **Velocity Overwrite**
     - ``obj.velocity(u, v, w)`` sets the initial velocity. ``obj.velocity(u, v, w, t=1.0)`` and ``obj.velocity_schedule([(t, [vx, vy, vz]), ...])`` set timed overrides, in seconds.
   * - Angular velocity
     - ``obj.angular_velocity(wx, wy, wz)`` spins about the centroid in radians per second. ``obj.angular_velocity_world(wx, wy, wz, t=...)`` and ``obj.angular_velocity_pca(pca_index, speed, t=...)`` start a spin at time ``t`` about a world axis or a principal axis (0 the longest extent, 2 the shortest).
   * - **Collision Active Duration Windows**
     - ``obj.collision_windows([(t_start, t_end), ...])`` in seconds, at most 8 windows.
   * - **PDRD** group and **Hinge**
     - ``obj.as_pdrd()`` on a triangle asset. ``obj.hinge(pca_axis=2)`` fixes the body's position and lets it turn only about one principal axis.
   * - **Static** group, moving
     - ``obj.pin()`` pins every vertex. ``obj.move(delta, t_start=0.0, t_end=1.0)`` and ``obj.animate_rotate(axis, angle, center=None, t_start=0.0, t_end=1.0)`` then keyframe its translation and its rotation in degrees, and the pin operations of :ref:`jupyterlab-pin-settings` move it as well.

External Force Fields
---------------------

``scene.force_field`` adds an acceleration (m/s², like gravity) to every
free vertex, evaluated once per step at the position the vertex starts
that step from. Two sources can be mixed. The add-on's **Force Fields** box
sets up the same two sources; see
:doc:`/blender_addon/workflow/params/force_fields`.

An exact Python function, compiled to run on the solver, evaluated at
each vertex's own position with no resolution and no domain:

.. code-block:: python

   import math

   def eval(x, y, z, t):
       r = math.sqrt(x * x + z * z) + 1e-6
       return (-z / r, 0.0, x / r)     # a swirl about the Y axis

   scene.force_field.script(eval)      # or the source text as a string

In a notebook, ``x``, ``y``, ``z`` and the returned vector use the solver's
axes, Y up. Pass ``z_up=True`` to ``script`` to write the function in
Blender's Z-up axes, as the add-on does. ``t`` is seconds.

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

The function must return ``(ax, ay, az)`` on every path. Anything else is
refused when ``script`` is called, with its line, and an accepted function
prints ``[Info] Force field script compiled: ...``.
``scene.force_field.builtins()`` prints every function, constant and
construct a script may use. The solver runs the function in single
precision. ``frontend.noise`` and ``frontend.curl_noise`` are the same
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

``values`` is ``(T, D, H, W, 3)``, or ``(D, H, W, 3)`` for a single instant,
which needs no ``times``. ``times`` holds one strictly increasing instant per
time sample, in seconds, and the end sample holds outside its range.
``box_min`` and ``box_max`` are in the solver's axes, Y up, and ``box_min``
must be below ``box_max`` on every axis. Every spatial axis needs at least 2
samples, and every value must be finite. ``kind`` is ``"acceleration"`` (the
default) or ``"air-velocity"``. Each refusal raises ``ValueError`` naming the
problem. ``sample`` places points at most ``spacing`` apart along every axis
of the box, so the counts follow from its size. Each prints
``[Info] Force field WxHxDxT: <N> MB estimated``. A grid of kind
``"air-velocity"`` adds to the scene wind inside the air drag instead, and
acts only where ``air-density`` is positive. Scale the field per object with
``obj.param.set("force-field-weight", w)``; ``0`` opts it out, and
fix-pinned vertices ignore it. Positions are in scene units, so World
Scaling needs no adjustment.

Starting, Resuming and Stopping a Run
-------------------------------------

- ``fixed_session.start(force=False, blocking=None, load=0)``. In a notebook
  it returns at once and the solver runs in the background. In a script it
  waits for the last frame, shows a progress bar, and prints
  ``*** Solver finished ***``, or ``*** Solver FAILED ***`` when the solver
  wrote to its error log, followed by the end of the log. Pass ``blocking`` to
  choose either behavior. ``load`` is the saved frame to start from, which
  ``resume`` sets.
- When the session has saved states and ``force`` is ``False``, ``start()``
  prints *Solver has saved states. Resuming from N* and resumes from the
  latest one. ``start(force=True)`` ends a running solver and starts from
  frame 0. ``run()`` starts from frame 0 only when no solver is running, so it
  never resumes and never interrupts one. ``resume(frame=-1)`` ends a running
  solver, then resumes from the latest saved frame, or from ``frame``. With no
  saved state, ``resume()`` does nothing.
- Starting from frame 0 deletes ``export/<app name>/<session name>``, the
  default folder of :ref:`exporting-frames`.
- One solver runs at a time. ``start()`` without ``force`` while one is
  running prints *Solver is already running. Terminate first.* and, in a
  notebook, shows a **Terminate Now** button. ``App.busy()`` reports whether a
  solver is running and ``App.terminate()`` ends it.
  ``fixed_session.is_running()`` reports whether a solver is running,
  ``fixed_session.initialize_finished()`` turns true once the solver has
  loaded the scene, and ``fixed_session.finished()`` turns true once it wrote
  its completion marker.
- A blocking start raises ``RuntimeError`` when the solver exits with a
  nonzero status, with the cause found in the log or *Solver exited with
  status N; see <error log>*, and when it exits with status 0 but wrote no
  completion marker. Text in the error log alone does not raise. A solver that
  stops while loading the scene raises ``ValueError``, and so does a start
  inside a notebook whose solver exits with an error within a fraction of a
  second.
- A CUDA run first checks the NVIDIA driver and raises ``ValueError`` with
  *Driver version is N. It must be 520 or newer* or *Driver version could not
  be detected.*

Stepping a Run Frame by Frame
-----------------------------

A session can hold its solver between two frames, with the process and
its memory alive, so a notebook can inspect the result and change the
force field before the next frame. Here ``scene`` is the
:class:`frontend.Scene` the session was built from:

.. code-block:: python

   fixed_session.run_until_frame(30)    # start if needed, hold at frame 30
   vert, _ = fixed_session.get.vertex(fixed_session.held_frame())
   scene.force_field.clear().script(new_source)
   fixed_session.update_force_field(scene.force_field)
   fixed_session.step_frame()           # advance one frame, hold again
   fixed_session.release()              # run on to the last frame

``fixed_session.update_params()`` re-exports the parameters of
``fixed_session.session`` for the next step. Only the ``dyn()`` schedules of
the application-wide keys the solver can animate (``gravity``, ``wind``,
``air-density``, ``air-friction``, ``isotropic-air-friction``, ``dt`` and
``playback``) and the per-object velocity schedules take effect. Keys for
times already past change nothing, and collision windows cannot change
mid-run and are refused. After its timeout (one hour by default), a held
solver saves its state and exits, and ``fixed_session.save_and_quit()``
works while held.
``examples/force-field.ipynb`` puts both features together.

Viewing a Run in the Notebook
-----------------------------

These calls draw widgets in JupyterLab. Outside Jupyter they do nothing.

- ``fixed_scene.preview()`` of a built scene draws it in 3D with wireframe,
  pin dots and stitch lines on. When a pin has an operation, an object has an
  initial velocity, or a static collider moves, a **time** slider scrubs the
  prescribed motion in steps of 0.01 s, up to the end of the longest pin
  operation or collider move (1 second where an operation has no end, and at
  least 1 second when an object has a velocity).
- ``fixed_session.preview()`` shows the latest frame and, while the solver
  runs, updates it live. Below it, the **Terminate Solver** and **Save and
  Quit** buttons and a table of the latest values appear while the solver
  runs. The table lists **Frame**, **Time/Frame**, **Time/Step**,
  **#Contact**, **#Newton**, **#PCG** and, when the log records a stretch,
  **Max Stretch**.
- ``fixed_session.animate()`` loads frame 0 up to, but not including, the
  highest frame number on disk (``get.latest_frame()``) into a **frame**
  slider, whose first position shows frame 0. **Reload** loads the frames
  written since, and a label reads *Loaded N frames*.
- ``fixed_session.stream(n_lines=40)`` shows the last ``n_lines`` lines of the
  solver's output. **Stop Live Stream** turns into **Start Live Stream**, and
  **Terminate Solver** and **Save and Quit** end or checkpoint the run. When
  the solver stops, the view adds *Terminated.* in red, followed by the error
  log, and with no log yet it shows *No log file found.*

See :ref:`render-options` for the ``options`` and ``engine`` arguments.

.. _exporting-frames:

Exporting Frames from a Notebook
--------------------------------

``fixed_session.export.animation()`` writes frame 0 up to, but not including,
the highest frame number on disk (``get.latest_frame()``), so the last frame of
a finished run is left out. ``fixed_session.export.frame(path, frame=None)``
writes one frame, the latest when ``frame`` is omitted, so it writes that last
one. While ``get.latest_frame()`` is 0, ``animation()`` waits as long as the
solver runs and otherwise prints *No animation frames available to export.*

.. code-block:: python

   fixed_session.export.animation()                  # frame_0.ply, frame_0.ply.png, ...
   fixed_session.export.animation(ext="obj").zip()   # OBJ meshes, then archive the folder
   fixed_session.export.frame("latest.ply")          # one frame

- **Folder.** Without ``path``, the files go to
  ``export/<app name>/<session name>``, relative to the notebook's working
  directory. Starting a fresh run (not a resume) deletes that folder first. A
  frame file that already exists is not rewritten unless ``clear=True``.
- **Files.** Each frame is a mesh ``frame_<n>.<ext>`` (``ply`` unless ``ext``
  says otherwise; ``obj`` also works) and a rendered picture
  ``frame_<n>.<ext>.png`` beside it. ``include_static=False`` leaves static
  colliders out of both.
- **Video.** When ffmpeg is available (the copy bundled with the solver, or
  one on ``PATH``) and pictures were rendered, they are encoded as
  ``frame.mp4`` in the same folder at 60 frames per second, and a notebook
  shows it inline.
- **Archive.** ``.zip()`` writes ``<folder>.zip`` next to the folder, replaces
  an existing archive, and prints its path.
- **Pictures.** The ``options`` dictionary picks the renderer.
  ``"renderer": "software"`` (the default) uses a built-in rasterizer that
  needs no GPU and draws 640 by 480 pixels unless ``"width"`` and
  ``"height"`` say otherwise. ``"renderer": "mitsuba"`` path-traces with the
  ``mitsuba`` command, which must be on ``PATH``, and the ``mitsuba`` Python
  package. ``"skip_render": True`` writes only the meshes, and no video.
  :ref:`render-options` lists the other keys.

.. _render-options:

Render Options
--------------

``preview()`` and ``animate()`` of a built scene or session take an
``options`` dictionary and an ``engine``. ``export.animation()`` and
``export.frame()`` take ``options`` too. A key left out keeps its default.

For ``preview()`` and ``animate()``:

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``flat_shading``
     - ``False``
     - Shade each triangle flat instead of smooth.
   * - ``wireframe``
     - ``False`` for a session, ``True`` for ``fixed_scene.preview()``
     - Draw the mesh edges over the surface.
   * - ``pin``
     - ``False`` for a session, ``True`` for ``fixed_scene.preview()``
     - Draw a dot on every pinned vertex.
   * - ``stitch``
     - Same as ``pin``
     - Draw a line for every stitch.
   * - ``max-area``
     - ``2``; a session that has run ``start()`` uses ``1`` plus the largest ``strain-limit`` in the scene
     - The ratio of a triangle's area to its starting area at which an object set with ``dyn_color("area")`` reaches red. Ratios from 1 up to this value run from blue to red.
   * - ``eye``, ``lookat``
     - Chosen from the mesh
     - Camera position and the point it looks at, as ``[x, y, z]``.
   * - ``fov``
     - ``50``
     - Camera field of view in degrees.
   * - ``width``, ``height``
     - ``600`` by ``600``
     - Size of the viewer in pixels.
   * - ``direct_intensity``, ``ambient_intensity``
     - ``1.0``, ``0.7``
     - Brightness of the directional and the ambient light.
   * - ``pts_color``
     - ``"white"``
     - Color of the pin dots.

For ``export.animation()`` and ``export.frame()``, a picture reads these keys:

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``renderer``
     - ``"software"``
     - ``"software"`` or ``"mitsuba"``.
   * - ``skip_render``
     - ``False``
     - Write only the meshes.
   * - ``width``, ``height``
     - ``640`` by ``480``
     - Size of the picture in pixels, for both renderers.
   * - ``max-area``
     - As above
     - The color ramp of ``dyn_color("area")`` objects.

With ``"mitsuba"``, the picture also reads ``variant`` (``"cuda_ad_rgb"``),
``max_depth`` (12), ``fov`` (20), ``sample_count`` (64), ``camera``
(``{"origin": [...], "target": [...]}``, chosen from the mesh when omitted) and
``up`` (``[0, 1, 0]``).

The ``engine`` argument of ``preview()`` and ``animate()`` is ``"threejs"``
(the default, an interactive 3D viewer) or ``"software"``. ``"software"``
draws a still picture with the rasterizer the exports use. It ignores the
camera, light, shading, wireframe and size keys and draws no pin dots; the
notebook warns *the software engine does not render point primitives; use the
threejs engine to display points*.

Reading Per-Object Statistics
-----------------------------

The solver records each object's center of mass, velocity, acceleration,
angular velocity, volume, surface area, rod length, their stretch ratios
and its contact count at every output frame, which is what the add-on's
**Object Statistics** panel shows; see
:doc:`/blender_addon/workflow/sim/simulating`. A built session reads them
through its ``get`` accessor (``app.session.get`` in a notebook the add-on
wrote):

- ``get.statistics_frames()`` lists the output frames that carry
  statistics, in ascending order.
- ``get.statistics(n)`` returns every object's values at frame ``n``, or at
  the latest frame when ``n`` is omitted, as ``{"frame", "time",
  "objects"}`` with ``objects`` keyed by object name. Each entry of
  ``objects`` is ``{"uuid", "type", "values"}``. A value the solver did not
  have at that frame is ``None`` (an object's first frame has no
  acceleration, for example), and a channel the object's type does not
  have, such as a rod's volume, is absent. With ``n`` omitted it returns
  ``None`` until the first frame is written, and a frame with no record
  raises ``OSError``.
- ``get.statistics_series(name, channel)`` returns one channel of one
  object over the run as two arrays, the simulated times in seconds and
  the values, leaving out the frames with no value.

In a scene built with ``scene.add``, an object is named after its asset: the
first ``scene.add("sheet")`` is ``"sheet"`` and later ones are ``"sheet_1"``,
``"sheet_2"`` and so on. ``scene.add("sheet", "flag")`` names it ``"flag"``,
and a name already in use raises an error. These names are the keys of
``get.statistics()``, the ``name`` of ``get.statistics_series()`` and the
argument of ``scene.select(name)``.

The channels that ``values`` holds and ``statistics_series`` takes are:

- ``location_x``, ``location_y`` and ``location_z``, the mass-weighted
  center.
- ``velocity_x``, ``velocity_y``, ``velocity_z`` and ``speed``.
- ``acceleration_x``, ``acceleration_y``, ``acceleration_z`` and
  ``acceleration_magnitude``.
- ``angular_velocity_x``, ``angular_velocity_y``, ``angular_velocity_z``,
  ``angular_speed``, and the axis ``angular_axis_x``, ``angular_axis_y`` and
  ``angular_axis_z``.
- ``volume``, ``surface_area`` and ``rod_length``, with ``volume_stretch``,
  ``area_stretch`` and ``length_stretch`` against the rest shape.
- ``contact_count``.

.. code-block:: python

   stats = app.session.get.statistics()      # the latest frame
   print(stats["objects"]["sheet"]["values"]["speed"])
   t, speed = app.session.get.statistics_series("sheet", "speed")

Reading Simulated Positions
---------------------------

- ``fixed_session.get.vertex(n=None)`` returns ``(vertices, frame)``, where
  ``vertices`` is a float32 array of shape ``(N, 3)`` with the rows of
  ``fixed_scene.vertex()`` (the simulated vertices; static colliders are not
  in it), or ``None`` when that frame is not on disk. Without ``n`` it returns
  the latest frame.
- ``get.latest_frame()`` is the highest frame number on disk, and 0 before the
  first frame.
- ``get.saved()`` lists the frames that have a saved state, which ``resume()``
  can start from.

.. code-block:: python

   vert, frame = fixed_session.get.vertex()
   print(vert.shape, frame)

Reference Pages
---------------

Four pages list the package's classes and methods, its application-wide
parameters, the per-object material parameters, and the solver's log
channels:

.. toctree::
   :maxdepth: 1

   module_reference
   meshes
   simulation_parameters
   material_parameters
   log_channels
