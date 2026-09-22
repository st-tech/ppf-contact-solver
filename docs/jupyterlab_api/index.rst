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
connection; a notebook answers it with four module-level functions:

.. code-block:: python

   import frontend

   frontend.list_backends()      # {'cuda': '.../target/cuda/release', ...}
   frontend.probe_backend("rocm")  # can this machine run that build?
   frontend.get_backend()        # what the next run will use
   frontend.set_backend("rocm")  # pin it; None returns to the automatic rule

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
scene whose geometry already overlaps. Three per-object parameters,
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

For the two cross-object parameters one side of a pair is enough. A pin
does the same for the elements it holds completely when it is created
with ``obj.pin(ind, allow_intersection=True)``. None of these affects the
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
