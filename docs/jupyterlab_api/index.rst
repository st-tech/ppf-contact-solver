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
