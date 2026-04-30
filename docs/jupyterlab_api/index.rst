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

   from frontend import App

   app = App.load()              # attach to the transferred project
   session = app.session         # the FixedSession you drive
   session.param.set("dt", 0.01) # override a simulation parameter
   session.run()                 # start / resume the solver

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
  three element types: triangles, tetrahedra, and rods.
- :doc:`log_channels` lists the named log streams the solver emits,
  retrievable with ``session.get.log.numbers(name)`` /
  ``session.get.log.stdout()`` for live plotting and post-run analysis.
