🧩 Module Reference
====================

The classes, methods, and properties exported by the ``frontend`` package,
the names in its ``__all__``. Import them from ``frontend``:

.. code-block:: python

   from frontend import App, Scene, FixedSession, ParamManager

Where to start:

- :class:`frontend.App` is the entry point for a scene built in Python. It
  provides the mesh, asset, scene, and session managers.
- :class:`frontend.BlenderApp` attaches to a project the Blender add-on
  transferred; see :doc:`index`.
- :class:`frontend.SceneManager`, :class:`frontend.Scene`, and
  :class:`frontend.Object` build a scene, and ``Scene.build()`` returns a
  :class:`frontend.FixedScene`.
- :class:`frontend.SessionManager`, :class:`frontend.Session`, and
  :class:`frontend.FixedSession` run it, and :class:`frontend.SessionGet`
  reads the results.
- :class:`frontend.ParamManager` holds the application-wide parameters of a
  session, listed on :doc:`simulation_parameters`.

Some objects that these classes return or raise are not exported, so they are
not listed below:

- ``Object.pin()`` returns a ``PinHolder``; its methods are described under
  :ref:`jupyterlab-pin-settings`.
- ``Object.param`` returns a ``ParamHolder`` with ``set``, ``get``,
  ``get_desc``, ``key_list``, ``items``, ``clear_all``, and ``copy``; the
  keys are listed on :doc:`material_parameters`.
- ``Wall.param`` and ``Sphere.param`` return a collider parameter holder with
  ``list()``, which returns the settings as a dictionary, and
  ``set(name, value)``, which refuses any name other than ``contact-gap``,
  ``friction``, ``active-duration``, and ``thickness``.
- ``session.get.log`` returns a ``SessionLog`` with ``names``, ``stdout``,
  ``stderr``, ``numbers``, ``number``, ``summary``, and ``average_summary``;
  the channels are listed on :doc:`log_channels`. ``summary()`` returns the
  latest values as strings with units: ``time-per-frame`` and
  ``time-per-step`` (``500ms``, ``1.50s``), ``num-contact`` (``1.50k``),
  ``newton-steps`` and ``pcg-iter``, plus ``stretch`` (``20.00%`` for a
  largest stretch ratio of 1.2) when the ``max-sigma`` channel is positive.
  ``average_summary()`` returns the same kind of strings averaged over the
  run (``num-contact (max)`` and ``dyn-consumed (max)`` are maxima), adds
  ``matrix-assembly``, ``pcg-linsolve``, ``line-search``, ``toi`` and
  ``toi-advanced``, and omits a metric whose log file is missing.
- ``session.export.animation()`` returns an object with ``zip()``, which
  archives the exported directory.
- ``Scene.build()`` raises ``ValidationError``, a subclass of ``ValueError``,
  when the scene checks find a violation; its ``violations`` attribute lists
  them.

.. automodule:: frontend
   :members:
   :undoc-members:
   :show-inheritance:

Signed Distance Functions
-------------------------

``frontend.sdf`` is a module, which the listing above leaves out. Its two
functions and its base class are listed here, and :doc:`index` has a
walkthrough.

.. autofunction:: frontend.sdf.sphere

.. autofunction:: frontend.sdf.capsule

.. autoclass:: frontend.sdf.SDF
   :members: bounds, save
   :special-members: __or__, __and__, __sub__
