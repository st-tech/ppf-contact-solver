🧱 Creating and Preparing Meshes
================================

``app.mesh`` builds the meshes a scene is made of and prepares them for the
solver. ``app.asset.add`` then registers a mesh under a name so that a scene
can use it. Every mesh is a tuple of NumPy arrays:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Type
     - Unpacks as
     - Contents
   * - ``TriMesh``
     - ``V, F``
     - Vertices and triangle indices. A closed 2D line shape unpacks the same
       way, with edges in place of triangles, until it is triangulated.
   * - ``TetMesh``
     - ``V, F, T``
     - Vertices, the surface triangles, and the tetrahedra.
   * - ``Rod``
     - ``V, E``
     - Vertices and the segments that join them.

.. code-block:: python

   from frontend import App

   app = App.create("sheet")
   V, F = app.mesh.square(res=64, size=2.0)
   app.asset.add.tri("sheet", V, F)

   V, F, T = app.mesh.icosphere(r=0.5).tetrahedralize()
   app.asset.add.tet("ball", V, F, T)

Primitives
----------

Each primitive returns a mesh that can be processed further with the methods
in the next section.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Call
     - Result
   * - ``square(res=32, size=2, ex=None, ey=None, gen_uv=True)``
     - A flat grid centered on the origin. ``res`` is the number of vertices
       along one side. ``ex`` and ``ey`` are the unit vectors of the plane
       (default ``[1, 0, 0]`` and ``[0, 1, 0]``). With ``gen_uv=True`` each
       vertex row carries two UV columns after the position.
   * - ``rectangle(res_x=32, width=2, height=1, ex=None, ey=None, gen_uv=True)``
     - The same for a rectangle. ``res_x`` is the number of vertices along
       the width.
   * - ``circle(n=32, r=1, ntri=1024)``
     - A disk whose boundary has ``n`` points, filled with about ``ntri``
       triangles.
   * - ``icosphere(r=1, subdiv_count=3)``
     - A sphere of radius ``r``.
   * - ``box(width=1, height=1, depth=1)``
     - A box surface.
   * - ``tet_box(width=1, height=1, depth=1)``
     - A box that is already tetrahedral, a ``TetMesh``, so it needs no
       ``tetrahedralize`` call.
   * - ``cylinder(r, min_x, max_x, n)``
     - A cylinder along the x axis from ``min_x`` to ``max_x``, with ``n``
       points around it.
   * - ``cone(Nr=16, Ny=16, Nb=4, radius=0.5, height=2, sharpen=1.0)``
     - A cone.
   * - ``torus(r=1, R=0.25, n=32)``
     - A torus. ``r`` is the major radius and ``R`` the minor radius.
   * - ``mobius(length_split=70, width_split=15, twists=1, r=1, flatness=1, width=1, scale=1)``
     - A Mobius strip.
   * - ``line(p0, p1, n)``
     - A ``Rod`` with ``n`` segments from ``p0`` to ``p1``.
   * - ``load_tri(path)``
     - A triangle mesh read from a file in any format that ``trimesh`` reads.
   * - ``load_tet(path)``
     - A ``TetMesh`` read from a MEDIT ``.mesh`` file. The surface is
       recovered from the tetrahedra when the file has none.
   * - ``preset(name)``
     - A named mesh. It is downloaded on first use and cached, so the first
       call needs internet access.

To build a mesh from arrays, use ``app.mesh.create``: ``create.rod(V, E)``,
``create.tri(V, F)`` and ``create.tet(V, F, T)``. ``create.tri(points)`` with
2D points and no faces makes a closed edge loop through all the points, which
``triangulate`` then fills:

.. code-block:: python

   import numpy as np

   outline = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
   V, F = app.mesh.create.tri(outline).triangulate(target=1024)

Preparing a mesh
----------------

These methods work on a triangle mesh and return a new mesh, except
``normalize`` and ``scale``, which change the vertices in place and return the
same mesh so that calls chain.

``decimate(target_tri)``
   Reduces the mesh to about ``target_tri`` triangles. A target at or above
   the current count returns the mesh unchanged.

``subdivide(n=1, method="midpoint")``
   Subdivides ``n`` times. ``method`` is ``"midpoint"`` or ``"loop"``.

``triangulate(target=1024, min_angle=20)``
   Fills a closed 2D line shape with about ``target`` triangles whose smallest
   angle is at least ``min_angle`` degrees. It refuses a mesh that is not a
   line shape.

``normalize()``
   Centers the mesh on the origin and scales it uniformly so that its longest
   side is 1, so every vertex lies in ``[-0.5, 0.5]``.

``scale(scale_x, scale_y=None, scale_z=None)``
   Scales around the centroid. ``scale_y`` and ``scale_z`` default to
   ``scale_x``.

``tetrahedralize(backend="ftetwild", **options)``
   Fills a closed surface with tetrahedra and returns a ``TetMesh``.

   ``backend="ftetwild"`` accepts open, cracked and non-manifold surfaces. It
   resamples the surface, so the original vertices survive only through the
   mesh's surface mapping: ``interpolate_surface()`` reconstructs positions at
   the original resolution from a deformed tetrahedral surface. Its options
   are ``edge_length_fac`` (default ``0.05``), ``epsilon``, ``stop_energy``,
   ``num_opt_iter``, ``optimize`` (default ``True``), ``simplify`` and
   ``coarsen``. Any other keyword is ignored. ``timeout`` (seconds) stops an
   attempt that stalls, and ``retries`` sets how many times it is repeated;
   the environment variables ``PPF_FTETWILD_TIMEOUT`` and
   ``PPF_FTETWILD_RETRIES`` set both without a code change.
   ``status_callback`` receives progress lines every ``status_interval``
   seconds (default ``5.0``). On Windows on ARM there is no fTetWild, and the
   call raises an error that names the platform instead of switching meshers.

   ``backend="tetgen"`` keeps the input surface exactly, so input vertices map
   one to one to the tetrahedral surface. It requires a clean, closed,
   manifold surface.

Caching
-------

``decimate``, ``subdivide``, ``triangulate``, ``tetrahedralize`` and the
downloaded presets are cached in the app's cache directory. The cache key is
the mesh and the arguments, so a repeated call with the same inputs returns the
stored result. Pass ``cache_dir`` to ``App.create`` to choose the directory,
and call ``app.clear_cache()`` to remove everything in it, for example to
regenerate a tetrahedral mesh.
