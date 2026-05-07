# File: _asset_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np

from . import _rust  # type: ignore[attr-defined]


class AssetManager:
    """Registry for mesh data assets.

    Holds uploaded meshes keyed by name and exposes an
    :class:`AssetUploader` for registering new assets and an
    :class:`AssetFetcher` for retrieving them.

    Storage lives in a Rust-backed ``_ppf_cts_py.AssetRegistry``
    that holds owning references to the user's numpy arrays so dtypes
    survive a roundtrip exactly: an ``np.float64`` vertex buffer reads
    back as ``np.float64``, an ``np.int32`` face buffer reads back as
    ``np.int32``.

    Example:
        Access the manager via :attr:`App.asset` to register a mesh,
        list the currently known names, and fetch arrays back out::

            from frontend import App

            app = App.create("demo")
            V, F = app.mesh.square(res=64)
            app.asset.add.tri("sheet", V, F)
            print(app.asset.list())
            V2, F2 = app.asset.fetch.tri("sheet")
    """

    def __init__(self):
        """Initialize the asset manager with an empty registry."""
        self._registry = _rust.AssetRegistry()
        self._add = AssetUploader(self)
        self._fetch = AssetFetcher(self)

    def list(self) -> list[str]:
        """List the names of all assets currently registered.

        Returns:
            list[str]: The registered asset names.

        Example:
            Check whether an asset has already been uploaded before
            re-registering it::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                assert "sheet" in app.asset.list()
        """
        return list(self._registry.list())

    def remove(self, name: str) -> bool:
        """Remove an asset from the manager.

        Args:
            name (str): The name of the asset to remove.

        Returns:
            bool: True if the asset existed and was removed, False if
            no asset with that name was registered.

        Example:
            Drop an asset by name and re-upload a fresh version::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                app.asset.remove("sheet")
                app.asset.add.tri("sheet", V, F)
        """
        return self._registry.remove(name)

    def clear(self):
        """Remove all assets from the manager.

        Example:
            Reset the asset registry before reloading a different set
            of meshes::

                from frontend import App

                app = App.create("demo")
                app.asset.clear()
                V, F, T = app.mesh.preset("armadillo").tetrahedralize()
                app.asset.add.tet("body", V, F, T)
        """
        self._registry.clear()

    def __getstate__(self) -> dict:
        """Pickle state. Snapshots the Rust registry into a plain
        ``dict[name, {"kind": str, "arrays": dict[str, ndarray]}]`` via
        :meth:`_ppf_cts_py.AssetRegistry.snapshot` so the wrapper
        roundtrips through ``pickle`` / ``copy.deepcopy``.
        """
        return {"snapshot": dict(self._registry.snapshot())}

    def __setstate__(self, state: dict) -> None:
        """Rehydrate the registry from :meth:`__getstate__` output via
        :meth:`_ppf_cts_py.AssetRegistry.restore`."""
        self._registry = _rust.AssetRegistry()
        self._add = AssetUploader(self)
        self._fetch = AssetFetcher(self)
        snapshot = state.get("snapshot", {})
        if snapshot:
            self._registry.restore(snapshot)

    @property
    def add(self) -> "AssetUploader":
        """The uploader used to register new assets.

        Example:
            Use ``app.asset.add`` to register triangle, tetrahedral, or
            rod meshes by name::

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
        """
        return self._add

    @property
    def fetch(self) -> "AssetFetcher":
        """The fetcher used to retrieve registered assets.

        Example:
            Use ``app.asset.fetch`` to pull mesh arrays back out by name::

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                V2, F2 = app.asset.fetch.tri("sheet")
        """
        return self._fetch


class AssetUploader:
    """Uploader used to register mesh assets with an :class:`AssetManager`.

    Example:
        Access the uploader via :attr:`App.asset.add` and register
        triangle, tetrahedral, and rod assets::

            from frontend import App

            app = App.create("demo")
            V, F = app.mesh.square(res=64)
            app.asset.add.tri("sheet", V, F)
            V, F, T = app.mesh.icosphere(r=0.25, subdiv_count=4).tetrahedralize()
            app.asset.add.tet("ball", V, F, T)
    """

    def __init__(self, manager: AssetManager):
        """Initialize the uploader bound to ``manager``."""
        self._manager = manager

    def check_bounds(self, V: np.ndarray, E: np.ndarray):
        """Check that every index in ``E`` refers to a valid row of ``V``.

        Args:
            V (np.ndarray): Vertex array whose row count defines the
                upper bound.
            E (np.ndarray): Element/index array to validate.

        Raises:
            Exception: If any index in ``E`` is greater than or equal
                to ``V.shape[0]``.

        Example:
            Validate a hand-built mesh before uploading it::

                import numpy as np
                from frontend import App

                app = App.create("demo")
                V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
                F = np.array([[0, 1, 2]], dtype=np.uint32)
                app.asset.add.check_bounds(V, F)
        """
        _rust.check_bounds(E, V.shape[0])

    def tri(self, name: str, V: np.ndarray, F: np.ndarray):
        """Upload a triangle mesh to the asset manager.

        Args:
            name (str): The name of the asset. Must not already exist.
            V (np.ndarray): The vertices (#x3 or #x5) of the mesh.
                If #x5, columns 0-2 are xyz coordinates and columns 3-4
                are UV coordinates.
            F (np.ndarray): The triangle elements (#x3) of the mesh.

        Raises:
            Exception: If ``V`` does not have 3 or 5 columns, ``F`` does
                not have 3 columns, ``name`` already exists, or any
                index in ``F`` is out of bounds for ``V``.

        Example:
            Register a square sheet and then an icosphere as triangle
            assets::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=128, ex=[1, 0, 0], ey=[0, 0, 1])
                app.asset.add.tri("sheet", V, F)
                V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
                app.asset.add.tri("sphere", V, F)
        """
        _rust.check_tri_v_cols(V)
        _rust.check_cols(F, "F", 3)
        if V.shape[1] == 5:
            # Extract xyz and UV coordinates
            V_xyz = V[:, :3].copy()
            V_uv = V[:, 3:5].copy()
            self.check_bounds(V_xyz, F)
            self._manager._registry.add_tri(name, V_xyz, F, V_uv)
        else:
            self.check_bounds(V, F)
            # Copy so later caller mutation doesn't reach the manager.
            self._manager._registry.add_tri(name, V.copy(), F, None)

    def tet(self, name: str, V: np.ndarray, F: np.ndarray, T: np.ndarray):
        """Upload a tetrahedral mesh to the asset manager.

        Args:
            name (str): The name of the asset. Must not already exist.
            V (np.ndarray): The vertices (#x3) of the mesh.
            F (np.ndarray): The surface triangle elements (#x3) of the mesh.
            T (np.ndarray): The tetrahedral elements (#x4) of the mesh.

        Raises:
            Exception: If the column counts of ``V``, ``F``, or ``T``
                are wrong, ``name`` already exists, or any index in
                ``F`` or ``T`` is out of bounds for ``V``.

        Example:
            Tetrahedralize an icosphere and register it as a volumetric
            asset::

                from frontend import App

                app = App.create("demo")
                V, F, T = app.mesh.icosphere(r=0.25, subdiv_count=4).tetrahedralize()
                app.asset.add.tet("sphere", V, F, T)
        """
        _rust.check_cols(V, "V", 3)
        _rust.check_cols(F, "F", 3)
        _rust.check_cols(T, "T", 4)
        self.check_bounds(V, F)
        self.check_bounds(V, T)
        self._manager._registry.add_tet(name, V, F, T)

    def rod(self, name: str, V: np.ndarray, E: np.ndarray):
        """Upload a rod mesh to the asset manager.

        Args:
            name (str): The name of the asset. Must not already exist.
            V (np.ndarray): The vertices (#x3) of the rod.
            E (np.ndarray): The edges (#x2) of the rod.

        Raises:
            Exception: If ``name`` already exists or any index in ``E``
                is out of bounds for ``V``.

        Example:
            Build a polyline from a numpy array of points and register
            it as a rod asset::

                import numpy as np
                from frontend import App

                app = App.create("demo")
                V = np.linspace([0, 0, 0], [1, 0, 0], 20)
                E = np.array([[i, i + 1] for i in range(len(V) - 1)], dtype=np.uint32)
                app.asset.add.rod("strand", V, E)
        """
        self.check_bounds(V, E)
        self._manager._registry.add_rod(name, V, E)

    def stitch(self, name: str, stitch: tuple[np.ndarray, np.ndarray]):
        """Upload a stitch asset to the asset manager.

        Args:
            name (str): The name of the asset. Must not already exist.
            stitch (tuple[np.ndarray, np.ndarray]): A pair ``(Ind, W)``
                where ``Ind`` is the index array (#x4) and ``W`` is the
                weight array (#x4) of the stitch.

        Raises:
            Exception: If ``Ind`` or ``W`` does not have 4 columns, or
                if ``name`` already exists.

        Example:
            Load a CIPC-format stitch mesh and register its stitch
            information alongside the dress geometry::

                from frontend import App

                app = App.create("demo")
                V, F, S = app.extra.load_CIPC_stitch_mesh("dress_stage.obj")
                app.asset.add.tri("dress", V, F)
                app.asset.add.stitch("glue", S)
        """
        Ind, W = stitch
        _rust.check_cols(Ind, "Ind", 4)
        _rust.check_cols(W, "W", 4)
        self._manager._registry.add_stitch(name, Ind, W)


class AssetFetcher:
    """Fetcher used to retrieve mesh assets from an :class:`AssetManager`.

    Example:
        Access the fetcher via :attr:`App.asset.fetch` to pull arrays
        for a registered mesh::

            from frontend import App

            app = App.create("demo")
            V, F = app.mesh.square(res=64)
            app.asset.add.tri("sheet", V, F)
            V2, F2 = app.asset.fetch.tri("sheet")
    """

    def __init__(self, manager: AssetManager):
        """Initialize the fetcher bound to ``manager``."""
        self._manager = manager

    def get_type(self, name: str) -> str:
        """Return the type tag of a registered asset.

        Args:
            name (str): The name of the asset.

        Returns:
            str: The type of the asset: one of ``"tri"``, ``"tet"``,
            ``"rod"``, or ``"stitch"``.

        Raises:
            Exception: If no asset is registered under ``name``.

        Example:
            Branch on the type tag before retrieving arrays of the
            correct shape::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                assert app.asset.fetch.get_type("sheet") == "tri"
        """
        return self._manager._registry.get_type(name)

    def get(self, name: str) -> dict[str, np.ndarray]:
        """Return the raw arrays stored for an asset.

        The keys present in the returned dictionary depend on the asset
        type:

        * ``"tri"``: ``V``, ``F``, and optionally ``UV``.
        * ``"tet"``: ``V``, ``F``, ``T``.
        * ``"rod"``: ``V``, ``E``.
        * ``"stitch"``: ``Ind``, ``W``.

        Args:
            name (str): The name of the asset.

        Returns:
            dict[str, np.ndarray]: Mapping from array name to array.

        Raises:
            Exception: If no asset is registered under ``name``.

        Example:
            Retrieve the raw arrays of a registered mesh as a
            dictionary and inspect their shapes::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                data = app.asset.fetch.get("sheet")
                print(data["V"].shape, data["F"].shape)
        """
        return self._manager._registry.get(name)

    def tri(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the vertex and face arrays of a triangle mesh asset.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The vertices (#x3) and the
            triangle elements (#x3) of the mesh.

        Raises:
            Exception: If no asset is registered under ``name``, or the
                asset exists but is not a triangle mesh.

        Example:
            Plot a previously registered triangle asset after fetching
            its arrays::

                from frontend import App

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                V, F = app.asset.fetch.tri("sheet")
                app.plot.create().tri(V, F)
        """
        return self._manager._registry.get_tri(name)

    def tet(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the arrays of a tetrahedral mesh asset.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The vertices
            (#x3), the surface triangle elements (#x3), and the
            tetrahedral elements (#x4) of the mesh.

        Raises:
            Exception: If no asset is registered under ``name``, or the
                asset exists but is not a tetrahedral mesh.

        Example:
            Fetch a tet asset and plot its volumetric connectivity::

                from frontend import App

                app = App.create("demo")
                V, F, T = app.mesh.icosphere(r=0.25, subdiv_count=3).tetrahedralize()
                app.asset.add.tet("ball", V, F, T)
                V, F, T = app.asset.fetch.tet("ball")
                app.plot.create().tet(V, T)
        """
        return self._manager._registry.get_tet(name)

    def rod(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the vertex and edge arrays of a rod mesh asset.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The vertices (#x3) and the
            edges (#x2) of the rod.

        Raises:
            Exception: If no asset is registered under ``name``, or the
                asset exists but is not a rod mesh.

        Example:
            Retrieve the vertex and edge arrays of a registered rod
            asset::

                import numpy as np
                from frontend import App

                app = App.create("demo")
                V = np.linspace([0, 0, 0], [1, 0, 0], 10)
                E = np.array([[i, i + 1] for i in range(len(V) - 1)], dtype=np.uint32)
                app.asset.add.rod("strand", V, E)
                V, E = app.asset.fetch.rod("strand")
        """
        return self._manager._registry.get_rod(name)

    def stitch(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the index and weight arrays of a stitch asset.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The index array ``Ind``
            (#x4) and the weight array ``W`` (#x4) of the stitch, as
            uploaded via :meth:`AssetUploader.stitch`.

        Raises:
            Exception: If no asset is registered under ``name``, or the
                asset exists but is not a stitch.

        Example:
            Fetch the index and weight arrays for a registered stitch
            asset::

                from frontend import App

                app = App.create("demo")
                V, F, S = app.extra.load_CIPC_stitch_mesh("dress_stage.obj")
                app.asset.add.stitch("glue", S)
                Ind, W = app.asset.fetch.stitch("glue")
        """
        return self._manager._registry.get_stitch(name)
