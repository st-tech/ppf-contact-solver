# File: _app_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import pickle

from typing import Optional

from ._asset_ import AssetManager
from ._extra_ import Extra
from ._mesh_ import MeshManager
from ._plot_ import PlotManager
from ._scene_ import SceneManager
from ._session_ import FixedSession, ParamManager, SessionManager
from ._utils_ import Utils

from . import _rust  # type: ignore[attr-defined]


def _suppress_stale_widget_errors():
    """Suppress TraitErrors caused by stale widget state in saved notebooks.

    When a notebook with saved widget state is loaded, IPY_MODEL_* references
    may point to non-existent widget models, causing TraitErrors. This function
    patches ipywidgets to silently ignore these specific errors.
    """
    try:
        import ipywidgets.widgets.widget as widget_module

        from traitlets import TraitError

        original_set_state = widget_module.Widget.set_state

        def patched_set_state(self, sync_data):
            try:
                original_set_state(self, sync_data)
            except TraitError as e:
                # Suppress errors about stale IPY_MODEL references
                if "IPY_MODEL_" in str(e):
                    pass  # Silently ignore stale widget state errors
                else:
                    raise

        widget_module.Widget.set_state = patched_set_state
    except ImportError:
        pass  # ipywidgets not available


# Apply the patch when this module is imported
_suppress_stale_widget_errors()


class App:
    """High-level entry point for the simulation frontend.

    An ``App`` bundles together the mesh, asset, scene, and session managers
    used to build and run a simulation. It is the canonical starting point
    for notebooks and scripts, typically constructed via :meth:`App.create`
    (new) or :meth:`App.load` (resume).

    Example:
        Build a trivial scene and run it to completion::

            from frontend import App

            app = App.create("intro")

            V, F = app.mesh.square(res=32)
            app.asset.add.tri("sheet", V, F)

            scene = app.scene.create().add("sheet").build()
            session = app.session.create(scene).build()
            session.start(blocking=True)
    """

    @staticmethod
    def create(name: str, cache_dir: str = "") -> "App":
        """Start a new application.

        Args:
            name (str): The name of the application.
            cache_dir (str): The directory used to store cached files. If empty, defaults to ``~/.cache/ppf-cts`` (or a project-relative ``cache/ppf-cts`` on Windows).

        Returns:
            App: A new instance of the App class.

        Example:
            Register a mesh as an asset, drop it into a scene, and run a
            short simulation::

                from frontend import App

                app = App.create("hello")

                V, F = app.mesh.square(res=64, ex=[1, 0, 0], ey=[0, 0, 1])
                app.asset.add.tri("sheet", V, F)

                scene = app.scene.create()
                scene.add("sheet").at(0, 0.6, 0)
                scene = scene.build()

                session = app.session.create(scene)
                session.param.set("frames", 60).set("dt", 0.01)
                session = session.build()
                session.start(blocking=True)
                session.export.animation().zip()
        """
        return App(name, True, cache_dir)

    @staticmethod
    def load(name: str, cache_dir: str = "") -> "App":
        """Load the saved state of the application if it exists.

        If no saved state is found on disk, a fresh App is initialized instead.

        Args:
            name (str): The name of the application.
            cache_dir (str): The directory used to store cached files. If empty, defaults to ``~/.cache/ppf-cts`` (or a project-relative ``cache/ppf-cts`` on Windows).

        Returns:
            App: A new instance of the App class.

        Example:
            Resume a named app, reusing the previously saved assets and
            scene when available::

                from frontend import App

                app = App.load("drape")
                print(app.asset.list())
        """
        return App(name, False, cache_dir)

    @staticmethod
    def get_proj_root() -> str:
        """Find the root directory of the project.

        Returns:
            str: Path to the root directory of the project (parent of frontend).

        Example:
            Resolve paths relative to the project root (for example, the
            bundled ``src`` directory)::

                import os
                from frontend import App

                src_dir = os.path.join(App.get_proj_root(), "src")
                print(os.path.isdir(src_dir))
        """
        return _rust.scene_project_root_from_frontend_file(os.path.abspath(__file__))

    @staticmethod
    def get_default_param() -> ParamManager:
        """Get default parameters for the application.

        Returns:
            ParamManager: The default parameters.

        Example:
            Inspect the default value of a solver parameter before
            building a session::

                from frontend import App

                param = App.get_default_param()
                print(param.get("gravity"))
        """
        return ParamManager()

    @staticmethod
    def busy() -> bool:
        """Return whether a simulation is currently running.

        Returns:
            bool: True if a simulation is running, False otherwise.

        Example:
            Guard against starting a second simulation while one is
            still in progress::

                from frontend import App

                if App.busy():
                    print("solver is still running; skipping")
                else:
                    print("ready to start a new session")
        """
        return Utils.busy()

    @staticmethod
    def is_fast_check() -> bool:
        """Check if fast check mode is enabled.

        Fast check mode forces simulations to run for only 1 frame,
        enabling quick validation of all examples.

        Returns:
            bool: True if fast check mode is enabled.

        Example:
            Branch on fast-check mode to shorten a CI run::

                from frontend import App

                frames = 1 if App.is_fast_check() else 200
                print("frames:", frames)
        """
        return Utils.is_fast_check()

    @staticmethod
    def set_fast_check(enabled: bool = True):
        """Set fast check mode.

        When enabled, simulations are forced to run for only 1 frame,
        enabling quick validation of examples.

        Args:
            enabled: Whether to enable fast check mode. Defaults to True.

        Example:
            Enable fast-check mode before building a session to run a
            single-frame smoke test::

                from frontend import App

                App.set_fast_check(True)
                assert App.is_fast_check()
        """
        Utils.set_fast_check(enabled)

    @staticmethod
    def terminate():
        """Terminate the running simulation if one is busy.

        Example:
            Stop a stuck solver process before starting a new run::

                from frontend import App

                if App.busy():
                    App.terminate()
        """
        Utils.terminate()

    @staticmethod
    def recover(name: str) -> FixedSession:
        """Recover the fixed session previously saved under ``name``.

        The session is located via a symlink in the data directory (or, on
        Windows, a ``.txt`` fallback file holding the target path), and
        otherwise via a fallback ``{data_dir}/{name}/session`` directory.

        Args:
            name (str): The name used to identify the session.

        Returns:
            FixedSession: The recovered fixed session.

        Raises:
            Exception: If no recoverable session can be found for ``name``.

        Example:
            Resume a long-running session after restarting the kernel
            and inspect its current status::

                from frontend import App

                session = App.recover("drape")
                print(session.finished())
        """
        from . import _cbor_bridge_ as _cbor

        def _load_fixed_session(p: str) -> FixedSession:
            with open(p, "rb") as f:
                blob = f.read()
            if _cbor.is_cbor(blob):
                # ``loads_pickle_blob`` handles both the current
                # dict-shaped payload and the older raw-bytes
                # envelopes left on disk by earlier builds.
                pickled = _cbor.loads_pickle_blob(blob, _cbor.KIND_FIXED_SESSION)
                return pickle.loads(pickled)
            return pickle.loads(blob)

        data_dir = App.get_data_dirpath()
        pickle_path = _rust.recover_session_path(name, data_dir)
        return _load_fixed_session(pickle_path)

    @staticmethod
    def get_data_dirpath():
        return _rust.get_data_dirpath(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    def __init__(self, name: str, renew: bool, cache_dir: str = ""):
        """Initialize the App.

        Creates an App instance with the given name. When ``renew`` is
        False, the constructor attempts to load a previously saved app
        pickle from disk; if none exists, it initializes a fresh state.

        Args:
            name (str): The name of the application.
            renew (bool): If True, discard any saved state and start fresh. If False, load saved state when available.
            cache_dir (str): The directory used to store cached files. If empty, defaults to ``~/.cache/ppf-cts`` (or a project-relative ``cache/ppf-cts`` on Windows).
        """
        self._extra = Extra()
        self._name = name
        # Keep `App.get_data_dirpath` patchable from tests (the CBOR
        # roundtrip suite monkeypatches it onto a tmp path), so resolve
        # the data dir through the bound method first and feed the
        # result into the bundled Rust path resolver only for the
        # non-data-dir fields.
        data_dirpath = self.get_data_dirpath()
        ci_dir = Utils.get_ci_dir() if self.ci else None
        self._root = ci_dir if ci_dir else os.path.join(data_dirpath, name)
        proj_root = App.get_proj_root()
        self._path = _rust.app_pickle_path(name, data_dirpath, ci_dir)
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = _rust.default_cache_dir(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            )
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        if os.path.exists(self._path) and not renew:
            from . import _cbor_bridge_ as _cbor

            with open(self._path, "rb") as f:
                blob = f.read()
            if _cbor.is_cbor(blob):
                # ``loads_pickle_blob`` accepts both the current
                # dict-shaped payload (with ``pickle_blob`` field) and
                # the older raw-bytes payload left on disk by
                # earlier App.save versions.
                pickled = _cbor.loads_pickle_blob(blob, _cbor.KIND_APP_STATE)
                (self._asset, self._scene, self._mesh, self._session, self._plot) = (
                    pickle.loads(pickled)
                )
            else:
                (self._asset, self._scene, self._mesh, self._session, self._plot) = (
                    pickle.loads(blob)
                )
        else:
            os.makedirs(self._root, exist_ok=True)
            self._plot = PlotManager()
            self._session = SessionManager(
                self._name, self._root, proj_root, App.get_data_dirpath()
            )
            self._asset = AssetManager()
            self._scene = SceneManager(self._plot, self.asset)
            self._mesh = MeshManager(self._cache_dir)

    @property
    def name(self) -> str:
        """Get the name of the application.

        Returns:
            str: The name of the application.

        Example:
            Read back the name assigned at construction time::

                app = App.create("hello")
                assert app.name == "hello"
        """
        return self._name

    @property
    def mesh(self) -> MeshManager:
        """Get the mesh manager.

        Returns:
            MeshManager: The mesh manager.

        Example:
            Use ``app.mesh`` to build primitives or load presets::

                app = App.create("demo")
                V, F = app.mesh.square(res=64)
                V2, F2, T = app.mesh.preset("armadillo").tetrahedralize()
        """
        return self._mesh

    @property
    def plot(self) -> PlotManager:
        """Get the plot manager.

        Returns:
            PlotManager: The plot manager.

        Example:
            Use ``app.plot`` to create interactive viewers from mesh data::

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                plot = app.plot.create().tri(V, F).build()
        """
        return self._plot

    @property
    def scene(self) -> SceneManager:
        """Get the scene manager.

        Returns:
            SceneManager: The scene manager.

        Example:
            Use ``app.scene`` to assemble a scene from registered assets::

                app = App.create("demo")
                scene = app.scene.create().add("sheet").at(0, 0.5, 0).build()
        """
        return self._scene

    @property
    def asset(self) -> AssetManager:
        """Get the asset manager.

        Returns:
            AssetManager: The asset manager.

        Example:
            Use ``app.asset`` to register mesh data and list what exists::

                app = App.create("demo")
                V, F = app.mesh.square(res=32)
                app.asset.add.tri("sheet", V, F)
                print(app.asset.list())
        """
        return self._asset

    @property
    def extra(self) -> Extra:
        """Get the extra manager.

        Returns:
            Extra: The extra manager.

        Example:
            Use ``app.extra`` for auxiliary helpers that do not belong to
            the core asset/scene/session flow::

                app = App.create("demo")
                helpers = app.extra
        """
        return self._extra

    @property
    def session(self) -> SessionManager:
        """Get the session manager.

        Returns:
            SessionManager: The session manager.

        Example:
            Use ``app.session`` to build and start a solver run from a
            finalized scene::

                app = App.create("demo")
                scene = app.scene.create().add("sheet").build()
                session = app.session.create(scene).build()
                session.start(blocking=True)
        """
        return self._session

    @property
    def ci(self) -> bool:
        """Check if the code is running in a CI environment.

        Returns:
            bool: True if the code is running in a CI environment, False otherwise.

        Example:
            Skip long-running steps when executing in CI::

                app = App.create("demo")
                if not app.ci:
                    session.start(blocking=True)
        """
        ci_name = Utils.ci_name()
        return ci_name is not None

    @property
    def cache_dir(self) -> str:
        """Get the path to the cache directory.

        Returns:
            str: The path to the cache directory.

        Example:
            Inspect where downloaded preset meshes and other cached files
            are stored on disk::

                app = App.create("demo")
                print(app.cache_dir)
        """
        return self._cache_dir

    @property
    def ci_dir(self) -> Optional[str]:
        """Get the path to the CI directory, if running under CI.

        Returns:
            Optional[str]: The CI directory path, or None when not running in CI.

        Example:
            Resolve a path under the CI workspace only when the app is
            running inside CI::

                app = App.create("demo")
                if app.ci_dir is not None:
                    print("CI workspace:", app.ci_dir)
        """
        if self.ci:
            return Utils.get_ci_dir()
        else:
            return None

    def clear(self) -> "App":
        """Clear the application state and return a fresh App instance.

        Example:
            Discard existing assets, scenes, and sessions before
            rebuilding a workflow from scratch::

                from frontend import App

                app = App.create("drape")
                app = app.clear()
                print(app.asset.list())
        """
        self.asset.clear()
        self._scene.clear()
        self._session.clear()
        return App(self._name, True, self._cache_dir)

    def save(self) -> "App":
        """Save the application state to disk and return self.

        Example:
            Persist the current assets and scene so a later call to
            :meth:`App.load` can resume them::

                from frontend import App

                app = App.create("drape")
                V, F = app.mesh.square(res=64)
                app.asset.add.tri("sheet", V, F)
                app.save()
        """
        from . import _cbor_bridge_ as _cbor

        # Native CBOR envelope: structured fields the user can inspect
        # with any CBOR reader, plus a ``pickle_blob`` carrying the
        # deep manager graph (no schema-level CBOR shape today).
        pickled = pickle.dumps(
            (self.asset, self._scene, self._mesh, self._session, self._plot),
        )
        payload = self._to_cbor_dict(pickled)
        with open(self._path, "wb") as f:
            f.write(_cbor.dumps_envelope(_cbor.KIND_APP_STATE, payload))
        return self

    def _to_cbor_dict(self, pickle_blob: bytes) -> dict:
        """Build the native CBOR map payload written by :meth:`save`.

        Inspectable metadata (project name, on-disk root) lives at the
        top of the map; ``pickle_blob`` carries the manager graph for
        rehydration in :meth:`__init__`.
        """
        return _rust.app_to_cbor_dict(
            self._name,
            self._root,
            list(self._asset.list()),
            pickle_blob,
        )

    def clear_cache(self) -> "App":
        """Remove all files and subdirectories under the cache directory.

        Example:
            Force downloaded meshes and other cached artifacts to be
            regenerated on the next run::

                from frontend import App

                app = App.create("drape")
                app.clear_cache()
        """
        _rust.clear_cache_dir(self._cache_dir)
        return self

    @staticmethod
    def run_tests() -> bool:
        """Run all frontend tests.

        Returns:
            bool: True if all tests pass, False otherwise.

        Example:
            Run the bundled frontend self-tests and exit non-zero on
            failure::

                import sys
                from frontend import App

                if not App.run_tests():
                    sys.exit(1)
        """
        from .tests._runner_ import run_all_tests

        return run_all_tests()
