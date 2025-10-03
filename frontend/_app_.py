# File: _app_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import pickle
import shutil

from ._asset_ import AssetManager
from ._extra_ import Extra
from ._mesh_ import MeshManager
from ._plot_ import PlotManager
from ._scene_ import SceneManager
from ._session_ import FixedSession, ParamManager, SessionManager
from ._utils_ import Utils

RECOVERABLE_FIXED_SESSION_NAME = "fixed_session.pickle"


class App:
    @staticmethod
    def create(name: str, cache_dir: str = "") -> "App":
        """Start a new application.

        Args:
            name (str): The name of the application. If not provided, it will use the current time as the name.
            cache_dir (str): The directory to store the cached files. If not provided, it will use `.cache/ppf-cts` directory.

        Returns:
            App: A new instance of the App class.
        """
        return App(name, True, cache_dir)

    @staticmethod
    def load(name: str, cache_dir: str = "") -> "App":
        """Load the saved state of the application if it exists.

        Args:
            name (str): The name of the application.
            cache_dir (str): The directory to store the cached files. If not provided, it will use `.cache/ppf-cts` directory.

        Returns:
            App: A new instance of the App class.
        """
        return App(name, False, cache_dir)

    @staticmethod
    def get_proj_root() -> str:
        """Find the root directory of the project.

        Returns:
            str: Path to the root directory of the project.
        """
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    @staticmethod
    def get_default_param() -> ParamManager:
        """Get default parameters for the application.

        Returns:
            Param: The default parameters.
        """
        return ParamManager()

    @staticmethod
    def busy() -> bool:
        """Returns if the application is running.

        Returns:
            bool: True if the application is running, False otherwise.
        """
        return Utils.busy()

    @staticmethod
    def terminate():
        """Terminates the running application if it is busy."""
        Utils.terminate()

    @staticmethod
    def recover(name: str) -> FixedSession:
        """Recovers the fixed session of the last run application.

        Args:
            name (str): The name used to identify the session.

        Returns:
            FixedSession: The fixed session of the last run application.
        """
        symlink_path = os.path.join(App.get_data_dirpath(), "symlinks", name)
        if os.path.islink(symlink_path):
            session_dir = os.readlink(symlink_path)
            pickle_path = os.path.join(session_dir, RECOVERABLE_FIXED_SESSION_NAME)
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, "rb"))
            else:
                raise Exception(
                    f"No recoverable fixed session found at named location: {name}"
                )
        else:
            # Try to find in git-{branch}/name/session/ directory
            fallback_path = os.path.join(App.get_data_dirpath(), name, "session")
            pickle_path = os.path.join(fallback_path, RECOVERABLE_FIXED_SESSION_NAME)
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, "rb"))
            else:
                raise Exception(f"No session found with name: {name}")

    @staticmethod
    def get_data_dirpath():
        import subprocess

        try:
            git_branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
            ).strip()
            if not git_branch:
                git_branch = "unknown"
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_branch = "unknown"

        return os.path.expanduser(
            os.path.join("~", ".local", "share", "ppf-cts", f"git-{git_branch}")
        )

    def __init__(self, name: str, renew: bool, cache_dir: str = ""):
        """Initializes the App class.

        Creates an instance of the App class with the given name.
        If the renew flag is set to False, it will try to load the saved state of the application from the disk.

        Args:
            name (str): The name of the application.
            renew (bool): A flag to indicate whether to renew the application state.
            cache_dir (str): The directory to store the cached files. If not provided, it will use `.cache/ppf-cts` directory.
        """
        self._extra = Extra()
        self._name = name
        if self.ci:
            self._root = Utils.get_ci_dir()
        else:
            self._root = os.path.join(self.get_data_dirpath(), name)
        self._path = os.path.join(self._root, "app.pickle")
        proj_root = App.get_proj_root()
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = os.path.expanduser(os.path.join("~", ".cache", "ppf-cts"))
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        if os.path.exists(self._path) and not renew:
            with open(self._path, "rb") as f:
                (self._asset, self._scene, self._mesh, self._session, self._plot) = (
                    pickle.load(f)
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
        """
        return self._name

    @property
    def mesh(self) -> MeshManager:
        """Get the mesh manager.

        Returns:
            MeshManager: The mesh manager.
        """
        return self._mesh

    @property
    def plot(self) -> PlotManager:
        """Get the plot manager.

        Returns:
            PlotManager: The plot manager.
        """
        return self._plot

    @property
    def scene(self) -> SceneManager:
        """Get the scene manager.

        Returns:
            SceneManager: The scene manager.
        """
        return self._scene

    @property
    def asset(self) -> AssetManager:
        """Get the asset manager.

        Returns:
            AssetManager: The asset manager.
        """
        return self._asset

    @property
    def extra(self) -> Extra:
        """Get the extra manager.

        Returns:
            Extra: The extra manager.
        """
        return self._extra

    @property
    def session(self) -> SessionManager:
        """Get the session manager.

        Returns:
            SessionManager: The session manager.
        """
        return self._session

    @property
    def ci(self) -> bool:
        """Check if the code is running in a CI environment.

        Returns:
            result (bool): True if the code is running in a CI environment, False otherwise.
        """
        ci_name = Utils.ci_name()
        return ci_name is not None

    @property
    def cache_dir(self) -> str:
        """Get the path to the cache directory.

        Returns:
            path (str): The path to the cache directory.
        """
        return self._cache_dir

    @property
    def ci_dir(self) -> str | None:
        """Get the path to the CI directory.

        Returns:
        path (str): The path to the CI directory.
        """
        if self.ci:
            return Utils.get_ci_dir()
        else:
            return None

    def clear(self) -> "App":
        """Clears the application state."""
        self.asset.clear()
        self._scene.clear()
        self._session.clear()
        return App(self._name, True, self._cache_dir)

    def save(self) -> "App":
        """Saves the application state."""
        with open(self._path, "wb") as f:
            pickle.dump(
                (self.asset, self._scene, self._mesh, self._session, self._plot),
                f,
            )
        return self

    def clear_cache(self) -> "App":
        """Clears the cache directory."""
        if os.path.exists(self._cache_dir) and os.path.isdir(self._cache_dir):
            for item in os.listdir(self._cache_dir):
                item_path = os.path.join(self._cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        open3d_data_path = os.path.expanduser(os.path.join("~", "open3d_data"))
        if os.path.exists(os.path.expanduser(open3d_data_path)):
            shutil.rmtree(open3d_data_path)
        return self
