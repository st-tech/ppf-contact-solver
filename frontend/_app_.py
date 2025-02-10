# File: _app_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ._asset_ import AssetManager
from ._extra_ import Extra
from ._scene_ import SceneManager
from ._mesh_ import MeshManager
from ._session_ import SessionManager, Param
from ._plot_ import PlotManager
from ._utils_ import Utils
import pickle
import os
import shutil
from typing import Optional


class App:
    @staticmethod
    def create(name: str, cache_dir: str = "") -> "App":
        """Start a new application.

        Args:
            name (str): The name of the application.
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
    def get_default_param() -> Param:
        """Get default parameters for the application.

        Returns:
            Param: The default parameters.
        """
        return Param(App.get_proj_root())

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
            self._root = os.path.expanduser(
                os.path.join("~", ".local", "share", "ppf-cts", name)
            )
        self._path = os.path.join(self._root, "app.pickle")
        proj_root = App.get_proj_root()
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.expanduser(os.path.join("~", ".cache", "ppf-cts"))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(self._path) and not renew:
            (self._asset, self._scene, self.mesh, self._session, self._plot) = (
                pickle.load(open(self._path, "rb"))
            )
        else:
            os.makedirs(self._root, exist_ok=True)
            self._plot = PlotManager()
            self._session = SessionManager(self._root, proj_root, self.save)
            self._asset = AssetManager()
            self._scene = SceneManager(self._plot, self.asset, self.save)
            self.mesh = MeshManager(self.cache_dir)  #: MeshManager: The mesh manager.

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
        if ci_name is not None:
            return True
        else:
            return False

    @property
    def ci_dir(self) -> Optional[str]:
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
        return App(self._name, True, self.cache_dir)

    def darkmode(self) -> "App":
        """Tunrs on the dark mode."""
        self._plot.darkmode(True)
        return self

    def save(self) -> "App":
        if not self.ci:
            """Saves the application state."""
            pickle.dump(
                (self.asset, self._scene, self.mesh, self._session, self._plot),
                open(self._path, "wb"),
            )
        return self

    def clear_cache(self) -> "App":
        """Clears the cache directory."""
        if os.path.exists(self.cache_dir) and os.path.isdir(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                item_path = os.path.join(self.cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        open3d_data_path = os.path.expanduser(os.path.join("~", "open3d_data"))
        if os.path.exists(os.path.expanduser(open3d_data_path)):
            shutil.rmtree(open3d_data_path)
        return self
