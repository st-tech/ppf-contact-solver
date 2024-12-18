# File: _app_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ._asset_ import AssetManager
from ._extra_ import Extra
from ._scene_ import SceneManager
from ._mesh_ import MeshManager
from ._session_ import SessionManager
from ._plot_ import PlotManager
import pickle
import os
import shutil


class App:
    def __init__(self, name: str, renew: bool = False, cache_dir: str = ""):
        self.extra = Extra()
        self._name = name
        self._root = os.path.expanduser(
            os.path.join("~", ".local", "share", "ppf-cts", name)
        )
        self._path = os.path.join(self._root, "app.pickle")
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.expanduser(os.path.join("~", ".cache", "ppf-cts"))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(self._path) and not renew:
            (self.asset, self.scene, self.mesh, self.session, self.plot) = pickle.load(
                open(self._path, "rb")
            )
        else:
            os.makedirs(self._root, exist_ok=True)
            self.plot = PlotManager()
            self.session = SessionManager(self._root, proj_root, self.save)
            self.asset = AssetManager()
            self.scene = SceneManager(self.plot, self.asset, self.save)
            self.mesh = MeshManager(self.cache_dir)

    def clear(self) -> "App":
        self.asset.clear()
        self.scene.clear()
        self.session.clear()
        return App(self._name, True, self.cache_dir)

    def darkmode(self) -> "App":
        self.plot.darkmode(True)
        return self

    def save(self) -> "App":
        pickle.dump(
            (self.asset, self.scene, self.mesh, self.session, self.plot),
            open(self._path, "wb"),
        )
        return self

    def clear_cache(self) -> "App":
        if os.path.exists(self.cache_dir) and os.path.isdir(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                item_path = os.path.join(self.cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        return self
