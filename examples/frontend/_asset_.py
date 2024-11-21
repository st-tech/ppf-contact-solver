# File: _asset_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np


class AssetManager:
    def __init__(self):
        self._mesh: dict[str, tuple] = {}
        self.add = AssetUploader(self)
        self.fetch = AssetFetcher(self)

    def list(self) -> list[str]:
        return list(self._mesh.keys())

    def remove(self, name: str) -> bool:
        if name in self._mesh.keys():
            del self._mesh[name]
            return True
        else:
            return False

    def clear(self):
        self._mesh = {}


class AssetUploader:
    def __init__(self, manager: AssetManager):
        self._manager = manager

    def check_bounds(self, V: np.ndarray, E: np.ndarray):
        max_ind = np.max(E)
        if max_ind >= V.shape[0]:
            raise Exception(f"E contains index {max_ind} out of bounds ({V.shape[0]})")

    def tri(self, name: str, V: np.ndarray, F: np.ndarray):
        if V.shape[1] != 3:
            raise Exception("V must have 3 columns")
        elif F.shape[1] != 3:
            raise Exception("F must have 3 columns")
        if name in self._manager._mesh.keys():
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, F)
            self._manager._mesh[name] = ("tri", V, F)

    def tet(self, name: str, V: np.ndarray, F: np.ndarray, T: np.ndarray):
        if V.shape[1] != 3:
            raise Exception("V must have 3 columns")
        elif F.shape[1] != 3:
            raise Exception("F must have 3 columns")
        elif T.shape[1] != 4:
            raise Exception("T must have 4 columns")
        if name in self._manager._mesh.keys():
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, F)
            self.check_bounds(V, T)
            self._manager._mesh[name] = ("tet", V, F, T)

    def rod(self, name: str, V: np.ndarray, E: np.ndarray):
        if name in self._manager._mesh.keys():
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, E)
            self._manager._mesh[name] = ("rod", V, E)

    def stitch(self, name: str, stitch: tuple[np.ndarray, np.ndarray]):
        Ind, W = stitch
        if Ind.shape[1] != 3:
            raise Exception("Ind must have 3 columns")
        elif W.shape[1] != 2:
            raise Exception("W must have 2 columns")
        for w in W:
            if abs(np.sum(w) - 1) > 1e-3:
                raise Exception("each row in W must sum to 1")
        if name in self._manager._mesh.keys():
            raise Exception(f"name '{name}' already exists")
        else:
            self._manager._mesh[name] = ("stitch", Ind, W)


class AssetFetcher:
    def __init__(self, manager: AssetManager):
        self._manager = manager

    def get(self, name: str) -> dict[str, np.ndarray]:
        result = {}
        if name not in self._manager._mesh.keys():
            raise Exception(f"Asset {name} does not exist")
        else:
            mesh = self._manager._mesh[name]
            if mesh[0] == "tri":
                result["V"] = mesh[1]
                result["F"] = mesh[2]
            elif mesh[0] == "tet":
                result["V"] = mesh[1]
                result["F"] = mesh[2]
                result["T"] = mesh[3]
            elif mesh[0] == "rod":
                result["V"] = mesh[1]
                result["E"] = mesh[2]
            elif mesh[0] == "stitch":
                result["Ind"] = mesh[1]
                result["W"] = mesh[2]
            return result

    def tri(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._manager._mesh.keys():
            raise Exception(f"Tri {name} does not exist")
        elif self._manager._mesh[name][0] != "tri":
            raise Exception(f"Tri {name} is not a valid")
        else:
            mesh = self._manager._mesh[name]
            assert mesh[0] == "tri"
            return mesh[1], mesh[2]

    def tet(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name not in self._manager._mesh.keys():
            raise Exception(f"Tet {name} does not exist")
        elif self._manager._mesh[name][0] != "tet":
            raise Exception(f"Tet {name} is not a valid")
        else:
            mesh = self._manager._mesh[name]
            assert mesh[0] == "tet"
            return mesh[1], mesh[2], mesh[3]

    def rod(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._manager._mesh.keys():
            raise Exception(f"Rod {name} does not exist")
        elif self._manager._mesh[name][0] != "rod":
            raise Exception(f"Rod {name} is not a valid")
        else:
            mesh = self._manager._mesh[name]
            assert mesh[0] == "rod"
            return mesh[1], mesh[2]

    def stitch(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._manager._mesh.keys():
            raise Exception(f"Stitch {name} does not exist")
        elif self._manager._mesh[name][0] != "stitch":
            raise Exception(f"Stitch {name} is not a valid")
        else:
            mesh = self._manager._mesh[name]
            assert mesh[0] == "stitch"
            return mesh[1], mesh[2]
