# File: _asset_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np


class AssetManager:
    """AssetManager class. Use this to register or remove data assets."""

    def __init__(self):
        """Initialize the asset manager."""
        self._mesh: dict[str, tuple] = {}
        self._add = AssetUploader(self)
        self._fetch = AssetFetcher(self)

    def list(self) -> list[str]:
        """List all the assets in the manager.

        Returns:
            list[str]: A list of asset names.
        """
        return list(self._mesh.keys())

    def remove(self, name: str) -> bool:
        """Remove an asset from the manager.

        Args:
            name (str): The name of the asset to remove.
        Returns:
            bool: True if the asset was removed, False otherwise.
        """
        if name in self._mesh:
            del self._mesh[name]
            return True
        else:
            return False

    def clear(self):
        """Clear all the assets in the manager."""
        self._mesh = {}

    @property
    def mesh(self) -> dict[str, tuple]:
        """Get the mesh dictionary."""
        return self._mesh

    @property
    def add(self) -> "AssetUploader":
        """Get the asset uploader."""
        return self._add

    @property
    def fetch(self) -> "AssetFetcher":
        """Get the asset fetcher."""
        return self._fetch


class AssetUploader:
    """AssetUploader class. Use this to upload data assets to the manager."""

    def __init__(self, manager: AssetManager):
        """Initialize the asset uploader."""
        self._manager = manager

    def check_bounds(self, V: np.ndarray, E: np.ndarray):
        """Check if the indices in E are within the bounds of V."""
        max_ind = np.max(E)
        if max_ind >= V.shape[0]:
            raise Exception(f"E contains index {max_ind} out of bounds ({V.shape[0]})")

    def tri(self, name: str, V: np.ndarray, F: np.ndarray):
        """Upload a triangle mesh to the asset manager.

        Args:
            name (str): The name of the asset.
            V (np.ndarray): The vertices (#x3) of the mesh.
            F (np.ndarray): The triangle elements (#x3) of the mesh.
        """
        if V.shape[1] != 3:
            raise Exception("V must have 3 columns")
        elif F.shape[1] != 3:
            raise Exception("F must have 3 columns")
        if name in self._manager.mesh:
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, F)
            self._manager.mesh[name] = ("tri", V, F)

    def tet(self, name: str, V: np.ndarray, F: np.ndarray, T: np.ndarray):
        """Upload a tetrahedral mesh to the asset manager.

        Args:
            name (str): The name of the asset.
            V (np.ndarray): The vertices (#x3) of the mesh.
            F (np.ndarray): The surface triangle (#x3) elements of the mesh.
            T (np.ndarray): The tetrahedral elements (#x4) of the mesh.
        """
        if V.shape[1] != 3:
            raise Exception("V must have 3 columns")
        elif F.shape[1] != 3:
            raise Exception("F must have 3 columns")
        elif T.shape[1] != 4:
            raise Exception("T must have 4 columns")
        if name in self._manager.mesh:
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, F)
            self.check_bounds(V, T)
            self._manager.mesh[name] = ("tet", V, F, T)

    def rod(self, name: str, V: np.ndarray, E: np.ndarray):
        """Upload a rod mesh to the asset manager.

        Args:
            name (str): The name of the asset.
            V (np.ndarray): The vertices (#x3) of the rod.
            E (np.ndarray): The edges of (#x2) the rod.
        """
        if name in self._manager.mesh:
            raise Exception(f"name '{name}' already exists")
        else:
            self.check_bounds(V, E)
            self._manager.mesh[name] = ("rod", V, E)

    def stitch(self, name: str, stitch: tuple[np.ndarray, np.ndarray]):
        """Upload a stitch mesh to the asset manager.

        Args:
            name (str): The name of the asset.
            stitch (tuple[np.ndarray, np.ndarray]): Ind (index, #x3) and W (weight #x2) of the stitch mesh.
            The weight encodes the liner interpolation between the last two vertices.
        """
        Ind, W = stitch
        if Ind.shape[1] != 3:
            raise Exception("Ind must have 3 columns")
        elif W.shape[1] != 2:
            raise Exception("W must have 2 columns")
        for w in W:
            if abs(np.sum(w) - 1) > 1e-3:
                raise Exception("each row in W must sum to 1")
        if name in self._manager.mesh:
            raise Exception(f"name '{name}' already exists")
        else:
            self._manager.mesh[name] = ("stitch", Ind, W)


class AssetFetcher:
    """AssetFetcher class. Use this to fetch data assets from the manager."""

    def __init__(self, manager: AssetManager):
        """Initialize the asset fetcher."""
        self._manager = manager

    def get_type(self, name: str) -> str:
        """Get the type of the asset.

        Args:
            name (str): The name of the asset.

        Returns:
            str: The type of the asset, such as "tri", "tet", "rod", or "stitch".
        """
        result = self._manager.mesh.get(name, None)
        if result is None:
            raise Exception(f"Asset {name} does not exist")
        else:
            return result[0]

    def get(self, name: str) -> dict[str, np.ndarray]:
        """Get the asset data.

        Args:
            name (str): The name of the asset.

        Returns:
            dict[str, np.ndarray]: The asset data, containing the vertices and elements, such as V, F, T, E, Ind, W.
        """
        result = {}
        if name not in self._manager.mesh:
            raise Exception(f"Asset {name} does not exist")
        else:
            mesh = self._manager.mesh[name]
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
        """Get the triangle mesh data.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The vertices (#x3) and elements (#x3) of the triangle mesh.
        """
        if name not in self._manager.mesh:
            raise Exception(f"Tri {name} does not exist")
        elif self._manager.mesh[name][0] != "tri":
            raise Exception(f"Tri {name} is not a valid")
        else:
            mesh = self._manager.mesh[name]
            assert mesh[0] == "tri"
            return mesh[1], mesh[2]

    def tet(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the tetrahedral mesh data.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The vertices (#x3), surface
            elements (#x3), and tetrahedral (#x4) elements of the mesh.
        """
        if name not in self._manager.mesh:
            raise Exception(f"Tet {name} does not exist")
        elif self._manager.mesh[name][0] != "tet":
            raise Exception(f"Tet {name} is not a valid")
        else:
            mesh = self._manager.mesh[name]
            assert mesh[0] == "tet"
            return mesh[1], mesh[2], mesh[3]

    def rod(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the rod mesh data.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The vertices and edges (#x2) of the rod mesh.
        """
        if name not in self._manager.mesh:
            raise Exception(f"Rod {name} does not exist")
        elif self._manager.mesh[name][0] != "rod":
            raise Exception(f"Rod {name} is not a valid")
        else:
            mesh = self._manager.mesh[name]
            assert mesh[0] == "rod"
            return mesh[1], mesh[2]

    def stitch(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the stitch mesh data.

        Args:
            name (str): The name of the asset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Ind (index, #x3) and W (weight, #x2) of the stitch mesh.
            The weight encodes the liner interpolation between the last two vertices.
        """
        if name not in self._manager.mesh:
            raise Exception(f"Stitch {name} does not exist")
        elif self._manager.mesh[name][0] != "stitch":
            raise Exception(f"Stitch {name} is not a valid")
        else:
            mesh = self._manager.mesh[name]
            assert mesh[0] == "stitch"
            return mesh[1], mesh[2]
