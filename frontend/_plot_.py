# File: _plot_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from meshplot import plot
import os
from meshplot.Viewer import Viewer
import pythreejs as p3s
import numpy as np
from IPython.display import display

"""Default shading settings for light mode."""
LIGHT_DEFAULT_SHADING = {
    "flat": False,
    "wireframe": True,
    "line_width": 1.0,
    "line_color": "black",
    "point_color": "black",
}

"""Default shading settings for dark mode."""
DARK_DEFAULT_SHADING = {
    "flat": False,
    "wireframe": True,
    "line_width": 1.0,
    "background": "#222222",
    "line_color": "white",
    "point_color": "white",
}


def in_jupyter_notebook():
    """Determine if the code is running in a Jupyter notebook."""
    dirpath = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(dirpath, ".CLI")):
        return False
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except (NameError, ImportError):
        return False


class PlotManager:
    """PlotManager class. Use this to create a plot."""

    def __init__(self) -> None:
        """Initialize the plot manager."""
        self._darkmode = True
        self._in_jupyter_notebook = in_jupyter_notebook()

    def darkmode(self, darkmode: bool) -> None:
        """Turn on or off dark mode.

        Args:
            darkmode (bool): True to turn on dark mode, False otherwise.
        """
        self._darkmode = darkmode

    def create(self) -> "Plot":
        """Create a plot."""
        return Plot(self._darkmode)

    def is_jupyter_notebook(self) -> bool:
        """Check if the code is running in a Jupyter notebook."""
        return self._in_jupyter_notebook


class PlotAdder:
    """PlotAdder class. Use this to add elements to a plot."""

    def __init__(self, parent: "Plot") -> None:
        """Initialize the plot adder."""
        self._parent = parent
        self._in_jupyter_notebook = in_jupyter_notebook()

    def tri(self, vert: np.ndarray, tri: np.ndarray, color: np.ndarray) -> "Plot":
        """Add a triangle mesh to the plot.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            tri (np.ndarray): The triangle elements (#x3) of the mesh.
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].

        Returns:
            Plot: The plot object.
        """

        if self._in_jupyter_notebook:
            viewer = self._parent._viewer
            shading = self._parent._shading
            if viewer is None:
                raise Exception("No plot to add to")
            else:
                viewer.add_mesh(vert, tri, color, shading=shading)
        return self._parent

    def edge(self, vert: np.ndarray, edge: np.ndarray) -> "Plot":
        """Add edges to the plot.

        Args:
            vert (np.ndarray): The vertices (#x3) of the edges.
            edge (np.ndarray): The edge elements (#x2) of the edges.

        Returns:
            Plot: The plot object.
        """
        if self._in_jupyter_notebook:
            viewer = self._parent._viewer
            shading = self._parent._shading
            edge = edge.copy().astype(np.uint32)
            vert = vert.copy().astype(np.float32)
            if viewer is None:
                raise Exception("No plot to add to")
            else:
                geometry = p3s.BufferGeometry(
                    attributes={
                        "position": p3s.BufferAttribute(vert, normalized=False),
                        "index": p3s.BufferAttribute(edge.flatten(), normalized=False),
                    }
                )
                material = p3s.LineBasicMaterial(
                    linewidth=shading["line_width"], color=shading["line_color"]
                )
                line = p3s.Line(geometry=geometry, material=material)
                obj = {
                    "geometry": geometry,
                    "mesh": line,
                    "material": material,
                    "max": np.max(vert, axis=0),
                    "min": np.min(vert, axis=0),
                    "type": "Lines",
                    "wireframe": None,
                }
                viewer._Viewer__add_object(obj)  # type: ignore
        return self._parent

    def point(self, vert: np.ndarray) -> "Plot":
        """Add points to the plot.

        Args:
            vert (np.ndarray): The vertices (#x3) of the points.

        Returns:
            Plot: The plot object.
        """
        if self._in_jupyter_notebook:
            viewer = self._parent._viewer
            shading = self._parent._shading
            if viewer is None:
                raise Exception("No plot to add to")
            else:
                viewer.add_points(vert, shading=shading)
        return self._parent


class Plot:
    """Plot class. Use this to create a plot."""

    def __init__(self, _darkmode: bool):
        """Initialize the plot.

        Args:
            _darkmode (bool): True to turn on dark mode, False otherwise.
        """
        self._in_jupyter_notebook = in_jupyter_notebook()
        self._darkmode = _darkmode
        self._viewer = None
        self._shading = {}
        self.add = PlotAdder(self)

    def is_jupyter_notebook(self) -> bool:
        """Check if the code is running in a Jupyter notebook."""
        return self._in_jupyter_notebook

    def to_html(self, path: str = ""):
        """Export an HTML file with the plot.

        Args:
            path (str): The filename to save the HTML file.
        """
        if self._in_jupyter_notebook:
            if self._viewer is None:
                raise Exception("No plot to save")
            else:
                self._viewer.save(path)

    def has_view(self) -> bool:
        """Return if the plot has a view."""
        return self._viewer is not None

    def overwrite_shading(self, shading: dict) -> dict:
        """Overwrite the shading settings with the default settings."""
        default_shading = (
            DARK_DEFAULT_SHADING if self._darkmode else LIGHT_DEFAULT_SHADING
        )
        for key in default_shading.keys():
            if key not in shading:
                shading[key] = default_shading[key]
        return shading

    def curve(
        self, vert: np.ndarray, _edge: np.ndarray = np.zeros(0), shading: dict = {}
    ) -> "Plot":
        """Plot a curve.

        Args:
            vert (np.ndarray): The vertices (#x3) of the curve.
            _edge (np.ndarray): The edge elements (#x2) of the curve.
            shading (dict): The shading settings.

        Returns:
            Plot: The plot object.
        """
        if self._in_jupyter_notebook:
            shading = self.overwrite_shading(shading)
            if _edge.size == 0:
                edge = np.array([[i, (i + 1) % len(vert)] for i in range(len(vert))])
            else:
                edge = _edge
            if vert.shape[1] == 2:
                _pts = np.concatenate([vert, np.zeros((vert.shape[0], 1))], axis=1)
            else:
                _pts = vert
            viewer = Viewer(shading)
            viewer.reset()
            self._viewer = viewer
            self._shading = shading
            self.add.edge(_pts, edge)
            display(self._viewer._renderer)
        return self

    def tri(
        self,
        vert: np.ndarray,
        tri: np.ndarray,
        stitch: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0)),
        color: np.ndarray = np.array([1.0, 0.85, 0.0]),
        shading: dict = {},
    ) -> "Plot":
        """Plot a triangle mesh.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            tri (np.ndarray): The triangle elements (#x3) of the mesh.
            stitch (tuple[np.ndarray, np.ndarray]): The stitch data (index #x3 and weight #x2).
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].
            sahding (dict): The shading settings.

        Returns:
            Plot: The plot object.
        """
        if self._in_jupyter_notebook:
            if tri.shape[1] != 3:
                raise ValueError("triangles must have 3 vertices")
            shading = self.overwrite_shading(shading)
            self._viewer = plot(vert, tri, color, shading=shading)
            self._shading = shading
            assert isinstance(self._viewer, Viewer)

            ind, w = stitch
            if len(ind) and len(w):
                stitch_vert, stitch_edge = [], []
                for ind, w in zip(ind, w):
                    x0, y0, y1 = vert[ind[0]], vert[ind[1]], vert[ind[2]]
                    w0, w1 = w[0], w[1]
                    idx0, idx1 = len(stitch_vert), len(stitch_vert) + 1
                    stitch_vert.append(x0)
                    stitch_vert.append(w0 * y0 + w1 * y1)
                    stitch_edge.append([idx0, idx1])
                stitch_vert = np.array(stitch_vert)
                stitch_edge = np.array(stitch_edge)
                self._viewer.add_edges(stitch_vert, stitch_edge, shading=shading)
        return self

    def tet(
        self,
        vert: np.ndarray,
        tet: np.ndarray,
        axis: int = 0,
        cut: float = 0.5,
        color: np.ndarray = np.array([1.0, 0.85, 0.0]),
        shading: dict = {},
    ) -> "Plot":
        """Plot a tetrahedral mesh.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            tet (np.ndarray): The tetrahedral elements (#x4) of the mesh.
            axis (int): The axis to cut the mesh.
            cut (float): The cut ratio.
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].
            shading (dict): The shading settings.

        Returns:
            Plot: The plot object.
        """
        if self._in_jupyter_notebook:

            def compute_hash(tri, n):
                n = np.int64(n)
                i0, i1, i2 = sorted(tri)
                return i0 + i1 * n + i2 * n * n

            assert vert.shape[1] == 3
            assert tet.shape[1] == 4
            shading = self.overwrite_shading(shading)
            max_coord = np.max(vert[:, axis])
            min_coord = np.min(vert[:, axis])
            tmp_tri = {}
            for t in tet:
                x = [vert[i] for i in t]
                c = (x[0] + x[1] + x[2] + x[3]) / 4
                if c[axis] > min_coord + cut * (max_coord - min_coord):
                    tri = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
                    for k in tri:
                        e = [t[i] for i in k]
                        hash = compute_hash(e, len(vert))
                        if hash not in tmp_tri:
                            tmp_tri[hash] = e
                        else:
                            del tmp_tri[hash]
            return self.tri(
                vert, np.array(list(tmp_tri.values())), color=color, shading=shading
            )
        else:
            return self

    def update(self, vert: np.ndarray):
        """Update the plot with new vertices.

        Args:
            vert (np.ndarray): The new vertices (#x3).
        """
        if self._in_jupyter_notebook:
            viewer = self._viewer
            if viewer is None:
                raise Exception("No plot to update")
            else:
                objects = viewer._Viewer__objects  # type: ignore
                x = vert.copy().astype(np.float32)
                geo = objects[0]["geometry"]
                geo.attributes["position"].array = x
                geo.attributes["position"].needsUpdate = True
                if self._shading["flat"]:
                    geo.exec_three_obj_method("computeFaceNormals")
                else:
                    geo.exec_three_obj_method("computeVertexNormals")
        return self
