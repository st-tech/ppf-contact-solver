# File: _plot_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import copy

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pythreejs as p3s  # pyright: ignore

from IPython.display import display

from frontend._utils_ import Utils

from ._render_ import OpenGLRenderer


class PlotManager:
    """PlotManager class. Use this to create a plot."""

    def __init__(self) -> None:
        """Initialize the plot manager."""
        self.param = PlotParam()

    def create(self, engine: str = "threejs") -> "Plot":
        """Create a plot."""
        return Plot(engine, self.param)

    def is_jupyter_notebook(self) -> bool:
        """Check if the code is running in a Jupyter notebook."""
        return Utils.in_jupyter_notebook()


class Plot:
    """Plot class. Use this to create a plot."""

    def __init__(self, engine: str, param: "PlotParam") -> None:
        """Initialize the plot.

        Args:
            _darkmode (bool): True to turn on dark mode, False otherwise.
        """
        if engine == "threejs":
            self._engine = ThreejsPlotEngine()
        elif engine == "opengl":
            self._engine = OpenGLRenderEngine()
        else:
            raise ValueError(f"Unknown engine: {engine}")

        self._vert = np.zeros(0)
        self._color = np.zeros(0)
        self.param = param

    def is_jupyter_notebook(self) -> bool:
        """Check if the code is running in a Jupyter notebook."""
        return Utils.in_jupyter_notebook()

    def plot(
        self,
        vert: np.ndarray,
        color: np.ndarray = np.zeros(0),
        tri: np.ndarray = np.zeros(0),
        seg: np.ndarray = np.zeros(0),
        pts: np.ndarray = np.zeros(0),
        param_override: Optional[dict] = None,
    ) -> "Plot":
        """Plot a mesh.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].
            tri (np.ndarray): The triangle elements (#x3) of the mesh.
            seg (np.ndarray): The edge elements (#x2) of the mesh.
            pts (np.ndarray): The point elements (#x1) of the mesh.
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            param = copy.deepcopy(self.param)
            for key, value in param_override.items():
                setattr(param, key, value)
            self._vert = vert.copy()
            self._color = color.copy()
            self._engine.plot(self._vert, self._color, tri, seg, pts, param)
        return self

    def update(
        self,
        vert: np.ndarray | None = None,
        color: np.ndarray | None = None,
        recompute_normals: bool = True,
    ):
        if vert is not None:
            self._vert[0 : len(vert)] = vert
            vert = self._vert
        if color is not None:
            self._color[0 : len(color)] = color
            color = self._color
        self._engine.update(vert, color, recompute_normals)

    def tri(
        self,
        vert: np.ndarray,
        tri: np.ndarray,
        stitch: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0)),
        color: np.ndarray = np.zeros(0),
        param_override: Optional[dict] = None,
    ) -> "Plot":
        """Plot a triangle mesh.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            tri (np.ndarray): The triangle elements (#x3) of the mesh.
            stitch (tuple[np.ndarray, np.ndarray]): The stitch data (index #x3 and weight #x2).
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            if tri.shape[1] != 3:
                raise ValueError("triangles must have 3 vertices")
            if vert.shape[1] == 2:
                vert = np.concatenate(
                    [vert, np.zeros((vert.shape[0], 1), dtype=np.uint32)], axis=1
                )
            else:
                vert = vert.copy()
            ind, w = stitch
            if len(ind) and len(w):
                edge = []
                new_vert = []
                for ind_item, w_item in zip(ind, w, strict=False):
                    x0, y0, y1 = vert[ind_item[0]], vert[ind_item[1]], vert[ind_item[2]]
                    w0, w1 = w_item[0], w_item[1]
                    idx0 = len(new_vert) + len(vert)
                    idx1 = idx0 + 1
                    new_vert.append(x0)
                    new_vert.append(w0 * y0 + w1 * y1)
                    edge.append([idx0, idx1])
                vert = np.vstack([vert, np.array(new_vert)])
                edge = np.array(edge)
            else:
                edge = np.zeros(0)
            self.plot(vert, color, tri, edge, np.zeros(0), param_override)

        return self

    def edge(
        self,
        vert: np.ndarray,
        edge: np.ndarray,
        color: np.ndarray,
        param_override: Optional[dict] = None,
    ) -> "Plot":
        """Add edges to the plot.

        Args:
            vert (np.ndarray): The vertices (#x3) of the edges.
            edge (np.ndarray): The edge elements (#x2) of the edges.
            color (np.ndarray): The color (#x3) of the edges. Each value should be in [0,1].
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            self.plot(vert, color, np.zeros(0), edge, np.zeros(0), param_override)

        return self

    def point(self, vert: np.ndarray, param_override: Optional[dict] = None) -> "Plot":
        """Add points to the plot.

        Args:
            vert (np.ndarray): The vertices (#x3) of the points.
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            self.plot(
                vert,
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.arange(len(vert)),
                param_override,
            )

        return self

    def curve(
        self,
        vert: np.ndarray,
        _edge: np.ndarray = np.zeros(0),
        color: np.ndarray = np.zeros(0),
        param_override: Optional[dict] = None,
    ) -> "Plot":
        """Plot a curve.

        Args:
            vert (np.ndarray): The vertices (#x3) of the curve.
            _edge (np.ndarray): The edge elements (#x2) of the curve.
            color (np.ndarray): The color (#x3) of the curve. Each value should be in [0,1].
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            if _edge.size == 0:
                edge = np.array(
                    [[i, (i + 1) % len(vert)] for i in range(len(vert))],
                    dtype=np.uint32,
                )
            else:
                edge = _edge
            if vert.shape[1] == 2:
                _pts = np.concatenate(
                    [vert, np.zeros((vert.shape[0], 1), dtype=np.uint32)], axis=1
                )
            else:
                _pts = vert
            self.edge(_pts, edge, color, param_override)

        return self

    def tet(
        self,
        vert: np.ndarray,
        tet: np.ndarray,
        axis: int = 0,
        cut: float = 0.5,
        color: np.ndarray = np.zeros(0),
        param_override: Optional[dict] = None,
    ) -> "Plot":
        """Plot a tetrahedral mesh.

        Args:
            vert (np.ndarray): The vertices (#x3) of the mesh.
            tet (np.ndarray): The tetrahedral elements (#x4) of the mesh.
            axis (int): The axis to cut the mesh.
            cut (float): The cut ratio.
            color (np.ndarray): The color (#x3) of the mesh. Each value should be in [0,1].
            param_override (dict): The parameter override.

        Returns:
            Plot: The plot object.
        """
        if param_override is None:
            param_override = {}
        if "flat_shading" not in param_override:
            param_override["flat_shading"] = True
        if Utils.in_jupyter_notebook():
            param = copy.deepcopy(self.param)
            for key, value in param_override.items():
                setattr(param, key, value)

            def compute_hash(tri, n):
                n = np.int64(n)
                i0, i1, i2 = sorted(tri)
                return i0 + i1 * n + i2 * n * n

            assert vert.shape[1] == 3
            assert tet.shape[1] == 4
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
                vert,
                np.array(list(tmp_tri.values())),
                color=color,
                param_override=param_override,
            )
        else:
            return self


@dataclass
class PlotBuffer:
    vert: p3s.BufferAttribute | None = None
    tri: p3s.BufferAttribute | None = None
    color: p3s.BufferAttribute | None = None
    pts: p3s.BufferAttribute | None = None
    seg: p3s.BufferAttribute | None = None


@dataclass
class PlotGeometry:
    tri: p3s.BufferGeometry | None = None
    pts: p3s.BufferGeometry | None = None
    seg: p3s.BufferGeometry | None = None


@dataclass
class PlotObject:
    tri: p3s.Mesh | None = None
    pts: p3s.Points | None = None
    seg: p3s.LineSegments | None = None
    wireframe: p3s.Mesh | None = None
    light_0: p3s.DirectionalLight | None = None
    light_1: p3s.AmbientLight | None = None
    camera: p3s.PerspectiveCamera | None = None
    scene: p3s.Scene | None = None
    renderer: p3s.Renderer | None = None


@dataclass
class PlotParam:
    direct_intensity: float = 1.0
    ambient_intensity: float = 0.7
    wireframe: bool = True
    flat_shading: bool = False
    pts_scale: float = 0.004
    pts_color: str = "white"
    default_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.2]))
    lookat: list[float] | None = None
    eyeup: float = 0.0
    fov: float = 50.0
    width: int = 600
    height: int = 600


class ThreejsPlotEngine:
    def __init__(self):
        self.buff = PlotBuffer()
        self.geom = PlotGeometry()
        self.obj = PlotObject()
        self.flat_shading = False

    def plot(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        tri: np.ndarray,
        seg: np.ndarray,
        pts: np.ndarray,
        param: PlotParam = PlotParam(),
    ):
        assert len(vert) > 0
        if len(color) == 0:
            color = np.tile(param.default_color, (len(vert), 1))
        assert len(color) == len(vert)

        color = color.astype("float32")
        vert = vert.astype("float32")

        bbox = np.max(vert, axis=0) - np.min(vert, axis=0)
        if param.lookat is None:
            center = list(-np.min(vert, axis=0) - 0.5 * bbox)
        else:
            center = list(-np.array(param.lookat))

        self.buff.vert = p3s.BufferAttribute(vert, normalized=False)
        self.buff.color = p3s.BufferAttribute(color)
        if len(tri):
            self.buff.tri = p3s.BufferAttribute(
                tri.astype("uint32").ravel(), normalized=False
            )
        else:
            self.buff.tri = None
        if len(pts):
            self.buff.pts = p3s.BufferAttribute(
                pts.astype("uint32").ravel(), normalized=False
            )
        else:
            self.buff.pts = None
        if len(seg):
            self.buff.seg = p3s.BufferAttribute(
                seg.astype("uint32").ravel(), normalized=False
            )
        else:
            self.buff.seg = None

        if self.buff.tri is not None:
            self.geom.tri = p3s.BufferGeometry(
                attributes={
                    "position": self.buff.vert,
                    "index": self.buff.tri,
                    "color": self.buff.color,
                }
            )
        else:
            self.geom.tri = None
        if self.buff.pts is not None:
            self.geom.pts = p3s.BufferGeometry(
                attributes={
                    "position": self.buff.vert,
                    "index": self.buff.pts,
                }
            )
        else:
            self.geom.pts = None
        if self.buff.seg is not None:
            self.geom.seg = p3s.BufferGeometry(
                attributes={
                    "position": self.buff.vert,
                    "index": self.buff.seg,
                    "color": self.buff.color,
                }
            )
        else:
            self.geom.seg = None

        if self.geom.tri is not None:
            self.flat_shading = param.flat_shading
            if param.flat_shading:
                self.geom.tri.exec_three_obj_method("computeFaceNormals")
            else:
                self.geom.tri.exec_three_obj_method("computeVertexNormals")

        if self.geom.tri is not None:
            self.obj.tri = p3s.Mesh(
                geometry=self.geom.tri,
                material=p3s.MeshStandardMaterial(
                    vertexColors="VertexColors",
                    side="DoubleSide",
                    flatShading=param.flat_shading,
                    polygonOffset=True,
                    polygonOffsetFactor=1,
                    polygonOffsetUnits=1,
                ),
                position=center,
            )
        else:
            self.obj.tri = None
        if self.geom.pts is not None:
            self.obj.pts = p3s.Points(
                geometry=self.geom.pts,
                material=p3s.PointsMaterial(
                    size=param.pts_scale,
                    color=param.pts_color,
                ),
                position=center,
            )
        else:
            self.obj.pts = None
        if self.geom.seg is not None:
            self.obj.seg = p3s.LineSegments(
                geometry=self.geom.seg,
                material=p3s.LineBasicMaterial(vertexColors="VertexColors"),
                position=center,
            )
        else:
            self.obj.seg = None
        if param.wireframe and self.obj.tri is not None:
            self.obj.wireframe = p3s.Mesh(
                geometry=self.geom.tri,
                material=p3s.MeshBasicMaterial(
                    color="black",
                    wireframe=True,
                ),
                position=center,
            )
        else:
            self.obj.wireframe = None

        scale = np.max(bbox)
        position = [0, scale * param.eyeup, 1.25 * scale]

        self.obj.light_0 = p3s.DirectionalLight(
            position=position, intensity=param.direct_intensity
        )
        self.obj.light_1 = p3s.AmbientLight(intensity=param.ambient_intensity)
        self.obj.camera = p3s.PerspectiveCamera(
            position=position,
            fov=param.fov,
            aspect=param.width / param.height,
            children=[self.obj.light_0],
        )

        children = [self.obj.camera, self.obj.light_1]
        if self.obj.tri is not None:
            children.append(self.obj.tri)
        if self.obj.wireframe is not None:
            children.append(self.obj.wireframe)
        if self.obj.pts is not None:
            children.append(self.obj.pts)
        if self.obj.seg is not None:
            children.append(self.obj.seg)

        self.obj.scene = p3s.Scene(children=children, background="#222222")
        self.obj.renderer = p3s.Renderer(
            camera=self.obj.camera,
            scene=self.obj.scene,
            controls=[p3s.OrbitControls(controlling=self.obj.camera)],
            antialias=True,
            width=param.width,
            height=param.height,
        )

        display(self.obj.renderer)

    def update(
        self,
        vert: np.ndarray | None = None,
        color: np.ndarray | None = None,
        recompute_normals: bool = True,
    ):
        if vert is not None:
            assert self.buff.vert is not None
            self.buff.vert.array = vert.astype("float32")
            self.buff.vert.needsUpdate = True
        if color is not None:
            assert self.buff.color is not None
            self.buff.color.array = color.astype("float32")
            self.buff.color.needsUpdate = True
        # Allow recomputing normals even without new vertices (for debounced updates)
        if recompute_normals and self.geom.tri is not None:
            if self.flat_shading:
                self.geom.tri.exec_three_obj_method("computeFaceNormals")
            else:
                self.geom.tri.exec_three_obj_method("computeVertexNormals")


class OpenGLRenderEngine:
    def __init__(self) -> None:
        self._handle = None

    def _render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        tri: np.ndarray,
        seg: np.ndarray,
    ):
        from IPython.display import (
            display,
        )

        engine = OpenGLRenderer()
        image = engine.render(
            vert,
            color,
            seg,
            tri,
            None,
        )
        if self._handle is None:
            self._handle = display(image, display_id=True)
        else:
            self._handle.update(image)

    def plot(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        tri: np.ndarray,
        seg: np.ndarray,
        pts: np.ndarray,
        param: PlotParam = PlotParam(),
    ):
        assert len(vert) > 0
        if len(color) == 0:
            color = np.tile(param.default_color, (len(vert), 1))
        assert len(color) == len(vert)

        self._vert = vert.copy()
        self._color = color.copy()
        self._tri = tri.copy()
        self._seg = seg.copy()
        self._pts = pts.copy()
        self._param = param
        self._render(self._vert, self._color, self._tri, self._seg)

    def update(
        self,
        vert: np.ndarray | None = None,
        color: np.ndarray | None = None,
        recompute_normals: bool = True,  # unused, for API compatibility
    ):
        if vert is not None:
            self._vert = vert.copy()
        if color is not None:
            self._color = color.copy()
        self._render(self._vert, self._color, self._tri, self._seg)
