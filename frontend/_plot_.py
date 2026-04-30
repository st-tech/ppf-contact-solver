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

from ._render_ import Rasterizer


class PlotManager:
    """Factory for creating :class:`Plot` instances with shared parameters.

    Example:
        Reach the plot manager from the app and render a sheet::

            app = App.create("demo")
            V, F = app.mesh.square(res=64)
            app.plot.create().tri(V, F)
    """

    def __init__(self) -> None:
        """Initialize the plot manager with default plot parameters."""
        self.param = PlotParam()

    def create(self, engine: str = "threejs") -> "Plot":
        """Create a new plot using the given rendering engine.

        Args:
            engine (str): The rendering engine to use. Either ``"threejs"`` or ``"software"``.

        Returns:
            Plot: A new plot bound to this manager's parameters.

        Example:
            Create a plot and chain a triangle viewer call::

                V, F = app.mesh.icosphere(r=1.0, subdiv_count=3)
                app.plot.create().tri(V, F)
        """
        return Plot(engine, self.param)

    def is_jupyter_notebook(self) -> bool:
        """Return True if the code is running inside a Jupyter notebook.

        Example:
            Only build a plot when running inside a notebook::

                if app.plot.is_jupyter_notebook():
                    app.plot.create().tri(V, F)
        """
        return Utils.in_jupyter_notebook()


class Plot:
    """A plot backed by either the pythreejs or software rasterizer engine.

    Example:
        Construct a plot via :meth:`PlotManager.create` and draw a mesh::

            plot = app.plot.create()
            V, F = app.mesh.square(res=32)
            plot.tri(V, F)
    """

    def __init__(self, engine: str, param: "PlotParam") -> None:
        """Initialize the plot with the selected rendering engine.

        Args:
            engine (str): The rendering engine. Either ``"threejs"`` or ``"software"``.
            param (PlotParam): The plot parameters shared with this plot.

        Raises:
            ValueError: If ``engine`` is not a recognized engine name.
        """
        if engine == "threejs":
            self._engine = ThreejsPlotEngine()
        elif engine == "software":
            self._engine = RasterizerEngine()
        else:
            raise ValueError(f"Unknown engine: {engine}")

        self._vert = np.zeros(0)
        self._color = np.zeros(0)
        self.param = param

    def is_jupyter_notebook(self) -> bool:
        """Return True if the code is running inside a Jupyter notebook.

        Example:
            Skip expensive mesh conversion when not in a notebook::

                plot = app.plot.create()
                if plot.is_jupyter_notebook():
                    plot.tri(V, F)
        """
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
        """Plot a mesh with optional triangles, edges, and points.

        Rendering is only performed when running inside a Jupyter notebook.

        Args:
            vert (np.ndarray): The vertices (Nx3) of the mesh.
            color (np.ndarray): Per-vertex colors (Nx3) with each value in [0, 1].
            tri (np.ndarray): The triangle elements (Fx3) of the mesh.
            seg (np.ndarray): The edge elements (Ex2) of the mesh.
            pts (np.ndarray): The point element indices (Px1) of the mesh.
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Example:
            Render triangles and edges together with a camera override::

                plot = app.plot.create()
                plot.plot(V, tri=F, seg=edges, param_override={"fov": 30})
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
        vert: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        recompute_normals: bool = True,
    ):
        """Update the cached vertex or color buffers and forward to the engine.

        Args:
            vert (Optional[np.ndarray]): New vertex positions. Only the leading
                ``len(vert)`` rows of the cached buffer are overwritten.
            color (Optional[np.ndarray]): New per-vertex colors. Only the leading
                ``len(color)`` rows of the cached buffer are overwritten.
            recompute_normals (bool): Whether to recompute normals after the update.

        Example:
            Animate a cached plot by pushing new vertex positions each frame::

                plot = app.plot.create().tri(V, F)
                for t in range(10):
                    V_t = V + np.array([0, 0.01 * t, 0])
                    plot.update(vert=V_t)
        """
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
        """Plot a triangle mesh, optionally with visualized stitch connections.

        If ``stitch`` is provided, additional vertices and edges are generated
        to visualize each stitch as a line segment.

        Args:
            vert (np.ndarray): The vertices (Nx3 or Nx2) of the mesh. 2D inputs
                are padded to 3D by appending a zero column.
            tri (np.ndarray): The triangle elements (Fx3) of the mesh.
            stitch (tuple[np.ndarray, np.ndarray]): Stitch data as a pair of
                arrays: an index array (Sx3) and a weight array (Sx2).
            color (np.ndarray): Per-vertex colors (Nx3) with each value in [0, 1].
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Raises:
            ValueError: If ``tri`` does not have exactly 3 columns.

        Example:
            Plot a tetrahedralized armadillo's surface triangles::

                V, F, T = app.mesh.preset("armadillo").decimate(19000).tetrahedralize()
                app.plot.create().tri(V, F)
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
        """Plot a set of edges.

        Args:
            vert (np.ndarray): The vertices (Nx3) used by the edges.
            edge (np.ndarray): The edge elements (Ex2) as pairs of vertex indices.
            color (np.ndarray): Per-vertex colors (Nx3) with each value in [0, 1].
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Example:
            Draw just the edges of a mesh colored uniformly::

                V, F = app.mesh.icosphere(r=1.0, subdiv_count=2)
                edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
                colors = np.tile([1.0, 0.3, 0.3], (len(V), 1))
                app.plot.create().edge(V, edges, colors)
        """
        if param_override is None:
            param_override = {}
        if Utils.in_jupyter_notebook():
            self.plot(vert, color, np.zeros(0), edge, np.zeros(0), param_override)

        return self

    def point(self, vert: np.ndarray, param_override: Optional[dict] = None) -> "Plot":
        """Plot a set of points.

        Args:
            vert (np.ndarray): The vertex positions (Nx3) of the points.
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Example:
            Scatter a random point cloud::

                pts = np.random.rand(200, 3)
                app.plot.create().point(pts)
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
        """Plot a curve as a sequence of connected line segments.

        If ``_edge`` is empty, a closed loop is built by connecting each vertex
        to the next one (with wrap-around).

        Args:
            vert (np.ndarray): The vertex positions (Nx3 or Nx2) of the curve.
                2D inputs are padded to 3D by appending a zero column.
            _edge (np.ndarray): Optional explicit edge elements (Ex2) of the curve.
            color (np.ndarray): Per-vertex colors (Nx3) with each value in [0, 1].
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Example:
            Plot a parametric 2D curve as a closed loop::

                N = 500
                t = np.linspace(0, 2 * np.pi, N, endpoint=False)
                r = 0.85 * np.cos(10 * t) + 1
                P = np.column_stack([r * np.cos(t), r * np.sin(t)])
                app.plot.create().curve(P)
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
        """Plot a tetrahedral mesh by rendering the cut surface as triangles.

        Tetrahedra whose centroid lies past ``cut`` along ``axis`` contribute
        their faces; internal faces shared by two such tetrahedra are removed,
        leaving only the visible surface of the cut region. ``flat_shading`` is
        forced on unless ``param_override`` explicitly sets it.

        Args:
            vert (np.ndarray): The vertices (Nx3) of the mesh.
            tet (np.ndarray): The tetrahedral elements (Tx4) of the mesh.
            axis (int): The axis index (0, 1, or 2) along which to cut.
            cut (float): The cut ratio in [0, 1] along ``axis``.
            color (np.ndarray): Per-vertex colors (Nx3) with each value in [0, 1].
            param_override (Optional[dict]): Fields to override on a copy of the plot parameters.

        Returns:
            Plot: ``self`` for method chaining.

        Example:
            Slice an armadillo tet mesh along the x axis and preview it::

                V, F, T = app.mesh.preset("armadillo").decimate(19000).tetrahedralize()
                app.plot.create().tet(V, T, axis=0, cut=0.5)
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
    vert: Optional[p3s.BufferAttribute] = None
    tri: Optional[p3s.BufferAttribute] = None
    color: Optional[p3s.BufferAttribute] = None
    pts: Optional[p3s.BufferAttribute] = None
    seg: Optional[p3s.BufferAttribute] = None


@dataclass
class PlotGeometry:
    tri: Optional[p3s.BufferGeometry] = None
    pts: Optional[p3s.BufferGeometry] = None
    seg: Optional[p3s.BufferGeometry] = None


@dataclass
class PlotObject:
    tri: Optional[p3s.Mesh] = None
    pts: Optional[p3s.Points] = None
    seg: Optional[p3s.LineSegments] = None
    wireframe: Optional[p3s.Mesh] = None
    light_0: Optional[p3s.DirectionalLight] = None
    light_1: Optional[p3s.AmbientLight] = None
    camera: Optional[p3s.PerspectiveCamera] = None
    scene: Optional[p3s.Scene] = None
    renderer: Optional[p3s.Renderer] = None


@dataclass
class PlotParam:
    direct_intensity: float = 1.0
    ambient_intensity: float = 0.7
    wireframe: bool = True
    flat_shading: bool = False
    pts_scale: float = 0.004
    pts_color: str = "white"
    default_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.2]))
    lookat: Optional[list[float]] = None
    eye: Optional[list[float]] = None
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
        if param.eye is not None:
            position = list(param.eye)
        else:
            position = [0, 0, 1.25 * scale]

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
        vert: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
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


class RasterizerEngine:
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

        engine = Rasterizer()
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
        vert: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        recompute_normals: bool = True,  # unused, for API compatibility
    ):
        if vert is not None:
            self._vert = vert.copy()
        if color is not None:
            self._color = color.copy()
        self._render(self._vert, self._color, self._tri, self._seg)
