# File: frontend/tests/_object_visibility_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for `Object.invisible`, the per-object draw-time lever.

An invisible object keeps its full role in the scene: it is built,
validated, and handed to the solver like any other. It only leaves the
pictures, which are the JupyterLab preview, the live session view that
replays solver frames through that same plot, and the images
`session.export` renders into `frame.mp4`. Exported `.ply` / `.obj` meshes
still carry the whole scene.

The preview path is the delicate one. A live session pushes frames indexed
over every dynamic vertex, hidden ones included, so the plot's cached buffer
has to stay that wide; only the upload to the render engine is compacted.
Both halves are covered here: what reaches the engine, and that a full-width
frame update still lands on it.
"""

import shutil
import tempfile

from pathlib import Path

import numpy as np

from .. import _plot_ as plot_mod
from .._asset_ import AssetManager
from .._mesh_ import MeshManager
from .._plot_ import PlotManager
from .._scene_ import Scene
from .._utils_ import Utils

# Vertex / triangle counts of the two assets every scene below is built
# from, so the assertions can name exact sizes instead of deltas.
SHEET_VERT, SHEET_TRI = 9, 8
BOX_VERT, BOX_TRI = 8, 12


class RecordingEngine:
    """Stands in for a plot engine and keeps whatever it was handed."""

    last: "RecordingEngine"

    def __init__(self) -> None:
        RecordingEngine.last = self
        self.plotted: dict = {}
        self.updated: dict = {}

    def plot(self, vert, color, tri, seg, pts, param=None):
        self.plotted = dict(vert=vert, color=color, tri=tri, seg=seg, pts=pts)

    def update(self, vert=None, color=None, recompute_normals=True):
        self.updated = dict(vert=vert, color=color)


class Workspace:
    """A scratch cache dir plus a plot layer that records instead of drawing.

    `Plot` only draws inside a notebook, so the notebook check is answered
    True for the duration and the three.js engine is swapped for a recorder.
    Both are restored on exit.
    """

    def __enter__(self) -> "Workspace":
        self.root = Path(tempfile.mkdtemp(prefix="ppf-visibility-"))
        self.asset = AssetManager()
        self.mesh = MeshManager(str(self.root / "cache"))
        self.plot = PlotManager()
        vert, tri = self.mesh.square(res=3)
        self.asset.add.tri("sheet", vert, tri)
        vert, tri = self.mesh.box(0.5, 0.5, 0.5)
        self.asset.add.tri("box", vert, tri)
        self._in_notebook = Utils.in_jupyter_notebook
        self._engine = plot_mod.ThreejsPlotEngine
        Utils.in_jupyter_notebook = staticmethod(lambda: True)
        plot_mod.ThreejsPlotEngine = RecordingEngine
        return self

    def __exit__(self, *_exc) -> None:
        Utils.in_jupyter_notebook = self._in_notebook
        plot_mod.ThreejsPlotEngine = self._engine
        shutil.rmtree(self.root, ignore_errors=True)

    def scene(self, name: str):
        """Two well-separated sheets plus a pinned box collider."""
        scene = Scene(name, self.plot, self.asset)
        lower = scene.add("sheet").at(0, 5, 0)
        upper = scene.add("sheet").at(0, 10, 0)
        collider = scene.add("box").at(0, 0, 0)
        collider.pin()
        return scene, lower, upper, collider


def _drawn(fixed) -> dict:
    """Preview a scene and return what the render engine received."""
    fixed.preview(show_slider=False)
    return RecordingEngine.last.plotted


def test_lever_is_chainable_and_reversible():
    """The lever reads back through `visible` and can be lifted again."""
    with Workspace() as ws:
        _, lower, _, _ = ws.scene("chain")
        assert lower.visible

        assert lower.invisible() is lower
        assert not lower.visible

        assert lower.invisible(False) is lower
        assert lower.visible
    print("    Chainable, readable through .visible, reversible: PASS")


def test_nothing_hidden_leaves_the_scene_untouched():
    """A scene with no hidden object carries no mask and compacts nothing."""
    with Workspace() as ws:
        scene, _, _, _ = ws.scene("plain")
        fixed = scene.build(quiet=True)

        assert fixed._invisible_vert is None
        assert fixed._invisible_static_vert is None
        assert fixed._draw_index(26, 2 * SHEET_VERT, BOX_VERT) is None

        drawn = _drawn(fixed)
        assert len(drawn["vert"]) == 2 * SHEET_VERT + BOX_VERT
        assert len(drawn["tri"]) == 2 * SHEET_TRI + BOX_TRI
    print("    Scene with nothing hidden draws its whole buffer: PASS")


def test_hidden_dynamic_object_leaves_the_picture():
    """Hiding a dynamic object drops its vertices and triangles from the draw."""
    with Workspace() as ws:
        scene, _, upper, _ = ws.scene("dynamic")
        upper.invisible()
        fixed = scene.build(quiet=True)

        assert int(fixed._invisible_vert.sum()) == SHEET_VERT
        assert fixed._invisible_static_vert is None

        drawn = _drawn(fixed)
        assert len(drawn["vert"]) == SHEET_VERT + BOX_VERT
        assert len(drawn["tri"]) == SHEET_TRI + BOX_TRI
        assert len(drawn["color"]) == len(drawn["vert"])
        # Every surviving triangle addresses the compacted buffer.
        assert drawn["tri"].max() < len(drawn["vert"])
    print("    Hidden dynamic object leaves the drawn buffer: PASS")


def test_hidden_static_collider_leaves_the_picture():
    """The static namespace is filtered on its own mask, not the dynamic one."""
    with Workspace() as ws:
        scene, _, _, collider = ws.scene("static")
        collider.invisible()
        fixed = scene.build(quiet=True)

        assert fixed._invisible_vert is None
        assert int(fixed._invisible_static_vert.sum()) == BOX_VERT

        drawn = _drawn(fixed)
        assert len(drawn["vert"]) == 2 * SHEET_VERT
        assert len(drawn["tri"]) == 2 * SHEET_TRI
        assert drawn["tri"].max() < len(drawn["vert"])
    print("    Hidden static collider leaves the drawn buffer: PASS")


def test_live_frame_update_lands_on_the_compacted_buffer():
    """A solver frame spans every dynamic vertex, hidden ones included.

    `session.preview` pushes exactly that array through `Plot.update`, so the
    cached buffer must stay full width and the compaction must happen on the
    way to the engine. A regression here surfaces as a live view that either
    throws on the first frame or draws the wrong vertices.
    """
    with Workspace() as ws:
        scene, _, upper, _ = ws.scene("live")
        upper.invisible()
        fixed = scene.build(quiet=True)

        plot = fixed.preview(show_slider=False)
        assert plot is not None
        engine = RecordingEngine.last
        drawn_vert_count = len(engine.plotted["vert"])

        frame = fixed.vertex(True) + np.array([0.0, 0.25, 0.0])
        assert len(frame) == 2 * SHEET_VERT, "frame must span every dynamic vertex"
        plot.update(frame, fixed.color(frame))

        assert len(engine.updated["vert"]) == drawn_vert_count
        assert len(engine.updated["color"]) == drawn_vert_count
        # The visible sheet moved with the frame; the hidden one is gone.
        assert np.isclose(engine.updated["vert"][:SHEET_VERT, 1].max(), 6.25)
    print("    Full-width frame update reaches the compacted buffer: PASS")


def test_hidden_object_drops_its_pin_markers():
    """Pin dots follow their object out of the picture.

    A partial pin, so both sheets stay dynamic; pinning every vertex would
    turn them into static colliders instead.
    """
    with Workspace() as ws:
        scene, lower, upper, _ = ws.scene("pins")
        pinned = [0, 1, 2]
        lower.pin(pinned)
        upper.pin(pinned)

        assert len(_drawn(scene.build(quiet=True))["pts"]) == 2 * len(pinned)

        upper.invisible()
        drawn = _drawn(scene.build(quiet=True))
        assert len(drawn["pts"]) == len(pinned)
        assert drawn["pts"].max() < len(drawn["vert"])
    print("    Pin markers of a hidden object are dropped: PASS")


def test_hiding_everything_fails_loudly():
    """An empty picture is a mistake worth naming, not a blank widget."""
    with Workspace() as ws:
        scene, lower, upper, collider = ws.scene("empty")
        lower.invisible()
        upper.invisible()
        collider.invisible()
        fixed = scene.build(quiet=True)

        try:
            fixed.preview(show_slider=False)
        except ValueError as e:
            assert "every object in this scene is invisible" in str(e)
        else:
            raise AssertionError("hiding every object must raise")
    print("    Hiding every object raises instead of drawing nothing: PASS")


def test_export_keeps_the_mesh_whole_but_not_the_picture():
    """`.ply` is data and keeps everything; the rendered `.png` is a picture."""
    with Workspace() as ws:
        scene, _, upper, _ = ws.scene("export")
        upper.invisible()
        fixed = scene.build(quiet=True)

        vert = fixed.vertex(True)
        path = ws.root / "export" / "frame_0.ply"
        fixed.export(vert, fixed.color(vert), str(path), delete_exist=True)

        # The mesh file is skipped under CI, where only the render matters.
        if Utils.ci_name() is None:
            import trimesh

            mesh = trimesh.load_mesh(str(path), process=False)
            assert len(mesh.vertices) == 2 * SHEET_VERT + BOX_VERT
            assert len(mesh.faces) == 2 * SHEET_TRI + BOX_TRI

        image = Path(str(path) + ".png")
        assert image.exists() and image.stat().st_size > 0

        # What the renderer was handed, assembled the way `export` does it.
        static_vert = fixed._static_vert[1]
        picture = fixed._picture_arrays(
            np.concatenate([vert, static_vert], axis=0),
            np.concatenate([fixed.color(vert), fixed._static_color], axis=0),
            fixed._rod,
            np.concatenate([fixed._tri, fixed._static_tri + len(vert)]),
            len(vert),
            len(static_vert),
        )
        assert len(picture[0]) == SHEET_VERT + BOX_VERT
        assert len(picture[3]) == SHEET_TRI + BOX_TRI
        assert picture[3].max() < len(picture[0])
    print("    Export writes the whole mesh and a partial picture: PASS")


def test_hidden_object_renders_as_if_it_were_absent():
    """Hiding is not dimming: the geometry leaves the auto-framing too.

    The renderer fits the camera to the vertices it is given, so a hidden
    object left in that array would silently zoom the picture out. Rendering
    the same visible geometry with and without a large hidden neighbor must
    produce the same bytes.
    """
    with Workspace() as ws:
        vert, tri = ws.mesh.box(20, 20, 20)
        ws.asset.add.tri("huge", vert, tri)

        def render(name: str, with_hidden: bool) -> bytes:
            scene = Scene(name, ws.plot, ws.asset)
            scene.add("sheet").at(0, 0, 0).color(1.0, 0.0, 0.0)
            if with_hidden:
                scene.add("huge").at(0, 40, 0).color(0.0, 1.0, 0.0).invisible()
            fixed = scene.build(quiet=True)
            positions = fixed.vertex(True)
            path = ws.root / name / "frame.ply"
            fixed.export(
                positions, fixed.color(positions), str(path), delete_exist=True
            )
            return Path(str(path) + ".png").read_bytes()

        assert render("with-hidden", True) == render("without", False), (
            "hidden geometry still influenced the picture"
        )
    print("    Hidden object renders identically to an absent one: PASS")


def test_visibility_never_reaches_the_solver():
    """Hiding an object changes the picture and nothing else about the scene."""
    with Workspace() as ws:
        scene, _, upper, _ = ws.scene("solver")
        shown = scene.build(quiet=True)

        upper.invisible()
        hidden = scene.build(quiet=True)

        assert np.array_equal(shown.vertex(True), hidden.vertex(True))
        assert np.array_equal(shown._tri, hidden._tri)
        assert np.array_equal(shown._rod, hidden._rod)
        assert np.array_equal(shown._vel, hidden._vel)
        assert np.array_equal(shown._color, hidden._color)
        assert np.array_equal(shown._static_tri, hidden._static_tri)
        assert np.array_equal(shown._static_vert[1], hidden._static_vert[1])
        assert shown.tri_param.keys() == hidden.tri_param.keys()
    print("    Hiding an object leaves the solver-bound scene identical: PASS")


def run_tests() -> bool:
    """Run all object-visibility tests. Returns True if all tests pass."""
    print("=" * 50)
    print("Object Visibility Tests")
    print("=" * 50)

    try:
        test_lever_is_chainable_and_reversible()
        test_nothing_hidden_leaves_the_scene_untouched()
        test_hidden_dynamic_object_leaves_the_picture()
        test_hidden_static_collider_leaves_the_picture()
        test_live_frame_update_lands_on_the_compacted_buffer()
        test_hidden_object_drops_its_pin_markers()
        test_hiding_everything_fails_loudly()
        test_export_keeps_the_mesh_whole_but_not_the_picture()
        test_hidden_object_renders_as_if_it_were_absent()
        test_visibility_never_reaches_the_solver()
        print("\nAll object visibility tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
