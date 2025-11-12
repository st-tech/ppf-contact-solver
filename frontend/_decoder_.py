# File: _decoder_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import pickle

from ._asset_ import AssetManager
from ._mesh_ import MeshManager
from ._plot_ import PlotManager
from ._scene_ import FixedScene, Scene
from ._session_ import FixedSession, Session
from ._utils_ import Utils


class BlenderApp:
    def __init__(self, name: str, verbose: bool = False):
        """Initialize the BlenderDecoder.

        Args:
            name (str): The name of the Blender project.
            verbose (bool): Enable verbose logging.
        """
        self._verbose = verbose
        self._name = name

        # Get current git branch name for directory structure
        import subprocess

        # First try to read from .git/branch_name.txt
        git_branch = "unknown"
        try:
            branch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                ".git",
                "branch_name.txt"
            )
            if os.path.exists(branch_file):
                with open(branch_file, "r") as f:
                    git_branch = f.read().strip()
                    if not git_branch:
                        git_branch = "unknown"
        except:
            pass

        # Fallback to git command if branch_name.txt not found or empty
        if git_branch == "unknown":
            try:
                git_branch = subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    text=True,
                ).strip()
                if not git_branch:
                    git_branch = "unknown"
            except Exception as _:
                git_branch = "unknown"

        self._data_dirpath = os.path.join(
            os.path.expanduser("~"),
            ".local",
            "share",
            "ppf-cts",
            f"git-{git_branch}",
        )
        self._root = os.path.join(
            self._data_dirpath,
            self._name,
        )
        cache_root = os.path.join(self._root, ".cash")
        os.makedirs(cache_root, exist_ok=True)
        self._asset_manager = AssetManager()
        self._mesh_manager = MeshManager(cache_root)
        self._scene = None
        self._session = None
        self._fixed_scene = None
        self._fixed_session = None

    def populate(self) -> "BlenderApp":
        """Populate the scene with objects from a Blender project."""
        data_path = os.path.join(self._root, "data.pickle")
        assert os.path.exists(data_path)

        self._scene = Scene("scene", PlotManager(), self._asset_manager)
        scene_decoder = SceneDecoder(data_path, self._asset_manager, self._mesh_manager)
        scene_decoder.populate_objects(self._scene, verbose=self._verbose)
        return self

    def make(self) -> "BlenderApp":
        """Make a runnable session from the populated scene with parameters updated."""
        assert self._scene is not None, "Scene must be populated before making the app"
        param_path = os.path.join(self._root, "param.pickle")
        assert os.path.exists(param_path)

        param_decoder = ParamDecoder().set_path(param_path)
        param_decoder.apply_to_objects(self._scene, verbose=self._verbose)
        self._fixed_scene = self._scene.build()
        if Utils.in_jupyter_notebook():
            self._fixed_scene.preview()

        self._session = Session(
            self._name,
            self._root,
            os.path.dirname(os.path.dirname(__file__)),
            self._data_dirpath,
            "session",
        ).init(self._fixed_scene)
        param_decoder.apply_to_session(self._session, verbose=self._verbose)
        self._fixed_session = self._session.build()
        return self

    @property
    def scene(self) -> Scene:
        """Get the scene object."""
        assert self._scene is not None, "Scene must be populated before accessing it"
        return self._scene

    @property
    def session(self) -> Session:
        """Get the session object."""
        assert self._session is not None, "Session must be made before accessing it"
        return self._session

    @property
    def fixed_scene(self) -> FixedScene:
        """Get the fixed scene object."""
        assert self._fixed_scene is not None, (
            "Fixed scene must be made before accessing it"
        )
        return self._fixed_scene

    @property
    def fixed_session(self) -> FixedSession:
        """Get the fixed session object."""
        assert self._fixed_session is not None, (
            "Fixed session must be made before accessing it"
        )
        return self._fixed_session


class ParamDecoder:
    def __init__(self):
        self._data = None

    def set_path(self, filepath: str) -> "ParamDecoder":
        """Set the path to the parameter file."""
        assert filepath.endswith(".pickle"), "File must be a pickle file."
        with open(filepath, "rb") as f:
            self._data = pickle.load(f)
        assert "group" in self._data, "Group parameters not found in the data."
        assert "scene" in self._data, "Scene parameters not found in the data."
        return self

    def apply_to_objects(self, scene: Scene, verbose: bool = False):
        """Decode the parameter file and apply it to the objects in a scene.

        Args:
            scene (Scene): The scene to which the parameters will be applied.
            verbose (bool): Enable verbose logging
        """
        assert self._data is not None, "Parameter data not set. Call set_path() first."
        if verbose:
            print("=== Object Parameters ===")
        for params, objects in self._data["group"]:
            for obj_name in objects:
                if verbose:
                    print(f"*** name: {obj_name} ***")
                obj = scene.select(obj_name)
                obj.param.clear_all()
                for key, val in params.items():
                    if verbose:
                        print(f"  {key}: {val}")
                    obj.param.set(key, val)

    def apply_to_session(self, session: Session, verbose: bool = False):
        """Decode the parameter file and apply it to the session.

        Args:
            filepath (str): Path to the pickle file containing the session parameters.
            verbose (bool): Enable verbose logging
        """
        assert self._data is not None, "Parameter data not set. Call set_path() first."
        if verbose:
            print("=== Session Parameters ===")
        session.param.clear_all()

        for k, v in self._data["scene"].items():
            if verbose:
                print(f"  {k}: {v}")
            if k == "fitting" and v > 0:
                fitting_time = float(v)
                session.param.set("fitting")
                session.param.dyn("fitting").time(fitting_time).hold().change(False)
            else:
                session.param.set(k, v)


class SceneDecoder:
    def __init__(
        self, filepath: str, asset_manager: AssetManager, mesh_manager: MeshManager
    ):
        assert filepath.endswith(".pickle"), "File must be a pickle file."
        with open(filepath, "rb") as f:
            self._data = pickle.load(f)
        self._asset = asset_manager
        self._mesh = mesh_manager

    def populate_objects(self, scene: Scene, verbose: bool = False) -> Scene:
        """
        Populate the scene with objects from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing the objects data.

        Returns:
            scene: The populated scene.
        """
        for group in self._data:
            objects = group.get("object", None)
            assert objects is not None, "Object data not found in the group."
            group_type = group.get("type")
            if verbose:
                print(f"--- new group: {group_type} ---")
            for obj in objects:
                name, vert, face, edge, uv = (
                    obj.get("name"),
                    obj.get("vert"),
                    obj.get("face"),
                    obj.get("edge"),
                    obj.get("uv", None),
                )
                if verbose:
                    if edge is not None:
                        print(
                            f"  * name: {name}, vert: {vert.shape}, edge: {edge.shape}, uv: {len(uv) if uv is not None else 'None'}"
                        )
                    else:
                        print(
                            f"  * name: {name}, vert: {vert.shape}, face: {face.shape if face is not None else 'None'}, uv: {len(uv) if uv is not None else 'None'}"
                        )
                if group_type == "STATIC":
                    if verbose:
                        print(
                            f"      > animation: {'YES' if 'animation' in obj else 'NO'}"
                        )
                    self._asset.add.tri(name, vert, face)
                    pin = scene.add(name).pin()
                    animation = obj.get("animation", None)
                    if animation:
                        import numpy as np

                        time = animation.get("time", [])
                        position = animation.get("position", [])
                        prev_t = 0.0
                        for t, pos in zip(time, position, strict=False):
                            if t > prev_t:
                                pin.move_to(np.array(pos), t_start=prev_t, t_end=t)
                                prev_t = t
                    _obj = None
                elif group_type == "SOLID":
                    V, F, T = self._mesh.create.tri(vert, face).tetrahedralize()
                    self._asset.add.tet(name, V, F, T)
                    _obj = scene.add(name)
                elif group_type == "SHELL":
                    self._asset.add.tri(name, vert, face)
                    _obj = scene.add(name)
                    if uv is not None:
                        assert len(uv) == len(face), "UV length must match face length."
                        _obj.set_uv(uv)
                elif group_type == "ROD":
                    assert edge is not None, (
                        f"Edge data not found for rod object {name}"
                    )
                    self._asset.add.rod(name, vert, edge)
                    _obj = scene.add(name)
                else:
                    raise ValueError(f"Unknown group type: {group_type}")
                if _obj is not None and "pin" in obj:
                    pin_index, pin_anim = obj["pin"]
                    if verbose:
                        print(
                            f"      > pin: {len(pin_index)}, animation: {'YES' if pin_anim else 'NO'}"
                        )
                    if pin_anim:
                        import numpy as np

                        pins = [_obj.pin([i]) for i in pin_index]
                        for i, _pin in enumerate(pins):
                            keyframe = pin_anim.get(pin_index[i], None)
                            if keyframe:
                                prev_t = 0.0
                                for t, pos in zip(
                                    keyframe["time"], keyframe["position"], strict=False
                                ):
                                    if t > prev_t:
                                        pins[i].move_to(
                                            np.array([pos]), t_start=prev_t, t_end=t
                                        )
                                        prev_t = t
                    else:
                        _obj.pin(pin_index)

                if _obj is not None and "stitch" in obj:
                    stitch_data = obj["stitch"]
                    stitch_name = f"{name}_stitch"
                    if verbose:
                        print(f"      > stitch: {len(stitch_data[0])} edges")
                    self._asset.add.stitch(stitch_name, stitch_data)
                    _obj.stitch(stitch_name)

        return scene
