# File: _session_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ._scene_ import FixedScene
from ._plot_ import Plot
from ._utils_ import Utils
from ._parse_ import ParamParser, CppRustDocStringParser
from tqdm import tqdm
import pandas as pd
import signal
import subprocess
import numpy as np
import shutil
import psutil
import os
import threading
import time
import copy
from typing import Any, Optional


PROCESS_NAME = "ppf-contact"
CONSOLE_STYLE = """
    <style>
        .no-scroll {
            overflow: hidden;
            white-space: pre-wrap;
            font-family: monospace;
        }
    </style>
    """


class Param:
    """Class to manage simulation parameters."""

    def __init__(self, app_root: str):
        """Initialize the Param class.

        Args:
            app_root (str): The root directory of the application.
        """
        path = os.path.abspath(os.path.join(app_root, "src", "args.rs"))
        self._key = None
        self._default_param = ParamParser.get_default_params(path)
        self._param = self._default_param.copy()
        self._time = 0.0
        self._dyn_param = {}

    def copy(self) -> "Param":
        """Copy the Param object.

        Returns:
            Param: The copied Param object.
        """
        return copy.deepcopy(self)

    def set(self, key: str, value: Optional[Any] = None) -> "Param":
        """Set a parameter value.

        Args:
            key (str): The parameter key.
            value (Any): The parameter value.

        Returns:
            Param: The updated Param object.
        """
        if "_" in key:
            raise ValueError("Key cannot contain underscore. Use '-' instead.")
        elif key not in self._param.keys():
            raise ValueError(f"Key {key} does not exist")
        else:
            if value is None:
                value = True
            self._param[key]["value"] = value
        return self

    def clear(self, key: str) -> "Param":
        """Clear a parameter.

        Args:
            key (str): The parameter key.
        """
        self._param[key]["value"] = self._default_param[key]["value"]
        if key in self._dyn_param.keys():
            del self._dyn_param[key]
        return self

    def dyn(self, key: str) -> "Param":
        """Set a current dynamic parameter key.

        Args:
            key (str): The dynamic parameter key.

        Returns:
            Param: The updated Param object.
        """
        if key not in self._param.keys():
            raise ValueError(f"Key {key} does not exist")
        else:
            self._time = 0.0
            self._key = key
        return self

    def change(self, value: Any) -> "Param":
        """Change the value of the dynamic parameter at the current time.

        Args:
            value (float): The new value of the dynamic parameter.

        Returns:
            Param: The updated Param object.
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param.keys():
                self._dyn_param[self._key].append((self._time, value))
            else:
                initial_val = self._param[self._key]["value"]
                self._dyn_param[self._key] = [
                    (0.0, initial_val),
                    (self._time, value),
                ]
            return self

    def hold(self) -> "Param":
        """Hold the current value of the dynamic parameter.

        Returns:
            Param: The updated Param object.
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param.keys():
                last_val = self._dyn_param[self._key][-1][1]
                self.change(last_val)
            else:
                val = self._param[self._key]["value"]
                val_type = self._param[self._key]["type"]
                if val_type == "f32" or val_type == "f64":
                    if isinstance(val, float):
                        self.change(val)
                    else:
                        raise ValueError(
                            f"Key must be float. {val} is given. type: {val_type}"
                        )
                elif val_type == "bool":
                    if isinstance(val, bool):
                        self.change(val)
                    else:
                        raise ValueError(
                            f"Key must be bool. {val} is given. type: {val_type}"
                        )
                else:
                    raise ValueError(
                        f"Key must be float or bool. {val} is given. type: {val_type}"
                    )
        return self

    def export(self, path: str):
        """Export the parameters to a file.

        Args:
            path (str): The path to the export directory.
        """
        if len(self._param.keys()):
            with open(os.path.join(path, "param.toml"), "w") as f:
                f.write("[param]\n")
                for key, value in self._param.items():
                    val = value["value"]
                    key = key.replace("-", "_")
                    if val is not None:
                        if isinstance(val, str):
                            f.write(f'{key} = "{val}"\n')
                        elif isinstance(val, bool):
                            if val:
                                f.write(f"{key} = true\n")
                            else:
                                f.write(f"{key} = false\n")
                        else:
                            f.write(f"{key} = {val}\n")
                    else:
                        f.write(f"{key} = false\n")
        if len(self._dyn_param.keys()):
            with open(os.path.join(path, "dyn_param.txt"), "w") as f:
                for key, vals in self._dyn_param.items():
                    f.write(f"[{key}]\n")
                    for entry in vals:
                        time, val = entry
                        if isinstance(val, float):
                            f.write(f"{time} {val}\n")
                        elif isinstance(val, bool):
                            f.write(f"{time} {float(val)}\n")
                        else:
                            raise ValueError(
                                f"Value must be float or bool. {val} is given."
                            )

    def time(self, time: float) -> "Param":
        """Set the current time for the dynamic parameter.

        Args:
            time (float): The current time.

        Returns:
            Param: The updated Param object.
        """
        if time <= self._time:
            raise ValueError("Time must be increasing")
        else:
            self._time = time
        return self

    def get(self, key: Optional[str] = None) -> bool | float:
        """Get the value of a parameter.

        Args:
            key (Optional[str], optional): The parameter key.
            If not specified, all parameters are returned.

        Returns:
            Any: The value of the parameter.
        """
        if key is None:
            raise ValueError("Key must be specified")
        else:
            return self._param[key]["value"]

    def items(self):
        """Get all parameter items.

        Returns:
            ItemsView: The parameter items.
        """
        return self._param.items()


class SessionManager:
    """Class to manage simulation sessions."""

    def __init__(self, app_root: str, proj_root: str, save_func):
        """Initialize the SessionManager class.

        Args:
            app_root (str): The root directory of the application.
            proj_root (str): The root directory of the project.
            save_func (Callable): The save function.
        """
        self._app_root = app_root
        self._proj_root = proj_root
        self._save_func = save_func
        self._sessions = {}
        self._curr = None

    def list(self):
        """List all sessions.

        Returns:
            dict: The sessions.
        """
        return self._sessions

    def select(self, name: str):
        """Select a session.

        Args:
            name (str): The name of the session.

        Returns:
            Session: The selected session.
        """
        if name not in self._sessions.keys():
            raise ValueError(f"Session {name} does not exist")
        self._curr = name
        return self._sessions[name]

    def current(self):
        """Get the current session.

        Returns:
            Session: The current session.
        """
        if self._curr is None:
            return None
        else:
            return self._sessions[self._curr]

    def create(
        self, scene: FixedScene, name: str = "", delete_if_exists: bool = True
    ) -> "Session":
        """Create a new session.

        Args:
            scene (FixedScene): The scene object.
            name (str): The name of the session. If not specified, current time is used.
            delete_if_exists (bool, optional): Whether to delete the session if it exists.

        Returns:
            Session: The created session.
        """
        if name == "":
            name = f"session-{len(self._sessions)}"
        if name in self._sessions.keys():
            if delete_if_exists:
                session = self._sessions[name]
                if is_running():
                    terminate()
                self._sessions[name].delete()
            else:
                raise ValueError(f"Session {name} already exists")
        session = Session(self._app_root, self._proj_root, name, self._save_func)
        self._sessions[name] = session
        self._curr = name
        return session.init(scene)

    def _terminate_or_raise(self, force: bool):
        """Terminate the solver if it is running, or raise an exception.

        Args:
            force (bool): Whether to force termination.
        """
        if is_running():
            if force:
                terminate()
            else:
                raise ValueError("Solver is running. Terminate first.")

    def delete(self, name: str, force: bool = True):
        """Delete a session.

        Args:
            name (str): The name of the session.
            force (bool, optional): Whether to force deletion.
        """
        self._terminate_or_raise(force)
        if name in self._sessions.keys():
            self._sessions[name].delete()
            del self._sessions[name]
            if name == self._curr:
                self._curr = None

    def clear(self, force: bool = True):
        """Clear all sessions.

        Args:
            force (bool, optional): Whether to force clearing.
        """
        self._terminate_or_raise(force)
        for session in self._sessions.values():
            session.delete()
        self._sessions = {}
        self._curr = None

    def param(self) -> Param:
        """Get a new Param object.

        Returns:
            Param: The Param object.
        """
        return Param(self._proj_root)


class SessionInfo:
    """Class to store session information."""

    def __init__(self, name: str):
        """Initialize the SessionInfo class.

        Args:
            name (str): The name of the session.
            path (str): The path to the session directory.
        """
        self._name = name
        self._path = ""

    def set_path(self, path: str):
        """Set the path to the session directory.

        Args:
            path (str): The path to the session directory.
        """
        self._path = path

    @property
    def name(self) -> str:
        """Get the name of the session."""
        return self._name

    @property
    def path(self) -> str:
        """Get the path to the session directory."""
        return self._path


class Zippable:
    def __init__(self, dirpath: str):
        self._dirpath = dirpath

    def zip(self):
        """Zip the directory."""
        ci_name = Utils.ci_name()
        if ci_name is not None:
            print("CI detected. Skipping zipping.")
        else:
            path = f"{self._dirpath}.zip"
            if os.path.exists(path):
                os.remove(path)
            print(f"zipping to {path}")
            shutil.make_archive(self._dirpath, "zip", self._dirpath)
            print("done")


class SessionExport:
    """Class to handle session export operations."""

    def __init__(self, session: "Session"):
        """Initialize the SessionExport class.

        Args:
            session (Session): The session object.
        """
        self._session = session

    def shell_command(
        self,
        param: Param,
    ) -> str:
        """Generate a shell command to run the solver.

        Args:
            param (Param): The simulation parameters.

        Returns:
            str: The shell command.
        """
        param.export(self._session.info.path)
        program_path = os.path.join(
            self._session._proj_root, "target", "release", "ppf-contact-solver"
        )
        if os.path.exists(program_path):
            command = " ".join(
                [
                    program_path,
                    f"--path {self._session.info.path}",
                    f"--output {self._session.output.path}",
                ]
            )
            path = os.path.join(self._session.info.path, "command.sh")
            with open(path, "w") as f:
                f.write(command)
            os.chmod(path, 0o755)
            return path
        else:
            raise ValueError("Solver does not exist")

    def animation(
        self,
        path: str = "",
        ext="ply",
        include_static: bool = True,
        clear: bool = False,
        options: dict = {},
    ) -> Zippable:
        """Export the animation frames.

        Args:
            path (str): The path to the export directory. If set empty, it will use the default path.
            ext (str, optional): The file extension. Defaults to "ply".
            include_static (bool, optional): Whether to include the static mesh.
            options (dict, optional): Additional arguments passed to a renderer.
            clear (bool, optional): Whether to clear the existing files.
        """
        frames = self._session.get.frame_list()
        if len(frames) == 0:
            raise ValueError("No frames found")
        else:
            options = self._session._update_options(options)
            ci_name = Utils.ci_name()
            if path == "":
                if ci_name is not None:
                    path = os.path.join(self._session.info.path, "preview")
                else:
                    session = self._session
                    scene = session._fixed
                    assert scene is not None
                    path = os.path.join("export", scene._name, session.info.name)

            if os.path.exists(path):
                if clear:
                    shutil.rmtree(path)
            else:
                os.makedirs(path)
            for n, i in enumerate(tqdm(frames, desc="export", ncols=70)):
                self.frame(
                    os.path.join(path, f"frame_{n}.{ext}"),
                    i,
                    include_static,
                    options,
                    delete_exist=clear,
                )
            if shutil.which("ffmpeg") is not None:
                vid_name = "frame.mp4"
                command = f"ffmpeg -hide_banner -loglevel error -y -r 60 -i frame_%d.{ext}.png -pix_fmt yuv420p -b:v 50000k {vid_name}"
                subprocess.run(command, shell=True, cwd=path)
                if self._session._in_jupyter_notebook:
                    from IPython.display import Video, display

                    display(Video(os.path.join(path, vid_name)))
                if ci_name is not None:
                    for file in os.listdir(path):
                        if file.endswith(".png"):
                            os.remove(os.path.join(path, file))

            return Zippable(path)

    def frame(
        self,
        path: str = "",
        frame: Optional[int] = None,
        include_static: bool = True,
        options: dict = {},
        delete_exist: bool = False,
    ) -> "Session":
        """Export a specific frame.

        Args:
            path (str): The path to the export file.
            frame (Optional[int], optional): The frame number. Defaults to None.
            include_static (bool, optional): Whether to include the static mesh.
            options (dict, optional): Additional arguments passed to a renderer.
            delete_exist (bool, optional): Whether to delete the existing file.

        Returns:
            Session: The session object.
        """

        options = self._session._update_options(options)
        if self._session._fixed is None:
            raise ValueError("Scene must be initialized")
        else:
            vert = self._session._fixed.vertex(True)
            if frame is not None:
                result = self._session.get.vertex(frame)
                if result is not None:
                    vert, _ = result
            else:
                result = self._session.get.vertex()
                if result is not None:
                    vert, _ = result
            color = self._session._fixed.color(vert, options)
            self._session._fixed.export(
                vert, color, path, include_static, options, delete_exist
            )
        return self._session


class SessionOutput:
    """Class to handle session output operations."""

    def __init__(self, session: "Session"):
        """Initialize the SessionOutput class.

        Args:
            session (Session): The session object.
        """
        self._session = session

    @property
    def path(self) -> str:
        """Get the path to the output directory."""
        return os.path.join(self._session.info.path, "output")


class SessionLog:
    """Class to handle session log retrieval operations."""

    def __init__(self, session: "Session") -> None:
        src_path = os.path.join(session._proj_root, "src")
        self._session = session
        self._log = CppRustDocStringParser.get_logging_docstrings(src_path)

    def names(self) -> list[str]:
        """Get the list of log names.

        Returns:
            list[str]: The list of log names.
        """
        return list(self._log.keys())

    def _tail_file(self, path: str, n_lines: Optional[int] = None) -> list[str]:
        """Get the last n lines of a file.

        Args:
            path (str): The path to the file.
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the file.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
                lines = [line.rstrip("\n") for line in lines]
                if n_lines is not None:
                    return lines[-n_lines:]
                else:
                    return lines
        return []

    def stdout(self, n_lines: Optional[int] = None) -> list[str]:
        """Get the last n lines of the stdout log file.

        Args:
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the stdout log file.
        """
        return self._tail_file(
            os.path.join(self._session.info.path, "stdout.log"), n_lines
        )

    def stderr(self, n_lines: Optional[int] = None) -> list[str]:
        """Get the last n lines of the stderr log file.

        Args:
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the stderr log file.
        """
        return self._tail_file(
            os.path.join(self._session.info.path, "error.log"), n_lines
        )

    def numbers(self, name: str):
        """Get a pair of numbers from a log file.

        Args:
            name (str): The name of the log file.

        Returns:
            list[list[float]]: The list of pair of numbers.
        """

        def float_or_int(var):
            var = float(var)
            if var.is_integer():
                return int(var)
            else:
                return var

        filename = self._log[name]["filename"]
        path = os.path.join(self._session.info.path, "output", "data", filename)
        entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    entry = line.split(" ")
                    entries.append([float_or_int(entry[0]), float_or_int(entry[1])])
            return entries
        else:
            return None

    def number(self, name: str):
        """Get the latest value from a log file.

        Args:
            name (str): The name of the log file.

        Returns:
            float: The latest value.
        """
        entries = self.numbers(name)
        if entries:
            return entries[-1][1]
        else:
            return None


class SessionGet:
    """Class to handle session data retrieval operations."""

    def __init__(self, session: "Session"):
        """Initialize the SessionGet class.

        Args:
            session (Session): The session object.
        """
        self._session = session
        self._log = SessionLog(session)

    @property
    def log(self) -> SessionLog:
        """Get the session log object."""
        return self._log

    def vertex_frame_count(self) -> int:
        """Get the vertex count.

        Returns:
            int: The vertex count.
        """
        path = os.path.join(self._session.info.path, "output")
        max_frame = 0
        if os.path.exists(path):
            files = os.listdir(path)
            for file in files:
                if file.startswith("vert") and file.endswith(".bin"):
                    frame = int(file.split("_")[1].split(".")[0])
                    max_frame = max(max_frame, frame)
        return max_frame

    def first_frame(self) -> Optional[int]:
        """Get the first frame number.

        Returns:
            int: The first frame number. If no frames are found, return None.
        """
        frames = self.frame_list()
        if len(frames) > 0:
            return frames[0]
        else:
            return None

    def latest_frame(self) -> Optional[int]:
        """Get the latest frame number.

        Returns:
            int: The latest frame number. If no frames are found, return None.
        """
        frames = self.frame_list()
        if len(frames) > 0:
            return frames[-1]
        else:
            return None

    def frame_list(self) -> list[int]:
        """Get the sorted list of frame numbers.

        Returns:
            list[int]: The list of frame numbers.
        """
        path = os.path.join(self._session.info.path, "output")
        if os.path.exists(path):
            files = os.listdir(path)
            frames = []
            for file in files:
                if file.startswith("vert") and file.endswith(".bin"):
                    frame = int(file.split("_")[1].split(".")[0])
                    frames.append(frame)
            if len(frames) > 0:
                return sorted(frames)
        return []

    def saved(self) -> list[int]:
        """Get the list of saved frame numbers.

        Returns:
            list[int]: The list of saved frame numbers.
        """
        result = []
        output_path = os.path.join(self._session.info.path, "output")
        if os.path.exists(output_path):
            for file in os.listdir(output_path):
                if file.startswith("state_") and file.endswith(".bin.gz"):
                    frame = int(file.split("_")[1].split(".")[0])
                    result.append(frame)
        return result

    def vertex(self, n: Optional[int] = None) -> Optional[tuple[np.ndarray, int]]:
        """Get the vertex data for a specific frame.

        Args:
            n (Optional[int], optional): The frame number. If not specified, the latest frame is returned. Defaults to None.

        Returns:
            Optional[tuple[np.ndarray, int]]: The vertex data and frame number.
        """
        if self._session._fixed is None:
            raise ValueError("Scene must be initialized")
        else:
            path = os.path.join(self._session.info.path, "output")
            if os.path.exists(path):
                if n is None:
                    files = os.listdir(path)
                    frames = []
                    for file in files:
                        if file.startswith("vert") and file.endswith(".bin"):
                            frame = int(file.split("_")[1].split(".")[0])
                            frames.append(frame)
                    if len(frames) > 0:
                        frames = sorted(frames)
                        last_frame = frames[-1]
                        path = os.path.join(path, f"vert_{last_frame}.bin")
                        try:
                            with open(path, "rb") as f:
                                data = f.read()
                                vert = np.frombuffer(data, dtype=np.float32).reshape(
                                    -1, 3
                                )
                            return (
                                vert,
                                last_frame,
                            )
                        except ValueError:
                            return None
                else:
                    try:
                        path = os.path.join(path, f"vert_{n}.bin")
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                data = f.read()
                                vert = np.frombuffer(data, dtype=np.float32).reshape(
                                    -1, 3
                                )
                            return (vert, n)
                    except ValueError:
                        pass
                    return None
        return None


class Session:
    """Class to manage a simulation session."""

    def __init__(self, app_root: str, proj_root: str, name: str, save_func):
        """Initialize the Session class.

        Args:
            app_root (str): The root directory of the application.
            proj_root (str): The root directory of the project.
            name (str): The name of the session.
            save_func (Callable): The save function.
        """
        self._in_jupyter_notebook = Utils.in_jupyter_notebook()
        self._app_root = app_root
        self._proj_root = proj_root
        self._fixed = None
        self._save_func = save_func
        self._update_preview_interval = 1.0 / 60.0
        self._update_terminal_interval = 1.0 / 30.0
        self._update_table_interval = 0.25
        self._info = SessionInfo(name)
        self._export = SessionExport(self)
        self._get = SessionGet(self)
        self._output = SessionOutput(self)
        self.param = None
        self._default_opts: dict[str, Any] = {
            "flat_shading": False,
            "wireframe": False,
            "pin": False,
            "stitch": False,
        }
        self.delete()

    @property
    def info(self) -> SessionInfo:
        """Get the session information."""
        return self._info

    @property
    def export(self) -> SessionExport:
        """Get the session export object."""
        return self._export

    @property
    def get(self) -> SessionGet:
        """Get the session get object."""
        return self._get

    @property
    def output(self) -> SessionOutput:
        """Get the session output object."""
        return self._output

    def print(self, message):
        """Print a message.

        Args:
            message (str): The message to print.
        """
        if self._in_jupyter_notebook:
            from IPython.display import display

            display(message)
        else:
            print(message)

    def delete(self):
        """Delete the session."""
        if os.path.exists(self.info.path):
            shutil.rmtree(self.info.path)

    def _check_ready(self):
        """Check if the session is ready."""
        if self._fixed is None:
            raise ValueError("Scene must be initialized")

    def init(self, scene: FixedScene) -> "Session":
        """Initialize the session with a fixed scene.

        Args:
            scene (FixedScene): The fixed scene.

        Returns:
            Session: The initialized session.
        """
        path = os.path.expanduser(os.path.join(self._app_root, self.info.name))
        self.info.set_path(path)
        if is_running():
            self.print("Solver is already running. Teriminate first.")
            if self._in_jupyter_notebook:
                from IPython.display import display

                display(self._terminate_button("Terminate Now"))
            return self

        self._fixed = scene

        if os.path.exists(self.info.path):
            shutil.rmtree(self.info.path)
        else:
            os.makedirs(self.info.path)

        if self._fixed is not None:
            self._fixed.export_fixed(self.info.path, True)
        else:
            raise ValueError("Scene and param must be initialized")

        return self

    def finished(self) -> bool:
        """Check if the session is finished.

        Returns:
            bool: True if the session is finished, False otherwise.
        """
        finished_path = os.path.join(self.output.path, "finished.txt")
        error = self.get.log.stderr()
        if len(error) > 0:
            for line in error:
                print(line)
        return os.path.exists(finished_path)

    def resume(
        self,
        frame: int = -1,
        param: Optional[Param] = None,
        force: bool = True,
        blocking: bool = False,
    ) -> "Session":
        if self.param is None:
            print("Session is not yet started")
            return self
        if frame == -1:
            saved = self.get.saved()
            if len(saved) > 0:
                frame = max(saved)
            else:
                return self
        if frame > 0:
            if param is not None:
                self.param = copy.deepcopy(param)
            self.param.set("load", frame)
            return self.start(self.param, force, blocking)
        else:
            print(f"No saved state found: frame: {frame}")
            return self

    def start(self, param: Param, force: bool = False, blocking=False) -> "Session":
        """Start the session.

        For Jupyter Notebook, the function will return immediately and the solver
        will run in the background. If blocking is set to True, the function will block
        until the solver is finished.
        When Jupiter Notebook is not detected, the function will block until the solver
        is finished.

        Args:
            param (Param): The simulation parameters.
            force (bool, optional): Whether to force starting the simulation.
            blocking (bool, optional): Whether to block the execution.

        Returns:
            Session: The started session.
        """
        gpu_count = Utils.get_gpu_count()
        if gpu_count == 0:
            raise ValueError("GPU is not detected.")

        driver_version = Utils.get_driver_version()
        min_driver_version = 520
        if driver_version:
            if driver_version < min_driver_version:
                raise ValueError(
                    f"Driver version is {driver_version}. It must be newer than {min_driver_version}"
                )
        else:
            raise ValueError("Driver version could not be detected.")

        self._check_ready()
        if is_running():
            if force:
                terminate()
            else:
                from IPython.display import display

                self.print("Solver is already running. Teriminate first.")
                display(self._terminate_button("Terminate Now"))
                return self

        frame = self.get.saved()
        if frame and not force:
            from IPython.display import display

            self.print(f"Solver has saved states. Resuming from {max(frame)}")
            return self.resume(max(frame), param, True, blocking)

        self.param = copy.deepcopy(param)
        self._save_func()
        cmd_path = self.export.shell_command(self.param)
        err_path = os.path.join(self.info.path, "error.log")
        log_path = os.path.join(self.info.path, "stdout.log")
        command = open(cmd_path, "r").read().split()
        process = subprocess.Popen(
            command,
            stdout=open(log_path, "w"),
            stderr=open(err_path, "w"),
            start_new_session=True,
            cwd=self._proj_root,
        )
        while not os.path.exists(log_path) and not os.path.exists(err_path):
            time.sleep(1)
        if process.poll() is not None:
            display_log(open(err_path, "r").readlines())
            raise ValueError("Solver failed to start")
        else:
            init_path = os.path.join(self.info.path, "output", "data", "initialize.out")
            time.sleep(1)
            while is_running():
                if os.path.exists(init_path):
                    break
                time.sleep(1)
            if not os.path.exists(init_path):
                display_log(open(err_path, "r").readlines())
                raise ValueError("Solver failed to start")
        if blocking or not self._in_jupyter_notebook:
            print(f">>> Log path: {log_path}")
            print(">>> Waiting for solver to finish...")
            total_frames = param.get("frames")
            assert isinstance(total_frames, int)
            with tqdm(total=total_frames, desc="Progress") as pbar:
                last_frame = 0
                while process.poll() is None:
                    frame = self.get.latest_frame()
                    if frame is not None and frame > last_frame:
                        pbar.update(frame - last_frame)
                        last_frame = frame
                    time.sleep(1)
            if os.path.exists(err_path):
                err_lines = open(err_path, "r").readlines()
            else:
                err_lines = []
            if len(err_lines) > 0:
                print("*** Solver FAILED ***")
            else:
                print("*** Solver finished ***")
            n_logs = 32
            log_lines = open(log_path, "r").readlines()
            print(">>> Log:")
            for line in log_lines[-n_logs:]:
                print(line.rstrip())
            if len(err_lines) > 0:
                print(">>> Error:")
                for line in err_lines:
                    print(line.rstrip())
                print(f">>> Error log path: {err_path}")

        strain_limit_eps = param.get("strain-limit-eps")
        self._default_opts["max-area"] = 1.0 + strain_limit_eps
        return self

    def _terminate_button(self, description: str = "Terminate Solver"):
        """Create a terminate button.

        Args:
            description (str, optional): The button description.

        Returns:
            Optional[widgets.Button]: The terminate button.
        """
        if self._in_jupyter_notebook:
            import ipywidgets as widgets

            def _terminate(button):
                button.disabled = True
                button.description = "Terminating..."
                terminate()
                while is_running():
                    time.sleep(0.25)
                button.description = "Terminated"

            button = widgets.Button(description=description)
            button.on_click(_terminate)
            return button
        else:
            return None

    def _save_and_quit_button(self, description: str = "Save and Quit"):
        """Create a save-and-quit button.

        Args:
            description (str, optional): The button description.

        Returns:
            Optional[widgets.Button]: The save-and-quit button.
        """
        if self._in_jupyter_notebook:
            import ipywidgets as widgets

            def _save_and_quit(button):
                button.disabled = True
                button.description = "Requesting..."
                request_save_and_quit(os.path.join(self.info.path, "output"))
                while is_running():
                    time.sleep(0.25)
                button.description = "Done"

            button = widgets.Button(description=description)
            button.on_click(_save_and_quit)
            return button
        else:
            return None

    def _update_options(self, options: dict) -> dict:
        options = dict(options)
        for key, value in self._default_opts.items():
            if key not in options.keys():
                options[key] = value
        return options

    def preview(
        self, options: dict = {}, live_update: bool = True, engine: str = "threejs"
    ) -> Optional["Plot"]:
        """Live view the session.

        Args:
            options (dict, optional): The render options.
            live_update (bool, optional): Whether to enable live update.
            engine (str, optional): The rendering engine. Defaults to "threejs".

        Returns:
            Optional[Plot]: The plot object.
        """
        options = self._update_options(options)
        if self._in_jupyter_notebook:
            import ipywidgets as widgets
            from IPython.display import display

            if self._fixed is None:
                raise ValueError("Scene must be initialized")
            else:
                result = self.get.vertex()
                if result is None:
                    vert, curr_frame = self._fixed.vertex(True), 0
                else:
                    vert, curr_frame = result
                plot = self._fixed.preview(
                    vert, options, show_slider=False, engine=engine
                )

            table = widgets.HTML()
            terminate_button = self._terminate_button()
            save_and_quit_button = self._save_and_quit_button()

            def convert_integer(number) -> str:
                if number is None:
                    return "N/A"
                elif number < 1000:
                    return str(number)
                elif number < 1_000_000:
                    return f"{number / 1_000:.2f}k"
                elif number < 1_000_000_000:
                    return f"{number / 1_000_000:.2f}M"
                else:
                    return f"{number / 1_000_000_000:.2f}B"

            def convert_time(time) -> str:
                if time is None:
                    return "N/A"
                elif time < 1_000:
                    return f"{int(time)}ms"
                elif time < 60_000:
                    return f"{time / 1_000:.2f}s"
                else:
                    return f"{time / 60_000:.2f}m"

            if live_update and is_running():

                def update_dataframe(table, curr_frame):
                    time_per_frame = convert_time(self.get.log.number("time-per-frame"))
                    time_per_step = convert_time(self.get.log.number("time-per-step"))
                    n_contact = convert_integer(self.get.log.number("num-contact"))
                    n_newton = convert_integer(self.get.log.number("newton-steps"))
                    max_sigma = self.get.log.number("max-sigma")
                    n_pcg = convert_integer(self.get.log.number("pcg-iter"))
                    data = {
                        "Frame": [str(curr_frame)],
                        "Time/Frame": [time_per_frame],
                        "Time/Step": [time_per_step],
                        "#Contact": [n_contact],
                        "#Newton": [n_newton],
                        "#PCG": [n_pcg],
                    }
                    if max_sigma is not None and max_sigma > 0.0:
                        stretch = f"{100.0 * (max_sigma - 1.0):.2f}%"
                        data["Max Stretch"] = [stretch]
                    df = pd.DataFrame(data)
                    table.value = df.to_html(
                        classes="table table-striped", border=0, index=False
                    )

                def live_preview(self):
                    nonlocal plot
                    nonlocal terminate_button
                    nonlocal save_and_quit_button
                    nonlocal table
                    nonlocal options
                    nonlocal curr_frame
                    assert plot is not None
                    while True:
                        last_frame = self.get.latest_frame()
                        if curr_frame != last_frame:
                            curr_frame = last_frame
                            result = self.get.vertex(curr_frame)
                            if result is not None:
                                vert, _ = result
                                color = self._fixed.color(vert, options)
                                update_dataframe(table, curr_frame)
                                plot.update(vert, color)
                        if not is_running():
                            break
                        time.sleep(self._update_preview_interval)
                    assert terminate_button is not None
                    assert save_and_quit_button is not None
                    terminate_button.disabled = True
                    terminate_button.description = "Terminated"
                    save_and_quit_button.disabled = True
                    time.sleep(self._update_preview_interval)
                    last_frame = self.get.latest_frame()
                    update_dataframe(table, last_frame)
                    vert, _ = self.get.vertex(last_frame)
                    color = self._fixed.color(vert, options)
                    plot.update(vert, color)

                def live_table(self):
                    nonlocal table
                    while True:
                        update_dataframe(table, curr_frame)
                        if not is_running():
                            break
                        time.sleep(self._update_table_interval)

                threading.Thread(target=live_preview, args=(self,)).start()
                threading.Thread(target=live_table, args=(self,)).start()
                display(widgets.HBox((terminate_button, save_and_quit_button)))

            display(table)
            return plot
        else:
            return None

    def animate(self, options: dict = {}, engine: str = "threejs") -> "Session":
        """Show the animation.

        Args:
            options (dict, optional): The render options.

        Returns:
            Session: The animated session.
        """
        frames = self.get.frame_list()
        if len(frames) == 0:
            raise ValueError("No frames found")
        else:
            options = self._update_options(options)
            offset = frames[0]

            if self._in_jupyter_notebook:
                import ipywidgets as widgets
                from IPython.display import display

                if self._fixed is None:
                    raise ValueError("Scene must be initialized")
                else:
                    try:
                        if self._fixed is not None:
                            verts = []
                            for i in tqdm(frames, desc="Loading frames", ncols=70):
                                result = self.get.vertex(i)
                                if result is not None:
                                    verts.append(result[0])

                            plot = self._fixed.preview(
                                verts[0],
                                options,
                                show_slider=False,
                                engine=engine,
                            )

                            def update(frame: int):
                                nonlocal offset
                                nonlocal verts
                                nonlocal plot
                                assert plot is not None
                                if self._fixed is not None:
                                    vert = verts[frame - offset]
                                    color = self._fixed.color(vert, options)
                                    plot.update(vert, color)

                            slider = widgets.IntSlider(
                                value=offset,
                                min=offset,
                                max=len(verts) + offset - 1,
                                step=1,
                                description="Frame",
                            )

                            def on_value_change(change):
                                update(change["new"])

                            slider.observe(on_value_change, names="value")
                            display(slider)
                    except Exception as _:
                        pass
            return self

    def stream(self, n_lines=40) -> "Session":
        """Stream the session logs.

        Args:
            n_lines (int, optional): The number of lines to stream. Defaults to 40.

        Returns:
            Session: The session object.
        """
        if self._in_jupyter_notebook:
            import ipywidgets as widgets
            from IPython.display import display

            log_widget = widgets.HTML()
            display(log_widget)
            button = widgets.Button(description="Stop Live Stream")
            terminate_button = self._terminate_button()
            save_and_quit_button = self._save_and_quit_button()
            display(widgets.HBox((button, terminate_button, save_and_quit_button)))

            assert button is not None
            assert terminate_button is not None
            assert save_and_quit_button is not None

            stop = False
            log_path = os.path.join(self.info.path, "stdout.log")
            err_path = os.path.join(self.info.path, "error.log")
            if os.path.exists(log_path):

                def live_stream(self):
                    nonlocal stop
                    nonlocal button
                    nonlocal log_widget
                    nonlocal log_path
                    nonlocal err_path
                    nonlocal terminate_button
                    nonlocal save_and_quit_button

                    assert button is not None
                    assert terminate_button is not None
                    assert save_and_quit_button is not None

                    while not stop:
                        result = subprocess.run(
                            ["tail", f"-n{n_lines}", log_path],
                            capture_output=True,
                            text=True,
                        )
                        log_widget.value = (
                            CONSOLE_STYLE
                            + f"<pre style='no-scroll'>{result.stdout.strip()}</pre>"
                        )
                        if not is_running():
                            log_widget.value += "<p style='color: red;'>Terminated.</p>"
                            if os.path.exists(err_path):
                                file = open(err_path, "r")
                                lines = file.readlines()
                                if len(lines) > 0:
                                    log_widget.value += "<p style='color: red;'>"
                                    for line in lines:
                                        log_widget.value += line + "\n"
                                    log_widget.value += "</p>"

                            button.disabled = True
                            terminate_button.disabled = True
                            save_and_quit_button.disabled = True
                            break
                        time.sleep(self._update_terminal_interval)

                thread = threading.Thread(target=live_stream, args=(self,))
                thread.start()

                def toggle_stream(b):
                    nonlocal stop
                    nonlocal thread
                    if thread.is_alive():
                        stop = True
                        thread.join()
                        b.description = "Start Live Stream"
                    else:
                        thread = threading.Thread(target=live_stream, args=(self,))
                        stop = False
                        thread.start()
                        b.description = "Stop Live Stream"

                button.on_click(toggle_stream)
            else:
                log_widget.value = "No log file found."
                terminate_button.disabled = True
                save_and_quit_button.disabled = True
                button.disabled = True

        return self


def is_running() -> bool:
    """Check if the solver is running.

    Returns:
        bool: True if the solver is running, False otherwise.
    """
    for proc in psutil.process_iter(["pid", "name", "status"]):
        if (
            PROCESS_NAME in proc.info["name"]
            and proc.info["status"] != psutil.STATUS_ZOMBIE
        ):
            return True
    return False


def terminate():
    """Terminate the solver."""
    for proc in psutil.process_iter(["pid", "name", "status"]):
        if (
            PROCESS_NAME in proc.info["name"]
            and proc.info["status"] != psutil.STATUS_ZOMBIE
        ):
            pid = proc.info["pid"]
            os.kill(pid, signal.SIGTERM)


def request_save_and_quit(watch_dir):
    """Request to save and quit the solver."""
    open(os.path.join(watch_dir, "save_and_quit"), "w").close()


def display_log(lines: list[str]):
    """Display the log lines.

    Args:
        lines (list[str]): The log lines.
    """
    if Utils.in_jupyter_notebook():
        import ipywidgets as widgets
        from IPython.display import display

        log_widget = widgets.HTML()
        text = "\n".join(lines)
        log_widget.value = CONSOLE_STYLE + f"<pre style='no-scroll'>{text}</pre>"
        display(log_widget)
    else:
        for line in lines:
            print(line)
