# File: _session_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import asyncio
import copy
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm

from ._param_ import ParamHolder, app_param
from ._parse_ import CppRustDocStringParser
from ._scene_ import FixedScene
from ._utils_ import Utils

if TYPE_CHECKING:
    from ._plot_ import Plot

RECOVERABLE_FIXED_SESSION_NAME = "fixed_session.pickle"

CONSOLE_STYLE = """
    <style>
        .no-scroll {
            overflow: hidden;
            white-space: pre-wrap;
            font-family: monospace;
        }
    </style>
    """


class ParamManager:
    """Class to manage simulation parameters."""

    def __init__(self):
        """Initialize the Param class.

        Args:
            app_root (str): The root directory of the application.
        """
        self._key = None
        self._param = ParamHolder(app_param())
        self._default_param = self._param.copy()
        self._time = 0.0
        self._dyn_param = {}

    def copy(self) -> "ParamManager":
        """Copy the Param object.

        Returns:
            Param: The copied Param object.
        """
        return copy.deepcopy(self)

    def set(self, key: str, value: Any | None = None) -> "ParamManager":
        """Set a parameter value.

        Args:
            key (str): The parameter key.
            value (Any): The parameter value.

        Returns:
            Param: The updated Param object.
        """
        if "_" in key:
            raise ValueError("Key cannot contain underscore. Use '-' instead.")
        elif key not in self._param.key_list():
            raise ValueError(f"Key {key} does not exist")
        else:
            if value is None:
                value = True
            self._param.set(key, value)
        return self

    def clear_all(self):
        """Clear all parameters to their default values."""
        self._param = self._default_param.copy()
        self._dyn_param = {}

    def clear(self, key: str) -> "ParamManager":
        """Clear a parameter.

        Args:
            key (str): The parameter key.
        """
        self._param.set(key, self._default_param.get(key))
        if key in self._dyn_param:
            del self._dyn_param[key]
        return self

    def dyn(self, key: str) -> "ParamManager":
        """Set a current dynamic parameter key.

        Args:
            key (str): The dynamic parameter key.

        Returns:
            Param: The updated Param object.
        """
        if key not in self._param.key_list():
            raise ValueError(f"Key {key} does not exist")
        else:
            self._time = 0.0
            self._key = key
        return self

    def change(self, value: Any) -> "ParamManager":
        """Change the value of the dynamic parameter at the current time.

        Args:
            value (float): The new value of the dynamic parameter.

        Returns:
            Param: The updated Param object.
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param:
                self._dyn_param[self._key].append((self._time, value))
            else:
                initial_val = self._param.get(self._key)
                self._dyn_param[self._key] = [
                    (0.0, initial_val),
                    (self._time, value),
                ]
            return self

    def hold(self) -> "ParamManager":
        """Hold the current value of the dynamic parameter.

        Returns:
            Param: The updated Param object.
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param:
                last_val = self._dyn_param[self._key][-1][1]
                self.change(last_val)
            else:
                val = self._param.get(self._key)
                self.change(val)
        return self

    def export(self, path: str):
        """Export the parameters to a file.

        Args:
            path (str): The path to the export directory.
        """
        if len(self._param.key_list()):
            with open(os.path.join(path, "param.toml"), "w") as f:
                f.write("[param]\n")
                for key, val in self._param.items():
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

    def time(self, time: float) -> "ParamManager":
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

    def get(self, key: str | None = None) -> Any:
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
            return self._param.get(key)

    def items(self):
        """Get all parameter items.

        Returns:
            ItemsView: The parameter items.
        """
        return self._param.items()


class SessionManager:
    """Class to manage simulation sessions."""

    def __init__(self, app_name: str, app_root: str, proj_root: str, data_dirpath: str):
        """Initialize the SessionManager class.

        Args:
            app_name (str): The name of the application.
            app_root (str): The root directory of the application.
            proj_root (str): The root directory of the project.
            data_dirpath (str): The data directory path.
        """
        self._app_name = app_name
        self._app_root = app_root
        self._proj_root = proj_root
        self._data_dirpath = data_dirpath
        self._sessions = {}

    def list(self):
        """List all sessions.

        Returns:
            dict: The sessions.
        """
        return self._sessions

    def select(self, name: str = "session"):
        """Select a session.

        Args:
            name (str): The name of the session.

        Returns:
            Session: The selected session.
        """
        if name not in self._sessions:
            raise ValueError(f"Session {name} does not exist")
        return self._sessions[name]

    def create(self, scene: FixedScene, name: str = "") -> "Session":
        """Create a new session.

        Args:
            scene (FixedScene): The scene object.
            name (str): The name of the session. If not specified, defaults to "session".

        Returns:
            Session: The created session.
        """
        assert isinstance(scene, FixedScene), "Scene must be a FixedScene object"
        autogenerated = None
        if name == "":
            base_name = "session"
            name = base_name
            counter = 0
            while name in self._sessions:
                counter += 1
                name = f"{base_name}-{counter}"
            autogenerated = counter
        session = Session(
            self._app_name,
            self._app_root,
            self._proj_root,
            self._data_dirpath,
            name,
            autogenerated,
        )
        self._sessions[name] = session
        return session.init(scene)

    def _terminate_or_raise(self, force: bool):
        """Terminate the solver if it is running, or raise an exception.

        Args:
            force (bool): Whether to force termination.
        """
        if Utils.busy():
            if force:
                Utils.terminate()
            else:
                raise ValueError("Solver is running. Terminate first.")

    def delete(self, name: str, force: bool = True):
        """Delete a session.

        Args:
            name (str): The name of the session.
            force (bool, optional): Whether to force deletion.
        """
        self._terminate_or_raise(force)
        if name in self._sessions:
            self._sessions[name].delete()
            del self._sessions[name]

    def clear(self, force: bool = True):
        """Clear all sessions.

        Args:
            force (bool, optional): Whether to force clearing.
        """
        self._terminate_or_raise(force)
        for session in self._sessions.values():
            session.delete()
        self._sessions = {}


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

    def set_path(self, path: str) -> "SessionInfo":
        """Set the path to the session directory.

        Args:
            path (str): The path to the session directory.
        """
        self._path = path
        return self

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

    def __init__(self, fixed_session: "FixedSession"):
        """Initialize the SessionExport class.

        Args:
            session (FixedSession): The fixed session object.
        """
        self._fixed_session = fixed_session
        self._session = fixed_session.session

    def shell_command(
        self,
        param: ParamManager,
    ) -> str:
        """Generate a shell command to run the solver.

        Args:
            param (Param): The simulation parameters.

        Returns:
            str: The shell command.
        """
        param.export(self._fixed_session.info.path)

        # Platform-specific solver path and script generation
        if os.name == "nt":  # Windows
            program_path = os.path.join(
                self._session.proj_root, "target", "release", "ppf-contact-solver.exe"
            )
            # Generate batch file
            command = f"""@echo off
set SOLVER_PATH={program_path}

if not exist "%SOLVER_PATH%" (
    echo Error: Solver does not exist at %SOLVER_PATH% >&2
    exit /b 1
)

"%SOLVER_PATH%" --path {self._fixed_session.info.path} --output {self._fixed_session.output.path} %*
"""
            path = os.path.join(self._fixed_session.info.path, "command.bat")
            with open(path, "w") as f:
                f.write(command)
        else:  # Linux/Mac
            program_path = os.path.join(
                self._session.proj_root, "target", "release", "ppf-contact-solver"
            )
            # Generate shell script that checks for solver existence at runtime
            command = f"""#!/bin/bash
SOLVER_PATH="{program_path}"

if [ ! -f "$SOLVER_PATH" ]; then
    echo "Error: Solver does not exist at $SOLVER_PATH" >&2
    exit 1
fi

"$SOLVER_PATH" --path {self._fixed_session.info.path} --output {self._fixed_session.output.path} "$@"
"""
            path = os.path.join(self._fixed_session.info.path, "command.sh")
            with open(path, "w") as f:
                f.write(command)
            if os.name != "nt":  # chmod not needed on Windows
                os.chmod(path, 0o755)
        return path

    def animation(
        self,
        path: str = "",
        ext="ply",
        include_static: bool = True,
        clear: bool = False,
        options: Optional[dict] = None,
    ) -> Zippable:
        """Export the animation frames.

        Args:
            path (str): The path to the export directory. If set empty, it will use the default path.
            ext (str, optional): The file extension. Defaults to "ply".
            include_static (bool, optional): Whether to include the static mesh.
            options (dict, optional): Additional arguments passed to a renderer.
            clear (bool, optional): Whether to clear the existing files.
        """
        if options is None:
            options = {}
        options = self._fixed_session.update_options(options)
        ci_name = Utils.ci_name()
        if path == "":
            if ci_name is not None:
                path = os.path.join(self._fixed_session.info.path, "preview")
            else:
                scene = self._session.fixed_scene
                assert scene is not None
                path = os.path.join(
                    "export",
                    self._fixed_session.session.app_name,
                    self._fixed_session.info.name,
                )

        # Check if frames are available
        latest_frame = self._fixed_session.get.latest_frame()
        if latest_frame == 0:
            if Utils.busy():
                print(
                    "No frames available yet. Waiting for simulation to generate frames..."
                )
                # Wait for frames to become available
                while Utils.busy() and self._fixed_session.get.latest_frame() == 0:
                    time.sleep(1)
                latest_frame = self._fixed_session.get.latest_frame()
                if latest_frame == 0:
                    print("Simulation finished but no frames were generated.")
                    print(
                        "Please ensure the simulation ran successfully and generated output frames."
                    )
                    return Zippable(
                        path if os.path.exists(path) else self._fixed_session.info.path
                    )
            else:
                print("No animation frames available to export.")
                print(
                    "Please run the simulation first using session.start() to generate frames."
                )
                return Zippable(
                    path if os.path.exists(path) else self._fixed_session.info.path
                )

        # Only print export message in CI mode
        if ci_name is not None:
            print(f"Exporting animation to {path}")
        if os.path.exists(path):
            if clear:
                shutil.rmtree(path)
        else:
            os.makedirs(path)

        # On Windows, skip rendering (pyrender doesn't work in headless mode)
        is_windows = sys.platform == "win32"
        if is_windows:
            options["skip_render"] = True

        for i in tqdm(range(latest_frame), desc="export", ncols=70):
            self.frame(
                os.path.join(path, f"frame_{i}.{ext}"),
                i,
                include_static,
                options,
                delete_exist=clear,
            )

        if shutil.which("ffmpeg") is not None:
            vid_name = "frame.mp4"
            command = f"ffmpeg -hide_banner -loglevel error -y -r 60 -i frame_%d.{ext}.png -pix_fmt yuv420p -b:v 50000k {vid_name}"
            subprocess.run(command, shell=True, cwd=path)
            if Utils.in_jupyter_notebook():
                from IPython.display import Video, display

                display(Video(os.path.join(path, vid_name), embed=True))

            if ci_name is not None:
                for file in os.listdir(path):
                    if file.endswith(".png"):
                        os.remove(os.path.join(path, file))

        return Zippable(path)

    def frame(
        self,
        path: str = "",
        frame: int | None = None,
        include_static: bool = True,
        options: Optional[dict] = None,
        delete_exist: bool = False,
    ) -> "FixedSession":
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

        if options is None:
            options = {}
        options = self._fixed_session.update_options(options)
        if self._fixed_session.fixed_scene is None:
            raise ValueError("Scene must be initialized")
        else:
            fixed_scene = self._fixed_session.session.fixed_scene
            if not fixed_scene:
                raise ValueError("Fixed scene is not initialized")
            else:
                vert = fixed_scene.vertex(True)
                if frame is not None:
                    result = self._fixed_session.get.vertex(frame)
                    if result is not None:
                        vert, _ = result
                else:
                    result = self._fixed_session.get.vertex()
                    if result is not None:
                        vert, _ = result
                color = self._fixed_session.fixed_scene.color(vert, options)
                fixed_scene.export(
                    vert, color, path, include_static, options, delete_exist
                )
        return self._fixed_session


class SessionOutput:
    """Class to handle session output operations."""

    def __init__(self, session: "FixedSession"):
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

    def __init__(self, fixed_session: "FixedSession") -> None:
        src_path = os.path.join(fixed_session.session.proj_root, "src")
        self._fixed_session = fixed_session
        self._log = CppRustDocStringParser.get_logging_docstrings(src_path)

    def names(self) -> list[str]:
        """Get the list of log names.

        Returns:
            list[str]: The list of log names.
        """
        return list(self._log.keys())

    def _tail_file(self, path: str, n_lines: int | None = None) -> list[str]:
        """Get the last n lines of a file.

        Args:
            path (str): The path to the file.
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the file.
        """
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                lines = [line.rstrip("\n") for line in lines]
                if n_lines is not None:
                    return lines[-n_lines:]
                else:
                    return lines
        return []

    def stdout(self, n_lines: int | None = None) -> list[str]:
        """Get the last n lines of the stdout log file.

        Args:
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the stdout log file.
        """
        return self._tail_file(
            os.path.join(self._fixed_session.info.path, "stdout.log"), n_lines
        )

    def stderr(self, n_lines: int | None = None) -> list[str]:
        """Get the last n lines of the stderr log file.

        Args:
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the stderr log file.
        """
        return self._tail_file(
            os.path.join(self._fixed_session.info.path, "error.log"), n_lines
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

        if name not in self._log:
            return None
        filename = self._log[name]["filename"]
        path = os.path.join(self._fixed_session.info.path, "output", "data", filename)
        entries = []
        if os.path.exists(path):
            with open(path) as f:
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

    def summary(self):
        """Get a summary of the session log.

        Returns:
            dict: A dictionary containing the summary of the session log. Each key is a log name and the value is the latest value.
        """
        time_per_frame = convert_time(self.number("time-per-frame"))
        time_per_step = convert_time(self.number("time-per-step"))
        n_contact = convert_integer(self.number("num-contact"))
        n_newton = convert_integer(self.number("newton-steps"))
        max_sigma = self.number("max-sigma")
        n_pcg = convert_integer(self.number("pcg-iter"))
        result = {
            "time-per-frame": time_per_frame,
            "time-per-step": time_per_step,
            "num-contact": n_contact,
            "newton-steps": n_newton,
            "pcg-iter": n_pcg,
        }
        if max_sigma is not None and max_sigma > 0.0:
            result["stretch"] = f"{100.0 * (max_sigma - 1.0):.2f}%"
        return result


class SessionGet:
    """Class to handle session data retrieval operations."""

    def __init__(self, fixed_session: "FixedSession"):
        """Initialize the SessionGet class.

        Args:
            session (Session): The session object.
        """
        self._fixed_session = fixed_session
        self._log = SessionLog(fixed_session)

    @property
    def log(self) -> SessionLog:
        """Get the session log object."""
        return self._log

    def vertex_frame_count(self) -> int:
        """Get the vertex count.

        Returns:
            int: The vertex count.
        """
        path = os.path.join(self._fixed_session.info.path, "output")
        max_frame = 0
        if os.path.exists(path):
            files = os.listdir(path)
            for file in files:
                if file.startswith("vert") and file.endswith(".bin"):
                    frame = int(file.split("_")[1].split(".")[0])
                    max_frame = max(max_frame, frame)
        return max_frame

    def latest_frame(self) -> int:
        """Get the latest frame number.

        Returns:
            int: The latest frame number.
        """
        path = os.path.join(self._fixed_session.info.path, "output")
        if os.path.exists(path):
            files = os.listdir(path)
            frames = []
            for file in files:
                if file.startswith("vert") and file.endswith(".bin"):
                    frame = int(file.split("_")[1].split(".")[0])
                    frames.append(frame)
            if len(frames) > 0:
                return sorted(frames)[-1]
        return 0

    def saved(self) -> list[int]:
        """Get the list of saved frame numbers.

        Returns:
            list[int]: The list of saved frame numbers.
        """
        result = []
        output_path = os.path.join(self._fixed_session.info.path, "output")
        if os.path.exists(output_path):
            for file in os.listdir(output_path):
                if file.startswith("state_") and file.endswith(".bin.gz"):
                    frame = int(file.split("_")[1].split(".")[0])
                    result.append(frame)
        return result

    def vertex(self, n: int | None = None) -> tuple[np.ndarray, int] | None:
        """Get the vertex data for a specific frame.

        Args:
            n (Optional[int], optional): The frame number. If not specified, the latest frame is returned. Defaults to None.

        Returns:
            Optional[tuple[np.ndarray, int]]: The vertex data and frame number.
        """
        path = os.path.join(self._fixed_session.info.path, "output")
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
                            vert = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
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
                            vert = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
                        return (vert, n)
                except ValueError:
                    pass
        return None

    def command(self) -> str | None:
        """Get the path to the command.sh file.

        Returns:
            Optional[str]: The path to the command.sh file if it exists, None otherwise.
        """
        if os.name == "nt":  # Windows
            command_path = os.path.join(self._fixed_session.info.path, "command.bat")
        else:
            command_path = os.path.join(self._fixed_session.info.path, "command.sh")
        if os.path.exists(command_path):
            return command_path
        return None

    def param_summary(self) -> list[str]:
        """Get the parameter summary from the param_summary.txt file.

        Returns:
            list[str]: The lines from the parameter summary file, or empty list if file doesn't exist.
        """
        summary_path = os.path.join(self._fixed_session.info.path, "param_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                return [line.rstrip("\n") for line in f.readlines()]
        return []

    def nvidia_smi(self) -> None:
        """Read and print the exported nvidia-smi outputs.

        Reads both nvidia-smi.txt and nvidia-smi-q.txt from the nvidia-smi directory
        and prints their concatenated contents.
        """
        nvidia_smi_dir = os.path.join(self._fixed_session.info.path, "nvidia-smi")
        nvidia_smi_path = os.path.join(nvidia_smi_dir, "nvidia-smi.txt")
        nvidia_smi_q_path = os.path.join(nvidia_smi_dir, "nvidia-smi-q.txt")

        output = ""

        if os.path.exists(nvidia_smi_path):
            with open(nvidia_smi_path) as f:
                output += f.read()
                output += "\n" + "=" * 80 + "\n\n"
        else:
            output += "nvidia-smi.txt not found\n\n"

        if os.path.exists(nvidia_smi_q_path):
            with open(nvidia_smi_q_path) as f:
                output += f.read()
        else:
            output += "nvidia-smi-q.txt not found\n"

        print(output)


class FixedSession:
    """Class to manage a fixed simulation session."""

    def __init__(self, session: "Session"):
        """Initialize the Session class.

        Args:
            session (Session): The session object.
        """
        self._session = session
        self._update_preview_interval = 0.1
        self._update_terminal_interval = 0.1
        self._update_table_interval = 0.1
        self._info = SessionInfo(session.name).set_path(
            os.path.join(session.app_root, session.name)
        )
        self._export = SessionExport(self)
        self._get = SessionGet(self)
        self._output = SessionOutput(self)
        self._param = session.param.copy()
        self._default_opts: dict[str, Any] = {
            "flat_shading": False,
            "wireframe": False,
            "pin": False,
            "stitch": False,
        }
        if self.fixed_scene is not None:
            self.delete()
            self.fixed_scene.export_fixed(self.info.path, True)
        else:
            raise ValueError("Scene and param must be initialized")
        self._cmd_path = self.export.shell_command(self._param)

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

    @property
    def session(self) -> "Session":
        """Get the session object."""
        return self._session

    def print(self, message):
        """Print a message.

        Args:
            message (str): The message to print.
        """
        if Utils.in_jupyter_notebook():
            from IPython.display import display

            display(message)
        else:
            print(message)

    def _analyze_solver_error(self, log_lines, err_lines):
        """Analyze log and error files for specific failure patterns.

        Args:
            log_lines (list): Lines from stdout log file
            err_lines (list): Lines from stderr log file

        Returns:
            str or None: Single most critical error message, or None if no specific error found
        """
        all_lines = log_lines + err_lines

        error_patterns = [
            (
                "cuda: no device found",
                "No CUDA device found",
            ),
            (
                "### ccd failed",
                "Continuous Collision Detection failed",
            ),
            (
                "### cg failed",
                "Linear solver failed",
            ),
            (
                "### intersection detected",
                "Intersection detected",
            ),
            (
                "Error: reduce buffer size is too small",
                "Insufficient GPU memory",
            ),
            (
                "stack overflow",
                "BVH traversal stack overflow",
            ),
            (
                "Overflow detected",
                "Numerical overflow",
            ),
            ("assert", "Internal assertion failed"),
        ]

        for line in all_lines:
            line_lower = line.lower().strip()
            for pattern, message in error_patterns:
                if pattern.lower() in line_lower:
                    return message

        return None

    def delete(self):
        """Delete the session."""
        if os.path.exists(self.info.path):
            shutil.rmtree(self.info.path)

    def _check_ready(self):
        """Check if the session is ready."""
        if self.fixed_scene is None:
            raise ValueError("Scene must be initialized")

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

    def initialize_finished(self) -> bool:
        """Check if the session initialization is finished.

        Returns:
            bool: True if the session initialization is finished, False otherwise.
        """
        initialize_finish_path = os.path.join(self.output.path, "initialize_finish.txt")
        error = self.get.log.stderr()
        if len(error) > 0:
            for line in error:
                print(line)
        return os.path.exists(initialize_finish_path)

    def resume(
        self,
        frame: int = -1,
        force: bool = True,
        blocking: bool | None = None,
    ) -> "FixedSession":
        if self._param is None:
            print("Session is not yet started")
            return self
        if frame == -1:
            saved = self.get.saved()
            if len(saved) > 0:
                frame = max(saved)
            else:
                return self
        if frame > 0:
            return self.start(force, blocking, frame)
        else:
            print(f"No saved state found: frame: {frame}")
            return self

    def start(
        self,
        force: bool = False,
        blocking: bool | None = None,
        load: int = 0,
    ) -> "FixedSession":
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
            load (int, optional): The frame number to load from saved states. Defaults to 0.

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

        nvidia_smi_dir = os.path.join(self.info.path, "nvidia-smi")
        os.makedirs(nvidia_smi_dir, exist_ok=True)

        nvidia_smi_path = os.path.join(nvidia_smi_dir, "nvidia-smi.txt")
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                with open(nvidia_smi_path, "w") as f:
                    f.write(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: Could not export nvidia-smi output: {e}")

        nvidia_smi_q_path = os.path.join(nvidia_smi_dir, "nvidia-smi-q.txt")
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                with open(nvidia_smi_q_path, "w") as f:
                    f.write(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: Could not export nvidia-smi -q output: {e}")

        if os.path.exists(self.save_and_quit_file_path()):
            os.remove(self.save_and_quit_file_path())
        self._check_ready()
        if Utils.busy():
            if force:
                Utils.terminate()
            else:
                from IPython.display import display

                self.print("Solver is already running. Teriminate first.")
                display(self._terminate_button("Terminate Now"))
                return self

        frame = self.get.saved()
        if frame and not force:
            from IPython.display import display

            self.print(f"Solver has saved states. Resuming from {max(frame)}")
            return self.resume(max(frame), True, blocking)

        if self._cmd_path:
            if load == 0:
                export_path = os.path.join(
                    "export",
                    self._session.app_name,
                    self.info.name,
                )
                if os.path.exists(export_path):
                    shutil.rmtree(export_path)

            err_path = os.path.join(self.info.path, "error.log")
            log_path = os.path.join(self.info.path, "stdout.log")
            if os.name == "nt":  # Windows
                command = f'"{self._cmd_path}" --load {load}'
            else:
                command = f"bash {self._cmd_path} --load {load}"
            with open(log_path, "w") as stdout_file, open(err_path, "w") as stderr_file:
                if os.name == "nt":  # Windows
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                        cwd=self._session.proj_root,
                    )
                else:
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        start_new_session=True,
                        cwd=self._session.proj_root,
                    )
            if blocking is None:
                blocking = not Utils.in_jupyter_notebook()
            if not blocking:
                # Wait for process to start (needed for Utils.busy() to detect it)
                for _ in range(10):  # Wait up to 1 second
                    if Utils.busy():
                        break
                    time.sleep(0.1)
            if blocking:
                while not os.path.exists(log_path) and not os.path.exists(err_path):
                    time.sleep(1)
                if process.poll() is not None:
                    if os.path.exists(log_path):
                        with open(log_path) as f:
                            log_lines = f.readlines()
                    else:
                        log_lines = []
                    if os.path.exists(err_path):
                        with open(err_path) as f:
                            err_lines = f.readlines()
                    else:
                        err_lines = []
                    display_log(log_lines)
                    display_log(err_lines)

                    error_message = self._analyze_solver_error(log_lines, err_lines)
                    if error_message:
                        raise ValueError(error_message)
                    else:
                        raise ValueError("Solver failed to start")
                else:
                    time.sleep(1)
                    while Utils.busy():
                        if self.initialize_finished():
                            break
                        time.sleep(1)
                    if not self.initialize_finished():
                        if os.path.exists(log_path):
                            with open(log_path) as f:
                                log_lines = f.readlines()
                        else:
                            log_lines = []
                        if os.path.exists(err_path):
                            with open(err_path) as f:
                                err_lines = f.readlines()
                        else:
                            err_lines = []
                        display_log(log_lines)
                        display_log(err_lines)

                        error_message = self._analyze_solver_error(log_lines, err_lines)
                        if error_message:
                            raise ValueError(error_message)
                        else:
                            raise ValueError(
                                "Solver initialization failed - check log files for details"
                            )
                print(f">>> Log path: {log_path}")
                print(">>> Waiting for solver to finish...")
                total_frames = self._param.get("frames")
                assert isinstance(total_frames, int)
                with tqdm(total=total_frames, desc="Progress") as pbar:
                    last_frame = 0
                    while process.poll() is None:
                        frame = self.get.latest_frame()
                        if frame > last_frame:
                            pbar.update(frame - last_frame)
                            last_frame = frame
                        time.sleep(1)
                if os.path.exists(err_path):
                    with open(err_path) as f:
                        err_lines = f.readlines()
                else:
                    err_lines = []
                if len(err_lines) > 0:
                    print("*** Solver FAILED ***")
                else:
                    print("*** Solver finished ***")
                n_logs = 32
                with open(log_path) as f:
                    log_lines = f.readlines()
                print(">>> Log:")
                for line in log_lines[-n_logs:]:
                    print(line.rstrip())
                if len(err_lines) > 0:
                    print(">>> Error:")
                    for line in err_lines:
                        print(line.rstrip())
                    print(f">>> Error log path: {err_path}")

            fixed_scene = self.fixed_scene
            max_strain_limit = 0.0
            if fixed_scene is not None:
                vals = [
                    x
                    for x in fixed_scene.tri_param.get("strain-limit", [])
                    if isinstance(x, float)
                ]
                if vals:
                    max_strain_limit = max(vals)
            self._default_opts["max-area"] = 1.0 + max_strain_limit
        else:
            raise ValueError("Command path is not set. Call build() first.")
        return self

    def _terminate_button(self, description: str = "Terminate Solver"):
        """Create a terminate button.

        Args:
            description (str, optional): The button description.

        Returns:
            Optional[widgets.Button]: The terminate button.
        """
        if Utils.in_jupyter_notebook():
            import ipywidgets as widgets

            def _terminate(button):
                button.disabled = True
                button.description = "Terminating..."
                Utils.terminate()
                while Utils.busy():
                    time.sleep(0.25)
                button.description = "Terminated"

            button = widgets.Button(description=description)
            button.on_click(_terminate)
            return button
        else:
            return None

    def save_and_quit_file_path(self) -> str:
        """Get the flagging file path for saving and quitting the session.
        If this file exists, the solver will save the session and quit.
        After the session is saved, the file will be removed."""
        return os.path.join(self.info.path, "output", "save_and_quit")

    def save_and_quit(self):
        """Save the session and quit the solver."""
        open(
            self.save_and_quit_file_path(),
            "w",
        ).close()

    def _save_and_quit_button(self, description: str = "Save and Quit"):
        """Create a save-and-quit button.

        Args:
            description (str, optional): The button description.

        Returns:
            Optional[widgets.Button]: The save-and-quit button.
        """
        if Utils.in_jupyter_notebook():
            import ipywidgets as widgets

            def _save_and_quit(button):
                button.disabled = True
                button.description = "Requesting..."
                self.save_and_quit()
                while Utils.busy():
                    time.sleep(0.25)
                button.description = "Done"

            button = widgets.Button(description=description)
            button.on_click(_save_and_quit)
            return button
        else:
            return None

    def update_options(self, options: dict) -> dict:
        options = dict(options)
        for key, value in self._default_opts.items():
            if key not in options:
                options[key] = value
        return options

    def preview(
        self,
        options: Optional[dict] = None,
        live_update: bool = True,
        engine: str = "threejs",
    ) -> Optional["Plot"]:
        """Live view the session.

        Args:
            options (dict, optional): The render options.
            live_update (bool, optional): Whether to enable live update.
            engine (str, optional): The rendering engine. Defaults to "threejs".

        Returns:
            Optional[Plot]: The plot object.
        """
        if options is None:
            options = {}
        options = self.update_options(options)
        if Utils.in_jupyter_notebook():
            import ipywidgets as widgets

            from IPython.display import display

            fixed_scene = self.fixed_scene
            if fixed_scene is None:
                raise ValueError("Scene must be initialized")
            else:
                result = self.get.vertex()
                if result is None:
                    vert, curr_frame = fixed_scene.vertex(True), 0
                else:
                    vert, curr_frame = result
                plot = fixed_scene.preview(
                    vert, options, show_slider=False, engine=engine
                )

            table = widgets.HTML()
            terminate_button = self._terminate_button()
            save_and_quit_button = self._save_and_quit_button()

            if live_update and Utils.busy():

                def update_dataframe(table, curr_frame):
                    summary = self.get.log.summary()
                    max_stretch = summary.get("stretch")
                    data = {
                        "Frame": [curr_frame],
                        "Time/Frame": [summary.get("time-per-frame")],
                        "Time/Step": [summary.get("time-per-step")],
                        "#Contact": [summary.get("num-contact")],
                        "#Newton": [summary.get("newton-steps")],
                        "#PCG": [summary.get("pcg-iter")],
                    }
                    if max_stretch is not None:
                        data["Max Stretch"] = [max_stretch]
                    df = pd.DataFrame(data)
                    table.value = df.to_html(
                        classes="table table-striped", border=0, index=False
                    )

                async def live_preview_async():
                    """Async coroutine for live preview updates.

                    Using async instead of threading allows the event loop to process
                    button events between updates, preventing UI unresponsiveness.
                    """
                    nonlocal plot
                    nonlocal terminate_button
                    nonlocal save_and_quit_button
                    nonlocal table
                    nonlocal options
                    nonlocal curr_frame
                    try:
                        assert plot is not None
                        while True:
                            last_frame = self.get.latest_frame()
                            if curr_frame != last_frame:
                                curr_frame = last_frame
                                result = self.get.vertex(curr_frame)
                                if result is not None:
                                    vert, _ = result
                                    color = self.fixed_scene.color(vert, options)
                                    update_dataframe(table, curr_frame)
                                    plot.update(vert, color)
                            if not Utils.busy():
                                break
                            await asyncio.sleep(self._update_preview_interval)
                        assert terminate_button is not None
                        assert save_and_quit_button is not None
                        terminate_button.disabled = True
                        terminate_button.description = "Terminated"
                        save_and_quit_button.disabled = True
                        await asyncio.sleep(self._update_preview_interval)
                        last_frame = self.get.latest_frame()
                        update_dataframe(table, last_frame)
                        vertex_data = self.get.vertex(last_frame)
                        if vertex_data is not None:
                            vert, _ = vertex_data
                            color = self.fixed_scene.color(vert, options)
                            plot.update(vert, color)
                    except Exception as e:
                        print(f"live_preview error: {e}")

                async def live_table_async():
                    """Async coroutine for table updates."""
                    nonlocal table
                    try:
                        while True:
                            update_dataframe(table, curr_frame)
                            if not Utils.busy():
                                break
                            await asyncio.sleep(self._update_table_interval)
                    except Exception as e:
                        print(f"live_table error: {e}")

                # Use async coroutines instead of threads to allow event loop
                # to process button events between updates
                asyncio.ensure_future(live_preview_async())
                asyncio.ensure_future(live_table_async())
                display(widgets.HBox((terminate_button, save_and_quit_button)))

            display(table)
            return plot
        else:
            return None

    def animate(
        self, options: Optional[dict] = None, engine: str = "threejs"
    ) -> "FixedSession":
        """Show the animation.

        Args:
            options (dict, optional): The render options.

        Returns:
            Session: The animated session.
        """
        if options is None:
            options = {}
        options = self.update_options(options)

        if Utils.in_jupyter_notebook():
            import ipywidgets as widgets

            from IPython.display import display

            fixed_scene = self.fixed_scene
            if fixed_scene is None:
                raise ValueError("Scene must be initialized")
            else:
                plot = fixed_scene.preview(
                    fixed_scene.vertex(True),
                    options,
                    show_slider=False,
                    engine=engine,
                )
                try:
                    if fixed_scene is not None:
                        # Wait for at least one frame to be ready
                        frame_count = self.get.vertex_frame_count()
                        if frame_count == 0:
                            print(
                                "Waiting for simulation to generate at least one frame..."
                            )
                            while self.get.vertex_frame_count() == 0:
                                if not Utils.busy():
                                    print(
                                        "Simulation finished but no frames were generated."
                                    )
                                    return self
                                time.sleep(0.5)
                            frame_count = self.get.vertex_frame_count()
                            print(f"Found {frame_count} frame(s). Loading animation...")

                        vert_list = []
                        for i in tqdm(
                            range(frame_count), desc="Loading frames", ncols=70
                        ):
                            result = self.get.vertex(i)
                            if result is not None:
                                vert, _ = result
                                vert_list.append(vert)

                        # Create status label and reload button
                        status_label = widgets.Label(
                            value=f"Loaded {len(vert_list)} frames"
                        )
                        reload_button = widgets.Button(description="Reload")
                        display(widgets.HBox([reload_button, status_label]))

                        def update(frame=1):
                            nonlocal vert_list
                            nonlocal plot
                            assert plot is not None
                            if fixed_scene is not None and frame - 1 < len(vert_list):
                                vert = vert_list[frame - 1]
                                color = fixed_scene.color(vert, options)
                                # Always recompute normals for correct lighting
                                plot.update(vert, color, recompute_normals=True)

                        # Create the interactive slider
                        slider = widgets.IntSlider(
                            min=1, max=frame_count, step=1, value=1, description="frame"
                        )
                        output = widgets.interactive_output(update, {"frame": slider})

                        def _reload(button):
                            nonlocal vert_list
                            nonlocal slider
                            nonlocal status_label
                            button.disabled = True
                            button.description = "Reloading..."
                            try:
                                # Reload frames from disk
                                new_frame_count = self.get.vertex_frame_count()
                                if new_frame_count > len(vert_list):
                                    for i in range(len(vert_list), new_frame_count):
                                        result = self.get.vertex(i)
                                        if result is not None:
                                            vert, _ = result
                                            vert_list.append(vert)

                                    # Update the slider range
                                    slider.max = new_frame_count

                                    # Update status label
                                    status_label.value = (
                                        f"Loaded {len(vert_list)} frames"
                                    )
                                button.description = "Reload"
                            except Exception as e:
                                button.description = "Reload"
                            finally:
                                button.disabled = False

                        reload_button.on_click(_reload)

                        # Display slider and output
                        display(slider, output)
                except Exception as _:
                    pass
        return self

    def stream(self, n_lines=40) -> "FixedSession":
        """Stream the session logs.

        Args:
            n_lines (int, optional): The number of lines to stream. Defaults to 40.

        Returns:
            Session: The session object.
        """
        if Utils.in_jupyter_notebook():
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
                        # Read last n_lines from log file (cross-platform)
                        if os.path.exists(log_path):
                            with open(log_path) as f:
                                lines = f.readlines()
                                tail_lines = (
                                    lines[-n_lines:] if len(lines) > n_lines else lines
                                )
                                tail_output = "".join(tail_lines).strip()
                        else:
                            tail_output = ""
                        log_widget.value = (
                            CONSOLE_STYLE
                            + f"<pre style='no-scroll'>{tail_output}</pre>"
                        )
                        if not Utils.busy():
                            log_widget.value += "<p style='color: red;'>Terminated.</p>"
                            if os.path.exists(err_path):
                                with open(err_path) as file:
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

    @property
    def fixed_scene(self) -> FixedScene | None:
        """Get the fixed scene."""
        return self._session.fixed_scene


class Session:
    """Class to setup a simulation session."""

    def __init__(
        self,
        app_name: str,
        app_root: str,
        proj_root: str,
        data_dirpath: str,
        name: str,
        autogenerated: Optional[int] = None,
    ):
        """Initialize the Session class.

        Args:
            app_name (str): The name of the application.
            app_root (str): The root directory of the application.
            proj_root (str): The root directory of the project.
            data_dirpath (str): The data directory path.
            name (str): The name of the session.
            autogenerated (Optional[int]): Counter value if autogenerated, None otherwise.
        """
        self._app_name = app_name
        self._name = name
        self._app_root = app_root
        self._proj_root = proj_root
        self._data_dirpath = data_dirpath
        self._autogenerated = autogenerated
        self._fixed_scene = None
        self._fixed_session = None
        self._param = ParamManager()

    @property
    def param(self) -> ParamManager:
        """Get the session parameter manager."""
        return self._param

    @property
    def fixed_scene(self) -> FixedScene | None:
        """Get the fixed scene.

        Returns:
            Optional[FixedScene]: The fixed scene object.
        """
        return self._fixed_scene

    @property
    def fixed_session(self) -> FixedSession | None:
        """Get the fixed session.val

        Returns:
            Optional[FixedSession]: The fixed session object.
        """
        return self._fixed_session

    @property
    def proj_root(self) -> str:
        """Get the project root directory."""
        return self._proj_root

    @property
    def app_name(self) -> str:
        """Get the application name."""
        return self._app_name

    @property
    def name(self) -> str:
        """Get the session name."""
        return self._name

    @property
    def app_root(self) -> str:
        """Get the application root directory."""
        return self._app_root

    def _check_ready(self):
        """Check if the session is ready."""
        if self._fixed_scene is None:
            raise ValueError("Scene must be initialized")

    def init(self, scene: FixedScene) -> "Session":
        """Initialize the session with a fixed scene.

        Args:
            scene (FixedScene): The fixed scene.

        Returns:
            Session: The initialized session.
        """
        self._fixed_scene = scene
        return self

    def build(self) -> FixedSession:
        self._fixed_session = FixedSession(self)
        # Use app name with counter suffix if autogenerated
        if self._autogenerated is not None:
            if self._autogenerated == 0:
                symlink_name = self._app_name
            else:
                symlink_name = f"{self._app_name}-{self._autogenerated}"
        else:
            symlink_name = self._name
        self._save_fixed_session(self._fixed_session, symlink_name)
        return self._fixed_session

    def _save_fixed_session(
        self, fixed_session: FixedSession, name: Optional[str] = None
    ):
        """Saves the fixed session to a recoverable file and creates symlink."""
        session_path = os.path.join(
            fixed_session.info.path, RECOVERABLE_FIXED_SESSION_NAME
        )
        with open(session_path, "wb") as f:
            pickle.dump(fixed_session, f)

        if name:
            symlink_dir = os.path.join(self._data_dirpath, "symlinks")
            os.makedirs(symlink_dir, exist_ok=True)
            symlink_path = os.path.join(symlink_dir, name)

            if os.path.islink(symlink_path):
                os.unlink(symlink_path)
            elif os.path.exists(symlink_path):
                os.remove(symlink_path)

            try:
                os.symlink(fixed_session.info.path, symlink_path)
            except OSError:
                # On Windows, symlinks may require elevated privileges
                # Fall back to writing a text file with the path
                with open(symlink_path + ".txt", "w") as f:
                    f.write(fixed_session.info.path)


def display_log(lines: list[str]):
    """Display the log lines.

    Args:
        lines (list[str]): The log lines.
    """
    lines = [line.rstrip("\n") for line in lines]
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


def convert_time(time) -> str:
    if time is None:
        return "N/A"
    elif time < 1_000:
        return f"{int(time)}ms"
    elif time < 60_000:
        return f"{time / 1_000:.2f}s"
    else:
        return f"{time / 60_000:.2f}m"


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
