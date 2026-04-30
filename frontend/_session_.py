# File: _session_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import asyncio
import copy
import os
import pickle
import platform
import shutil
import subprocess
import threading
import time

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from tqdm.auto import tqdm

from ._param_ import ParamHolder, app_param
from ._parse_ import CppRustDocStringParser
from ._scene_ import FixedScene
from ._utils_ import Utils, get_export_base_path

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
    """Class to manage simulation parameters.

    Example:
        Configure the standard simulation parameters for a session before
        building it::

            session = app.session.create(scene)
            (
                session.param.set("frames", 120)
                             .set("dt", 0.01)
                             .set("fps", 30)
            )
            session = session.build()
    """

    def __init__(self):
        """Initialize the ParamManager."""
        self._key = None
        self._param = ParamHolder(app_param())
        self._default_param = self._param.copy()
        self._time = 0.0
        self._dyn_param = {}

    def copy(self) -> "ParamManager":
        """Copy the ParamManager object.

        Returns:
            ParamManager: The copied ParamManager object.

        Example:
            Snapshot the current parameters before tweaking one of them::

                baseline = session.param.copy()
                session.param.set("dt", 0.005)
        """
        return copy.deepcopy(self)

    def set(self, key: str, value: Optional[Any] = None) -> "ParamManager":
        """Set a parameter value.

        If ``value`` is ``None``, the parameter is set to ``True``.

        Args:
            key (str): The parameter key. Must not contain an underscore;
                use ``-`` instead.
            value (Any, optional): The parameter value. Defaults to ``None``
                (interpreted as ``True``).

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``key`` contains an underscore or does not exist.

        Example:
            Chain several ``.set`` calls to configure a session. Keys use
            hyphens, never underscores::

                (
                    session.param.set("frames", 250)
                                 .set("dt", 0.01)
                                 .set("fps", 30)
                                 .set("min-newton-steps", 32)
                                 .set("gravity", [0, 0, 0])
                )
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
        """Clear all parameters to their default values.

        Example:
            Reset every parameter back to its default before configuring a
            new run::

                session.param.clear_all()
                session.param.set("frames", 60).set("dt", 0.01)
        """
        self._param = self._default_param.copy()
        self._dyn_param = {}

    def clear(self, key: str) -> "ParamManager":
        """Reset a parameter to its default value and drop any dynamic entries for it.

        Args:
            key (str): The parameter key.

        Returns:
            ParamManager: The updated ParamManager object.

        Example:
            Revert a single parameter back to its default after trying an
            override::

                session.param.set("dt", 0.001)
                session.param.clear("dt")
        """
        self._param.set(key, self._default_param.get(key))
        if key in self._dyn_param:
            del self._dyn_param[key]
        return self

    def dyn(self, key: str) -> "ParamManager":
        """Select the current dynamic parameter key and reset the internal time cursor.

        Args:
            key (str): The dynamic parameter key.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``key`` does not exist.

        Example:
            Flip gravity between t=1s and t=2s, then restore it::

                g = session.param.get("gravity")
                (session.param.dyn("gravity")
                              .time(1.0).hold()
                              .time(1.5).change([-x for x in g])
                              .time(2.0).change(g))
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
            value (Any): The new value of the dynamic parameter. May be a
                scalar, bool, or list/tuple of floats, depending on the key.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If no dynamic key is currently selected.

        Example:
            Slow playback to 10% after the third second via a dynamic key::

                (
                    session.param.dyn("playback")
                                 .time(2.99).hold()
                                 .time(3.0).change(0.1)
                )
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
        """Hold the current value of the dynamic parameter at the current time.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If no dynamic key is currently selected.

        Example:
            Keep playback steady until t=2.99s, then drop it at t=3.0s::

                (
                    session.param.dyn("playback")
                                 .time(2.99).hold()
                                 .time(3.0).change(0.1)
                )
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
        """Export the parameters to ``param.toml`` (and ``dyn_param.txt`` when present).

        In fast-check mode, ``frames`` is forced to ``1``.

        Args:
            path (str): The path to the export directory.

        Example:
            Write the parameters alongside a session directory for inspection
            or external launching::

                session.param.export(fixed_session.info.path)
        """
        # Force frames=1 in fast check mode
        if Utils.is_fast_check():
            self._param.set("frames", 1)

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
                        elif hasattr(val, "__len__") and not isinstance(val, str):
                            items = ", ".join(str(float(x)) for x in val)
                            f.write(f"{key} = [{items}]\n")
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
                        if isinstance(val, (list, tuple)):
                            items = " ".join(str(float(x)) for x in val)
                            f.write(f"{time} {items}\n")
                        elif isinstance(val, float):
                            f.write(f"{time} {val}\n")
                        elif isinstance(val, bool):
                            f.write(f"{time} {float(val)}\n")
                        else:
                            raise ValueError(
                                f"Value must be float, bool, or list. {val} is given."
                            )

    def time(self, time: float) -> "ParamManager":
        """Advance the current time cursor for the dynamic parameter.

        Args:
            time (float): The new current time. Must be strictly greater than
                the previous value.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``time`` is not strictly increasing.

        Example:
            Advance the cursor between two dynamic-value updates::

                (
                    session.param.dyn("playback")
                                 .time(1.0).hold()
                                 .time(2.0).change(0.5)
                )
        """
        if time <= self._time:
            raise ValueError("Time must be increasing")
        else:
            self._time = time
        return self

    def get(self, key: Optional[str] = None) -> Any:
        """Get the value of a parameter.

        Args:
            key (Optional[str]): The parameter key. Must be specified.

        Returns:
            Any: The value of the parameter.

        Raises:
            ValueError: If ``key`` is ``None``.

        Example:
            Read the current gravity vector so a dynamic override can flip
            its sign later::

                g = session.param.get("gravity")
                print(g)
        """
        if key is None:
            raise ValueError("Key must be specified")
        else:
            return self._param.get(key)

    def items(self):
        """Get all parameter items.

        Returns:
            ItemsView: The parameter items.

        Example:
            Inspect every parameter currently configured on the session::

                for key, value in session.param.items():
                    print(f"{key} = {value}")
        """
        return self._param.items()


class SessionManager:
    """Class to manage simulation sessions.

    Example:
        Create a session from a built scene and launch it::

            session = app.session.create(scene)
            session.param.set("frames", 60).set("dt", 0.01)
            session = session.build()
            session.start(blocking=True)
    """

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

        Example:
            Print the names of every session currently tracked by the app::

                for name in app.session.list():
                    print(name)
        """
        return self._sessions

    def select(self, name: str = "session"):
        """Select an existing session by name.

        Args:
            name (str): The name of the session. Defaults to ``"session"``.

        Returns:
            Session: The selected session.

        Raises:
            ValueError: If no session with the given name exists.

        Example:
            Re-fetch a previously-created session by name::

                app.session.create(scene, name="run-A")
                session = app.session.select("run-A")
        """
        if name not in self._sessions:
            raise ValueError(f"Session {name} does not exist")
        return self._sessions[name]

    def create(self, scene: FixedScene, name: str = "") -> "Session":
        """Create a new session.

        If ``name`` is empty, an auto-generated name is used: ``"session"``
        for the first call, then ``"session-1"``, ``"session-2"``, ... for
        subsequent calls.

        Args:
            scene (FixedScene): The scene object.
            name (str): The name of the session. Defaults to ``""``
                (auto-generated).

        Returns:
            Session: The created session.

        Raises:
            Exception: If the scene has violations (self-intersections,
                contact-offset violations, etc.).

        Example:
            Create a session from a built fixed scene and configure it::

                scene = app.scene.create()
                scene.add("sheet").at(0, 0.5, 0)
                scene = scene.build()

                session = app.session.create(scene)
                session.param.set("frames", 60).set("dt", 0.01)
                session = session.build()
        """
        assert isinstance(scene, FixedScene), "Scene must be a FixedScene object"
        if scene.has_violations:
            messages = scene.get_violation_messages()
            raise Exception(f"Cannot create session: {'; '.join(messages)}. ")
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
        """Terminate the solver if it is running, or raise if ``force`` is ``False``.

        Args:
            force (bool): If ``True``, terminate a running solver; otherwise
                raise a ``ValueError``.

        Raises:
            ValueError: If the solver is running and ``force`` is ``False``.
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

        Example:
            Tear down a named session, terminating the solver if it is
            still running::

                app.session.delete("run-A", force=True)
        """
        self._terminate_or_raise(force)
        if name in self._sessions:
            self._sessions[name].delete()
            del self._sessions[name]

    def clear(self, force: bool = True):
        """Clear all sessions.

        Args:
            force (bool, optional): Whether to force clearing.

        Example:
            Remove every session and any running solver before starting
            fresh::

                app.session.clear(force=True)
        """
        self._terminate_or_raise(force)
        for session in self._sessions.values():
            session.delete()
        self._sessions = {}


class SessionInfo:
    """Class to store session information.

    Example:
        Read the on-disk directory path for a built session::

            fixed_session = session.build()
            print(fixed_session.info.name)
            print(fixed_session.info.path)
    """

    def __init__(self, name: str):
        """Initialize the SessionInfo class.

        The session directory path is initialized empty and should be set via
        :meth:`set_path`.

        Args:
            name (str): The name of the session.
        """
        self._name = name
        self._path = ""

    def set_path(self, path: str) -> "SessionInfo":
        """Set the path to the session directory.

        Args:
            path (str): The path to the session directory.

        Returns:
            SessionInfo: This instance, for chaining.

        Example:
            Normally this is called by :class:`SessionManager` during session
            construction. A direct call may be useful when relocating an
            existing session on disk::

                info = SessionInfo("my_run")
                info.set_path("/data/sessions/my_run")
                print(info.path)
        """
        self._path = path
        return self

    @property
    def name(self) -> str:
        """Get the name of the session.

        Example:
            Inspect the session name before starting a run::

                session = app.session.create(fixed_scene).build()
                print(session.info.name)
        """
        return self._name

    @property
    def path(self) -> str:
        """Get the path to the session directory.

        Example:
            Read the on-disk session directory after building::

                session = app.session.create(fixed_scene).build()
                print(session.info.path)
        """
        return self._path


class Zippable:
    def __init__(self, dirpath: str):
        self._dirpath = dirpath

    def zip(self):
        """Zip the directory.

        No-op when running in a CI environment (as detected by
        :meth:`Utils.ci_name`).
        """
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
    """Class to handle session export operations.

    Example:
        Export every simulated frame and zip the output directory::

            session.start(blocking=True)
            session.export.animation().zip()
    """

    def __init__(self, fixed_session: "FixedSession"):
        """Initialize the SessionExport class.

        Args:
            fixed_session (FixedSession): The fixed session object.
        """
        self._fixed_session = fixed_session
        self._session = fixed_session.session

    def shell_command(
        self,
        param: ParamManager,
    ) -> str:
        """Generate a platform-specific launcher script for the solver.

        On Windows, writes a ``command.bat`` file; on Linux/macOS, writes a
        ``command.sh`` script (marked executable).

        Args:
            param (ParamManager): The simulation parameters.

        Returns:
            str: The path to the generated launcher script.

        Example:
            Regenerate the solver launcher alongside a session (useful for
            re-running from the command line)::

                path = session.export.shell_command(session.session.param)
                print(path)
        """
        param.export(self._fixed_session.info.path)

        # Platform-specific solver path and script generation
        if platform.system() == "Windows":  # Windows
            program_path = os.path.join(
                self._session.proj_root, "target", "release", "ppf-contact-solver.exe"
            )
            lib_path = os.path.join(
                self._session.proj_root, "src", "cpp", "build", "lib"
            )
            # Generate batch file with PATH set for DLLs
            command = f"""@echo off
set SOLVER_PATH={program_path}
set LIB_PATH={lib_path}

REM CUDA_PATH should be set by start.bat or the environment
set PATH=%LIB_PATH%;%CUDA_PATH%\\bin;%PATH%

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
            if platform.system() != "Windows":  # chmod not needed on Windows
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

        If no frames are available yet, waits for the simulation if it is
        running, otherwise returns early. When ``ffmpeg`` is available and
        rendered PNGs are produced, also encodes an ``frame.mp4`` video in
        the export directory.

        Args:
            path (str): The path to the export directory. If empty, a default
                path is used.
            ext (str, optional): The file extension. Defaults to ``"ply"``.
            include_static (bool, optional): Whether to include the static
                mesh. Defaults to ``True``.
            clear (bool, optional): Whether to clear the existing files.
                Defaults to ``False``.
            options (dict, optional): Additional arguments passed to the
                renderer.

        Returns:
            Zippable: A handle to the export directory that can be zipped.

        Example:
            Export every frame as PLY and zip the result::

                session.start(blocking=True)
                session.export.animation().zip()
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
                    get_export_base_path(),
                    self._fixed_session.session.app_name,
                    self._fixed_session.info.name,
                )

        # Check if frames are available
        latest_frame = self._fixed_session.get.latest_frame()
        if latest_frame == 0:
            if self._fixed_session.is_running():
                print(
                    "No frames available yet. Waiting for simulation to generate frames..."
                )
                # Wait for frames to become available
                while self._fixed_session.is_running() and self._fixed_session.get.latest_frame() == 0:
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

        for i in tqdm(range(latest_frame), desc="export"):
            self.frame(
                os.path.join(path, f"frame_{i}.{ext}"),
                i,
                include_static,
                options,
                delete_exist=clear,
            )

        # Check if any PNG images were rendered before attempting video creation
        png_files = [f for f in os.listdir(path) if f.endswith(".png")]
        # Look for ffmpeg in multiple locations
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ffmpeg_candidates = [
            os.path.join(project_root, "bin", "ffmpeg"),  # Linux/Docker
            os.path.join(project_root, "bin", "ffmpeg.exe"),  # Windows bundled (dist/)
            os.path.join(
                project_root, "build-win-native", "ffmpeg", "ffmpeg.exe"
            ),  # Windows dev
        ]
        ffmpeg_path = next((p for p in ffmpeg_candidates if os.path.isfile(p)), None)
        if ffmpeg_path is None:
            ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None and png_files:
            vid_name = "frame.mp4"
            command = f'"{ffmpeg_path}" -hide_banner -loglevel error -y -r 60 -i frame_%d.{ext}.png -pix_fmt yuv420p -c:v libx264 {vid_name}'
            subprocess.run(command, shell=True, cwd=path)
            if Utils.in_jupyter_notebook():
                from IPython.display import Video, display

                display(Video(os.path.join(path, vid_name), embed=True))

            if ci_name is not None:
                for file in png_files:
                    os.remove(os.path.join(path, file))

        return Zippable(path)

    def frame(
        self,
        path: str = "",
        frame: Optional[int] = None,
        include_static: bool = True,
        options: Optional[dict] = None,
        delete_exist: bool = False,
    ) -> "FixedSession":
        """Export a specific frame.

        Args:
            path (str): The path to the export file.
            frame (Optional[int], optional): The frame number. If ``None``,
                the latest available frame is used. Defaults to ``None``.
            include_static (bool, optional): Whether to include the static
                mesh. Defaults to ``True``.
            options (dict, optional): Additional arguments passed to the
                renderer.
            delete_exist (bool, optional): Whether to delete the existing
                file. Defaults to ``False``.

        Returns:
            FixedSession: The owning fixed session object.

        Example:
            Export just the latest frame as an OBJ file::

                session.export.frame("latest.obj")
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
    """Class to handle session output operations.

    Example:
        Locate the solver output directory for a built session (used by
        exporters and log readers)::

            print(session.output.path)
    """

    def __init__(self, session: "FixedSession"):
        """Initialize the SessionOutput class.

        Args:
            session (FixedSession): The fixed session object.
        """
        self._session = session

    @property
    def path(self) -> str:
        """Get the path to the output directory.

        Example:
            Locate the solver output directory for post-processing::

                session = session.build().start(blocking=True)
                output_dir = session.output.path
        """
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

        Example:
            List every log channel the solver can emit, then confirm a
            channel of interest is present::

                names = session.get.log.names()
                assert "time-per-frame" in names
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
            with open(path) as f:
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

        Example:
            Print the last 8 lines of solver stdout for a quick health
            check::

                for line in session.get.log.stdout(n_lines=8):
                    print(line)
        """
        return self._tail_file(
            os.path.join(self._fixed_session.info.path, "stdout.log"), n_lines
        )

    def stderr(self, n_lines: Optional[int] = None) -> list[str]:
        """Get the last n lines of the stderr log file.

        Args:
            n_lines (Optional[int], optional): The number of lines. Defaults to None.

        Returns:
            list[str]: The last n lines of the stderr log file.

        Example:
            Surface any solver errors without loading the full log::

                for line in session.get.log.stderr(n_lines=20):
                    print(line)
        """
        return self._tail_file(
            os.path.join(self._fixed_session.info.path, "error.log"), n_lines
        )

    def numbers(self, name: str):
        """Get the list of (x, y) number pairs recorded for a given log.

        Args:
            name (str): The name of the log.

        Returns:
            Optional[list[list[float | int]]]: The list of number pairs, or
            ``None`` if the log name is unknown or the backing file does not
            exist. Integer-valued entries are returned as ``int``.

        Example:
            Average wall-clock time per video frame::

                pairs = session.get.log.numbers("time-per-frame")
                avg_ms = sum(n for _, n in pairs) / len(pairs)
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
        """Get the latest value from a log.

        Args:
            name (str): The name of the log.

        Returns:
            Optional[float | int]: The latest recorded value, or ``None`` if
            unavailable.

        Example:
            Read the most recent Newton iteration count::

                latest = session.get.log.number("newton-steps")
                print(latest)
        """
        entries = self.numbers(name)
        if entries:
            return entries[-1][1]
        else:
            return None

    def summary(self):
        """Get a summary of the session log using the latest values of key metrics.

        Returns:
            dict: A dictionary mapping metric name to a formatted latest value.
            When ``max-sigma`` is positive, a ``"stretch"`` entry is also
            included as a percentage.

        Example:
            Print a compact summary of the solver's current status::

                print(session.get.log.summary())
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

    def average_summary(self):
        """Get averages for log-backed metrics only.

        Returns:
            dict: A dictionary containing averaged statistics. Metrics without a
            corresponding existing ``.out`` file are omitted.

        Example:
            Print the run-averaged metrics after a simulation has finished::

                print(session.get.log.average_summary())
        """

        def average(name: str):
            entries = self.numbers(name)
            if not entries:
                return None
            values = [entry[1] for entry in entries]
            if not values:
                return None
            return sum(values) / len(values)

        def maximum(name: str):
            entries = self.numbers(name)
            if not entries:
                return None
            values = [entry[1] for entry in entries]
            if not values:
                return None
            return max(values)

        result = {}
        time_per_frame = average("time-per-frame")
        if time_per_frame is not None:
            result["time-per-frame"] = convert_time(time_per_frame)

        time_per_step = average("time-per-step")
        if time_per_step is not None:
            result["time-per-step"] = convert_time(time_per_step)

        n_contact = maximum("num-contact")
        if n_contact is not None:
            result["num-contact (max)"] = convert_integer(round(n_contact))

        n_newton = average("newton-steps")
        if n_newton is not None:
            result["newton-steps"] = f"{n_newton:.2f}"

        n_pcg = average("pcg-iter")
        if n_pcg is not None:
            result["pcg-iter"] = f"{n_pcg:.2f}"

        max_sigma = average("max-sigma")
        if max_sigma is not None and max_sigma > 0.0:
            result["stretch"] = f"{100.0 * (max_sigma - 1.0):.2f}%"

        return result


class SessionGet:
    """Class to handle session data retrieval operations.

    Example:
        Pull the most recent vertex buffer and a log channel from a running
        or completed session::

            vert, frame = session.get.vertex()
            per_frame_ms = session.get.log.numbers("time-per-frame")
    """

    def __init__(self, fixed_session: "FixedSession"):
        """Initialize the SessionGet class.

        Args:
            fixed_session (FixedSession): The fixed session object.
        """
        self._fixed_session = fixed_session
        self._log = SessionLog(fixed_session)

    @property
    def log(self) -> SessionLog:
        """Get the session log object.

        Example:
            Fetch the list of emitted log channels::

                session = session.build().start(blocking=True)
                channels = session.get.log.names()
        """
        return self._log

    def vertex_frame_count(self) -> int:
        """Get the highest frame index that has an exported vertex buffer.

        Returns:
            int: The highest frame index found in ``output/vert_*.bin``, or
            ``0`` if none exist.

        Example:
            Wait until at least one frame is on disk before replaying::

                while session.get.vertex_frame_count() == 0:
                    time.sleep(0.5)
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

        Example:
            Poll the most recent frame index while the solver runs::

                frame = session.get.latest_frame()
                print(f"solver is on frame {frame}")
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

        Example:
            Resume from the newest saved state if any exist::

                saved = session.get.saved()
                if saved:
                    session.resume(max(saved))
        """
        result = []
        output_path = os.path.join(self._fixed_session.info.path, "output")
        if os.path.exists(output_path):
            for file in os.listdir(output_path):
                if file.startswith("state_") and file.endswith(".bin.gz"):
                    try:
                        frame = int(file.split("_")[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue
                    result.append(frame)
        return result

    def vertex(self, n: Optional[int] = None) -> Optional[tuple[np.ndarray, int]]:
        """Get the vertex data for a specific frame.

        Args:
            n (Optional[int], optional): The frame number. If ``None``, the
                latest frame is returned. Defaults to ``None``.

        Returns:
            Optional[tuple[np.ndarray, int]]: A tuple ``(vertices, frame)``
            where ``vertices`` has shape ``(N, 3)`` and dtype ``float32``, or
            ``None`` if no data is available.

        Example:
            Read the latest exported vertex buffer from disk::

                result = session.get.vertex()
                if result is not None:
                    vert, frame = result
                    print(vert.shape, frame)
        """
        path = os.path.join(self._fixed_session.info.path, "output")
        if os.path.exists(path):
            if n is None:
                files = os.listdir(path)
                frames = []
                for file in files:
                    if file.startswith("vert") and file.endswith(".bin"):
                        try:
                            frame = int(file.split("_")[1].split(".")[0])
                        except (IndexError, ValueError):
                            continue
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

    def command(self) -> Optional[str]:
        """Get the path to the solver launcher script.

        On Windows this points at ``command.bat``; elsewhere at
        ``command.sh``.

        Returns:
            Optional[str]: The path to the launcher script if it exists,
            ``None`` otherwise.

        Example:
            Print the launcher path so it can be re-run from a shell::

                path = session.get.command()
                if path:
                    print(path)
        """
        if platform.system() == "Windows":  # Windows
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

        Example:
            Print the parameter summary captured by the solver at launch::

                for line in session.get.param_summary():
                    print(line)
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

        Example:
            Inspect the GPU state captured at the start of a run::

                session.get.nvidia_smi()
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
    """Class to manage a fixed simulation session.

    Returned by :meth:`Session.build`. Use it to launch the solver, monitor
    it, pull results, and export.

    Example:
        Build a session from a configured :class:`Session` and run it to
        completion, then export::

            fixed_session = session.build()
            fixed_session.start(blocking=True)
            fixed_session.export.animation().zip()
    """

    def __init__(self, session: "Session"):
        """Initialize the FixedSession from a parent Session.

        Deletes any prior on-disk session directory, exports the fixed
        scene, and writes the solver launcher script.

        Args:
            session (Session): The parent session object.

        Raises:
            ValueError: If the parent session has no fixed scene.
        """
        self._session = session
        self._process: Optional[subprocess.Popen] = None
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
        """Get the session information.

        Example:
            Read the session name and directory path::

                session = app.session.create(fixed_scene).build()
                print(session.info.name, session.info.path)
        """
        return self._info

    @property
    def export(self) -> SessionExport:
        """Get the session export object.

        Example:
            Export the session shell command for reproducible replays::

                session = app.session.create(fixed_scene).build()
                cmd_path = session.export.shell_command(session.session.param)
        """
        return self._export

    @property
    def get(self) -> SessionGet:
        """Get the session get object.

        Example:
            Retrieve solver-emitted log channel names::

                session = session.build().start(blocking=True)
                channels = session.get.log.names()
        """
        return self._get

    @property
    def output(self) -> SessionOutput:
        """Get the session output object.

        Example:
            Locate the solver output directory::

                session = app.session.create(fixed_scene).build()
                print(session.output.path)
        """
        return self._output

    @property
    def session(self) -> "Session":
        """Get the session object.

        Example:
            Reach back to the parent :class:`Session` for its parameters::

                fixed = app.session.create(fixed_scene).build()
                params = fixed.session.param
        """
        return self._session

    def print(self, message):
        """Print a message.

        Args:
            message (str): The message to print.

        Example:
            Emit a status line that renders nicely in Jupyter or stdout::

                session.print("Launching solver...")
        """
        if Utils.in_jupyter_notebook():
            from IPython.display import display

            display(message)
        else:
            print(message)

    def is_running(self) -> bool:
        """Check if the solver process is running.

        This method first checks the stored process handle (most reliable),
        then falls back to Utils.busy() for broader detection.

        Returns:
            bool: True if the solver is running, False otherwise.

        Example:
            Poll the solver state after launching non-blocking::

                session.start()
                while session.is_running():
                    time.sleep(1)
        """
        # First check the stored process handle (most reliable)
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    return True
                # Process has exited, clear the reference
                self._process = None
            except OSError:
                # Process handle became invalid
                self._process = None

        # Fall back to Utils.busy() for cases where process was started externally
        return Utils.busy()

    def running(self) -> bool:
        """Alias for :meth:`is_running`.

        Example:
            Shorthand form for polling solver state::

                session.start()
                while session.running():
                    time.sleep(1)
        """
        return self.is_running()

    def run(self, blocking: Optional[bool] = None) -> "FixedSession":
        """Idempotent launcher: ensure the solver is running.

        - If the solver is already running (this process or another host
          process detected via ``Utils.busy()``), return without
          relaunching.
        - Otherwise, start a fresh simulation from frame 0.

        To pick up a previously-saved state, use :meth:`resume` explicitly,
        since ``run()`` never auto-resumes.

        Args:
            blocking (Optional[bool]): If ``True``, block until the solver
                finishes. If ``None``, defaults to blocking outside Jupyter
                and non-blocking inside Jupyter.

        Returns:
            FixedSession: This session.

        Example:
            Ensure the solver is running without restarting it if it already
            is::

                session.run(blocking=True)
        """
        if self.is_running():
            return self
        # force=True here bypasses start()'s auto-resume branch so we
        # always begin from frame 0. It won't terminate anything because
        # is_running() was just False.
        return self.start(force=True, blocking=blocking)

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
            (
                "e.squarednorm() > sqr(offset)",
                "Contact offset too large: a rod vertex is within offset distance "
                "of a triangle face. Reduce contact-offset or increase separation.",
            ),
        ]

        for line in all_lines:
            line_lower = line.lower().strip()
            for pattern, message in error_patterns:
                if pattern.lower() in line_lower:
                    return message

        # Check for panic/assertion with surrounding context
        for i, line in enumerate(all_lines):
            if "panicked at" in line or "assertion failed" in line.lower():
                context_lines = []
                # Include up to 3 lines before and after for context
                for j in range(max(0, i - 3), min(len(all_lines), i + 4)):
                    stripped = all_lines[j].strip()
                    if stripped:
                        context_lines.append(stripped)
                return "\n".join(context_lines)

        return None

    def delete(self):
        """Delete the session.

        Example:
            Remove the on-disk session directory before a clean rerun::

                session.delete()
        """
        if os.path.exists(self.info.path):
            shutil.rmtree(self.info.path)

    def _check_ready(self):
        """Check if the session is ready."""
        if self.fixed_scene is None:
            raise ValueError("Scene must be initialized")

    def finished(self) -> bool:
        """Check if the session has finished.

        Any stderr lines present are printed as a side effect.

        Returns:
            bool: ``True`` if a ``finished.txt`` marker exists in the output
            directory, ``False`` otherwise.

        Example:
            Assert the run completed cleanly in CI::

                if app.ci:
                    assert session.finished()
        """
        finished_path = os.path.join(self.output.path, "finished.txt")
        error = self.get.log.stderr()
        if len(error) > 0:
            for line in error:
                print(line)
        return os.path.exists(finished_path)

    def initialize_finished(self) -> bool:
        """Check if the session initialization has finished.

        Any stderr lines present are printed as a side effect.

        Returns:
            bool: ``True`` if an ``initialize_finish.txt`` marker exists in
            the output directory, ``False`` otherwise.

        Example:
            Wait for solver initialization to complete before continuing::

                while not session.initialize_finished():
                    time.sleep(1)
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
        blocking: Optional[bool] = None,
    ) -> "FixedSession":
        """Resume the solver from a saved state.

        Args:
            frame (int): The saved frame to resume from. If ``-1``, resumes
                from the most recent saved frame. Defaults to ``-1``.
            force (bool): Forwarded to :meth:`start`. Defaults to ``True``.
            blocking (Optional[bool]): Forwarded to :meth:`start`.

        Returns:
            FixedSession: This session.

        Example:
            Pick up where a previous run left off::

                session.resume()                 # latest saved frame
                session.resume(frame=120)        # specific saved frame
        """
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
        blocking: Optional[bool] = None,
        load: int = 0,
    ) -> "FixedSession":
        """Start the session.

        Inside a Jupyter notebook the function returns immediately by
        default and the solver runs in the background; outside Jupyter it
        blocks until the solver finishes. Pass ``blocking`` explicitly to
        override this behavior.

        If saved states exist and ``force`` is ``False``, this delegates to
        :meth:`resume` from the latest saved frame.

        Args:
            force (bool, optional): If ``True``, terminate any running
                solver and skip the auto-resume branch. Defaults to
                ``False``.
            blocking (Optional[bool], optional): Whether to block until the
                solver finishes. Defaults to ``None`` (auto-detect based on
                Jupyter).
            load (int, optional): The frame number to load from saved
                states. Defaults to ``0`` (start fresh).

        Returns:
            FixedSession: The started session.

        Example:
            Launch the solver and return immediately for notebook-style
            monitoring::

                session.start().preview()
                session.stream()

            Or block until finished when running as a script::

                session.start(blocking=True)
        """
        Utils.check_gpu()

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
        if self.is_running():
            if force:
                Utils.terminate()
                self._process = None
            else:
                from IPython.display import display

                self.print("Solver is already running. Terminate first.")
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
                    get_export_base_path(),
                    self._session.app_name,
                    self.info.name,
                )
                if os.path.exists(export_path):
                    shutil.rmtree(export_path)

            err_path = os.path.join(self.info.path, "error.log")
            log_path = os.path.join(self.info.path, "stdout.log")
            if platform.system() == "Windows":  # Windows
                command = f'"{self._cmd_path}" --load {load}'
            else:
                command = f"bash {self._cmd_path} --load {load}"
            with open(log_path, "w") as stdout_file, open(err_path, "w") as stderr_file:
                if platform.system() == "Windows":  # Windows
                    self._process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # pyright: ignore[reportAttributeAccessIssue]
                        cwd=self._session.proj_root,
                    )
                else:
                    self._process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        start_new_session=True,
                        cwd=self._session.proj_root,
                    )
            process = self._process
            if blocking is None:
                blocking = not Utils.in_jupyter_notebook()
            if not blocking:
                # Wait briefly for process to initialize and verify it's stable
                time.sleep(0.2)  # Give process time to start
                if not self.is_running():
                    # Process exited within 0.2s. Could be a startup crash
                    # (rc != 0) or a legitimately-short run that finished
                    # before the poll (rc == 0). The latter happens in
                    # practice when resuming from a saved frame near the
                    # end of the simulation: load state, run remaining
                    # frame(s), exit cleanly. Without this distinction
                    # the solver wrapper would misclassify a successful
                    # short run as ``"Solver failed to start"``.
                    rc = process.poll() if process is not None else None
                    if rc is None or rc == 0:
                        # Clean exit (or process not yet observable but
                        # is_running() reported false — give it the benefit
                        # of the doubt rather than raise on stale state).
                        return self
                    if os.path.exists(err_path):
                        with open(err_path) as f:
                            err_lines = f.readlines()
                    else:
                        err_lines = []
                    if os.path.exists(log_path):
                        with open(log_path) as f:
                            log_lines = f.readlines()
                    else:
                        log_lines = []
                    error_message = self._analyze_solver_error(log_lines, err_lines)
                    if error_message:
                        raise ValueError(error_message)
                    elif err_lines:
                        raise ValueError(f"Solver failed: {''.join(err_lines[:5])}")
                    else:
                        raise ValueError(
                            f"Solver failed to start (rc={rc})"
                        )
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
                    while self.is_running():
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
                with tqdm(total=total_frames, desc="progress") as pbar:
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

            async def _terminate_async(button):
                button.disabled = True
                button.description = "Terminating..."
                Utils.terminate()
                self._process = None
                while self.is_running():
                    await asyncio.sleep(0.25)
                button.description = "Terminated"

            def _terminate(button):
                asyncio.ensure_future(_terminate_async(button))

            button = widgets.Button(description=description)
            button.on_click(_terminate)
            return button
        else:
            return None

    def save_and_quit_file_path(self) -> str:
        """Get the flag-file path that signals the solver to save and quit.

        If this file exists, the solver will save the session and quit.
        After the session is saved, the file is removed.

        Example:
            Check for the save-and-quit sentinel file::

                path = session.save_and_quit_file_path()
                print(os.path.exists(path))
        """
        return os.path.join(self.info.path, "output", "save_and_quit")

    def save_and_quit(self):
        """Save the session and quit the solver.

        Example:
            Ask a running solver to checkpoint and exit gracefully::

                session.save_and_quit()
                while session.is_running():
                    time.sleep(1)
        """
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

            async def _save_and_quit_async(button):
                button.disabled = True
                button.description = "Requesting..."
                self.save_and_quit()
                while self.is_running():
                    await asyncio.sleep(0.25)
                self._process = None
                button.description = "Done"

            def _save_and_quit(button):
                asyncio.ensure_future(_save_and_quit_async(button))

            button = widgets.Button(description=description)
            button.on_click(_save_and_quit)
            return button
        else:
            return None

    def update_options(self, options: dict) -> dict:
        """Return a copy of ``options`` with missing defaults filled in.

        Args:
            options (dict): User-supplied render options.

        Returns:
            dict: A new dictionary combining ``options`` with the session's
            default option values.

        Example:
            Add the session's default render flags to a user-provided
            options dict::

                opts = session.update_options({"flat_shading": True})
        """
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
        """Live-view the session inside a Jupyter notebook.

        Outside Jupyter this is a no-op and returns ``None``.

        Args:
            options (dict, optional): The render options.
            live_update (bool, optional): Whether to enable live updates.
                Defaults to ``True``.
            engine (str, optional): The rendering engine. Defaults to
                ``"threejs"``.

        Returns:
            Optional[Plot]: The plot object, or ``None`` when not running
            inside a Jupyter notebook.

        Example:
            Kick off a run and watch it in-notebook while tailing stdout::

                session.start().preview()   # live frame playback
                session.stream()            # tail solver stdout
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

            if live_update and self.is_running():

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
                    from ._utils_ import dict_to_html_table

                    table.value = dict_to_html_table(
                        data, classes="table table-striped"
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
                        assert self.fixed_scene is not None
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
                            if not self.is_running():
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
                            if not self.is_running():
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
        """Show the animation inside a Jupyter notebook.

        Outside Jupyter this is a no-op. Loads all available frames from
        disk and exposes a slider plus a reload button to pull in frames
        produced after the initial load.

        Args:
            options (dict, optional): The render options.
            engine (str, optional): The rendering engine. Defaults to
                ``"threejs"``.

        Returns:
            FixedSession: This session.

        Example:
            Replay frames once the run has finished (or has enough frames
            on disk)::

                session.animate()
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
                                if not self.is_running():
                                    print(
                                        "Simulation finished but no frames were generated."
                                    )
                                    return self
                                time.sleep(0.5)
                            frame_count = self.get.vertex_frame_count()
                            print(f"Found {frame_count} frame(s). Loading animation...")

                        vert_list = []
                        for i in tqdm(range(frame_count), desc="loading frames"):
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
                            except Exception:
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
        """Stream the tail of the session stdout log inside a Jupyter notebook.

        Outside Jupyter this is a no-op.

        Args:
            n_lines (int, optional): The number of trailing lines to
                display. Defaults to ``40``.

        Returns:
            FixedSession: This session.

        Example:
            Kick off a run and watch both the live preview and stdout tail
            in a notebook cell::

                session.start().preview()
                session.stream()
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
                        if not self.is_running():
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

                async def toggle_stream_async(b):
                    nonlocal stop
                    nonlocal thread
                    if thread.is_alive():
                        stop = True
                        b.disabled = True
                        b.description = "Stopping..."
                        while thread.is_alive():
                            await asyncio.sleep(0.1)
                        b.disabled = False
                        b.description = "Start Live Stream"
                    else:
                        thread = threading.Thread(target=live_stream, args=(self,))
                        stop = False
                        thread.start()
                        b.description = "Stop Live Stream"

                def toggle_stream(b):
                    asyncio.ensure_future(toggle_stream_async(b))

                button.on_click(toggle_stream)
            else:
                log_widget.value = "No log file found."
                terminate_button.disabled = True
                save_and_quit_button.disabled = True
                button.disabled = True

        return self

    @property
    def fixed_scene(self) -> Optional[FixedScene]:
        """Get the fixed scene.

        Example:
            Retrieve the bound scene from a fixed session::

                fixed = app.session.create(scene).build()
                scene_ref = fixed.fixed_scene
        """
        return self._session.fixed_scene


class Session:
    """Class to setup a simulation session.

    Instances are created via :meth:`SessionManager.create`, configured via
    the :attr:`param` manager, then finalized with :meth:`build` to produce
    a :class:`FixedSession`.

    Example:
        Configure a session and build it into a runnable fixed session::

            session = app.session.create(scene)
            session.param.set("frames", 120).set("dt", 0.01)
            fixed_session = session.build()
            fixed_session.start(blocking=True)
    """

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
        """Get the session parameter manager.

        Example:
            Configure solver parameters before building the session::

                session = app.session.create(scene)
                session.param.set("frames", 120).set("dt", 0.01)
        """
        return self._param

    @property
    def fixed_scene(self) -> Optional[FixedScene]:
        """Get the fixed scene.

        Returns:
            Optional[FixedScene]: The fixed scene object.

        Example:
            Inspect the bound scene before finalizing::

                session = app.session.create(scene)
                print(session.fixed_scene)
        """
        return self._fixed_scene

    @property
    def fixed_session(self) -> Optional[FixedSession]:
        """Get the fixed session.

        Returns:
            Optional[FixedSession]: The fixed session object, or ``None`` if
            :meth:`build` has not been called yet.

        Example:
            Access the built runnable session after calling :meth:`build`::

                session = app.session.create(scene)
                session.build()
                fixed = session.fixed_session
        """
        return self._fixed_session

    @property
    def proj_root(self) -> str:
        """Get the project root directory.

        Example:
            Print the project root for the current session::

                session = app.session.create(scene)
                print(session.proj_root)
        """
        return self._proj_root

    @property
    def app_name(self) -> str:
        """Get the application name.

        Example:
            Read the owning application name::

                session = app.session.create(scene)
                print(session.app_name)
        """
        return self._app_name

    @property
    def name(self) -> str:
        """Get the session name.

        Example:
            Retrieve the session name for logging or display::

                session = app.session.create(scene)
                print(session.name)
        """
        return self._name

    @property
    def app_root(self) -> str:
        """Get the application root directory.

        Example:
            Locate the application root on disk::

                session = app.session.create(scene)
                print(session.app_root)
        """
        return self._app_root

    def _check_ready(self):
        """Check if the session is ready."""
        if self._fixed_scene is None:
            raise ValueError("Scene must be initialized")

    def init(self, scene: FixedScene) -> "Session":
        """Attach a fixed scene to this session.

        Args:
            scene (FixedScene): The fixed scene.

        Returns:
            Session: This session, for chaining.

        Example:
            :meth:`SessionManager.create` calls this internally, but it can
            also be used to re-bind a scene to an existing session::

                session.init(scene).param.set("frames", 60)
        """
        self._fixed_scene = scene
        return self

    def build(self) -> FixedSession:
        """Build and persist a :class:`FixedSession` from this session.

        Pickles the built session into its directory and creates a symlink
        (or a ``.txt`` fallback on Windows without symlink privileges) under
        the data dir for convenient access.

        Returns:
            FixedSession: The newly built fixed session.

        Example:
            Finalize a configured session and start it::

                session.param.set("frames", 60).set("dt", 0.01)
                fixed_session = session.build()
                fixed_session.start(blocking=True)
        """
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
        """Save the fixed session to a recoverable file and create a symlink."""
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


def read_average_summary_from_disk(session_root: str) -> dict:
    """Read averaged simulation stats from .out files on disk.

    Standalone function usable without a FixedSession — uses the same
    docstring parser and metric definitions as SessionLog.average_summary().

    Args:
        session_root: Path to the session directory (containing output/data/*.out).

    Returns:
        dict with formatted stats (same format as SessionLog.average_summary).
    """
    data_dir = os.path.join(session_root, "output", "data")
    if not os.path.isdir(data_dir):
        return {}

    # Discover log name → filename mapping from source code
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    from ._parse_ import CppRustDocStringParser
    log_map = CppRustDocStringParser.get_logging_docstrings(src_path)

    def _read_values(name: str):
        if name not in log_map:
            return None
        path = os.path.join(data_dir, log_map[name]["filename"])
        if not os.path.exists(path):
            return None
        values = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1]))
                    except ValueError:
                        pass
        return values if values else None

    def _avg(name):
        v = _read_values(name)
        return sum(v) / len(v) if v else None

    def _max(name):
        v = _read_values(name)
        return max(v) if v else None

    result = {}
    val = _avg("time-per-frame")
    if val is not None:
        result["time-per-frame"] = convert_time(val)
    val = _avg("time-per-step")
    if val is not None:
        result["time-per-step"] = convert_time(val)
    val = _max("num-contact")
    if val is not None:
        result["num-contact (max)"] = convert_integer(round(val))
    val = _avg("newton-steps")
    if val is not None:
        result["newton-steps"] = f"{val:.2f}"
    val = _avg("pcg-iter")
    if val is not None:
        result["pcg-iter"] = f"{val:.2f}"
    val = _avg("max-sigma")
    if val is not None and val > 0.0:
        result["stretch"] = f"{100.0 * (val - 1.0):.2f}%"
    return result
