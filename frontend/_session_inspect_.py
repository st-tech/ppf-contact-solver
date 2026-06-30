# File: _session_inspect_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Session inspection / export helpers split out of ``_session_.py``.

Holds:

* ``Zippable`` (directory zipper)
* ``SessionExport`` (animation + frame + shell-command exporters)
* ``SessionOutput`` (output-directory accessor)
* ``SessionLog`` (stdout/stderr/log readers)
* ``SessionGet`` (vertex/log/command getters)
* The ``display_log`` helper.
* The ``CONSOLE_STYLE`` HTML snippet shared with ``FixedSession``.

Everything is re-exported from ``_session_.py`` for backward compatibility.
"""

import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from tqdm.auto import tqdm

from . import _rust  # type: ignore[attr-defined]

from ._parse_ import CppRustDocStringParser
from ._utils_ import Utils, get_export_base_path

if TYPE_CHECKING:
    from ._session_ import FixedSession
    from ._session_param_ import ParamManager


CONSOLE_STYLE = """
    <style>
        .no-scroll {
            overflow: hidden;
            white-space: pre-wrap;
            font-family: monospace;
        }
    </style>
    """


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
            # `prepare_zip_target` removes any existing `.zip` and
            # returns the destination path so the print line and the
            # `make_archive` call agree on it. `shutil.make_archive`
            # itself stays Python (stdlib zip writer).
            path = _rust.prepare_zip_target(self._dirpath)
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
        param: "ParamManager",
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
        which = Utils.platform_which()
        return _rust.write_shell_command_script(
            self._fixed_session.info.path,
            self._fixed_session.output.path,
            self._session.proj_root,
            which,
        )

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
                path = _rust.export_base_path_for(
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
        # Look for a bundled ffmpeg first (Rust handles the path
        # arithmetic + isfile check); fall back to PATH via
        # `shutil.which` (Python only, walks `$PATH`).
        project_root = _rust.project_root_from_frontend_file(os.path.abspath(__file__))
        ffmpeg_path = _rust.locate_bundled_ffmpeg(project_root)
        if ffmpeg_path is None:
            ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None and png_files:
            vid_name = "frame.mp4"
            command = _rust.ffmpeg_video_command(ffmpeg_path, ext, vid_name)
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


def _harvest_log_docstrings(proj_root: str) -> dict:
    """Walk every plausible source root for `// Name:` / `logging.push("...")`
    log-channel docstrings and return a merged ``name -> entry`` dict.

    Two source roots host log-channel docstrings: the CUDA driver at
    ``<proj_root>/crates/ppf-cts-solver/src`` and the Rust kernels at
    ``<proj_root>/crates/ppf-cts-core/src``. The legacy
    ``<proj_root>/src`` is also probed for older checkouts that still
    carry it; missing roots are skipped so any subset still resolves
    every channel that physically exists.
    """
    roots = [
        os.path.join(proj_root, "src"),
        os.path.join(proj_root, "crates", "ppf-cts-solver", "src"),
        os.path.join(proj_root, "crates", "ppf-cts-core", "src"),
    ]
    merged: dict = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        merged.update(CppRustDocStringParser.get_logging_docstrings(r))
    return merged


class SessionLog:
    """Class to handle session log retrieval operations."""

    def __init__(self, fixed_session: "FixedSession") -> None:
        self._fixed_session = fixed_session
        self._log = _harvest_log_docstrings(fixed_session.session.proj_root)

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
        path = _rust.session_log_tail_path(self._fixed_session.info.path, "stdout")
        return _rust.read_log_tail(path, n_lines)

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
        path = _rust.session_log_tail_path(self._fixed_session.info.path, "stderr")
        return _rust.read_log_tail(path, n_lines)

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
        if name not in self._log:
            return None
        log_filenames = [(k, v["filename"]) for k, v in self._log.items()]
        path = _rust.session_log_filename_path(
            self._fixed_session.info.path, name, log_filenames
        )
        if path is None:
            return None
        return _rust.read_log_numbers_squashed(path)

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
        max_sigma = self.number("max-sigma")
        return _rust.log_summary(
            float(self.number("time-per-frame") or 0.0),
            float(self.number("time-per-step") or 0.0),
            float(self.number("num-contact") or 0.0),
            float(self.number("newton-steps") or 0.0),
            float(self.number("pcg-iter") or 0.0),
            float(max_sigma) if max_sigma is not None else None,
        )

    def average_summary(self):
        """Get averages for log-backed metrics only.

        Returns:
            dict: A dictionary containing averaged statistics. Metrics without a
            corresponding existing ``.out`` file are omitted.

        Example:
            Print the run-averaged metrics after a simulation has finished::

                print(session.get.log.average_summary())
        """
        # All six per-metric file reads + average/max + format happen
        # in Rust via `average_summary_from_disk`. Pass the same
        # `(name, filename)` mapping we use for `numbers()`.
        log_filenames = [(k, v["filename"]) for k, v in self._log.items()]
        data_dir = os.path.join(self._fixed_session.output.path, "data")
        return _rust.average_summary_from_disk(data_dir, log_filenames)


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

    def latest_frame(self) -> int:
        """Get the latest frame number.

        Returns:
            int: The latest frame number.

        Example:
            Poll the most recent frame index while the solver runs::

                frame = session.get.latest_frame()
                print(f"solver is on frame {frame}")
        """
        path = self._fixed_session.output.path
        return int(_rust.latest_vertex_frame(path))

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
        output_path = self._fixed_session.output.path
        return [int(n) for n in _rust.list_saved_states(output_path)]

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
        path = self._fixed_session.output.path
        if n is None:
            got = _rust.read_latest_vertex(path)
            if got is None:
                return None
            vert, frame = got
            return (vert, int(frame))
        arr = _rust.read_vertex_bin(path, int(n))
        if arr is None:
            return None
        return (arr, int(n))

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
        which = Utils.platform_which()
        return _rust.command_path(self._fixed_session.info.path, which)

    def param_summary(self) -> list[str]:
        """Get the parameter summary from the param_summary.txt file.

        Returns:
            list[str]: The lines from the parameter summary file, or empty list if file doesn't exist.

        Example:
            Print the parameter summary captured by the solver at launch::

                for line in session.get.param_summary():
                    print(line)
        """
        return _rust.param_summary_lines(self._fixed_session.info.path)

    def nvidia_smi(self) -> None:
        """Read and print the exported nvidia-smi outputs.

        Reads both nvidia-smi.txt and nvidia-smi-q.txt from the nvidia-smi directory
        and prints their concatenated contents.

        Example:
            Inspect the GPU state captured at the start of a run::

                session.get.nvidia_smi()
        """
        print(_rust.nvidia_smi_text(self._fixed_session.info.path))


def display_log(lines: list[str]):
    """Display the log lines.

    Args:
        lines (list[str]): The log lines.
    """
    lines = _rust.rstrip_newlines(list(lines))
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


