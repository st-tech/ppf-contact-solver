# File: _session_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ._scene_ import FixedScene
from ._plot_ import Plot
from ._plot_ import in_jupyter_notebook
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
import re
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
    def __init__(self, app_root: str):
        path = os.path.abspath(os.path.join(app_root, "src", "args.rs"))
        self._key = None
        self._param = get_default_params(path)
        self._time = 0.0
        self._dyn_param = {}

    def set(self, key: str, value: Any) -> "Param":
        if "_" in key:
            raise ValueError("Key cannot contain underscore. Use '-' instead.")
        elif key not in self._param.keys():
            raise ValueError(f"Key {key} does not exist")
        else:
            self._param[key] = value
        return self

    def dyn(self, key: str) -> "Param":
        if key not in self._param.keys():
            raise ValueError(f"Key {key} does not exist")
        else:
            self._key = key
        return self

    def change(self, value: float) -> "Param":
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param.keys():
                self._dyn_param[self._key].append((self._time, value))
            else:
                initial_val = self._param[self._key]
                self._dyn_param[self._key] = [
                    (0.0, initial_val),
                    (self._time, value),
                ]
            return self

    def hold(self) -> "Param":
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param.keys():
                last_val = self._dyn_param[self._key][-1][1]
                self.change(last_val)
            else:
                val = self._param[self._key]
                if isinstance(val, float):
                    self.change(val)
                else:
                    raise ValueError("Key must be float")
        return self

    def export(self, path: str):
        if len(self._param.keys()):
            with open(os.path.join(path, "param.toml"), "w") as f:
                f.write("[param]\n")
                for key, value in self._param.items():
                    key = key.replace("-", "_")
                    if value is not None:
                        if isinstance(value, str):
                            f.write(f'{key} = "{value}"\n')
                        elif isinstance(value, bool):
                            if value:
                                f.write(f"{key} = true\n")
                        else:
                            f.write(f"{key} = {value}\n")
                    else:
                        f.write(f"{key} = false\n")
        if len(self._dyn_param.keys()):
            with open(os.path.join(path, "dyn_param.txt"), "w") as f:
                for key, vals in self._dyn_param.items():
                    f.write(f"[{key}]\n")
                    for entry in vals:
                        time, value = entry
                        f.write(f"{time} {value}\n")

    def time(self, time: float) -> "Param":
        if time <= self._time:
            raise ValueError("Time must be increasing")
        else:
            self._time = time
        return self

    def get(self, key: Optional[str] = None):
        if key is None:
            return self._param
        else:
            return self._param[key]

    def items(self):
        return self._param.items()

    def delete(self, key: str):
        del self._param[key]

    def clear(self):
        self._param = {}


class SessionManager:
    def __init__(self, app_root: str, proj_root: str, save_func):
        self._app_root = app_root
        self._proj_root = proj_root
        self._save_func = save_func
        self._sessions = {}
        self._curr = None

    def list(self):
        return self._sessions

    def select(self, name: str):
        if name not in self._sessions.keys():
            raise ValueError(f"Session {name} does not exist")
        self._curr = name
        return self._sessions[name]

    def current(self):
        if self._curr is None:
            return None
        else:
            return self._sessions[self._curr]

    def create(self, name: str, delete_if_exists: bool = True) -> "Session":
        if name in self._sessions.keys():
            if delete_if_exists:
                session = self._sessions[name]
                if is_running():
                    raise ValueError(f"Session {name} is running")
                else:
                    self._sessions[name].delete()
            else:
                raise ValueError(f"Session {name} already exists")
        session = Session(self._app_root, self._proj_root, name, self._save_func)
        self._sessions[name] = session
        self._curr = name
        return session

    def _terminate_or_raise(self, force: bool):
        if is_running():
            if force:
                terminate()
            else:
                raise ValueError("Solver is running. Terminate first.")

    def delete(self, name: str, force: bool = True):
        self._terminate_or_raise(force)
        if name in self._sessions.keys():
            self._sessions[name].delete()
            del self._sessions[name]
            if name == self._curr:
                self._curr = None

    def clear(self, force: bool = True):
        self._terminate_or_raise(force)
        for session in self._sessions.values():
            session.delete()
        self._sessions = {}
        self._curr = None

    def param(self) -> Param:
        return Param(self._proj_root)


class SessionInfo:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


class SessionExport:
    def __init__(self, session: "Session"):
        self._session = session

    def shell_command(
        self,
        param: Param,
    ) -> str:
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

    def animation(self, path: str, ext="ply", include_static: bool = True):
        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            os.makedirs(path)
        for i in tqdm(range(self._session.get.latest_frame()), desc="export", ncols=70):
            self.frame(os.path.join(path, f"frame_{i}.{ext}"), i, include_static)

    def frame(
        self, path: str, frame: Optional[int] = None, include_static: bool = True
    ) -> "Session":
        if self._session._fixed is None:
            raise ValueError("Scene must be initialized")
        else:
            vert = self._session._fixed._vert
            if frame is not None:
                result = self._session.get.vertex(frame)
                if result is not None:
                    vert, _ = result
            else:
                result = self._session.get.vertex()
                if result is not None:
                    vert, _ = result
            self._session._fixed.export(vert, path, include_static)
        return self._session


class SessionOutput:
    def __init__(self, session: "Session"):
        self._session = session
        self.path = os.path.join(self._session.info.path, "output")


class SessionGet:
    def __init__(self, session: "Session"):
        self._session = session

    def log(self) -> list[str]:
        path = os.path.join(self._session.info.path, "output", "data")
        result = []
        for file in os.listdir(path):
            if file.endswith(".out"):
                result.append(file.replace(".out", ""))
        return result

    def numbers(self, name: str):
        def float_or_int(var):
            var = float(var)
            if var.is_integer():
                return int(var)
            else:
                return var

        path = os.path.join(self._session.info.path, "output", "data", f"{name}.out")
        entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    entry = line.split(" ")
                    entries.append([float_or_int(entry[0]), float_or_int(entry[1])])
        return entries

    def number(self, name: str):
        entries = self.numbers(name)
        if entries:
            return entries[-1][1]
        else:
            return None

    def vertex_frame_count(self) -> int:
        path = os.path.join(self._session.info.path, "output")
        max_frame = 0
        if os.path.exists(path):
            files = os.listdir(path)
            for file in files:
                if file.startswith("vert") and file.endswith(".bin"):
                    frame = int(file.split("_")[1].split(".")[0])
                    max_frame = max(max_frame, frame)
        return max_frame

    def latest_frame(self) -> int:
        path = os.path.join(self._session.info.path, "output")
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

    def vertex(self, n: Optional[int] = None) -> Optional[tuple[np.ndarray, int]]:
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
                            if len(vert) == len(self._session._fixed._vert):
                                return (vert, last_frame)
                            else:
                                return None
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
    def __init__(self, app_root: str, proj_root: str, name: str, save_func):
        self._in_jupyter_notebook = in_jupyter_notebook()
        self._app_root = app_root
        self._proj_root = proj_root
        self._fixed = None
        path = os.path.expanduser(os.path.join(app_root, "session", name))
        self._save_func = save_func
        self._update_preview_interval = 1.0 / 60.0
        self._update_terminal_interval = 1.0 / 30.0
        self._update_table_interval = 0.25
        self.info = SessionInfo(name, path)
        self.export = SessionExport(self)
        self.get = SessionGet(self)
        self.output = SessionOutput(self)
        self.delete()

    def print(self, message):
        if self._in_jupyter_notebook:
            from IPython.display import display

            display(message)
        else:
            print(message)

    def delete(self):
        if os.path.exists(self.info.path):
            shutil.rmtree(self.info.path)

    def _check_ready(self):
        if self._fixed is None:
            raise ValueError("Scene must be initialized")

    def init(self, scene: FixedScene) -> "Session":
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

        self._save_func()
        return self

    def finished(self) -> bool:
        finished_path = os.path.join(self.output.path, "finished.txt")
        err_path = os.path.join(self.info.path, "error.log")
        if os.path.exists(err_path):
            file = open(err_path, "r")
            lines = file.readlines()
            if len(lines) > 0:
                for line in lines:
                    print(line)
        return os.path.exists(finished_path)

    def start(self, param: Param, force: bool = True, blocking=False) -> "Session":
        self._check_ready()
        if is_running():
            if force:
                terminate()
            else:
                from IPython.display import display

                self.print("Solver is already running. Teriminate first.")
                display(self._terminate_button("Terminate Now"))
                return self
        cmd_path = self.export.shell_command(param)
        err_path = os.path.join(self.info.path, "error.log")
        log_path = os.path.join(self.info.path, "stdout.log")
        if blocking or not self._in_jupyter_notebook:
            subprocess.run(cmd_path, cwd=self._proj_root, shell=True)
            return self
        else:
            command = open(cmd_path, "r").read()
            process = subprocess.Popen(
                command.split(),
                stdout=open(log_path, "w"),
                stderr=open(err_path, "w"),
                start_new_session=True,
                cwd=self._proj_root,
            )
            if process.poll() is not None:
                raise ValueError("Solver failed to start")
            else:
                init_path = os.path.join(
                    self.info.path, "output", "data", "initialize.out"
                )
                time.sleep(1)
                while is_running():
                    if os.path.exists(init_path):
                        return self
                    time.sleep(0.25)
                err_content = open(err_path, "r").readlines()
                display_log(err_content)
                raise ValueError("Solver failed to start")

    def _terminate_button(self, description: str = "Terminate Solver"):
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

    def preview(self, live_update: bool = True) -> Optional["Plot"]:
        if self._in_jupyter_notebook:
            import ipywidgets as widgets
            from IPython.display import display

            shading = {"wireframe": False}
            if self._fixed is None:
                raise ValueError("Scene must be initialized")
            else:
                result = self.get.vertex()
                if result is None:
                    vert, curr_frame = self._fixed._vert, 0
                else:
                    vert, curr_frame = result
                plot = self._fixed.preview(
                    vert, shading=shading, show_pin=False, show_stitch=False
                )

            table = widgets.HTML()
            button = self._terminate_button()

            def convert_integer(number) -> str:
                if number is None:
                    return "N/A"
                elif number < 1000:
                    return str(number)
                elif number < 1_000_000:
                    return f"{number/1_000:.2f}k"
                elif number < 1_000_000_000:
                    return f"{number/1_000_000:.2f}M"
                else:
                    return f"{number/1_000_000_000:.2f}B"

            def convert_time(time) -> str:
                if time is None:
                    return "N/A"
                elif time < 1_000:
                    return f"{int(time)}ms"
                elif time < 60_000:
                    return f"{time/1_000:.2f}s"
                else:
                    return f"{time/60_000:.2f}m"

            if live_update and is_running():

                def update_dataframe(table, curr_frame):
                    time_per_frame = convert_time(self.get.number("per_video_frame"))
                    time_per_step = convert_time(self.get.number("advance"))
                    n_contact = convert_integer(self.get.number("advance.num_contact"))
                    n_newton = convert_integer(self.get.number("advance.newton_steps"))
                    max_sigma = self.get.number("advance.max_sigma")
                    n_pcg = convert_integer(self.get.number("advance.iter"))
                    data = {
                        "Frame": [str(curr_frame)],
                        "Time/Frame": [time_per_frame],
                        "Time/Step": [time_per_step],
                        "#Contact": [n_contact],
                        "#Newton": [n_newton],
                        "#PCG": [n_pcg],
                    }
                    if max_sigma is not None:
                        stretch = f"{100.0 * (max_sigma - 1.0):.2f}%"
                        data["Max Stretch"] = [stretch]
                    df = pd.DataFrame(data)
                    table.value = df.to_html(
                        classes="table table-striped", border=0, index=False
                    )

                def live_preview(self):
                    nonlocal plot
                    nonlocal button
                    nonlocal table
                    nonlocal curr_frame
                    assert plot is not None
                    while True:
                        last_frame = self.get.latest_frame()
                        if curr_frame != last_frame:
                            curr_frame = last_frame
                            result = self.get.vertex(curr_frame)
                            if result is not None:
                                vert, _ = result
                                update_dataframe(table, curr_frame)
                                plot.update(vert)
                        if not is_running():
                            break
                        time.sleep(self._update_preview_interval)
                    assert button is not None
                    button.disabled = True
                    button.description = "Terminated"

                def live_table(self):
                    nonlocal table
                    while True:
                        update_dataframe(table, curr_frame)
                        if not is_running():
                            break
                        time.sleep(self._update_table_interval)

                threading.Thread(target=live_preview, args=(self,)).start()
                threading.Thread(target=live_table, args=(self,)).start()
                display(button)

            display(table)
            return plot
        else:
            return None

    def animate(self) -> "Session":
        if self._in_jupyter_notebook:
            import ipywidgets as widgets

            shading = {"wireframe": False}
            if self._fixed is None:
                raise ValueError("Scene must be initialized")
            else:
                plot = self._fixed.preview(
                    self._fixed._vert,
                    shading=shading,
                    show_pin=False,
                    show_stitch=False,
                )
                try:
                    if self._fixed is not None:
                        frame_count = self.get.vertex_frame_count()
                        vert_list = []
                        for i in tqdm(
                            range(frame_count), desc="Loading frames", ncols=70
                        ):
                            result = self.get.vertex(i)
                            if result is not None:
                                vert, _ = result
                                vert_list.append(vert)

                        def update(frame=1):
                            nonlocal vert_list
                            nonlocal plot
                            assert plot is not None
                            if self._fixed is not None:
                                plot.update(vert_list[frame - 1])

                        widgets.interact(update, frame=(1, frame_count))
                except Exception as _:
                    pass
        return self

    def stream(self, n_lines=40) -> "Session":
        if self._in_jupyter_notebook:
            import ipywidgets as widgets
            from IPython.display import display

            log_widget = widgets.HTML()
            display(log_widget)
            button = widgets.Button(description="Stop Live Stream")
            terminate_button = self._terminate_button()
            display(widgets.HBox((button, terminate_button)))

            stop = False
            log_path = os.path.join(self.info.path, "output", "cudasim_log.txt")
            err_path = os.path.join(self.info.path, "error.log")
            if os.path.exists(log_path):

                def live_stream(self):
                    nonlocal stop
                    nonlocal button
                    nonlocal log_widget
                    nonlocal log_path
                    nonlocal err_path
                    nonlocal terminate_button
                    assert terminate_button is not None

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
                button.disabled = True

        return self


def is_running() -> bool:
    for proc in psutil.process_iter(["pid", "name", "status"]):
        if (
            PROCESS_NAME in proc.info["name"]
            and proc.info["status"] != psutil.STATUS_ZOMBIE
        ):
            return True
    return False


def terminate():
    for proc in psutil.process_iter(["pid", "name", "status"]):
        if (
            PROCESS_NAME in proc.info["name"]
            and proc.info["status"] != psutil.STATUS_ZOMBIE
        ):
            pid = proc.info["pid"]
            os.kill(pid, signal.SIGTERM)


def display_log(lines: list[str]):
    if in_jupyter_notebook():
        import ipywidgets as widgets
        from IPython.display import display

        log_widget = widgets.HTML()
        text = "\n".join(lines)
        log_widget.value = CONSOLE_STYLE + f"<pre style='no-scroll'>{text}</pre>"
        display(log_widget)


def get_default_params(path: str) -> dict[str, Any]:
    att_pattern = re.compile(r"#\[(.*?)\]")
    field_pattern = re.compile(r"pub\s+(\w+):\s*([^,]+),?")
    struct_start_pattern = re.compile(r"^pub\s+struct\s+Args\s*\{")
    struct_end_pattern = re.compile(r"^\s*\}")
    curr_attributes = []
    inside_struct = False
    result = {}

    with open(path, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if not inside_struct:
                if struct_start_pattern.match(line.strip()):
                    inside_struct = True
                continue
            if struct_end_pattern.match(line.strip()):
                break
            if not line.strip() or line.strip().startswith("//"):
                continue
            attr_match = att_pattern.match(line.strip())
            if attr_match:
                curr_attributes.append(attr_match.group(1).strip())
            else:
                field_match = field_pattern.match(line.strip())
                if field_match:
                    field_name = field_match.group(1).replace("_", "-")
                    default_value = None
                    for attr in curr_attributes:
                        clap_match = re.match(r"clap\((.*?)\)", attr)
                        if clap_match:
                            args = clap_match.group(1)
                            arg_list = re.findall(r'(?:[^,"]|"(?:\\.|[^"\\])*")+', args)
                            for arg in arg_list:
                                arg = arg.strip()
                                if "=" in arg:
                                    key, value = map(str.strip, arg.split("=", 1))
                                    value = value.strip('"').strip("'")
                                    if "default_value" in key:
                                        default_value = value
                    if default_value is not None:
                        try:
                            float_value = float(default_value)
                            default_value = (
                                int(float_value)
                                if float_value.is_integer()
                                else float_value
                            )
                        except ValueError:
                            pass
                    result[field_name] = default_value
                    curr_attributes = []
                else:
                    curr_attributes = []
    return result
