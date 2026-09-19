# File: scenarios/bl_remote_device_select.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Compute Device and GPU Backend choices reaching a server on ANOTHER
# machine, over SSH and over Docker.
#
# WHAT WOULD GO WRONG WITHOUT THIS, and it is the failure the whole device
# mechanism exists to prevent: a run that used a build the artist did not
# choose, with nothing reporting it. Both solvers are real and both answer
# `--backend` honestly about themselves, so the only symptom is a run roughly
# 30x faster or slower than asked for. A remote launch that named ONE fixed
# path, `target/release/ppf-cts-server`, would make the Compute Device a
# control that moves and changes no run.
#
# A SINGLE FIXED PATH ALSO CANNOT REACH A DISTRIBUTION AT ALL.
# `build-linux-native/bundle.sh` ships `target/<backend>/release` and no
# `target/release`, so that one path is the one path an unpacked Linux release
# does not have. Check E is that.
#
# WHAT THIS RUNS, AND WHY IT RUNS IN Blender CI. Nothing here needs a second
# machine, a daemon, an sshd or a GPU. A backend is reached through exactly one
# abstraction, `exec_command`, so a stand-in for it exercises the REAL launch
# path: `effect_runner._do_launch_server` composes the real script, which is
# recorded and read back. `bl_solver_gpu_select` check L already drives the
# launch this way, and this scenario is that technique applied to the question
# of WHICH BUILD rather than which GPU.
#
# CHECK H IS THE ONE THAT CATCHES A UI THAT LOOKS RIGHT. The rows are drawn
# only once a connection is up, so the artist's answer arrives at Start Server,
# and a launch reading the backend's connect-time copy would leave them movable
# and inert.
#
# WHAT IT DOES NOT PROVE, stated because a launch-command assertion is easy to
# over-read: that the far side honored the directory. Two other scenarios carry
# that half. `bl_ssh_remote_solve` check C is the remote one, and it is the only
# place in CI where the listing comes off a machine that is not the one running
# Blender: it resolves the Compute Device against a REAL remote host's listing
# and compares the answer with the directory that host's own server reports its
# runs use. `bl_native_device_real_solve` is the local one, and goes further: it
# runs a solve per device and asks each session launcher which solver binary
# executed.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Path resolution and launch-script composition; no solver runs, so the answer
# is the same on every backend.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import posixpath
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


ROOT = "/home/u/ppf-contact-solver"


def listing(*pairs):
    # A build listing in the shape `core.remote_builds` parses out of the
    # solver host's own answer: absolute directory -> marker.
    return {posixpath.join(ROOT, sub): marker for sub, marker in pairs}


class _Recorder:
    """A stand-in for the one abstraction a remote backend is reached through.

    It records the launch script and then FAILS the invocation, so the launch
    stops with the command already captured instead of waiting out its
    readiness poll against a server that will never come up.
    """

    def __init__(self, backend_type, device, gpu_backend):
        self.backend_type = backend_type
        self.server_port = 9090
        self.current_directory = ROOT
        self._device = device
        self._gpu_backend = gpu_backend
        self.script = ""
        self.commands = []

    def exec_command(self, command, shell=False, cwd=None, timeout=None):
        self.commands.append(command)
        if "start_server.sh\n" in command:
            self.script = command
        return {"exit_code": 1, "stdout": [], "stderr": []}


def launch(runner, backend, builds, device="", gpu_backend=""):
    # Drive the REAL launch effect and return (script, error). *device* and
    # *gpu_backend* are what Start Server passes; empty means the launch falls
    # back to what the backend was connected with.
    remote_builds.load_builds(
        [f"{m}\t{d}" for d, m in builds.items()], ROOT
    )
    saved = runner._backend
    runner._backend = backend
    error = ""
    try:
        runner._do_launch_server(-1, "", device, gpu_backend)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        runner._backend = saved
    return backend.script, error


try:
    facade = __import__(pkg + ".core.facade", fromlist=["runner"])
    conn = __import__(pkg + ".core.connection", fromlist=["remote_server_binary"])
    remote_builds = __import__(pkg + ".core.remote_builds",
                               fromlist=["load_builds"])
    backends_mod = __import__(pkg + ".core.backends", fromlist=["create_backend"])
    runner = facade.runner

    # ---- A: the probe asks for every layout, and its answer parses back ----
    command = remote_builds.probe_command(ROOT)
    record(
        "A_probe_command_covers_every_layout",
        all(
            # Each path is joined whole and then quoted, so it reads as one
            # shell word: `'<root>/<sub>/ppf-cts-server'`.
            f"'{ROOT}/{sub}/ppf-cts-server'" in command
            for sub in (
                "target/release",
                "target/cuda/release",
                "target/rocm/release",
                "target/cpu/release",
            )
        )
        and ".ppf-backend" in command,
        {"command": command[:300]},
    )
    # Each line is `<marker><TAB><directory>`, and the marker comes first
    # because every transport strips the whole output: a line ending in the
    # separator, which is what an unmarked directory would produce the other
    # way round, loses it and stops parsing.
    parsed = remote_builds.parse_listing(
        [
            f"cuda\t{ROOT}/target/cuda/release",
            f"cpu\t{ROOT}/target/cpu/release",
            "mesg: ttyname failed: Inappropriate ioctl for device",
            f"\t{ROOT}/target/release",
        ]
    )
    record(
        "A_listing_parses_and_ignores_a_banner",
        parsed
        == {
            f"{ROOT}/target/cuda/release": "cuda",
            f"{ROOT}/target/cpu/release": "cpu",
            f"{ROOT}/target/release": "",
        },
        {"parsed": parsed},
    )

    # ---- B: each selection resolves to its OWN directory ----
    both = listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    gpu = conn.remote_server_binary(ROOT, both, "GPU")
    cpu = conn.remote_server_binary(ROOT, both, "CPU")
    record(
        "B_each_device_resolves_its_own_build",
        gpu == f"{ROOT}/target/cuda/release/ppf-cts-server"
        and cpu == f"{ROOT}/target/cpu/release/ppf-cts-server",
        {"gpu": gpu, "cpu": cpu},
    )
    record(
        "B_a_missing_build_is_absent_rather_than_substituted",
        conn.remote_server_binary(
            ROOT, listing(("target/cuda/release", "cuda")), "CPU"
        )
        is None,
        {},
    )
    three = listing(
        ("target/cuda/release", "cuda"),
        ("target/rocm/release", "rocm"),
        ("target/cpu/release", "cpu"),
    )
    record(
        "B_a_named_accelerator_resolves_to_that_build",
        conn.remote_server_binary(ROOT, three, "GPU", "ROCM")
        == f"{ROOT}/target/rocm/release/ppf-cts-server",
        {"got": conn.remote_server_binary(ROOT, three, "GPU", "ROCM")},
    )
    record(
        "B_a_named_accelerator_that_is_absent_resolves_to_nothing",
        conn.remote_server_binary(ROOT, both, "GPU", "ROCM") is None,
        {},
    )
    record(
        "B_the_marker_wins_over_the_directory_name",
        conn.remote_server_binary(
            ROOT, listing(("target/cuda/release", "rocm")), "GPU", "ROCM"
        )
        == f"{ROOT}/target/cuda/release/ppf-cts-server",
        {},
    )

    # ---- C: the SSH launch names the resolved build AND its target dir ----
    #
    # BOTH HALVES, because naming only the binary is the split the native path
    # already guards: the build worker's frontend loads the cdylib from
    # whichever target directory it finds first and writes THAT into the
    # session's command.sh, so a CPU server can drive a GPU solve with nothing
    # reporting it.
    for device, backend_name in (("GPU", "cuda"), ("CPU", "cpu")):
        backend = _Recorder("ssh", device, "AUTO")
        script, error = launch(runner, backend, both)
        want_bin = f"{ROOT}/target/{backend_name}/release/ppf-cts-server"
        want_dir = f'export CARGO_TARGET_DIR="{ROOT}/target/{backend_name}"'
        record(
            f"C_ssh_{device.lower()}_launch_names_the_build",
            want_bin in script and want_dir in script,
            {"want_bin": want_bin, "want_dir": want_dir, "script": script[:400],
             "error": error},
        )

    # ---- D: Docker composes the same script through the same path ----
    backend = _Recorder("docker", "CPU", "AUTO")
    script, error = launch(runner, backend, both)
    record(
        "D_docker_cpu_launch_names_the_build",
        f"{ROOT}/target/cpu/release/ppf-cts-server" in script
        and f'export CARGO_TARGET_DIR="{ROOT}/target/cpu"' in script,
        {"script": script[:400], "error": error},
    )

    # ---- E: a DISTRIBUTION layout is reachable ----
    #
    # An unpacked Linux release has no `target/release` at all, so a launch
    # that named only that path could reach nothing in one.
    distribution = listing(
        ("target/cuda/release", "cuda"), ("target/cpu/release", "cpu")
    )
    backend = _Recorder("ssh", "GPU", "AUTO")
    script, error = launch(runner, backend, distribution)
    record(
        "E_a_distribution_layout_launches",
        f"{ROOT}/target/cuda/release/ppf-cts-server" in script,
        {"script": script[:400], "error": error},
    )
    record(
        "E_a_distribution_root_validates",
        conn.remote_holds_any_server(ROOT, distribution) is True
        and conn.remote_holds_any_server(ROOT, {}) is False,
        {},
    )

    # ---- F: a selection the host cannot serve is REFUSED, by name ----
    #
    # Refused rather than substituted, and the refusal says which build IS
    # there, so the artist can act on it without changing the path.
    gpu_only = listing(("target/cuda/release", "cuda"))
    backend = _Recorder("ssh", "CPU", "AUTO")
    script, error = launch(runner, backend, gpu_only)
    record(
        "F_an_unservable_device_is_refused_and_launches_nothing",
        script == "" and "GPU build" in error and "CPU" in error,
        {"error": error, "script": script[:200]},
    )
    backend = _Recorder("ssh", "GPU", "ROCM")
    script, error = launch(runner, backend, gpu_only)
    record(
        "F_an_unservable_accelerator_is_refused_by_name",
        script == "" and "rocm" in error.lower() and "cuda" in error.lower(),
        {"error": error, "script": script[:200]},
    )

    # ---- G: the panel's choice reaches the backend that will launch ----
    #
    # The two halves above meet here: `create_backend` is what carries the
    # selection from the connection onto the object `_do_launch_server` reads
    # it back off, and a backend that dropped it would leave every check above
    # passing while the artist's choice reached nothing.
    # Asserted on the classes rather than through `create_backend`, which opens
    # a real paramiko or Docker client: what is in question is whether the
    # selection is HELD, and the factory's own two lines that pass it are read
    # by `bl_docker_connect_gate`, which already stands a container in.
    ssh_backend = backends_mod.SSHBackend(
        instance=None, directory=ROOT, port=9090, device="CPU", gpu_backend="ROCM"
    )
    docker_backend = backends_mod.DockerBackend(
        instance=None, directory=ROOT, port=9090, container="c",
        device="CPU", gpu_backend="ROCM",
    )
    record(
        "G_both_remote_backends_hold_the_selection",
        ssh_backend._device == "CPU" and ssh_backend._gpu_backend == "ROCM"
        and docker_backend._device == "CPU"
        and docker_backend._gpu_backend == "ROCM",
        {},
    )
    record(
        "G_the_default_is_gpu_so_saved_files_keep_their_behavior",
        backends_mod.SSHBackend(
            instance=None, directory=ROOT, port=9090
        )._device == "GPU",
        {},
    )

    # ---- H: the choice made AFTER connecting is the one that launches ----
    #
    # THE REMOTE ROWS ARE DRAWN ONLY ONCE A CONNECTION IS UP, because before
    # that there is no solver host to ask what it holds. So the artist's answer
    # arrives at Start Server, after the connection was made. A launch that
    # read only the backend's connect-time copy would leave those rows movable
    # and inert: the panel would say CPU and every solve would run the GPU
    # build, with nothing reporting it. That is the silent substitution the
    # whole device mechanism exists to prevent, arriving through the UI rather
    # than through a path.
    connected_as_gpu = _Recorder("ssh", "GPU", "AUTO")
    script, error = launch(runner, connected_as_gpu, both, device="CPU")
    record(
        "H_the_device_given_at_start_server_wins_over_the_one_from_connect",
        f"{ROOT}/target/cpu/release/ppf-cts-server" in script
        and f'export CARGO_TARGET_DIR="{ROOT}/target/cpu"' in script,
        {"script": script[:400], "error": error},
    )
    # And the same for the accelerator.
    connected_as_auto = _Recorder("ssh", "GPU", "AUTO")
    script, error = launch(
        runner, connected_as_auto, three, device="GPU", gpu_backend="ROCM"
    )
    record(
        "H_the_accelerator_given_at_start_server_wins_too",
        f"{ROOT}/target/rocm/release/ppf-cts-server" in script,
        {"script": script[:400], "error": error},
    )
    # Naming neither keeps what the connection was made with, which is what
    # every caller that passes no device gets.
    connected_as_cpu = _Recorder("ssh", "CPU", "AUTO")
    script, error = launch(runner, connected_as_cpu, both)
    record(
        "H_naming_no_device_keeps_the_connection_s_own",
        f"{ROOT}/target/cpu/release/ppf-cts-server" in script,
        {"script": script[:400], "error": error},
    )

    # ---- I: THE PANEL RESOLVES AGAINST THE ROOT THE PROBE USED, on EVERY
    #      remote transport, including a pure SSH connection.
    #
    # THIS IS THE CHECK THE REST OF THIS SCENARIO COULD NOT MAKE, and its
    # absence shipped a defect. Every check above drives the LAUNCH, which
    # resolves from `backend.current_directory`; the probe resolves from the
    # same attribute, so the two cannot disagree however hard they are tested.
    # The PANEL was a third resolver, and it derived its own root from
    # `communicator.normalized_remote_root()`, which answers the DATA root
    # (`<share>/ppf-cts/git-<branch>/<project>`) and not the solver root. Every
    # lookup then missed, which is indistinguishable from a host holding no
    # build, so the rows went dead and the panel said "No solver build found
    # under <data root>" while the listing held both. Measured on a real
    # Docker-over-SSH connection to a container that had both.
    #
    # THE DATA ROOT IS DELIBERATELY DIFFERENT FROM THE SOLVER ROOT HERE. If
    # they were the same the check would pass against the defect, which is
    # exactly how this went unnoticed: on a tree where the connection happens to
    # name the checkout, the two spellings coincide.
    #
    # IT RUNS FOR EVERY REMOTE TYPE because the probe does: the resolution is
    # transport-independent, so a regression would hit all five at once, and
    # naming them one by one is what says which.
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["_draw_remote_device"])
    conn_ops = __import__(pkg + ".ui.connection_ops",
                          fromlist=["REMOTE_SERVER_TYPES"])
    groups_mod = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    class _PanelLayout:
        # Records what the draw emitted. A real UILayout cannot be built
        # outside a draw callback, and every method here returns self so a
        # chain of column()/row() lands in one record.
        def __init__(self):
            self.labels = []
            self.props = []
            self.enabled = True

        def column(self, *a, **k):
            return self

        def row(self, *a, **k):
            return self

        def label(self, text="", icon="", **k):
            self.labels.append((text, icon))

        def prop(self, data, name, **k):
            self.props.append(name)

    class _PanelCom:
        # Connected, nothing running, and a DATA root that is NOT the solver
        # root, which is the whole point of the check.
        def __init__(self, data_root):
            self._data_root = data_root

        def is_connected(self):
            return True

        def is_server_running(self):
            return False

        def is_server_launching(self):
            return False

        def normalized_remote_root(self):
            return self._data_root

    DATA_ROOT = "/home/u/.local/share/ppf-cts/git-unknown/proj"
    panel_props = groups_mod.get_addon_data(bpy.context.scene).ssh_state
    saved_com = main_panel.com
    saved_type = panel_props.server_type
    saved_device = panel_props.native_device
    try:
        main_panel.com = _PanelCom(DATA_ROOT)
        for server_type in conn_ops.REMOTE_SERVER_TYPES:
            remote_builds.load_builds(
                [f"{m}\t{d}" for d, m in both.items()], ROOT
            )
            panel_props.server_type = server_type
            panel_props.native_device = "GPU"
            lay = _PanelLayout()
            main_panel._draw_remote_device(lay, panel_props)
            texts = [t for t, _ in lay.labels]
            found_nothing = any("No solver build found" in t for t in texts)
            record(
                f"I_the_panel_finds_both_builds_on_{server_type.lower()}",
                not found_nothing and "native_device" in lay.props,
                {
                    "server_type": server_type,
                    "solver_root": ROOT,
                    "data_root": DATA_ROOT,
                    "labels": lay.labels,
                    "props": lay.props,
                },
            )
        # And the converse, so the check above cannot pass by never drawing the
        # message at all: a root the listing holds nothing under MUST say so.
        remote_builds.load_builds([], ROOT)
        panel_props.server_type = "CUSTOM"
        lay = _PanelLayout()
        main_panel._draw_remote_device(lay, panel_props)
        record(
            "I_an_empty_listing_still_reports_no_build_found",
            any("No solver build found" in t for t, _ in lay.labels),
            {"labels": lay.labels},
        )
    finally:
        main_panel.com = saved_com
        panel_props.server_type = saved_type
        panel_props.native_device = saved_device

    # ---- J: a Docker-over-SSH launch refuses a port that is not published ----
    #
    # The server is reached through a port PUBLISHED ON THE DOCKER HOST, so
    # `docker port <container> <port>` has to answer before a launch is worth
    # attempting. Without the gate the launch proceeds, the server comes up
    # inside the container, and nothing outside can reach it: the artist sees a
    # connection that never comes up rather than the one fact that explains it,
    # a container started without `-p`. Measured on a real container started
    # that way.
    #
    # IT IS CHECKED HERE RATHER THAN IN A HOST TEST because the guard sits
    # inside `_do_launch_server`, and this scenario is the only place that
    # drives that function for real.
    class _PortChannel:
        def __init__(self, code):
            self._code = code

        def recv_exit_status(self):
            return self._code

    class _PortStream:
        def __init__(self, code):
            self.channel = _PortChannel(code)

        def read(self):
            return b""

    class _PortInstance:
        """The paramiko handle the guard reaches for, answering as asked."""

        def __init__(self, code):
            self._code = code
            self.asked = []

        def exec_command(self, command, timeout=None):
            self.asked.append(command)
            return None, _PortStream(self._code), _PortStream(0)

    def _launch_with_port_answer(code):
        be = _Recorder("ssh", "GPU", "AUTO")
        be._container = "ppf-test"
        be._instance = _PortInstance(code)
        script, error = launch(runner, be, both)
        return be, script, error

    be_bad, _script, error = _launch_with_port_answer(1)
    record(
        "J_an_unpublished_port_stops_the_launch_by_name",
        "is not exposed on container" in error and "-p" in error,
        {"error": error, "asked": be_bad._instance.asked},
    )
    record(
        "J_the_gate_asks_docker_port_on_the_ssh_host",
        any(c.startswith("docker port ppf-test ") for c in be_bad._instance.asked),
        {"asked": be_bad._instance.asked},
    )
    # The converse, so the check above cannot pass by refusing everything: a
    # published port must let the launch through to composing its script.
    be_ok, script, error = _launch_with_port_answer(0)
    record(
        "J_a_published_port_lets_the_launch_proceed",
        # `both` holds the DISTRIBUTION layout, so GPU is target/cuda/release.
        # Naming target/release here would assert the checkout layout this
        # listing does not have, and pass or fail for the wrong reason.
        "is not exposed on container" not in error
        and f"{ROOT}/target/cuda/release/ppf-cts-server" in script,
        {"error": error, "script": script[:200]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    try:
        remote_builds.forget_builds()
    except Exception:
        pass
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions: every path here is a fabricated remote one, and nothing
    is opened, executed or connected to.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
