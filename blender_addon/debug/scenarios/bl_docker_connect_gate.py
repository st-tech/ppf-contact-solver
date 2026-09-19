# File: scenarios/bl_docker_connect_gate.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Local Docker: the Connect button's gate, the shipped defaults, and the
# refusals the user reads.
#
# The Docker backend reaches the daemon over the Docker socket. No SSH key
# takes part in it, and the panel does not draw the SSH Key field in Docker
# mode. Gating the button on that field therefore made Connect unpressable
# for a reason nothing on screen could explain: the field defaults to a path
# under the user's home directory, so a Windows account whose name holds a
# space put a space in it and the shell-safety test rejected it. That is the
# community report "the addon doesn't seem to want to connect to the docker
# server", where the button simply never enables.
#
# The other half of the same report is what the shipped defaults name. The
# project's own public instructions create a container called
# ``ppf-contact-solver`` and an image that builds the solver at
# ``/root/ppf-contact-solver``, so those are what the Container and Container
# Path fields default to and what this scenario pins: a default naming
# anything else is a default no reader of those instructions has, and the
# container path in particular can be read only from inside the container.
#
# Checks:
#
#   * REMOTE_OT_Connect.poll enables Docker with an SSH key path holding a
#     space, and with the SSH key field cleared entirely.
#   * it still refuses a blank container name and a container path holding a
#     shell metacharacter, which is what the Docker exec path interpolates.
#   * the panel's Docker branch draws the container / container path / port
#     fields and does NOT draw the SSH key field, so the gate matches the UI.
#   * the shipped container default is the name the project's public
#     ``docker run`` creates, Container Path defaults to where the published
#     image puts the solver, and Container Path carries a description that
#     says it is a path inside the container.
#   * a container that publishes no host port is refused by name, with the
#     ``-p`` flag that fixes it, instead of connecting and failing later on
#     every query.
#
# ``module_exists(["docker"])`` is one conjunct of the real poll, and the rig's
# Blender has no docker-py installed, so the scenario substitutes that one
# predicate for the duration and restores it. Nothing here contacts a Docker
# daemon: the published-port check is exercised against a stand-in carrying
# the same ``attrs`` shape docker-py returns.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure UI-gate and defaults logic; no solver is involved.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _FakeLayout:
    """Records prop() and label() calls so the panel's Docker branch can be
    inspected without a real Blender UILayout."""

    def __init__(self):
        self.props = []
        self.labels = []

    def prop(self, _data, name, **kw):
        self.props.append(name)

    def label(self, text="", icon="", **kw):
        self.labels.append((text, icon))

    def row(self, **kw):
        return self

    def column(self, **kw):
        return self

    def box(self):
        return self

    def operator(self, *a, **kw):
        return self


class _StubContainer:
    """Carries the ``attrs`` shape docker-py fills from the daemon."""

    def __init__(self, ports, network_mode="bridge"):
        self.attrs = {
            "NetworkSettings": {"Ports": ports},
            "HostConfig": {"NetworkMode": network_mode},
        }


conn_ops = None
saved_module_exists = None
try:
    conn_ops = __import__(pkg + ".ui.connection_ops",
                          fromlist=["REMOTE_OT_Connect"])
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["MAIN_PT_RemotePanel"])
    backends = __import__(pkg + ".core.backends", fromlist=["create_backend"])
    state_mod = __import__(pkg + ".ui.state", fromlist=["SSHState"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "docker_gate"
    props = root.ssh_state

    # The real poll asks whether docker-py is importable. The rig's Blender
    # has no docker-py, and that conjunct is not what this scenario is about,
    # so stand it in for the duration and put the real one back at the end.
    saved_module_exists = conn_ops.module_exists
    conn_ops.module_exists = lambda packages: True

    props.server_type = "DOCKER"
    props.container = "ppf-contact-solver"
    props.docker_path = "/root/ppf-contact-solver"

    # ---- the gate does not depend on a field Docker never uses ----
    props.key_path = "C:\\Users\\John Smith\\.ssh\\id_rsa"
    record("docker_enabled_with_space_in_ssh_key_path",
           bool(bpy.ops.ssh.run_command.poll()) is True,
           {"key_path": props.key_path})

    props.key_path = ""
    record("docker_enabled_with_ssh_key_cleared",
           bool(bpy.ops.ssh.run_command.poll()) is True,
           {"key_path": props.key_path})

    props.key_path = "/home/artist/.ssh/id_ed25519"
    record("docker_enabled_with_ordinary_ssh_key",
           bool(bpy.ops.ssh.run_command.poll()) is True,
           {"key_path": props.key_path})

    # ---- the gate still refuses what Docker genuinely cannot use ----
    props.container = "   "
    record("docker_refuses_blank_container",
           bool(bpy.ops.ssh.run_command.poll()) is False,
           {"container": repr(props.container)})
    props.container = "ppf-contact-solver"

    # The container path is interpolated into the commands the add-on runs
    # inside the container, so a shell metacharacter there stays refused.
    props.docker_path = "/root/ppf;rm -rf /"
    record("docker_refuses_shell_unsafe_container_path",
           bool(bpy.ops.ssh.run_command.poll()) is False,
           {"docker_path": props.docker_path})
    props.docker_path = "/root/ppf-contact-solver"

    # ---- the panel draws the fields the gate reads, and no others ----
    # A Blender Panel cannot be instantiated from Python
    # (``bpy_struct.__new__`` demands its own argument), but ``draw`` is an
    # ordinary function on the class, so calling it unbound with a stand-in
    # carrying ``layout`` exercises the real branch.
    class _PanelSelf:
        def __init__(self, layout):
            self.layout = layout

    fl = _FakeLayout()
    root.state.show_connection = True
    main_panel.MAIN_PT_RemotePanel.draw(_PanelSelf(fl), bpy.context)
    record("panel_docker_draws_container", "container" in fl.props,
           {"props": fl.props})
    record("panel_docker_draws_container_path", "docker_path" in fl.props,
           {"props": fl.props})
    record("panel_docker_draws_port", "docker_port" in fl.props,
           {"props": fl.props})
    record("panel_docker_does_not_draw_ssh_key", "key_path" not in fl.props,
           {"props": fl.props})

    # ---- the shipped defaults name what the public instructions create ----
    defaults = state_mod.SSHState.bl_rna.properties
    container_default = defaults["container"].default
    record("container_default_matches_public_docker_run",
           container_default == "ppf-contact-solver",
           {"default": container_default})
    docker_path_default = defaults["docker_path"].default
    record("container_path_default_matches_published_image",
           docker_path_default == "/root/ppf-contact-solver",
           {"default": docker_path_default})
    record("container_path_has_a_description",
           "container" in (defaults["docker_path"].description or "").lower(),
           {"description": defaults["docker_path"].description})

    # ---- a container that publishes nothing is refused by name ----
    require = backends._require_published_port
    published = _StubContainer({"9090/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9090"}]})
    try:
        require(published, "ppf-contact-solver", 9090)
        record("published_port_accepted", True, {})
    except Exception as exc:
        record("published_port_accepted", False, {"error": str(exc)})

    unpublished = _StubContainer({"9090/tcp": None})
    try:
        require(unpublished, "ppf-contact-solver", 9090)
        record("unpublished_port_refused", False, {"error": "no refusal"})
    except Exception as exc:
        text = str(exc)
        record("unpublished_port_refused",
               "ppf-contact-solver" in text and "-p 9090:9090" in text,
               {"error": text})

    # ---- exec keeps the container's two streams apart ----
    # Muxed together, a diagnostic written to stderr is swallowed by the
    # caller reading stdout, and a script's partial output is discarded on a
    # non-zero exit, which left a container-side failure during Start Server
    # reading as a bare 16 s timeout with nothing in it.
    class _StubExecContainer:
        def __init__(self, code, out, err):
            self._r = (code, (out, err))
            self.calls = []

        def exec_run(self, command, **kw):
            self.calls.append((command, kw))
            return self._r

    DockerBackend = backends.DockerBackend
    failing = _StubExecContainer(3, b"got as far as here\n", b"boom: no such file\n")
    be = DockerBackend(failing, "/root/ppf-contact-solver", 9090, "ppf-contact-solver")
    res = be.exec_command("/bin/false", shell=True)
    record("exec_demuxes_streams",
           failing.calls and failing.calls[0][1].get("demux") is True,
           {"kwargs": str(failing.calls[0][1]) if failing.calls else None})
    record("exec_keeps_stdout_on_failure",
           res["stdout"] == ["got as far as here"], {"got": res["stdout"]})
    record("exec_reports_stderr_separately",
           res["stderr"] == ["boom: no such file"], {"got": res["stderr"]})
    record("exec_reports_the_exit_code", res["exit_code"] == 3,
           {"got": res["exit_code"]})

    ok = _StubExecContainer(0, b"SERVER_READY\n", b"")
    be_ok = DockerBackend(ok, "/root/ppf-contact-solver", 9090, "ppf-contact-solver")
    res_ok = be_ok.exec_command("cat progress.log", shell=True)
    record("exec_success_keeps_stderr_empty",
           res_ok["stdout"] == ["SERVER_READY"] and res_ok["stderr"] == [],
           {"stdout": res_ok["stdout"], "stderr": res_ok["stderr"]})

    # Host networking publishes nothing and needs nothing, so it is accepted.
    host_net = _StubContainer({}, network_mode="host")
    try:
        require(host_net, "ppf-contact-solver", 9090)
        record("host_network_accepted", True, {})
    except Exception as exc:
        record("host_network_accepted", False, {"error": str(exc)})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if conn_ops is not None and saved_module_exists is not None:
        conn_ops.module_exists = saved_module_exists
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the scenario reads the shipped property
    defaults, exercises the Connect operator's poll and the panel's draw
    against a recording layout, and checks the published-port refusal against
    a stand-in carrying docker-py's ``attrs`` shape. It never reaches a Docker
    daemon, so it runs on any host.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
