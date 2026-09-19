# File: addon_host_tests/_native_device_selection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gates for the GPU/CPU choice on the two LOCAL NATIVE backends
# (``blender_addon/core/connection.py``).
#
# THE BACKEND IS A PROPERTY OF A BUILD, NOT A RUNTIME FLAG. The solver links
# one backend, and ``crates/ppf-cts-solver/build.rs`` refuses to put a second
# one in a directory that already holds another, so choosing GPU or CPU is
# choosing WHICH DIRECTORY the server comes from. These tests are therefore
# about path resolution and nothing else, which is why they run on a plain
# interpreter: no Blender data is read and no server is started.
#
# WHAT WOULD GO WRONG WITHOUT THEM, and it is the failure this whole mechanism
# exists to prevent: a caller that asked for CPU and silently got the GPU
# server. Both are real solvers and both answer ``--backend`` honestly, so
# neither reports the substitution; the only symptom is that the run is
# roughly 30x faster or slower than the artist asked for. A resolver that
# falls back on an unknown device produces exactly that, which is why the
# no-fallback case below is asserted rather than assumed.

from __future__ import annotations

import pytest



# THE THREE NATIVE BACKENDS, as `connection.NATIVE_BACKENDS` names them. Every
# case below runs against all three: what they resolve is one rule, and a
# platform that answered differently would be the drift this file exists to
# catch.
NATIVES = ("win_native", "mac_native", "linux_native")


def _exe(native: str) -> str:
    """The server's file name for *native*."""
    return "ppf-cts-server.exe" if native == "win_native" else "ppf-cts-server"


def _resolver(conn, native):
    return getattr(conn, f"{native}_server_binary")


@pytest.fixture(scope="module")
def conn():
    """``blender_addon.core.connection``, loaded from source.

    It reaches `.gpu_devices` and `.status` and nothing that touches Blender
    data, so the conftest's stub `bpy` is enough. The package has to exist in
    `sys.modules` first, exactly as the other fixtures here arrange it.
    """
    from conftest import ADDON_ROOT, _ensure_package, load_addon_module

    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.connection")


def _root(tmp_path, *, gpu: bool, cpu: bool, native: str):
    """A solver root holding the requested builds, in the shipped layout.

    NO BACKEND MARKER IS WRITTEN, deliberately: this is the DISTRIBUTED
    bundle's shape, where `bundle.sh` copies the binaries and not the marker,
    and it is what every resolution below the marker tests exercises. A root
    that carries one is built by ``_marked`` instead.
    """
    exe = _exe(native)
    if gpu:
        d = tmp_path / "target" / "release"
        d.mkdir(parents=True, exist_ok=True)
        (d / exe).write_text("")
    if cpu:
        d = tmp_path / "target" / "cpu" / "release"
        d.mkdir(parents=True, exist_ok=True)
        (d / exe).write_text("")
    return str(tmp_path)


def _marked(tmp_path, *parts, backend: str, native: str):
    """A solver root whose ``<parts>`` directory records *backend*.

    ``crates/ppf-cts-solver/build.rs`` writes this beside the artifacts of
    every build it links, so a root produced by cargo always has one and this
    is the shape the panel meets on a developer's machine.
    """
    exe = _exe(native)
    d = tmp_path.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / exe).write_text("")
    (d / ".ppf-backend").write_text(backend)
    return str(tmp_path)


@pytest.mark.parametrize("native", NATIVES)
def test_each_device_resolves_its_own_build(conn, tmp_path, native):
    """GPU and CPU resolve to DIFFERENT directories under one root.

    This is the property the whole feature rests on. If the two resolved to
    the same path the selector would be decorative, and the artist would get
    whichever backend happened to be built last.
    """
    root = _root(tmp_path, gpu=True, cpu=True, native=native)
    resolve = _resolver(conn, native)
    gpu = resolve(root, conn.DEVICE_GPU)
    cpu = resolve(root, conn.DEVICE_CPU)
    assert gpu is not None and cpu is not None
    assert gpu != cpu, "GPU and CPU resolved to one path, so the choice does nothing"
    assert "cpu" in cpu.replace("\\", "/").split("/"), cpu
    assert "cpu" not in gpu.replace("\\", "/").split("/"), gpu


@pytest.mark.parametrize("native", NATIVES)
def test_the_default_is_gpu_so_saved_files_keep_their_behavior(conn, tmp_path, native):
    """Called with no device, a root resolves exactly as it always did.

    Every ``.blend`` saved before the device property existed carries no
    answer, and the property reads GPU for them. A default that resolved
    anywhere else would silently move those files onto another backend.
    """
    root = _root(tmp_path, gpu=True, cpu=True, native=native)
    resolve = _resolver(conn, native)
    assert resolve(root) == resolve(root, conn.DEVICE_GPU)


@pytest.mark.parametrize("native", NATIVES)
def test_a_missing_build_is_absent_rather_than_substituted(conn, tmp_path, native):
    """Asking for a device this root does not hold yields None.

    NOT the other device. The panel reads this to decide whether to offer the
    choice at all, and a fallback here would let it advertise a mode that the
    spawn would then satisfy with the wrong binary.
    """
    resolve = _resolver(conn, native)
    gpu_only = _root(tmp_path / "gpu_only", gpu=True, cpu=False, native=native)
    assert resolve(gpu_only, conn.DEVICE_CPU) is None
    assert resolve(gpu_only, conn.DEVICE_GPU) is not None
    cpu_only = _root(tmp_path / "cpu_only", gpu=False, cpu=True, native=native)
    assert resolve(cpu_only, conn.DEVICE_GPU) is None
    assert resolve(cpu_only, conn.DEVICE_CPU) is not None


@pytest.mark.parametrize("native", NATIVES)
def test_an_unknown_device_never_falls_back(conn, tmp_path, native):
    """An unrecognized device resolves to nothing, not to the GPU layout.

    A fallback would be the silent substitution described at the top of this
    file, arriving through a typo rather than through a missing build.
    """
    root = _root(tmp_path, gpu=True, cpu=True, native=native)
    resolve = _resolver(conn, native)
    assert resolve(root, "TPU") is None
    assert resolve(root, "") is None


def test_both_platforms_offer_the_same_device_set(conn):
    """The two natives cannot drift into supporting different DEVICES.

    Their LAYOUTS legitimately differ: Windows accepts a ``bin/`` bundle
    beside a checkout, and macOS and Linux accept only the checkout rows, which
    ``bl_mac_native_root_resolve`` asserts by requiring a ``bin/``-only root
    to resolve to nothing there. An earlier version of this file merged the
    tables and turned that scenario red, so what is asserted here is the
    DEVICE dimension only.
    """
    for table in (
        conn._WIN_DEVICE_SUBDIRS,
        conn._MAC_DEVICE_SUBDIRS,
        conn._LINUX_DEVICE_SUBDIRS,
    ):
        assert set(table) == {conn.DEVICE_GPU, conn.DEVICE_CPU}


def test_macos_does_not_accept_the_windows_bundle_layout(conn, tmp_path):
    """macOS resolves a ``bin/``-only root to nothing, for either device.

    This is the property the merged table broke. It is asserted here as well
    as in the rig because this tier runs on every push and the rig does not.
    """
    for sub in ("bin", "bin-cpu"):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppf-cts-server").write_text("")
    root = str(tmp_path)
    assert conn.mac_native_server_binary(root, conn.DEVICE_GPU) is None
    assert conn.mac_native_server_binary(root, conn.DEVICE_CPU) is None


# ---------------------------------------------------------------------------
# The backend marker
#
# A PATH IS A CONVENTION AND THE MARKER IS EVIDENCE. Every backend links the
# same executable name, so the directory a server sits in cannot say which
# backend is in it; `build.rs` writes `.ppf-backend` beside its artifacts
# precisely because the name cannot carry that. These tests are the addon
# reading it, which is what `frontend.solver_dir` already did for the same
# question.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native", NATIVES)
@pytest.mark.parametrize("backend", ["cuda", "metal", "rocm"])
def test_a_gpu_marker_in_the_gpu_layout_resolves_as_gpu(conn, tmp_path, native, backend):
    """The ordinary developer tree, and it must not change.

    Every accelerated backend NAME answers to the one device the artist picks:
    which accelerator a host has is the host's business, so `cuda`, `metal` and
    `rocm` are the same answer here.
    """
    root = _marked(tmp_path, "target", "release", backend=backend, native=native)
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_GPU) is not None
    assert resolve(root, conn.DEVICE_CPU) is None


@pytest.mark.parametrize("native", NATIVES)
def test_a_cpu_build_in_the_gpu_layout_is_cpu(conn, tmp_path, native):
    """THE BUG THIS PINS, and it is the command the panel used to prescribe.

    `cargo build --release --features cpu` links the CPU backend into
    `target/release` whenever that directory is free to take it, which
    `build.rs`'s own banner says in as many words ("it overwrites the solver at
    target/release/"). Reading the layout alone, the panel then offered GPU and
    would have spawned the CPU solver, roughly 30x slower with nothing
    reporting the swap, while telling the artist in the same panel that no CPU
    build existed and to run the command that had just produced one.
    """
    root = _marked(tmp_path, "target", "release", backend="cpu", native=native)
    resolve = _resolver(conn, native)
    cpu = resolve(root, conn.DEVICE_CPU)
    assert cpu is not None, "a CPU build went unseen because of where it sits"
    assert resolve(root, conn.DEVICE_GPU) is None, (
        "a CPU build was offered as GPU, which is the silent substitution "
        "this whole mechanism exists to prevent"
    )


@pytest.mark.parametrize("native", NATIVES)
def test_a_marker_never_promotes_the_other_device_for_free(conn, tmp_path, native):
    """A GPU-marked tree still holds no CPU build.

    The resolver visits every device's layout so a marker can override the
    row it was found by; that widening must not turn "found something else"
    into "found what you asked for".
    """
    root = _marked(tmp_path, "target", "cpu", "release", backend="metal", native=native)
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_CPU) is None
    assert resolve(root, conn.DEVICE_GPU) is not None


@pytest.mark.parametrize("native", NATIVES)
def test_an_unmarked_directory_answers_only_for_its_own_layout(conn, tmp_path, native):
    """A bundle carries no marker, so the layout is all there is.

    Without a marker there is nothing to override the row with, so an unmarked
    directory found on ANOTHER device's behalf is refused. Admitting it would
    reinstate, as a bug, exactly the fallback
    `test_a_missing_build_is_absent_rather_than_substituted` forbids.
    """
    root = _root(tmp_path, gpu=True, cpu=False, native=native)
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_CPU) is None
    assert resolve(root, conn.DEVICE_GPU) is not None


@pytest.mark.parametrize("native", NATIVES)
def test_an_unreadable_or_empty_marker_falls_back_to_the_layout(conn, tmp_path, native):
    """A marker that says nothing is not a marker.

    An empty file is what a half-written one looks like, and a root is not
    worth refusing over it: the layout is the answer that shipped before
    markers were read at all.
    """
    root = _root(tmp_path, gpu=True, cpu=True, native=native)
    for parts in (("target", "release"), ("target", "cpu", "release")):
        (tmp_path.joinpath(*parts) / ".ppf-backend").write_text("   \n")
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_GPU) is not None
    assert resolve(root, conn.DEVICE_CPU) is not None
    assert resolve(root, conn.DEVICE_GPU) != resolve(root, conn.DEVICE_CPU)


@pytest.mark.parametrize("native", NATIVES)
def test_an_unknown_backend_name_is_not_accelerated(conn, tmp_path, native):
    """A name this addon has never heard of answers to CPU, not to GPU.

    The GPU list is a CLOSED set of accelerator names, so a backend added to
    `build.rs` and not here reads as the unaccelerated one. That direction is
    the safe one: the artist is told their build is slow when it is fast,
    rather than being handed a 30x slowdown labeled GPU.
    """
    root = _marked(tmp_path, "target", "release", backend="vulkan", native=native)
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_GPU) is None
    assert resolve(root, conn.DEVICE_CPU) is not None


def test_a_marker_does_not_give_macos_the_windows_bundle_layout(conn, tmp_path):
    """The layout tables still bound WHERE macOS looks.

    Widening the search to every device's rows widened it within one platform's
    table only. A `bin/` root stays invisible on macOS however it is marked,
    which is what `bl_mac_native_root_resolve` asserts.
    """
    for sub in ("bin", "bin-cpu"):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppf-cts-server").write_text("")
        (d / ".ppf-backend").write_text("metal")
    root = str(tmp_path)
    assert conn.mac_native_server_binary(root, conn.DEVICE_GPU) is None
    assert conn.mac_native_server_binary(root, conn.DEVICE_CPU) is None


@pytest.mark.parametrize("native", NATIVES)
def test_a_cpu_only_tree_is_a_valid_solver_root(conn, tmp_path, native):
    """The root walk asks a DIFFERENT question from the device probe.

    "Does this folder hold the solver" is not "does it hold the backend I
    picked", and the path resolver asks only the first. A tree built with
    `--features cpu` and nothing else is a solver root; resolving it with a
    GPU-shaped probe walked past it and told the artist the folder held no
    solver, which is both wrong and unactionable.
    """
    resolve_root = getattr(conn, f"resolve_{native}_root")
    marked = _marked(tmp_path, "target", "release", backend="cpu", native=native)
    assert resolve_root(marked) == marked
    # And from a subdirectory, which is the selection the walk exists for.
    assert resolve_root(str(tmp_path / "target" / "release")) == marked


@pytest.mark.parametrize("native", NATIVES)
def test_a_second_target_directory_is_a_valid_solver_root(conn, tmp_path, native):
    """A root whose only build sits in the sanctioned second target directory.

    `build.rs` prescribes `CARGO_TARGET_DIR=target/cpu` for holding two
    backends at once, and it is equally what a tree holding only that one
    looks like.
    """
    resolve_root = getattr(conn, f"resolve_{native}_root")
    root = _root(tmp_path, gpu=False, cpu=True, native=native)
    assert resolve_root(root) == root


def test_the_root_walk_still_refuses_the_windows_bundle_layout_on_macos(conn, tmp_path):
    """Widening the walk to every device widened it within ONE table.

    macOS accepts only the checkout layout, so no `bin/` row exists for it to
    find, whichever device is being considered.
    """
    for sub in ("bin", "bin-cpu"):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppf-cts-server").write_text("")
    assert conn.resolve_mac_native_root(str(tmp_path)) is None


@pytest.mark.parametrize("native", NATIVES)
def test_a_marker_written_by_cmd_exe_reads_the_same(conn, tmp_path, native):
    """The Windows bundle writes its marker with `echo cpu>file`, which is CRLF.

    Measured on Windows Server 2025: the file is exactly ``b'cpu\\r\\n'``. The
    add-on runs on that platform too, so the reader has to survive it. This is
    also why the marker is written with no space before the redirect: `echo cpu
    >file` would put the space INSIDE the name.
    """
    exe = _exe(native)
    d = tmp_path / "target" / "release"
    d.mkdir(parents=True, exist_ok=True)
    (d / exe).write_text("")
    (d / ".ppf-backend").write_bytes(b"cpu\r\n")
    root = str(tmp_path)
    resolve = _resolver(conn, native)
    assert resolve(root, conn.DEVICE_CPU) is not None
    assert resolve(root, conn.DEVICE_GPU) is None


# ---------------------------------------------------------------------------
# A server that is already running
#
# DISCONNECT LEAVES A NATIVE SERVER RUNNING, on purpose, and Connect and Start
# Server then attach to whatever answers on the port. The device selection
# chooses which server to LAUNCH, so on the attach path it reached nothing: a
# CPU selection connected to a GPU server left by an earlier connection drove
# every solve on the GPU build. The server now reports the target directory
# its runs use, and these tests hold the add-on to refusing a mismatch.
# ---------------------------------------------------------------------------


def _both_builds(tmp_path, native):
    """A root holding a marked GPU build and a marked CPU build."""
    _marked(tmp_path, "target", "release", backend="cuda", native=native)
    _marked(tmp_path, "target", "cpu", "release", backend="cpu", native=native)
    return str(tmp_path)


def _response(conn, target_dir, backend, *, version=None):
    """What a server running *target_dir*'s build answers to a status query."""
    return {
        "protocol_version": conn.PROTOCOL_VERSION if version is None else version,
        "solver_target_dir": str(target_dir),
        "solver_backend": backend,
    }


def _resolver(conn, native):
    return getattr(conn, f"{native}_server_binary")


@pytest.mark.parametrize("native", NATIVES)
def test_a_server_running_the_selected_build_is_attached(conn, tmp_path, native):
    """The ordinary reconnect: the server is the build the panel names."""
    root = _both_builds(tmp_path, native)
    resolve = _resolver(conn, native)
    conn.check_running_server(
        _response(conn, tmp_path / "target", "cuda"),
        root, 9090, conn.DEVICE_GPU, resolve,
    )
    conn.check_running_server(
        _response(conn, tmp_path / "target" / "cpu", "cpu"),
        root, 9090, conn.DEVICE_CPU, resolve,
    )


@pytest.mark.parametrize("native", NATIVES)
@pytest.mark.parametrize(
    "selected, running, backend, other",
    [
        ("CPU", ("target",), "cuda", "GPU"),
        ("GPU", ("target", "cpu"), "cpu", "CPU"),
    ],
    ids=["cpu_selected_gpu_running", "gpu_selected_cpu_running"],
)
def test_a_server_running_the_other_build_is_refused(
    conn, tmp_path, native, selected, running, backend, other
):
    """THE BUG: the selection must not be satisfied by the other build.

    The message names both directories, and the way out that works from
    inside the add-on: select the device the running server IS, connect to
    it, and stop it.
    """
    root = _both_builds(tmp_path, native)
    running_dir = tmp_path.joinpath(*running)
    with pytest.raises(conn.NativeServerMismatch) as caught:
        conn.check_running_server(
            _response(conn, running_dir, backend),
            root, 9090, selected, _resolver(conn, native),
        )
    message = str(caught.value)
    assert str(running_dir) in message, message
    assert f"Compute Device is set to {selected}" in message, message
    assert f"set Compute Device to {other}" in message, message


@pytest.mark.parametrize("native", NATIVES)
def test_a_server_from_another_tree_is_refused_for_the_same_device(conn, tmp_path, native):
    """A backend NAME would pass this; the directory does not.

    A GPU server built in some other checkout answers "cuda" exactly like this
    root's GPU build, and its runs still execute the other tree's solver.
    Selecting a different device cannot reach it, so the message says to stop
    the process instead.
    """
    root = _both_builds(tmp_path / "mine", native)
    elsewhere = tmp_path / "elsewhere" / "target"
    elsewhere.mkdir(parents=True)
    with pytest.raises(conn.NativeServerMismatch) as caught:
        conn.check_running_server(
            _response(conn, elsewhere, "cuda"),
            root, 9090, conn.DEVICE_GPU, _resolver(conn, native),
        )
    message = str(caught.value)
    assert str(elsewhere) in message, message
    assert "set Compute Device to" not in message, message
    assert "port 9090" in message, message


@pytest.mark.parametrize("native", NATIVES)
def test_a_server_that_does_not_report_its_build_is_refused(conn, tmp_path, native):
    """Silence is not agreement: a server that cannot say is not attached to."""
    root = _both_builds(tmp_path, native)
    response = _response(conn, "", "")
    with pytest.raises(conn.NativeServerMismatch):
        conn.check_running_server(
            response, root, 9090, conn.DEVICE_GPU, _resolver(conn, native),
        )


@pytest.mark.parametrize("native", NATIVES)
def test_another_protocol_and_an_empty_port_are_not_this_checks_to_refuse(
    conn, tmp_path, native
):
    """Two cases pass through untouched, each for a stated reason.

    A server on another protocol version is stopped by the handshake on its
    first response, which is a better outcome than a refusal that leaves it
    running. And nothing answering is nothing to compare.
    """
    root = _both_builds(tmp_path, native)
    resolve = _resolver(conn, native)
    conn.check_running_server(
        _response(conn, tmp_path / "target", "cuda", version="0.0"),
        root, 9090, conn.DEVICE_CPU, resolve,
    )
    conn.check_running_server(None, root, 9090, conn.DEVICE_CPU, resolve)


@pytest.mark.parametrize("native", NATIVES)
def test_connect_refuses_a_running_server_of_the_other_build(
    conn, tmp_path, native, monkeypatch
):
    """The check is WIRED into Connect, in both modes.

    Connect is where a disconnected-but-running server is picked up again, so
    it is where the refusal has to land. The test mode, where an orchestrator
    owns the server, is covered too, because an orchestrator-owned server
    running the other build is the same substitution.
    """
    root = _both_builds(tmp_path, native)
    gpu_server = _response(conn, tmp_path / "target", "cuda")
    asked = []

    def query(port, timeout=1.5, name=""):
        asked.append(name)
        return gpu_server

    monkeypatch.setattr(conn, "_query_ppf_cts_server", query)
    connect = getattr(conn, f"connect_{native}")
    no_spawn = f"PPF_{native.split(chr(95))[0].upper()}_NATIVE_NO_SPAWN"
    for mode in (None, "1"):
        if mode is None:
            monkeypatch.delenv(no_spawn, raising=False)
        else:
            monkeypatch.setenv(no_spawn, mode)
        with pytest.raises(conn.NativeServerMismatch):
            connect(root, 9090, conn.DEVICE_CPU, "my_project")
        info, _process = connect(root, 9090, conn.DEVICE_GPU, "my_project")
        assert info.current_directory == root
    # A status query SELECTS the project it names on the server, so Connect
    # asks under the add-on's own project rather than a dummy one, which
    # would move a running server's selection away from the user's project.
    assert asked and set(asked) == {"my_project"}, asked


@pytest.mark.parametrize("native", NATIVES)
def test_start_server_refuses_to_attach_to_the_other_build(
    conn, tmp_path, native, monkeypatch
):
    """The check is WIRED into Start Server's attach path as well.

    A server can start on the port between Connect and Start, and the spawn
    then attaches rather than launching. Attaching to the matching build still
    returns ``None``, the attach-mode result every caller already handles.
    """
    root = _both_builds(tmp_path, native)
    gpu_server = _response(conn, tmp_path / "target", "cuda")
    monkeypatch.setattr(conn, "_port_is_in_use", lambda port: True)
    monkeypatch.setattr(
        conn, "_query_ppf_cts_server", lambda port, timeout=1.5, name="": gpu_server
    )
    if native == "win_native":
        monkeypatch.delenv("PPF_WIN_NATIVE_NO_SPAWN", raising=False)

        def spawn(device):
            return conn.spawn_win_native_server(root, 9090, device=device)
    else:
        monkeypatch.delenv("PPF_MAC_NATIVE_NO_SPAWN", raising=False)

        def spawn(device):
            return conn.spawn_mac_native_server(root, 9090, device)
    with pytest.raises(conn.NativeServerMismatch):
        spawn(conn.DEVICE_CPU)
    assert spawn(conn.DEVICE_GPU) is None


# ---------------------------------------------------------------------------
# A root that holds ONE build
#
# The Windows ARM64 distribution ships the CPU build alone, and a tree built
# only with `--features cpu` looks the same. The Compute Device defaults to
# GPU, so the first Connect against such a folder asks for a build it does not
# hold. These tests hold the add-on to letting the artist change that, and to
# refusing by name rather than running the other build.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gpu,cpu,selected,is_open",
    [
        (True, True, "GPU", True),
        (True, True, "CPU", True),
        (False, True, "GPU", True),
        (False, True, "CPU", False),
        (True, False, "CPU", True),
        (True, False, "GPU", False),
    ],
)
def test_the_selector_is_open_exactly_when_there_is_something_to_change(
    conn, gpu, cpu, selected, is_open
):
    """Open with both builds, or while the selection names the absent one.

    A CPU-only folder starts on GPU, the property's default. A selector locked
    there offers nothing Connect can run; one left open on the build that IS
    there would let the artist move off it onto nothing.
    """
    have = {conn.DEVICE_GPU: gpu, conn.DEVICE_CPU: cpu}
    assert conn.native_device_choice_open(have, selected) is is_open


@pytest.mark.parametrize("native", NATIVES)
def test_a_one_build_root_refuses_the_other_device_by_name(conn, tmp_path, native):
    """The refusal names the build that is there, and the folder stays right.

    Telling the artist to point Solver Path elsewhere would send them away from
    a folder that is correct. A root holding neither build keeps the path
    advice, because there the folder IS the problem.
    """
    message = getattr(conn, f"{native}_not_found_message")
    cpu_only = _root(tmp_path / "cpu_only", gpu=False, cpu=True, native=native)
    text = message(cpu_only, conn.DEVICE_GPU)
    assert "holds the CPU build" in text and "Set Compute Device to CPU" in text, text
    gpu_only = _root(tmp_path / "gpu_only", gpu=True, cpu=False, native=native)
    text = message(gpu_only, conn.DEVICE_CPU)
    assert "holds the GPU build" in text and "Set Compute Device to GPU" in text, text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert message(str(empty), conn.DEVICE_GPU) == message(str(empty))
    assert "Point Solver Path" in message(str(empty), conn.DEVICE_GPU)


@pytest.mark.parametrize("native", NATIVES)
def test_launching_a_cpu_only_root_as_gpu_is_refused_by_name(
    conn, tmp_path, monkeypatch, native
):
    """The launch refuses with the named text, and never runs the CPU server."""
    root = _root(tmp_path, gpu=False, cpu=True, native=native)
    monkeypatch.setattr(conn, "_port_is_in_use", lambda port: False)
    if native == "win_native":
        monkeypatch.delenv("PPF_WIN_NATIVE_NO_SPAWN", raising=False)

        def spawn():
            return conn.spawn_win_native_server(root, 9090, device=conn.DEVICE_GPU)
    else:
        monkeypatch.delenv("PPF_MAC_NATIVE_NO_SPAWN", raising=False)

        def spawn():
            return conn.spawn_mac_native_server(root, 9090, conn.DEVICE_GPU)
    with pytest.raises(FileNotFoundError, match="holds the CPU build"):
        spawn()
