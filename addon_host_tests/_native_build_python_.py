# File: addon_host_tests/_native_build_python_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gates for WHICH PYTHON a POSIX native spawn hands its build worker
# (``blender_addon/core/connection.py``).
#
# ONE ANSWER FOR macOS AND LINUX. `build-mac-native/bundle.sh` and
# `build-linux-native/bundle.sh` put the interpreter in the same place inside
# the distribution, so the question belongs to the ROOT rather than to the
# platform and both spawns ask it through `native_build_python`.
#
# WHY THIS IS ITS OWN SUBJECT. ppf-cts-server does not tetrahedralize; it
# spawns a worker under a Python that must carry the frontend dependency set
# (numpy, scipy, pytetwild, tetgen, ...). Naming the wrong interpreter does not
# fail at launch, it fails minutes later inside a SOLID build as
# ``ModuleNotFoundError``, which reads as a broken distribution rather than as
# the add-on pointing somewhere else.
#
# THE FAILURE THESE PIN. The spawn named the developer venv
# (``~/.local/share/ppf-cts/venv``) unconditionally. A DISTRIBUTION ships its
# own fully provisioned interpreter inside the folder the artist selects, and
# its own ``start.sh`` names that one in ``PPF_CTS_BUILD_PYTHON``, which is why
# its JupyterLab tetrahedralizes fine while the same distribution driven from
# the add-on reported no pytetwild.

from __future__ import annotations

import os

import pytest

# The macOS native spawn is POSIX code: it joins PATH with ":", names
# bin/python3, and resolves the developer venv through HOME, which Windows
# ignores when expanding "~". Windows never runs it.
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the macOS native spawn is POSIX-only"
)


@pytest.fixture(scope="module")
def conn():
    from conftest import ADDON_ROOT, _ensure_package, load_addon_module

    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.connection")


def _dist(tmp_path):
    """A distribution root, in the layout ``bundle.sh`` publishes."""
    interpreter = tmp_path / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("")
    return str(tmp_path), str(interpreter)


def _dev_venv(home):
    """A developer venv at the convention ``warmup.py`` resolves."""
    interpreter = home / ".local" / "share" / "ppf-cts" / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("")
    (interpreter.parent.parent / "pyvenv.cfg").write_text("home = /usr\n")
    return str(interpreter)


def test_a_distribution_uses_the_interpreter_it_ships(conn, tmp_path, monkeypatch):
    """THE BUG THIS PINS.

    A distribution carries its dependencies in ``<root>/python``, and that is
    the interpreter its own launcher exports and its own JupyterLab runs. The
    add-on must not reach past it to a venv belonging to some checkout
    elsewhere on the machine.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    venv_py = _dev_venv(home)
    root, shipped = _dist(tmp_path / "dist")
    chosen = conn.native_build_python(root)
    assert chosen == shipped, "reached past the distribution's own interpreter"
    assert chosen != venv_py


def test_a_checkout_falls_back_to_the_developer_venv(conn, tmp_path, monkeypatch):
    """The mirror case, and the behavior that must not change.

    A checkout ships no interpreter of its own, so the venv every provisioning
    path here builds is the answer, exactly as before.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    venv_py = _dev_venv(home)
    checkout = tmp_path / "dev"
    checkout.mkdir()
    assert conn.native_build_python(str(checkout)) == venv_py


def test_nothing_is_invented_when_no_interpreter_exists(conn, tmp_path, monkeypatch):
    """``None`` rather than a plausible path that is not there.

    The server's own resolution then reports what it could not find. Naming a
    nonexistent interpreter would replace that report with an exec failure that
    says nothing about the dependency set.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    checkout = tmp_path / "dev"
    checkout.mkdir()
    assert conn.native_build_python(str(checkout)) is None


def test_a_blank_root_still_answers_for_the_machine(conn, tmp_path, monkeypatch):
    """No root is not a reason to find no interpreter.

    Only the FIRST candidate is keyed on the root; the venv is a property of
    the machine, and a caller with nothing to say about a root still gets it.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    venv_py = _dev_venv(home)
    assert conn.native_build_python("") == venv_py


def _spawn_env(conn, monkeypatch, root):
    """The environment ``spawn_mac_native_server`` would launch with.

    Captured by intercepting ``Popen`` rather than by re-deriving it, so what
    is asserted is what the child actually receives.
    """
    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")

    monkeypatch.setattr(conn.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(conn, "_port_is_in_use", lambda port: False)
    conn.spawn_mac_native_server(root, 9090)
    return captured["env"]


def _solver_root(tmp_path, backend="metal"):
    d = tmp_path / "target" / "release"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ppf-cts-server").write_text("")
    (d / ".ppf-backend").write_text(backend)
    return str(tmp_path)


def test_the_spawn_names_the_interpreter_explicitly(conn, tmp_path, monkeypatch):
    """``PPF_CTS_BUILD_PYTHON`` is set, not left to ``VIRTUAL_ENV`` or PATH.

    Only the explicit variable can express "the interpreter that belongs to
    THIS root". macOS's own ``python3`` is 3.9 and cannot parse the frontend,
    so the PATH step at the end of the server's resolution is never an answer
    here.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    root = _solver_root(tmp_path / "dist")
    _, shipped = _dist(tmp_path / "dist")
    env = _spawn_env(conn, monkeypatch, root)
    assert env["PPF_CTS_BUILD_PYTHON"] == shipped
    assert env["PATH"].split(":")[0] == os.path.dirname(shipped)


def test_the_spawn_keeps_virtual_env_for_a_real_venv(conn, tmp_path, monkeypatch):
    """A venv answer still exports ``VIRTUAL_ENV``, and a shipped tree does not.

    ``pyvenv.cfg`` is what makes a prefix a venv, and it is the distribution's
    relocatable interpreter that is NOT one: a venv records absolute paths and
    would not survive being copied, which is why the distribution installs into
    the tree instead. Exporting ``VIRTUAL_ENV`` for it would be a claim about a
    layout that is not there.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    venv_py = _dev_venv(home)
    checkout = _solver_root(tmp_path / "dev")
    env = _spawn_env(conn, monkeypatch, checkout)
    assert env["PPF_CTS_BUILD_PYTHON"] == venv_py
    assert env["VIRTUAL_ENV"] == os.path.dirname(os.path.dirname(venv_py))

    dist = _solver_root(tmp_path / "dist")
    _dist(tmp_path / "dist")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    env = _spawn_env(conn, monkeypatch, dist)
    assert "VIRTUAL_ENV" not in env


def test_an_inherited_override_is_not_clobbered(conn, tmp_path, monkeypatch):
    """Someone who set ``PPF_CTS_BUILD_PYTHON`` meant it.

    It is also how the add-on is launched from inside a distribution's own
    environment, where the launcher has already exported the right answer.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _dev_venv(home)
    monkeypatch.setenv("PPF_CTS_BUILD_PYTHON", "/opt/chosen/bin/python")
    root = _solver_root(tmp_path / "dist")
    _dist(tmp_path / "dist")
    env = _spawn_env(conn, monkeypatch, root)
    assert env["PPF_CTS_BUILD_PYTHON"] == "/opt/chosen/bin/python"


def test_a_stale_inherited_virtual_env_is_cleared(conn, tmp_path, monkeypatch):
    """Blender is commonly started from a shell with some venv active.

    That inherited value is what put the developer venv in front of a
    distribution's build worker in the first place. Once an interpreter is
    named explicitly and it is not a venv, leaving the variable behind would
    have the environment contradict the choice, and any message the server
    writes about how it resolved its interpreter would name the wrong one.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    stale = _dev_venv(home)
    monkeypatch.setenv("VIRTUAL_ENV", os.path.dirname(os.path.dirname(stale)))
    root = _solver_root(tmp_path / "dist")
    _, shipped = _dist(tmp_path / "dist")
    env = _spawn_env(conn, monkeypatch, root)
    assert env["PPF_CTS_BUILD_PYTHON"] == shipped
    assert "VIRTUAL_ENV" not in env, "the stale venv survived beside the choice"


def test_the_target_directory_matches_the_server_that_was_spawned(conn, tmp_path, monkeypatch):
    """The device choice has to reach the SOLVER, not just the server.

    The build worker loads the cdylib out of ``CARGO_TARGET_DIR`` and the
    session script then names that same directory as ``SOLVER_PATH``. Spawn the
    CPU server without saying this and the run executes the GPU solver out of
    ``target/release``, with each binary answering ``--backend`` honestly about
    itself and neither asked about the other.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _dev_venv(home)
    root = tmp_path / "tree"
    for parts, backend in (
        (("target", "release"), "metal"),
        (("target", "cpu", "release"), "cpu"),
    ):
        d = root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppf-cts-server").write_text("")
        (d / ".ppf-backend").write_text(backend)

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            captured["env"] = kwargs.get("env")

    monkeypatch.setattr(conn.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(conn, "_port_is_in_use", lambda port: False)

    conn.spawn_mac_native_server(str(root), 9090, conn.DEVICE_CPU)
    assert captured["env"]["CARGO_TARGET_DIR"] == str(root / "target" / "cpu")

    conn.spawn_mac_native_server(str(root), 9090, conn.DEVICE_GPU)
    assert captured["env"]["CARGO_TARGET_DIR"] == str(root / "target")


def test_an_inherited_target_dir_is_not_allowed_to_redirect_the_run(conn, tmp_path, monkeypatch):
    """A ``CARGO_TARGET_DIR`` exported in the user's own shell is searched
    BEFORE the tree's own ``target``, so an inherited one would send the
    frontend looking for the cdylib somewhere unrelated to the folder the
    artist selected. The distribution's own launcher unsets it for this reason.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _dev_venv(home)
    monkeypatch.setenv("CARGO_TARGET_DIR", "/somewhere/else")
    root = _solver_root(tmp_path / "tree")
    env = _spawn_env(conn, monkeypatch, root)
    assert env["CARGO_TARGET_DIR"] == os.path.join(root, "target")
