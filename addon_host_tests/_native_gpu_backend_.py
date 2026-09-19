# File: addon_host_tests/_native_gpu_backend_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# WHICH ACCELERATOR a local native connection runs, once Compute Device says
# GPU and the root holds more than one GPU build.
#
# A Windows x64 distribution carries CUDA and ROCm together, and a machine can
# have an NVIDIA and an AMD card at once, so the add-on has to choose. The rule
# is the frontend's (`frontend/_backends_.py`, driven by
# `frontend/tests/_backend_resolution_.py`), restated in the add-on because it
# runs inside Blender and reaches the solver through the server rather than
# through the solver's Python package. THESE CASES ARE THAT FILE'S CASES, so
# the two cannot drift apart unnoticed.
#
# The probe is a callable this file controls, which is the only way to run the
# branch where the AMD device answers and the NVIDIA one does not: no machine
# this project owns has an AMD GPU.

import pytest


@pytest.fixture(scope="module")
def conn():
    """``blender_addon.core.connection``, loaded from source."""
    from conftest import ADDON_ROOT, _ensure_package, load_addon_module

    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.connection")


def _build(root, *parts, backend=None, windows=True):
    """A server under ``<root>/<parts>``, with *backend* recorded beside it."""
    exe = "ppf-cts-server.exe" if windows else "ppf-cts-server"
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / exe).write_text("")
    if backend is not None:
        (directory / ".ppf-backend").write_text(backend)
    return str(directory / exe)


def _probe_returning(answers):
    """A probe answering from *answers* (``{backend: (usable, detail)}``)."""
    asked = []

    def probe(server_path):
        for name, answer in answers.items():
            if name in server_path.replace("\\", "/").split("/"):
                asked.append(name)
                return answer
        raise AssertionError(f"nothing in {answers} matches {server_path}")

    return probe, asked


def _never_probed(server_path):
    raise AssertionError(f"{server_path} was asked, and nothing needed asking")


def test_a_backend_directory_is_found_by_its_marker(conn, tmp_path):
    cuda = _build(tmp_path, "target", "cuda", "release", backend="cuda")
    rocm = _build(tmp_path, "target", "rocm", "release", backend="rocm")
    assert conn.win_native_gpu_builds(str(tmp_path)) == {"cuda": cuda, "rocm": rocm}


def test_a_marker_outranks_the_directory_name(conn, tmp_path):
    """A directory called cuda holding a ROCm build IS a ROCm build.

    The marker is the only evidence of what a directory holds; the name is a
    convention, and answering by the name would offer CUDA and run ROCm.
    """
    _build(tmp_path, "target", "cuda", "release", backend="rocm")
    assert list(conn.win_native_gpu_builds(str(tmp_path))) == ["rocm"]


def test_an_unmarked_historical_build_is_unnamed(conn, tmp_path):
    """The bundle that shipped before markers cannot say which accelerator it is."""
    legacy = _build(tmp_path, "target", "release")
    assert conn.win_native_gpu_builds(str(tmp_path)) == {conn._GPU_UNNAMED: legacy}


def test_an_unmarked_root_resolves_exactly_as_it_always_did(conn, tmp_path):
    """The historical layout keeps its answer, with no marker and no probe."""
    legacy = _build(tmp_path, "target", "release")
    assert (
        conn.win_native_server_binary(str(tmp_path), conn.DEVICE_GPU, probe=_never_probed)
        == legacy
    )


def test_a_named_choice_takes_that_build(conn, tmp_path):
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    rocm = _build(tmp_path, "target", "rocm", "release", backend="rocm")
    resolved = conn.win_native_server_binary(
        str(tmp_path), conn.DEVICE_GPU, "ROCM", _never_probed
    )
    assert resolved == rocm


def test_a_named_choice_with_no_build_resolves_to_nothing(conn, tmp_path):
    """Never to another accelerator, which is the substitution being prevented."""
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    assert (
        conn.win_native_server_binary(str(tmp_path), conn.DEVICE_GPU, "ROCM", _never_probed)
        is None
    )


def test_the_refusal_names_the_builds_that_are_here(conn, tmp_path):
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    message = conn._absent_gpu_backend_message(
        str(tmp_path), "ROCM", conn.win_native_gpu_builds
    )
    assert "cuda" in message and "no rocm build" in message


def test_the_one_gpu_build_is_taken_without_asking_a_solver(conn, tmp_path):
    cuda = _build(tmp_path, "target", "cuda", "release", backend="cuda")
    _build(tmp_path, "target", "cpu", "release", backend="cpu")
    resolved = conn.win_native_server_binary(
        str(tmp_path), conn.DEVICE_GPU, conn.GPU_BACKEND_AUTO, _never_probed
    )
    assert resolved == cuda


def test_the_panel_takes_the_first_in_order_without_a_probe(conn, tmp_path):
    """Drawing must not run a solver, so with no probe the order decides."""
    cuda = _build(tmp_path, "target", "cuda", "release", backend="cuda")
    _build(tmp_path, "target", "rocm", "release", backend="rocm")
    assert conn.win_native_server_binary(str(tmp_path), conn.DEVICE_GPU) == cuda


def test_where_both_answer_cuda_is_chosen(conn, tmp_path):
    cuda = _build(tmp_path, "target", "cuda", "release", backend="cuda")
    _build(tmp_path, "target", "rocm", "release", backend="rocm")
    probe, asked = _probe_returning(
        {"cuda": (True, "NVIDIA L40S"), "rocm": (True, "gfx1100")}
    )
    resolved = conn.win_native_server_binary(
        str(tmp_path), conn.DEVICE_GPU, conn.GPU_BACKEND_AUTO, probe
    )
    assert resolved == cuda
    assert asked == ["cuda"]


def test_where_only_the_amd_device_answers_rocm_is_chosen(conn, tmp_path):
    """The branch no machine here can produce: an AMD GPU and no NVIDIA one."""
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    rocm = _build(tmp_path, "target", "rocm", "release", backend="rocm")
    probe, asked = _probe_returning(
        {"cuda": (False, "No NVIDIA GPU detected."), "rocm": (True, "gfx1100")}
    )
    resolved = conn.win_native_server_binary(
        str(tmp_path), conn.DEVICE_GPU, conn.GPU_BACKEND_AUTO, probe
    )
    assert resolved == rocm
    assert asked == ["cuda", "rocm"]


def test_where_neither_answers_the_spawn_is_refused_naming_both(conn, tmp_path):
    """And never moved onto the CPU build, which the artist did not choose."""
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    _build(tmp_path, "target", "rocm", "release", backend="rocm")
    _build(tmp_path, "target", "cpu", "release", backend="cpu")
    probe, _ = _probe_returning(
        {
            "cuda": (False, "No NVIDIA GPU detected."),
            "rocm": (False, "No AMD GPU detected."),
        }
    )
    with pytest.raises(conn.NoUsableGpuBackend) as refusal:
        conn.win_native_server_binary(
            str(tmp_path), conn.DEVICE_GPU, conn.GPU_BACKEND_AUTO, probe
        )
    message = str(refusal.value)
    assert "No NVIDIA GPU detected." in message and "No AMD GPU detected." in message
    assert "Compute Device to CPU" in message


def test_the_cpu_device_is_untouched_by_the_backend_choice(conn, tmp_path):
    cpu = _build(tmp_path, "target", "cpu", "release", backend="cpu")
    _build(tmp_path, "target", "cuda", "release", backend="cuda")
    resolved = conn.win_native_server_binary(
        str(tmp_path), conn.DEVICE_CPU, "CUDA", _never_probed
    )
    assert resolved == cpu


@pytest.mark.parametrize(
    "builds,selected,expected",
    [
        ({"cuda": "a"}, "AUTO", False),
        ({"cuda": "a", "rocm": "b"}, "AUTO", True),
        ({"cuda": "a", "rocm": "b"}, "CUDA", True),
        # The saved choice names a backend this root does not hold, so the
        # selector stays open: a `.blend` carrying it must be correctable.
        ({"cuda": "a"}, "ROCM", True),
        ({"cuda": "a"}, "CUDA", False),
        # An unmarked historical build is not a named choice to offer.
        ({"": "a"}, "AUTO", False),
    ],
)
def test_when_the_backend_selector_is_open(conn, builds, selected, expected):
    assert conn.native_gpu_backend_choice_open(builds, selected) is expected


def test_macos_answers_the_same_way_with_one_accelerator(conn, tmp_path):
    """The macOS twin takes the same parameters, and Metal is the only build."""
    metal = _build(tmp_path, "target", "release", backend="metal", windows=False)
    assert conn.mac_native_gpu_builds(str(tmp_path)) == {"metal": metal}
    assert (
        conn.mac_native_server_binary(str(tmp_path), conn.DEVICE_GPU, probe=_never_probed)
        == metal
    )
