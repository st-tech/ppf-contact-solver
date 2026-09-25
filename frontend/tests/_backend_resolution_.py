# File: frontend/tests/_backend_resolution_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Which backend a run uses where several are built: `frontend/_backends_.py`.
#
# THE RULE IS DRIVEN HERE WITH DIRECTORIES THIS FILE MAKES AND A PROBE IT
# CONTROLS, so every branch runs on any machine. One of them cannot be produced
# any other way: no machine this project owns has an AMD GPU, so "the AMD device
# answers and the NVIDIA one does not" has no hardware to happen on.
#
# The module is loaded by path because it imports nothing from the package, and
# importing the package loads the extension module, which a tree need not have
# built. The last case does need a build and skips itself without one.

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "frontend" / "_backends_.py"
    spec = importlib.util.spec_from_file_location("_backends_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backends = _load_module()


def _build(root: Path, directory: str, backend: str, profile: str = "release") -> str:
    """A build directory holding *backend*, as a bundler or `build.rs` leaves it."""
    made = root / directory / profile
    made.mkdir(parents=True)
    (made / backends.MARKER).write_text(f"{backend}\n")
    return str(root / directory)


def _probe_returning(answers):
    """A probe returning *answers* (``{backend: (usable, detail)}``), recording asks."""
    asked = []

    def probe(name):
        asked.append(name)
        usable, detail = answers[name]
        return backends.Probe(name, usable, detail, "stamp")

    return probe, asked


def _never_probed(name):
    raise AssertionError(f"the {name} solver was asked, and nothing needed asking")


def test_a_marker_names_what_a_directory_holds_whatever_it_is_called(tmp_path):
    target = _build(tmp_path, "target/cuda", "rocm")
    assert backends.builds([target]) == {"rocm": os.path.join(target, "release")}


def test_the_first_directory_holding_a_backend_wins(tmp_path):
    first = _build(tmp_path, "named", "cuda")
    second = _build(tmp_path, "target/cuda", "cuda")
    found = backends.builds([first, second])
    assert found == {"cuda": os.path.join(first, "release")}


def test_a_directory_with_no_marker_or_an_empty_one_holds_no_build(tmp_path):
    (tmp_path / "bare" / "release").mkdir(parents=True)
    empty = _build(tmp_path, "empty", "")
    (Path(empty) / "release" / backends.MARKER).write_text("   \n")
    assert backends.builds([str(tmp_path / "bare"), empty]) == {}


def test_a_marker_written_by_cmd_exe_reads_the_same(tmp_path):
    made = tmp_path / "target" / "cpu" / "release"
    made.mkdir(parents=True)
    (made / backends.MARKER).write_text("cpu\r\n")
    assert backends.builds([str(tmp_path / "target" / "cpu")]) == {"cpu": str(made)}


def test_an_explicit_choice_is_returned_without_asking_any_solver(tmp_path):
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
        ]
    )
    assert backends.resolve(built, "rocm", _never_probed).backend == "rocm"


def test_an_explicit_choice_with_no_build_is_refused_naming_what_is_built(tmp_path):
    built = backends.builds([_build(tmp_path, "target/cuda", "cuda")])
    with pytest.raises(RuntimeError) as refusal:
        backends.resolve(built, "rocm", _never_probed)
    assert "no rocm build" in str(refusal.value)
    assert "cuda" in str(refusal.value)


def test_an_unknown_backend_name_is_refused(tmp_path):
    built = backends.builds([_build(tmp_path, "target/cuda", "cuda")])
    with pytest.raises(ValueError):
        backends.resolve(built, "vulkan", _never_probed)


def test_the_one_gpu_backend_built_is_chosen_when_its_device_answers(tmp_path):
    """One GPU build beside a CPU one, and the GPU can run: the GPU is the answer.

    THE PROBE IS ASKED EVEN THOUGH THERE IS NOTHING TO CHOOSE BETWEEN, which is
    what makes the fallback below reach the machine it exists for: a
    distribution for one accelerator carries exactly one GPU backend, so the
    machine with no GPU is precisely the one where an earlier rule skipped the
    question and answered "cuda".
    """
    built = backends.builds(
        [_build(tmp_path, "target/cuda", "cuda"), _build(tmp_path, "target/cpu", "cpu")]
    )
    probe, asked = _probe_returning({"cuda": (True, "NVIDIA L40S")})
    answer = backends.resolve(built, None, probe)
    assert answer.backend == "cuda"
    assert answer.notice == ""
    assert asked == ["cuda"]


def test_the_one_gpu_backend_built_is_chosen_without_a_probe_when_nothing_else_is(tmp_path):
    """With no CPU build to fall back to, the single GPU build is the answer.

    Asking its solver could not change it, so it is not asked: the failure of a
    backend that cannot run belongs to the run, which reports it against the
    machine, rather than to this choice.
    """
    built = backends.builds([_build(tmp_path, "target/cuda", "cuda")])
    assert backends.resolve(built, None, _never_probed).backend == "cuda"


def test_where_both_gpus_answer_cuda_is_chosen(tmp_path):
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
            _build(tmp_path, "target/cpu", "cpu"),
        ]
    )
    probe, asked = _probe_returning(
        {"cuda": (True, "NVIDIA L40S"), "rocm": (True, "gfx1100")}
    )
    assert backends.resolve(built, None, probe).backend == "cuda"
    assert asked == ["cuda", "rocm"]


def test_where_only_the_amd_device_answers_rocm_is_chosen(tmp_path):
    """The branch no machine here can produce: an AMD GPU and no NVIDIA one."""
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
        ]
    )
    probe, _ = _probe_returning(
        {
            "cuda": (False, "No NVIDIA GPU detected."),
            "rocm": (True, "gfx1100"),
        }
    )
    assert backends.resolve(built, None, probe).backend == "rocm"


def test_a_backend_whose_runtime_is_absent_is_unusable_rather_than_fatal():
    """The Windows loader refusing a backend must not abort a run another serves.

    A distribution carries every backend its architecture can build, while a
    machine carries only the runtimes it has: the x64 archive holds both the
    CUDA and the ROCm solver, and the ROCm one imports an AMD runtime DLL that
    an NVIDIA machine does not have. The loader then kills the process before
    `--probe` prints anything, so the answer has to come from the exit status.
    """
    for status in backends.WINDOWS_LOADER_FAILURE:
        answer = backends.parse_probe("", "", status, "solver.exe", "rocm")
        assert answer.backend == "rocm"
        assert answer.usable is False
        assert str(status) in answer.detail


def test_a_solver_that_ran_and_answered_nothing_is_still_a_hard_failure():
    """The loader-failure reading must not swallow a stale or broken binary."""
    with pytest.raises(RuntimeError, match="did not answer"):
        backends.parse_probe("", "", 1, "solver.exe", "rocm")
    # Without a marker there is no backend to report as unusable, so even a
    # loader status is a failure to answer rather than a silent skip.
    with pytest.raises(RuntimeError, match="did not answer"):
        backends.parse_probe("", "", backends.WINDOWS_LOADER_FAILURE[0], "solver.exe")


def test_an_absent_rocm_runtime_leaves_cuda_serving_the_run(tmp_path):
    """The shape of the CI failure this pair of tests exists for."""
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
            _build(tmp_path, "target/cpu", "cpu"),
        ]
    )

    def probe(name):
        if name == "rocm":
            return backends.parse_probe(
                "", "", backends.WINDOWS_LOADER_FAILURE[0], "solver.exe", "rocm"
            )
        return backends.Probe(name, True, "NVIDIA L40S", "stamp")

    assert backends.resolve(built, None, probe).backend == "cuda"


def test_where_no_gpu_answers_the_cpu_backend_is_chosen_and_the_run_says_so(tmp_path):
    """The machine with no usable GPU runs, and is told why it is slow.

    THIS IS THE ONE FALLBACK IN THE RULE, and the notice is what makes it
    honest rather than silent. A reader opening a notebook on a laptop gets a
    run instead of a refusal they would answer by writing
    `App.set_backend("cpu")` at the top of every notebook, including the
    ones they later open on a machine that has a GPU.
    """
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
            _build(tmp_path, "target/cpu", "cpu"),
        ]
    )
    probe, _ = _probe_returning(
        {
            "cuda": (False, "No NVIDIA GPU detected."),
            "rocm": (False, "No AMD GPU detected."),
        }
    )
    answer = backends.resolve(built, None, probe)
    assert answer.backend == "cpu"
    assert "No NVIDIA GPU detected." in answer.notice
    assert "No AMD GPU detected." in answer.notice
    assert "CPU backend is selected" in answer.notice


def test_the_cpu_fallback_never_overrides_an_explicit_gpu_choice(tmp_path):
    """Naming a backend that cannot run here is an error, not a fallback.

    The whole safety of the fallback rests on this: an automatic choice may
    land on the CPU backend, and a choice the caller made may not be moved off
    what they asked for, however unusable it is. A run that silently ran
    somewhere other than where it was told to is the defect both rules exist to
    prevent, approached from opposite sides.
    """
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/cpu", "cpu"),
        ]
    )
    answer = backends.resolve(built, "cuda", _never_probed)
    assert answer.backend == "cuda"
    assert answer.notice == ""


def test_where_no_gpu_answers_and_no_cpu_is_built_the_run_is_refused(tmp_path):
    """Nothing here can run, so there is nothing to fall back to.

    The refusal names what each GPU backend reported and the command that adds
    the build that would have served it.
    """
    built = backends.builds(
        [
            _build(tmp_path, "target/cuda", "cuda"),
            _build(tmp_path, "target/rocm", "rocm"),
        ]
    )
    probe, _ = _probe_returning(
        {
            "cuda": (False, "No NVIDIA GPU detected."),
            "rocm": (False, "No AMD GPU detected."),
        }
    )
    with pytest.raises(RuntimeError) as refusal:
        backends.resolve(built, None, probe)
    message = str(refusal.value)
    assert "No NVIDIA GPU detected." in message and "No AMD GPU detected." in message
    assert "--features cpu" in message


def test_a_tree_with_no_gpu_build_runs_on_the_cpu(tmp_path):
    built = backends.builds([_build(tmp_path, "target/cpu", "cpu")])
    assert backends.resolve(built, None, _never_probed).backend == "cpu"


def test_nothing_built_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        backends.resolve({}, None, _never_probed)


def test_a_usable_answer_is_read(tmp_path):
    answer = backends.parse_probe(
        "backend: cuda\nstamp: abc123\ndevice: NVIDIA L40S\n", "", 0, "solver"
    )
    assert answer == backends.Probe("cuda", True, "NVIDIA L40S", "abc123")


def test_an_unusable_answer_is_read(tmp_path):
    answer = backends.parse_probe(
        "backend: rocm\nlinked: rocm\nstamp: abc123\nunusable: No AMD GPU detected.\n",
        "",
        backends.PROBE_UNUSABLE,
        "solver",
    )
    assert answer.backend == "rocm"
    assert not answer.usable
    assert answer.detail == "No AMD GPU detected."


@pytest.mark.parametrize(
    "stdout,returncode",
    [
        # A solver built before --probe existed: clap refuses the argument.
        ("", 2),
        # A verdict with no device named, and a device named with no verdict.
        ("backend: cuda\nstamp: abc123\n", 0),
        (
            "backend: cuda\nstamp: abc123\ndevice: NVIDIA L40S\n",
            backends.PROBE_UNUSABLE,
        ),
        # No stamp, so nothing can be compared against the extension module.
        ("backend: cuda\ndevice: NVIDIA L40S\n", 0),
        # A crash, which must never read as "unusable".
        ("backend: cuda\nstamp: abc123\n", 1),
    ],
)
def test_anything_that_is_not_an_answer_is_refused(stdout, returncode):
    with pytest.raises(RuntimeError) as refusal:
        backends.parse_probe(stdout, "", returncode, "solver")
    assert "--probe did not answer" in str(refusal.value)


def test_the_built_cpu_solver_answers_and_carries_this_tree_s_stamp():
    """The real thing, end to end: `--probe`, the package API, and the stamp.

    Skipped rather than passed over nothing where the CPU backend is not built.
    """
    target = REPO_ROOT / "target" / "cpu"
    solver = (
        "ppf-contact-solver.exe" if sys.platform == "win32" else "ppf-contact-solver"
    )
    if not (target / "release" / solver).is_file():
        pytest.skip("this tree has no CPU build to probe")
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from frontend import App;"
            "import _ppf_cts_py;"
            "answer = App.probe_backend('cpu');"
            "print(App.get_backend(), answer.usable,"
            " answer.stamp == _ppf_cts_py.__source_stamp__)",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "cpu True True", result.stdout


def test_app_and_the_package_functions_hold_one_choice():
    """`App.set_backend` is the package's `set_backend`, not a second copy.

    A notebook imports only `App`, and a script written earlier calls the
    package function, so a choice made through either must be what the other
    reads back, and a refusal must come from the one rule both share.
    """
    target = REPO_ROOT / "target" / "cpu"
    solver = (
        "ppf-contact-solver.exe" if sys.platform == "win32" else "ppf-contact-solver"
    )
    if not (target / "release" / solver).is_file():
        pytest.skip("this tree has no CPU build to choose")
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    script = "\n".join(
        [
            "import frontend",
            "from frontend import App",
            "assert App.list_backends() == frontend.list_backends()",
            "App.set_backend('cpu')",
            "assert frontend.get_backend() == 'cpu'",
            "assert App.get_backend() == 'cpu'",
            "try:",
            "    App.set_backend('no-such-backend')",
            "except ValueError as refusal:",
            "    print('refused:', refusal)",
            "else:",
            "    raise SystemExit('a name that is no backend was accepted')",
            "unbuilt = [n for n in ('cuda', 'rocm', 'metal') if n not in App.list_backends()]",
            "for name in unbuilt[:1]:",
            "    try:",
            "        App.set_backend(name)",
            "    except RuntimeError as refusal:",
            "        print('refused:', refusal)",
            "    else:",
            "        raise SystemExit(f'the unbuilt {name} backend was accepted')",
            "assert App.get_backend() == 'cpu', 'a refused choice replaced the last one'",
            "App.set_backend(None)",
            "assert frontend._BACKEND_CHOICE is None",
            "print('ok')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip().splitlines()[-1] == "ok", result.stdout
