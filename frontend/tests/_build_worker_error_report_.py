# File: frontend/tests/_build_worker_error_report_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for the build worker's failure report.
#
# The worker is the only thing between a Python exception raised anywhere
# in the build and what the add-on shows the user, and it speaks a
# line-oriented protocol the Rust server parses. The properties fixed
# here are the ones that decide whether a failure is actionable:
#
#   * every marker is exactly one physical line, whatever the payload;
#   * the headline still starts with ``<Type>: <message>``, which is what
#     the server matches on to classify a missing dependency;
#   * the reported frame is the deepest one inside the ``frontend``
#     package, not the innermost stdlib frame (``zipfile`` for a failing
#     ``np.savez``, ``subprocess`` for a failing spawn), so it names the
#     operation instead of the syscall;
#   * an explicit cause and the OS-level fields of an ``OSError`` survive;
#   * a non-UTF-8 stdio encoding cannot silence the report;
#   * a failure of the report itself is diagnosable rather than a bare
#     type name, and cannot become the failure that is reported;
#   * cleanup in ``_mesh_``'s ``finally`` cannot replace the real error,
#     including on an interpreter that has no stderr to warn on.

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager

import numpy as np
import pytest

try:
    from .. import build_worker as bw
    from .._mesh_ import TriMesh
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


FRONTEND_DIR = os.path.dirname(os.path.abspath(bw.__file__))


@contextmanager
def _worker_stdout():
    """Capture the worker's protocol output and isolate its stage global."""
    buf = io.StringIO()
    saved_out, saved_stage = sys.stdout, bw._LAST_STAGE
    sys.stdout = buf
    bw._LAST_STAGE = ""
    try:
        yield buf
    finally:
        sys.stdout = saved_out
        bw._LAST_STAGE = saved_stage


def _lines(buf):
    # Every ``_emit`` ends in a newline, so the trailing split field is
    # always empty and dropping it never hides a line.
    return buf.getvalue().split("\n")[:-1]


def _caught(fn):
    """Run ``fn``, return the exception it raised (with its traceback)."""
    try:
        fn()
    except BaseException as exc:
        return exc
    raise AssertionError("expected the callable to raise")


def _report(exc):
    """Emitted lines for ``exc``, as ``(error_line, detail_lines)``."""
    with _worker_stdout() as buf:
        bw._emit_error(exc)
    return _split_report(_lines(buf))


def _split_report(lines):
    error_lines = [ln for ln in lines if ln.startswith("ERROR ")]
    detail_lines = [ln for ln in lines if ln.startswith("ERRORDETAIL ")]
    assert len(error_lines) == 1, f"expected one ERROR line, got {lines!r}"
    return error_lines[0], detail_lines


def _raise(exc):
    raise exc


def _raise_from_self():
    try:
        raise ValueError("recursive cause")
    except ValueError as e:
        raise e from e


class _UnprintableError(Exception):
    """An exception the report cannot format.

    A PyO3 error object backed by a Rust type is the shape this stands
    in for: the report asks for the message and the ask itself raises.
    """

    def __str__(self):
        raise RuntimeError("__str__ is broken")


def _the_os_call_that_failed():
    raise OSError(22, "Invalid argument")


def _wrap_the_failure():
    try:
        _the_os_call_that_failed()
    except OSError as e:
        raise ValueError("Rock: could not tetrahedralize") from e


def _savez_into_a_missing_directory(path):
    np.savez(path, vert=np.zeros((2, 3)))


def _ping(depth):
    # Alternating with _pong so no two consecutive frames share a source
    # line: `traceback` collapses a repeated line into "[Previous line
    # repeated N more times]", which would keep the dump short and leave
    # the cap unexercised.
    if depth == 0:
        raise RuntimeError("bottom of the recursion")
    _pong(depth - 1)


def _pong(depth):
    _ping(depth - 1)


def test_error_line_is_single_line():
    exc = _caught(
        lambda: _raise(RuntimeError("line one\nline two\r\nthree\rtail"))
    )
    with _worker_stdout() as buf:
        bw._emit_error(exc)
    lines = _lines(buf)
    error_line, _ = _split_report(lines)
    # An embedded break would be read as a second line by the server and
    # either dropped or parsed as another marker.
    assert not [ln for ln in lines if ln.startswith(("PROGRESS ", "META "))]
    for fragment in ("line one", "line two", "three", "tail"):
        assert fragment in error_line, f"lost {fragment!r} from {error_line!r}"


def test_headline_head_is_type_colon_message():
    exc = _caught(
        lambda: _raise(ModuleNotFoundError("No module named 'pytetwild'"))
    )
    error_line, _ = _report(exc)
    # The server classifies a missing dependency by substring, so the head
    # must stay exactly what it was before any context was appended.
    assert error_line.startswith(
        "ERROR ModuleNotFoundError: No module named 'pytetwild'"
    )


def test_os_error_fields_print_all_five():
    exc = _caught(lambda: _raise(OSError(22, "Invalid argument")))
    error_line, _ = _report(exc)
    for field in (
        "errno=22",
        "winerror=",
        "strerror='Invalid argument'",
        "filename=None",
        "filename2=None",
    ):
        assert field in error_line, f"lost {field!r} from {error_line!r}"


def test_frame_is_the_deepest_project_frame():
    missing = os.path.join(
        tempfile.gettempdir(), "ppf-cts-no-such-directory", "input.npz"
    )
    exc = _caught(lambda: _savez_into_a_missing_directory(missing))
    error_line, _ = _report(exc)
    # np.savez fails several frames deep in numpy and the standard
    # library; the frame worth reporting is the project call that made it.
    assert " in _savez_into_a_missing_directory" in error_line
    assert os.path.basename(__file__) in error_line
    for stdlib in ("zipfile", "_npyio_impl"):
        assert stdlib not in error_line, f"reported {stdlib} frame: {error_line!r}"


def test_reports_root_cause_frame():
    exc = _caught(_wrap_the_failure)
    error_line, _ = _report(exc)
    assert error_line.startswith("ERROR ValueError: Rock: could not tetrahedralize")
    assert "caused by OSError: [Errno 22] Invalid argument" in error_line
    # The frame comes from the root cause, so it names the call that
    # failed rather than the handler that re-raised.
    assert " in _the_os_call_that_failed" in error_line
    assert " in _wrap_the_failure" not in error_line


def test_names_last_progress_stage():
    stage = "Tetrahedralizing Rock_01 (1/1, new)..."
    exc = _caught(lambda: _raise(RuntimeError("boom")))
    with _worker_stdout() as buf:
        bw._progress(0.17, stage)
        bw._emit_error(exc)
    error_line, _ = _split_report(_lines(buf))
    assert f"while: {stage}" in error_line


def test_detail_carries_traceback_and_interpreter():
    exc = _caught(_wrap_the_failure)
    error_line, detail_lines = _report(exc)
    body = "\n".join(detail_lines)
    assert "Traceback (most recent call last):" in body
    assert "OSError: [Errno 22] Invalid argument" in body
    assert sys.executable in error_line


def test_detail_lines_capped_and_elision_noted():
    exc = _caught(lambda: _ping(200))
    _, detail_lines = _report(exc)
    assert len(detail_lines) <= bw.MAX_DETAIL_LINES
    assert any("traceback line(s) across" in ln for ln in detail_lines)
    # Each block keeps its tail, so the exception line itself survives.
    assert detail_lines[-1].endswith("RuntimeError: bottom of the recursion")


def _chained(depth: int) -> BaseException:
    """An ``OSError`` wrapped ``depth`` times, innermost raised first."""

    def wrap(n):
        if n == 0:
            raise OSError(22, "Invalid argument")
        try:
            wrap(n - 1)
        except Exception as err:
            raise RuntimeError(f"stage {n} failed") from err

    return _caught(lambda: wrap(depth))


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8, 20, 60])
def test_root_cause_survives_the_detail_cap(depth):
    # Python prints a chain root first, so slicing the flattened
    # traceback to its tail reports the wrappers and drops the failure
    # they wrap. The budget is spread per exception instead, so the
    # exception that actually failed is always on the wire.
    exc = _chained(depth)
    _, detail_lines = _report(exc)
    body = "\n".join(detail_lines)
    assert "OSError: [Errno 22] Invalid argument" in body
    assert len(detail_lines) <= bw.MAX_DETAIL_LINES


def test_flat_tail_would_lose_the_root_cause_at_this_depth():
    # Establishes that at this chain depth the last MAX_DETAIL_LINES of
    # the flattened traceback carry none of the root exception. That is
    # what makes the parametrized test above a gate: the property it
    # asserts is one a plain tail slice cannot satisfy here.
    exc = _chained(8)
    flat = "".join(traceback.format_exception(exc)).splitlines()
    tail = flat[-bw.MAX_DETAIL_LINES:]
    assert not any(ln.startswith("OSError: [Errno 22]") for ln in tail)


def test_detail_under_the_cap_is_the_verbatim_traceback():
    # A report that fits is not rewritten at all, so the common case
    # reads exactly as Python would print it.
    exc = _chained(1)
    _, detail_lines = _report(exc)
    expected = "".join(traceback.format_exception(exc)).splitlines()
    assert len(expected) <= bw.MAX_DETAIL_LINES
    assert [ln[len("ERRORDETAIL "):] for ln in detail_lines] == expected


def test_non_ascii_survives_a_non_utf8_locale():
    # Reproduces the Windows console code page on any platform: cp932
    # cannot encode U+1F4A5, so without the UTF-8 pin the report raises
    # UnicodeEncodeError inside its own handler and stdout stays empty.
    # The child source is pure ASCII with escapes so the result cannot
    # depend on how the command line itself is decoded.
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{FRONTEND_DIR}')\n"
        "import build_worker\n"
        "try:\n"
        "    raise RuntimeError('\\u30c6\\u30b9\\u30c8 \\U0001f4a5')\n"
        "except BaseException as exc:\n"
        "    build_worker._emit_error(exc)\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp932"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")
    error_line, _ = _split_report(text.split("\n")[:-1])
    assert "\u30c6\u30b9\u30c8" in error_line


def test_progress_stays_single_line():
    with _worker_stdout() as buf:
        bw._progress(0.5, "stage a\nstage b\r\nstage c")
    lines = _lines(buf)
    assert lines == ["PROGRESS percent=0.5000 info=stage a stage b stage c"]


def test_a_self_caused_chain_terminates():
    # ``raise e from e`` leaves ``e.__cause__ is e``, and it is the
    # shortest construct that makes a chain self-referential: CPython
    # refuses to set ``__context__`` to the exception itself, so the
    # explicit forms are the only source of one. Both walks bound
    # themselves with a seen-set; without it each spins on the same
    # object and the report never reaches the wire.
    exc = _caught(_raise_from_self)
    assert exc.__cause__ is exc
    assert [item for item, _ in bw._exception_chain(exc)] == [exc]
    assert bw._headline_root(exc) is exc
    error_line, _ = _report(exc)
    assert error_line.startswith("ERROR ValueError: recursive cause")


def test_report_failure_is_diagnosable_from_stderr(capsys):
    # The wire carries a type name for this case, because re-entering
    # the exception is what broke the report. A type name cannot say
    # which of the report's steps raised, so the reporting failure's own
    # traceback goes to stderr, where the server picks it up.
    exc = _caught(lambda: _raise(_UnprintableError()))
    with _worker_stdout() as buf:
        bw._emit_error(exc)
    error_line, _ = _split_report(_lines(buf))
    assert error_line.startswith(
        "ERROR _UnprintableError: building the error report failed (RuntimeError)"
    )
    err = capsys.readouterr().err
    assert "building the error report failed" in err, err
    assert "RuntimeError: __str__ is broken" in err, err
    # The frame is the point: it names where the report broke, which is
    # what a bare type name on the wire cannot.
    assert "in __str__" in err, err


def _tetrahedralize_with_denied_cleanup(monkeypatch):
    """Run the fTetWild path with both the savez and the unlink failing.

    Returns ``(sentinel, raised, attempted)``: the exception the build
    itself raised, the exception that reached the caller, and the temp
    files whose removal the cleanup was denied.
    """
    sentinel = OSError(22, "savez sentinel")
    attempted: list[str] = []

    def _raising_savez(path, **arrays):
        raise sentinel

    real_unlink = os.unlink

    def _raising_unlink(path, *args, **kwargs):
        # Deny only the fTetWild temp files. `tempfile` unlinks probe
        # files of its own while resolving the temp directory, and denying
        # those makes it report that no usable directory exists, which is
        # a different failure than the one under test.
        if not str(path).endswith(".npz"):
            return real_unlink(path, *args, **kwargs)
        attempted.append(path)
        raise PermissionError(13, "Permission denied", path)

    from .. import _mesh_

    monkeypatch.setattr(_mesh_.np, "savez", _raising_savez)
    monkeypatch.setattr(_mesh_.os, "unlink", _raising_unlink)
    mesh = TriMesh((np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int32)))
    try:
        raised = _caught(lambda: mesh._tetrahedralize_ftetwild({}, None, 5.0))
    finally:
        # Undone here so the leftovers below are removed through the real
        # ``os.unlink``, and so a test that patched ``sys.stderr`` gets it
        # back before pytest reports.
        monkeypatch.undo()
        for path in attempted:
            if os.path.exists(path):
                os.unlink(path)
    return sentinel, raised, attempted


def test_cleanup_failure_does_not_mask_the_real_error(monkeypatch, capsys):
    sentinel, raised, attempted = _tetrahedralize_with_denied_cleanup(monkeypatch)
    # A failing unlink in the cleanup would otherwise become the exception
    # the build reports, hiding the failure that actually stopped it.
    assert raised is sentinel, f"cleanup replaced the build error: {raised!r}"
    assert attempted, "cleanup never reached the temp file"
    assert "could not remove the fTetWild temp file" in capsys.readouterr().err


def test_cleanup_warning_survives_an_absent_stderr(monkeypatch):
    # ``pythonw.exe`` and an embedded host with no console leave
    # ``sys.stderr`` as ``None``, so the warning about the failed cleanup
    # has nowhere to go. An unguarded write raises ``AttributeError``,
    # which is not an ``OSError``, so it escapes the ``finally`` and
    # becomes the exception the build reports.
    monkeypatch.setattr(sys, "stderr", None)
    sentinel, raised, attempted = _tetrahedralize_with_denied_cleanup(monkeypatch)
    assert raised is sentinel, f"cleanup replaced the build error: {raised!r}"
    assert attempted, "cleanup never reached the temp file"
