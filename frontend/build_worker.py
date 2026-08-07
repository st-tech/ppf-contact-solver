# File: frontend/build_worker.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build pipeline subprocess driver. ``ppf-cts-server`` (Rust binary)
# spawns this script when handling ``DoSpawnBuild``: it cannot link
# against libpython without widening its dependency footprint, so the
# decode + tetrahedralize + FixedScene work runs here.
#
# Wire format (stdout, line-buffered):
#
#   PROGRESS percent=NN info=<text>\n   per-stage progress update
#   META frames=<int>\n                 total frames from param.pickle
#   ERROR <message>\n                   fatal error, followed by exit 1
#   ERRORDETAIL <text>\n                one traceback line of that ERROR
#
# Every marker occupies exactly one physical line: ``_emit`` folds
# embedded line breaks into spaces, so no payload it writes can split one
# marker into two or be read as another marker. Other frontend code also
# prints to this stream; the reader discards any line it does not
# recognize, so that text costs a debug-log line and nothing more.
#
# Side channel (file): on a scene-validation failure the structured
# violation payload (world-space geometry for the viewport overlay) is
# written to ``<root>/build_violations.json``. The stdout protocol only
# carries a flat ERROR string, so geometry travels via this file; the
# server reads it back on failure and forwards it to the add-on.
#
# Cancellation: the parent sends SIGTERM. The handler raises
# ``KeyboardInterrupt``, BlenderApp.populate().make() unwinds, and we
# exit with code 130 so the parent can distinguish cancel from crash.

import faulthandler
import json
import os
import signal
import sys
import traceback

from typing import Optional

# Dump a Python-level traceback to stderr on a hard crash (SIGSEGV /
# Windows access violation, etc.). A native crash inside a C extension,
# e.g. an ABI-mismatched scipy/numpy blowing up in the SuperLU solve of
# the partial-pin SOLID harmonic extension, or a tetrahedralizer binding,
# kills this worker WITHOUT raising a Python exception, so the ``except``
# below never runs and no ``ERROR`` line is emitted. The parent then only
# sees the bare exit code and reports the useless "build worker exited
# with code 1". faulthandler turns that into a stack that names the exact
# frame, which the server forwards to ``server.log`` as ``[BUILD stderr]``.
#
# ``all_threads=False`` is load-bearing, not a default worth changing back.
# The dump is written by the signal handler as one raw ``write(2)`` per
# fragment, and with every thread enabled CPython emits the NON-current
# threads first and the crashing thread LAST. Walking another thread's
# ``PyThreadState`` is itself a dereference, so a crash that happens while
# the interpreter is tearing those states down (a native fault during
# ``Py_FinalizeEx``) faults a SECOND time inside the handler, on the first
# foreign state it touches. The dump then stops after the literal bytes
# ``Thread 0x`` and the stack that named the fault is never written. That
# is measured, not theorized: pointing faulthandler at a SOCK_DGRAM socket
# makes each write one countable datagram, and write #0 is ``Thread 0x``
# while write #1 is the first read out of the thread state. Restricting
# the dump to the crashing thread touches only the current state, so the
# frames arrive first and survive. The cost is the other threads' stacks,
# which is the right trade: they are worth nothing when the dump dies
# before reaching the one stack that explains the crash.
faulthandler.enable(all_threads=False)


def _force_utf8_streams() -> None:
    """Pin stdout / stderr to UTF-8 so a non-ASCII report cannot be lost.

    The default stdout encoding follows the console code page on Windows
    (cp932 on a Japanese system). A scene object, path, or exception
    message carrying a character outside that page then raises
    ``UnicodeEncodeError`` inside the very handler that is reporting the
    failure, and the parent sees an empty stdout plus a bare exit code.
    ``backslashreplace`` keeps an unencodable character legible rather
    than dropping the line. The parent also sets ``PYTHONIOENCODING``,
    which covers an import-time failure that raises before this runs.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except AttributeError:
            # A harness may substitute a stream that is not a TextIOWrapper.
            pass


_force_utf8_streams()

# Caps on what the report puts on the wire. stderr always carries the
# untruncated traceback, so these bound the protocol, not the record.
MAX_DETAIL_LINES = 40
MAX_DETAIL_LINE_CHARS = 500

# Per-segment caps for the headline. Each segment is clipped on its own
# before the join, so a long value costs only its own tail and every
# other segment reaches the wire whole. Each cap is sized from the
# longest value this tree can produce for that segment, and each segment
# is written so that its most actionable text comes first, which is what
# makes a clip cost the least informative part of it.
#
#   message      the scene-validation report is the longest one a build
#                raises: four violation clauses prefixed with an object
#                name, 390 characters at a full 63-character name.
#   stage        the longest per-object progress beat is 163 characters
#                at that same name length. The build-planning beat
#                enumerates up to five objects and runs longer, and it
#                opens with the job count, so a clip keeps the count.
#   frame        the file, line and function are short; the source text
#                appended after them is arbitrary and clips first.
#   caused by    a second opinion on a failure the message already
#                names, so it is the segment worth the least.
#   os fields    errno, winerror and strerror total 79 characters; the
#                two filenames after them are paths, and one full
#                Windows MAX_PATH filename fits inside this cap.
#   interpreter  version and platform are short; the executable is a
#                path, and a full Windows MAX_PATH one fits.
MAX_MESSAGE_CHARS = 400
MAX_STAGE_CHARS = 200
MAX_FRAME_CHARS = 200
MAX_CAUSE_CHARS = 200
MAX_OS_FIELDS_CHARS = 400
MAX_INTERPRETER_CHARS = 320

_SEPARATOR = " | "

# The bound on an assembled headline: every segment above plus the
# separators between them. Derived rather than declared, so the clip in
# ``_error_headline`` is a backstop against an unbounded segment being
# added later and never fires on the six that exist.
MAX_HEADLINE_CHARS = (
    MAX_MESSAGE_CHARS
    + MAX_STAGE_CHARS
    + MAX_FRAME_CHARS
    + MAX_CAUSE_CHARS
    + MAX_OS_FIELDS_CHARS
    + MAX_INTERPRETER_CHARS
    + 5 * len(_SEPARATOR)
)

# Floor on what each exception in a chain keeps once the detail is over
# budget. Four lines hold the per-block elision marker, one frame (its
# ``File`` line plus the source line) and the ``Type: message`` line, so
# every exception still names itself and where it was raised.
MIN_BLOCK_DETAIL_LINES = 4

_TRUNCATION_MARKER = " ...(truncated)"

# Python's own wording between chained exceptions, reproduced so a
# budgeted report still reads as a traceback. The stdlib spellings are
# private, so they are stated here rather than imported.
_CAUSE_JOINER = "The above exception was the direct cause of the following exception:"
_CONTEXT_JOINER = "During handling of the above exception, another exception occurred:"

# Most recent ``_progress`` text, so a failure can name what the build was
# doing when it hit. Written only by ``_progress``.
_LAST_STAGE = ""

# Directory of the ``frontend`` package. A frame under it is project code;
# anything else is stdlib or a third-party library.
_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


def _warn(text: str) -> None:
    """Write ``text`` to stderr, or do nothing if that is not possible.

    Every caller runs while an exception is propagating or while a report
    about one is being assembled, so an exception raised by the write
    itself would replace the failure being reported with a failure to
    report it. ``sys.stderr`` is ``None`` under a GUI-subsystem
    interpreter (``pythonw.exe``, an embedded host with no console) and a
    harness may substitute a stream that is closed or absent, so the
    stream is resolved rather than assumed and every failure mode of the
    write is absorbed. Warning text is diagnostic; the exception on its
    way past is the record.
    """
    try:
        stream = sys.stderr
        if stream is None:
            return
        stream.write(text)
    except BaseException:
        pass


def _emit(line: str) -> None:
    """Write ``line`` to stdout as exactly one protocol line, then flush.

    The Rust side reads the stream line by line, so a payload carrying an
    embedded line break would arrive as two lines and the second one
    would be dropped as unrecognized, or worse be parsed as another
    marker. ``splitlines`` splits on every boundary the reader can split
    on and more, so folding it back with spaces makes the one-line
    property hold for any input.

    The guarantee is over ``_emit``'s own payloads. Other frontend code
    prints to this same stream (a verbose build narrates itself, and a
    successful build reports its scene checks), and the reader discards
    every line it cannot parse, so that text is harmless. The marker
    namespace, though, is not owned here: a ``print()`` added elsewhere
    whose text begins with ``ERROR `` would be read as the failure
    headline.

    ``flush()`` is required: without it the kernel pipe buffer can hold
    progress updates indefinitely and the UI freezes mid-build.
    """
    sys.stdout.write(" ".join(line.splitlines()))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _progress(progress: float, info: str) -> None:
    # Record the stage before emitting it so a failure raised by the work
    # this beat announces can name that work. ``_emit`` owns the
    # single-line guarantee for the text. The percent truncates to two
    # decimals worth of precision, which is plenty for a UI progress bar.
    global _LAST_STAGE
    _LAST_STAGE = info or ""
    pct = max(0.0, min(1.0, float(progress)))
    _emit(f"PROGRESS percent={pct:.4f} info={_LAST_STAGE}")


def _on_sigterm(_signum, _frame):  # pragma: no cover - signal path
    # Translate SIGTERM into KeyboardInterrupt so the build's Python
    # frames unwind through their normal exception machinery (closing
    # files, releasing GPU handles) instead of dying mid-syscall.
    raise KeyboardInterrupt("SIGTERM received")


def _exception_chain(
    exc: BaseException,
) -> list[tuple[BaseException, str]]:
    """``exc`` and its cause / context ancestors, in printed order.

    Python prints a chain root first and the raised exception last, so
    the list is ordered the same way. Each element pairs an exception
    with the joiner text printed directly after it, which is ``""`` for
    the last element.

    ``raise ... from e``, and any exception raised while handling
    another, keep the original attached, and the original is the one
    that names the operation that actually failed. ``__suppress_context__``
    marks an explicit ``from None``, i.e. the author declared the context
    irrelevant, so the walk stops there. The seen-set bounds the walk
    against a cycle assembled by hand (CPython will not set an
    exception's ``__context__`` to itself, but it does not stop an
    author from linking two exceptions to each other). ``raise e from e``
    is the shortest construct that produces one: it leaves
    ``e.__cause__ is e``, and the seen-set is what makes the chain come
    back as a single element instead of looping.
    """
    seen = {id(exc)}
    # Built outermost first, so each step records the joiner that belongs
    # between the exception it just found and the one it came from. That
    # joiner precedes the earlier exception in printed order, which is
    # what reversing at the end produces.
    walk: list[tuple[BaseException, str]] = [(exc, "")]
    cur = exc
    while True:
        nxt = cur.__cause__
        joiner = _CAUSE_JOINER
        if nxt is None and not cur.__suppress_context__:
            nxt = cur.__context__
            joiner = _CONTEXT_JOINER
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        walk.append((nxt, joiner))
        cur = nxt
    walk.reverse()
    return walk


def _headline_root(exc: BaseException) -> BaseException:
    """Deepest exception reached from ``exc`` through explicit causes.

    The headline names one exception's location, one exception's OS
    fields and at most one cause, and all three have to describe the same
    exception or they point a reader at three different places. This is
    that one exception.

    Only ``__cause__`` is followed, which an author sets with
    ``raise ... from``, i.e. by asserting that the earlier failure is why
    this one happened. ``__context__`` carries any exception that merely
    happened to be in flight, so an optional-dependency probe
    (``try: import scipy`` / ``except ImportError:``) that failed and was
    handled sits on the chain of every exception raised in its handler
    even though nothing about the failure came from it.

    The ERRORDETAIL block keeps walking ``__context__``: a traceback is a
    record and an incidental exception is part of what happened, while
    the headline is a verdict and has room for one subject.

    The seen-set bounds the walk against a hand-assembled cycle, the same
    guard and for the same reason as ``_exception_chain``.
    """
    seen = {id(exc)}
    cur = exc
    while True:
        nxt = cur.__cause__
        if nxt is None or id(nxt) in seen:
            return cur
        seen.add(id(nxt))
        cur = nxt


def _clip(text: str, limit: int) -> str:
    """``text`` bounded to ``limit`` characters, marked when shortened."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _project_frame(exc: BaseException) -> Optional[traceback.FrameSummary]:
    """Deepest traceback frame of ``exc`` that lies inside this package.

    The innermost frame of a failing OS call belongs to the standard
    library (``zipfile`` for ``np.savez``, ``subprocess`` for a spawn),
    which names the syscall but not the operation the build was
    performing. The deepest frame under ``frontend`` is the project code
    that made the call, so it is the label a reader can act on, and it is
    derived rather than hand-annotated so every call site gets one.

    Falls back to the innermost frame when no frame belongs to the
    package, which is what an import-time failure looks like, and returns
    ``None`` for an exception that was never raised (so it carries no
    traceback).
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    for frame in reversed(frames):
        try:
            path = os.path.normcase(os.path.abspath(frame.filename))
        except (OSError, ValueError):
            # A filename that cannot be resolved at all (an unreadable
            # cwd, an embedded NUL) is not one of this package's.
            continue
        if path.startswith(_PACKAGE_DIR + os.sep):
            return frame
    return frames[-1]


def _os_error_fields(exc: BaseException) -> str:
    """OS-level fields of an ``OSError``, or ``""`` for anything else.

    All five are printed even when unset: ``winerror=None`` beside a
    populated ``errno`` says the failure came from the C runtime rather
    than from Win32, which is the distinction that narrows a Windows root
    cause, so an absent field is a measurement and not an omission.
    ``winerror`` needs ``getattr`` because the attribute exists only on
    Windows.
    """
    if not isinstance(exc, OSError):
        return ""
    return (
        f"errno={exc.errno} winerror={getattr(exc, 'winerror', None)} "
        f"strerror={exc.strerror!r} filename={exc.filename!r} "
        f"filename2={exc.filename2!r}"
    )


def _error_headline(exc: BaseException) -> str:
    """One-line summary of ``exc``: what failed, where, and under what.

    The head stays ``"<Type>: <message>"``, which is what the server
    matches on to classify a missing dependency and what the add-on panel
    shows, so the fields appended after it are context and never change
    the identity of the failure.

    The two surfaces that show this line clip it: a Blender panel label
    shows on the order of 45 characters at a typical sidebar width, and
    the operator status bar is no wider. Segments are therefore ordered
    by how far they get a reader who sees only the opening: the stage and
    the frame localize the failure, so they come directly after the
    identity, while the cause, the OS fields and the interpreter answer
    questions a reader asks after already knowing what broke and where.

    ``caused by`` is emitted only when it carries text the message does
    not already have. The wrapping idiom on this path prefixes an object
    name to the cause's own message, so the clause would otherwise repeat
    the head verbatim and spend the budget saying it twice.
    """
    root = _headline_root(exc)
    parts = [_clip(f"{type(exc).__name__}: {exc}", MAX_MESSAGE_CHARS)]
    if _LAST_STAGE:
        parts.append(_clip(f"while: {_LAST_STAGE}", MAX_STAGE_CHARS))
    frame = _project_frame(root)
    if frame is not None:
        head, tail = os.path.split(frame.filename)
        short = f"{os.path.basename(head)}/{tail}" if head else tail
        where = f"at {short}:{frame.lineno} in {frame.name}"
        if frame.line:
            # The source text names which call on a multi-statement line
            # failed, so it is kept and clipped rather than dropped.
            where += f": {frame.line}"
        parts.append(_clip(where, MAX_FRAME_CHARS))
    if root is not exc and str(root) not in str(exc):
        parts.append(
            _clip(f"caused by {type(root).__name__}: {root}", MAX_CAUSE_CHARS)
        )
    fields = _os_error_fields(root) or _os_error_fields(exc)
    if fields:
        parts.append(_clip(fields, MAX_OS_FIELDS_CHARS))
    parts.append(
        _clip(
            f"python {sys.version.split()[0]} {sys.platform} at {sys.executable}",
            MAX_INTERPRETER_CHARS,
        )
    )
    headline = _SEPARATOR.join(parts)
    # Backstop on the assembled line. Every segment above is bounded and
    # ``MAX_HEADLINE_CHARS`` is their sum, so this catches only a segment
    # added later without a cap of its own.
    return _clip(headline, MAX_HEADLINE_CHARS)


def _budgeted_detail(exc: BaseException, total_lines: int) -> list[str]:
    """Traceback lines for ``exc`` fitted into ``MAX_DETAIL_LINES``.

    The budget is spread across the exceptions in the chain and each one
    keeps its own tail, so every exception contributes its innermost
    frames and its ``Type: message`` line. Slicing the flattened
    traceback instead would keep only its tail, and Python prints a
    chain root first, so a chain whose outer exceptions fill the budget
    would report the wrappers and drop the failure they wrap.

    Short blocks take less than an equal share and hand the remainder to
    the longer ones, so the budget is spent rather than reserved.
    """
    blocks = [
        (
            "".join(traceback.format_exception(item, chain=False)).splitlines(),
            joiner,
        )
        for item, joiner in _exception_chain(exc)
    ]
    # A chain can be longer than the budget can seat at the per-block
    # floor. Keep the root, which names the failure, and the outermost
    # exceptions, which name what was being attempted, and collapse the
    # middle: an intermediate wrapper is the least informative part of a
    # chain. Each seated block costs its own lines plus the joiner
    # printed after it, and the summary line costs one.
    chain_length = len(blocks)
    omitted = 0
    if len(blocks) > (MAX_DETAIL_LINES - 1) // (MIN_BLOCK_DETAIL_LINES + 1):
        # One line announces the collapsed run, so it takes a seat too.
        seats = max(1, (MAX_DETAIL_LINES - 2) // (MIN_BLOCK_DETAIL_LINES + 1))
        if seats > 1:
            blocks = blocks[:1] + blocks[1 - seats:]
        else:
            blocks = blocks[:1]
        omitted = chain_length - len(blocks)

    budget = MAX_DETAIL_LINES - 1 - sum(1 for _, j in blocks if j)
    if omitted:
        budget -= 1
    keep: dict[int, int] = {}
    remaining = len(blocks)
    # Ascending length, so a block that wants less than its share
    # releases the difference to the blocks still unassigned.
    for index in sorted(range(len(blocks)), key=lambda i: len(blocks[i][0])):
        share = max(MIN_BLOCK_DETAIL_LINES, budget // remaining)
        keep[index] = min(len(blocks[index][0]), share)
        budget -= keep[index]
        remaining -= 1
    out: list[str] = []
    for index, (lines, joiner) in enumerate(blocks):
        kept = keep[index]
        if kept < len(lines):
            # The marker is one of this block's kept lines, so a block
            # never costs more than it was allotted.
            elided = len(lines) - kept + 1
            lines = [f"  ...({elided} line(s) elided)"] + lines[-(kept - 1):]
        out.extend(lines)
        if joiner:
            out.append(joiner)
        if omitted and index == 0:
            out.append(f"...({omitted} intermediate exception(s) omitted)")
    # Counted from what was actually built, so the figure cannot drift
    # from the output it describes.
    out.insert(
        0,
        f"showing {len(out)} of {total_lines} traceback line(s) across "
        f"{chain_length} chained exception(s); "
        "full traceback in the server log",
    )
    return out


def _report_failed(err: BaseException, step: str) -> None:
    """Send a failure of the reporting itself to stderr.

    The stdout protocol carries a type name for these, because the report
    is being assembled for an exception that may raise from its own
    ``__str__`` and re-entering it is what broke the step. A type name
    does not say which of the report's steps raised, so the record goes
    here, where the server picks it up as ``[BUILD stderr]``. The write
    is guarded, so this cannot become the third failure in the stack.
    """
    _warn(f"build worker: {step} failed\n")
    try:
        stream = sys.stderr
        if stream is None:
            return
        traceback.print_exception(type(err), err, err.__traceback__, file=stream)
    except BaseException:
        pass


def _emit_error(exc: BaseException) -> None:
    """Report ``exc`` on stdout: one ERROR headline, then ERRORDETAIL lines.

    The headline goes first so a worker killed mid-dump still delivers
    the verdict, and the parent keeps the detail that follows the last
    headline it saw.
    """
    try:
        headline = _error_headline(exc)
    except BaseException as report_err:
        # Only type names here: an exception whose ``__str__`` or
        # ``__repr__`` raises is the case this branch exists for, so
        # nothing that could re-enter user code goes into the message.
        # The type name alone cannot say which step of the report broke,
        # so the reporting failure's own traceback goes to stderr and
        # reaches the server log as ``[BUILD stderr]``.
        _report_failed(report_err, "building the error report")
        _emit(
            f"ERROR {type(exc).__name__}: building the error report failed "
            f"({type(report_err).__name__}); see the build worker's stderr"
        )
        return
    _emit(f"ERROR {headline}")
    try:
        detail = "".join(traceback.format_exception(exc)).splitlines()
    except BaseException as fmt_err:
        _report_failed(fmt_err, "formatting the traceback")
        _emit(
            f"ERRORDETAIL formatting the traceback failed "
            f"({type(fmt_err).__name__})"
        )
        return
    if len(detail) > MAX_DETAIL_LINES:
        try:
            detail = _budgeted_detail(exc, len(detail))
        except BaseException as budget_err:
            # Falling back to the flat tail keeps a report on the wire.
            # It can drop the root cause of a long chain, which is the
            # whole reason the budgeted form exists, so say so rather
            # than letting a shorter report look like a complete one.
            _report_failed(budget_err, "budgeting the traceback")
            detail = [
                f"budgeting the traceback failed "
                f"({type(budget_err).__name__}); showing the tail only, "
                f"which may omit the root cause"
            ] + detail[-MAX_DETAIL_LINES:]
    for text in detail:
        _emit(f"ERRORDETAIL {text[:MAX_DETAIL_LINE_CHARS]}")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        _emit("ERROR usage: build_worker.py <name> <root>")
        return 2
    name = argv[1]
    root = argv[2]
    # Optional trailing flag: when present, the build re-decodes the scene
    # input while keeping the solver `output/` subtree (saved checkpoints),
    # so a resume can pick up edited animation without wiping the states.
    # Parsed from argv[3:] to leave the `<name> <root>` positional contract
    # intact.
    preserve_output = "--preserve-output" in argv[3:]

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Patches the production ``frontend`` module when the test rig
        # invokes us with ``PPF_CTS_DATA_ROOT`` set. Production runs
        # without the env var and skip this entirely. Importing the
        # helper here (not at module top) keeps the ``ERROR`` path
        # clean if the frontend package fails to import.
        from frontend import _debug_runtime_
        _debug_runtime_.install_debug_patches()

        # Imported lazily so ``ERROR`` can still be reported if the
        # frontend package or its dependencies fail to import.
        from frontend import BlenderApp

        app = BlenderApp(name, progress_callback=_progress)
        # ppf-cts-server stores `data.pickle` and `param.pickle` under
        # the path the addon supplied (its remote `current_directory`
        # plus the project name) and passes that path back as `root`.
        # When that differs from the canonical `<data_dirpath>/<name>`
        # BlenderApp computed for itself, override `BlenderApp._root`
        # so the build worker reads from the same location the addon
        # wrote to. Skip when the test rig already steered us via
        # `PPF_CTS_DATA_ROOT`.
        if not os.environ.get("PPF_CTS_DATA_ROOT") and root != app._root:
            app._data_dirpath = os.path.dirname(root)
            app._root = root
            cache_root = os.path.join(root, ".cash")
            os.makedirs(cache_root, exist_ok=True)
            if hasattr(app, "_mesh_manager"):
                # `set_cache_dir` redirects BOTH `MeshManager._cache_dir`
                # and its `CreateManager._cache_dir`. The tetra cache path
                # is derived through `create.tri(...)` off the latter, so
                # updating only `MeshManager._cache_dir` left the per-project
                # tet cache pointed at the canonical path
                # `BlenderApp.__init__` computed via `blender_app_paths`,
                # while the upload pickles lived next to the project root.
                # On a fresh project name (or after the canonical cache was
                # lost), every build re-ran fTetWild from scratch and that's
                # intrinsically non-deterministic across runs. Two builds of
                # the same SOLID mesh produced different tet hulls (~1% size
                # difference); for a SHELL tucked inside the SOLID's
                # contact-gap zone, that difference flipped the kite
                # between "free" and "in contact" and the user saw the
                # kite locally wrinkle "for no reason."
                app._mesh_manager.set_cache_dir(cache_root)
        app.populate().make(preserve_output=preserve_output)
        # Forward static build metadata the response builder needs to
        # publish solver progress (frame / total_frames). Pulled from
        # the FixedSession's resolved param set so static + dyn merges
        # are respected. Read once here so the META line and the
        # scene_info "Total Frames" row share one value and one guard.
        # Best-effort: a missing or non-int `frames` becomes None.
        try:
            total_frames = int(app.session._param.get("frames"))
        except (AttributeError, TypeError, ValueError):
            total_frames = None
        if total_frames is not None and total_frames > 0:
            _emit(f"META frames={total_frames}")
        # Drop a scene_info.json next to the project so ppf-cts-server's
        # response builder can splice it into every status response. The
        # build runs in this subprocess, so any field derived from the
        # parsed app must be persisted to disk before we exit. The
        # addon's panel renders this dict as the "Scene Info" box after
        # a successful build.
        try:
            info: dict[str, str] = {}
            fmt = lambda n: f"{n:,}"
            fs = app._fixed_scene
            if fs is not None:
                if fs._vert is not None and len(fs._vert) > 0:
                    info["Vertices"] = fmt(len(fs._vert[0]))
                if fs._tri is not None:
                    info["Triangles"] = fmt(len(fs._tri))
                if fs._tet is not None:
                    info["Tetrahedra"] = fmt(len(fs._tet))
                if fs._rod is not None and len(fs._rod) > 0:
                    info["Rod Edges"] = fmt(len(fs._rod))
            sd = getattr(app, "_scene_decoder", None)
            if sd is not None and getattr(sd, "_data", None) is not None:
                n_dynamic = 0
                n_static = 0
                ref_groups: dict[str, str] = {}
                canonical_refs: dict[str, set[str]] = {}
                for group in sd._data:
                    gt = group.get("type", "")
                    objs = group.get("object", [])
                    if gt == "STATIC":
                        n_static += len(objs)
                    else:
                        n_dynamic += len(objs)
                    for obj in objs:
                        mesh_ref = obj.get("mesh_ref")
                        if mesh_ref:
                            ref_groups.setdefault(mesh_ref, gt)
                            canonical_refs.setdefault(mesh_ref, set()).add(
                                obj.get("name", "")
                            )
                            canonical_refs[mesh_ref].add(mesh_ref)
                info["Dynamic Objects"] = fmt(n_dynamic)
                info["Static Objects"] = fmt(n_static)
                if canonical_refs:
                    by_type: dict[str, list[int]] = {}
                    for ref_name, names in canonical_refs.items():
                        gt = ref_groups.get(ref_name, "")
                        by_type.setdefault(gt, []).append(len(names))
                    for gt in sorted(by_type):
                        counts = sorted(by_type[gt], reverse=True)
                        label = f"Shared {gt.capitalize()}s"
                        info[label] = "(" + ",".join(str(c) for c in counts) + ")"
            # Static session params (resolved by `make()` from static +
            # dyn merges) populate the "Total Frames" / "FPS" rows.
            # Dynamic rows ("Simulated Frames" / "Last Saved") are added
            # by the response builder per poll, since they change as the
            # solver runs.
            if total_frames is not None and total_frames > 0:
                info["Total Frames"] = fmt(total_frames)
            try:
                fss = app.session
                if fss is not None and getattr(fss, "_param", None) is not None:
                    fps = fss._param.get("fps")
                    if fps is not None:
                        # Float-safe: the session fps carries Time Scale
                        # (scene fps * time_scale) and is fractional whenever
                        # the scale is not 1; int() would misreport the rate
                        # the solver is actually running at.
                        info["FPS"] = f"{float(fps):g}"
            except Exception:
                pass
            with open(os.path.join(root, "scene_info.json"), "w") as fp:
                json.dump(info, fp)
        except Exception as exc:
            _warn(f"scene_info write failed: {exc}\n")
        # Final progress beat so the parent sees 100% before EOF.
        _progress(1.0, "Build complete.")
        return 0
    except KeyboardInterrupt:
        # Cancel path: stay silent on stdout (parent already knows it
        # asked for cancel) and exit with the conventional cancel code.
        return 130
    except SystemExit as exc:
        # Honor explicit exit codes, but route any error message through
        # the wire format so the parent can surface it. A bare
        # ``sys.exit()`` / ``raise SystemExit`` leaves ``exc.code`` None,
        # which by Python convention means success, so map it to 0.
        if exc.code is None:
            return 0
        code = int(exc.code) if isinstance(exc.code, int) else 1
        if exc.code and not isinstance(exc.code, int):
            _emit(f"ERROR {exc.code}")
        return code
    except BaseException as exc:
        # When scene validation fails, ValidationError carries a
        # structured ``violations`` payload (self-intersecting triangles,
        # contact-offset pairs, wall/sphere hits) with world-space
        # geometry. Persist it to a sidecar the server reads back on
        # failure and forwards to the add-on, which highlights the
        # offending faces in the viewport. ``getattr`` keeps this a
        # no-op for ordinary errors that carry no violations.
        violations = getattr(exc, "violations", None)
        if violations:
            try:
                with open(os.path.join(root, "build_violations.json"), "w") as fp:
                    json.dump({"violations": violations}, fp)
            except Exception as werr:  # best-effort; never mask the build error
                _warn(f"build_violations write failed: {werr}\n")
        _emit_error(exc)
        # stderr carries the untruncated traceback for the server log, so
        # the caps on the stdout report bound the protocol and never the
        # record; stdout stays parseable for the line-oriented protocol.
        # Guarded like every other write on this path: the ERROR line is
        # already on the wire, and the record must not cost the verdict.
        try:
            stream = sys.stderr
            if stream is not None:
                traceback.print_exc(file=stream)
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
