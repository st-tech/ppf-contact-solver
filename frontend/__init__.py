# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Entry point for the frontend package.

Import :class:`App` from this package to start the application; the other
names listed in ``__all__`` expose the scene, session, mesh, plotting,
asset, and utility APIs.
"""

# Resolve the Rust extension module exactly once, at the top of the
# package. Submodules import it as ``from . import _rust`` so a missing
# build surfaces this single, actionable error instead of a cascade of
# ``ModuleNotFoundError`` from whichever submodule loaded first.
#
# Tree-local cdylib load. ``_ppf_cts_py`` is built by ``cargo build
# --release`` (it is a workspace default-member) into
# ``<tree-root>/target/<profile>/`` as ``lib_ppf_cts_py.so`` (Linux),
# ``lib_ppf_cts_py.dylib`` (macOS) or ``_ppf_cts_py.dll`` (Windows). We
# load that file DIRECTLY by absolute path via importlib and register it
# as ``sys.modules["_ppf_cts_py"]``, so two worktrees (e.g. one on `dev`,
# one on `main`) on the same host always pick up their own build with no
# shared venv / site-packages in the picture. There is no maturin wheel
# and no fallback: a missing cdylib is a hard, actionable error. The tree
# root is taken relative to *this file*, so the right tree's build wins
# regardless of caller cwd.
import importlib.machinery as _machinery
import importlib.util as _ilu
import os as _os
import subprocess as _subprocess
import sys as _sys

from . import _backends_

_FRONTEND_TREE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))


def _cdylib_filename():
    """The cargo cdylib filename for ``_ppf_cts_py`` on this platform."""
    if _sys.platform == "darwin":
        return "lib_ppf_cts_py.dylib"
    if _sys.platform == "win32":
        return "_ppf_cts_py.dll"
    return "lib_ppf_cts_py.so"


def _target_dirs():
    """The one target directory this process loads its build from, as a list.

    ``CARGO_TARGET_DIR`` when it is set, and ``<tree root>/target`` otherwise.
    A relative value is relative to the tree root, as Cargo reads it.

    The variable is honoured because it is how two backends are held at once:
    they link the same executable name, so ``crates/ppf-cts-solver/build.rs``
    refuses to overwrite one with another in a shared directory and tells the
    caller to give the second its own.  The same variable then has to reach
    here, or the build would succeed and the cdylib beside it would never be
    found.

    WHEN IT IS SET, IT IS THE ONLY PLACE SEARCHED.  Falling back to
    ``<tree root>/target`` when the named directory holds no cdylib loads a
    DIFFERENT build from the one the caller named, and ``artifact_dir`` then
    writes that build's solver into the session launcher, so a run that named
    one backend executes another with nothing reporting it.  An empty named
    directory is therefore the same hard error as an empty tree.
    ``ppf-cts-server`` restates this rule to report which build its runs use
    (``executor::build::target_dir_for``), so the two change together.
    """
    override = _os.environ.get("CARGO_TARGET_DIR")
    if override:
        return [
            override
            if _os.path.isabs(override)
            else _os.path.join(_FRONTEND_TREE_ROOT, override)
        ]
    return [_os.path.join(_FRONTEND_TREE_ROOT, "target")]


def _load_dirs():
    """Where the extension module is searched for, in order.

    ``_target_dirs`` first. When ``CARGO_TARGET_DIR`` is unset, the named
    backend directories follow the tree's own ``target``: a distribution carries
    one directory per backend and no ``target/release``, and any of their
    modules serves, because which backend a run uses is decided separately
    (``get_backend``) and checked against this module's source stamp before a
    run (``_run_directory``). The CPU directory comes first among them because
    its module imports no GPU runtime.
    """
    if _os.environ.get("CARGO_TARGET_DIR"):
        return _target_dirs()
    root = _os.path.join(_FRONTEND_TREE_ROOT, "target")
    return [root] + [
        _os.path.join(root, name) for name in ("cpu", "cuda", "rocm", "metal")
    ]


def _find_cdylib():
    """Locate the tree-local cdylib, preferring a release build over debug."""
    name = _cdylib_filename()
    for target in _load_dirs():
        for profile in ("release", "debug"):
            cand = _os.path.join(target, profile, name)
            if _os.path.isfile(cand):
                return cand
    return None


def backend_of(target_dir=None, profile="release"):
    """Which backend a built tree holds, or ``None`` if nothing is built.

    ``build.rs`` writes this beside the artifacts.  It answers the question the
    executable's fixed name cannot: every backend links ``ppf-contact-solver``,
    so the path says nothing about what is in it.
    """
    for target in [target_dir] if target_dir else _target_dirs():
        marker = _os.path.join(target, profile, ".ppf-backend")
        try:
            with open(marker) as handle:
                found = handle.read().strip()
            if found:
                return found
        except OSError:
            continue
    return None


class _RustCdylibFinder:
    """A ``sys.meta_path`` finder that maps ``_ppf_cts_py`` to the tree-local
    cdylib. ``sys.modules`` registration below already satisfies a plain
    ``import _ppf_cts_py``; this keeps the name resolvable if the module is
    ever evicted from ``sys.modules`` (e.g. an addon reload)."""

    def __init__(self, path):
        self._path = path

    def find_spec(self, name, path=None, target=None):
        if name != "_ppf_cts_py":
            return None
        loader = _machinery.ExtensionFileLoader(name, self._path)
        return _ilu.spec_from_file_location(name, self._path, loader=loader)


_LOADED_CDYLIB = None


def artifact_dir():
    """The directory this process loaded its extension module from.

    A directory names exactly one backend, which is what
    ``crates/ppf-cts-solver/build.rs`` enforces: every backend links the same
    executable name, so it refuses to put two backends in one directory.

    IT IS NOT NECESSARILY WHERE A RUN'S SOLVER COMES FROM. A distribution
    carries a directory per backend, and the run takes its solver from the
    backend ``get_backend`` resolves to, which ``_run_directory`` hands the
    launcher script after checking that solver was built from the same sources
    as this module. Where nothing chooses otherwise, the two directories are the
    same one.

    RESOLVED RATHER THAN REMEMBERED WHEN IT HAS TO BE. ``_load_rust`` returns
    early when ``_ppf_cts_py`` is already in ``sys.modules``, and that path
    never recorded the file it came from, so a caller in a process that had the
    module registered by some other route got ``None`` here and wrote an EMPTY
    directory into the launcher script. The fallback repeats the same search
    ``_load_rust`` would have done, which is the same answer by construction.

    ``None`` only when nothing is built, which the callers below turn into a
    named failure rather than an empty path.
    """
    path = _LOADED_CDYLIB or _find_cdylib()
    return None if path is None else _os.path.dirname(path)


# The two compute devices a caller can ask for, and what each MEANS here.
#
# "gpu" is the host's own accelerated backend, which is CUDA or ROCm on Windows
# and Linux and Metal on macOS.  It is deliberately not spelled "cuda" or "metal":
# a caller choosing a device is answering "accelerated or not", and which
# accelerator a host has is the host's business rather than the caller's.
# `backend_of` still reports the specific name, because a diagnostic has to be
# specific where a choice does not.
GPU = "gpu"
CPU = "cpu"

# Which backend names count as the accelerated answer.  A directory records the
# name `build.rs` selected, so this is the mapping from that vocabulary to the
# caller's.  A ROCm distribution's target/release is marked "rocm", and without
# that name here its GPU build would answer as the CPU device.
_GPU_BACKENDS = ("cuda", "metal", "rocm")


def solver_dir(device=None, profile="release"):
    """The directory holding a solver for *device*, or ``None``.

    THE BACKEND IS A PROPERTY OF A BUILD, NOT A RUNTIME FLAG, so choosing a
    device is choosing a DIRECTORY.  ``crates/ppf-cts-solver/build.rs`` links
    one backend per build, refuses to put a second one in a directory that
    already holds another, and writes ``.ppf-backend`` beside the artifacts
    saying which it is.  This reads those markers and hands back the directory
    whose backend answers to *device*.

    ``device`` is ``GPU``, ``CPU``, or ``None`` for "whatever this process
    already loaded", which is what every existing caller wants and is why the
    default preserves the previous behavior exactly.

    IT SEARCHES THE SAME PLACES ``artifact_dir`` DOES, PLUS THE SANCTIONED
    SECOND DIRECTORY.  Two backends coexist by living in two target
    directories (``CARGO_TARGET_DIR=target/cpu``), which is what `build.rs`
    prescribes in its own refusal message, so a tree that has both built has
    them at ``target/<profile>`` and ``target/<name>/<profile>``.

    Returns ``None`` when nothing built answers to *device*.  A caller turns
    that into a named failure rather than falling back silently to the other
    device: a CPU run that quietly became a GPU run, or the reverse, is the
    split this module already exists to prevent.
    """
    if device is None:
        return artifact_dir()
    if device not in (GPU, CPU):
        raise ValueError(
            f"unknown compute device {device!r}: expected {GPU!r} or {CPU!r}"
        )
    for target in _candidate_target_dirs():
        found = backend_of(target, profile)
        if found is None:
            continue
        is_gpu = found in _GPU_BACKENDS
        if (device == GPU) == is_gpu:
            return _os.path.join(target, profile)
    return None


def _candidate_target_dirs():
    """Every target directory a build may have used, most specific first.

    ``_target_dirs`` names the directory this process loads from, and the
    tree's own ``target`` is added whether or not that is it: this answers
    which directory holds a device's build, which does not depend on which one
    this process happened to load.  A second backend lives in a NAMED sibling,
    and the names are the backend names themselves because that is what the
    build command spells:
    ``CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu``.
    """
    seen = []
    root = _os.path.join(_FRONTEND_TREE_ROOT, "target")
    for target in _target_dirs() + [root]:
        if target not in seen:
            seen.append(target)
    for name in ("cpu", "cuda", "metal", "rocm", "gpu"):
        cand = _os.path.join(root, name)
        if cand not in seen:
            seen.append(cand)
    return seen


def available_devices(profile="release"):
    """Which devices this tree has a solver built for, as a sorted tuple.

    The addon and any UI offering a choice needs to know which options are
    real before it draws them, and the honest answer is a fact about the disk
    rather than about the host: a machine with a GPU that has only built the
    CPU backend can offer only CPU.
    """
    found = []
    for device in (GPU, CPU):
        if solver_dir(device, profile) is not None:
            found.append(device)
    return tuple(found)


# THE BACKEND A RUN USES. `_backends_` holds the rule; what follows wires in
# where this tree keeps its builds and how a solver is asked.
#
# `set_backend`'s choice for this process, or None for the automatic rule.
_BACKEND_CHOICE = None
# `--probe` answers, keyed by the solver binary's real path, size and
# modification time, so a rebuilt solver is asked again and an unchanged one is
# not asked twice.
_PROBES = {}
# Long enough for a GPU runtime's first initialization on a cold machine.
_PROBE_TIMEOUT_S = 120


def _solver_path(directory):
    name = (
        "ppf-contact-solver.exe" if _sys.platform == "win32" else "ppf-contact-solver"
    )
    return _os.path.join(directory, name)


def _run_profile():
    """The profile a run takes its solver from: the one this module was loaded from."""
    loaded = artifact_dir()
    return _os.path.basename(loaded) if loaded else "release"


def _probe_directory(directory):
    """Ask the solver in *directory* ``--probe``, remembering its answer."""
    solver = _solver_path(directory)
    try:
        status = _os.stat(solver)
    except OSError as error:
        raise RuntimeError(f"there is no solver at {solver}: {error}") from error
    key = (_os.path.realpath(solver), status.st_size, status.st_mtime_ns)
    if key in _PROBES:
        return _PROBES[key]
    env = dict(_os.environ)
    if _sys.platform == "win32":
        # The solver imports its backend DLL, so the directories the session
        # launcher puts on PATH have to be on it for this exec too. The list is
        # ppf-cts-core's `scripts::windows_library_dirs`, followed by the CUDA
        # toolkit's `bin` as the launcher script follows it.
        dirs = list(_rust.windows_library_dirs(_FRONTEND_TREE_ROOT))
        cuda = env.get("CUDA_PATH")
        if cuda:
            dirs.append(_os.path.join(cuda, "bin"))
        env["PATH"] = _os.pathsep.join(dirs + [env.get("PATH", "")])
    try:
        result = _subprocess.run(
            [solver, "--probe"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env=env,
        )
    except _subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{solver} --probe did not answer within {_PROBE_TIMEOUT_S} s"
        ) from error
    except OSError as error:
        raise RuntimeError(f"{solver} --probe could not be started: {error}") from error
    answer = _backends_.parse_probe(
        result.stdout,
        result.stderr,
        result.returncode,
        solver,
        _backends_.read_marker(directory),
    )
    _PROBES[key] = answer
    return answer


def _explicit_choice(profile):
    """``set_backend``'s choice, else the build ``CARGO_TARGET_DIR`` names, else ``None``."""
    if _BACKEND_CHOICE is not None:
        return _BACKEND_CHOICE
    if _os.environ.get("CARGO_TARGET_DIR"):
        named = _target_dirs()[0]
        found = _backends_.read_marker(_os.path.join(named, profile))
        if found is None:
            raise RuntimeError(
                f"CARGO_TARGET_DIR names {named}, and its {profile} directory holds "
                "no solver build: build the solver into it with the same variable "
                "set, or unset it"
            )
        return found
    return None


# Whether the automatic rule's notice has been printed in this process. A
# notice is about the MACHINE rather than about a run, so it is worth saying
# once: `_resolve` is called by every run and by several of the functions
# below, and a line repeated per frame would be noise the reader learns to skip.
_NOTICE_SHOWN = False


def _resolve(profile):
    """``(backend, build directory)`` a run with *profile* uses.

    Prints the automatic rule's notice the first time there is one. That is the
    whole of "and it says so" in `_backends_`'s rule 3: without it, a machine
    with no usable GPU would quietly run about 30x slower than the reader
    expects, which is the silent substitution this project refuses.
    """
    global _NOTICE_SHOWN
    built = _backends_.builds(_candidate_target_dirs(), profile)
    answer = _backends_.resolve(
        built,
        _explicit_choice(profile),
        lambda backend: _probe_directory(built[backend]),
    )
    if answer.notice and not _NOTICE_SHOWN:
        _NOTICE_SHOWN = True
        print(answer.notice)
    return answer.backend, built[answer.backend]


def list_backends():
    """The backends built into this tree or distribution, as ``{name: directory}``.

    A fact about the disk and not about the machine: whether a backend has a
    usable device here is :func:`probe_backend`'s question, and which one a run
    uses is :func:`get_backend`'s.

    Example::

        import frontend
        frontend.list_backends()
        # {'cuda': '.../target/cuda/release', 'rocm': '.../target/rocm/release',
        #  'cpu': '.../target/cpu/release'}
    """
    return _backends_.builds(_candidate_target_dirs(), _run_profile())


def probe_backend(name):
    """Ask backend *name*'s own solver whether it can run on this machine.

    Returns a named tuple ``(backend, usable, detail, stamp)``, where ``detail``
    is the device's name when ``usable`` is true and the backend's own reason
    when it is not. An unusable answer is not an error here: it is the answer
    the automatic choice reads, and where every GPU backend built here gives
    one, it is what selects the CPU backend.

    This is the question the automatic choice behind :func:`get_backend` asks
    of each GPU backend built here, CUDA first, then ROCm, then Metal, until one
    reports a usable device.
    An answer is remembered for the solver binary that gave it (its real path,
    size and modification time), so asking twice runs the solver once and a
    rebuilt solver is asked again.
    """
    built = list_backends()
    if name not in built:
        listing = ", ".join(sorted(built)) or "nothing"
        raise RuntimeError(f"there is no {name} build here (built: {listing})")
    return _probe_directory(built[name])


def set_backend(name):
    """Choose the backend this process's runs use.

    *name* is ``"cuda"``, ``"rocm"``, ``"metal"`` or ``"cpu"``, or ``None`` to
    return to the automatic choice :func:`get_backend` describes. A backend with
    no build here is refused at once rather than at the next run, and an
    explicit choice NEVER falls back to another backend: naming a GPU backend
    on a machine that cannot run it is an error, where leaving the choice
    automatic would have selected the CPU backend and said so.

    ``None`` returns to the automatic choice except where ``CARGO_TARGET_DIR``
    is set, because the build that variable names is itself an explicit choice:
    unsetting the variable is what returns the process to the automatic rule
    there.

    Example, on a machine with both an NVIDIA and an AMD GPU, where the
    automatic choice is CUDA::

        import frontend
        frontend.set_backend("rocm")
    """
    global _BACKEND_CHOICE
    if name is not None:
        _backends_.resolve(list_backends(), name, None)
    _BACKEND_CHOICE = name


def get_backend():
    """The backend this process's next run uses.

    An explicit choice comes first and never falls back: :func:`set_backend`'s,
    else the build ``CARGO_TARGET_DIR`` names. Otherwise the choice is
    automatic, and it is the first of CUDA, ROCm and Metal whose own solver
    reports a usable device on this machine. Failing that it is the CPU
    backend, which needs no GPU and is substantially slower, and choosing it
    that way prints one line naming what each GPU backend reported, once per
    process.

    Two automatic answers are given without asking a solver, because asking
    could not change them: a tree with no GPU build answers the CPU backend,
    and the one GPU build in a tree with no CPU build beside it answers itself.
    Raises, naming why, when nothing can be chosen: nothing is built here, or
    no GPU backend built here reports a usable device and there is no CPU build
    to fall back to.
    """
    return _resolve(_run_profile())[0]


def _run_directory():
    """The solver directory a run launches from.

    Refused when that solver was built from different sources than this
    process's extension module. This module writes the session and the solver
    reads it, so the two must agree on its formats, and
    ``ppf_cts_formats::SOURCE_STAMP`` is equal exactly when they were built from
    the same sources. That is what makes taking the solver from another
    backend's directory safe.
    """
    name, directory = _resolve(_run_profile())
    answer = _probe_directory(directory)
    loaded = getattr(_rust, "__source_stamp__", None)
    if answer.stamp != loaded:
        raise RuntimeError(
            f"the {name} solver in {directory} was built from different sources "
            f"(stamp {answer.stamp}) than the extension module this process "
            f"loaded from {artifact_dir()} (stamp {loaded}). Rebuild both from "
            "this tree."
        )
    return directory


def _require_usable_backend():
    """The backend a run uses, refused by name when it has no usable device here.

    ONLY A RESOLUTION WHOSE BACKEND WAS NEVER PROBED REACHES THIS REFUSAL,
    because a probed one is returned exactly when it answered usable. Three
    resolutions are of that kind: an explicit choice, which is never moved off
    what it names; the one GPU build in a tree with no CPU build beside it,
    where asking could not change the answer; and the CPU backend, which the
    rule selects without asking either way.

    So the message names the recovery its own resolution leaves open: returning
    the choice to automatic where a choice made it and another build here can
    change the answer, since the automatic rule selects the CPU backend where
    one is built, and building that CPU backend where there is none. The CPU
    backend refusing here leaves neither recovery, and every GPU backend built
    beside it has already reported that it cannot run, so the message stops at
    the reason. What is refused is unchanged: a run never starts on a backend
    whose own solver reports no usable device.
    """
    profile = _run_profile()
    name, directory = _resolve(profile)
    answer = _probe_directory(directory)
    if answer.usable:
        return name
    built = _backends_.builds(_candidate_target_dirs(), profile)
    others = sorted(other for other in built if other != name)
    parts = [
        f"the {name} backend cannot run on this machine: {answer.detail.rstrip('.')}."
    ]
    if others:
        parts.append(
            f"Also built here: {', '.join(others)}; frontend.set_backend(name) "
            "chooses one."
        )
    if others and _explicit_choice(profile) is not None:
        # Named where it would change the answer, and only there: with nothing
        # else built, the automatic choice resolves to this same backend.
        parts.append(
            "It is chosen explicitly, and an explicit choice is never moved off "
            "what it names: "
            + (
                "frontend.set_backend(None)"
                if _BACKEND_CHOICE is not None
                else "unsetting CARGO_TARGET_DIR"
            )
            + " returns to the automatic choice."
        )
    if "cpu" in others:
        parts.append(
            "The automatic choice selects the CPU backend built here and prints why."
        )
    elif "cpu" not in built:
        parts.append(
            "There is no CPU build here to fall back to: build one with "
            "`CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu`, "
            "and the automatic choice selects it on a machine no GPU build can "
            "serve, printing why."
        )
    raise RuntimeError(" ".join(parts))


# Directories registered with ``os.add_dll_directory``, kept for the life of the
# process, since a directory leaves the search path when its handle is closed.
_DLL_DIRECTORY_HANDLES = []


def _register_windows_dll_directories():
    """Let a Windows cdylib's own DLL imports resolve from the tree's ``bin``.

    Python loads an extension module with the default DLL search directories,
    which leave PATH out, so a DLL the cdylib imports is found only beside it, in
    System32, or in a directory registered with ``os.add_dll_directory``. A ROCm
    build's cdylib imports ``amdhip64_7.dll`` through ppf-cts-core's GPU probe,
    and a distribution ships that runtime, with the 121 MB ``amd_comgr.dll`` it
    imports, once, in ``<tree root>\\bin`` beside the solver's other runtime DLLs,
    rather than a second time beside the cdylib. So that directory is registered
    when it exists, which is the directory the Linux distribution's RPATH names.
    A tree without it registers nothing, and an import that resolves nowhere
    still fails the load.
    """
    if _sys.platform != "win32" or _DLL_DIRECTORY_HANDLES:
        return
    runtime = _os.path.join(_FRONTEND_TREE_ROOT, "bin")
    if _os.path.isdir(runtime):
        _DLL_DIRECTORY_HANDLES.append(_os.add_dll_directory(runtime))


def _load_rust():
    """Load and register the ``_ppf_cts_py`` cdylib built into this tree."""
    if "_ppf_cts_py" in _sys.modules:
        return _sys.modules["_ppf_cts_py"]
    path = _find_cdylib()
    if path is None:
        name = _cdylib_filename()
        searched = " and ".join(
            _os.path.join(target, profile)
            for target in _load_dirs()
            for profile in ("release", "debug")
        )
        named = (
            " CARGO_TARGET_DIR is set, so that directory is the only one "
            "searched: build into it with the same variable set, or unset it."
            if _os.environ.get("CARGO_TARGET_DIR")
            else ""
        )
        raise ImportError(
            f"_ppf_cts_py extension not found (looked for {name} in "
            f"{searched}).{named} "
            "Build it with `cargo build --release` from the repo root, which "
            "selects the real backend for this host (CUDA where a toolkit is "
            "present, Metal on macOS) and hard-errors when neither exists. "
            "There is no fallback build and no installed wheel to fall back to."
        )
    if not any(isinstance(f, _RustCdylibFinder) for f in _sys.meta_path):
        _sys.meta_path.insert(0, _RustCdylibFinder(path))
    _register_windows_dll_directories()
    loader = _machinery.ExtensionFileLoader("_ppf_cts_py", path)
    spec = _ilu.spec_from_file_location("_ppf_cts_py", path, loader=loader)
    module = _ilu.module_from_spec(spec)
    loader.exec_module(module)
    _sys.modules["_ppf_cts_py"] = module
    global _LOADED_CDYLIB
    _LOADED_CDYLIB = path
    return module


# Load the tree-local cdylib (built by `cargo build --release`) and register
# it as the top-level `_ppf_cts_py` module. A missing cdylib raises a clear
# ImportError from _load_rust; there is no fallback to an installed wheel.
#
# No cross-tree guard is needed: _load_rust always loads the cdylib from THIS
# frontend's own tree (``<_FRONTEND_TREE_ROOT>/target/<profile>/``), so it can
# never pick up another tree's module the way a shared-venv wheel once could.
# A guard comparing the cdylib's compile-time ``__build_manifest_dir__`` to the
# tree root would also wrongly reject a distribution bundle, where bundle.bat
# copies a dll built in the source tree into ``dist/target/release/``.
_rust = _load_rust()  # noqa: N816

__all__ = [
    "App",
    "get_cache_dir",
    "fetch_asset",
    "AssetManager",
    "AssetFetcher",
    "AssetUploader",
    "SceneManager",
    "Scene",
    "SceneInfo",
    "ObjectAdder",
    "FixedScene",
    "Object",
    "InvisibleAdder",
    "Wall",
    "Sphere",
    "Extra",
    "MeshManager",
    "CreateManager",
    "Rod",
    "TetMesh",
    "TriMesh",
    "PlotManager",
    "Plot",
    "SessionManager",
    "Session",
    "FixedSession",
    "SessionInfo",
    "SessionExport",
    "SessionOutput",
    "SessionGet",
    "CppRustDocStringParser",
    "ParamManager",
    "Utils",
    "BlenderApp",
    "sdf",
    "list_backends",
    "probe_backend",
    "set_backend",
    "get_backend",
]

from . import _sdf_ as sdf
from ._app_ import App
from ._asset_ import AssetFetcher, AssetManager, AssetUploader
from ._decoder_ import BlenderApp
from ._extra_ import Extra
from ._mesh_ import CreateManager, MeshManager, Rod, TetMesh, TriMesh
from ._parse_ import CppRustDocStringParser
from ._plot_ import Plot, PlotManager
from ._scene_ import (
    FixedScene,
    InvisibleAdder,
    Object,
    ObjectAdder,
    Scene,
    SceneInfo,
    SceneManager,
    Sphere,
    Wall,
)

# Backward-compat: FixedScene / EnumColor / ValidationError moved from the
# former ``_scene_fixed_`` module into ``_scene_``. Pickles written before
# that merge (``fixed_session.pickle``, ``app_state.pickle``) store the old
# ``frontend._scene_fixed_`` class path, so alias the old module name to
# ``_scene_`` and let those pickles keep unpickling.
_sys.modules.setdefault(f"{__name__}._scene_fixed_", _sys.modules[f"{__name__}._scene_"])
from ._session_ import (
    FixedSession,
    ParamManager,
    Session,
    SessionExport,
    SessionGet,
    SessionInfo,
    SessionManager,
    SessionOutput,
)
from ._utils_ import Utils, fetch_asset, get_cache_dir
