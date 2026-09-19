#!/usr/bin/env python3
# File: .github/workflows/scripts/check-neutral-solver.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# `ppf-cts-solver` is NEUTRAL. Nothing in it may name a backend.
#
# THE RULE: every backend-specific thing lives in `ppf-cts-compute`, and no
# other crate contains backend-specific logic or code, kernels included. The end
# state is three commands over this crate, and this script is those commands,
# run on every build.
#
# IT IS PART RATCHET AND PART WALL, and which half a name falls under is decided
# by whether the neutral spelling already exists.
#
# A RATCHET is for a rule being converged on rather than held. While files are
# still moving, a wall fails every build until the last one lands, so nobody
# runs it and the rule is enforced by memory instead. A ratchet fails the moment
# the count GROWS, and has to be lowered by hand as files move, which makes each
# move visible in a diff.
#
# A WALL is for a rule that is already satisfied. Four of the five counts below
# are zero: no file in the crate names a compiler, no file's own NAME says which
# backend it serves, no directory names a target, and no symbol carries a
# backend name that has since been given a neutral one. Each zero fails the
# build on the first reintroduction.
#
# WHY THE CEILING MUST BE LOWERED IN THE SAME CHANGE THAT MOVES A FILE: a
# ceiling left stale re-permits every move already made. That is the same
# argument rule 9 of check-shared-wiring.py carries for hand-written launchers,
# and it was learned there by measurement.

import importlib.util
import os
import re
import subprocess
import sys

# THE MASKER IS BORROWED, NOT REWRITTEN, AND THAT IS A CORRECTNESS DECISION.
#
# The API ratchet below asks whether a file NAMES a backend API in CODE. Answering
# that needs comments and string literals blanked first, and a second masker
# written here would be a second thing to get wrong: the one in
# check-shared-wiring.py is the one that has been measured against four lexical
# traps that each once passed a forbidden construct with a zero exit (a digit
# separator read as a character literal, an attribute split across two lines, a
# line comment ending in a backslash, and a multi-line block comment whose mask
# did not carry newlines). So this script imports it rather than copying it, and
# the two gates share one answer to "what is code here".
#
# BOTH MASKERS ARE CONSERVATIVE IN THE SAME DIRECTION, which is what makes them
# safe to reuse for a rule whose whole job is to notice a backend call. Every
# uncertainty they carry UNDER-masks: `kernel_code` gives the apostrophe no
# meaning, ends a line comment at the newline whether or not a backslash
# continues it, and ends a string at a newline. Each of those can only report a
# comment as code, never hide code as a comment. So a false report is possible
# and fails LOUDLY with a file and a line, and a missed backend call is not.
#
# WHY IT IS IMPORTED AND NOT COPIED: a copy is a fork, and the argument against
# forking a kernel body is the argument against forking a lexer. The import must
# fail loudly if it cannot resolve, because the one outcome worse than a wrong
# count is an unmasked count reported as a masked one.
_WIRING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check-shared-wiring.py")


def _load_maskers():
    """`kernel_code` and `mask_rust` out of the sibling gate.

    The file name carries hyphens, so it is not importable by name; it is loaded
    by path. It runs no code at import (module-level regex compiles and
    constants, with its own `main` under an `if __name__` guard), so loading it
    costs nothing and has no side effect.
    """
    spec = importlib.util.spec_from_file_location("check_shared_wiring",
                                                  _WIRING)
    if spec is None or spec.loader is None:
        sys.exit(f"check-neutral-solver: cannot load {_WIRING}. The API ratchet "
                 f"below reads code with comments and string literals masked, and "
                 f"an unmasked count reported as a masked one is the silent pass "
                 f"this gate exists to prevent.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [n for n in ("kernel_code", "mask_rust") if not hasattr(mod, n)]
    if missing:
        sys.exit(f"check-neutral-solver: {_WIRING} no longer defines "
                 f"{', '.join(missing)}. It was renamed or removed; point this "
                 f"script at the new name rather than falling back to an "
                 f"unmasked scan.")
    return mod.kernel_code, mod.mask_rust


MASK_CXX, MASK_RUST = _load_maskers()

# The crate that must be neutral.
CRATE = "crates/ppf-cts-solver"

# Extensions that name a COMPILER, which is the same claim a backend-named
# directory makes. `.cu` and `.cuh` are nvcc's, `.mm` is clang's Objective-C++,
# `.metal` and `.msl` are the Metal shader compiler's. None belongs to a crate
# whose contents belong to no compiler in particular.
BACKEND_EXTENSIONS = (".cu", ".cuh", ".mm", ".metal", ".msl")

# A NAME THAT SAYS WHICH BACKEND IT SERVES, applied to a directory and to a
# file's stem alike, because the claim a name makes does not depend on whether
# the thing named holds other files.
#
# THE FILE HALF EXISTS BECAUSE NEITHER OTHER CHECK REACHES IT, and the gap was
# measured rather than reasoned about: a tracked
# `crates/ppf-cts-solver/src/cuda_device.rs` passed this gate green. The
# extension list below reads the EXTENSION, and `.rs` is not one of nvcc's; the
# directory scan reads the DIRECTORY, and a file sitting in `src/` has no
# backend-named directory above it. So the one place a new backend would
# plausibly land now that the C++ side is at zero, a Rust module in the neutral
# crate, was the one place nothing looked.
#
# IT IS MATCHED ON THE STEM, NOT THE WHOLE BASENAME, so the two file checks stay
# disjoint: `foo.metal` is a compiler's extension and is counted once, by
# `BACKEND_EXTENSIONS`, rather than twice. A stem that names a backend fails
# whatever its extension, which is what makes this a rule about the NAME.
BACKEND_NAME_RE = re.compile(r"(cuda|metal|cpu)", re.IGNORECASE)

# THE TWO NAME FAMILIES, AND THEY ARE CHECKED DIFFERENTLY ON PURPOSE.
#
# A WALL name is a SYMBOL the project has already given a neutral spelling. The
# neutral name exists, every backend answers to it, and nothing in the crate is
# entitled to the old one, so the honest measure is ZERO OCCURRENCES rather than
# zero files. Counting files would let one back in: a file already counted for
# some other reason absorbs a new occurrence without moving any number, which is
# a violation that lands green. So the wall is counted per occurrence, over
# every tracked text file in the crate rather than over four extensions, and it
# reports the file and line.
#
#   cpu_*     an entry point named for the backend that compiles it. The
#                 spelling is `ppf_<stem>_entry` for a dispatchable entry point
#                 and `ppf_<stem>_abi` for a C-ABI helper, on every target, which
#                 is what lets one Rust call site be answered by three backends.
#   CpuDiag    the ARGUMENT RECORD a hand-written entry point takes, named for
#   Cpu*Args   
#                 the backend that compiles it. `kernelgen.py` spells a generated
#                 one `Ppf<Camel>Args` beside its Rust twin `<Camel>Args`
#                 (`record_cpp`, `record_rust`), so a hand-written record lands on
#                 the name its generated replacement will carry and the conversion
#                 becomes a pure deletion. The one that was not an `*Args`,
#                 `CpuDiag`, is `ChunkDiag`: one record per chunk, which is
#                 what the transport merges over.
#                 SPELLED OUT rather than left as a bare `Cpu`, which is the
#                 same shape the `Cuda*` alternatives below already have. A
#                 bare alternative also matches `Backend::Cpu`, the variant
#                 `build.rs` selects a backend with, and a build script that
#                 chooses among three backends has to name all three: the
#                 sibling `Backend::Cuda` and `Backend::Metal` are not
#                 matched either, so a bare spelling here would refuse one
#                 of the three and admit the other two.
#   CudaVec       the OWNING ALLOCATION, `ArenaVec<T>`. A view is `Vec<T>`, a
#   CudaVecVec    handle is `ArenaHandle`, a jagged view is `VecVec<T>`, and the
#   CudaPtr       jagged owner is `ArenaVecVec<T>`; the owning array and the
#   cuda_arena    (arena, offset) pointer had no neutral name until they were
#                 given these. The allocator namespace is `compute::arena`.
#   cuda_diagnostics
#                 the DIAGNOSTIC TRANSPORT's namespace, `diagnostics`.
#                 Reporting an assertion or a trace out of a device is platform
#                 machinery one of the five verbs requires, and every target
#                 answers the same macro vocabulary over it (DIAG_ARG,
#                 DIAG_ASSERT, DIAG_BOUNDS_CHECK, DIAG_TRACE), so the namespace is
#                 named for the thing exactly as `compute::arena` is.
#   cudaMemcpy    HOST-DEVICE TRANSFER, `mem::copy_from_device_to_host` and
#                 `mem::copy_from_host_to_device`, and ALLOCATION and FREE,
#   cudaMalloc    `ArenaVec<T>::alloc` and `ArenaVec<T>::free` over
#   cudaFree      `compute::arena`. Three of the five verbs, so the solver
#                 states WHAT to move or reserve and the compute crate states
#                 how. The helper takes typed pointers and a COUNT, where the
#                 CUDA call takes void pointers and a byte length, so a size
#                 that does not match its pointers stops being expressible: the
#                 two template deductions have to agree on the element type.
#                 The other two verbs are not here, and neither is at zero: a
#                 launch is still spelled `<<<>>>` in three headers, and the
#                 diagnostic channel that a launch reports through is reached
#                 by macro.
#   cudaStream    THE QUEUE a launcher orders its work on, `DeviceQueue`. Each
#                 backend prologue aliases it to its own handle (nvcc's
#                 `cudaStream_t` in seam_cuda.cuh, an opaque pointer in
#                 seam_host.h), so a launcher declaration in this crate names a
#                 queue without naming who runs it, exactly as `ArenaVec<T>`
#                 lets one name an owning array without naming the allocator.
#                 The alias is transparent, so a `.cu` definition written
#                 against nvcc's spelling and a declaration written against this
#                 one declare the same function.
#   CudaDataSet   THE ARENA-OWNED SCENE MODEL and the nine records it is built
#   CudaVertexSet from, `ArenaDataSet` and its `Arena*` siblings. `DataSet` and
#   CudaMeshInfo  its siblings in data.hpp are the VIEW every kernel reads
#   CudaPropSet   through; these are the same records held as `ArenaVec<T>`
#   CudaParamArrays instead of `Vec<T>`, which is what makes them the owning
#   CudaCollisionMesh form. The pairing is the one `Vec<T>` and `ArenaVec<T>`
#   CudaConstraint already carry, so the owning form is named the way the
#   CudaVertexNeighbor container is: for what it IS, an allocation out of an
#   CudaHingeNeighbor arena, rather than for the compiler that reserves it.
#   CudaEdgeNeighbor
WALL_RE = re.compile(
    r"\b(cpu_|CpuDiag|Cpu\w*Args|CudaVecVec|CudaVec|CudaPtr|cuda_arena|cuda_diagnostics"
    r"|cudaMemcpy|cudaMalloc|cudaFree|cudaStream"
    r"|CudaDataSet|CudaVertexSet|CudaMeshInfo|CudaPropSet|CudaParamArrays"
    r"|CudaCollisionMesh|CudaConstraint"
    r"|CudaVertexNeighbor|CudaHingeNeighbor|CudaEdgeNeighbor)")

# A RATCHET name is a backend API the crate still genuinely calls, or a compiler
# keyword it still genuinely spells. Those are a RESIDENCY debt: the declaration
# headers that carry them are nvcc's, and they leave when the headers leave, so
# the count falls in steps.
# `cudaMalloc`, `cudaMemcpy`, `cudaFree` and `cudaStream` are NOT here: each
# moved to the wall above when the last occurrence became a `mem::` call, a
# `compute::arena` call, or a `DeviceQueue`. A name may be in one list or the
# other and never both, or the same occurrence would be counted twice and the
# two ceilings would move together for one edit.
#
# IT IS MATCHED ON CODE, WITH COMMENTS AND STRING LITERALS MASKED, AND THAT
# CHOICE WAS MEASURED RATHER THAN ASSUMED. Reading raw source instead, on the
# argument that a neutral file has no reason to name a backend even in prose,
# flags 30 files, of which 25 are flagged for a COMMENT, and every one of those
# comments is a
# kernel body explaining what its own rendering becomes ("`__global__` is what
# nvcc compiles this into"). Those comments are correct and belong where they
# are: a design whose whole claim is that one body is rendered per target needs
# to be able to write down what each rendering looks like, and a rule that
# forbids naming the output is a rule against documenting the mechanism.
#
# THE REASON TO CHANGE IT IS NOT TIDINESS, IT IS THAT A COUNT WHICH IS 80
# PERCENT PROSE IS A PLACE A REGRESSION HIDES. The ratchet fails on GROWTH, and
# growth is measured per file. A file already flagged for a comment absorbs a
# genuinely new backend call without moving any number, so the very files most
# likely to acquire one, the kernel bodies that describe backends in prose, were
# the files the ratchet had already stopped watching. Masking the prose out took
# the ceiling from 30 to 5, and those 5 are the real debt: three headers that
# hand-write a launch, one that dumps the Newton system, and one PDRD projector.
#
# TWO THINGS FOLLOW FROM MASKING, AND BOTH ARE STATED RATHER THAN DISCOVERED.
# A string literal is now prose too, so a message naming `cudaDeviceProp` in a
# panic does not count, which is right (a string is data, and a backend API
# named inside one is not called). What that cannot see is a file EMITTING
# backend source as a string, and nothing here would catch it; the wall above is
# read UNMASKED for exactly this reason, so the renamed-symbol family is still
# caught inside a string, and a build script's recipe is check-compute-neutral
# and review. And the prose count does not vanish from the report: it is printed
# beside the code count every run, so a jump in it is visible even though it is
# not a ceiling.
BACKEND_API_RE = re.compile(
    r"\b(cudaGraph|cudaDevice"
    r"|__global__|__syncthreads|MTLDevice|MTLBuffer|MTLCommandQueue"
    r"|newBufferWithLength|id<MTL"
    # DISPATCH_START and DISPATCH_END are defined in exactly one place,
    # ppf-cts-compute/cuda/utility/dispatcher.cu, so a file spelling either is
    # CUDA-only however it looks. Without them a header carrying twelve
    # hand-written launches and no `__global__` counts as neutral.
    r"|DISPATCH_START|DISPATCH_END)")

# THE PATHS THAT HAVE A HOME, AND IT IS NOT HERE.
#
# A THIRD KIND OF WALL, and it exists because neither of the other two catches
# what it catches. A mechanism header is a `.hpp`, so `CEILING_FILES` (which
# reads extensions) does not see it; it declares `__device__` rather than
# calling a `cuda*` API, so `BACKEND_API_RE` does not see it either. Each of the
# four below was in this crate, was one of the five verbs, and moved to
# `crates/ppf-cts-compute/cuda` on its own merits. Reintroducing one is a
# residency regression that would otherwise land green, so it is named.
#
# THIS IS A PATH CHECK AND NOT A CONTENT CHECK, deliberately. What it defends is
# the DECISION about where a file lives, and a decision is identified by the
# thing decided about. A rewritten copy under a different name is not caught
# here, and the honest answer to that is review plus check-compute-neutral.py's
# census, which fails on any file in that crate nobody has classified.
MOVED_TO_COMPUTE = {
    "src/kernels/arena/arena.hpp":
        "ALLOC/FREE/COPY: now cuda/arena/arena.hpp, beside the arena.cu that "
        "answers it",
    "src/kernels/diagnostics/diagnostics.hpp":
        "PLATFORM: now cuda/diagnostics/diagnostics.hpp. The RECORD layout "
        "stays here, in diagnostic_record.hpp, because both sides read it",
    "src/kernels/main/mem.hpp":
        "COPY: now cuda/mem.hpp. Host-device transfer is one of the five verbs",
    "src/kernels/utility/dispatcher.hpp":
        "LAUNCH: now cuda/utility/dispatcher.hpp, the name its callers reach "
        "the dispatch primitive by",
}

# THE CEILINGS. Lower each in the change that moves a file, never separately.
CEILING_FILES = 0           # no file in the crate names a compiler
CEILING_NAMES = 0           # no file's stem names a target
CEILING_DIRS = 0            # no directory in the crate names a target
# ZERO, AND ZERO IS A WALL. No file under this crate names a backend API in
# code at all, so this ceiling refuses the first one rather than counting down
# from a debt.
#
# WHERE THE HOST SIDE OF A PASS LIVES, because a reader meeting a zero here will
# ask what carries the work a hand-written launcher would: the neutral Rust
# driver in `src/driver/` reaches every pass through a generated entry point and
# the `Device` trait, so a `<<<>>>` in this crate would have no caller to serve
# in the first place.
CEILING_API_FILES = 0       # the .hpp/.cpp/.rs/.h files that still name one
# AND A SECOND RATCHET OVER THE SAME HITS, BECAUSE A FILE IS TOO COARSE A UNIT
# TO NOTICE A NEW LAUNCH. Masking answers the case where a new call lands in a
# file flagged only for a comment. It does not answer the case where a new call
# lands in one of the five above, which are flagged already: the file count
# stays at 5 and the growth is invisible. This is the argument the wall carries
# verbatim, so it gets the wall's remedy, an occurrence count. The two ceilings
# measure the same hits at two granularities and must fall together; a file
# leaving takes its occurrences with it.
CEILING_API_HITS = 0        # occurrences of a backend API name, in code
CEILING_WALL = 0            # occurrences of a name that already has a neutral one

# WHAT IS LEFT IN THE TWO BACKEND DIRECTORIES IS MECHANISM AND NOTHING ELSE:
# `crates/ppf-cts-compute/cuda` holds the `.cuh`
# prologue, the recipes, the architecture manifest and the three `.cu` that are
# mechanism (the arena, the diagnostic transport, the dispatch primitive), and
# `crates/ppf-cts-compute/metal` holds the device, arena, shader assembler,
# pipeline cache, diagnostic transport, bring-up sequence and the recipe. The
# simulation sits in this crate instead, one `*.kernel.cpp` per kernel, rendered
# per target. What makes that possible on both sides is the same thing: a recipe
# there takes the neutral kernel tree as a caller-supplied `KERNEL_ROOT` rather
# than hardcoding a path back into this crate, so the dependency edge points one
# way.
#
# THIS CHECK AND check-compute-neutral.py ARE A PAIR AND NEITHER SUBSTITUTES FOR
# THE OTHER. This one asks whether the SOLVER names a backend; that one asks
# whether the COMPUTE crate carries simulation. Satisfying this one by moving
# files is exactly how that one came to be broken.
#
# The first three ceilings are therefore ZERO, and a zero ceiling is a wall: any
# `.cu`, `.cuh`, `.mm`, `.metal` or `.msl` added under this crate, any file whose
# stem names a target whatever its extension, and any directory named for a
# target, fails the build. The end state of section 2.1d is reached for all
# three, and it stays reached.
#
# NO FILE NAMES A BACKEND API IN CODE, for the reason the ceiling above gives:
# nothing in this crate hand-writes a launch.
#
# THE OTHER 25 ARE PROSE AND ARE NOT COUNTED, and what they are is worth
# reading before assuming they are untidy. Twenty are kernel bodies whose
# comments explain what the transcompiler makes of them, `build.rs` names
# `cudaDeviceProp` in the panic that fires on the wrong toolkit,
# `csrmat/asm_profile.hpp` describes the dispatch blocks whose cost it profiles,
# `entrypoints/` names the keyword in two file headers, `seam/backend_abi.h`
# names it once explaining what the ABI replaces, and `src/mesh.rs` mentions it
# in passing. None of them calls anything. The report below still prints how
# many there are, because the number falling to zero would be a different kind
# of news than the ceiling falling to zero.
#
# NOTHING IS WAITING ON THE ENTRY GENERATOR ANY MORE, and it is worth being
# precise about why, because "the generator learned every launch shape" would
# be the wrong lesson. It did not. The launches that were waiting on it belonged
# to an orchestrator this design does not have, so the question they posed was
# retired rather than answered. A future body that needs threadgroup scratch,
# the block index, more than one scatter target or a gather through an index
# read from another buffer will pose it again, and extending the entry generator
# is where that goes.
#
# THE SYMBOL HALF OF SECTION 2.1D IS SETTLED, and the wall is what holds it. A
# symbol is named for what it IS: an entry point is `ppf_<stem>_entry` and a
# C-ABI helper `ppf_<stem>_abi` on every target, an owning allocation is
# `ArenaVec<T>` or `ArenaVecVec<T>`, an (arena, offset) pointer is `ArenaPtr<T>`,
# and the allocator namespace is `compute::arena`. Each has one declaration in
# the neutral tree that a CUDA build, a Metal build and a host build can each
# satisfy, which is the property a backend-specific spelling cannot have: a
# header naming `CudaVec` can only ever be compiled by one of the three, so the
# name decides the architecture rather than describing it.
#
# THE RESIDENCY DEBT AND THE NAMING DEBT WERE INDEPENDENT, and settling one
# never settled the other. The naming half is closed: no header in this crate
# is nvcc-only, because the ones that were are deleted. The residency half is
# a different question and lives in `check-shared-wiring.py`, which counts the
# buffers a record still reaches by host pointer; a header can be perfectly
# neutral and still name a buffer the host addresses directly.
#
# THERE IS NO THIRD CRATE TO MOVE ANYTHING INTO. A file that is
# backend-specific AND carries simulation fits neither crate, and the answer is
# not a third home for it: such a file is MIXED, and it is separated rather than
# rehoused.
#
# THE FOUR THAT WERE MECHANISM HAVE MOVED, and each was one of the five verbs
# rather than a naming problem: the allocator (`arena/arena.hpp`), the
# diagnostic transport (`diagnostics/diagnostics.hpp`), the host-device transfer
# helpers (`main/mem.hpp`) and the dispatch primitive's interface
# (`utility/dispatcher.hpp`) are all in `crates/ppf-cts-compute/cuda`. What a
# neutral header names of them it names WITHOUT a path, which is the same
# arrangement `seam/seam.hpp` uses for `seam_cuda.cuh`: the CUDA build puts its
# own directory on the include path, so the neutral tree carries no route into
# that crate.


def tracked_paths():
    """Every tracked path under the crate.

    Read from git rather than the filesystem so a build directory, a stale
    artifact or an editor's scratch file cannot change the verdict. An untracked
    `.cu` is somebody's experiment; a tracked one is the architecture.
    """
    out = subprocess.run(
        ["git", "ls-files", CRATE],
        capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def unstaged_candidates():
    """New files that WOULD count, and do not because git does not list them.

    THE COUNT ABOVE IS DELIBERATELY OVER TRACKED FILES ONLY, and this does not
    change that: a build directory's `.cu` must not decide the verdict. What it
    answers is the confusing half of that choice. Splitting a header that names
    a backend API leaves the halves untracked, so the count FALLS, and the fall
    is an artifact of `git ls-files` rather than of the tree; measured, a split
    that moved nothing reported 50 hits as 48. The ratchet does catch it, since
    the ceiling is matched for equality and staging turns the false fall into a
    failure as growth, but only after the author has believed the fall once.
    Naming the files here turns that into one line.
    """
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", CRATE],
        capture_output=True, text=True, check=True).stdout
    named = []
    for path in (p for p in out.splitlines() if p):
        if path.endswith(BACKEND_EXTENSIONS):
            named.append(path)
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        if BACKEND_API_RE.search(raw):
            named.append(path)
    return sorted(named)


def main():
    paths = tracked_paths()
    unstaged = unstaged_candidates()
    if not paths:
        sys.exit(f"check-neutral-solver: git lists no tracked file under {CRATE}. "
                 f"Either the crate moved or this script is looking in the wrong "
                 f"place, and reporting a clean count over nothing is the one "
                 f"outcome worse than reporting a violation.")

    files = sorted(p for p in paths if p.endswith(BACKEND_EXTENSIONS))

    # THE STEM, so this and the extension scan above never count one file twice.
    # `os.path.splitext` splits at the LAST dot, so `foo.kernel.cpp` has the stem
    # `foo.kernel` and a dotfile keeps its whole name; neither reading admits a
    # backend name that is not written there.
    named = sorted(p for p in paths
                   if BACKEND_NAME_RE.search(
                       os.path.splitext(os.path.basename(p))[0]))

    dirs = set()
    for p in paths:
        parts = os.path.dirname(p).split("/")
        for i, part in enumerate(parts):
            if BACKEND_NAME_RE.search(part):
                dirs.add("/".join(parts[:i + 1]))
    dirs = sorted(dirs)

    # THE API SCAN READS CODE. Comments and string literals are blanked first,
    # by the masker the sibling gate uses, so a kernel body explaining what its
    # own rendering becomes is not counted beside a header that launches. Both
    # maskers preserve offsets, so a hit's position in the masked text is its
    # position in the original and the line number reported is the real one.
    #
    # A FILE WITH NO CODE HIT BUT A RAW HIT IS COUNTED SEPARATELY AND REPORTED.
    # It is not a ceiling, because a comment is allowed to name a backend and
    # the number moves whenever somebody rewrites one. It is printed because a
    # number that is not watched should at least be visible: a large jump in it
    # is a reason to read a diff even though it is not a reason to fail a build.
    api_hits = []       # (path, line, name) for every hit in CODE
    api_files = []      # the files those hits are in
    prose_only = []     # named in a comment or a string, and nowhere else
    for p in paths:
        if not p.endswith((".rs", ".hpp", ".cpp", ".h")):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except OSError:
            continue
        if not BACKEND_API_RE.search(raw):
            continue
        code = MASK_RUST(raw) if p.endswith(".rs") else MASK_CXX(raw)
        hits = [(p, code.count("\n", 0, m.start()) + 1, m.group(1))
                for m in BACKEND_API_RE.finditer(code)]
        if hits:
            api_files.append(p)
            api_hits.extend(hits)
        else:
            prose_only.append(p)
    api_files.sort()
    prose_only.sort()
    api_hits.sort()

    # A path that moved out and came back. Read from the same tracked list, so
    # an untracked copy on someone's disk is not a verdict.
    tracked = set(paths)
    returned = sorted(rel for rel in MOVED_TO_COMPUTE
                      if f"{CRATE}/{rel}" in tracked)

    # The wall is read over EVERY tracked file, whatever its extension, and per
    # occurrence. A recipe, a manifest or a comment naming one of these is the
    # same violation as a declaration naming it, and each is reported with its
    # file and line so the message names the offender rather than a count.
    wall_hits = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    # Every match on the line, not the first: the count has to
                    # be occurrences for the ceiling of zero to mean what it
                    # says, and two on one line is two.
                    for m in WALL_RE.finditer(line):
                        wall_hits.append((p, n, m.group(1), line.rstrip()))
        except (OSError, UnicodeDecodeError):
            continue

    print("neutral-solver census (ratchet)")
    print(f"  files naming a compiler   : {len(files)} (ceiling {CEILING_FILES})")
    print(f"  files naming a target      : {len(named)} (ceiling {CEILING_NAMES})")
    print(f"  directories naming a target: {len(dirs)} (ceiling {CEILING_DIRS})")
    print(f"  files naming a backend API : {len(api_files)}"
          + (f" (ceiling {CEILING_API_FILES})" if CEILING_API_FILES is not None else " (unratcheted)"))
    print(f"  those names, in code       : {len(api_hits)}"
          + (f" (ceiling {CEILING_API_HITS})" if CEILING_API_HITS is not None else " (unratcheted)"))
    print(f"  files naming one in prose  : {len(prose_only)} (not a ceiling; a "
          f"comment may name a backend)")
    print(f"  renamed-symbol occurrences : {len(wall_hits)} (ceiling {CEILING_WALL})")
    print(f"  moved-out paths back again : {len(returned)} "
          f"(ceiling 0, of {len(MOVED_TO_COMPUTE)} watched)")
    for d in dirs:
        print(f"      dir  {d}")
    for p in named:
        print(f"      name {p}")
    for p in files[:12]:
        print(f"      file {p}")
    if len(files) > 12:
        print(f"      ... and {len(files) - 12} more")
    # Every API file with its hits, so a growth failure is read off the report
    # rather than reconstructed. A count alone cannot say which file moved.
    for path in api_files:
        by_name = {}
        for _, line, name in (h for h in api_hits if h[0] == path):
            by_name.setdefault(name, []).append(line)
        detail = ", ".join(
            f"{name} x{len(lines)} (first at line {min(lines)})"
            for name, lines in sorted(by_name.items()))
        print(f"      api  {path}: {detail}")
    for path, n, name, text in wall_hits[:12]:
        print(f"      wall {path}:{n}: {name}   {text.strip()[:80]}")
    if len(wall_hits) > 12:
        print(f"      ... and {len(wall_hits) - 12} more")

    failed = []
    for rel in returned:
        failed.append(f"{rel} is back under {CRATE}; it lives in "
                      f"crates/ppf-cts-compute ({MOVED_TO_COMPUTE[rel]})")
    if len(wall_hits) > CEILING_WALL:
        names = sorted({h[2] for h in wall_hits})
        plural = "occurrence" if len(wall_hits) == 1 else "occurrences"
        failed.append(f"{len(wall_hits)} {plural} of a renamed symbol "
                      f"({', '.join(names)}); each already has a neutral spelling "
                      f"every backend answers to")
    if len(files) > CEILING_FILES:
        failed.append(f"{len(files)} files name a compiler, over the {CEILING_FILES} recorded")
    if len(named) > CEILING_NAMES:
        # NAMED, not counted. A stem is the whole evidence here, so a count
        # alone would leave a reader grepping for which file arrived.
        failed.append(f"{len(named)} files name a target in their own name, over "
                      f"the {CEILING_NAMES} recorded: {', '.join(named)}. A file "
                      f"whose name says which backend it serves declares this "
                      f"crate has backends in it, and it has none; backend code "
                      f"is written in ppf-cts-compute or emitted by the "
                      f"transcompiler from a neutral source")
    if len(dirs) > CEILING_DIRS:
        failed.append(f"{len(dirs)} directories name a target, over the {CEILING_DIRS} recorded")
    if CEILING_API_FILES is not None and len(api_files) > CEILING_API_FILES:
        failed.append(f"{len(api_files)} files name a backend API in code, over "
                      f"the {CEILING_API_FILES} recorded")
    if CEILING_API_HITS is not None and len(api_hits) > CEILING_API_HITS:
        # Name the lines, not just the count. The occurrence ceiling exists for
        # a call landing in a file that is already flagged, and in that case the
        # file list above is unchanged, so the line is the only thing that says
        # what happened.
        excess = len(api_hits) - CEILING_API_HITS
        failed.append(f"{len(api_hits)} occurrences of a backend API name in "
                      f"code, over the {CEILING_API_HITS} recorded ({excess} "
                      f"more than the ceiling). The hits are listed above; a "
                      f"file already at the file ceiling still fails here, "
                      f"which is the point.")

    # A ceiling that is not lowered as files move re-permits every move already
    # made, so a count BELOW its ceiling is also a failure until the ceiling
    # follows it down.
    stale = []
    if len(files) < CEILING_FILES:
        stale.append(f"CEILING_FILES to {len(files)}")
    if len(named) < CEILING_NAMES:
        stale.append(f"CEILING_NAMES to {len(named)}")
    if len(dirs) < CEILING_DIRS:
        stale.append(f"CEILING_DIRS to {len(dirs)}")
    if CEILING_API_FILES is not None and len(api_files) < CEILING_API_FILES:
        stale.append(f"CEILING_API_FILES to {len(api_files)}")
    if CEILING_API_HITS is not None and len(api_hits) < CEILING_API_HITS:
        stale.append(f"CEILING_API_HITS to {len(api_hits)}")

    if failed:
        print("\nFAILED")
        for m in failed:
            print(f"  {m}")
        print(f"\n  {CRATE} is neutral: every backend-specific thing belongs in "
              f"ppf-cts-compute. See BACKEND_ARCHITECTURE.md section 2.1-0.")
        sys.exit(1)
    if stale:
        print("\nFAILED")
        print(f"  the count fell but the ceiling did not. Lower {', '.join(stale)} "
              f"in this change: a ceiling left stale re-permits every move already "
              f"made.")
        if unstaged:
            print("  BUT READ THE LIST BELOW FIRST, because the fall may not be "
                  "real: this scan reads `git ls-files`, and these files are not "
                  "in it yet, so whatever they name is uncounted. `git add` them "
                  "and re-run before lowering anything.")
            for path in unstaged:
                print(f"      unstaged  {path}")
        sys.exit(1)

    if unstaged:
        print(f"\n  NOT COUNTED, because `git ls-files` does not list them yet: "
              f"{len(unstaged)} untracked file(s) under this crate name a backend "
              f"API or carry a backend extension. Staging them can only RAISE the "
              f"counts above.")
        for path in unstaged:
            print(f"      unstaged  {path}")

    if len(files) == 0 and len(named) == 0 and len(dirs) == 0 and len(api_files) == 0:
        print("\nOK: the solver crate names no backend")
    elif len(files) == 0 and len(named) == 0 and len(dirs) == 0:
        # The file and directory ceilings being zero is NOT the whole rule, and
        # saying so here would be the silent pass this gate exists to prevent:
        # a crate with no backend-named FILE can still hold a backend-named
        # SYMBOL in a neutrally named one.
        print(f"\nOK: no growth. No file or directory names a backend, and "
              f"{len(api_files)} file(s) still name a backend API; that count is "
              f"the debt and it is what must fall.")
    else:
        print("\nOK: no growth (the crate is not neutral yet; the ceilings are the debt)")


if __name__ == "__main__":
    main()
