# macOS Build

Native macOS build glue for the Metal backend, mirroring `build-win-native/`.
Same entry points, same rules: `warmup.sh` provisions, `build.sh` compiles,
`bundle.sh` packages, `config.sh` holds local settings, and
`scripts/downloads.txt` is the single source of truth for every URL this
directory fetches.

This directory is build and packaging glue only. The solver's C++, MSL and
shared math live under `crates/ppf-cts-solver/src/`, and nothing physics-related
belongs here.

## Status

**The shader pre-compile (build.sh step 4) is currently a graceful SKIP.** The dump tool
it drives, `ppf-metal-shader-dump`, calls `metal_backend::backend_bringup`, which lives in
`bringup.mm` and is deliberately not in the shipped `libppfbe_metal.dylib` (the dylib is the
runtime ABI entry points only), so the tool does not link against the runtime backend.
Restoring the pre-compile means porting the tool to link the bring-up and shader-assembly
objects instead. Until then build.sh completes and the backend compiles its shader at run
time (about 6.5 s, OS-cached), so nothing is broken, the first solver run is just slower.
"The Metal toolchain" section below describes the tool as it will be once
ported.


`build.sh` and `bundle.sh` have been run on an Apple M1 and complete. That
run was of an earlier version of these scripts, one that carried no bundled
interpreter, and what it produced was a flat `dist/` directory. That shape is
close to the distribution directory `bundle.sh` produces, which is what makes
its measurement worth recording; the steps that carry the interpreter, the
thinning and the launcher have not been run, which "Not run" below sets out.
That run turned up three defects, all fixed: both scripts resolved the backend
with `otool -L` and then tested the result with `-f`, which cannot succeed
because
`otool -L` prints the install name `@rpath/libppfbe_metal.dylib` and
`@rpath` is a loader placeholder rather than a directory (the resolution now
lives in `scripts/backend-path.sh` and consults `LC_RPATH`); and `bundle.sh`
never rewrote the cdylib's own `LC_ID_DYLIB`, which cargo writes as an absolute
build-tree path. The `dist/` that run produced was verified by relocation:
copied to an unrelated directory, no Mach-O in it references the build tree,
and the relocated solver runs.

**A relocation test on the build machine cannot settle whether a distribution
runs elsewhere**, and the defect recorded below is why that is worth stating
rather than leaving to be inferred: that defect survived exactly this test.
Moving the distribution directory within one machine leaves the build tree
where it was, so a path compiled into a binary and resolved at run time still
resolves. What that test does establish is that no LOADER reference escaped,
which is what it was run for.

**The distribution carries its own Python and is a plain terminal program.**
`warmup.sh` provisions a relocatable CPython into `build-mac-native/python`,
and `bundle.sh` packages everything into `dist/<name>/`, a
directory whose `ppf-contact-solver` launcher starts JupyterLab in the
foreground and streams its log to the terminal. A user needs no Python of their
own and no setup step before the first run. See "The bundled interpreter" and
"The distribution" below.

**One defect stopped a finished payload from starting on a Mac that is not
the one that built it, and it is worth recording because gate C is the only
thing that would ever have caught it and because the shape of it invites a
future reader to put it back.** `crates/ppf-cts-solver/build.rs` emitted
`PPF_BACKEND_LIBRARY_DIR`, an absolute path into the build tree;
`crates/ppf-cts-solver/src/driver/launch.rs` read it with `option_env!` at
compile time and handed it to the backend as `OpenConfig.library_dir`; and
`be_open` in `crates/ppf-cts-compute/metal/backend/backend.mm` refused to open
without one, then loaded `ppf_entries.metallib` out of the directory it was
given. So the shipped solver carried a build-tree path that it resolved itself
at run time rather than through the loader. No load command was wrong and no
loader check would have failed. Gate C is what caught it, because it reads file
contents rather than load commands, and `bundle.sh` refused rather than
packaging past it, which was the correct outcome and not a gate to widen:
relaxing it would have shipped a payload whose backend cannot open, and the
first person to find that out would be a user who has to work out from a
traceback what a `library_dir` is.

The three files now say the opposite. `build.rs` emits nothing, `launch.rs`
names no `library_dir`, and `be_open` falls back to the directory this dylib
was itself loaded from, resolved through `dladdr` by the same
`context_prebuilt_library_dir` the shader cache already used.
`ppf_entries.metallib` sits beside the dylib in the build tree and in the
distribution alike, so the question has one answer in both layouts and nothing
has to be told. `be_open` still refuses, by name, when the caller names no
directory and it cannot resolve its own, which is the case that should never
arise and would otherwise be a silent load of nothing. **The code is fixed and
the build has still not been run**, for the reason in "Not run" below; what is
settled is the cause, not that gate C now passes.

Two things about the packaging work were measured, one is a property taken from
upstream rather than measured here, and three have not been run at all.

Measured:

- The interpreter pointer was verified reachable from a networked host on
  2026-09-07: `curl -fsSLI` on `URL_PYTHON` returns HTTP 200, so
  `check-downloads.sh`'s HEAD probe passes on it. A first attempt can lose the
  connection to the release asset host, which is what that probe's retry policy
  is there for, so the verdict is the retried result rather than one sample.
- `SHA256_PYTHON` was taken from the line for that exact file in the release's
  own `SHA256SUMS` asset, and re-read from it on 2026-09-07. It is NOT the hash
  of a local download, which would certify only that a file arrived intact from
  wherever it arrived from.

Taken from upstream, not measured here:

- The rest of the pin, that the archive extracts to a top-level `python`
  directory and that `install_only` is relocatable. Those are properties of
  how python-build-standalone builds that artifact, and the first thing a real
  `warmup.sh` run will confirm or contradict.

Not run:

- `warmup.sh` has NOT been run. Its first act is a network preflight over
  `scripts/downloads.txt`, and the macOS machine available here has no
  outbound network by design, so provisioning is the one path still resting on
  review rather than on a run. Correct whatever it turns up on its first real
  use.
- The `bundle.sh` steps that need an interpreter have NOT been run, for the
  same reason: they need one `warmup.sh` has not yet provisioned anywhere.
  That covers the interpreter copy, the thinning pass, the generated launcher,
  the smoke test and the signing loop.
- **A real BROWSER download onto a second Mac has not been run.** What is now
  covered is the mechanism rather than the transport: `release.yml`'s
  verification job marks the extracted distribution with
  `com.apple.quarantine` itself and asserts the launcher clears it, and the
  same mechanism was measured by hand on macOS 26.6 through a `ditto` archive
  that was marked and re-extracted. What no step reaches is a mark set by an
  actual browser, which is the only thing that sets it in production.

## Prerequisites

- A Mac with Apple silicon. The backend is built `-arch arm64` and Intel Macs
  are out of scope.
- The **Xcode command line tools**, not full Xcode:
  `xcode-select --install`. They carry clang, make, git, curl, `python3` and
  the macOS SDK whose `Metal.framework` is the only framework the backend
  links.
- A Python 3.10 or newer, **to build**. `warmup.sh` builds the developer
  environment with it and installs nothing without it. macOS ships 3.9, which
  the frontend's PEP 604 unions do not parse, so a stock machine needs one
  installed (`brew install python@3.11`, `uv python install 3.11`, or the
  python.org installer). This is a build requirement only; see below.

**Running a finished distribution requires none of the above.** It carries its
own interpreter, its own frontend dependencies and its own JupyterLab. It
installs nothing, downloads nothing, and needs no Xcode, no command line
tools, no Homebrew and no Python of the user's; an example notebook it ships
may need both a network and `git`, which the next paragraph and "What an
example notebook fetches" below set out. The prerequisites in this section are
for the machine that BUILDS.

**An example notebook is a separate question, and the difference is worth
stating rather than leaving a reader to find it.** The distribution is closed:
it starts, serves JupyterLab and runs the solver out of its own contents, and
nothing it does by itself reaches the network or the user's tools. An EXAMPLE
is a program the user chose to open. `bundle.sh` ships the notebooks and the
scripts and none of the assets they run on, so ten of the twenty-nine shipped
notebooks fetch their own geometry the first time they are run, into
`cache/ppf-cts` inside the distribution. Four of those ten shell out to `git` and so need `git` on
`PATH`; the other six fetch over HTTPS from Python and need only a network.
The remaining nineteen build their scenes from geometry they generate and run
with no network at all. Which notebook is which is under "What an example
notebook fetches" below.

Stated as two claims rather than one: nothing the distribution itself runs
comes from the user's machine, and four of the examples it ships need `git`
while six more need a network. A user who opens none of those ten never finds
the second claim out.

No admin privileges are required, and nothing is installed system wide.
`warmup.sh` writes only into this directory and into the developer
environment.

## The Metal toolchain, and why nothing here provisions one

**The Metal shader compiler is a SEPARATE COMPONENT, and installing full Xcode
does not supply it**: it needs `xcodebuild -downloadComponent MetalToolchain`.
Nothing here provisions it, and nothing here requires it. The backend can always
assemble its shader and compile it at run time through `newLibraryWithSource`,
which is what a machine without the component does.

Test for it by INVOKING the tool, never by locating it. `xcrun --find metal`
resolves to a path inside the Xcode toolchain whether or not the component is
installed, so a check that only looks for the binary reports a toolchain that
cannot run. `xcrun metal --version` is what settles it, and without the component
it fails with "cannot execute tool 'metal' due to missing Metal Toolchain".

Availability varies by machine. On one Apple M1 machine running macOS 26.5.2 the
component was installed as of 2026-08-14, reporting version 32023.864.

`build.sh` step 4 uses it when it is there. It runs the backend once with no
scene, through `ppf-metal-shader-dump`, to get the assembled shader text and the
exact compiler arguments the backend's own compile options amount to; compiles
each library into a `.metallib` under those arguments; installs the results
beside `libppfbe_metal.dylib`; and then re-runs the tool and requires every
library to load pre-compiled. Without the component the step prints one
paragraph and continues. `PPF_MAC_METALLIB=0` skips it either way.

**A stale artifact is a wrong answer, not a slow start**, so each file is NAMED
by a hash over the assembled source, the math mode read back out of the compile
options, and the language version. An edited kernel names a file that is not
there and the run compiles from source. This script never computes that name: it
asks the backend for it, which is what makes the two sides agree. Measured on the
M1: one comment line added to a shared header moved the solver library's key,
3 of 4 libraries stayed pre-compiled and one fell back to a 6.4 s source compile;
a truncated, empty or garbage file is reported and falls back the same way.

What it is worth, one small scene on that M1: 4.5 s against 10.5 s when the OS
shader cache has been evicted, and nothing at all when that cache is warm, which
is the steady state on a machine that has run the shader before. The point is
that the OS cache is evictable and not ours to key.

The OTHER half of startup, creating the compute pipeline states, is 16.2 s cold
and is cached by the backend itself through `MTLBinaryArchive`
(`crates/ppf-cts-compute/metal/pipeline_cache.hpp`). Nothing here has to
provision it and no build step produces it: the archive is written on the first
run into `~/.cache/ppf-cts/metal-pipeline-archive` and keyed on the assembled
source, the device, the OS build, the compile options and which route produced
the library, so a stale one misses and recompiles rather than loading. It cannot
be shipped, because a binary archive is keyed to a device and a driver. Two
consequences for packaging. A distribution ships with an empty archive, so a
user's FIRST run pays that half of the cold startup (about 20.6 s on that M1 for
a small scene with the shader libraries pre-compiled, about 27.7 s without) and
later runs about 4.5 s. And the archive lives under the user's home, never
inside the distribution directory or the source tree, so a read-only install
directory is fine. `PPF_METAL_ARCHIVE=0` turns it off, `PPF_METAL_ARCHIVE_DIR`
relocates it.

A macOS build is therefore one command with no flags:

```bash
cargo build --release
```

`crates/ppf-cts-solver/build.rs` selects the real backend for the host, which
on macOS is Metal, and hard-errors when a host has neither Metal nor CUDA. There
is nothing to fall back to, and that is the point rather than a gap: a backend
that computed no physics while writing to the same `target/release/` path would
silently overwrite a working solver, and the live session that followed would
get fake results from a binary that looks right.

The Rust CPU backend (`cargo build --release --features cpu`) is the
replacement: it is driven by the same neutral Newton driver as CUDA and Metal,
runs every example scene at about 30x the wall clock, and refuses by name the
few capabilities it does not carry yet (`src/driver/refusal.rs`).
On this Metal-capable machine it prints a banner and proceeds rather than
refusing, because it computes the same physics from the same neutral kernel
bodies and is incomplete rather than fake. It still lands on the same
`target/release/` path, so rebuild the Metal backend before any live use.

## Build steps

### 1. Provision

```bash
./warmup.sh
```

Checks the host, then verifies every URL in `scripts/downloads.txt` **before**
downloading anything, then installs Rust into `rust/` if the machine has no
cargo, then builds the developer environment, then provisions the interpreter
that `bundle.sh` will ship (`PPF_MAC_PYTHON=0` skips that last step; see "The
bundled interpreter").

The dependency list is read out of the repository's own `warmup.py`
(`python_packages()`, `tetra_packages()`) rather than restated here, and the
result is verified against that file's `REQUIRED_PACKAGES` in the environment
that was just created. A partial pip install fails the run: a frontend without
scipy silently takes a different pin-diffusion path than every other host, and
one without a tetrahedralizer cannot build a SOLID scene at all.

The environment lands where `warmup.py`'s `get_venv_path()` says, which is
`$HOME/.local/share/ppf-cts/venv`. That is also what the Metal fixture target
defaults `FIXTURE_PYTHON` to, so the fixtures work with no further setup. Set
`PPF_CTS_VENV` in `config.sh` to put it elsewhere.

### 2. Build

```bash
./build.sh
```

Runs `cargo build --release`, then verifies that all three workspace
default-members landed in `target/release` and that the solver binary really
does link `libppfbe_metal.dylib`, then pre-compiles the shader libraries
where the Metal Toolchain is installed, then writes `start.sh`.

The shader step also runs the backend's two parity self-tests, since those are
what compile the vec-op and solver libraries in the first place. A machine that
cannot reproduce this backend's arithmetic therefore cannot produce an artifact,
and the build says which comparison failed.

### 3. Run

```bash
./start.sh
```

JupyterLab on the port in `config.sh`, serving the source tree's own
`examples/`. This is the developer path: `build.sh` writes it, and it runs the
developer environment rather than the bundled interpreter. The distribution has
an entry point of its own, a different file with different defaults, described
under "The distribution" below.

### 4. Package (optional)

```bash
./bundle.sh
```

Packages `dist/<name>/`: the binaries, the backend dylib, the
generated entry library `ppf_entries.metallib`, the pre-compiled shader
libraries when `build.sh` produced any, the frontend, the examples, the crate
source roots the frontend harvests log-channel names from, and the bundled
interpreter. It rewrites each Mach-O file's load commands so nothing points
back into the build tree, thins every universal file to arm64, generates the
launcher and the documentation, pre-compiles the Python bytecode, asserts that
no reference escaped, imports the whole stack once as a smoke test, and only
then signs. It produces that one directory and nothing else: no archive, no
disk image, and no option to build anything but a directory.

**Signing is the last thing that touches the contents**, because rewriting a
file after it is signed invalidates that file's signature. Generation, pruning,
thinning and bytecode compilation therefore all happen before it, and so do the
gates, so a payload that is going to be rejected is rejected in seconds rather
than after the signing loop.

The shader libraries travel with the dylib, in `bin/`, because the backend
resolves that directory through `dladdr` on its own code. Verified by relocating
the flat `dist/` an earlier run produced: it reports its own directory as the
shader library directory and loads the hash-named shader libraries out of it,
never the build tree's. That measurement covers the shader cache and nothing
else; `ppf_entries.metallib` sits in the same directory and is resolved by a
different route, which the next paragraph takes up.

`ppf_entries.metallib` sits in that same directory and is resolved differently,
which is worth keeping straight. The shader libraries are a cache: each is
named by a hash over the text it was compiled from, absence is a supported
state, and the backend finds them through `dladdr`. The entry library is not
optional, so the packaging asserts that file by name rather than counting it
among the cache. `be_open` takes the directory the CALLER names in
`OpenConfig.library_dir` and, where the caller names none, resolves the one
this dylib was loaded from. Nothing in the shipped solver names one, which is
what the defect recorded under "Status" above was about.

`bundle.sh` does **not** notarize, and it says so at the end rather than
pretending. See "Signing and notarization" below.

## The bundled interpreter

So that a finished distribution runs on a Mac with no Python, `warmup.sh`
provisions one into `build-mac-native/python` and `bundle.sh` copies it in. The
pin lives in `scripts/downloads.txt` with every other download:

```
cpython-3.12.14+20260901-aarch64-apple-darwin-install_only.tar.gz
```

a python-build-standalone `install_only` build for arm64, about 24 MB, from
release `20260901`. It is the 3.12 series `.github/workflows/both-backends.yml`
and `unit-tests.yml` already pin for their own frontend dependencies, so the
series this ships is exercised elsewhere in the repository. Each of those
workflows installs a subset of its own rather than `warmup.py`'s whole list,
and neither installs JupyterLab at all, so what they establish is the series
and not the package set. `scripts/downloads.txt` says the same beside the pin
itself. Three properties of that choice are what the rest of this section
rests on.

**`install_only` is relocatable by construction.** Such an interpreter computes
its home from its own path, so the tree can be moved, copied into a
distribution, and copied again by the user with no rewriting. A CPython that is
not relocatable produces a distribution that runs only on the machine that
built it, which is the failure this directory exists to avoid.

**The dependencies install into that tree, not into a venv over it.** A venv
records an absolute `home =` in its `pyvenv.cfg` and its `bin/python` points at
the base interpreter by absolute path, so a venv is exactly the thing that
would undo the relocation. `pip install` therefore targets
`build-mac-native/python/bin/python3` directly and lands in that tree's own
`site-packages`. The package list is not restated here: it is the same one the
developer environment gets, read out of the repository's own `warmup.py`.

**The canonical path is `python/bin/python3`.** In an `install_only` tree that
is a relative symlink to the versioned binary, so it survives `rsync -a`,
`ditto` and a Finder copy, and no script has to know the minor version.
`bin/python` may not exist and is not used. Where a version-numbered directory
is genuinely needed, `bundle.sh` asks the interpreter
(`sysconfig.get_paths()`) rather than writing `3.12` down a second time.

### Two environments, with different jobs

| | path | who uses it |
| --- | --- | --- |
| Developer | `$HOME/.local/share/ppf-cts/venv` | `build.sh`'s `start.sh`, the Metal fixture target's `FIXTURE_PYTHON`, anything run from the source tree |
| Bundled | `build-mac-native/python` | `bundle.sh`, and the distribution it produces |

The second does not replace the first, and nothing about the developer
environment changed. They have different lifetimes: one belongs to this
machine, the other is a build artifact that gets copied to a user's Mac.

### Re-running warmup

A second `./warmup.sh` verifies rather than reinstalls. It checks that the
interpreter runs, that it reports the pinned version, that every name in
`warmup.py`'s `REQUIRED_PACKAGES` imports under it, and that `jupyterlab`
imports. That last one is checked separately and by name because
`REQUIRED_PACKAGES` does not list it, and launching `-m jupyterlab` is the
entire purpose of the distribution.

A failed check stops the run and prints the recovery,
`rm -rf build-mac-native/python && ./warmup.sh`. It does not wipe and redo on
its own: a half-provisioned tree quietly replaced hides the cause, and the
whole point of a shipped interpreter is that the user cannot repair it.

`PPF_MAC_PYTHON=0` skips the step, so a developer who never packages a
distribution is not blocked by a 24 MB download. `bundle.sh` then refuses,
naming both the script and the switch, rather than producing a distribution
that silently needs the user's own Python.

### The cached tarball, and the fleet with no egress

`warmup.sh` caches the archive in `build-mac-native/downloads` and skips the
download when the file is already there, the same shape `rustup-init.sh`
already has. That is the path for a Mac with no outbound network: copy the file
into that directory by hand and warmup uses it rather than trying to fetch.

`SHA256_PYTHON` is checked on every run that provisions from the archive,
whether that run downloaded the file or found a relayed copy already in
`downloads/`, because the hand-relayed copy is the one that needs checking. A
hash computed only after our own download would certify nothing. A later run
finds the provisioned tree instead and verifies that, reading the archive not
at all, so the tarball can be deleted once `python/` exists. The macOS spelling
is `shasum -a 256`; `sha256sum` does not exist there. A truncated relay is
otherwise a half-extracted interpreter that mostly works.

This does not make a fully offline Mac able to provision. `check-downloads.sh`
still runs first and still needs the network, which is already true today and
recorded under "Running the example suite on a Mac with no egress" below. What
the cache buys is a machine that can reach the network for the preflight and
has had the 24 MB placed by hand.

## The distribution

`bundle.sh` produces one thing a user can act on:

```
build-mac-native/dist/
    ppf-contact-solver/
```

and nothing else. There is no archive step and no disk image step: a
distributor who wants one file wraps that directory themselves, and
`bundle.sh` prints the `ditto` command for it at the end.

This section describes the distribution the scripts are written to produce.
Nothing in this section has been observed on a finished payload: the run that
would produce one has not happened, which "Not run" under "Status" above
records.

### Layout

```
ppf-contact-solver/                    the distribution directory, and the tree root

`<name>` is `ppf-contact-solver` for a local build and the archive's own stem
for a release, `ppf-contact-solver-<date>-macos-arm64`, which `bundle.sh` takes
from `PPF_MAC_DIST_NAME`. `ditto --keepParent` carries that name into the
archive, so a release unzips to a folder carrying its own version rather than
onto the previous download. A local build keeps the short name, since nothing
there unzips anything and the path is the one this file names throughout.
    ppf-contact-solver                 the launcher, a shell script, what a user runs
    config.sh  README.txt  THIRD_PARTY_LICENSES.txt
    bin/libppfbe_metal.dylib           the Metal backend
    bin/ppf_entries.metallib           the generated entry library, required
    bin/<hash>.metallib                the shader cache, optional, zero or more
    target/release/                    solver, server, the PyO3 cdylib
    frontend/  examples/  crates/*/src/
    python/                            the bundled interpreter
    python/share/jupyter/              what JUPYTER_PATH points at
    .git/branch_name.txt               the branch stamp, read by branch_for
```

**The distribution directory is the tree root**, which is the role the
repository root plays for a developer running out of a source checkout.
`frontend/__init__.py` resolves the root from its own `__file__` as
`dirname(__file__)/..`, so it finds
`target/release/lib_ppf_cts_py.dylib` under that directory, and the relative
depth from `target/release` to `bin` is `../../bin`, which is what the
`@executable_path` and `@loader_path` rewriting in `bundle.sh` step 6 writes.

**Two files in the distribution carry the basename `ppf-contact-solver`, and
they are different programs.** The one at the top is the launcher, a shell
script, and is what a person runs. `target/release/ppf-contact-solver` is the
solver, a Mach-O the frontend runs once per scene. Nothing in the launcher runs
the solver directly.

Neither the version nor the minimum macOS is written down. Both are read off
the artifacts and stamped into the launcher: the version from the solver
binary's own `--version`, and the minimum from the highest `minos` over
every Mach-O file in the payload after thinning. The distribution cannot run
below what its own binaries require, so that is the only correct source for it,
and a number typed in by hand would be a claim nobody checks. The launcher
refuses on an older macOS naming both numbers, because a directory has nothing
that would refuse for it and the alternative is a dyld diagnostic naming a
symbol, which reads as a corrupt download.

### What running it does

```bash
cd path/to/ppf-contact-solver
./ppf-contact-solver
```

1. Resolves the distribution directory from the launcher's own path, walking
   any symlink chain by hand and bounding that walk, so it runs from any
   working directory, through a symlink, and from a path containing a space.
   An unbounded walk over a symlink cycle would hang with nothing printed,
   which is the failure this program exists to make legible.
2. Refuses on an Intel Mac, on a macOS below the stamped minimum, on a missing
   payload file naming its path, and on a `config.sh` whose `PORT` is empty or
   not a number. Each refusal names what failed and what to do about it, in
   plain text on the terminal. `--help` prints the usage and exits without
   starting an interpreter, which is what makes it cheap enough to be the
   build-time and CI-time probe of the launcher.
3. Reads `PORT` from the `config.sh` beside it, and honors `PPF_CTS_VENV` as an
   override for a developer who wants their own packages. With no override it
   uses the interpreter at `python/bin/python3`.
4. **Keeps the user's `PATH` and appends `/usr/bin:/bin:/usr/sbin:/sbin`.**
   Nothing the program itself runs is resolved through `PATH`: the interpreter
   and every binary are named by absolute path. What `PATH` decides is what a
   NOTEBOOK finds, and the four notebooks that clone need `git` while video
   export needs `ffmpeg`, both of which are the user's to supply. Appending the
   system directories is what guarantees the launcher's own `dirname`,
   `readlink`, `sleep`, `mkdir`, `rsync` and `uname` are reachable from a
   `PATH` that carries none.
5. **Unsets `CARGO_TARGET_DIR`, `PYTHONHOME` and `PYTHONOPTIMIZE`, and assigns
   `PYTHONPATH` rather than appending to it.** Each of those four arrives from
   the user's own shell, which is the difference a terminal program makes.
   `_target_dirs()` in `frontend/__init__.py` honors `CARGO_TARGET_DIR` and
   searches ONLY it, never the tree's own `target`, so a developer who exports
   it would have the frontend look for the cdylib somewhere else. `PYTHONHOME`
   hides the stdlib. `PYTHONOPTIMIZE` redirects every import to `.opt-N.pyc`,
   which the build does not write, so with `PYTHONDONTWRITEBYTECODE` set the
   whole import graph recompiles on every start and nothing is stored, which
   presents as a program that is simply slow. A `PYTHONPATH` of the user's
   shadowing `frontend`, `numpy` or `scipy` changes what the solver computes,
   and this project treats a frontend package difference as a physics variable
   rather than a preference. Someone who wants their own packages uses
   `PPF_CTS_VENV`.
6. Points the frontend, the build worker and Jupyter at the selected
   interpreter and at that interpreter's own kernelspec, keeps Jupyter's state
   under `local/share/ppf-cts/jupyter` inside the distribution directory, and
   sets `PYTHONNOUSERSITE` so user-site packages do not reach it. A stray
   kernelspec or a user-site numpy would otherwise silently change what runs.
7. Serves the distribution's own `examples/` in place, so a notebook saves where
   it was shipped. That is why the directory has to be writable, and the
   launcher refuses by name when it is not.
8. Prints the distribution directory, the interpreter, the notebook directory,
   the state directory and the URL, then starts JupyterLab **in the
   foreground**. The server's log goes straight to this terminal, unfiltered.
   If that port is taken the server takes the next free one, and the exact
   URL is in that log; the launcher says so rather than pretending to know the
   port in advance.

The two interpreter paths are spelled differently on purpose: a venv always
carries `bin/python`, and an `install_only` interpreter tree is addressed
through `bin/python3`. The launcher refuses in two cases rather than falling
back. `PPF_CTS_VENV` set to a directory that holds no interpreter at
`bin/python` is a mistake worth naming, since a silent fallback would run the
wrong environment. No interpreter at `python/bin/python3` with `PPF_CTS_VENV`
unset says the distribution was packaged from a tree where `warmup.sh` had not
provisioned one, which step 1 of `bundle.sh` refuses to do.

JupyterLab opens the browser itself. That is a convenience and never a
requirement: it logs a failure to open and keeps serving, and the URL is on the
terminal either way. Opening it from the launcher instead would need a
readiness probe, a JSON parse of the server's runtime record and a poll loop,
all to learn a URL the server already prints.

Every failure prints to the terminal in plain text, naming what failed and what
to do. There is no dialog, no notification and nothing revealed in a file
browser: there is a terminal to print to.

### Stopping it

- **Ctrl+C.** The launcher traps INT, TERM and HUP, asks the server to stop,
  and exits with it. A user-requested stop exits 0; a server that fails on its
  own passes its status through.
- The launcher runs the server with job control on, so the server gets a
  process group of its own and one signal reaches it AND the kernels it
  started. Without that the server and its kernels sit in the launcher's own
  group, there is no group to signal, and a killed server leaves its kernels
  behind.
- The launcher sends SIGTERM first, because `jupyter_server` handles SIGTERM by
  shutting its kernels down and exiting, and SIGKILL cannot be handled at all.
  A server that has not exited within ten seconds is killed, and the kill goes
  to the group so the kernels go with it. Anything that has already left that
  group is a plain process the user can end; this program ends only what it
  started.
- JupyterLab's own **File > Shut Down** also works. The server exits and the
  launcher follows it.
- There is no lock file, no pid file, no log file and no separate stopper. The
  terminal is the log. Two copies started at once are two servers on two ports,
  which Jupyter's own port retry resolves, and they share nothing but the state
  directory.

There is no idle-shutdown timeout, on purpose. Killing a server a user
deliberately left open is a worse failure than one that lingers.

### Thinning to arm64

The distribution is arm64 by construction, so a slice for any other
architecture is dead weight that also has to be signed and walked. A census
over the payload found 15 universal files, from `debugpy`, `fontTools`,
`charset_normalizer`, `tornado` and `pyzmq`, each carrying an `x86_64` slice
that cannot run here, and zero `x86_64`-only files.

`bundle.sh` thins them with `lipo`, and three things about where that sits in
the run are what make it correct. It happens BEFORE signing, because rewriting
a file invalidates that file's signature. It happens before the deployment
target is read, because `otool -l` on a universal file prints the load commands
of every slice, so a minimum taken over a fat payload could name an
architecture that does not ship. And a file that is already single-architecture
is skipped rather than passed to `lipo -thin`, which errors on one; the
decision is a read of `lipo -archs` rather than an attempt and a recovery.

The result is asserted rather than assumed: after the pass, and again over the
final list once pruning is done, every Mach-O file in the payload must report
`arm64` and nothing else, and any other answer fails the build. A file that
carries no arm64 slice at all fails it too, because nothing here invents one.
The bytes saved are reported.

## What it writes, and how to remove it

Everything goes inside the distribution directory, and nothing goes anywhere
else:

```
<distribution>/local/share/ppf-cts/
    git-<branch>/<scene>/session/   the solver's own session output
    jupyter/config/                 JUPYTER_CONFIG_DIR
    jupyter/data/                   JUPYTER_DATA_DIR
    jupyter/runtime/                JUPYTER_RUNTIME_DIR
    jupyter/ipython/                IPYTHONDIR
<distribution>/cache/ppf-cts/       the asset cache and the Metal pipeline
                                    archive
    <hash>__<name>.npz              meshes and tetrahedralizations, cached
    downloads/                      preset meshes as fetched
    Codim-IPC/  tet-assets/         repositories an example sparse-cloned
    metal-pipeline-archive/         the compute pipeline states, per device
<distribution>/examples/            the notebooks, served and saved in place
```

**Removing the product is deleting the distribution directory.** There is
nothing else to undo.

The `.ppf-selfcontained` marker `bundle.sh` writes at the tree root is what
decides this. `compose_data_dir` and `default_cache_dir` in
`crates/ppf-cts-core/src/datamodel/app.rs` root the session data and the asset
cache inside a tree that carries it, which is also where a Windows tree has
always kept them. A file rather than a variable, because the launcher is not the
only way in: the Blender add-on spawns the distribution's `ppf-cts-server`
directly. The one cache the solver process writes itself, the Metal pipeline
archive, follows the same marker: the session launch script
(`crates/ppf-cts-core/src/datamodel/session/scripts.rs`) exports
`PPF_METAL_ARCHIVE_DIR` into the tree's cache directory before every solve,
keeping a value the user already set. The launcher sets the Jupyter and IPython
directories itself.

Most of `cache/ppf-cts` is put there by an example notebook rather than by the
program itself; see "What an example notebook fetches" below. Everything in it
can be produced again, so deleting it costs a re-fetch, a
re-tetrahedralization and a slower first solve, and nothing else.

A developer checkout carries no marker and keeps using `~/.local/share/ppf-cts`
and `~/.cache/ppf-cts`, where `warmup.sh`'s developer environment also lives.

**The signed payload is never written to.** What the program creates are new
directories beside the signed files. `bundle.sh` pre-compiles every `.pyc` at
build time with `compileall -f --invalidation-mode unchecked-hash`, and both
entry points set `PYTHONDONTWRITEBYTECODE=1`, so nothing is recompiled per run
and no signed file changes. `unchecked-hash` is what makes the pre-compiled
files valid regardless of the source timestamps a copy may or may not have
preserved. The directory has to be writable, and the launcher refuses by name
when it is not.

## What an example notebook fetches

The distribution ships the notebooks and the scripts, not the geometry some of
them run on: `bundle.sh` copies `examples/*.ipynb` and `examples/*.py` and
nothing else. A notebook whose scene is built from a mesh that is not in that
set therefore fetches the mesh itself, once, into the distribution's own
`cache/ppf-cts`, and reuses it on every later run.

A fetch started by a notebook is the only thing reached from inside the
distribution that touches the network or a tool of the user's, and it is why
the unqualified sentence "it downloads nothing" is not written anywhere in this
tree without this section beside it. The distribution is closed. An example is
a program the user chose to open. Ten of the twenty-nine shipped notebooks
fetch on their first run:

| Notebook | What it fetches | Needs |
| --- | --- | --- |
| `fitting` | two asset directories from the Codim-IPC repository | `git` |
| `large-animals` | `assets.zip` from Barrier-Free-Supplementary | `git` |
| `large-fluffy` | the same archive | `git` |
| `trapped-919539a` | the same archive | `git` |
| `fishingknot` | `fishingknot.ply` from this project's own release assets | network |
| `codim` | the `armadillo` preset mesh | network |
| `friction` | the `armadillo` preset mesh | network |
| `trampoline` | the `armadillo` preset mesh | network |
| `walkthrough` | the `armadillo` preset mesh | network |
| `roller` | the `knot` preset mesh | network |

The other nineteen build their scenes from geometry they generate, so they run
with no network at all and with nothing on the machine but the distribution.

**The four that need `git` need it on `PATH`, and the distribution does not
carry one.** `app.extra.sparse_clone(...)` runs `git clone --filter=blob:none
--no-checkout` and then a sparse checkout per path
(`crates/ppf-cts-core/src/extra.rs`), so those four notebooks are the only
place in this product that depends on a tool of the user's. On a Mac with the
Xcode command line tools installed, `git` is there and they work. On a Mac
without them, `/usr/bin/git` is not git: it is the developer-tools shim, and
invoking it raises the "install the command line developer tools" dialog. That
is the documented behavior of the shim rather than something measured here,
since no Mac without the tools was available; what is measured is that the four
notebooks reach `git` at all.

**Those four notebooks are the only remaining path to `git`, and getting the
count down to four took two changes worth naming so that neither is undone.**
`App.create(...)` resolves its session directory through `data_dirpath_for`,
which names the branch, and the branch came from `git branch --show-current`
run in the tree root. In a distribution the tree root carries no repository, so
that query fired on the first cell of every notebook, the nineteen that fetch
nothing included. `bundle.sh` stamps `.git/branch_name.txt` at the tree root at
build time, and `branch_for` in
`crates/ppf-cts-core/src/datamodel/app.rs` runs the query only where a `.git`
entry exists. Either change alone closes the path. Both are there because the
stamp is what gives the session directories a real branch name instead of
`git-unknown`, and the guard is what holds for a payload assembled by something
other than this script. Each is commented where it lives; this paragraph is
here so a reader of the README knows the start path was a real defect and not a
hypothetical one.

`sparse_clone` refuses rather than proceeding when it finds no git, so the
failure is loud and lands in the notebook cell. The message it refuses with was
written for the Windows payload and names MinGit, `start.bat` and
`choco install git`, none of which exists on macOS, so a Mac user is told to
repair something that was never there. Correcting it belongs in
`crates/ppf-cts-core/src/extra.rs`, outside this directory.

The other six fetch over `urllib` from Python and need only a network.
`frontend/_mesh_.py` resolves a preset name to a URL and writes the result into
`cache/ppf-cts/downloads`; `fishingknot` calls `urllib` itself in its first
cell and writes `fishingknot.ply` at the top of the cache instead. A machine
with no network fails those six at the cell that fetches, with the URL in the
traceback.

Everything fetched lands under the distribution's own `cache/ppf-cts`, so none
of this changes what removing the product is.

## The distribution never touches the user's macOS

This is a constraint on the design rather than a summary of it, so it is worth
stating as a list of things that are NOT done. Every item below is about the
distribution: what it does when the user runs the launcher, and what it does
while it serves JupyterLab. It is not a claim about an example notebook the
user then chooses to run, which is a program of its own and may fetch its own
assets; that half is "What an example notebook fetches" above, and the two
together are the whole of it.

- **No installer.** No `.pkg`, no postinstall script, nothing written to
  `/Library`, `/usr/local`, `/opt` or anywhere outside the user's own home. The
  release is a directory, and removing it is deleting that directory.
- **It installs nothing and downloads nothing at run time.** No Homebrew, no
  MacPorts, no `pip install`, no download, and no `xcode-select --install`
  prompt. Everything it needs to start, to serve JupyterLab and to run the
  solver is inside the distribution directory, put there at build time. Four of
  the notebooks it ships do run `git`, and six more do download, and that is
  the one line this list does not cover.
- **No reliance on the user's machine having anything, to start and to run a
  self-contained notebook.** No system or user Python, no Xcode, no command
  line tools, no toolchain of any kind. Nothing the program itself runs is
  resolved through `PATH`: every binary it starts is named by absolute path,
  and the `PATH` it exports is the user's own with the system directories
  appended, so a tool the user installed IS reachable from a notebook. What
  this rule forbids is depending on one. Two build-time gates enforce the rest
  rather than leaving it to review: one rejects any Mach-O in the payload that
  depends on an absolute path outside `/usr/lib` and `/System/Library`, which
  is how a wheel that linked the build machine's Homebrew would be caught, and
  one rejects any file anywhere in the payload that names the build tree.
- **Nothing persistent is registered.** No LaunchAgent, no LaunchDaemon, no
  login item, no PATH edit and no shell rc edit.
- **Nothing of the user's is modified, with one exception that is stated
  rather than hidden: the download mark on this folder.** The launcher clears
  `com.apple.quarantine` from the distribution directory at startup, prints one
  line saying it did, and clears it nowhere else: it walks with `xattr -s`, so
  a symbolic link is cleared as itself rather than followed to whatever it
  points at. Nothing in the BUILD clears anything, because what `bundle.sh`
  packages was never marked. See "Gatekeeper, and what the first run looks
  like" for why the launcher can do this at all and why it is the whole folder
  rather than the few binaries it starts.

Nothing the program writes is an exception to this either. Sessions, the asset
cache, the Metal pipeline archive, the Jupyter state and what a notebook fetches
all land inside the distribution directory, so removing that directory removes
them.

## What this directory does not provision

- **A Python interpreter for the BUILD host.** `warmup.sh` requires one that
  already exists, 3.10 or newer, and says how to get one when it finds none.
  This is separate from the interpreter it provisions FOR THE DISTRIBUTION,
  which is pinned in `scripts/downloads.txt` and described under "The bundled
  interpreter" above. The build host needs its own because the developer
  environment is built before there is a bundled interpreter to build it with,
  and because `warmup.py`'s canonical package list is what both of them are
  read from.
- **ffmpeg.** The frontend falls back to `shutil.which("ffmpeg")` and skips
  video export when there is none. If a macOS ffmpeg build is ever added, it
  takes the pinned pointers from `build-win-native/scripts/downloads.txt`,
  which is already read by two platforms, rather than adding a second copy
  here.
- **git.** The build host gets it from the command line tools. Nothing puts one
  inside the distribution either, and four example notebooks call it; see "What
  an example notebook fetches" above. Bundling a git would make the
  distribution carry a second toolchain to serve four of twenty-nine examples,
  so the requirement is documented rather than removed.
- **The Metal Toolchain component.** `build.sh` uses it when it is installed and
  says so and continues when it is not. Downloading it here would make it a
  prerequisite of every build from source, in exchange for a startup cost the OS
  shader cache already absorbs on any machine that has run the shader once.

## Running the example suite on a Mac with no egress

`examples/run_suite.py` runs the notebooks on whichever backend the host carries.
Getting it to run on a locked-down Mac took four fixes, all of them one-time and
none of them obvious, so they are recorded rather than rediscovered.

- **`warmup.sh` / `warmup.py` cannot run there at all.** The box has an empty
  egress list, so pip does not fail, it HANGS on connect. Provision by carrying
  wheels in, never by running warmup on the Mac.
- **The venv may have no `pip` module.** Ship pip's own wheel and bootstrap
  through it, which needs no install step of its own:

  ```bash
  # on a networked host: download for the MAC's tags, not this host's
  python -m pip download --only-binary=:all: --platform macosx_15_0_arm64 \
      --python-version 311 --implementation cp --no-deps --dest wheels \
      pytetwild==0.4.2
  python -m pip download --only-binary=:all: --no-deps --dest wheels pip
  # on the Mac, offline
  python wheels/pip-*.whl/pip install --no-index --find-links wheels pytetwild==0.4.2
  ```

  `--no-index` is what guarantees the install cannot reach for the network and
  stall.
- **The platform tag has to be read off PyPI, not guessed.** `pytetwild` 0.4.2
  publishes only `macosx_15_0_arm64` for macOS, so `--platform macosx_11_0_arm64` and
  `macosx_14_0_arm64` both report "no matching distribution" and read like the
  package having no macOS build at all. `tetgen` 0.8.4 is `macosx_11_0_arm64`.
  The two differ, so they are downloaded in separate invocations.
- **Copying the packages from a Linux host does not work**, and cannot be made
  to: `pytetwild` and `tetgen` are compiled extensions, so an x86-64 Linux
  site-packages is the wrong architecture and the wrong object format.

**The dependency set is discovered by RUNNING the suite, not by reading a list.**
Two packages bit in one session, each surfacing only when a notebook reached the
line that needed it: `pytetwild` and `tetgen` (13 of the 24 notebooks
tetrahedralize) and then `trimesh`, which `app.mesh.preset(...)` imports and
which `roller.ipynb` reaches at its second line. Expect more, and expect each to
present as a `ModuleNotFoundError` seconds into a scene rather than at startup.
The versions installed for the Mac, matching the CUDA reference host:
numpy 2.4.4, scipy 1.17.1, tetgen 0.8.4, trimesh 5.0.0. pytetwild is pinned in
`warmup.py`'s `tetra_packages()` (0.4.2), which every provisioning path reads.

**Pin the versions to whatever the CUDA reference host has, and treat that as a
physics requirement rather than tidiness.** Cross-platform simulation divergence
in this project has rooted in the Python frontend rather than the solver: a
frontend without scipy took a surface-only fallback in `_build_solid_pin_fields`
and produced a different pin set, and an unpinned numpy flips borderline pin
vertices. Measured here, the Mac had numpy 2.4.6 against the reference host's
2.4.4 and was pinned back. 13 of the 24 non-large notebooks tetrahedralize, so
without `pytetwild` most of the suite cannot even build its scene.

## Signing and notarization

Every Mach-O file in the payload is signed, one at a time, and nothing else is.
Per-file signing is required because rewriting a load command invalidates the
signature and an arm64 binary with an invalid signature does not run at all.
The interpreter's own binaries arrive already ad-hoc signed and are re-signed
anyway, since `bundle.sh` rewrites some of them and a mixed set of signatures
under one identity is not what a distributor wants to ship.

**A directory carries no seal over its non-executable files, and that loss is
worth stating rather than working around.** The `.metallib` files, the Python
sources and the launcher itself are covered by no signature; each Mach-O file
carries its own. `codesign` on a non-Mach-O file stores the signature in an
extended attribute rather than in the file, nothing on the running path
verifies it, and an extended attribute does not survive every way a user may
unpack an archive, so signing those files would be a gesture rather than a
check. This is a property of the form. Do not add a checksum manifest to
compensate, and do not sign anything a loader will not verify.

- With no `MAC_CODESIGN_IDENTITY`, `bundle.sh` signs **ad-hoc**. The payload
  runs on the machine that built it and Gatekeeper rejects it anywhere else.
- With `MAC_CODESIGN_IDENTITY` set to a Developer ID identity in the keychain,
  it signs with the hardened runtime and a secure timestamp, and fails loudly
  when the identity is not there.

When an identity is set, the signature also carries three entitlements
(`allow-jit`, `allow-unsigned-executable-memory`, `disable-library-validation`)
because `numba` is in the frontend's package list and JIT-compiles. Under the
hardened runtime, which comes with a Developer ID signature and not with an
ad-hoc one, llvmlite's executable memory is refused without them. That defect
would appear only in the signed release and never in the ad-hoc build anyone
tests, which is why the entitlements are specified rather than discovered.

**Notarizing and STAPLING are two different reaches, and only the staple needs
a container.** `xcrun notarytool submit` accepts a `.zip`, which `bundle.sh`
prints the `ditto` command for at the end, so a distribution can be notarized
as it stands given an Apple Developer account. What cannot be done to a
directory is stapling the ticket to it: `xcrun stapler staple` takes an
`.app`, a `.dmg` or a `.pkg`. An unstapled notarization still clears
Gatekeeper, by an online check against Apple at first launch, so what the
missing staple costs is a first run with no network. A distributor who needs
an offline-verifiable ticket wraps the directory in a container of their own;
nothing here produces one. `config.sh` is the single place credentials would be
named. Never put an app-specific password in it, since it is tracked in git;
use a `notarytool` keychain profile, or export the password.

### Gatekeeper, and what the first run looks like

The payload is signed but not notarized, so a copy the user downloaded through
a browser is marked `com.apple.quarantine` and Gatekeeper would refuse it.
**The launcher clears that mark on its own folder at startup**, prints one line
saying so, and starts normally. The mark is set by the application that
downloaded the file, so a copy fetched with `curl`, `scp`, `git` or `rsync`
carries none and nothing is printed.

**Why the launcher can do this at all.** Gatekeeper assesses the exec of a
Mach-O file. The launcher is a shell script, read by `/bin/bash`, which is the
system's own binary, so it runs while marked and can clear the folder before
anything it starts is assessed.

**Why it is the whole folder and not the binaries the launcher starts.** Every
Mach-O is assessed when it is LOADED, not only when it is started, and the
bundled interpreter brings hundreds of extension modules that are `dlopen`ed
rather than exec'd. Clearing only the launcher's own targets would get past the
launcher and fail at the first `import`. Plain data files are readable while
marked, so it is the loadable files that make the walk mandatory.

**Why `-s`.** Without it `xattr` follows a symbolic link, which leaves the
link's own mark in place and reaches whatever the link points at, possibly
outside the folder; and a link pointing nowhere is an error that would end the
launcher under `set -e`. With `-s` the walk exits 0, prints nothing, clears the
links themselves and touches nothing outside the distribution.

**What it deliberately does not do.** It does not decide for someone who has
not decided: it runs because they started this program, it clears the mark on
that folder only, and it says what it did. If part of the folder stays marked,
which is what a folder owned by another user or on a read-only volume looks
like, it says that too, and the interpreter check further down names both the
Open Anyway click and `xattr -s -d -r com.apple.quarantine` by hand.

Measured on macOS 26.6, and each of these decides a line of the code above: a
quarantined shell script run from a terminal executes normally; an ad-hoc
signed Mach-O whose mark was cleared execs, while `spctl --assess` still
reports `rejected`, so **`spctl`'s verdict is not the gate the exec meets**;
`xattr -d -r` without `-s` exits 1 and prints one line per broken symbolic link
even on a clean tree, while `xattr -s -d -r` exits 0 silently; and a real
`ditto` archive, marked and re-extracted, carries the mark onto every entry,
after which the generated launcher clears all of them and the payload binaries
run.

No build-time assertion reaches this risk, so a gate in the workflow covers it:
`release.yml`'s `verify-macos` job marks the extracted distribution itself
and asserts the launcher clears it. What that gate cannot reach is a real
browser download, which is the only thing that sets the mark in production.

Notarization would still buy something this does not: a copy Gatekeeper accepts
with no file modified at all. It remains the better answer wherever an Apple
Developer account is available, and `notarytool submit` accepts a `.zip`, so
only the STAPLE needs a container. Without a staple Gatekeeper checks Apple
online at first launch.

## Updating tool versions

Every URL lives in `scripts/downloads.txt`. To bump a tool, edit that entry's
`URL_*` and `FILE_*` lines together, then verify the new pointer:

```bash
scripts/check-downloads.sh
```

`warmup.sh` runs the same check on every invocation, before its first download,
so a rotted pointer costs seconds instead of a whole provision. Do not loosen
its timeouts to make a run pass; an unreachable pointer is a result.

## Script reference

| Script | Purpose |
| --- | --- |
| `warmup.sh` | One-time provisioning: host checks, Rust, the developer environment, the bundled interpreter |
| `build.sh` | `cargo build --release`, artifact verification, writes `start.sh` |
| `bundle.sh` | Packages `dist/<name>/`, rewrites load commands, thins to arm64, verifies, signs |
| `config.sh` | Local settings: port, environment path, signing identity |
| `start.sh` | Generated by `build.sh`; runs JupyterLab over the source tree from the developer environment |
| `scripts/downloads.txt` | Single source of truth for tool URLs |
| `scripts/check-downloads.sh` | Verifies every URL before any download |
| `scripts/load-downloads.sh` | Parses the manifest into the caller's environment |

There are two entry points and they are not interchangeable.
`build-mac-native/start.sh` is the one in the table: `build.sh` writes it, it
requires the developer environment, it serves the source tree's `examples/`,
and it keeps Jupyter's state in `build-mac-native/jupyter`. The distribution's
entry point is its own `ppf-contact-solver` launcher, written by `bundle.sh`,
which defaults to the bundled interpreter with `PPF_CTS_VENV` as an override,
serves the distribution's own `examples/`, and keeps Jupyter's state under its
`local/share/ppf-cts`. See "The distribution" above.

Two environment switches, each defaulting to on, each skipping one optional
step. Neither is a `config.sh` key: they are read where they are used, the way
`PPF_MAC_METALLIB` already was.

| Switch | Read by | Setting it to 0 |
| --- | --- | --- |
| `PPF_MAC_METALLIB` | `build.sh` | Skips pre-compiling the shader libraries |
| `PPF_MAC_PYTHON` | `warmup.sh` | Skips the bundled interpreter, and `bundle.sh` then refuses |

`clean-build.sh`, `clear-cache.sh` and `clear-all.sh` have Windows
counterparts and are not written yet. Removing `rust/`, `downloads/`,
`python/`, `jupyter/` and `dist/` by hand is the whole of what they would do
here.

## Notes

- Once `warmup.sh` has run, this directory holds a provisioned toolchain,
  including the bundled interpreter under `python/`. Never rsync it and never
  `--delete` against it; exclude it from any sync, as the Metal port's own
  sync command does.
- Everything provisioned or generated here is gitignored. Only the scripts,
  `scripts/downloads.txt` and this file are tracked.
