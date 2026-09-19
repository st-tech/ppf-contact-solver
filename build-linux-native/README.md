# Linux Build

`build-linux-native/` builds ZOZO's Contact Solver on a Linux x86_64 or aarch64
host and packages it as a self-contained distribution: one directory holding the
solver backends, a relocatable Python interpreter with every frontend package,
JupyterLab, a slim ffmpeg and a launcher. The person who runs it needs the GPU
vendor's driver for the GPU backend, where the distribution carries one, and
nothing else: no CUDA toolkit, no ROCm installation, no Python, no compiler, no
root.

It is the counterpart of `build-win-native/` and `build-mac-native/`, and
follows their shape: `warmup.sh` provisions, `build.sh` compiles, `bundle.sh`
packages, and `scripts/downloads.txt` is the single source of truth for what is
downloaded.

## Architectures and backends

Every script builds for the host it runs on, and nothing cross-compiles.
`scripts/platform.sh` decides, from `uname -m` and `PPF_LINUX_BACKENDS` in
`config.sh`, what each of them builds and ships:

| Host | Backends by default | Other sets | What else differs |
| --- | --- | --- | --- |
| x86_64 | CUDA, ROCm and CPU | `cuda cpu`, `rocm cpu`, `cpu` | the CUDA toolkit (NVIDIA's linux-x86_64 archives) and the ROCm SDK (AMD's TheRock tarball) are provisioned for the GPU backends built |
| aarch64 | CUDA and CPU | `cpu` | the CUDA toolkit is NVIDIA's linux-sbsa archives; `triangle`, which publishes no aarch64 wheel, is built from its pinned upstream source; no nasm |

A distribution carries every backend it was built with, each in its own
`target/<backend>/release`, and which one a run uses is resolved when the run
starts: the one GPU build present, or the first whose solver reports a usable
device (`frontend.get_backend`). `rocm` on aarch64 is refused because AMD
publishes ROCm for x86_64 hosts only, and a set without `cpu` is refused because
every distribution ships the CPU backend. Per-architecture downloads carry an
`_X86_64` or `_AARCH64` suffix in `scripts/downloads.txt`, and a host has no
fallback to another architecture's file.

## Prerequisites

The build host provides:

- Linux x86_64 or aarch64 with glibc. A musl host such as Alpine cannot build or
  run it.
- A C and C++ compiler (when CUDA is built, one CUDA 12.8's nvcc accepts: GCC up
  to 14), make, binutils (`readelf`, `strip`, `strings`, `objdump`), curl, tar,
  xz, git, rsync, `sha256sum` and pkg-config.
- A Python 3.10 or newer for the developer environment and for `bundle.sh`'s
  ELF audit. On aarch64 its headers too (`python3-dev`), because a wheel is
  compiled against it.

Nothing needs root. `warmup.sh` writes into this directory and into the
developer environment, and names each missing host tool with the package
manager command that installs it.

The host does NOT need a CUDA toolkit or a ROCm installation. `warmup.sh`
provisions whichever the backends need here, see "The CUDA toolkit" and "The
ROCm SDK" below.

## Build steps

### 1. Provision

```bash
./build-linux-native/warmup.sh
```

It installs, each only when absent and each verified on every run:

- Rust into `rust/`, only when the host has no cargo.
- The CUDA 12.8 toolkit into `cuda/`, from NVIDIA's per-component archives, when
  CUDA is among the backends.
- The ROCm SDK into `rocm/`, from AMD's TheRock tarball, when ROCm is among the
  backends. The math libraries the project neither compiles against nor ships
  (MIOpen, rocBLAS, hipBLASLt and their payloads) are left in the tarball, which
  brings the unpacked SDK from about 20 GiB to about 5.3 GiB.
- On aarch64, wheels built from pinned upstream source into `wheels/`, for the
  frontend packages that publish none there (see "Wheels built from source").
- A developer Python environment with the packages `warmup.py` lists, at
  `PPF_CTS_VENV` (default `$HOME/.local/share/ppf-cts/venv`).
- A relocatable CPython into `python/` carrying the same packages and
  JupyterLab, for the distribution.
- patchelf into `patchelf/`.
- nasm into `nasm/`, only on x86_64 and only when the host has no nasm 2.13 or
  newer.
- The slim ffmpeg into `ffmpeg/`, built from source with x264 and zlib linked
  statically, with its license texts and a record of the exact sources.

A second run is a check rather than a reinstall. The whole run is logged to
`warmup.log`. Two switches skip an optional part: `PPF_LINUX_PYTHON=0` skips the
bundled interpreter and `PPF_LINUX_FFMPEG=0` skips ffmpeg; `bundle.sh` then
refuses to package and names the switch.

### 2. Build

```bash
./build-linux-native/build.sh
```

It builds each backend `scripts/platform.sh` names, each in its own target
directory:

| Directory | Backend | Command |
| --- | --- | --- |
| `target/cuda/release` | CUDA | `CARGO_TARGET_DIR=target/cuda cargo build --release --features cuda` |
| `target/rocm/release` | ROCm | `CARGO_TARGET_DIR=target/rocm cargo build --release --features rocm`, with `ROCM_PATH` and `HIP_PLATFORM=amd` |
| `target/cpu/release` | CPU | `CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu` |

Every backend links the same executable name, so the directory is what keeps
them apart, and `build.sh` asks each binary `--backend` rather than trusting
the path. When CUDA is built it also checks that the CUDA backend library needs
no toolkit library at run time; when ROCm is built, the checks "The ROCm SDK"
lists. It writes `start.sh`, which serves the source tree's examples from the
developer environment and pins no target directory: the frontend searches the
built ones and resolves which backend a run uses.

### 3. Run from the source tree

```bash
./build-linux-native/start.sh
```

### 4. Package

```bash
./build-linux-native/bundle.sh
```

The distribution is written to `dist/ppf-contact-solver`, or to
`dist/$PPF_LINUX_DIST_NAME` when that is set. To hand it to someone:

```bash
tar -C build-linux-native/dist -czf ppf-contact-solver.tar.gz ppf-contact-solver
```

A tar archive keeps the executable bits and the interpreter's symbolic links.

## The CUDA toolkit

`warmup.sh` assembles the toolkit from NVIDIA's redistributable component
archives (`cuda_nvcc`, `cuda_cudart`, `cuda_cuobjdump`, `cuda_nvdisasm` and
their dependencies, listed as `URL_CUDA_*` in `scripts/downloads.txt`) into
`cuda/`. No installer runs and nothing is written outside this directory.

`build.sh` selects it through `PPF_CUDA_ROOT`, which the CUDA Makefile reads as
`NVCC := $(PPF_CUDA_ROOT)/bin/nvcc` (default `/usr/local/cuda`), and the solver's
build script asks that Makefile which nvcc it will use before checking the
release. `cuobjdump` must resolve from the same toolkit, because the release
build's FP64 guard reads the device code with it.

**The distribution ships no CUDA toolkit library.** The CUDA backend,
`libppfbe_cuda.so`, links the CUDA runtime statically and loads the driver's
`libcuda.so.1` at run time, so its only run-time dependencies are the C and C++
runtime libraries. `build.sh` asserts that the library names no toolkit library
and leaves no `cuda*` symbol undefined.

## The ROCm SDK

`warmup.sh` unpacks AMD's TheRock distribution tarball (the multiarch form,
which carries the device libraries for every target in
`crates/ppf-cts-compute/rocm/rocm_arch.txt`) into `rocm/`, and checks the HIP
release `hipcc` reports against `ROCM_HIP_VERSION` in `scripts/downloads.txt`.
AMD publishes no digest or signature for this channel, so the pinned hash was
computed from a downloaded copy: it detects a changed or truncated file and says
nothing about origin.

`build.sh` selects the SDK through `ROCM_PATH` and builds for AMD's platform.
The same sources build for NVIDIA hardware under `HIP_PLATFORM=nvidia`, and that
library loads and reports `rocm` too, so `build.sh` sets `amd` and refuses an
environment that asks for anything else. The library records the SDK's `lib/`
as its search path at link time, because an unpacked SDK registers no loader
path; the distribution rewrites that to `$ORIGIN`.

No AMD GPU is needed to build, and none was available to run, so `build.sh`
checks the library statically: it resolves `libppfbe_rocm.so` through the
solver's own search path, requires every function `backend_abi.h` declares to
be exported, refuses a solver that prints `linked: hip-nvidia`, and runs
`.github/workflows/scripts/check-rocm-fp-flags.py` (no compile selects a
fast-math device library) and `check-rocm-code-objects.py` (every target's
device image is present, disassembles to a non-zero instruction count and holds
no FP64 instruction).

**The distribution ships the ROCm runtime.** Unlike CUDA's, it is loaded
dynamically: `bundle.sh` walks the backend library's search path and copies
everything it loads outside the system set (HIP, the HSA runtime, comgr, the
profiler hook, the kpack loader, and TheRock's own copies of libdrm, libelf and
libnuma) into `bin/`, points each at `$ORIGIN`, and copies every license text the
SDK carries into `licenses/`.

## The bundled interpreter

The distribution's interpreter is a python-build-standalone `install_only`
build, which is relocatable as an installation. Packages are installed into
that tree rather than into a venv over it, because a venv records absolute
paths and does not survive being copied.

Every use of that interpreter during the build ignores a user site-packages
directory, `PYTHONPATH` and `PYTHONHOME`. pip treats a package found in
`~/.local/lib/python3.X/site-packages` as already installed, so without that an
install on a host whose user site holds numpy leaves numpy out of the tree and
reports success. `warmup.sh` also requires every package to resolve inside the
tree and runs `pip check`.

A host with no route to PyPI installs from a wheelhouse instead: set
`PPF_LINUX_WHEELHOUSE` (in `config.sh` or the environment) to a directory of
wheels, and pip runs with `--no-index`. Fill it elsewhere with
`pip download --only-binary=:all:` for Python 3.12 on the build host's
architecture; the example below is x86_64, and aarch64 takes the same tags with
`aarch64` in place of `x86_64`. pip's
`--platform` does not expand manylinux tags, so list every tag the target glibc
accepts, for glibc 2.28:

```bash
pip download --only-binary=:all: --python-version 3.12 --implementation cp \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
  --platform manylinux_2_26_x86_64 --platform manylinux_2_25_x86_64 \
  --platform manylinux_2_24_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 --platform manylinux_2_12_x86_64 \
  --platform manylinux2010_x86_64 --platform manylinux_2_5_x86_64 \
  --platform manylinux1_x86_64 --platform linux_x86_64 \
  -d wheels <packages from warmup.py> jupyterlab
```

On aarch64 the wheelhouse needs no `triangle`: `warmup.sh` builds it from
source, and finds the source in `downloads/src/triangle` when the host cannot
reach its repository.

## Wheels built from source

A frontend package that publishes no wheel for the host's architecture is built
from its upstream repository by `scripts/source-wheel.sh`, once for each
interpreter that needs it (the developer environment and the bundled one).
`scripts/platform.sh` names the packages; today that is `triangle` on aarch64.

- **The source is fetched, never carried.** This repository holds no
  third-party code. The package is fetched at the commit `scripts/downloads.txt`
  pins (`URL_<NAME>_GIT` and `<NAME>_COMMIT`, plus a pinned commit per
  submodule, checked against the one its parent tree records) into
  `downloads/src/<name>`, and built unmodified.
- **The build tools are pinned by hash** in
  `scripts/wheel-build-requirements.txt`, and the build runs with
  `--no-build-isolation`, so no build requirement is resolved at build time.
- **Floating-point contraction is off** (`-ffp-contract=off`). These packages
  carry exact geometric predicates, which a fused multiply-add breaks, and GCC
  and clang both fuse by default on aarch64. The built extension's instructions
  are counted, and a single fused one fails the build.
- **A record travels with the wheel.** `wheels/<wheel>.sources.txt` names the
  repositories, commits and flags, and `bundle.sh` copies it into the
  distribution's `licenses/`.

python-build-standalone records clang as the compiler it was built with. On a
host without clang the script builds with `cc`, which setuptools also uses for
the link.

## The distribution

### Layout

```
ppf-contact-solver/
  ppf-contact-solver           the launcher, which is what a person runs
  config.sh                    PORT
  README.txt                   the user-facing instructions
  THIRD_PARTY_LICENSES.txt
  licenses/                    CPython, ffmpeg, x264, zlib, this project, the GPU runtime's
                               terms (NVIDIA CUDA, or every text the ROCm SDK carries), and the
                               source record of each wheel built from source
  bin/libppfbe_<backend>.so    one per GPU backend shipped, and with ROCm the runtime
                               libraries it loads
  bin/ffmpeg
  target/<backend>/release/    that backend's solver, server and Python extension, one
                               directory per backend shipped
  python/                      the interpreter and every package
  frontend/  examples/  crates/
  .ppf-selfcontained           roots all state inside this directory
```

Two files carry the basename `ppf-contact-solver`: the launcher at the root, a
bash script, and the solver the frontend starts for each scene, in the
`target/<backend>/release/` of the backend that run resolved to. Setting
`CARGO_TARGET_DIR` to one of those directories names the backend explicitly.

### What the launcher does

```
./ppf-contact-solver                        start JupyterLab and serve the examples
./ppf-contact-solver python FILE [ARGS...]  run a script in the same environment, no JupyterLab
./ppf-contact-solver --help
```

Before starting anything it checks, and names by what is wrong:

- that the machine's architecture is the one the distribution was built for,
  which the launcher carries as `PPF_DIST_ARCH`;
- that the system C library is glibc at or above the floor read off the
  binaries when the distribution was built;
- that every shipped solver starts, which is where an older libstdc++ shows;
- that the interpreter starts, after clearing `PYTHONHOME`, `PYTHONPATH` and
  `PYTHONOPTIMIZE` and setting `PYTHONNOUSERSITE`.

It then points Jupyter, IPython, matplotlib and numba at `local/share/ppf-cts`
inside the distribution, sets `SSL_CERT_FILE` to the shipped CA bundle unless one
is already set, and leaves `LD_LIBRARY_PATH` alone. A browser is opened only when
`DISPLAY` or `WAYLAND_DISPLAY` is set; otherwise it prints the URL and an
`ssh -L` line for reaching it from another machine. Ctrl+C stops JupyterLab and
its kernels.

In a distribution with both backends, `CARGO_TARGET_DIR=<distribution>/target/cpu`
selects the CPU solver; a CPU-only distribution selects it by default. The
launcher keeps that variable only when it names a build inside the distribution
and drops any other value with a note, because the frontend would otherwise load
a build the distribution did not ship. `PPF_CTS_VENV` substitutes a developer's
own Python environment for `python/`.

### What the GPU needs

With the CUDA backend, the NVIDIA driver, 570 or newer, which is the oldest
branch the CUDA 12.8 runtime supports. With the ROCm backend, an AMD GPU the
`amdgpu` kernel driver supports, and read access to `/dev/kfd`. The CPU backend
runs without either, and a CPU-only distribution needs no GPU at all.

The floor for the system libraries is what the binaries were built against.
A release is built in an AlmaLinux 8 container, so it needs glibc 2.28 and
libstdc++ `GLIBCXX_3.4.25`: RHEL, AlmaLinux and Rocky 8, Ubuntu 20.04, Debian 10
and anything newer.

## What it writes, and how to remove it

Everything the program writes is inside the distribution directory:

| Path | What |
| --- | --- |
| `local/share/ppf-cts/` | sessions, and the Jupyter, IPython, matplotlib and numba state |
| `cache/ppf-cts/` | meshes and tetrahedralizations the examples fetch or compute, and the NVIDIA driver's compute cache |
| `examples/` | the notebooks, which save in place |

The `.ppf-selfcontained` marker is what roots the session data and the cache
there, and every entry point reads it (`datamodel::app::is_selfcontained`).
Deleting the directory removes everything. Nothing is installed elsewhere.

Some examples download meshes over HTTPS on their first run and keep them in
`cache/ppf-cts`. A machine with no network runs the examples that fetch nothing,
or runs every example once its `cache/ppf-cts` has been filled from a machine
that has.

## The gates `bundle.sh` applies

`bundle.sh` refuses to finish unless each of these holds over the final
payload:

- **ELF audit**, `scripts/elf-audit.py`. Every ELF file is found by its magic
  bytes, not its name. No `NEEDED`, `RPATH` or `RUNPATH` entry names the build
  tree. Every `NEEDED` library resolves inside the payload, or is on a short
  list of system libraries (the glibc family with the architecture's dynamic
  loader, `libstdc++.so.6`, `libgcc_s.so.1`, `libz.so.1`), or is covered by a
  named exemption that must match a file. No NVIDIA driver library is shipped.
  Resolution is modeled the way the loader does it and never asks `ldd`, which
  answers for the build host.
- **Architecture.** Every ELF file is a 64-bit little-endian object for the
  host architecture, read off its header's `e_machine`, or is named in
  `bundle.sh`'s `MACHINE_EXEMPTIONS` with its reason. A wheel that claims no
  platform can still carry a native library for another one: on aarch64 the one
  entry is debugpy's x86_64 attach helper.
- **Search paths.** Where a GPU backend ships, its solver carries `DT_RPATH`
  `$ORIGIN/../../bin`, which the loader consults before `LD_LIBRARY_PATH`, so a
  stale backend library elsewhere cannot be loaded in its place, and every
  library in `bin/` that loads another one there carries `$ORIGIN`. The smoke
  test proves it with a broken library on `LD_LIBRARY_PATH` and the loader's own
  report.
- **Floors.** The highest `GLIBC_` and `GLIBCXX_` versions any binary needs are
  stamped into the launcher, and `PPF_LINUX_MAX_GLIBC` and
  `PPF_LINUX_MAX_GLIBCXX`, when set, fail a payload that needs more and name the
  file that raises it. A file the program does not load in normal use can be
  left out of the floors by name, with its reason, in `bundle.sh`'s
  `FLOOR_EXEMPTIONS`; it is printed as left out, and an entry matching no file
  fails the audit. The one entry is debugpy's attach helper, built against
  glibc 2.34 and loaded only to attach a debugger to a running process by PID.
- **No build-tree path.** No file names the directory it was built in, apart
  from one named stamp in the Python extension. The search is for that path as
  text, so `bundle.sh` refuses a build tree of fewer than two path components:
  a path like `/src` occurs inside unrelated source paths. `build.sh` also
  remaps cargo's home to `/cargo` in the Rust binaries, which otherwise record
  every dependency crate's absolute source path, and refuses a binary that
  still names it.
- **No broken symlink.**
- **Smoke test.** The interpreter imports the frontend and JupyterLab under a
  poisoned environment; every setting the launcher passes to Jupyter exists;
  every shipped solver and server, and ffmpeg, start; the frontend resolves its
  data and cache paths inside the distribution and writes nothing under an empty
  HOME; the launcher answers through symlinks whose names carry a space; and
  `python` mode runs the distribution's own interpreter, selects `target/cpu`
  only from inside it, and defaults to it where no GPU build ships.

## Verifying a distribution

`scripts/verify-distribution.sh` checks a distribution on the machine that runs
it, which `bundle.sh` cannot, and writes a report:

```bash
build-linux-native/scripts/verify-distribution.sh DIST REPORT_DIR \
  [--driver-only] [--cuda-scenes all|none|"a b"] [--cpu-scenes all|none|"a b"]
```

It reads the backends the launcher was stamped with (`PPF_DIST_BACKENDS`) and
checks the layout, the launcher, each shipped solver and server, where the
loader takes the CUDA backend from when it ships, the interpreter through
`ppf-contact-solver python`, and then runs example notebooks through
`examples/run_suite.py --fast-check` on each backend, judged by the frames they
produce. `--cuda-scenes` against a distribution without the CUDA backend fails
rather than skipping. While a CUDA
scene runs it samples the shared libraries the solver maps, and requires that
they come from the distribution or the system library directories, that no CUDA
toolkit library is among them, and that the driver's `libcuda` is, which is what
shows a solve reached the GPU. HOME is an empty directory for the run, and it
fails if anything appears there or anywhere in the distribution outside
`local/`, `cache/` and `examples/`. `--driver-only` also requires the machine to
have a driver and no toolkit.

Every check runs regardless of the previous one, so the report names every
failure. Each check's output is `REPORT_DIR/<check>.log`.

## Releasing

`.github/workflows/release.yml` is dispatched by a person. It runs:

| Job | Where | What |
| --- | --- | --- |
| `version` | a short job | the stamp both archives of one release carry |
| `build-linux` | an AlmaLinux 8.10 container with gcc-toolset-13 | `warmup.sh`, `build.sh`, `bundle.sh` with `PPF_LINUX_MAX_GLIBC=2.28` and `PPF_LINUX_MAX_GLIBCXX=3.4.25`, then the archive |
| `verify-floor` | a fresh AlmaLinux 8.10 container with no compiler | `verify-distribution.sh --cpu-scenes <input>` |
| `build-linux-arm64` | the same container on a GitHub-hosted `ubuntu-24.04-arm` runner | the same three scripts, which build the CPU backend alone on aarch64 |
| `verify-floor-arm64` | a fresh aarch64 AlmaLinux 8.10 container with no compiler | `verify-distribution.sh --cpu-scenes <input>` |
| `verify-linux` | an EC2 GPU instance from a plain Ubuntu 24.04 image with only an NVIDIA server driver: the lowest branch at or above 570 that has a prebuilt module for the image's kernel | `verify-distribution.sh --driver-only --cuda-scenes <input>`, detached and polled |
| `release` | a short job | tag and GitHub release, unless a dry run |

gcc-toolset links the libstdc++ symbols newer than the system's statically,
which is what keeps a C++20 build at the system's `GLIBCXX_3.4.25`. The GPU
verification runs detached because an Instance Connect tunnel closes at 60
minutes, and the wait is split into windows with a credential refresh between
them. Both verification jobs upload their report directories.

## A build host with no egress

A file from `scripts/downloads.txt` already in `downloads/` with a matching
checksum is neither probed nor fetched. The ffmpeg sources are used from
`downloads/src/` (`ffmpeg-<tag>`, `x264`, the zlib tarball) when present, and
`PPF_LINUX_WHEELHOUSE` covers the Python packages. A run that needs nothing from
the network says so and opens no connection for its downloads.
`scripts/check-downloads.sh [KEY...]` probes only the URLs named, or all of them.

## Updating tool versions

Change the entry in `scripts/downloads.txt`, with its `SHA256_*`, and remove the
provisioned directory so `warmup.sh` installs the new one. The ffmpeg, x264 and
zlib pins live in `build-win-native/scripts/downloads.txt`, which the Windows
build and `.github/workflows/scripts/make-slim-ffmpeg.sh` share.

## Script reference

| File | Purpose |
| --- | --- |
| `warmup.sh` | Provisions the toolchain, both Python environments, patchelf, nasm (x86_64), ffmpeg, and the wheels built from source (aarch64) |
| `build.sh` | Builds the backends `scripts/platform.sh` names and writes `start.sh` |
| `bundle.sh` | Packages, audits and smoke-tests the distribution |
| `config.sh` | `PORT`, `PPF_LINUX_BACKENDS`, `PPF_CTS_VENV`, `PPF_CTS_PYTHON`, `PPF_LINUX_WHEELHOUSE` |
| `start.sh` | Generated by `build.sh`; JupyterLab over the source tree |
| `scripts/downloads.txt` | Single source of truth for download URLs, checksums and source pins |
| `scripts/platform.sh` | The host architecture, its backends and its source-built wheels |
| `scripts/source-wheel.sh` | Builds one package's wheel from pinned upstream source and checks it |
| `scripts/wheel-build-requirements.txt` | The hash-pinned build tools `source-wheel.sh` uses |
| `scripts/load-downloads.sh` | Parses a manifest into the caller's environment |
| `scripts/check-downloads.sh` | Probes the manifest's URLs |
| `scripts/backend-path.sh` | Resolves a `NEEDED` library through a binary's own search path |
| `scripts/elf-audit.py` | The ELF dependency audit and the floor report |
| `scripts/verify-distribution.sh` | Verifies an unpacked distribution where it runs |

Environment read by `bundle.sh`: `PPF_LINUX_DIST_NAME`, `PPF_LINUX_MAX_GLIBC`,
`PPF_LINUX_MAX_GLIBCXX`.
