# Windows Build

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with driver installed (for runtime execution)

**No admin privileges required** - all build tools are installed locally.

## Build Steps

### 1. Setup Environment

```batch
warmup.bat
```

This downloads and installs locally (no admin required):
- 7-Zip (portable, for CUDA extraction)
- MinGit (portable)
- CUDA Toolkit 12.8 (extracted locally)
- MSVC compiler (portable, via [portable-msvc](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977))
- Rust (local installation)
- Embedded Python 3.11 with required packages

First run takes 20-30 minutes to download everything (~10GB total).

Before any download starts, `warmup.bat` runs `scripts\check-downloads.bat`
to verify every URL in `scripts\downloads.txt` (the single source of truth
for tool URLs) is reachable. If a pointer has rotted upstream, warmup
aborts with the failing URL listed; fix the entry in
`scripts\downloads.txt` and re-run.

### 2. Build

```batch
build.bat
```

This builds:
- CUDA library (`libsimbackend_cuda.dll`)
- Rust executable (`ppf-contact-solver.exe`)
- Launcher scripts (`start.bat`, `start-jupyterlab.pyw`)

## Running

```batch
start.bat
```

JupyterLab opens at http://localhost:8080

### 3. Bundle (Optional)

```batch
bundle.bat
```

Creates a self-contained distribution in `dist/` containing:
- Solver binaries and CUDA runtime
- Embedded Python environment
- MinGit for repository cloning
- Example notebooks
- Launcher scripts

The bundle can run on any Windows machine with only NVIDIA driver installed.

## Clean

```batch
clean-build.bat   # Remove build artifacts only
clean-env.bat     # Remove all local tools (Python, Rust, MSVC, CUDA, etc.)
clear-cache.bat   # Clear runtime caches
clear-all.bat     # Full cleanup including session data
```

When to run which:

| Goal                                 | Sequence                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| Rebuild from cached env              | `clean-build.bat` -> `build.bat`                                                      |
| Fresh tools (re-download everything) | `clean-env.bat` -> `warmup.bat` -> `build.bat`                                        |
| Clear runtime caches only            | `clear-cache.bat` (use `clear-all.bat` instead to also drop session data)             |
| Total reset                          | `clean-env.bat` + `clean-build.bat` + `clear-all.bat` -> `warmup.bat` -> `build.bat`  |

`clean-env.bat` first kills any process whose executable lives under this
directory (orphan `git.exe`, `python.exe`, etc.) so a stuck background
task does not silently lock files and leave them behind. If a `rmdir`
still fails, the script exits non-zero and prints the offending path.

## Updating tool versions

All external download URLs and filenames are centralized in
`scripts\downloads.txt`. To bump a tool (for example CUDA 12.8 -> 12.9),
edit the matching `URL_*` and `FILE_*` lines together, then verify the
new pointer:

```batch
scripts\check-downloads.bat
```

`warmup.bat` runs the same check on every invocation, so a bad pointer
fails fast at the manifest layer rather than partway through a multi-GB
download.

## Script Reference

| Script | Purpose |
|--------|---------|
| `warmup.bat` | First-time environment setup (downloads all tools locally) |
| `build.bat` | Compile CUDA library and Rust binary |
| `bundle.bat` | Create distribution package |
| `start.bat` | Launch JupyterLab |
| `fast-check-all.bat` | Run all example notebooks as tests |
| `clean-build.bat` | Remove build artifacts only |
| `clean-env.bat` | Remove all local tools (Python, Rust, MSVC, CUDA, etc.) |
| `clear-cache.bat` | Clear runtime caches |
| `clear-all.bat` | Full cleanup including session data |
| `git-pull.bat` | Update source via git pull |
| `scripts\downloads.txt` | Single source of truth for tool URLs and filenames |
| `scripts\check-downloads.bat` | Verify every URL in `scripts\downloads.txt` is reachable |

## Notes

- GPU driver still requires system installation for runtime execution
- Compilation works without GPU driver
- All tools are self-contained in the `build-win-native` directory
