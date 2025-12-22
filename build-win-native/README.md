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

## Notes

- GPU driver still requires system installation for runtime execution
- Compilation works without GPU driver
- All tools are self-contained in the `build-win-native` directory
