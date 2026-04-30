# Windows Build Testing Guide

This guide explains how to manually test the Windows build process using two AWS instances.

## Overview

The testing process uses two Windows instances:

| Instance | Access | Purpose | Environment |
|----------|--------|---------|-------------|
| **Build** | `ssh win-build` | Compile and bundle | NVIDIA driver only (all tools installed locally) |
| **Clean** | `ssh win-clean` | Verify bundle | NVIDIA driver only (no build tools) |

**Important**: Both instances only need NVIDIA driver installed. All build tools are installed locally by warmup.bat.

## Prerequisites

Both instances should have:
- Windows Server 2025
- NVIDIA GPU (e.g., L4 on G6e instances)
- NVIDIA driver installed
- SSH access configured

**No system-wide build tools required** - warmup.bat installs everything locally:
- 7-Zip (portable)
- MinGit (portable)
- CUDA Toolkit (extracted locally)
- MSVC compiler (portable)
- Rust (local installation)
- Python (embedded)

## Phase 1: Building on the Build Instance

### 1.1 Connect to the Build Instance

```bash
ssh win-build
```

### 1.2 Transfer the Source Code

From your local machine, create and transfer the source archive:

```bash
# On your local machine (includes uncommitted changes)
cd /home/ubuntu/dev
zip -rq /tmp/repo.zip . -x '.git/*' -x 'target/*' -x 'build-win-native/downloads/*' \
    -x 'build-win-native/python/*' -x 'build-win-native/rust/*' -x 'build-win-native/msvc/*' \
    -x 'build-win-native/cuda/*' -x 'build-win-native/7zip/*' -x 'build-win-native/mingit/*' \
    -x 'build-win-native/dist/*' -x 'src/cpp/build/*'

# Transfer to the build instance
scp /tmp/repo.zip win-build:C:/source.zip
```

On the Windows instance:

```powershell
# Remove old source if exists
if (Test-Path 'C:\ppf-contact-solver') { Remove-Item -Recurse -Force 'C:\ppf-contact-solver' }

# Extract source
New-Item -ItemType Directory -Path 'C:\ppf-contact-solver' -Force
Expand-Archive -Path 'C:\source.zip' -DestinationPath 'C:\ppf-contact-solver' -Force
Remove-Item 'C:\source.zip'
```

### 1.3 Run Warmup (First Time Setup)

The warmup script downloads and installs locally (no admin required):
- 7-Zip (portable, for CUDA extraction)
- MinGit (portable)
- CUDA Toolkit 12.8.1 (extracted locally via 7-Zip)
- MSVC compiler (VS 2022, portable via [portable-msvc](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977))
- Rust (local installation)
- Embedded Python 3.11.9 with pip and packages

**Note**: Downloads use `curl.exe` (built into Windows 10+) for faster transfers.

```cmd
cd C:\ppf-contact-solver\build-win-native
warmup.bat /nopause
```

This creates:
- `build-win-native\7zip\` - Portable 7-Zip
- `build-win-native\cuda\` - CUDA Toolkit (extracted)
- `build-win-native\msvc\` - Portable MSVC compiler
- `build-win-native\python\` - Embedded Python environment
- `build-win-native\rust\` - Local Rust installation
- `build-win-native\mingit\` - MinGit for Git operations
- `build-win-native\downloads\` - Downloaded installers (cached)

**Note**: First run takes 20-30 minutes to download ~10GB of tools. Subsequent runs automatically skip already-installed components (checks for nvcc.exe, cargo.exe, python.exe, etc.).

### 1.4 Build the Solver

```cmd
cd C:\ppf-contact-solver\build-win-native
build.bat /nopause
```

This performs:
1. Downloads Eigen 3.4.0 (if not present)
2. Sets up MSVC environment (from portable installation)
3. Builds CUDA library (`libsimbackend_cuda.dll`) using nvcc directly
4. Builds Rust binary (`ppf-contact-solver.exe`) using Cargo
5. Creates launcher scripts

Build outputs:
- `src\cpp\build\lib\libsimbackend_cuda.dll` - CUDA backend
- `target\release\ppf-contact-solver.exe` - Main solver binary

### 1.5 Create the Bundle

```cmd
cd C:\ppf-contact-solver\build-win-native
bundle.bat /nopause
```

This creates a self-contained distribution in `build-win-native\dist\` containing:
- `target\release\ppf-contact-solver.exe` - Solver binary
- `bin\` - DLLs (libsimbackend_cuda.dll, cudart64_12.dll)
- `python\` - Embedded Python environment
- `mingit\` - MinGit for repository cloning
- `frontend\` - Python frontend module
- `src\` - Source Python modules
- `examples\` - Jupyter notebooks
- Launcher scripts (start.bat, headless.bat, etc.)

### 1.6 Test the Bundle on Build Instance

Run a quick headless test:

```cmd
cd C:\ppf-contact-solver\build-win-native\dist
headless.bat /nopause
```

Run all example notebooks:

```cmd
cd C:\ppf-contact-solver\build-win-native\dist
fast-check-all.bat /nopause
```

### 1.7 Create the Release Archive

```powershell
$VERSION = "test-$(Get-Date -Format 'yyyy-MM-dd-HH-mm')"
Compress-Archive -Path 'C:\ppf-contact-solver\build-win-native\dist\*' -DestinationPath "C:\ppf-contact-solver-${VERSION}-win64.zip" -Force
```

### 1.8 Clean Up Test Artifacts

Before archiving, clean up any test artifacts:

```powershell
Remove-Item -Recurse -Force 'C:\ppf-contact-solver\build-win-native\dist\local' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\ppf-contact-solver\build-win-native\dist\cache' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\ppf-contact-solver\build-win-native\dist\export' -ErrorAction SilentlyContinue
Get-ChildItem -Path 'C:\ppf-contact-solver\build-win-native\dist' -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
```

## Phase 2: Testing on the Clean Instance

The clean instance verifies that the bundle has no external DLL dependencies and works on a fresh Windows installation with only NVIDIA drivers.

### 2.1 Transfer the Bundle

From your local machine:

```bash
# Download from build instance
scp win-build:'C:/ppf-contact-solver-*-win64.zip' /tmp/bundle.zip

# Upload to clean instance
scp /tmp/bundle.zip win-clean:C:/bundle.zip
```

Or directly between instances if network allows.

### 2.2 Connect to the Clean Instance

```bash
ssh win-clean
```

### 2.3 Extract and Test the Bundle

```powershell
# Clean up any previous test
if (Test-Path 'C:\bundle') { Remove-Item -Recurse -Force 'C:\bundle' }

# Extract
New-Item -ItemType Directory -Path 'C:\bundle' -Force
Expand-Archive -Path 'C:\bundle.zip' -DestinationPath 'C:\bundle' -Force
```

### 2.4 Run Headless Test

```cmd
cd C:\bundle
headless.bat /nopause
```

If this succeeds, the bundle has no missing DLL dependencies.

### 2.5 Run All Example Tests

```cmd
cd C:\bundle
fast-check-all.bat /nopause
```

This converts each example notebook to a Python script and runs it.

### 2.6 Clean Up the Clean Instance

**Important**: Always clean up after testing to keep the instance clean.

```powershell
Remove-Item -Recurse -Force 'C:\bundle' -ErrorAction SilentlyContinue
Remove-Item -Force 'C:\bundle.zip' -ErrorAction SilentlyContinue
```

## Troubleshooting

### Build Errors

**CUDA not found**:
```
ERROR: nvcc not found
```
Ensure warmup.bat completed successfully. The CUDA toolkit should be extracted to `build-win-native\cuda\`.

**MSVC not found**:
```
ERROR: No MSVC found
```
Re-run warmup.bat. The portable MSVC should be at `build-win-native\msvc\`.

**Rust not found**:
```
ERROR: Rust not found
```
Re-run warmup.bat to install Rust locally.

### Bundle Test Errors

**DLL not found on clean instance**:
```
The code execution cannot proceed because xyz.dll was not found
```
This means the bundle is missing a required DLL. Check:
1. Is the DLL in `bin\` directory?
2. Is it a CUDA runtime DLL that should be bundled?
3. Is it a system DLL that should be available?

**Python import errors**:
```
ModuleNotFoundError: No module named 'xyz'
```
The embedded Python is missing a required package. Add it to warmup.bat's package list and rebuild.

### Path Length Issues

Windows has a 260-character path limit by default. The build avoids hitting it
by running `shorten_webpack_chunks.py` from `bundle.bat`, which rewrites the
long webpack chunk filenames in the embedded JupyterLab tree. If a path-length
error appears during bundle, check that the shrink pass ran and that no
newly added asset uses a long generated name; do not work around it by
enabling `LongPathsEnabled`, which is a non-reversible registry change to the
build host.

## Clean Build (Full Rebuild)

To perform a clean build from scratch:

```cmd
cd C:\ppf-contact-solver\build-win-native
clean-build.bat /nopause
build.bat /nopause
bundle.bat /nopause
```

This removes:
- C++/CUDA build output
- Rust target directory
- Previous distribution folder

## Script Reference

| Script | Purpose |
|--------|---------|
| `warmup.bat` | First-time environment setup (downloads all tools locally) |
| `build.bat` | Compile CUDA library (nvcc) and Rust binary |
| `bundle.bat` | Create distribution package |
| `clean-build.bat` | Remove all build artifacts |
| `clean-env.bat` | Remove all local tools (Python, Rust, MSVC, CUDA, etc.) |
| `start.bat` | Launch JupyterLab (development) |
| `headless.bat` | Run headless example |
| `fast-check-all.bat` | Run all example notebooks as tests |
| `clear-cache.bat` | Clear runtime caches |
| `clear-all.bat` | Full cleanup including session data |

## CI/CD Reference

The automated GitHub Actions workflow (`.github/workflows/release-win.yml`) performs these same steps:

1. Launches a fresh Windows EC2 instance (NVIDIA driver only)
2. Transfers source and runs warmup/build/bundle (all tools installed locally)
3. Tests with headless.bat and fast-check-all.bat
4. Terminates build instance
5. Launches a clean verification instance (NVIDIA driver only)
6. Transfers bundle and runs tests
7. Creates GitHub release (if not dry run)
8. Terminates all instances and cleans up
