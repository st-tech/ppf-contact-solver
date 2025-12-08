# Windows Build (Experimental)

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads) installed

## Build Steps

### 1. Setup Environment

Run as Administrator (required for Chocolatey installation):

```batch
warmup.bat
```

This installs:
- Chocolatey package manager
- Git
- Visual Studio 2022 Build Tools
- Rust (local installation)
- Embedded Python 3.11 with required packages

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

## Clean

```batch
clean-build.bat
clean-env.bat
```