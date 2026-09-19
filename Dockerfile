# File: Dockerfile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

# Base image stage - always from NVIDIA CUDA
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS base-image
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute
ENV LANG=en_US.UTF-8
ENV PROJ_NAME=ppf-contact-solver

COPY . /root/${PROJ_NAME}
WORKDIR /root/${PROJ_NAME}

RUN apt-get update && \
  apt-get install -y python3 python3-venv && \
  python3 warmup.py --skip-confirmation && \
  /root/.cargo/bin/cargo build && \
  rm -rf /root/${PROJ_NAME}

WORKDIR /root
RUN rm -rf /var/lib/apt/lists/*

# Builder stage for compiled mode - builds from base-image
FROM base-image AS builder
ENV PROJ_NAME=ppf-contact-solver

COPY . /root/${PROJ_NAME}
WORKDIR /root/${PROJ_NAME}

# Capture git branch name and save to .git/branch_name.txt
RUN mkdir -p .git && \
  (git branch --show-current > .git/branch_name.txt 2>/dev/null || echo "unknown" > .git/branch_name.txt)

# Build slim ffmpeg (PNG to MP4 only, ~4MB vs ~28MB for full package). The
# script installs nothing itself, so the tools it builds with are installed here.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential nasm pkg-config curl git ca-certificates && \
  .github/workflows/scripts/make-slim-ffmpeg.sh

RUN /root/.cargo/bin/cargo build --release

# THE CPU BACKEND BESIDE THE CUDA ONE, because an image that ships one backend
# is an image that runs on one kind of machine. This one's runtime stage is
# plain ubuntu:24.04 and starts without `--gpus`, so a reader trying the project
# on a laptop gets a container that comes up and then fails at its first solve
# with no usable device. With this build present, `frontend`'s automatic rule
# answers with the CPU backend and prints why, and the Blender add-on's Compute
# Device row has both answers to offer over a Docker connection.
#
# IT IS ALSO WHAT EVERY OTHER DISTRIBUTION OF THIS PROJECT SHIPS.
# `build-linux-native/scripts/platform.sh` refuses a backend set that leaves out
# `cpu`, in those words: "Every distribution ships the CPU backend, which is
# what runs on a machine with no supported GPU." This image is a distribution by
# the same argument.
#
# CARGO_TARGET_DIR IS THE WHOLE ARRANGEMENT, not a convenience.
# `crates/ppf-cts-solver/build.rs` refuses to put a second backend into a
# directory that already holds another, because every backend links the same
# executable name, so two backends means two directories and this variable is
# how the second one is named.
RUN CARGO_TARGET_DIR=target/cpu /root/.cargo/bin/cargo build --release --features cpu

# Runtime stage for compiled mode (minimal Ubuntu)
FROM ubuntu:24.04 AS runtime-image
ENV LANG=en_US.UTF-8
ENV PROJ_NAME=ppf-contact-solver
ENV BUILT_MODE=compiled

# Install Python runtime and required dependencies for notebooks
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  ca-certificates \
  git \
  libgomp1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/*.bin /tmp/* /var/tmp/*

# Copy only necessary files from builder
COPY --from=builder /root/${PROJ_NAME}/target/release/ppf-contact-solver /root/${PROJ_NAME}/target/release/ppf-contact-solver
COPY --from=builder /root/${PROJ_NAME}/target/release/ppf-cts-server /root/${PROJ_NAME}/target/release/ppf-cts-server
COPY --from=builder /root/${PROJ_NAME}/target/release/build/ppf-cts-solver-*/out/lib/*.so /usr/local/lib/
# The PyO3 cdylib (ppf-cts-py is a workspace default-member, so the release
# build above produces it). frontend/__init__.py loads it directly by
# absolute path from target/release and registers it as _ppf_cts_py, so the
# runtime image needs it at this exact path. No glob: a missing cdylib must
# fail the build loudly (there is no wheel fallback).
COPY --from=builder /root/${PROJ_NAME}/target/release/lib_ppf_cts_py.so /root/${PROJ_NAME}/target/release/lib_ppf_cts_py.so
# The backend marker `crates/ppf-cts-solver/build.rs` writes beside the
# artifacts, naming the backend this directory holds. It is what tells the
# frontend a build is here at all: `frontend/_backends_.py` reads the marker in
# each `target/<profile>` to decide which backend a run uses, and a directory
# without one holds nothing as far as that rule is concerned, so every run is
# refused with "no solver is built here (built: nothing)" even though the
# binaries above are in place. The executable name cannot answer instead,
# because every backend links the same one. No glob, for the reason the cdylib
# above carries one: a missing marker must fail this build rather than the
# image's first run.
COPY --from=builder /root/${PROJ_NAME}/target/release/.ppf-backend /root/${PROJ_NAME}/target/release/.ppf-backend
# The CPU build, the same four files out of its own target directory. It needs
# no entry in /usr/local/lib beside them: the CPU backend is the Rust driver and
# links into the binaries, so unlike the CUDA one it loads no backend library,
# and its solver names only libgcc_s, libm and libc, all of which this stage
# already has.
#
# THE MARKER MATTERS TWICE OVER HERE. `frontend._backends_` reads it to decide
# which backend a run uses, which is how a machine with no usable GPU lands on
# this build, and `blender_addon/core/remote_builds.py` reads it over the
# connection to decide which Compute Device a Docker connection can offer. A
# directory without one holds nothing as far as either rule is concerned.
COPY --from=builder /root/${PROJ_NAME}/target/cpu/release/ppf-contact-solver /root/${PROJ_NAME}/target/cpu/release/ppf-contact-solver
COPY --from=builder /root/${PROJ_NAME}/target/cpu/release/ppf-cts-server /root/${PROJ_NAME}/target/cpu/release/ppf-cts-server
COPY --from=builder /root/${PROJ_NAME}/target/cpu/release/lib_ppf_cts_py.so /root/${PROJ_NAME}/target/cpu/release/lib_ppf_cts_py.so
COPY --from=builder /root/${PROJ_NAME}/target/cpu/release/.ppf-backend /root/${PROJ_NAME}/target/cpu/release/.ppf-backend
COPY --from=builder /root/${PROJ_NAME}/*.py /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/Cargo.toml /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/LICENSE /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/examples /root/${PROJ_NAME}/examples
COPY --from=builder /root/${PROJ_NAME}/blender_addon /root/${PROJ_NAME}/blender_addon
COPY --from=builder /root/${PROJ_NAME}/frontend /root/${PROJ_NAME}/frontend
# frontend/_session_inspect_.py:_harvest_log_docstrings walks these
# three source roots to discover log channel names from `// Name:`
# docstrings; without them session.get.log.names() returns [] and
# notebooks like drape.ipynb fail at `assert "time-per-frame" in logs`.
# They are the same three ppf-cts-server/src/main.rs probes, and the list
# has to be read from there rather than guessed: a root copied in here that
# the server does not probe contributes nothing, and one the server probes
# that is missing here costs every channel declared under it.
#
# The neutral driver in ppf-cts-solver/src/driver declares most channels now,
# so the first root carries them. The CUDA root is still copied because the
# backend declares its own, and it is under ppf-cts-compute because that is
# where every backend lives.
COPY --from=builder /root/${PROJ_NAME}/crates/ppf-cts-solver/src /root/${PROJ_NAME}/crates/ppf-cts-solver/src
COPY --from=builder /root/${PROJ_NAME}/crates/ppf-cts-compute/cuda /root/${PROJ_NAME}/crates/ppf-cts-compute/cuda
COPY --from=builder /root/${PROJ_NAME}/crates/ppf-cts-core/src /root/${PROJ_NAME}/crates/ppf-cts-core/src
COPY --from=builder /root/${PROJ_NAME}/.git/branch_name.txt /root/${PROJ_NAME}/.git/branch_name.txt
COPY --from=builder /root/${PROJ_NAME}/.github/workflows/scripts/examples.txt /root/${PROJ_NAME}/examples.txt
COPY --from=builder /root/${PROJ_NAME}/bin/ffmpeg /root/${PROJ_NAME}/bin/ffmpeg

# Copy virtual environment from base-image (which has the venv created).
# The venv provides the pure-python deps (cbor2, psutil, jupyter); the PyO3
# extension is NOT installed here. frontend/__init__.py loads the cdylib
# directly from target/release/lib_ppf_cts_py.so (copied above).
COPY --from=base-image /root/.local/share/ppf-cts/venv /root/.local/share/ppf-cts/venv

# Clean up venv cache files and unnecessary content to reduce image size
RUN find /root/.local/share/ppf-cts/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv -type f -name "*.pyo" -delete 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/matplotlib/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/setuptools/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/jupyterlab/tests 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv/lib/python*/site-packages -type f -name "*.so" -exec strip --strip-debug {} + 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/pip/_vendor/distlib/*.exe 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/share/jupyter/lab/staging 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/share/locale 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/share/man 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv -type d -name "*.dist-info" -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER "$1"/WHEEL 2>/dev/null || true' _ {} \;

# Update library cache
RUN ldconfig

WORKDIR /root

# Final stages with proper CMD
FROM base-image AS base-final
CMD ["/bin/bash"]

FROM runtime-image AS runtime-final
CMD ["/bin/sh", "-c", "cd /root/${PROJ_NAME} && python3 warmup.py jupyter"]
