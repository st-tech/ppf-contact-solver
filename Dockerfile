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

RUN /root/.cargo/bin/cargo build --release

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
  libx11-6 \
  libgomp1 \
  libgl1 \
  libosmesa6 \
  libxrender1 \
  ffmpeg && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/*.bin /tmp/* /var/tmp/*

# Copy only necessary files from builder
COPY --from=builder /root/${PROJ_NAME}/target/release/ppf-contact-solver /root/${PROJ_NAME}/target/release/ppf-contact-solver
COPY --from=builder /root/${PROJ_NAME}/target/release/build/ppf-contact-solver-*/out/lib/*.so /usr/local/lib/
COPY --from=builder /root/${PROJ_NAME}/*.py /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/Cargo.toml /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/LICENSE /root/${PROJ_NAME}/
COPY --from=builder /root/${PROJ_NAME}/examples /root/${PROJ_NAME}/examples
COPY --from=builder /root/${PROJ_NAME}/blender_addon /root/${PROJ_NAME}/blender_addon
COPY --from=builder /root/${PROJ_NAME}/frontend /root/${PROJ_NAME}/frontend
COPY --from=builder /root/${PROJ_NAME}/src /root/${PROJ_NAME}/src
COPY --from=builder /root/${PROJ_NAME}/.git/branch_name.txt /root/${PROJ_NAME}/.git/branch_name.txt

# Copy virtual environment from base-image (which has the venv created)
COPY --from=base-image /root/.local/share/ppf-cts/venv /root/.local/share/ppf-cts/venv

# Clean up venv cache files and unnecessary content to reduce image size
RUN find /root/.local/share/ppf-cts/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
  find /root/.local/share/ppf-cts/venv -type f -name "*.pyo" -delete 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/pandas/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/matplotlib/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/setuptools/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/jupyterlab/tests 2>/dev/null || true && \
  rm -rf /root/.local/share/ppf-cts/venv/lib/python*/site-packages/sklearn/tests 2>/dev/null || true && \
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
