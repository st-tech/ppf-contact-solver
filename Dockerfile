# File: Dockerfile
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG BUILD_MODE=base
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute
ENV LANG=en_US.UTF-8
ENV PROJ_NAME=ppf-contact-solver
ENV BUILT_MODE=${BUILD_MODE}

COPY . /root/${PROJ_NAME}
WORKDIR /root/${PROJ_NAME}

RUN apt update
RUN apt install -y git python3 curl
RUN python3 warmup.py

RUN if [ "$BUILD_MODE" = "compile" ]; then \
        /root/.cargo/bin/cargo build --release; \
    else \
        cd /root && rm -rf /root/${PROJ_NAME}; \
    fi

WORKDIR /root
RUN rm -rf /var/lib/apt/lists/*

CMD if [ "$BUILT_MODE" = "compile" ]; then \
        cd /root/${PROJ_NAME} && python3 warmup.py jupyter; \
    else \
        bash; \
    fi