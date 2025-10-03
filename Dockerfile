# File: Dockerfile
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

ARG IMAGE
FROM $IMAGE

ARG MODE
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute
ENV LANG=en_US.UTF-8
ENV PROJ_NAME=ppf-contact-solver
ENV BUILT_MODE=$MODE

COPY . /root/${PROJ_NAME}
WORKDIR /root/${PROJ_NAME}

RUN echo "building in ${MODE} mode"

RUN if [ "$MODE" = "compiled" ]; then \
  /root/.cargo/bin/cargo build --release; \
  elif [ "$MODE" = "base" ]; then \
  apt-get update && \
  apt-get install -y python3 python3-venv && \
  python3 warmup.py && \
  /root/.cargo/bin/cargo build && \
  rm -rf /root/${PROJ_NAME}; \
  else \
  echo "unknown build mode ${BUILT_MODE}"; \
  exit 1; \
  fi

WORKDIR /root
RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/sh", "-c", "\
  if [ \"$BUILT_MODE\" = \"compiled\" ]; then \
  cd /root/${PROJ_NAME} && python3 warmup.py jupyter; \
  elif [ \"$BUILT_MODE\" = \"base\" ]; then \
  bash; \
  fi\
  "]
