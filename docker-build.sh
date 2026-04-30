#!/bin/bash

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
IMAGE_NAME="ppf-contact-solver-compiled-${BRANCH_NAME}:latest"

docker build --no-cache -t "${IMAGE_NAME}" .

echo ""
echo "=== Running headless.py on built image ==="
docker run --rm --gpus all "${IMAGE_NAME}" \
  /bin/sh -c "cd /root/ppf-contact-solver && PYTHONPATH=/root/ppf-contact-solver /root/.local/share/ppf-cts/venv/bin/python examples/headless.py"

echo ""
echo "=== Running fast_check on built image ==="
docker run --rm --gpus all "${IMAGE_NAME}" \
  /bin/sh -c "cd /root/ppf-contact-solver && /root/.local/share/ppf-cts/venv/bin/python warmup.py fast_check && /root/.local/share/ppf-cts/venv/bin/python warmup.py clear_all"
