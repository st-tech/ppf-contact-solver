#!/bin/bash

ARG=$1
if [ "$ARG" = "create" ]; then

  VAST_API_KEY=$2

  echo "helper: provision..."
  bash .github/workflows/vast/provision.sh $VAST_API_KEY

  echo "helper: transfer..."
  bash /tmp/vast-ci/rsync-command.sh

  echo "helper: build..."
  bash .github/workflows/vast/run.sh build

  echo "helper: convert..."
  bash .github/workflows/vast/run.sh convert

elif [ "$ARG" = "run" ]; then

  SCRIPT_NAME=$2
  NAME=$3

  echo "helper: run..."
  bash .github/workflows/vast/run.sh run $SCRIPT_NAME $NAME

elif [ "$ARG" = "collect" ]; then

  echo "helper: collect..."

  NAME=$2
  bash /tmp/vast-ci/collect.sh $NAME

elif [ "$ARG" = "delete" ]; then

  echo "helper: delete..."
  bash /tmp/vast-ci/delete-instance.sh

fi
