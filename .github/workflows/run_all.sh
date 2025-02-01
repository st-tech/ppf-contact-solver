#!/bin/bash

if [ -z "$1" ]; then
	echo "Error: Missing branch parameter."
	exit 1
fi

if [ -z "$2" ]; then
	echo "Error: Missing runner parameter."
	exit 1
fi

branch="$1"
runner="$2"

WORKFLOW_PATTERN="example_*"
WORKFLOW_FILES=$(ls $WORKFLOW_PATTERN*.yml 2>/dev/null)

if [ -z "$WORKFLOW_FILES" ]; then
	echo "No workflow files found."
	exit 1
fi

for WORKFLOW_FILE in $WORKFLOW_FILES; do
	echo "Triggering GitHub Action workflow: $WORKFLOW_FILE with runner=$runner"
	gh workflow run "$WORKFLOW_FILE" -f runner="$runner" --ref $branch
done
