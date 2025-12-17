#!/bin/bash
# File: cancel-all.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

gh run list --json status,workflowName,databaseId --jq '.[] | select(.status=="in_progress" or .status=="queued")' |
	while read -r run; do
		id=$(echo "$run" | jq -r '.databaseId')
		workflow_name=$(echo "$run" | jq -r '.workflowName')

		echo "Canceling running workflow '$workflow_name' (#$id)"
		gh run cancel $id
	done
