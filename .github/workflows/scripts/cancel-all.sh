#!/bin/bash
# File: cancel-all.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

gh run list --limit 1024 --json status,workflowName,databaseId \
	-q '.[] | select(.status=="in_progress" or .status=="queued") | "\(.databaseId)\t\(.workflowName)"' |
	while IFS=$'\t' read -r id workflow_name; do
		echo "Canceling '$workflow_name' (#$id)"
		gh run cancel "$id"
	done
