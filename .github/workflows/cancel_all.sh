#!/bin/bash

gh run list --json status,workflowName,databaseId --jq '.[] | select(.status=="in_progress")' |
	while read -r run; do
		id=$(echo "$run" | jq -r '.databaseId')
		workflow_name=$(echo "$run" | jq -r '.workflowName')

		echo "Canceling running workflow '$workflow_name' (#$id)"
		gh run cancel $id
	done
