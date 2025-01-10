# File: provision.sh
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

# Working directory
WORKDIR=/tmp/vast-ci

# CUDA tester file path
CUDA_TESTER_PATH=$(pwd)/.github/workflows/vast/cuda-tester.cu

# your local public ssh key
SSH_PUB_KEY=$WORKDIR/id_ed25519.pub

# disk space 32GB
DISK_SPACE=32

# GPU
GPU_NAME=RTX_4090

# Image
VAST_IMAGE="ghcr.io/st-tech/ppf-contact-solver-base:latest"

# Retry interval
RETRY_INTERVAL=10

# max retries
MAX_RETRIES=30

# check if the CUDA tester file exists
if [ ! -f "$CUDA_TESTER_PATH" ]; then
  echo "Error: CUDA tester file not found: $CUDA_TESTER_PATH"
  exit 1
fi

if [ -d "$WORKDIR" ]; then
  echo "$WORKDIR already exists."
  cd $WORKDIR
  source $WORKDIR/venv/bin/activate
else
  echo "Setting up..."

  # Check if $1 is not set (empty)
  if [ -z "$1" ]; then
    echo "Error: Argument not provided"
    exit 1
  fi

  # set api key
  VAST_API_KEY=$1

  # create a working directory
  rm -rf $WORKDIR
  mkdir -p $WORKDIR
  cd $WORKDIR

  # create virtual env
  echo "create virtual env"
  python3 -m venv $WORKDIR/venv

  # activate virtual env
  source $WORKDIR/venv/bin/activate

  # install requests
  echo "install requests"
  pip3 install requests

  # download the vast CLI
  echo "download vast CLI"
  wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast
  chmod +x vast

  # set api key
  ./vast set api-key $VAST_API_KEY

  # generate a new SSH key
  echo "generate a new SSH key"
  ssh-keygen -q -t ed25519 -N '' -f $WORKDIR/id_ed25519

fi

# https://vast.ai/docs/cli/commands
query=""
query+="reliability > 0.99 "           # high reliability
query+="num_gpus=1 "                   # single gpu
query+="gpu_name=$GPU_NAME "           # GPU
query+="driver_version >= 535.154.05 " # driver version
query+="cuda_vers >= 11.8 "            # cuda version
query+="compute_cap >= 750 "           # compute capability
query+="geolocation in [US] "          # country US,CA,IS,TW,VN,GB,NO
query+="rentable=True "                # rentable only
query+="verified=True "                # verified by vast.ai
query+="disk_space >= $DISK_SPACE "    # available disk space
query+="dph <= 1.0 "                   # less than $1 per hour
query+="duration >= 7 "                # at least 7 days online
query+="inet_up >= 200 "               # at least 200MB/s upload
query+="inet_down >= 200 "             # at least 200MB/s download
query+="cpu_ram >= 32 "                # at least 32GB ram
query+="inet_up_cost <= 0.25 "         # upload cheaper than $0.25/GB
query+="inet_down_cost <= 0.25 "       # download cheaper than $0.25/GB

TRIED_LIST=()
while true; do

  echo "find offer cheapest"
  if [ ${#TRIED_LIST[@]} -gt 0 ]; then
    condition="host_id not in ["
    for host_id in "${TRIED_LIST[@]}"; do
      condition+="$host_id,"
    done
    condition="${condition%,}"
    condition+="]"
  else
    condition=""
  fi

  OFFER_CMD="./vast search offers \"$query $condition\" -o 'dph' --raw"
  echo $OFFER_CMD
  OFFER=$(eval $OFFER_CMD)
  INSTANCE_ID=$(printf "%s\n" "$OFFER" | jq -r '.[0].id')
  HOST_ID=$(printf "%s\n" "$OFFER" | jq -r '.[0].host_id')
  TRIED_LIST+=($HOST_ID)

  # verify that the instance ID is valid
  echo "instance_id: $INSTANCE_ID"
  echo "host_id: $HOST_ID"

  if [[ -z "$INSTANCE_ID" ]]; then
    echo "No valid instance found"
    sleep $RETRY_INTERVAL
    continue
  fi

  # create an instance
  RESULT=$(./vast create instance $INSTANCE_ID \
    --label "github-actions" \
    --image "$VAST_IMAGE" \
    --disk $DISK_SPACE --ssh \
    --raw)
  RESULT=$(printf "%s\n" "$RESULT" | sed "s/'/\"/g" | sed "s/True/true/g")
  success=$(printf "%s\n" "$RESULT" | jq -r '.success')
  echo $RESULT
  INSTANCE_ID=$(printf "%s\n" "$RESULT" | jq -r '.new_contract')
  if [[ "$success" == "true" ]]; then
    echo "new INSTANCE_ID: $INSTANCE_ID"
  else
    echo "success: $success"
    echo "instance creation failed."
    sleep $RETRY_INTERVAL
    continue
  fi

  # write down the delete command
  echo "source $WORKDIR/venv/bin/activate; $WORKDIR/vast destroy instance $INSTANCE_ID" >$WORKDIR/delete-instance.sh
  chmod +x $WORKDIR/delete-instance.sh

  # wait until the instance is loaded
  VAST_INSTANCE_JSON=$WORKDIR/instance.json
  retry_count=0
  host_ready=false
  while true; do
    $WORKDIR/vast show instances --raw >$VAST_INSTANCE_JSON
    STATUS=$(jq -r --argjson id "$INSTANCE_ID" '.[] | select(.id == $id) | .actual_status' "$VAST_INSTANCE_JSON")
    if [[ "$STATUS" == "running" ]]; then
      echo "host ready"
      host_ready=true
      break
    elif [[ "$STATUS" == "offline" || "$STATUS" == "error" ]]; then
      echo "host failed"
      $WORKDIR/delete-instance.sh
      break
    else
      echo "instance status: $STATUS. retrying in $RETRY_INTERVAL seconds..."
      ((retry_count++))
      if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
        echo "Maximum retries reached. Exiting..."
        $WORKDIR/delete-instance.sh
        break
      fi
      sleep $RETRY_INTERVAL
    fi
  done
  if [ "$host_ready" = false ]; then
    continue
  fi

  # register ssh key
  echo "register ssh key"
  ./vast attach ssh $INSTANCE_ID "$(cat $SSH_PUB_KEY)"

  # write the ssh command to a file
  ssh_command="ssh -i $WORKDIR/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=5 $(./vast ssh-url $INSTANCE_ID)"
  echo "$ssh_command \$@" >$WORKDIR/ssh-command.sh
  chmod +x $WORKDIR/ssh-command.sh

  # write the rsync command to a file
  port=$(echo $(./vast ssh-url $INSTANCE_ID) | sed -E 's/^.*:(.*)$/\1/')
  hostname=$(echo $(./vast ssh-url $INSTANCE_ID) | sed -E 's/^[a-zA-Z]+:\/\/[a-zA-Z0-9._-]+@([^:]+):.*/\1/')
  echo "rsync -avz --exclude='.git' --exclude='asset' -e \"ssh -i $WORKDIR/id_ed25519 -o StrictHostKeyChecking=no -p $port\" . root@${hostname}:/root/ppf-contact-solver" >$WORKDIR/rsync-command.sh
  chmod +x $WORKDIR/rsync-command.sh

  # Loop until SSH connection is successful
  retry_count=0
  ssh_ready=false
  cuda_ready=false
  while true; do
    sleep $RETRY_INTERVAL
    echo "trying to establish SSH connection..."
    eval "$ssh_command \"nvidia-smi\""
    if [ $? -eq 0 ]; then
      echo "SSH connection ready"
      ssh_ready=true
      break
    else
      echo "Connection failed. Retrying in $RETRY_INTERVAL seconds..."
      ((retry_count++))
      if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
        echo "Maximum retries reached."
        $WORKDIR/delete-instance.sh
        break
      fi
    fi
  done
  if [ "$ssh_ready" = true ]; then
    scp_command="scp -i $WORKDIR/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=5 -P $port $CUDA_TESTER_PATH root@${hostname}:/tmp/"
    echo "==== copy cuda-tester.cu ======"
    echo $scp_command
    eval $scp_command
    echo "==== compile cuda ======"
    $WORKDIR/ssh-command.sh "nvcc /tmp/cuda-tester.cu -o /tmp/cuda-tester"
    echo "==== run cuda ======"
    timeout 10s $WORKDIR/ssh-command.sh "/tmp/cuda-tester"
    if [ $? -eq 0 ]; then
      cuda_ready=true
      echo "=== apt update ==="
      $WORKDIR/ssh-command.sh "apt update"
      break
    else
      echo "CUDA test failed"
      $WORKDIR/delete-instance.sh
    fi
  fi
done