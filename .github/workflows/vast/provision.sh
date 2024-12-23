#!/bin/bash

# Check if $1 is not set (empty)
if [ -z "$1" ]; then
  echo "Error: Argument not provided"
  exit 1
fi

# set api key
VAST_API_KEY=$1

# create a working directory
WORKDIR=/tmp/vast-ci
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
ssh-keygen -q -t rsa -N '' -f $WORKDIR/id_rsa

# your local public ssh key
SSH_PUB_KEY=$WORKDIR/id_rsa.pub

# disk space 64GB
DISK_SPACE=64

# GPU
GPU_NAME=RTX_4090

# Image
VAST_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"

# Retry interval
RETRY_INTERVAL=10

# max retries
MAX_RETRIES=10

# https://vast.ai/docs/cli/commands
query=""
query+="reliability > 0.98 " # high reliability
query+="num_gpus=1 " # single gpu
query+="gpu_name=$GPU_NAME " # GPU
query+="cuda_vers >= 11.8 " # cuda version
query+="compute_cap >= 750 " # compute capability
query+="geolocation=TW " # location country code
query+="rentable=True " # rentable only
query+="verified=True " # verified by vast.ai
query+="disk_space >= $DISK_SPACE " # available disk space
query+="dph <= 1.0 " # less than $1 per hour
query+="duration >= 3 " # at least 3 days online
query+="inet_up >= 300 " # at least 300MB/s upload
query+="inet_down >= 300 " # at least 300MB/s download
query+="cpu_ram >= 32 " # at least 32GB ram
query+="inet_up_cost <= 0.05 " # upload cheaper than $0.05/GB
query+="inet_down_cost <= 0.05 " # download cheaper than $0.05/GB

echo "find offer cheapest"
INSTANCE_ID=$(./vast search offers $query -o 'dph' | awk 'NR==2 {print $1}')

# verify that the instance ID is valid
echo "instance_id: $INSTANCE_ID"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "No valid instance found"
  exit 1
fi

# create an instance
RESULT=$(./vast create instance $INSTANCE_ID \
   --label "github-actions" \
   --image "$VAST_IMAGE" \
   --disk $DISK_SPACE --ssh --direct \
   --env TZ=Asia/Tokyo \
   --raw)
RESULT=$(echo "$RESULT" | sed "s/'/\"/g" | sed "s/True/true/g")
success=$(echo "$RESULT" | jq -r '.success')
echo $RESULT
INSTANCE_ID=$(echo "$RESULT" | jq -r '.new_contract')
if [[ "$success" == "true" ]]; then
  echo "new INSTANCE_ID: $INSTANCE_ID"
else
  echo "success: $success"
  echo "instance creation failed."
fi

# write down the delete command
echo "source $WORKDIR/venv/bin/activate; $WORKDIR/vast destroy instance $INSTANCE_ID" > $WORKDIR/delete-instance.sh
chmod +x $WORKDIR/delete-instance.sh

# register ssh key
echo "register ssh key"
sleep 3
./vast attach ssh $INSTANCE_ID "$(cat $SSH_PUB_KEY)" --raw

# write the ssh command to a file
ssh_command="ssh -i $WORKDIR/id_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=5 $(./vast ssh-url $INSTANCE_ID)"
echo "$ssh_command \$@" > $WORKDIR/ssh-command.sh
chmod +x $WORKDIR/ssh-command.sh

# write the rsync command to a file
port=$(echo $(./vast ssh-url $INSTANCE_ID) | sed -E 's/^.*:(.*)$/\1/')
hostname=$(echo $(./vast ssh-url $INSTANCE_ID) | sed -E 's/^[a-zA-Z]+:\/\/[a-zA-Z0-9._-]+@([^:]+):.*/\1/')
echo "rsync -avz --exclude='.git' --exclude='asset' -e \"ssh -i $WORKDIR/id_rsa -o StrictHostKeyChecking=no -p $port\" . root@${hostname}:/root/ppf-contact-solver" > $WORKDIR/rsync-command.sh
chmod +x $WORKDIR/rsync-command.sh

# Loop until SSH connection is successful
retry_count=0
while true; do
  echo "trying to establish SSH connection..."
  eval "$ssh_command \"exit\""
  if [ $? -eq 0 ]; then
    echo "SSH connection ready"
    break
  else
    echo "Connection failed. Retrying in $RETRY_INTERVAL seconds..."
    ((retry_count++))
    if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
      echo "Maximum retries reached. Exiting..."
      $WORKDIR/delete-instance.sh
      exit 1
    fi
    sleep $RETRY_INTERVAL
  fi
done