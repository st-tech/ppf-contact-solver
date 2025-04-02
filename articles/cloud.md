## 📡 CLI Instructions on Cloud Services

This article summarizes how to deploy our solver on major cloud platforms using the provided CLI.
The CLI enables faster and more consistent creation and deletion of instances. It can also be extended to create a cleaner UI that facilitates instance management.

### 📦 [vast.ai](https://vast.ai)

[vast.ai](https://vast.ai) provides a CLI for deploying instances.
Here’s an example bash script to automate the task.
First, we setup [vast.ai](https://vast.ai) CLI:

```bash
# install vast-ai CLI (https://cloud.vast.ai/cli/)
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast
chmod +x vast

# api key
VAST_API_KEY="get an API key at https://cloud.vast.ai/cli/"

# set API key
./vast set api-key $VAST_API_KEY

# jq must be installed (sudo apt install jq)
jq --version
```

Set variables

```bash
# your local public ssh key
SSH_PUB_KEY=$HOME/.ssh/id_ed25519.pub

# disk space 64GB
DISK_SPACE=64

# GPU
GPU_NAME=RTX_4090

# Image
VAST_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"
```

Search an instance

```bash
# https://vast.ai/docs/cli/commands
query=""
query+="reliability > 0.98 " # high reliability
query+="num_gpus=1 " # single gpu
query+="gpu_name=$GPU_NAME " # GPU
query+="driver_version >= 535.154.05 " # driver version
query+="cuda_vers >= 11.8 " # cuda version
query+="compute_cap >= 750 " # compute capability
query+="geolocation in [TW,VN,JP]" # location country code
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

# find offer cheapest
RESULT=$(./vast create instance $INSTANCE_ID \
  --label "github-actions" \
  --image "$VAST_IMAGE" \
  --disk $DISK_SPACE --ssh \
  --env TZ=Asia/Tokyo \
  --raw)
RESULT=$(printf "%s\n" "$RESULT" | sed "s/'/\"/g" | sed "s/True/true/g")
success=$(printf "%s\n" "$RESULT" | jq -r '.success')
INSTANCE_ID=$(printf "%s\n" "$RESULT" | jq -r '.new_contract')
if [[ "$success" == "true" ]]; then
  echo "new INSTANCE_ID: $INSTANCE_ID"
else
  echo "success: $success"
  echo "instance creation failed."
fi
```

Deploy

```bash
# create an instance
./vast create instance $INSTANCE_ID \
   --label "ppf-contact-solver" \
   --image "$VAST_IMAGE" \
   --disk $DISK_SPACE --ssh \

# ssh info fetch
VAST_INSTANCE_JSON=/tmp/vast-instance.json
while true; do
  ./vast show instances --raw > $VAST_INSTANCE_JSON
  SSH_IP=$(jq -r '.[0].public_ipaddr' "$VAST_INSTANCE_JSON" 2>/dev/null)
  SSH_PORT=$(jq -r '.[0].ports["22/tcp"][] | select(.HostIp == "0.0.0.0") | .HostPort' "$VAST_INSTANCE_JSON" 2>/dev/null)
  if [[ -n "$SSH_IP" && -n "$SSH_PORT" ]]; then
    sleep 1
    break  # exit the loop if both are valid
  else
    echo "failed to fetch SSH details. retrying in 5 seconds..."
    sleep 5  # wait for 5 seconds before retrying
  fi
done

# register ssh key
echo "register ssh key"
./vast attach ssh $(./vast show instances -q) "$(cat $SSH_PUB_KEY)"
```

Now connect via SSH.
If the first connection attempt fails, try again after a few seconds.

```bash
# ssh into the server port forwarding 8080 <--> 8080
ssh -p $SSH_PORT root@${SSH_IP} -L 8080:localhost:8080
```

After logging in, follow the instructions from [Both Systems](../articles/install.md#-both-systems) to install our solver.
Once the JupyterLab frontend is up, you can access it at <http://localhost:8080>.
After use, follow the instructions below to destroy the instance.

```bash
# destroy instance
./vast destroy instance $(./vast show instances -q)

# list all instances
./vast show instances

echo "visit web interface https://cloud.vast.ai/instances/"
echo "to make sure that all instances are deleted"
```

If you wish to wipe the entire [vast.ai CLI](https://vast.ai/docs/cli/commands) installation, run the commands below:

```bash
# (optional) delete vast CLI and config
rm -f vast
rm -rf $HOME/.config/vastai
```

### 📦 [RunPod](https://runpod.io)

[RunPod](https://runpod.io) also provides a CLI for deploying instances.
Here’s an example bash script to automate the task.

First, install [runpodctl](https://github.com/runpod/runpodctl).
Note that, as of late 2024, the official binary release does not offer SSH connection support. For this use, a direct GitHub clone is required.

```bash
# clone runpodctl latest copy
git clone https://github.com/runpod/runpodctl.git $HOME/runpodctl

# compile runpodctl
cd $HOME/runpodctl; make; cd -

# set ephemeral path only valid in the current shell
PATH=$PATH:$HOME/runpodctl/bin/

# this must return greater than 1.0.0-test
# as of late 2024 the official release does not offer ssh connect
runpodctl --version

# set API key (generate at https://www.runpod.io/console/user/settings)
RUNPOD_API_KEY="...your_api_key..."
runpodctl config; runpodctl config --apiKey $RUNPOD_API_KEY
```

Next, set the necessary variables.

```bash
# disk space 64GB
DISK_SPACE=64

# GPU
GPU_NAME="RTX 4090"

# Image
RUNPOD_IMAGE="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04"

# go must be installed at this point (https://go.dev/doc/install)
go version
```

Now deploy an instance

```bash
# create a pod. rent cost must be less than $1 per hour
runpodctl create pod \
  --name ppf-contact-solver --startSSH \
  --ports '22/tcp' --cost 1.0 --gpuCount 1 \
  --gpuType "NVIDIA GeForce $GPU_NAME" \
  --containerDiskSize $DISK_SPACE \
  --imageName "$RUNPOD_IMAGE"

# get pod_id
POD_ID=$(runpodctl get pod | grep -v '^ID' | cut -f1)
echo "pod_id: $POD_ID"
```

Now connect via SSH.
If the first connection attempt fails, try again after a few seconds.

```bash
# connect ssh portforward 8080 <-> 8080
eval $(runpodctl ssh connect $POD_ID) -L 8080:localhost:8080
```

After logging in, follow the instructions from [Both Systems](../articles/install.md#-both-systems) to install our solver.
Once the JupyterLab frontend is up, you can access it at <http://localhost:8080>.
After use, follow the instructions below to destroy the instance.

```bash
# remove pod
runpodctl remove pod $POD_ID

# list pods
runpodctl get pod

echo "also check the web interface to confirm the pod is removed"
```

If you wish to wipe the entire [runpodctl](https://github.com/runpod/runpodctl) installation, run the commands below:

```bash
# remove runpod if desired
rm -rf $HOME/runpodctl
rm -rf .runpod
```

### 📦 [Scaleway](https://www.scaleway.com/en/)

Set up your Scaleway CLI by following 📚 [this guide](https://www.scaleway.com/en/cli/).
Also, register your public SSH key.
Here’s how to create a GPU instance and SSH into the instance with port forwarding.
This setup costs approximately €0.76 per hour.

```bash
# set zone
zone=fr-par-2

# set name
name=ppf-contact-solver

# set type L4-1-24G or GPU-3070-S
type=L4-1-24G

# create
result=$(scw instance server create \
         --output json \
         name=$name \
         type=$type \
         image=ubuntu_jammy_gpu_os_12 \
         zone=$zone)
id=$(jq -r '.id' <<< "$result")
ip=$(jq -r '.public_ip.address' <<< "$result")

# print info
echo "ID: $id IP: $ip"

# ssh into the server
ssh root@${ip} -L 8080:localhost:8080
```

SSH might fail until the instance is fully loaded; try again at intervals.
Once connected, run the same Docker 🐧 [Linux](../articles/install.md#-linux) command on the instance to set up a 🐳 Docker environment.
After use, run the following command to clean up.

```bash
# cleanup
scw instance server terminate $id zone=$zone with-ip=true with-block=true

# check
scw instance server list zone=$zone
```

Double-check from the 🖥️ web console to confirm that the instance has been successfully ✅ deleted.
Also, check that both the flexible IP and its associated storage are deleted.

### 📦 [Google Compute Engine](https://cloud.google.com/products/compute)

First, set up your `gcloud` CLI by following 📚 [this guide](https://cloud.google.com/sdk/docs/install?hl=en).
Next, run the command below to provision an ⚡ NVIDIA L4 GPU instance.
As of late 2024, this setup costs approximately 💵 $1 per hour.

```bash
IMAGE="projects/ml-images/global/images/c0-deeplearning-common-gpu-v20241118-debian-11-py310"
ZONE="asia-east1-c"
INSTANCE_NAME="ppf-contact-solver"
INSTANCE_TYPE="g2-standard-8"

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$INSTANCE_TYPE \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --accelerator=count=1,type=nvidia-l4 \
    --create-disk=auto-delete=yes,boot=yes,image=$IMAGE,mode=rw,size=50,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm
```

After creating the instance, try connecting to it via `ssh` using the 🌐 `gcloud` interface. Since port `8080` is taken by the deployed image, make sure to select a different port on the host side.
Here, we set the host-side port to `8888`.
Note that the local port remains `8080` so that the JupyterLab interface can be accessed at <http://localhost:8080>.
I know this might be confusing, so just copy and paste if you're okay with it.

```bash
# Port number 8080 is taken, so let's use 8888
gcloud compute ssh --zone=$ZONE $INSTANCE_NAME -- -L 8080:localhost:8888
```

As shown in this [(Video)](https://drive.google.com/file/d/1dj6TvR2IVLKLFXtO8QRrRl-8xQ7G547A/view?usp=sharing), the instance may take a few minutes to load, so early SSH access fails.
Keep trying at intervals; it should connect once the host is ready.

Next, run the same Docker 🐧 [Linux](../articles/install.md#-linux) command in the instance to set up a 🐳 Docker environment. Be sure to change `$MY_WEB_PORT` to `8888` in this case.

```
MY_WEB_PORT=8888  # Make sure to set the port to 8888
MY_TIME_ZONE=Asia/Tokyo  # Your time zone
MY_CONTAINER_NAME=ppf-contact-solver  # Container name
```

The rest of the installation process is identical. After use, don't forget to ❌ delete the instance, or you will continue to be 💸 charged.  

```bash
gcloud compute instances stop --zone=$ZONE $INSTANCE_NAME
gcloud compute instances delete --zone=$ZONE $INSTANCE_NAME
```

Just to be sure, double-check from the 🖥️ web console to confirm that the instance has been successfully ✅ deleted.

