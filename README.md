# ZOZO's Contact Solver 🫶 [![Getting Started](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml)

A contact solver for physics-based simulations involving 👚 shells, 🪵 solids and 🪢 rods. All made by ZOZO.
Published in [ACM Transactions on Graphics (TOG)](https://dl.acm.org/doi/abs/10.1145/3687908).

<img src="./asset/image/teaser-image.jpg" alt="solver logo">
  
## ✨ Highlights

- **💪 Robust**: Contact resolutions are penetration-free. No snagging intersections.
- **⏲ Scalable**: An extreme case includes beyond 150M contacts. Not just one million.
- **🚲 Cache Efficient**: All on the GPU runs in single precision. No double precision.
- **🥼 Inextensible**: Cloth never extends beyond very strict upper bounds, such as 1%.
- **📐 Physically Accurate**: Our deformable solver is driven by the Finite Element Method.
- **🚀 Massively Parallel**: Both contact and elasticity solvers are run on the GPU.
- **🐳 Docker Sealed**: Everything is designed to work out of the box.
- **🌐 JupyterLab Included**: Open your browser and run examples right away [[Video]](https://drive.google.com/file/d/1n068Ai_hlfgapf2xkAutOHo3PkLpJXA4/view?usp=sharing).
- **☁️ Cloud-Ready**: Our solver can be seamlessly deployed on major cloud platforms.
- **✨ Stay Clean**: You can remove all traces after use.
- **👌 Open**: We have opted the Apache v2.0 license.

## 🔖 Table of Contents

- [🎓 Technical Materials](#-technical-materials)
- [⚡️ Requirements](#️-requirements)
- [📝 Change History](#-change-history)
- [🐍 How To Use](#-how-to-use)
- [🖼️ Catalogue](#️-catalogue)
- [💨 Getting Started](#-getting-started)
    - [🎥 Installation Videos](#-installation-videos)
    - [🐳 Installing Docker](#-installing-docker)
    - [🪟 Windows](#-windows)
    - [🐧 Linux](#-linux)
    - [🧹 Cleaning Up](#-cleaning-up)
- [📡 Deploying on Cloud Services](#-deploying-on-cloud-services)
    - [📦 Deploying on vast.ai](#-deploying-on-vastai)
    - [📦 Deploying on RunPod](#-deploying-on-runpod)
    - [📦 Deploying on Google Compute Engine](#-deploying-on-google-compute-engine)
- [📃 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [🖋 Citation](#-citation)

## 🎓 Technical Materials

- 🎥 Main video [[Video]](https://drive.google.com/file/d/1OzPbUoqddUYDvXMvRnUHH7kz0nZhmt7K/view?usp=drive_link)
- 🎥 Additional video examples [[Directory]](https://drive.google.com/drive/folders/1O4t3CBcG8qqju_qun0RP60OULK4_1tTf?usp=drive_link)
- 🎥 Presentation videos [[Short]](https://drive.google.com/file/d/1axAbFRtbOxhkU7K3Wf9F5gh2iDNJn6CZ/view?usp=sharing)[[Long]](https://drive.google.com/file/d/1zybHydN0a0cZ-ifl_D_LYLwdMOnz2YnP/view?usp=sharing)
- 📃 Main paper [[PDF]](https://drive.google.com/file/d/1OrOKJH_im1L4j1cJB18sfvNHEbZVSqjL/view?usp=drive_link)[[Hindsight]](./articles/hindsight.md)
- 📊 Supplementary PDF [[PDF]](https://drive.google.com/file/d/1ptjFNVufPBV4-vb5UDh1yTgz8-esjaSF/view?usp=drive_link)
- 🤖 Supplementary scripts [[Directory]](https://drive.google.com/drive/folders/13CO068xLkd6ZSxsqtJQdNadgMrbbfSug?usp=drive_link)
- 🔍 Singular-value eigenanalysis [[Markdown]](./articles/eigensys.md)

## ⚡️ Requirements

- 🔥 A modern NVIDIA GPU (Turing or newer).
- 🐳 A Docker environment (see [below](#-getting-started)).

## 📝 Change History

- (2024.12.18) Added a [frictional contact example](./examples/friction.ipynb): armadillo sliding on the slope [[Video]](https://drive.google.com/file/d/12WGdfDTFIwCT0UFGEZzfmQreM6WSSHet/view?usp=sharing).
- (2024.12.18) Added a [hindsight](./articles/hindsight.md) noting that the tilt angle was not $30^\circ$, but rather $26.57^\circ$.
- (2024.12.16) Removed thrust dependencies to fix runtime errors for the driver version `560.94` [[Issue Link]](https://github.com/st-tech/ppf-contact-solver/issues/1).

## 🐍 How To Use

Our frontend is accessible through 🌐 a browser using our built-in JupyterLab 🐍 interface.
All is set up when you open it for the first time.
Results can be interactively viewed through the browser and exported as needed.

This allows you to interact with the simulator on your 💻 laptop while the actual simulation runs on a remote headless server over 🌍 the internet.
This means that you don't have to buy ⚙️ hardware, but can rent it at [vast.ai](https://vast.ai) or [RunPod](https://www.runpod.io/) for less than 💵 $1 per hour.
For example, this [[Video]](https://drive.google.com/file/d/1n068Ai_hlfgapf2xkAutOHo3PkLpJXA4/view?usp=sharing) was recorded on a [vast.ai](https://vast.ai) instance.
The experience is 👍 good!

Here's an example of draping five sheets over a sphere with two corners pinned.
Please look into the [examples](./examples/) directory for more examples.

```python
# import our frontend
from frontend import App

# make an app with the label "drape"
app = App("drape", renew=True)

# create a square mesh resolution 128 spanning the xz plane
V, F = app.mesh.square(res=128, ex=[1,0,0], ey=[0,0,1])

# add to the asset and name it "sheet"
app.asset.add.tri("sheet", V, F)

# create an icosphere mesh radius 0.5 and 5 subdivisions
V, F = app.mesh.icosphere(r=0.5, subdiv_count=5)

# add to the asset and name it "sphere"
app.asset.add.tri("sphere", V, F)

# create a scene "five-sheets"
scene = app.scene.create("five-sheets")

# define gap between sheets
gap = 0.01

for i in range(5):
    
    # add a sheet to the scene
    obj = scene.add("sheet")

    # pick two vertices max towards directions [1,0,-1] and [-1,0,-1]
    corner = obj.grab([1, 0, -1]) + obj.grab([-1, 0, -1])

    # place it with a vertical offset and pin the corners
    obj.at(0, gap * i, 0).pin(corner)

    # set fiber directions required for the Baraff-Witkin model
    obj.direction([1, 0, 0], [0, 0, 1])

# add a sphere mesh at a lower position and set it to a static collider
scene.add("sphere").at(0, -0.5 - gap, 0).pin()

# compile the scene and report stats
fixed = scene.build().report()

# interactively preview the built scene (image left)
fixed.preview()

# set simulation parameter(s)
param = app.session.param()
param.set("dt", 0.01)

# create a new session with a name
session = app.session.create("dt-001").init(fixed)

# start the simulation and live-preview the results (image right)
session.start(param).preview()

# also show streaming logs
session.stream()

# or interactively view the animation sequences
session.animate()

# export all simulated frames (downloadable from the file browser)
session.export_animation(f"export/{session.info.name}")
```
<img src="./asset/image/drape.jpg" alt="drape">
  
## 🖼️ Catalogue

|||||
|---|---|---|---|
|woven|stack|trampoline|needle|
|![](./asset/image/catalogue/woven.mp4.gif)|![](./asset/image/catalogue/stack.mp4.gif)|![](./asset/image/catalogue/trampoline.mp4.gif)|![](./asset/image/catalogue/needle.mp4.gif)|
|cards|codim|hang|trapped|
|![](./asset/image/catalogue/cards.mp4.gif)|![](./asset/image/catalogue/codim.mp4.gif)|![](./asset/image/catalogue/hang.mp4.gif)|![](./asset/image/catalogue/trapped.mp4.gif)|
|domino|noodle|drape|quintuple|
|![](./asset/image/catalogue/domino.mp4.gif)|![](./asset/image/catalogue/noodle.mp4.gif)|![](./asset/image/catalogue/drape.mp4.gif)|![](./asset/image/catalogue/quintupletwist.mp4.gif)|
|ribbon|curtain|fishingknot|friction|
|![](./asset/image/catalogue/ribbon.mp4.gif)|![](./asset/image/catalogue/curtain.mp4.gif)|![](./asset/image/catalogue/fishingknot.mp4.gif)|![](./asset/image/catalogue/friction-armadillo.mp4.gif)|

At the moment, not all examples are ready yet, but they will be added/updated one by one.
The author is actively woriking on it.

## 💨 Getting Started

🛠️ All the steps below are verified to run without errors via automated GitHub Actions ⚙️ (see `.github/workflows/getting-started.yml`
).

The tested 🚀 runner is the Ubuntu NVIDIA GPU-Optimized Image for AI and HPC with an NVIDIA Tesla T4 (16 GB VRAM) with Driver version 550.127.05.
This is not a self-hosted runner, meaning that each time the runner launches, all environments are 🌱 fresh.

### 🎥 Installation Videos

We provide uninterrupted recorded installation videos (🪟 Windows [[Video]](https://drive.google.com/file/d/1Np3MwUtSlppQPMrawtobzoGtZZWrmFgG/view?usp=sharing), 🐧 Linux [[Video]](https://drive.google.com/file/d/1ZDnzsn46E1I6xNzyg0S8Q6xvgXw_Lw7M/view?usp=sharing), ☁ [vast.ai](https://vast.ai) [[Video]](https://drive.google.com/file/d/1k0LnkPKXuEwZZvElaKohWZeDd6M3ONe1/view?usp=sharing), and ☁️ [Google Cloud](https://cloud.google.com/products/compute) [[Video]](https://drive.google.com/file/d/1dj6TvR2IVLKLFXtO8QRrRl-8xQ7G547A/view?usp=sharing))
to reduce stress 😣 during the installation process. We encourage you to 👀 check them out to get a sense of how things go ⏳ and how long ⏱️ each step takes.

### 🐳 Installing Docker

To get the ball ⚽ rolling, we'll configure a Docker environment 🐳 to minimize any trouble 🤯 that 🥊 hits you.

> [!NOTE]
> If you wish to install our solver on a headless remote machine, SSH into the server with port forwarding using the following command:
> ```
> ssh -L 8080:localhost:8080 user@remote_server_address
> ```
> This port will be used to access the frontend afterward.
> The two port numbers of `8080` must match the value we set for `$MY_WEB_PORT` below.

First, install the CUDA Toolkit [[Link]](https://developer.nvidia.com/cuda-downloads) along with the driver on your host system.
Next, follow the instructions below specific to the operating system running on the host.

### 🪟 Windows

Install the latest version of Docker Desktop [[Link]](https://docs.docker.com/desktop/setup/install/windows-install/) on the host computer.
You may need to log out or reboot after the installation. After logging back in, launch Docker Desktop to ensure that Docker is running.
Then, create a container 📦 by running the following Docker command in PowerShell:

```
$MY_WEB_PORT = 8080  # Port number for JupyterLab web browsing
$MY_TIME_ZONE = "Asia/Tokyo"  # Your time zone
$MY_CONTAINER_NAME = "ppf-contact-solver"  # Container name

docker run -it `
    --gpus all `
    -p ${MY_WEB_PORT}:8080 `
    -e TERM `
    -e TZ=$MY_TIME_ZONE `
    -e LANG=en_US.UTF-8 `
    --hostname ppf-dev `
    --name $MY_CONTAINER_NAME `
    -e NVIDIA_DRIVER_CAPABILITIES="graphics,compute,utility" `
    nvidia/cuda:11.8.0-devel-ubuntu22.04
```

Windows users do not need to install the NVIDIA Container Toolkit.

### 🐧 Linux

Linux users will also need to install Docker 🐋 on their system.
Please refer to the installation guide [[Link]](https://docs.docker.com/engine/install/).
Also, install the NVIDIA Container Toolkit by following the guide [[Link]](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Then, create a container 📦 by running the following Docker command:

```
MY_WEB_PORT=8080  # Port number for JupyterLab web browsing
MY_TIME_ZONE=Asia/Tokyo  # Your time zone
MY_CONTAINER_NAME=ppf-contact-solver  # Container name

docker run -it \
    --gpus all \
    -p $MY_WEB_PORT:8080 \
    -e TERM -e TZ=$MY_TIME_ZONE \
    -e LANG=en_US.UTF-8 \
    --hostname ppf-dev \
    --name $MY_CONTAINER_NAME -e \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    nvidia/cuda:11.8.0-devel-ubuntu22.04
```

### 🪟🐧 Both Systems

At the end of the line, you should see:

```
root@ppf-dev:/#
```

From here on, all commands will happen in the 📦 container, not on your host.
Next, we'll make sure that a NVIDIA driver is visible from the Docker container. Try this

```
nvidia-smi
```

If successful, this will get back to you with something like this

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:C1:00.0 Off |                  Off |
| 64%   51C    P2   188W / 450W |   4899MiB / 24564MiB |     91%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```

> [!NOTE]
> If an error occurs 🥵, ensure that `nvidia-smi` is working on your host. For Linux users, make sure the NVIDIA Container Toolkit is properly installed. If the issue persists, try running `sudo service docker restart` on your host to resolve it.

Please confirm that your GPU is listed here.
Now let's get the installation started.
No worries 🤙; all the commands below only disturb things in the container, so your host environment stays clean ✨.
First, install following packages

```
apt update
apt install -y git python3
```

Next, clone our respository

```
git clone https://github.com/st-tech/ppf-contact-solver.git
```

Move into the ```ppf-contact-solver``` and let ```warmup.py``` do all the rest 💤:

```
cd ppf-contact-solver
python3 warmup.py
```

> [!NOTE]
> If you’re suspicious, you can look around ```warmup.py``` before you proceed. Run `less warmup.py`, scroll all the way to the bottom, and hit `q` to quit.

Now we're set. Let's kick in the compilation!🏃

```
source "$HOME/.cargo/env"
cargo build --release
```

Be patient; this takes some time... ⏰⏰ If the last line says 

```
Finished `release` profile [optimized] target(s) in ...
```

We're done! 🎉 Start our frontend by

```
python3 warmup.py jupyter
```

and now you can access our JupyterLab frontend from http://localhost:8080 on your 🌐 browser.
The port number `8080` is the one we set for `$MY_WEB_PORT`.
Enjoy! 😄

### 🧹 Cleaning Up

To remove all traces, simply stop 🛑 the container and ❌ delete it.
Be aware that all simulation data will be also lost. Back up any important data if needed.

```
docker stop $MY_CONTAINER_NAME
docker rm $MY_CONTAINER_NAME
```

> [!NOTE]
> If you wish to completely wipe what we’ve done here, you may also need to purge the Docker image by:
> ```
> docker rmi $(docker images | grep 'nvidia/cuda' | grep '11.8.0-devel-ubuntu22.04' | awk '{print $3}')
> ```
> but don't do this if you still need it.

## 📡 Deploying on Cloud Services

### 📦 Deploying on [vast.ai](https://vast.ai)

The exact same steps above should work (see `.github/workflows/getting-started-vast.yml`), except that you'll need to create a Docker template. Here's one:

- **Image Path/Tag**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Docker Options**: `-e TZ=Asia/Tokyo` (Your time zone, of course)
- Make sure to select ✅ ***Run interactive shell server, SSH.***
- When connecting via SSH, make sure to include `-L 8080:localhost:8080` in the command.
- For a better experience, choose a geographically nearby server with a high connection speed.
- Also, make sure to allocate a large disk space, such as 64GB.

**Important**: Don't forget to ❌ delete the instance after use, or you’ll be 💸 charged for nothing.

[vast.ai](https://vast.ai) provides a CLI interface for deploying instances.
Here’s an example bash script to automate the task.
First, install [vast.ai CLI](https://cloud.vast.ai/cli/) and set variables:

```bash
# api key
VAST_API_KEY="get an API key at https://cloud.vast.ai/cli/"

# your local public ssh key
SSH_PUB_KEY=$HOME/.ssh/id_ed25519.pub

# disk space 64GB
DISK_SPACE=64

# GPU
GPU_NAME=RTX_4090

# jq must be installed (sudo apt install jq)
jq --version

# install vast-ai CLI (https://cloud.vast.ai/cli/)
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast
chmod +x vast

./vast set api-key $VAST_API_KEY
```

Next, search an instance

```bash
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
query+="inet_up_cost <= 0.05 " # upload cheaper than $0.5/GB
query+="inet_down_cost <= 0.05 " # download cheaper than $0.5/GB

# find offer cheapest
INSTANCE_ID=$(./vast search offers $query -o 'dph' | awk 'NR==2 {print $1}')

# verify that the instance ID is valid
echo "instance_id: $INSTANCE_ID"
```

and deploy

```bash
# create an instance
./vast create instance $INSTANCE_ID \
   --label "ppf-contact-solver" \
   --image "nvidia/cuda:11.8.0-devel-ubuntu22.04" \
   --disk $DISK_SPACE --ssh --direct \
   --env TZ=Asia/Tokyo

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
    echo "failed to fetch SSH details. Retrying in 5 seconds..."
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

After logging in, follow the instructions from [Both Systems](#-both-systems) to install our solver.
Once the JupyterLab frontend is up, you can access it at http://localhost:8080.
After use, follow the instructions below to destroy the instance.

```bash
# destroy instance
./vast destroy instance $(./vast show instances -q)

# list all instances
./vast show instances

echo "visit web interface https://cloud.vast.ai/instances/ to make sure that all instances are deleted"
```

If you wish to wipe the entire [vast.ai CLI](https://vast.ai/docs/cli/commands) installation, run the commands below:

```bash
# (optional) delete vast CLI and config
rm -f vast
rm -rf $HOME/.config/vastai
```

### 📦 Deploying on [RunPod](https://runpod.io)

You can deploy our solver on a RunPod instance. To do this, we need to select an official RunPod Docker image instead.

- **Container Image**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Expose HTTP Ports**: Empty
- **Expose TCP Ports**: `22`
- When connecting via SSH, make sure to include `-L 8080:localhost:8080` in the command.
- For a better experience, choose a geographically nearby server with a high connection speed.
- Also, make sure to allocate a large disk space, such as 64GB.
- ✅ Make sure to select `SSH Terminal Access`
- ❌ Deselect `Start Jupyter Notebook`

**Important**: Don't forget to ❌ delete the instance after use, or you’ll be 💸 charged for nothing.

[RunPod](https://runpod.io) also provides a CLI interface for deploying instances.
Here’s an example bash script to automate the task.
First, set the necessary variables.

```bash
# set API key (generate at https://www.runpod.io/console/user/settings)
RUNPOD_API_KEY="..."

# disk space 64GB
DISK_SPACE=64

# GPU
GPU_NAME="RTX 4090"

# go must be installed at this point (https://go.dev/doc/install)
go version
```

Next, install [runpodctl](https://github.com/runpod/runpodctl).
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

runpodctl config; runpodctl config --apiKey $RUNPOD_API_KEY
```

Now deploy an instance

```bash
# create a pod. rent cost must be less than $1 per hour
runpodctl create pod \
  --name ppf-contact-solver --startSSH \
  --ports '22/tcp' --cost 1.0 --gpuCount 1 \
  --gpuType "NVIDIA GeForce $GPU_NAME" \
  --containerDiskSize $DISK_SPACE \
  --imageName 'runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04'

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

After logging in, follow the instructions from [Both Systems](#-both-systems) to install our solver.
Once the JupyterLab frontend is up, you can access it at http://localhost:8080.
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

### 📦 Deploying on [Google Compute Engine](https://cloud.google.com/products/compute)

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
Note that the local port remains `8080` so that the JupyterLab interface can be accessed at `http://localhost:8080`.
I know this might be confusing, so just copy and paste if you're okay with it.

```bash
# Port number 8080 is taken, so let's use 8888
gcloud compute ssh --zone=$ZONE $INSTANCE_NAME -- -L 8080:localhost:8888
```

As shown in this [[Video]](https://drive.google.com/file/d/1dj6TvR2IVLKLFXtO8QRrRl-8xQ7G547A/view?usp=sharing), the instance may take a few minutes to load, so early SSH access fails.
Keep trying at intervals; it should connect once the host is ready.

Next, run the same Docker 🐧 [Linux](#-linux) command in the instance to set up a 🐳 Docker environment. Be sure to change `$MY_WEB_PORT` to `8888` in this case.

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

## 📃 License

📝 This project is licensed under Apache v2.0 license.

## 🙏 Acknowledgements

The author would like to thank ZOZO, Inc. for allowing him to work on this topic as part of his main workload.
The author also extends thanks to the teams in the IP department for permitting the publication of our technical work and the release of our code, as well as to many others for assisting with the internal paperwork required for publication.

## 🖋 Citation

```
@article{Ando2024CB,
author = {Ando, Ryoichi},
title = {A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness},
year = {2024},
issue_date = {December 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3687908},
doi = {10.1145/3687908},
journal = {ACM Trans. Graph.},
month = nov,
articleno = {224},
numpages = {13},
keywords = {collision, contact}
}
```

It should be emphasized that this work was strongly inspired by the IPC.
The author kindly encourages citing their [original work](https://dl.acm.org/doi/10.1145/3386569.3392425) as well.
