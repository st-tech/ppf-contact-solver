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
- [🧑‍💻 Setting Up Your Development Environment](#-setting-up-your-development-environment)
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

- (2024.12.21) Added a [house of cards example](./examples/cards.ipynb) [[Video]](https://drive.google.com/file/d/14KXaapSIM0qYcdp6yFVyNlfFWGwqCpGA/view?usp=sharing)
- (2024.12.18) Added a [frictional contact example](./examples/friction.ipynb): armadillo sliding on the slope [[Video]](https://drive.google.com/file/d/12WGdfDTFIwCT0UFGEZzfmQreM6WSSHet/view?usp=sharing)
- (2024.12.18) Added a [hindsight](./articles/hindsight.md) noting that the tilt angle was not $30^\circ$, but rather $26.57^\circ$
- (2024.12.16) Removed thrust dependencies to fix runtime errors for the driver version `560.94` [[Issue Link]](https://github.com/st-tech/ppf-contact-solver/issues/1)

## 🐍 How To Use

Our frontend is accessible through 🌐 a browser using our built-in JupyterLab 🐍 interface.
All is set up when you open it for the first time.
Results can be interactively viewed through the browser and exported as needed.

This allows you to interact with the simulator on your 💻 laptop while the actual simulation runs on a remote headless server over 🌍 the internet.
This means that you don't have to buy ⚙️ hardware, but can rent it at [vast.ai](https://vast.ai) or [RunPod](https://www.runpod.io/) for less than 💵 $1 per hour.
For example, this [[Video]](https://drive.google.com/file/d/1n068Ai_hlfgapf2xkAutOHo3PkLpJXA4/view?usp=sharing) was recorded on a [vast.ai](https://vast.ai) instance.
The experience is 👍 good!

Our Python interface is designed with the following principles in mind:

- **🛠️ Non-Destructive Mesh Creation**: Relying on third-party tools for mesh creation and loading can make it difficult to change resolutions and result in excessive data storage. We have built-in mesh creation tools to avoid these issues.

- **🔗 Method Chaining**: We adopt the method chaining style from JavaScript, making the API intuitive and easy to use.

- **📦 Single Import for Everything**: All frontend features are accessible by simply importing with `from frontend import App`.

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
> This port forwarding will be used to access the frontend afterward.
> The second port number must match `$MY_WEB_PORT` below.

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

Our contact solver is designed for heavy use in cloud services ☁️, which supports on-demand flexible scalability 🔄 and reduces hardware investment costs 💰.
This has been challenging because the computational performance of CPUs/GPUs designed for data centers 🖥️ is far behind that of those targeted for consumer use 🏠.

With our scalable contact solver framework 🧩 and purely web-based frontends 🌐, this limitation is reduced. Our solver also runs on an NVIDIA L4 🖱️, a data-center designed GPU that offers reasonable pricing 💲, delivering both satisfactory performance 💪 and practical scalability 📈.

### 📦 Deploying on [vast.ai](https://vast.ai)

The exact same steps above should work (see `.github/workflows/getting-started-vast.yml`), except that you'll need to create a Docker template. Here's one:

- **Image Path/Tag**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Docker Options**: `-e TZ=Asia/Tokyo` (Your time zone, of course)
- Make sure to select ✅ ***Run interactive shell server, SSH.***
- When connecting via SSH, make sure to include `-L 8080:localhost:8080` in the command.
- For a better experience, choose a geographically nearby server with a high connection speed.
- Also, make sure to allocate a large disk space, such as 64GB.
- CLI instructions are described in [[Markdown]](./articles/cloud.md#-vastai).

**Important**: Don't forget to ❌ delete the instance after use, or you’ll be 💸 charged for nothing.

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
- CLI instructions are described in [[Markdown]](./articles/cloud.md#-runpod).

**Important**: Don't forget to ❌ delete the instance after use, or you’ll be 💸 charged for nothing.

### 📦 Deploying on [Google Compute Engine](https://cloud.google.com/products/compute)

- Select `GPUs`. We recommend the GPU type `NVIDIA L4` because it's affordable and accessible, as it does not require a high quota. You may select `T4` instead for testing purposes.
- Do **not** check `Enable Virtual Workstation (NVIDIA GRID)`.
- We recommend the machine type `g2-standard-8`.
- Choose the OS type `Deep Learning VM with CUDA 11.8 M126` and set the disk size to `50GB`.
- As of late 2024, this configuration costs approximately $0.86 per hour in `us-central1 (Iowa)` and $1.00 per hour in `asia-east1 (Taiwan)`. 
- Port number `8080` is reserved by the OS image. Set `$MY_WEB_PORT` to `8888`. When connecting via `gcloud`, use the following format:  `gcloud compute ssh --zone "xxxx" "instance-name" -- -L 8080:localhost:8888`.

- CLI instructions are described in [[Markdown]](./articles/cloud.md#-google-compute-engine).

## 🧑‍💻 Setting Up Your Development Environment

Advanced users may be interested in inspecting our 📜 core code to examine how each component ⚙️ contributes to our solver pipeline 🔄. To facilitate this task, we provide a guide below for setting up a comfortable development environment using either 🖥️ [VSCode](https://azure.microsoft.com/en-us/products/visual-studio-code) or ⌨️ [NeoVim](https://neovim.io/).
In fact, this is how we 🚀 develop.

### 🖥️ Complete Installation

First, complete the entire [installation process](#-getting-started) and keep the Docker container 🚢 running.
Make sure that your terminal is attached to the container, with the current directory pointing to `ppf-contact-solver` directory.

### 🛠️ [clangd](https://clangd.llvm.org/) Setup

Just to avoid confusion, all the `python3 warmup.py ...` commands below must be executed in the Docker container on the remote, not on your local machine!

Our code is not compatible with [C/C++ IntelliSense](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) provided by Microsoft.
We instead employ [clangd](https://clangd.llvm.org/) for linting, so make sure not to install IntelliSense into the container.
Otherwise, you will be overwhelmed by 🐞 errors and ⚠️ warnings.
First, run the following command:

```bash
python3 warmup.py clangd
```

This generates the `.clangd` and `.clang-format` config files, which we adhere to when writing code 💻.
They will be automatically detected by [clangd](https://clangd.llvm.org/).

### 🖥️ [VSCode](https://azure.microsoft.com/en-us/products/visual-studio-code) Users

If you intend to use [VSCode](https://azure.microsoft.com/en-us/products/visual-studio-code), run the following command to generate `.vscode/extensions.json` file.

```bash
python3 warmup.py vscode
```

The generated file contains a list of recommended extensions.
You will be prompted to install these extensions when your VSCode connects to the container.
Finally, connect to the container using the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Now you're good to go! 🚀

### ⌨️ [NeoVim](https://neovim.io/) Users

We provide one-liners to install [NeoVim](https://neovim.io/) and other handy tools into the container:

- [🖥️ NeoVim](https://neovim.io/): `python3 warmup.py nvim`
- [💤 LazyVim](http://www.lazyvim.org/): `python3 warmup.py lazyvim`
- [🛠️ Lazygit](https://github.com/jesseduffield/lazygit): `python3 warmup.py lazygit`
- [🐟 fish shell](https://fishshell.com/): `python3 warmup.py fish`
- [⌨️ oh-my-zsh](https://ohmyz.sh/): `python3 warmup.py ohmyzsh`

Of course, if you need a specific version to be installed, simply do so within the container.
If you are a careful person (and even though you are not), we strongly encourage that you 👀 scout `warmup.py` before you do this for security reasons.
It's not lengthy code.

Once you have a [💤 LazyVim](http://www.lazyvim.org/) environment installed in the container, turn on the `clangd` and `rust` plugins.

> [!NOTE]
> When you attach to a Docker container and explore the shell, you will quickly notice that the Emacs binding `ctrl-p` does not work as intended.
> This is because Docker assigns `ctrl-p ctrl-q` as a special key sequence to detach from the container.
> 
> To change this behavior, create a Docker config file `$HOME/.docker/config.json` on the remote machine, **not in the container on the remote!**
> Set its contents to
> ```
> {
>   "detachKeys": "ctrl-q"
> }
> ```
> The value `ctrl-q` defines the new key combination for detaching.
Replace this with your preferred combination.
You can now detach from the container by pressing `ctrl-q`.

Now you're good to go! 🚀

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
